#!/usr/bin/env python3
"""
Predictive FactLedger tests (PHASE_PREDICTIVE_FACTLEDGER_PLAN v1).

Levels 1 + 2 of the plan's validation strategy, all offline:

  1. Unit (pure math) — EdgeFilter beta-binomial updates,
     ActivityPrior fit/predict recovery, PredictiveLayer cold start,
     sidecar roundtrip, reset semantics, warm-from-history.
  2. Integration with StoryDirector — tick() runs with the layer
     enabled, the drift check fires on a divergent activity, the
     sidecar persists across Director instances, the activity
     history JSONL is written, /story/state carries the block.

Usage:
    python tests/test_predictive_factledger.py
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

NPC_ROOT = Path(__file__).parent.parent.resolve()
PIE_ROOT = (NPC_ROOT.parent / "plug-in-intelligence-engine").resolve()

# NOTE: no TextIOWrapper here — test_story_director (imported below
# for its stub harness) wraps sys.stdout.buffer at import time, and a
# second wrapper would orphan the first and close the shared buffer
# on GC (same double-wrap hazard e2e_stress.py documents).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NPC_ENGINE_DEV_MODE", "1")
logging.basicConfig(level=logging.WARNING)
for n in ["httpx", "huggingface_hub", "sentence_transformers", "faiss",
          "tqdm", "llama_cpp", "engine.npc_knowledge", "engine.npc_capabilities"]:
    logging.getLogger(n).setLevel(logging.ERROR)

sys.path.insert(0, str(NPC_ROOT))
sys.path.insert(0, str(PIE_ROOT))
sys.path.insert(0, str(NPC_ROOT / "tests"))

import numpy as np  # noqa: E402

from npc_engine.predictive_factledger import (  # noqa: E402
    ActivityPrior, EdgeFilter, PredictiveLayer, EDGE_PRIOR_BOOST_CAP,
)
from npc_engine.story_director import ArcPlanner  # noqa: E402

# Re-use the offline stub harness from the main Story Director tests
# (stub engine + state-file isolation) so integration tests here look
# and behave like the 192 existing ones.
from test_story_director import (  # noqa: E402
    _make_stub_engine, _isolate_state_file,
)
from npc_engine.story_director import StoryDirector  # noqa: E402

_LABELS = ["idle", "in_combat", "in_dialogue", "in_dungeon", "in_menu",
           "in_town", "traveling", "unknown", "wandering"]


def _cleanup_predictive_sidecars():
    """Remove tmp predictive sidecars created via isolated state files."""
    d = NPC_ROOT / "data" / "story_director"
    for p in list(d.glob("_tmp_*.predictive.npz")) + \
             list(d.glob("_tmp_*.activity_history.jsonl")):
        try:
            p.unlink()
        except Exception:
            pass


# ── Level 1 — unit (pure math) ──────────────────────────────────


def test_edge_filter_cold_returns_half():
    filt = EdgeFilter("npc_beat", bucket_count=8)
    assert filt.predict(("kael",), 0) == 0.5
    assert filt.predict(("nobody",), 7) == 0.5
    print("  [PASS] edge_filter_cold_returns_half")


def test_edge_filter_beta_update_and_predict():
    """Synthetic 2-bucket cycle: kael always active in bucket 0,
    never in bucket 1. Posterior means must match the Beta(1,1)
    conjugate update exactly."""
    filt = EdgeFilter("npc_beat", bucket_count=2)
    for _ in range(8):
        filt.update(("kael",), 0, True)
        filt.update(("kael",), 1, False)
    # Beta(1+8, 1) mean = 9/10; Beta(1, 1+8) mean = 1/10
    assert abs(filt.predict(("kael",), 0) - 9 / 10) < 1e-9, filt.predict(("kael",), 0)
    assert abs(filt.predict(("kael",), 1) - 1 / 10) < 1e-9, filt.predict(("kael",), 1)
    # Bucket wraps modulo bucket_count
    assert filt.predict(("kael",), 2) == filt.predict(("kael",), 0)
    print("  [PASS] edge_filter_beta_update_and_predict")


def test_edge_filter_serialization_roundtrip():
    filt = EdgeFilter("npc_beat", bucket_count=4)
    filt.update(("kael",), 0, True)
    filt.update(("bess",), 3, False)
    clone = EdgeFilter.from_json(filt.to_json())
    assert clone.edge_kind == "npc_beat"
    assert clone.bucket_count == 4
    assert clone.counts == filt.counts, (clone.counts, filt.counts)
    print("  [PASS] edge_filter_serialization_roundtrip")


def test_activity_prior_cold_uniform():
    prior = ActivityPrior(_LABELS)
    assert not prior.is_fitted
    dist = prior.predict(np.ones(4))
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert all(abs(v - 1 / len(_LABELS)) < 1e-9 for v in dist.values())
    print("  [PASS] activity_prior_cold_uniform")


def _separable_pairs(n_per_class: int = 6, dim: int = 16):
    """Two well-separated latent clusters -> two labels."""
    rng = np.random.default_rng(42)
    a_center = np.zeros(dim); a_center[0] = 1.0
    b_center = np.zeros(dim); b_center[1] = 1.0
    pairs = []
    for _ in range(n_per_class):
        pairs.append((a_center + rng.normal(0, 0.05, dim), "in_town"))
        pairs.append((b_center + rng.normal(0, 0.05, dim), "in_dungeon"))
    return pairs, a_center, b_center


def test_activity_prior_fit_recovers_labels():
    """Plan level-1 spec: a linear projection trained on a fixture
    mapping recovers the label distribution within epsilon."""
    prior = ActivityPrior(_LABELS)
    pairs, a_center, b_center = _separable_pairs()
    assert prior.fit(pairs) is True
    dist_a = prior.predict(a_center)
    dist_b = prior.predict(b_center)
    assert max(dist_a, key=dist_a.get) == "in_town", dist_a
    assert max(dist_b, key=dist_b.get) == "in_dungeon", dist_b
    assert dist_a["in_town"] > 0.6, dist_a["in_town"]
    assert dist_b["in_dungeon"] > 0.6, dist_b["in_dungeon"]
    print("  [PASS] activity_prior_fit_recovers_labels")


def test_activity_prior_single_class_stays_cold():
    prior = ActivityPrior(_LABELS)
    rng = np.random.default_rng(0)
    pairs = [(rng.normal(0, 1, 8), "in_town") for _ in range(20)]
    assert prior.fit(pairs) is False
    assert not prior.is_fitted
    print("  [PASS] activity_prior_single_class_stays_cold")


def test_activity_prior_below_min_samples_stays_cold():
    prior = ActivityPrior(_LABELS)
    pairs, _, _ = _separable_pairs(n_per_class=2)  # 4 < 8 min samples
    assert prior.fit(pairs) is False
    assert not prior.is_fitted
    print("  [PASS] activity_prior_below_min_samples_stays_cold")


def test_activity_prior_dim_mismatch_falls_back_uniform():
    prior = ActivityPrior(_LABELS)
    pairs, _, _ = _separable_pairs(dim=16)
    assert prior.fit(pairs)
    dist = prior.predict(np.ones(32))  # wrong latent dim
    assert all(abs(v - 1 / len(_LABELS)) < 1e-9 for v in dist.values())
    print("  [PASS] activity_prior_dim_mismatch_falls_back_uniform")


def _fake_ledger(entries):
    return SimpleNamespace(entries=entries)


def _entry(text, npc_id, tick, vec):
    return {"text": text, "npc_id": npc_id, "kind": "event", "tick": tick,
            "embedding": list(vec)}


def test_unknown_edge_kind_ignored_not_crashed(tmp_base: Path):
    """Plan risk register: unknown edge kinds are ignored, not
    crashed; new kinds need an explicit registration call."""
    layer = PredictiveLayer(tmp_base / "p0.npz", _LABELS)
    assert "faction_presence" not in layer.edge_filters
    # predict_next only consults registered kinds — a missing kind
    # must not raise and must contribute nothing.
    pred, edge_priors = layer.predict_next(_fake_ledger([]), tick=0,
                                           current_activity="unknown")
    assert edge_priors == {}
    # Registration is explicit and idempotent.
    f1 = layer.register_edge_kind("faction_presence", bucket_count=4)
    f2 = layer.register_edge_kind("faction_presence")
    assert f1 is f2 and f1.bucket_count == 4
    print("  [PASS] unknown_edge_kind_ignored_not_crashed")


def test_predictive_layer_empty_ledger_cold(tmp_base: Path):
    """Plan level-1 spec: predict_next with an empty ledger returns
    uniform + cold=True."""
    layer = PredictiveLayer(tmp_base / "p1.npz", _LABELS)
    pred, edge_priors = layer.predict_next(_fake_ledger([]), tick=0,
                                           current_activity="unknown")
    assert pred.cold is True
    assert pred.drift is False
    assert edge_priors == {}
    assert all(abs(v - 1 / len(_LABELS)) < 1e-9
               for v in pred.distribution.values())
    print("  [PASS] predictive_layer_empty_ledger_cold")


def test_predictive_layer_record_observation_updates_filters(tmp_base: Path):
    layer = PredictiveLayer(tmp_base / "p2.npz", _LABELS)
    vec = np.zeros(8); vec[0] = 1.0
    # kael gets beats at even ticks only; bess every tick.
    for tick in range(16):
        delta = [_entry("beat", "bess", tick, vec)]
        if tick % 2 == 0:
            delta.append(_entry("beat", "kael", tick, vec))
        layer.record_observation(tick, "in_town", delta)
    filt = layer.edge_filters["npc_beat"]
    # Bucket cycle is 8, so even ticks land in even buckets.
    assert filt.predict(("kael",), 0) > 0.5
    assert filt.predict(("kael",), 1) < 0.5
    assert filt.predict(("bess",), 0) > 0.5
    assert filt.predict(("bess",), 1) > 0.5
    # predict_next surfaces per-NPC edge priors for the next bucket
    _, edge_priors = layer.predict_next(_fake_ledger([]), tick=15,
                                        current_activity="in_town")
    assert set(edge_priors) == {"kael", "bess"}, edge_priors
    assert edge_priors["kael"] > 0.5  # next bucket after 15 is 0 (even)
    print("  [PASS] predictive_layer_record_observation_updates_filters")


def test_predictive_layer_sidecar_roundtrip(tmp_base: Path):
    path = tmp_base / "p3.npz"
    layer = PredictiveLayer(path, _LABELS)
    vec = np.zeros(8); vec[0] = 1.0
    for tick in range(10):
        layer.record_observation(tick, "in_town",
                                 [_entry("beat", "kael", tick, vec)])
    pairs, _, _ = _separable_pairs()
    layer.prior.fit(pairs)
    assert layer.prior.is_fitted
    layer.save()

    reloaded = PredictiveLayer(path, _LABELS)
    assert reloaded._loaded_from_sidecar is True
    assert reloaded.prior.is_fitted, "prior matrix must survive the sidecar"
    assert np.allclose(reloaded.prior.weight_matrix, layer.prior.weight_matrix)
    assert (reloaded.edge_filters["npc_beat"].counts
            == layer.edge_filters["npc_beat"].counts)
    print("  [PASS] predictive_layer_sidecar_roundtrip")


def test_predictive_layer_sidecar_label_change_discards_prior(tmp_base: Path):
    path = tmp_base / "p4.npz"
    layer = PredictiveLayer(path, _LABELS)
    pairs, _, _ = _separable_pairs()
    layer.prior.fit(pairs)
    layer.save()
    reloaded = PredictiveLayer(path, _LABELS + ["new_activity"])
    assert not reloaded.prior.is_fitted, \
        "stale prior must be discarded when the label set changes"
    print("  [PASS] predictive_layer_sidecar_label_change_discards_prior")


def test_predictive_layer_reset_zeroes_edges_keeps_prior(tmp_base: Path):
    """Plan composition table, game-reset row: reset() zeroes the edge
    filters but keeps the prior matrix."""
    layer = PredictiveLayer(tmp_base / "p5.npz", _LABELS)
    vec = np.zeros(8); vec[0] = 1.0
    layer.record_observation(0, "in_town", [_entry("b", "kael", 0, vec)])
    pairs, _, _ = _separable_pairs()
    layer.prior.fit(pairs)
    assert layer.edge_filters["npc_beat"].counts
    layer.reset()
    assert not layer.edge_filters["npc_beat"].counts
    assert layer.prior.is_fitted, "reset must NOT clear the prior matrix"
    print("  [PASS] predictive_layer_reset_zeroes_edges_keeps_prior")


def test_warm_from_history_fits_prior_and_filters(tmp_base: Path):
    history = tmp_base / "h1.jsonl"
    dim = 8
    town_vec = np.zeros(dim); town_vec[0] = 1.0
    dungeon_vec = np.zeros(dim); dungeon_vec[1] = 1.0
    entries = []
    lines = []
    now = datetime.now(timezone.utc)
    # Alternate 8-tick town blocks and 8-tick dungeon blocks; the
    # activity is posted at the start of each block.
    tick = 0
    for block in range(6):
        in_town = (block % 2 == 0)
        vec = town_vec if in_town else dungeon_vec
        activity = "in_town" if in_town else "in_dungeon"
        for _ in range(4):
            entries.append(_entry("beat", "kael" if in_town else "bess",
                                  tick, vec))
            lines.append(json.dumps({
                "tick": tick, "activity": activity,
                "ts": now.isoformat(),
            }))
            tick += 1
    history.write_text("\n".join(lines) + "\n", encoding="utf-8")

    layer = PredictiveLayer(tmp_base / "p6.npz", _LABELS,
                            history_path=history)
    report = layer.warm_from_history(_fake_ledger(entries))
    assert report["history_records"] == len(lines), report
    assert report["pairs"] > 0, report
    assert report["fitted"] is True, report
    assert layer.prior.is_fitted
    assert layer.edge_filters["npc_beat"].counts
    # The warmed prior should map a town-tail latent to in_town.
    dist = layer.prior.predict(town_vec)
    assert max(dist, key=dist.get) == "in_town", dist
    print("  [PASS] warm_from_history_fits_prior_and_filters")


def test_warm_from_history_prunes_stale_records(tmp_base: Path):
    history = tmp_base / "h2.jsonl"
    old_ts = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    new_ts = datetime.now(timezone.utc).isoformat()
    lines = [
        json.dumps({"tick": 0, "activity": "in_town", "ts": old_ts}),
        json.dumps({"tick": 1, "activity": "in_town", "ts": new_ts}),
    ]
    history.write_text("\n".join(lines) + "\n", encoding="utf-8")
    layer = PredictiveLayer(tmp_base / "p7.npz", _LABELS,
                            history_path=history)
    report = layer.warm_from_history(_fake_ledger([]))
    assert report["pruned"] == 1, report
    assert report["history_records"] == 1, report
    kept = history.read_text(encoding="utf-8").strip().splitlines()
    assert len(kept) == 1 and json.loads(kept[0])["tick"] == 1
    print("  [PASS] warm_from_history_prunes_stale_records")


def test_edge_prior_boost_flips_tied_clusters_and_respects_cap(tmp_base: Path):
    """Two equally dense ledger clusters; a warm edge prior on the
    second cluster's NPC must flip the argmax, and a cold (0.5)
    prior must reproduce the unboosted choice bit-for-bit."""
    dim = 8
    a = np.zeros(dim); a[0] = 1.0
    b = np.zeros(dim); b[1] = 1.0
    entries = []
    for i in range(3):
        entries.append(_entry(f"thread A beat {i}", "kael", i, a))
    for i in range(3):
        entries.append(_entry(f"thread B beat {i}", "bess", 3 + i, b))
    ledger = SimpleNamespace(entries=entries, np=np)

    def propose(boost):
        planner = ArcPlanner(tmp_base / "arcs_boost.json")
        return planner._propose_from_ledger(
            ledger, ["kael", "bess"], current_tick=10,
            edge_prior_boost=boost,
        )

    baseline = propose(None)
    assert baseline is not None
    # Ties resolve to the first index — thread A.
    assert baseline.focus_npcs == ["kael"], baseline.focus_npcs

    cold = propose({"kael": 0.5, "bess": 0.5})
    assert cold is not None and cold.focus_npcs == ["kael"], \
        "cold priors (0.5) must not change the choice"

    boosted = propose({"bess": 0.95})
    assert boosted is not None and boosted.focus_npcs == ["bess"], \
        boosted.focus_npcs

    # Cap check: a huge prior can nudge by at most EDGE_PRIOR_BOOST_CAP,
    # so it can flip a TIE but never beat a strictly denser cluster.
    entries_dense = entries + [_entry("thread A beat 3", "kael", 6, a)]
    ledger_dense = SimpleNamespace(entries=entries_dense, np=np)
    planner = ArcPlanner(tmp_base / "arcs_boost2.json")
    arc = planner._propose_from_ledger(
        ledger_dense, ["kael", "bess"], current_tick=10,
        edge_prior_boost={"bess": 1.0},
    )
    assert arc is not None and arc.focus_npcs[0] == "kael", \
        f"boost (cap {EDGE_PRIOR_BOOST_CAP}) must never dominate density"
    print("  [PASS] edge_prior_boost_flips_tied_clusters_and_respects_cap")


# ── Level 2 — integration with StoryDirector ─────────────────────


def test_tick_runs_with_predictive_layer_enabled():
    """Plan level-2 spec: tick() runs to completion with the layer
    enabled, and /story/state carries the predictive block."""
    restore = _isolate_state_file("pred_tick")
    try:
        engine = _make_stub_engine(responses=[
            '{"action": "event", "target": "kael", '
            '"description": "Kael hammers a strange blade."}',
        ])
        director = StoryDirector(engine)
        assert director._predictive is not None, "layer should be on by default"
        result = director.tick()
        assert result["dispatch"]["ok"], result
        assert director._last_activity_pred is not None
        assert director._last_activity_pred["cold"] is True, \
            "fresh ledger + unfitted prior must be cold"
        state = director.get_state()
        assert state["predictive"]["enabled"] is True
        assert state["predictive"]["boost_enabled"] is False, \
            "boost must be OFF by default (plan step 5 gating)"
        assert state["predictive"]["last_prediction"] is not None
        assert state["predictive"]["observation_count"] >= 1
    finally:
        restore()
        _cleanup_predictive_sidecars()
    print("  [PASS] tick_runs_with_predictive_layer_enabled")


def test_drift_check_fires_when_prior_disagrees():
    """Plan level-2 spec: the drift check fires when the (fitted)
    prior confidently disagrees with the reported activity."""
    restore = _isolate_state_file("pred_drift")
    try:
        engine = _make_stub_engine(responses=[
            '{"action": "event", "target": "kael", '
            '"description": "Dungeon torches gutter in the deep dark."}',
        ])
        director = StoryDirector(engine)
        layer = director._predictive
        assert layer is not None
        # Seed the ledger with dungeon-flavored content (real
        # embeddings via the ledger's embedder).
        for i, text in enumerate([
            "The player descends into the crypt beneath the old keep.",
            "Skeletons stir in the dungeon's flooded lower gallery.",
            "A torch sputters against the dungeon's wet stone walls.",
        ]):
            director.ledger.add(text=text, npc_id="kael", kind="event",
                                tick=i + 1)
        current_latent = layer.summary_latent(director.ledger.entries)
        assert current_latent is not None
        # Fit the prior so the CURRENT ledger latent maps to
        # in_dungeon and a contrasting latent maps to in_town.
        pairs = [(current_latent, "in_dungeon") for _ in range(5)]
        pairs += [(-current_latent, "in_town") for _ in range(5)]
        assert layer.prior.fit(pairs), "fixture prior must fit"
        # Game client claims the player is in town -> drift.
        director._player_activity = "in_town"
        result = director.tick()
        assert result["dispatch"]["ok"], result
        pred = director._last_activity_pred
        assert pred is not None and pred["cold"] is False
        assert pred["top_activity"] == "in_dungeon", pred
        assert pred["drift"] is True, pred
        assert layer._drift_count >= 1
    finally:
        restore()
        _cleanup_predictive_sidecars()
    print("  [PASS] drift_check_fires_when_prior_disagrees")


def test_sidecar_persists_across_director_instances():
    """Plan level-2 spec: graceful shutdown writes the sidecar;
    restart loads it without re-fitting."""
    restore = _isolate_state_file("pred_persist")
    try:
        engine1 = _make_stub_engine(responses=[
            '{"action": "event", "target": "kael", '
            '"description": "Kael quenches a blade in the rain barrel."}',
            '{"action": "fact", "target": "bess", '
            '"fact": "The inn cellar flooded overnight."}',
        ])
        director1 = StoryDirector(engine1)
        director1.tick()
        director1.tick()
        stats1 = director1._predictive.stats()
        assert stats1["tracked_npcs"] >= 1, stats1

        engine2 = _make_stub_engine()
        director2 = StoryDirector(engine2)
        stats2 = director2._predictive.stats()
        assert stats2["loaded_from_sidecar"] is True, stats2
        assert stats2["tracked_npcs"] == stats1["tracked_npcs"], \
            (stats1, stats2)
        assert (director2._predictive.edge_filters["npc_beat"].counts
                == director1._predictive.edge_filters["npc_beat"].counts)
    finally:
        restore()
        _cleanup_predictive_sidecars()
    print("  [PASS] sidecar_persists_across_director_instances")


def test_activity_history_jsonl_written():
    """Plan schema section: every /story/activity POST appends one
    JSONL record with tick, activity, and an ISO timestamp."""
    restore = _isolate_state_file("pred_hist")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        director.set_player_activity("in_town")
        director.set_player_activity("in_dungeon")
        director.set_player_activity("in_dungeon")  # repeat also logs
        path = director._activity_history_file
        assert path.exists(), path
        lines = [json.loads(x) for x in
                 path.read_text(encoding="utf-8").strip().splitlines()]
        assert len(lines) == 3, lines
        assert [x["activity"] for x in lines] == \
            ["in_town", "in_dungeon", "in_dungeon"]
        assert all("tick" in x and "ts" in x for x in lines)
        datetime.fromisoformat(lines[0]["ts"])  # parses
    finally:
        restore()
        _cleanup_predictive_sidecars()
    print("  [PASS] activity_history_jsonl_written")


def test_game_reset_zeroes_edge_filters_keeps_prior():
    restore = _isolate_state_file("pred_reset")
    try:
        engine = _make_stub_engine(responses=[
            '{"action": "event", "target": "kael", '
            '"description": "Kael sharpens the town gate hinges."}',
        ])
        director = StoryDirector(engine)
        director.tick()
        layer = director._predictive
        assert layer.edge_filters["npc_beat"].counts
        pairs, _, _ = _separable_pairs()
        layer.prior.fit(pairs)
        director.reset_to_initial_state()
        assert not layer.edge_filters["npc_beat"].counts, \
            "game reset must zero edge filters"
        assert layer.prior.is_fitted, \
            "game reset must keep the prior matrix"
        assert director._last_activity_pred is None
    finally:
        restore()
        _cleanup_predictive_sidecars()
    print("  [PASS] game_reset_zeroes_edge_filters_keeps_prior")


def test_disable_env_removes_layer():
    restore = _isolate_state_file("pred_disable")
    os.environ["NPC_ENGINE_PREDICTIVE_DISABLE"] = "1"
    try:
        engine = _make_stub_engine(responses=[
            '{"action": "event", "target": "kael", '
            '"description": "Nothing predictive about this beat."}',
        ])
        director = StoryDirector(engine)
        assert director._predictive is None
        result = director.tick()  # must still run cleanly
        assert result["dispatch"]["ok"], result
        assert director.get_state()["predictive"] == {"enabled": False}
        # No sidecar, no history file.
        assert not director._predictive_file.exists()
        director.set_player_activity("in_town")
        assert not director._activity_history_file.exists()
    finally:
        os.environ.pop("NPC_ENGINE_PREDICTIVE_DISABLE", None)
        restore()
        _cleanup_predictive_sidecars()
    print("  [PASS] disable_env_removes_layer")


def test_boost_env_gates_arc_proposal_boost():
    restore = _isolate_state_file("pred_boost_gate")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        assert director._predictive_boost_enabled is False
        os.environ["NPC_ENGINE_PREDICTIVE_BOOST"] = "1"
        try:
            engine2 = _make_stub_engine()
            director2 = StoryDirector(engine2)
            assert director2._predictive_boost_enabled is True
        finally:
            os.environ.pop("NPC_ENGINE_PREDICTIVE_BOOST", None)
    finally:
        restore()
        _cleanup_predictive_sidecars()
    print("  [PASS] boost_env_gates_arc_proposal_boost")


def test_note_activity_harvests_pairs_and_fits():
    """Posting activities against a ledger with embeddings harvests
    supervision pairs; once the sample + diversity floor is met the
    prior fits and predictions go warm."""
    restore = _isolate_state_file("pred_note")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        layer = director._predictive
        town_texts = [
            "Market stalls open along the town square.",
            "The innkeeper waters down the morning ale.",
            "Children chase a stray dog past the chapel.",
        ]
        dungeon_texts = [
            "Something claws at the sealed crypt door.",
            "The dungeon air tastes of rust and mold.",
            "A rotted rope bridge spans the chasm below the keep.",
        ]
        tick = 0
        for round_i in range(3):
            for text in town_texts:
                tick += 1
                director.ledger.add(text=f"{text} ({round_i})",
                                    npc_id="kael", kind="event", tick=tick)
            director.set_player_activity("in_town")
            for text in dungeon_texts:
                tick += 1
                director.ledger.add(text=f"{text} ({round_i})",
                                    npc_id="bess", kind="event", tick=tick)
            director.set_player_activity("in_dungeon")
        assert len(layer._activity_pairs) == 6, len(layer._activity_pairs)
        # 6 pairs < min 8 -> still cold; two more rounds push it over.
        for text in town_texts:
            tick += 1
            director.ledger.add(text=f"{text} (x)", npc_id="kael",
                                kind="event", tick=tick)
        director.set_player_activity("in_town")
        for text in dungeon_texts:
            tick += 1
            director.ledger.add(text=f"{text} (x)", npc_id="bess",
                                kind="event", tick=tick)
        director.set_player_activity("in_dungeon")
        assert layer.prior.is_fitted, \
            f"{len(layer._activity_pairs)} pairs should be enough to fit"
        # The freshest ledger tail is dungeon-flavored.
        pred, _ = layer.predict_next(director.ledger, tick, "in_dungeon")
        assert pred.cold is False
        assert pred.top_activity == "in_dungeon", pred.to_dict()
        assert pred.drift is False
    finally:
        restore()
        _cleanup_predictive_sidecars()
    print("  [PASS] note_activity_harvests_pairs_and_fits")


# ── Runner ──────────────────────────────────────────────────────


def main():
    import tempfile
    print("Predictive FactLedger — unit tests (level 1)")
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        test_edge_filter_cold_returns_half()
        test_edge_filter_beta_update_and_predict()
        test_edge_filter_serialization_roundtrip()
        test_activity_prior_cold_uniform()
        test_activity_prior_fit_recovers_labels()
        test_activity_prior_single_class_stays_cold()
        test_activity_prior_below_min_samples_stays_cold()
        test_activity_prior_dim_mismatch_falls_back_uniform()
        test_unknown_edge_kind_ignored_not_crashed(base)
        test_predictive_layer_empty_ledger_cold(base)
        test_predictive_layer_record_observation_updates_filters(base)
        test_predictive_layer_sidecar_roundtrip(base)
        test_predictive_layer_sidecar_label_change_discards_prior(base)
        test_predictive_layer_reset_zeroes_edges_keeps_prior(base)
        test_warm_from_history_fits_prior_and_filters(base)
        test_warm_from_history_prunes_stale_records(base)
        test_edge_prior_boost_flips_tied_clusters_and_respects_cap(base)

    print("\nPredictive FactLedger — StoryDirector integration (level 2)")
    test_tick_runs_with_predictive_layer_enabled()
    test_drift_check_fires_when_prior_disagrees()
    test_sidecar_persists_across_director_instances()
    test_activity_history_jsonl_written()
    test_game_reset_zeroes_edge_filters_keeps_prior()
    test_disable_env_removes_layer()
    test_boost_env_gates_arc_proposal_boost()
    test_note_activity_harvests_pairs_and_fits()

    print("\nAll predictive FactLedger tests passed.")


if __name__ == "__main__":
    main()
