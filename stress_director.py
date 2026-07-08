#!/usr/bin/env python3
"""
Story Director GPU stress harness — drives one phase per run via a
common `boot_engine` path that matches bench_story_director. Each
phase function returns a findings dict that the caller serializes
into ``logs/stress_<phase>.json``.

Usage:
    python stress_director.py --phase 4a --world port_blackwater_zoned
    python stress_director.py --phase 4b
    python stress_director.py --phase 4c
    python stress_director.py --phase 3a
    python stress_director.py --phase 5

Assumes the caller has already run ``bench_story_director.py`` once
or has the prod models available at
``../plug-in-intelligence-engine/models/qwen2.5-3b-instruct-q4_k_m.gguf``.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

NPC_ROOT = Path(__file__).parent.resolve()
PIE_ROOT = (NPC_ROOT.parent / "plug-in-intelligence-engine").resolve()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NPC_ENGINE_DEV_MODE", "1")
logging.basicConfig(level=logging.WARNING)
for n in ["httpx", "huggingface_hub", "sentence_transformers", "faiss",
          "tqdm", "llama_cpp", "engine.npc_knowledge", "engine.npc_capabilities"]:
    logging.getLogger(n).setLevel(logging.ERROR)

sys.path.insert(0, str(NPC_ROOT))
sys.path.insert(0, str(PIE_ROOT))


def boot_engine(world: str = "port_blackwater_zoned", reset: bool = True):
    """Boot a real NPCEngine pointed at the requested world. Borrows
    the config-writing pattern from bench_story_director."""
    import yaml

    world_dir = NPC_ROOT / "data" / "worlds" / world
    model = PIE_ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
    if not model.exists():
        raise RuntimeError(f"model file missing: {model}")

    # Clean per-run state.
    if reset:
        for fname in ("state.json", "fact_ledger.json",
                       "fact_ledger.embeddings.npy", "arcs.json",
                       "arcs.embeddings.npy",
                       # Predictive-lane sidecars — without these a
                       # previous run's fitted prior leaks into a
                       # "fresh" scenario boot.
                       "state.predictive.npz",
                       "state.activity_history.jsonl"):
            p = world_dir / "story" / fname
            if p.exists():
                p.unlink()

    raw = yaml.safe_load((PIE_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["base_model"]["path"] = str(model)
    raw["base_model"]["context_length"] = 8192
    raw["base_model"]["temperature"] = 0.6
    raw["fusion"] = raw.get("fusion") or {}
    raw["fusion"]["chat_format"] = "chatml"
    raw["npc"] = raw.get("npc") or {}
    raw["npc"]["enabled"] = True
    raw["npc"]["profiles_dir"] = str(world_dir / "npc_profiles")
    raw["npc"]["state_dir"] = str(world_dir / "npc_state")

    temp_pie = PIE_ROOT / f"config_stress_{world}.yaml"
    temp_pie.write_text(yaml.dump(raw, default_flow_style=False), encoding="utf-8")
    npc_cfg = {
        "world_dir": str(world_dir),
        "world_name": world_dir.name,
        "active_npc": "captain_reva" if "port_blackwater" in world else "noah",
        "pie_config": str(temp_pie),
    }
    temp_npc = NPC_ROOT / f"config_stress_{world}.yaml"
    temp_npc.write_text(yaml.dump(npc_cfg, default_flow_style=False), encoding="utf-8")

    os.chdir(PIE_ROOT)
    from npc_engine.engine import NPCEngine
    engine = NPCEngine(str(temp_npc))
    engine.initialize()
    os.chdir(NPC_ROOT)
    # Re-bind story director paths to per-world story/ dir (resolve_paths
    # already runs in __init__; re-reading state_dir ensures the runtime
    # directory exists).
    (world_dir / "story").mkdir(parents=True, exist_ok=True)
    return engine, temp_pie, temp_npc


def _tick(sd, label: str, **kwargs):
    """Time a tick and return a compact record. ``sd`` is the
    StoryDirector instance, not the engine."""
    t0 = time.monotonic()
    result = sd.tick(**kwargs)
    elapsed = time.monotonic() - t0
    action = result.get("action") or {}
    return {
        "label": label,
        "elapsed_s": round(elapsed, 3),
        "tick": result.get("tick"),
        "paused": bool(result.get("paused")),
        "paused_reason": result.get("paused_reason"),
        "action_kind": action.get("action"),
        "next_hint_s": result.get("next_tick_recommended_in_seconds"),
        "n_subactions": len(result.get("sub_actions", []))
            if isinstance(result.get("sub_actions"), list) else 0,
    }


# ── Phase 4a — activity gates ────────────────────────────────────

def phase_4a(world: str, out: Path) -> dict:
    engine, _, _ = boot_engine(world=world)
    sd = engine.story_director
    sd.set_narration_mode("terse")
    sd.set_active_zones(["dock_district", "lighthouse_bluffs"])

    report = {"phase": "4a", "world": world, "transitions": []}

    # 1. Default (unknown) tick — should run normally, single action.
    report["transitions"].append(_tick(sd, "unknown_default"))

    # 2. Combat — paused noop, no LLM call.
    sd.set_player_activity("in_combat")
    rec = _tick(sd, "in_combat")
    rec["next_expected"] = 10
    report["transitions"].append(rec)

    # 3. Menu — paused with 30s hint.
    sd.set_player_activity("in_menu")
    rec = _tick(sd, "in_menu")
    rec["next_expected"] = 30
    report["transitions"].append(rec)

    # 4. Idle — 120 s hint.
    sd.set_player_activity("idle")
    rec = _tick(sd, "idle")
    rec["next_expected"] = 120
    report["transitions"].append(rec)

    # 5. Dungeon — live tick with quest dropped from rotation.
    sd.set_player_activity("in_dungeon")
    kind_counts: dict = {}
    for i in range(4):
        # Force the rotation to start at quest slot so we actually
        # exercise the "drop quest" branch.
        sd._kind_rotation_index = 1  # quest index
        record = _tick(sd, f"in_dungeon_{i}", actions_per_tick=1)
        kind_counts[record["action_kind"]] = kind_counts.get(
            record["action_kind"], 0
        ) + 1
        report["transitions"].append(record)
    report["dungeon_kind_counts"] = kind_counts

    # 6. Dialogue — single-action forced even with actions_per_tick=3.
    sd.set_player_activity("in_dialogue")
    rec = _tick(sd, "in_dialogue", actions_per_tick=3)
    rec["multi_action_requested"] = 3
    report["transitions"].append(rec)

    # 7. Wandering — single-action forced.
    sd.set_player_activity("wandering")
    rec = _tick(sd, "wandering", actions_per_tick=2)
    rec["multi_action_requested"] = 2
    report["transitions"].append(rec)

    # 8. in_town — normal live multi-action tick.
    sd.set_player_activity("in_town")
    rec = _tick(sd, "in_town", actions_per_tick=3)
    report["transitions"].append(rec)

    report["final_state"] = {
        "tick_count": sd.tick_count,
        "player_activity": sd._player_activity,
        "ledger_entries": len(sd.ledger.entries),
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ── Phase 4b — quest pacing ──────────────────────────────────────

def phase_4b(world: str, out: Path) -> dict:
    engine, _, _ = boot_engine(world=world)
    sd = engine.story_director
    sd.set_narration_mode("terse")
    sd.set_active_zones(["dock_district", "lighthouse_bluffs"])

    # Tighten pacing so we can actually observe the gates in a short
    # run.
    sd.set_quest_pacing(max_unoffered=1, cooldown_ticks=3)

    report = {
        "phase": "4b", "world": world,
        "pacing": sd.get_quest_pacing(),
        "ticks": [],
        "npc_quests_over_time": [],
    }

    # Run 12 ticks with actions_per_tick=3 on a 6-NPC zoned cast so
    # the 4b gates fire naturally.
    for i in range(12):
        rec = _tick(sd, f"t{i+1}", actions_per_tick=3)
        # Count quests per NPC this tick.
        per_npc = {
            nid: sum(1 for q in npc.quests
                     if q.status in ("available", "active"))
            for nid, npc in engine.pie.npc_knowledge.profiles.items()
        }
        rec["per_npc_open_quests"] = per_npc
        report["ticks"].append(rec)

    # Final state — did any NPC breach _MAX_UNOFFERED_QUESTS_PER_NPC?
    final_open: dict[str, int] = {}
    for nid, npc in engine.pie.npc_knowledge.profiles.items():
        final_open[nid] = sum(
            1 for q in npc.quests if q.status == "available"
        )
    report["final_unoffered_per_npc"] = final_open
    max_unoffered_observed = max(final_open.values(), default=0)
    report["max_unoffered_observed"] = max_unoffered_observed
    report["cap_respected"] = max_unoffered_observed <= max(
        1, sd._effective_max_unoffered_quests(),
    )
    report["last_quest_dispatched_per_npc"] = dict(
        sd._last_quest_dispatched_per_npc,
    )
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ── Phase 4c — pause + budget ────────────────────────────────────

def phase_4c(world: str, out: Path) -> dict:
    engine, _, _ = boot_engine(world=world)
    sd = engine.story_director
    sd.set_narration_mode("terse")
    sd.set_active_zones(["dock_district", "lighthouse_bluffs"])

    report = {"phase": "4c", "world": world, "events": []}

    # 1. Normal tick, record duration.
    rec = _tick(sd, "pre_pause", actions_per_tick=1)
    report["events"].append(rec)

    # 2. Pause → tick. No LLM call expected.
    pre_llm_calls = len(
        getattr(engine.pie, "base_model", None)._call_trace
        if hasattr(getattr(engine.pie, "base_model", None), "_call_trace")
        else []
    )
    sd.pause_ticks()
    rec = _tick(sd, "paused_tick", actions_per_tick=1)
    rec["expect_paused"] = True
    report["events"].append(rec)

    # 3. Resume → tick, real LLM again.
    sd.resume_ticks()
    rec = _tick(sd, "post_resume", actions_per_tick=1)
    report["events"].append(rec)

    # 4. Set tight budget, run a tick to push us over, then the next
    # tick should return budget_exceeded.
    sd.set_tick_budget(0.1)   # 0.1 s per 60 s — trivially tripped
    rec_over = _tick(sd, "post_budget_set", actions_per_tick=1)
    rec_over["budget_state"] = sd.get_pause_state()
    report["events"].append(rec_over)
    rec_blocked = _tick(sd, "blocked_by_budget", actions_per_tick=1)
    rec_blocked["expect_paused_reason"] = "budget_exceeded"
    report["events"].append(rec_blocked)

    # 5. Clear budget, tick runs again.
    sd.set_tick_budget(-1.0)
    rec = _tick(sd, "post_budget_clear", actions_per_tick=1)
    report["events"].append(rec)

    # 6. Adaptive hint snapshot across activities.
    hints = {}
    for act in ("in_combat", "in_menu", "idle", "wandering",
                 "in_dungeon", "in_town", "unknown"):
        sd.set_player_activity(act)
        hints[act] = sd._compute_next_tick_hint()
    report["activity_hints"] = hints
    sd.set_player_activity("in_town")

    # 7. Arc-at-confront accelerator check. Stage an active arc at
    # beat >= 2, set wandering (900 s baseline), hint should
    # collapse to 60 s.
    from npc_engine.story_director import NarrativeArc
    arc = NarrativeArc(
        id="stress_arc", theme="climax",
        focus_npcs=["captain_reva"],
        beat_goals=["seed", "escalate", "confront", "resolve"],
        current_beat=2, status="active", started_at_tick=1,
    )
    sd.arc_planner.arcs.append(arc)
    sd.arc_planner.active_arc_ids.append("stress_arc")
    sd.set_player_activity("wandering")
    report["confront_accelerator_hint"] = sd._compute_next_tick_hint()
    sd.set_player_activity("in_town")
    report["confront_town_hint"] = sd._compute_next_tick_hint()

    report["pause_state_end"] = sd.get_pause_state()
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ── Phase 3a — main-line on PB zoned ─────────────────────────────

def phase_3a(world: str, out: Path) -> dict:
    # Write a main_dark_lighthouse line into the world's quest_lines.yaml
    # before booting so _load_quest_lines picks it up.
    import yaml as _yaml
    world_dir = NPC_ROOT / "data" / "worlds" / world
    qlines_path = world_dir / "quest_lines.yaml"
    original_content = qlines_path.read_text(encoding="utf-8") if qlines_path.exists() else None
    qlines_path.write_text(
        _yaml.safe_dump({
            "quest_lines": {
                "main_dark_lighthouse": {
                    "type": "main",
                    "title": "The Dark Lighthouse",
                    "description": "Discover why the lighthouse went dark.",
                    "beats": [
                        {"quest_id": "lighthouse_mystery",
                          "giver": "captain_reva"},
                        {"quest_id": "witness_account",
                          "giver": "thessa",
                          "requires": ["lighthouse_mystery"]},
                        {"quest_id": "broken_lantern",
                          "giver": "brom",
                          "requires": ["witness_account"]},
                    ],
                    "protected_givers": ["captain_reva", "thessa", "brom"],
                    "reward_track": [
                        "Harbor Master's seal",
                        "Thessa's sea-tongue charm",
                        "Brom's pilgrim medallion",
                    ],
                },
            },
            "director": {
                "quest_auto_refuse": {
                    "enabled": True,
                    "player_configurable": True,
                },
            },
        }),
        encoding="utf-8",
    )

    try:
        engine, _, _ = boot_engine(world=world)
        sd = engine.story_director
        sd.set_narration_mode("terse")
        sd.set_active_zones(["dock_district", "lighthouse_bluffs"])
        sd.set_autonomous_lifecycle(True)

        report = {
            "phase": "3a",
            "world": world,
            "main_line_cast_initial": sorted(sd._main_line_cast()),
            "protected_givers_initial": sorted(sd._protected_givers()),
            "dev_auto_refuse_enabled": sd._quest_auto_refuse_enabled,
            "ticks": [],
            "focus_npc_counts": {},
        }

        # Run 15 autonomous ticks, record focus.
        focus_counts: dict[str, int] = {}
        for i in range(15):
            pre = set(sd._alive_npcs().keys())
            rec = _tick(sd, f"t{i+1}", actions_per_tick=3)
            post = set(sd._alive_npcs().keys())
            dead_this_tick = list(pre - post)
            # Extract focus NPCs used in this tick from npc_last_planned_tick.
            rec["deaths_this_tick"] = dead_this_tick
            report["ticks"].append(rec)
            # Last-planned tick records give us per-tick focus list.
            this_tick_focus = [
                nid for nid, tk in sd._npc_last_planned_tick.items()
                if tk == sd.tick_count
            ]
            for nid in this_tick_focus:
                focus_counts[nid] = focus_counts.get(nid, 0) + 1
        report["focus_npc_counts"] = focus_counts

        # Main-line vs non ratio.
        ml_cast = sd._main_line_cast()
        ml_focus = sum(focus_counts.get(n, 0) for n in ml_cast)
        non_ml_focus = sum(
            v for n, v in focus_counts.items() if n not in ml_cast
        )
        report["main_line_focus"] = ml_focus
        report["non_main_line_focus"] = non_ml_focus
        report["ratio_ml_to_non_ml"] = (
            ml_focus / max(1, non_ml_focus)
        )

        # Protection check — every protected giver still alive?
        report["deceased_at_end"] = sorted(sd._deceased_npcs.keys())
        report["protected_survived"] = all(
            nid in sd._alive_npcs() for nid in sd._protected_givers()
        )

        # Main-line reward track — simulate a player completion on
        # the first main-line beat. PB zoned profiles already ship a
        # ``lighthouse_mystery`` quest without Phase 3a tags, so we
        # add one under a stress-local id that carries the main-line
        # metadata. Shows reward accumulation works when the quest
        # actually has the quest_line tag.
        reva = engine.pie.npc_knowledge.profiles.get("captain_reva")
        if reva is not None:
            from npc_engine.knowledge import Quest
            reva.add_quest(Quest(
                id="stress_main_beat_0",
                name="Lighthouse Mystery (stress)",
                description="Find out why the lighthouse is dark.",
                quest_line="main_dark_lighthouse",
                quest_line_beat=0,
            ))
        sd.record_player_action(
            "Player solved the lighthouse mystery.",
            target="captain_reva",
            quest_completed="stress_main_beat_0",
        )
        report["quest_line_state"] = sd._quest_line_state.get(
            "main_dark_lighthouse"
        )
        report["reva_quest_statuses"] = [
            {"id": q.id, "status": q.status,
              "quest_line": q.quest_line, "beat": q.quest_line_beat}
            for q in reva.quests
        ] if reva is not None else []

        # Auto-refuse stress — enable it, set intents, dispatch a cruel
        # quest via LLM path is messy; instead use the synchronous
        # dispatch helper (real engine) which routes through the same
        # code path.
        sd.set_player_auto_refuse(["cruel", "dark"])
        ar_result = sd._dispatch_quest({
            "action": "quest",
            "npc_id": "old_bones",
            "quest": {
                "id": "stress_cruel", "name": "Cruel Stress",
                "description": "Test cruel quest auto-refuse path.",
                "intent": "cruel", "moral_weight": 0.95,
            },
        })
        report["auto_refuse_dispatch"] = ar_result
        old_bones = engine.pie.npc_knowledge.profiles.get("old_bones")
        q_ob = next((q for q in old_bones.quests if q.id == "stress_cruel"), None)
        report["auto_refused_quest_status"] = q_ob.status if q_ob else None

        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        # Restore or delete the quest_lines.yaml.
        if original_content is not None:
            qlines_path.write_text(original_content, encoding="utf-8")
        else:
            try:
                qlines_path.unlink()
            except Exception:
                pass


# ── Phase 5 — identity split end-to-end ──────────────────────────

def phase_5(world: str, out: Path) -> dict:
    engine, _, _ = boot_engine(world=world)
    sd = engine.story_director
    sd.set_narration_mode("terse")
    sd.set_active_zones(["dock_district", "lighthouse_bluffs"])

    report = {"phase": "5", "world": world, "steps": []}

    # Register a visible feature and introduce under two identities.
    sd.register_visible_feature("dragonslayer_cloak", "the_dragonslayer")
    sd.set_player_visible_feature("dragonslayer_cloak")
    intro1 = sd.introduce_player(
        to_npc="captain_reva", name="Jordan",
        titles=["the Dragonslayer"],
    )
    report["steps"].append({"step": "introduce_reva", "result": intro1})

    intro2 = sd.introduce_player(to_npc="thessa", name="Jordan")
    report["steps"].append({"step": "introduce_thessa", "result": intro2})

    vouch = sd.vouch_player_to(
        voucher_npc="captain_reva", to_npc="old_bones",
    )
    report["steps"].append({"step": "vouch_reva_to_old_bones", "result": vouch})

    # Player does something shady in front of varro + finn under a
    # different identity (cloak off).
    sd.set_player_visible_feature(None)
    action_result = sd.record_player_action(
        "Someone stole a loaf from the bakery.",
        target="old_bones",
        witness_npcs=["varro", "finn"],
        subject_identity="hooded_thief",
    )
    report["steps"].append({"step": "shady_deed", "result": action_result})

    # Run one live tick so the Director reacts.
    tick_rec = _tick(sd, "post_action_tick", actions_per_tick=3)
    report["steps"].append({"step": "live_tick", "result": tick_rec})

    # Reputation summary.
    rep = sd.get_player_reputation()
    report["reputation"] = rep

    # Hint for an unrecognized NPC (brom — never met).
    brom_hint = sd.build_reputation_hint_for_npc("brom")
    reva_hint = sd.build_reputation_hint_for_npc("captain_reva")
    report["brom_hint_present"] = brom_hint is not None
    report["reva_hint_suppressed"] = reva_hint is None
    report["brom_hint_preview"] = (brom_hint or "")[:240]

    # Postgen name guard e2e — hand-craft dialogue with "Jordan" and
    # verify validate_and_repair replaces it when the speaker
    # hasn't been introduced.
    from npc_engine.postgen import validate_and_repair
    import json as _json
    all_names = {"jordan", "the_dragonslayer"}
    # Brom never met the player.
    raw = _json.dumps({
        "dialogue": "Welcome, Jordan. The lighthouse calls me.",
        "emotion": "neutral", "action": None,
    })
    brom_profile = {"identity": {"name": "Brom", "role": "pilgrim"}}
    brom_known = {
        n.lower() for n in (
            engine.pie.npc_knowledge.profiles["brom"]
            .player_knowledge["known_as"]
        )
    }
    repaired_brom = validate_and_repair(
        raw, npc_id="brom", profile=brom_profile,
        all_player_names=all_names, player_known_names=brom_known,
    )
    report["postgen_brom_repaired"] = "Jordan" not in repaired_brom

    reva_known = {
        n.lower() for n in (
            engine.pie.npc_knowledge.profiles["captain_reva"]
            .player_knowledge["known_as"]
        )
    }
    repaired_reva = validate_and_repair(
        raw, npc_id="captain_reva", profile={"identity":
            {"name": "Captain Reva", "role": "harbor_master"}},
        all_player_names=all_names, player_known_names=reva_known,
    )
    report["postgen_reva_kept"] = "Jordan" in repaired_reva

    # Per-NPC player_knowledge snapshot for eyeballing.
    report["per_npc_player_knowledge"] = {
        nid: {
            "met": npc.player_knowledge.get("met"),
            "recognized": npc.player_knowledge.get("recognized"),
            "known_as": list(npc.player_knowledge.get("known_as", [])),
            "witnessed_n": len(npc.player_knowledge.get("witnessed_deeds", [])),
            "heard_n": len(npc.player_knowledge.get("heard_deeds", [])),
        }
        for nid, npc in engine.pie.npc_knowledge.profiles.items()
    }

    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ── Entry point ──────────────────────────────────────────────────

PHASES = {
    "4a": phase_4a,
    "4b": phase_4b,
    "4c": phase_4c,
    "3a": phase_3a,
    "5":  phase_5,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=list(PHASES.keys()), required=True)
    p.add_argument("--world", default="port_blackwater_zoned")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = Path(args.out) if args.out else NPC_ROOT / "logs" / f"stress_phase{args.phase}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"=== Stress phase {args.phase} on {args.world} ===")
    report = PHASES[args.phase](args.world, out)
    # Print a short summary.
    print(f"--- wrote {out} ---")
    print(json.dumps(
        {k: v for k, v in report.items()
          if k in ("phase", "world", "dungeon_kind_counts",
                    "max_unoffered_observed", "cap_respected",
                    "ratio_ml_to_non_ml", "protected_survived",
                    "activity_hints", "confront_accelerator_hint",
                    "brom_hint_present", "reva_hint_suppressed",
                    "postgen_brom_repaired", "postgen_reva_kept")},
        indent=2,
    ))


if __name__ == "__main__":
    main()
