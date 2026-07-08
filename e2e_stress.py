#!/usr/bin/env python3
"""
End-to-end stress scenarios for the Story Director.

Each scenario simulates a realistic game-client flow against a real
Qwen 2.5 3B boot, asserts state at every step, and writes a JSON
report to ``logs/e2e_<scenario>.json``.

Usage:
    python e2e_stress.py --scenario gameplay
    python e2e_stress.py --scenario persistence
    python e2e_stress.py --scenario refusal_decay
    python e2e_stress.py --scenario live_dialogue
    python e2e_stress.py --scenario identity_accuracy
    python e2e_stress.py --scenario activity_transitions
    python e2e_stress.py --scenario scale_100

Scenarios can be composed — run ``--scenario all`` to execute every
non-scale scenario in sequence.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

NPC_ROOT = Path(__file__).parent.resolve()
PIE_ROOT = (NPC_ROOT.parent / "plug-in-intelligence-engine").resolve()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NPC_ENGINE_DEV_MODE", "1")
logging.basicConfig(level=logging.WARNING)
for n in ["httpx", "huggingface_hub", "sentence_transformers", "faiss",
          "tqdm", "llama_cpp", "engine.npc_knowledge", "engine.npc_capabilities"]:
    logging.getLogger(n).setLevel(logging.ERROR)

sys.path.insert(0, str(NPC_ROOT))
sys.path.insert(0, str(PIE_ROOT))


# Re-use the boot helper from stress_director. Importing before we
# touch sys.stdout avoids a double-wrap (stress_director's module-
# level TextIOWrapper would otherwise land on top of ours and close
# the inner on GC).
from stress_director import boot_engine, _tick  # noqa: E402

# Ensure UTF-8 stdout for the scenario prints. If stress_director
# already wrapped, leave it — reconfigure is idempotent.
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass


# ── Check helpers ────────────────────────────────────────────────

class StepLog:
    """Collect step-by-step check results and pretty-print on
    teardown. Every ``.check(cond, label, detail)`` call records a
    PASS/FAIL so the scenario report has a readable trail even when
    the underlying state is complex."""

    def __init__(self, scenario: str):
        self.scenario = scenario
        self.checks: list[dict] = []
        self.steps: list[dict] = []
        self.failed = 0

    def check(self, cond: bool, label: str, detail=None):
        result = {"label": label, "ok": bool(cond)}
        if detail is not None:
            result["detail"] = detail
        self.checks.append(result)
        if not cond:
            self.failed += 1
            print(f"  [FAIL] {label}: {detail!r}")
        else:
            print(f"  [PASS] {label}")

    def step(self, name: str, payload=None):
        self.steps.append({"name": name, "payload": payload})
        print(f"=== {self.scenario} :: {name} ===")

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "failed": self.failed,
            "check_count": len(self.checks),
            "checks": self.checks,
            "steps": self.steps,
        }


def _write_quest_lines(world: str, body: dict) -> tuple[Path, str | None]:
    """Write body to <world>/quest_lines.yaml. Returns the path and
    the prior content (or None). Caller restores in a finally."""
    import yaml as _yaml
    world_dir = NPC_ROOT / "data" / "worlds" / world
    p = world_dir / "quest_lines.yaml"
    prior = p.read_text(encoding="utf-8") if p.exists() else None
    p.write_text(_yaml.safe_dump(body), encoding="utf-8")
    return p, prior


def _restore_quest_lines(path: Path, prior: str | None):
    if prior is None:
        try:
            path.unlink()
        except Exception:
            pass
    else:
        path.write_text(prior, encoding="utf-8")


# ── Scenario 1 — 30-tick scripted gameplay arc ───────────────────

def scenario_gameplay(out: Path) -> dict:
    """Full-session e2e. Runs a scripted timeline against PB zoned
    with an authored main-line, exercising every Phase 3a/4/5
    feature. Every step asserts the expected state transition."""
    world = "port_blackwater_zoned"
    log = StepLog("gameplay")

    # Configure a main-line with a decay-refusal cruel beat so we
    # can stress both main-line progression AND refusal decay in
    # one run.
    qlines_path, prior = _write_quest_lines(world, {
        "quest_lines": {
            "main_test": {
                "type": "main",
                "title": "Main test line",
                "beats": [
                    {"quest_id": "gp_beat_0", "giver": "captain_reva"},
                    {"quest_id": "gp_beat_1", "giver": "thessa",
                      "requires": ["gp_beat_0"]},
                ],
                "protected_givers": ["captain_reva", "thessa", "brom"],
                "reward_track": ["Harbor seal", "Sea-tongue charm"],
            },
        },
        "director": {
            "quest_auto_refuse": {
                "enabled": True, "player_configurable": True,
            },
        },
    })

    try:
        engine, _, _ = boot_engine(world=world)
        sd = engine.story_director
        sd.set_narration_mode("terse")
        sd.set_active_zones(["dock_district", "lighthouse_bluffs"])

        # Tick 1–2: baseline in_town tick, then player introduces.
        log.step("tick_1_baseline")
        rec = _tick(sd, "t1", actions_per_tick=3)
        log.check(not rec["paused"], "t1 live tick", rec)
        log.check(sd.tick_count == 1, "tick_count advanced to 1")
        log.check(sd._player_activity == "unknown", "default activity")

        log.step("introduce_player_to_reva")
        r = sd.introduce_player(to_npc="captain_reva", name="Jordan")
        log.check(r["ok"], "introduce ok")
        reva_pk = engine.pie.npc_knowledge.profiles["captain_reva"].player_knowledge
        log.check(reva_pk["met"] and reva_pk["recognized"],
                   "reva met + recognized")
        log.check("jordan" in reva_pk["known_as"], "reva knows jordan")

        # Tick 3: register a visible feature + equip. Introduce
        # thessa WITH cloak on → thessa auto-recognizes both
        # identities.
        log.step("register_feature_and_equip")
        sd.register_visible_feature("dragonslayer_cloak", "the_dragonslayer")
        sd.set_player_visible_feature("dragonslayer_cloak")
        sd.introduce_player(to_npc="thessa", name="Jordan")
        thessa_pk = engine.pie.npc_knowledge.profiles["thessa"].player_knowledge
        log.check("the_dragonslayer" in thessa_pk["known_as"],
                   "thessa auto-recognized the_dragonslayer via feature")
        log.check("jordan" in thessa_pk["known_as"], "thessa knows jordan")

        log.step("tick_3_live")
        _tick(sd, "t3", actions_per_tick=3)
        log.check(sd.tick_count == 2, f"tick_count now 2 got {sd.tick_count}")

        # Tick 4: cloak off, shady deed under "hooded_killer" in
        # front of varro + finn.
        log.step("shady_deed_no_cloak")
        sd.set_player_visible_feature(None)
        r = sd.record_player_action(
            "A hooded figure stabbed Old Bones in the alley.",
            target="old_bones",
            witness_npcs=["varro", "finn"],
            subject_identity="hooded_killer",
        )
        log.check(r["ok"], "record_player_action ok")
        varro_pk = engine.pie.npc_knowledge.profiles["varro"].player_knowledge
        finn_pk = engine.pie.npc_knowledge.profiles["finn"].player_knowledge
        log.check("hooded_killer" in varro_pk["known_as"],
                   "varro recognized hooded_killer")
        log.check("hooded_killer" in finn_pk["known_as"],
                   "finn recognized hooded_killer")
        log.check("jordan" not in varro_pk["known_as"],
                   "varro does NOT know jordan (no introduction)")
        log.check("jordan" not in finn_pk["known_as"],
                   "finn does NOT know jordan")
        brom_pk = engine.pie.npc_knowledge.profiles["brom"].player_knowledge
        log.check(not brom_pk["met"],
                   "brom never met, just heard the gossip")
        log.check(len(brom_pk["heard_deeds"]) >= 1,
                   "brom has heard_deeds entry")

        # Tick 5: live LLM tick — Director should react to the
        # shady deed.
        log.step("tick_5_reactive")
        pre_tick = sd.tick_count
        _tick(sd, "t5", actions_per_tick=3)
        log.check(sd.tick_count == pre_tick + 1,
                   "tick advanced from reactive")

        # Tick 6: player REFUSES a cruel quest (decay mode).
        log.step("refuse_cruel_quest_decay")
        old_bones = engine.pie.npc_knowledge.profiles["old_bones"]
        from npc_engine.knowledge import Quest
        old_bones.add_quest(Quest(
            id="gp_cruel", name="Cruel Task", description="d",
            intent="cruel", moral_weight=0.8,
            refusal_mode="decay", refusal_decay_ticks=4,
        ))
        r = sd.process_refusal(quest_id="gp_cruel", npc_id="old_bones")
        log.check(r["ok"], "refusal ok")
        log.check(r["trust_delta"] == int(0.8 * -15),
                   f"trust delta scales w/moral weight, got {r['trust_delta']}")
        q = next(q for q in old_bones.quests if q.id == "gp_cruel")
        log.check(q.status == "refused", "quest status=refused")
        log.check(("old_bones", "gp_cruel") in sd._refused_quest_timers,
                   "decay timer scheduled")

        # Tick 7–11: live ticks to advance time; refusal should
        # reopen at tick (tick_count_at_refusal + 1 + 4).
        pre_ref_tick = r.get("unlock_tick")
        log.step("advance_ticks_toward_refusal_unlock")
        log.check(pre_ref_tick is not None, "unlock_tick present",
                   pre_ref_tick)
        # Run enough ticks to cross the unlock boundary.
        for i in range(pre_ref_tick - sd.tick_count + 2):
            _tick(sd, f"advance_{i}", actions_per_tick=1)
        q = next(q for q in old_bones.quests if q.id == "gp_cruel")
        log.check(q.status == "available",
                   f"decayed refusal reopened (status={q.status})")
        log.check(("old_bones", "gp_cruel") not in sd._refused_quest_timers,
                   "decay timer cleared after reopen")

        # Tick N: complete the main-line beat-0 (stress-local).
        #
        # record_player_action(quest_completed=...) routes through
        # ``engine.complete_quest`` which updates the player_quests
        # tracker but does NOT auto-flip the NPC's quest status to
        # "completed" unless the quest is on player_quests.active.
        # The reward accumulator reads Quest dataclass fields (via
        # _find_quest_by_id) so it still works. To exercise the
        # prereq unlock path below we explicitly flip status to
        # completed — that's what the game client does via
        # engine.player_quests.complete(...) in production.
        log.step("complete_main_line_beat_0")
        reva = engine.pie.npc_knowledge.profiles["captain_reva"]
        reva.add_quest(Quest(
            id="gp_beat_0", name="Beat 0", description="d",
            quest_line="main_test", quest_line_beat=0,
        ))
        sd.record_player_action(
            "Solved the harbor puzzle.",
            target="captain_reva",
            quest_completed="gp_beat_0",
        )
        # engine.complete_quest now flips NPC Quest.status directly
        # (even without a formal accept step), so no manual flip
        # needed. Verify the status actually flipped.
        q_beat0 = next(
            (q for q in reva.quests if q.id == "gp_beat_0"), None
        )
        log.check(
            q_beat0 is not None and q_beat0.status == "completed",
            "engine.complete_quest flipped NPC status",
            q_beat0.status if q_beat0 else "quest not found",
        )
        st = sd._quest_line_state.get("main_test")
        log.check(st is not None, "main_test line state present")
        log.check("gp_beat_0" in (st or {}).get("completed_quests", []),
                   "beat 0 in completed_quests")
        log.check("Harbor seal" in (st or {}).get("rewards_earned", []),
                   "reward 0 earned", st)
        log.check((st or {}).get("line_status") == "active",
                   "line still active (1/2 beats)")

        # Prereq unlock: add gp_beat_1 BEFORE completion so it starts locked,
        # then the next lifecycle sweep should unlock because gp_beat_0
        # is now completed.
        log.step("prereq_unlock_beat_1")
        thessa = engine.pie.npc_knowledge.profiles["thessa"]
        # Dispatch via the Director to fire prereq gating.
        r = sd._dispatch_quest({
            "action": "quest", "npc_id": "thessa",
            "quest": {
                "id": "gp_beat_1", "name": "Beat 1",
                "description": "Second beat.",
                "quest_line": "main_test", "quest_line_beat": 1,
                "prerequisite_quests": ["non_existent_prereq"],
            },
        })
        log.check(r["ok"] and r["status"] == "locked",
                   "beat 1 locked (unknown prereq)", r)
        # Patch the prereq to the completed one, re-sweep.
        q1 = next(q for q in thessa.quests if q.id == "gp_beat_1")
        q1.prerequisite_quests = ["gp_beat_0"]
        unlocked = sd._unlock_quests_if_prereqs_met()
        log.check(q1.status == "available", "beat 1 unlocked post-sweep")
        log.check(any(u["quest_id"] == "gp_beat_1" for u in unlocked),
                   "unlock record present")

        # Pause → tick — must not bump tick_count.
        log.step("pause_then_tick")
        pre_tick = sd.tick_count
        sd.pause_ticks()
        rec = _tick(sd, "paused", actions_per_tick=1)
        log.check(rec["paused"] and rec["paused_reason"] == "explicit_pause",
                   "paused response", rec)
        log.check(sd.tick_count == pre_tick,
                   f"tick_count unchanged (was {pre_tick})")

        sd.resume_ticks()
        _tick(sd, "post_resume", actions_per_tick=1)
        log.check(sd.tick_count == pre_tick + 1, "tick advances after resume")

        # Activity transitions: in_dungeon (drops quest) → in_combat
        # (paused, but BUMPS tick per Phase 4a design) → in_town.
        log.step("activity_transitions")
        sd.set_player_activity("in_dungeon")
        pre_tick = sd.tick_count
        sd._kind_rotation_index = 1  # force quest slot
        rec = _tick(sd, "dungeon", actions_per_tick=1)
        log.check(rec["action_kind"] in ("fact", "event"),
                   f"dungeon dropped quest, got {rec['action_kind']}")

        sd.set_player_activity("in_combat")
        pre_tick2 = sd.tick_count
        rec = _tick(sd, "combat", actions_per_tick=1)
        log.check(rec["paused"] is True and rec["paused_reason"] == "activity:in_combat",
                   "combat paused", rec)
        log.check(sd.tick_count == pre_tick2 + 1,
                   "combat BUMPS tick_count per Phase 4a design",
                   f"{pre_tick2} -> {sd.tick_count}")

        sd.set_player_activity("in_town")
        _tick(sd, "post_combat", actions_per_tick=3)

        # Reputation snapshot at end.
        log.step("final_reputation")
        rep = sd.get_player_reputation()
        log.check("hooded_killer" in rep["known_identities"],
                   "hooded_killer identity recorded")
        log.check("jordan" in rep["known_identities"],
                   "jordan identity recorded")
        log.check(rep["total_npcs_who_recognize_you"] >= 4,
                   f"≥4 NPCs recognize player, got {rep['total_npcs_who_recognize_you']}")

        log.steps.append({"name": "final_state", "payload": {
            "tick_count": sd.tick_count,
            "ledger_size": len(sd.ledger.entries),
            "arcs_active": len(sd.arc_planner.active_arcs()),
            "quest_line_state": sd._quest_line_state,
            "reputation": rep,
            "refused_timers": [
                [n, q, t] for (n, q), t in sd._refused_quest_timers.items()
            ],
            "per_npc_player_knowledge": {
                nid: {
                    "met": npc.player_knowledge["met"],
                    "recognized": npc.player_knowledge["recognized"],
                    "known_as": list(npc.player_knowledge["known_as"]),
                }
                for nid, npc in engine.pie.npc_knowledge.profiles.items()
            },
        }})

        report = log.to_dict()
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        _restore_quest_lines(qlines_path, prior)


# ── Scenario 2 — cross-session persistence round-trip ────────────

def scenario_persistence(out: Path) -> dict:
    """Run a rich state setup, tear the engine down, boot fresh,
    assert every Phase 3a/4/5 field survived."""
    world = "port_blackwater_zoned"
    log = StepLog("persistence")

    qlines_path, prior = _write_quest_lines(world, {
        "quest_lines": {
            "persist_line": {
                "type": "main",
                "title": "Persistence",
                "beats": [{"quest_id": "p_beat_0", "giver": "captain_reva"}],
                "protected_givers": ["captain_reva"],
                "reward_track": ["Persistence reward"],
            }
        },
        "director": {
            "quest_auto_refuse": {
                "enabled": True, "player_configurable": True,
            },
        },
    })

    try:
        log.step("session_1_setup")
        engine, _, _ = boot_engine(world=world)
        sd = engine.story_director
        sd.set_narration_mode("terse")
        sd.set_active_zones(["dock_district", "lighthouse_bluffs"])

        # Introduce, feature, refuse-decay, quest-line setup.
        sd.register_visible_feature("cloak_red", "the_red")
        sd.set_player_visible_feature("cloak_red")
        sd.introduce_player(to_npc="captain_reva", name="Jordan",
                             titles=["the Red"])
        sd.set_quest_pacing(max_unoffered=3, cooldown_ticks=7)
        sd.set_tick_budget(12.5)
        sd.set_player_auto_refuse(["cruel"])

        # Witness a deed.
        sd.record_player_action(
            "The red one took a crate from the dock.",
            target="finn",
            witness_npcs=["finn"],
            subject_identity="the_red",
        )

        # Refuse a decay-mode quest.
        old_bones = engine.pie.npc_knowledge.profiles["old_bones"]
        from npc_engine.knowledge import Quest
        old_bones.add_quest(Quest(
            id="persist_ref", name="Ref", description="d",
            intent="gray", moral_weight=0.5,
            refusal_mode="decay", refusal_decay_ticks=6,
        ))
        sd.process_refusal(quest_id="persist_ref", npc_id="old_bones")

        # Lock-pause before teardown.
        sd.pause_ticks()

        # Snapshot everything we expect to round-trip.
        snapshot = {
            "tick_count": sd.tick_count,
            "paused": sd._paused,
            "paused_at_tick": sd._paused_at_tick,
            "tick_budget_seconds": sd._tick_budget_seconds,
            "player_visible_feature": sd._player_visible_feature,
            "visible_feature_to_identity": dict(sd._visible_feature_to_identity),
            "player_auto_refuse_intents": sorted(sd._player_auto_refuse_intents),
            "quest_line_state": {k: dict(v) for k, v in sd._quest_line_state.items()},
            "refused_quest_timers": sorted(
                ((n, q, t) for (n, q), t in sd._refused_quest_timers.items())
            ),
            "npc_player_identity_trust": {
                k: dict(v) for k, v in sd._npc_player_identity_trust.items()
            },
            "max_unoffered_override": sd._max_unoffered_quests_override,
            "quest_cooldown_override": sd._quest_cooldown_ticks_override,
            "reva_known_as": list(
                engine.pie.npc_knowledge.profiles["captain_reva"]
                    .player_knowledge["known_as"]
            ),
            "finn_witnessed": list(
                engine.pie.npc_knowledge.profiles["finn"]
                    .player_knowledge["witnessed_deeds"]
            ),
            "old_bones_quests": [
                (q.id, q.status, q.refusal_mode)
                for q in old_bones.quests
                if q.id == "persist_ref"
            ],
        }
        # Force a save — the pause_ticks call above already saved,
        # but be explicit.
        sd._save_state()
        # Drop references to the engine (simulate process exit).
        del sd
        del engine

        # Session 2 — boot fresh and compare.
        log.step("session_2_boot_and_compare")
        engine2, _, _ = boot_engine(world=world, reset=False)
        sd2 = engine2.story_director
        log.check(sd2.tick_count == snapshot["tick_count"],
                   "tick_count survived",
                   (sd2.tick_count, snapshot["tick_count"]))
        log.check(sd2._paused is True, "paused flag survived")
        log.check(sd2._paused_at_tick == snapshot["paused_at_tick"],
                   "paused_at_tick survived")
        log.check(abs(sd2._tick_budget_seconds - 12.5) < 1e-6,
                   "tick_budget survived", sd2._tick_budget_seconds)
        log.check(sd2._player_visible_feature == "cloak_red",
                   "visible_feature survived")
        log.check(sd2._visible_feature_to_identity.get("cloak_red") == "the_red",
                   "feature registry survived")
        log.check("cruel" in sd2._player_auto_refuse_intents,
                   "auto_refuse intents survived")
        log.check("persist_line" in sd2._quest_line_state,
                   "quest_line_state survived")
        log.check(sd2._max_unoffered_quests_override == 3,
                   "quest_pacing override survived")
        log.check(sd2._quest_cooldown_ticks_override == 7,
                   "cooldown override survived")
        log.check(("old_bones", "persist_ref") in sd2._refused_quest_timers,
                   "refused_quest_timers survived",
                   list(sd2._refused_quest_timers.keys()))
        reva2_pk = engine2.pie.npc_knowledge.profiles["captain_reva"].player_knowledge
        # Phase 5a — player_knowledge now persisted in state.json.
        # known_as should survive a fresh boot.
        log.check(set(reva2_pk["known_as"]) >= {"jordan", "the_red"},
                   "player_knowledge survived",
                   reva2_pk["known_as"])
        log.check(reva2_pk["met"] is True, "met survived")
        log.check(reva2_pk["recognized"] is True, "recognized survived")
        finn2_pk = engine2.pie.npc_knowledge.profiles["finn"].player_knowledge
        log.check(len(finn2_pk["witnessed_deeds"]) >= 1,
                   "finn witnessed_deeds survived",
                   finn2_pk["witnessed_deeds"])
        # Identity trust IS persisted though.
        log.check(
            "jordan" in sd2._npc_player_identity_trust.get("captain_reva", {}),
            "identity trust survived",
            sd2._npc_player_identity_trust,
        )

        # Tick after resume — real LLM call to confirm engine boots
        # cleanly post-restore.
        sd2.resume_ticks()
        rec = _tick(sd2, "post_restore_tick", actions_per_tick=1)
        log.check(not rec["paused"],
                   "post-restore tick fires", rec)

        log.steps.append({"name": "snapshot", "payload": snapshot})
        log.steps.append({"name": "session2_after_resume", "payload": {
            "tick_count": sd2.tick_count,
            "paused": sd2._paused,
        }})
        report = log.to_dict()
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        _restore_quest_lines(qlines_path, prior)


# ── Scenario 3 — refusal decay on live LLM ───────────────────────

def scenario_refusal_decay(out: Path) -> dict:
    world = "port_blackwater_zoned"
    log = StepLog("refusal_decay")
    engine, _, _ = boot_engine(world=world)
    sd = engine.story_director
    sd.set_narration_mode("terse")
    sd.set_active_zones(["dock_district", "lighthouse_bluffs"])

    from npc_engine.knowledge import Quest
    old_bones = engine.pie.npc_knowledge.profiles["old_bones"]
    finn = engine.pie.npc_knowledge.profiles["finn"]
    old_bones.add_quest(Quest(
        id="decay_q", name="Decay", description="d",
        intent="gray", moral_weight=0.4,
        refusal_mode="decay", refusal_decay_ticks=3,
    ))
    finn.add_quest(Quest(
        id="perm_q", name="Perm", description="d",
        intent="gray", moral_weight=0.4,
        refusal_mode="permanent",
    ))

    log.step("refuse_both")
    r_decay = sd.process_refusal(quest_id="decay_q", npc_id="old_bones")
    r_perm = sd.process_refusal(quest_id="perm_q", npc_id="finn")
    log.check(r_decay["unlock_tick"] is not None,
               "decay unlock_tick set", r_decay["unlock_tick"])
    log.check(r_perm["unlock_tick"] is None,
               "permanent refusal has no unlock_tick")

    unlock = r_decay["unlock_tick"]
    # Run enough live ticks to cross the unlock boundary.
    log.step("run_ticks_to_unlock")
    for i in range(unlock - sd.tick_count + 2):
        _tick(sd, f"t{i}", actions_per_tick=1)
    q_decay = next(q for q in old_bones.quests if q.id == "decay_q")
    log.check(q_decay.status == "available",
               f"decay refusal reopened (status={q_decay.status})")
    q_perm = next(q for q in finn.quests if q.id == "perm_q")
    log.check(q_perm.status == "refused",
               f"permanent refusal stays refused (status={q_perm.status})")

    # Further ticks — permanent must stay refused indefinitely.
    log.step("drive_more_ticks")
    for i in range(5):
        _tick(sd, f"tmore_{i}", actions_per_tick=1)
    q_perm = next(q for q in finn.quests if q.id == "perm_q")
    log.check(q_perm.status == "refused",
               "permanent still refused after extra ticks")

    log.steps.append({"name": "state_end", "payload": {
        "decay_final_status": q_decay.status,
        "perm_final_status": q_perm.status,
        "refused_timers": list(sd._refused_quest_timers.keys()),
        "tick_count": sd.tick_count,
    }})
    report = log.to_dict()
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ── Scenario 4 — live NPC dialogue through postgen guard ─────────

def scenario_live_dialogue(out: Path) -> dict:
    world = "port_blackwater_zoned"
    log = StepLog("live_dialogue")
    engine, _, _ = boot_engine(world=world)
    sd = engine.story_director
    sd.set_narration_mode("terse")

    # Introduce player only to captain_reva. brom stays unrecognized.
    sd.introduce_player(to_npc="captain_reva", name="Jordan")

    # Call engine.process() so we get a real end-to-end NPC dialogue
    # through PIE + capabilities + postgen. The base path doesn't
    # know about player_known_names unless the caller threads them
    # in explicitly — we're validating that the postgen HELPERS work
    # correctly when the game client integrates them.
    log.step("reva_dialogue_with_name")
    try:
        reva_response = engine.process("Hello, it's Jordan.",
                                         npc_id="captain_reva")
        log.check(bool(reva_response),
                   "reva responded", (reva_response or "")[:200])
    except Exception as e:
        log.check(False, "reva dialogue errored", str(e))
        reva_response = ""

    log.step("brom_dialogue_unrecognized")
    try:
        brom_response = engine.process("Hello, it's Jordan.",
                                         npc_id="brom")
        log.check(bool(brom_response),
                   "brom responded", (brom_response or "")[:200])
    except Exception as e:
        log.check(False, "brom dialogue errored", str(e))
        brom_response = ""

    # engine.process() now auto-threads player_known_names and
    # all_player_names into validate_and_repair (wired in engine.py).
    # The deterministic test below uses hand-crafted JSON to verify
    # the guard logic independently of model randomness — the model
    # may or may not use "Jordan" in free-form output, so checking
    # the live response is flaky. This manual path proves the
    # helpers fire correctly regardless.
    from npc_engine.postgen import validate_and_repair
    all_names = {"jordan"}
    # Reva's known_as includes jordan.
    reva_known = {
        n.lower() for n in (
            engine.pie.npc_knowledge.profiles["captain_reva"]
                .player_knowledge["known_as"]
        )
    }
    brom_known: set = set()

    # Hand-craft dialogue using the player name to exercise the guard
    # (model output varies; this is deterministic).
    reva_raw = json.dumps({
        "dialogue": "Good morning, Jordan. The docks are quiet.",
        "emotion": "friendly", "action": None,
    })
    brom_raw = json.dumps({
        "dialogue": "Welcome, Jordan. The lighthouse calls to me.",
        "emotion": "neutral", "action": None,
    })
    reva_profile = {"identity": {"name": "Captain Reva",
                                   "role": "harbor master"}}
    brom_profile = {"identity": {"name": "Brom", "role": "pilgrim"}}
    reva_repaired = validate_and_repair(
        reva_raw, npc_id="captain_reva", profile=reva_profile,
        all_player_names=all_names, player_known_names=reva_known,
    )
    brom_repaired = validate_and_repair(
        brom_raw, npc_id="brom", profile=brom_profile,
        all_player_names=all_names, player_known_names=brom_known,
    )
    log.check("Jordan" in reva_repaired,
               "reva kept Jordan (recognized)", reva_repaired[:200])
    log.check("Jordan" not in brom_repaired,
               "brom stripped Jordan (unrecognized)", brom_repaired[:200])
    log.check("tranger" in brom_repaired.lower() or "Stranger" in brom_repaired,
               "brom rewrote to stranger", brom_repaired[:200])

    log.steps.append({"name": "live_responses", "payload": {
        "reva_response": reva_response[:300],
        "brom_response": brom_response[:300],
        "reva_repaired": reva_repaired,
        "brom_repaired": brom_repaired,
    }})
    report = log.to_dict()
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ── Scenario 5 — identity accuracy under mixed witnesses ─────────

def scenario_identity_accuracy(out: Path) -> dict:
    world = "port_blackwater_zoned"
    log = StepLog("identity_accuracy")
    engine, _, _ = boot_engine(world=world)
    sd = engine.story_director
    sd.set_narration_mode("terse")
    sd.set_active_zones(["dock_district", "lighthouse_bluffs"])

    # Deed 1: varro + finn witness under "hooded_killer".
    sd.record_player_action(
        "A hooded figure stabbed a merchant on the dock.",
        target="old_bones",
        witness_npcs=["varro", "finn"],
        subject_identity="hooded_killer",
    )
    # Deed 2: captain_reva witnesses under "jordan".
    sd.record_player_action(
        "Jordan delivered the harbor manifest on time.",
        target="captain_reva",
        witness_npcs=["captain_reva"],
        subject_identity="jordan",
    )
    # Deed 3: thessa witnesses at the lighthouse under "the_pilgrim".
    sd.record_player_action(
        "A quiet pilgrim lit a candle at the bluffs.",
        target="brom",
        witness_npcs=["thessa"],
        subject_identity="the_pilgrim",
    )

    log.step("assert_per_npc_known_as")
    expected = {
        "varro": {"hooded_killer"},
        "finn": {"hooded_killer"},
        "captain_reva": {"jordan"},
        "thessa": {"the_pilgrim"},
        "old_bones": set(),   # non-witness for all 3
        "brom": set(),        # non-witness for all 3
    }
    for npc_id, want in expected.items():
        pk = engine.pie.npc_knowledge.profiles[npc_id].player_knowledge
        got = set(pk["known_as"])
        log.check(
            got == want,
            f"{npc_id}: known_as matches exactly",
            {"got": sorted(got), "want": sorted(want)},
        )

    # Non-witness NPCs should have heard_deeds for every deed they
    # didn't witness.
    log.step("assert_non_witness_heard_all")
    # Total entries added: 3. Each non-witness should have heard
    # all 3 (they were added one-by-one, and the fallback propagates
    # to every non-witness at the time of the record).
    for npc_id in ("old_bones", "brom"):
        pk = engine.pie.npc_knowledge.profiles[npc_id].player_knowledge
        log.check(
            len(pk["heard_deeds"]) == 3,
            f"{npc_id} heard all 3 deeds",
            pk["heard_deeds"],
        )

    # Reputation grouping should split cleanly by identity.
    log.step("reputation_grouping")
    rep = sd.get_player_reputation()
    for ident in ("hooded_killer", "jordan", "the_pilgrim"):
        log.check(
            ident in rep["known_identities"],
            f"identity {ident} present in reputation",
            list(rep["known_identities"].keys()),
        )
    log.check(
        set(rep["known_identities"]["hooded_killer"]["known_by"]) == {"varro", "finn"},
        "hooded_killer known_by exactly varro+finn",
    )
    log.check(
        rep["known_identities"]["jordan"]["known_by"] == ["captain_reva"],
        "jordan known_by exactly captain_reva",
    )
    log.check(
        rep["known_identities"]["the_pilgrim"]["known_by"] == ["thessa"],
        "the_pilgrim known_by exactly thessa",
    )

    # A hint for brom — unrecognized, heard all 3 — should reference
    # every identity.
    log.step("brom_hint_covers_identities")
    hint = sd.build_reputation_hint_for_npc("brom")
    log.check(hint is not None, "brom hint present")
    if hint:
        for name in ("hooded killer", "jordan", "the pilgrim"):
            log.check(name in hint,
                       f"hint mentions {name!r}", hint[:400])

    log.steps.append({"name": "final", "payload": {
        "reputation": rep,
        "brom_hint": hint,
    }})
    report = log.to_dict()
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ── Scenario 6 — activity transitions under load ─────────────────

def scenario_activity_transitions(out: Path) -> dict:
    world = "port_blackwater_zoned"
    log = StepLog("activity_transitions")
    engine, _, _ = boot_engine(world=world)
    sd = engine.story_director
    sd.set_narration_mode("terse")
    sd.set_active_zones(["dock_district", "lighthouse_bluffs"])

    # Recorded sequence: [(activity, expected_paused, expects_tick_bump)]
    sequence = [
        ("in_town",     False, True),
        ("in_combat",   True,  True),     # 4a activity pause bumps
        ("in_town",     False, True),
        ("in_dialogue", False, True),
        ("in_dungeon",  False, True),
        ("idle",        True,  True),
        ("in_menu",     True,  True),
        ("in_town",     False, True),
        ("wandering",   False, True),
    ]
    pre_tick = sd.tick_count
    for act, expect_paused, expect_bump in sequence:
        sd.set_player_activity(act)
        rec = _tick(sd, f"act_{act}", actions_per_tick=1)
        log.check(
            bool(rec["paused"]) == expect_paused,
            f"activity={act} paused={expect_paused}",
            rec,
        )
        log.check(
            sd.tick_count == pre_tick + 1,
            f"activity={act} tick bumped",
            f"{pre_tick} -> {sd.tick_count}",
        )
        pre_tick = sd.tick_count

    # Now interleave an explicit pause — tick_count should NOT bump.
    log.step("explicit_pause_during_town")
    sd.set_player_activity("in_town")
    sd.pause_ticks()
    pre_tick = sd.tick_count
    rec = _tick(sd, "paused_during_town", actions_per_tick=1)
    log.check(rec["paused"] and rec["paused_reason"] == "explicit_pause",
               "explicit pause takes precedence", rec)
    log.check(sd.tick_count == pre_tick,
               f"explicit pause does NOT bump tick ({pre_tick})")

    sd.resume_ticks()
    log.steps.append({"name": "final_tick_count", "payload": sd.tick_count})
    report = log.to_dict()
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ── Scenario 7 — scale test on synthetic_100 ─────────────────────

def scenario_scale_100(out: Path) -> dict:
    world = "synthetic_100"
    log = StepLog("scale_100")
    world_dir = NPC_ROOT / "data" / "worlds" / world
    if not world_dir.exists():
        print(f"[SKIP] {world} not present; run generate_synthetic_world.py first")
        return log.to_dict()
    engine, _, _ = boot_engine(world=world)
    sd = engine.story_director
    sd.set_narration_mode("terse")

    # No zones for synthetic — keep rotation uniform.
    ticks: list[dict] = []
    for i in range(10):
        rec = _tick(sd, f"t{i+1}", actions_per_tick=3)
        ticks.append(rec)
    cast_size = len(engine.pie.npc_knowledge.profiles)
    log.check(cast_size >= 100, "cast size ≥ 100", cast_size)
    avg_latency = sum(t["elapsed_s"] for t in ticks) / max(1, len(ticks))
    log.check(avg_latency < 60, "avg tick < 60 s", avg_latency)
    # Check rotation breadth — how many distinct NPCs did the
    # architect plan pick?
    touched = set(sd._npc_last_planned_tick.keys())
    log.check(len(touched) >= 15, "rotation touched ≥15 NPCs in 10 ticks",
               len(touched))
    log.steps.append({"name": "ticks", "payload": ticks})
    log.steps.append({"name": "summary", "payload": {
        "cast_size": cast_size,
        "touched_count": len(touched),
        "avg_latency_s": avg_latency,
        "ledger_size": len(sd.ledger.entries),
    }})
    report = log.to_dict()
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def scenario_game_reset(out: Path) -> dict:
    """Drive a rich mid-session state (ticks, birth, death, identity,
    quests, refusals, pause), call reset, then verify every subsystem
    returns to its initial-cast baseline."""
    world = "port_blackwater_zoned"
    log = StepLog("game_reset")
    engine, _, _ = boot_engine(world=world)
    sd = engine.story_director
    sd.set_narration_mode("terse")
    sd.set_active_zones(["dock_district", "lighthouse_bluffs"])

    log.step("accumulate_state")
    # Run 5 real ticks to accumulate ledger, arcs, rotation state.
    for i in range(5):
        _tick(sd, f"t{i+1}", actions_per_tick=3)
    log.check(sd.tick_count == 5, f"tick_count=5, got {sd.tick_count}")
    log.check(len(sd.ledger.entries) > 0, "ledger non-empty")

    # Introduce, deed, refusal, identity trust.
    sd.introduce_player(to_npc="captain_reva", name="Jordan")
    sd.register_visible_feature("cloak", "hero")
    sd.set_player_visible_feature("cloak")
    sd.record_player_action(
        "A hooded figure punched finn.",
        target="finn", witness_npcs=["finn"],
        subject_identity="hooded_figure",
    )
    from npc_engine.knowledge import Quest
    old_bones = engine.pie.npc_knowledge.profiles["old_bones"]
    old_bones.add_quest(Quest(
        id="ref_q", name="Refuse Me", description="d",
        intent="dark", moral_weight=0.7,
        refusal_mode="decay", refusal_decay_ticks=10,
    ))
    sd.process_refusal(quest_id="ref_q", npc_id="old_bones")

    # Kill an NPC.
    sd.queue_death_request("brom", cause="test reset")
    sd._lifecycle_tick()
    brom = engine.pie.npc_knowledge.profiles["brom"]
    log.check(brom.status == "deceased", "brom deceased pre-reset")

    # Simulate a birth (add to profiles + birth_history).
    engine.pie.npc_knowledge.profiles["born_npc_pip"] = type(
        brom  # same class as existing stubs
    ).__new__(type(brom))
    engine.pie.npc_knowledge.profiles["born_npc_pip"].__dict__.update(
        engine.pie.npc_knowledge.profiles["finn"].__dict__.copy()
    )
    engine.pie.npc_knowledge.profiles["born_npc_pip"].identity = {
        "name": "Pip", "role": "urchin",
    }
    sd._birth_history.append({"npc_id": "born_npc_pip"})
    log.check("born_npc_pip" in engine.pie.npc_knowledge.profiles,
               "born NPC present pre-reset")

    sd.pause_ticks()
    sd.set_tick_budget(2.5)
    sd.set_quest_pacing(max_unoffered=0, cooldown_ticks=99)
    sd.set_player_auto_refuse(["cruel", "dark"])
    sd.set_player_activity("in_combat")

    pre_reset_size = len(engine.pie.npc_knowledge.profiles)
    log.check(pre_reset_size == 7,
               f"7 profiles pre-reset (6+1 born), got {pre_reset_size}")

    log.step("reset")
    result = sd.reset_to_initial_state()
    log.check(result["ok"] is True, "reset ok", result)
    log.check("born_npc_pip" in result["born_removed"],
               "born removed in report")
    log.check("brom" in result["deceased_restored"],
               "brom restored in report")

    log.step("verify_clean_slate")
    # Cast should be exactly the initial 6.
    post_size = len(engine.pie.npc_knowledge.profiles)
    log.check(post_size == 6, f"6 profiles post-reset, got {post_size}")
    log.check("born_npc_pip" not in engine.pie.npc_knowledge.profiles,
               "born NPC removed")

    # Brom alive.
    brom2 = engine.pie.npc_knowledge.profiles.get("brom")
    log.check(brom2 is not None and brom2.status == "alive",
               "brom alive post-reset")

    # State zeroed.
    log.check(sd.tick_count == 0, f"tick_count=0, got {sd.tick_count}")
    log.check(len(sd.ledger.entries) == 0, "ledger empty")
    log.check(sd.arc_planner.arcs == [], "arcs empty")
    log.check(sd._paused is False, "not paused")
    log.check(sd._tick_budget_seconds == -1.0, "budget cleared")
    log.check(sd._player_activity == "unknown", "activity reset")
    log.check(sd._max_unoffered_quests_override is None, "pacing cleared")
    log.check(sd._player_auto_refuse_intents == set(), "auto-refuse cleared")
    log.check(sd._visible_feature_to_identity == {}, "feature registry cleared")
    log.check(sd._refused_quest_timers == {}, "refusal timers cleared")
    log.check(sd._npc_player_identity_trust == {}, "identity trust cleared")
    log.check(sd._deceased_npcs == {}, "deceased cleared")
    log.check(sd._birth_history == [], "birth history cleared")

    # Captain Reva's player_knowledge should be fresh (no more Jordan).
    reva = engine.pie.npc_knowledge.profiles["captain_reva"]
    log.check(reva.player_knowledge["met"] is False,
               "reva player_knowledge reset")
    log.check(reva.player_knowledge["known_as"] == [],
               "reva known_as empty")

    # A live tick should work after reset.
    log.step("post_reset_tick")
    rec = _tick(sd, "post_reset", actions_per_tick=1)
    log.check(not rec["paused"], "post-reset tick fires", rec)
    log.check(sd.tick_count == 1, "tick_count incremented to 1")

    report = log.to_dict()
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ── Scenario 9 — predictive FactLedger drift cycle ───────────────

def scenario_predictive_drift(out: Path) -> dict:
    """PHASE_PREDICTIVE_FACTLEDGER_PLAN v1 validation, level 3.

    30-tick scripted cycle that pushes activity transitions on a
    fixed 3-tick cadence (in_town <-> in_dungeon), so the predictive
    layer harvests one supervision pair per transition. Asserts:

      - the layer is enabled + observes every real tick;
      - the ActivityPrior FITS within the cycle (enough history);
      - once fitted, warm (cold=False) predictions appear;
      - the drift check is *conditionally correct* over the natural
        session: drift fires exactly when a confident prediction
        disagrees with the reported activity (v1 observes only, so
        the natural fire COUNT is reported, not asserted);
      - a forced-drift probe — a contrast prior fitted on the real
        session ledger's latent — makes drift fire within ONE tick
        (well inside the plan's five-tick bound);
      - the edge filters accumulated per-NPC counts and their
        next-tick priors feed a non-empty edge_priors map;
      - arc proposal with the edge boost enabled still proposes
        arcs grounded in ledger NPCs (distribution recorded in the
        report for the with/without comparison).
    """
    world = "port_blackwater_zoned"
    log = StepLog("predictive_drift")
    engine, _, _ = boot_engine(world=world)
    sd = engine.story_director
    sd.set_narration_mode("terse")
    sd.set_active_zones(["dock_district", "lighthouse_bluffs"])

    layer = sd._predictive
    log.check(layer is not None, "predictive layer enabled by default")
    if layer is None:
        report = log.to_dict()
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    log.step("30_tick_cycle_with_3_tick_activity_cadence")
    activities = ["in_town", "in_dungeon"]
    fitted_at_tick = None
    warm_ticks = 0
    natural_drift_ticks = 0
    conditional_violations = 0
    tick_trail: list[dict] = []
    for i in range(30):
        if i % 3 == 0:
            sd.set_player_activity(activities[(i // 3) % 2])
        rec = _tick(sd, f"t{i+1}", actions_per_tick=1)
        pred = sd._last_activity_pred or {}
        if fitted_at_tick is None and layer.prior.is_fitted:
            fitted_at_tick = sd.tick_count
        if pred and not pred.get("cold", True):
            warm_ticks += 1
            if pred.get("drift"):
                natural_drift_ticks += 1
            # Conditional correctness: drift <=> confident disagreement.
            confident = pred.get("confidence", 0) >= 0.55
            disagrees = (pred.get("top_activity")
                         != pred.get("reported_activity"))
            if bool(pred.get("drift")) != bool(confident and disagrees):
                conditional_violations += 1
        tick_trail.append({
            "tick": sd.tick_count,
            "activity": sd._player_activity,
            "pred": pred,
            "paused": bool(rec.get("paused")),
        })

    log.check(layer._observation_count >= 25,
               "layer observed (nearly) every tick",
               layer._observation_count)
    log.check(fitted_at_tick is not None,
               "ActivityPrior fitted within the 30-tick cycle",
               {"fitted_at_tick": fitted_at_tick,
                "pairs": len(layer._activity_pairs)})
    log.check(warm_ticks > 0, "warm (cold=False) predictions after fit",
               warm_ticks)
    log.check(conditional_violations == 0,
               "drift flag == (confident AND disagrees) on every warm tick",
               conditional_violations)
    log.steps.append({"name": "natural_drift_ticks",
                       "payload": natural_drift_ticks})

    # Forced-drift probe: fit a contrast prior on the REAL session
    # ledger latent so the prior confidently says in_dungeon while
    # the client reports in_town. Drift must fire on the next tick —
    # the plan allows five ticks; one suffices when the prior is
    # already confident.
    log.step("forced_drift_probe")
    latent = layer.summary_latent(sd.ledger.entries)
    log.check(latent is not None, "session ledger produces a latent")
    if latent is not None:
        pairs = [(latent, "in_dungeon") for _ in range(8)]
        pairs += [(-latent, "in_town") for _ in range(8)]
        # Seed the layer's OWN pair store and fit from it. Anything
        # else gets clobbered: every /story/activity post refits the
        # prior from _activity_pairs (the prior is data-owned), so a
        # bare prior.fit() would not survive the next activity post.
        # The activity is set directly for the same reason — the probe
        # tests the drift check in tick(), not the harvest path.
        layer._activity_pairs = list(pairs)
        log.check(layer.prior.fit(layer._activity_pairs),
                   "contrast prior fits")
        sd._player_activity = "in_town"
        drift_before = layer._drift_count
        _tick(sd, "drift_probe", actions_per_tick=1)
        pred = sd._last_activity_pred or {}
        log.check(pred.get("drift") is True,
                   "drift fires within 1 tick of confident disagreement",
                   pred)
        log.check(layer._drift_count > drift_before,
                   "drift counter incremented", layer._drift_count)

    # Edge filters + boost path.
    log.step("edge_priors_and_boost")
    _, edge_priors = layer.predict_next(sd.ledger, sd.tick_count,
                                        sd._player_activity)
    log.check(len(edge_priors) >= 2,
               "edge priors cover >= 2 NPCs", edge_priors)
    stats = sd.get_predictive_state()
    log.check(stats["tracked_npcs"] >= 2, "tracked_npcs >= 2",
               stats["tracked_npcs"])
    log.check(stats["last_prediction"] is not None,
               "/story/state predictive block carries last prediction")

    # Arc-proposal comparison (recorded, not asserted — the plan's
    # with/without distribution check): propose from the same ledger
    # with no boost, a cold boost, and the live edge priors.
    from npc_engine.story_director import ArcPlanner
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        avail = list(engine.pie.npc_knowledge.profiles.keys())
        results = {}
        for tag, boost in (
            ("no_boost", None),
            ("cold_boost", {n: 0.5 for n in avail}),
            ("live_boost", edge_priors),
        ):
            planner = ArcPlanner(Path(td) / f"arcs_{tag}.json")
            arc = planner._propose_from_ledger(
                sd.ledger, avail, current_tick=sd.tick_count,
                edge_prior_boost=boost,
            )
            results[tag] = arc.focus_npcs if arc else None
        log.check(results["no_boost"] == results["cold_boost"],
                   "cold boost reproduces unboosted proposal", results)
        log.steps.append({"name": "arc_proposal_comparison",
                           "payload": results})

    log.steps.append({"name": "tick_trail", "payload": tick_trail})
    log.steps.append({"name": "predictive_stats", "payload": stats})
    report = log.to_dict()
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


SCENARIOS = {
    "gameplay":              scenario_gameplay,
    "persistence":           scenario_persistence,
    "refusal_decay":         scenario_refusal_decay,
    "live_dialogue":         scenario_live_dialogue,
    "identity_accuracy":     scenario_identity_accuracy,
    "activity_transitions":  scenario_activity_transitions,
    "game_reset":            scenario_game_reset,
    "scale_100":             scenario_scale_100,
    "predictive_drift":      scenario_predictive_drift,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=list(SCENARIOS.keys()) + ["all"],
                    required=True)
    args = p.parse_args()
    logs = NPC_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    total_failed = 0
    scenarios = (
        list(SCENARIOS.keys()) if args.scenario == "all"
        else [args.scenario]
    )
    summary: list[dict] = []
    for sc in scenarios:
        if sc == "scale_100" and args.scenario == "all":
            # Skip from "all" runs — heavyweight.
            continue
        out = logs / f"e2e_{sc}.json"
        print(f"\n\n###### SCENARIO {sc} ######")
        try:
            report = SCENARIOS[sc](out)
            total_failed += int(report.get("failed", 0))
            summary.append({
                "scenario": sc,
                "failed": report.get("failed", 0),
                "checks": report.get("check_count", 0),
            })
        except Exception as e:
            print(f"[FATAL] scenario {sc}: {e}")
            traceback.print_exc()
            total_failed += 1
            summary.append({"scenario": sc, "error": str(e)})
    print("\n\n=== E2E SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"TOTAL FAILED CHECKS: {total_failed}")
    sys.exit(1 if total_failed else 0)


if __name__ == "__main__":
    main()
