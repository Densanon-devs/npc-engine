#!/usr/bin/env python3
"""
Predictive FactLedger — build-time verification harness.

Run on a cadence while iterating on the predictive lane. Four gates:

  1. Predictive unit + integration suite (tests/test_predictive_factledger.py)
  2. Full Story Director suite (tests/test_story_director.py) — the
     layer is default-on, so this doubles as a regression gate.
  3. Behavior-neutrality A/B — two stub Directors driven by identical
     scripted responses, one with the layer enabled and one with
     NPC_ENGINE_PREDICTIVE_DISABLE=1, must produce byte-identical
     action/dispatch decisions across a multi-tick session. This is
     the plan's core v1 promise ("can't be worse than today") checked
     directly rather than inferred.
  4. Plan-conformance spot checks — cheap static asserts that the
     load-bearing plan requirements are still wired (boost cap
     constant, activity JSONL append, sidecar gitignore entries).

Usage:
    python verify_predictive.py            # all gates
    python verify_predictive.py --fast     # skip gate 2 (slow suite)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

NPC_ROOT = Path(__file__).parent.resolve()
FAILURES: list[str] = []


def gate(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def run_suite(script: str) -> bool:
    print(f"\n== Gate: {script} ==")
    proc = subprocess.run([sys.executable, script], cwd=NPC_ROOT,
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    tail = "\n".join((proc.stdout or "").splitlines()[-3:])
    print(f"  exit={proc.returncode}  tail: {tail!r}")
    return proc.returncode == 0


def gate_behavior_neutrality() -> None:
    """Same scripted responses, layer on vs disabled -> identical
    Director decisions."""
    print("\n== Gate: behavior-neutrality A/B ==")
    sys.path.insert(0, str(NPC_ROOT))
    sys.path.insert(0, str(NPC_ROOT / "tests"))
    sys.path.insert(0, str((NPC_ROOT.parent / "plug-in-intelligence-engine").resolve()))
    os.environ.setdefault("NPC_ENGINE_DEV_MODE", "1")
    from test_story_director import _make_stub_engine, _isolate_state_file
    from npc_engine.story_director import StoryDirector

    responses = [
        '{"action": "event", "target": "kael", "description": "Kael finds a cracked anvil."}',
        '{"action": "quest", "npc_id": "bess", "quest": {"id": "q_ab", "name": "Cellar Rats", "description": "Clear the inn cellar."}}',
        '{"action": "fact", "target": "noah", "fact": "The old mill burned down years ago."}',
        '{"action": "event", "target": "bess", "description": "A trader arrives with salted fish."}',
    ]

    def run_session(disable: bool) -> list[dict]:
        tag = "ab_off" if disable else "ab_on"
        restore = _isolate_state_file(tag)
        if disable:
            os.environ["NPC_ENGINE_PREDICTIVE_DISABLE"] = "1"
        try:
            engine = _make_stub_engine(responses=list(responses))
            director = StoryDirector(engine)
            trail = []
            for _ in range(4):
                r = director.tick()
                trail.append({"action": r["action"], "dispatch": r["dispatch"]})
            return trail
        finally:
            os.environ.pop("NPC_ENGINE_PREDICTIVE_DISABLE", None)
            restore()
            for p in (NPC_ROOT / "data" / "story_director").glob(
                    f"_tmp_{tag}_state.*"):
                try:
                    p.unlink()
                except Exception:
                    pass

    trail_on = run_session(disable=False)
    trail_off = run_session(disable=True)
    gate("layer on/off produce identical decisions",
         json.dumps(trail_on, sort_keys=True) == json.dumps(trail_off, sort_keys=True),
         f"on={trail_on}\noff={trail_off}")


def gate_plan_conformance() -> None:
    print("\n== Gate: plan-conformance spot checks ==")
    pf = (NPC_ROOT / "npc_engine" / "predictive_factledger.py").read_text(encoding="utf-8")
    sd = (NPC_ROOT / "npc_engine" / "story_director.py").read_text(encoding="utf-8")
    gi = (NPC_ROOT / ".gitignore").read_text(encoding="utf-8")
    srv = (NPC_ROOT / "npc_engine" / "server.py").read_text(encoding="utf-8")

    gate("boost cap constant is 0.10 (plan: cap < 0.10 of total)",
         "EDGE_PRIOR_BOOST_CAP = 0.10" in pf)
    gate("drift check logs at DEBUG in v1 (plan risk register)",
         "logger.debug(" in pf and "predicted_drift" in pf)
    gate("Beta(1,1) cold prediction is exactly 0.5",
         "(s + 1) / (s + f + 2)" in pf)
    gate("activity JSONL appended in set_player_activity",
         "_activity_history_file" in sd and '"activity": activity' in sd)
    gate("edge_prior_boost plumbed through arc proposal",
         "edge_prior_boost=edge_prior_boost" in sd)
    gate("boost env-gated off by default (plan step 5)",
         "_PREDICTIVE_BOOST_ENV" in sd and "== \"1\"" in sd)
    gate("predictive sidecars gitignored (both runtime dirs)",
         "data/story_director/*.predictive.npz" in gi
         and "data/worlds/*/story/*.predictive.npz" in gi
         and "data/story_director/*.activity_history.jsonl" in gi
         and "data/worlds/*/story/*.activity_history.jsonl" in gi)
    gate("GET /story/predictive endpoint exists",
         "/story/predictive" in srv)
    gate("game reset keeps prior matrix (reset() has no prior clear)",
         "def reset(self)" in pf and "prior generalizes across resets" in pf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="skip the full story-director suite")
    args = parser.parse_args()

    ok1 = run_suite("tests/test_predictive_factledger.py")
    gate("predictive suite green", ok1)
    if not args.fast:
        ok2 = run_suite("tests/test_story_director.py")
        gate("story director suite green", ok2)
    gate_behavior_neutrality()
    gate_plan_conformance()

    print("\n=== VERIFY SUMMARY ===")
    if FAILURES:
        print(f"FAILED GATES ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("All verification gates passed.")


if __name__ == "__main__":
    main()
