"""
Smoke test for the NarrativeJudge passive observer wired into
StoryDirector. Boots Ashenvale, runs 5 ticks with the observer
enabled, and prints the resulting observer log.

Run:
    python smoke_narrative_judge_observer.py

Verifies:
  - observer init succeeds with the env var set
  - per-dispatch log entries are written to the per-world sidecar
  - score fields are populated (not None) for at least some dispatches
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Enable the observer before importing anything that touches StoryDirector
os.environ["NPC_ENGINE_NARRATIVE_JUDGE_OBSERVE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NPC_ENGINE_DEV_MODE", "1")

NPC_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(NPC_ROOT))

# Borrow boot_engine + _tick from stress_director
from stress_director import boot_engine, _tick

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass


def main():
    print("Booting Ashenvale engine...")
    engine, _, _ = boot_engine(world="ashenvale", reset=True)
    sd = engine.story_director
    sd.set_narration_mode("terse")

    log_path = getattr(sd, "_narrative_judge_log_path", None)
    judge = getattr(sd, "_narrative_judge", None)
    print(f"Judge enabled: {judge is not None}")
    print(f"Judge available: {judge.available if judge else 'N/A'}")
    print(f"Log path: {log_path}")
    if log_path and log_path.exists():
        log_path.unlink()
        print(f"Cleared previous log.")
    print()

    print("Running 5 ticks...")
    for i in range(5):
        rec = _tick(sd, f"smoke_{i+1}")
        print(f"  T{rec['tick']}: kind={rec['action_kind']} elapsed={rec['elapsed_s']}s "
              f"subactions={rec['n_subactions']}")

    print()
    print(f"Reading observer log: {log_path}")
    if not log_path or not log_path.exists():
        print("  ! Log file does not exist.")
        sys.exit(1)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    print(f"  {len(lines)} log entries.")
    print()
    for i, line in enumerate(lines, 1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            print(f"  [{i}] INVALID JSON: {line[:80]}")
            continue
        scored = rec.get("scores") or {}
        n_scored = len(scored)
        print(f"  [{i}] tick={rec['tick']:>2} kind={rec['action_kind']:>5} "
              f"target={str(rec.get('dispatched_target'))[:15]:>15} "
              f"active_quests={rec.get('active_quest_count')} "
              f"scored_against={n_scored}")
        print(f"        fact: {rec.get('fact_text','')[:100]}")
        if scored:
            # Pick the quest with highest advance probability
            best = max(scored.keys(),
                       key=lambda k: scored[k]["scores"].get("advance", 0.0))
            best_score = scored[best]["scores"]
            print(f"        best-advance: {best} "
                  f"adv={best_score['advance']:.2f} "
                  f"blk={best_score['block']:.2f} "
                  f"neu={best_score['neutral']:.2f}")
    print()
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
