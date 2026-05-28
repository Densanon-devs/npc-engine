"""
Smoke test for the latent-align passive observer wired into the
gossip propagator. Simulates a drift cascade: inject multiple
semantically-similar facts about the same event into the same NPC
across turns. Verify the observer correctly identifies duplicates.

Run:
    python smoke_latent_align_observer.py

Verifies:
  - observer activates with the env var set
  - per-injection log entries are written to the sidecar JSONL
  - drift-cascade pairs are flagged as duplicates
  - unrelated facts are NOT flagged as duplicates
  - same-topic-different-event pairs are NOT flagged as duplicates
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["NPC_ENGINE_LATENT_ALIGN_OBSERVE"] = "1"

NPC_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(NPC_ROOT))
sys.path.insert(0, str(NPC_ROOT.parent / "plug-in-intelligence-engine"))
sys.path.insert(0, str(NPC_ROOT.parent / "densanon-core"))

from npc_engine.config import GossipRules
from npc_engine.social.network import SocialGraph
from npc_engine.social.propagation import GossipFact, GossipPropagator


class _StubKnowledge:
    def __init__(self):
        self.events_by_npc: dict[str, list[str]] = {}

    def inject_event(self, npc_id: str, text: str) -> None:
        self.events_by_npc.setdefault(npc_id, []).append(text)


def main():
    log_path = NPC_ROOT / "data" / "story_director" / "latent_align_observations.jsonl"
    if log_path.exists():
        log_path.unlink()
        print(f"Cleared previous log: {log_path}")

    # Minimal social graph: every NPC connects to "old_bones" (the
    # innkeeper who hears everything). All other NPCs gossip to him.
    # We bypass graph propagation entirely and call _inject_gossip
    # directly, so the graph is only needed to satisfy GossipPropagator.
    npcs = ["finn", "reva", "varro", "elara", "kael", "bess"]
    graph = SocialGraph()
    for npc_id in npcs:
        graph.add_connection({
            "from": npc_id, "to": "old_bones",
            "relationship": "friend", "closeness": 0.9, "gossip_filter": "all",
        })

    rules = GossipRules(max_hops=1, decay_per_hop=0.0,
                       min_significance=0.1, propagation_delay=0)
    prop = GossipPropagator(graph, rules)
    knowledge = _StubKnowledge()
    cap_managers: dict = {}

    # Drift-cascade scenario: 6 facts about the SAME event from
    # different NPCs, in different phrasings. Each one propagates to
    # old_bones. After fact #2, the observer should start flagging
    # duplicates against the existing history.
    drift_chain = [
        ("finn", "The king is unwell.", "personal", 0.7),
        ("reva", "The king is feeling sick.", "personal", 0.7),
        ("varro", "The king is ill.", "personal", 0.7),
        ("elara", "The king has fallen sick.", "personal", 0.7),
        ("kael", "The king's health is failing.", "personal", 0.7),
        ("bess", "The king is gravely sick.", "personal", 0.7),
    ]

    # Distinct events for the same NPC (different topic, NOT duplicates)
    distinct = [
        ("finn", "A storm hit the eastern berth last night.", "lore", 0.6),
        ("reva", "Mara opened her shop late this morning.", "trade", 0.5),
    ]

    # Same-topic different-event (NOT duplicates per our corpus)
    same_topic = [
        ("varro", "The queen has been dancing at the harvest festival.", "personal", 0.6),
        ("elara", "The king's son announced a tournament.", "personal", 0.6),
    ]

    print("Simulating gossip propagation...\n")
    all_facts = drift_chain + distinct + same_topic
    for source, text, category, significance in all_facts:
        prop._turn += 1
        fact = GossipFact(
            text=text, category=category, significance=significance,
            source_npc=source, source_turn=prop._turn,
        )
        # Directly inject (skip the extraction layer)
        prop._inject_gossip("old_bones", fact, knowledge, cap_managers)
        print(f"  T{prop._turn} {source} -> old_bones: {text}")
    print()

    # Read the observer log
    if not log_path.exists():
        print("FAIL — no observer log written")
        sys.exit(1)
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").strip().splitlines()]
    print(f"Observer entries: {len(records)}")
    print()

    print(f"{'T':>3} {'src':>10} {'hist':>4} {'sim':>5} {'dup':>5}  new_fact")
    print("-" * 100)
    for r in records:
        print(f"{r['turn']:>3} {r['source_npc']:>10} {r['history_size_before']:>4} "
              f"{r['best_similarity']:>5.2f} {str(r['is_duplicate_at_default']):>5}  "
              f"{r['new_fact'][:75]}")

    # Verify the expected pattern:
    # - Drift chain entries 2-6 should be flagged as duplicates (sim >= 0.70)
    # - Distinct events should NOT be flagged (low sim)
    # - Same-topic different-event should NOT be flagged
    print()
    print("Validation:")
    drift_records = [r for r in records if r['turn'] in (2, 3, 4, 5, 6)]
    drift_dup_count = sum(1 for r in drift_records if r['is_duplicate_at_default'])
    print(f"  Drift cascade (turns 2-6): {drift_dup_count}/5 flagged as duplicates")
    print(f"    Expected: 4-5 (some paraphrases may dip below threshold)")

    distinct_records = [r for r in records if r['turn'] in (7, 8)]
    distinct_dup_count = sum(1 for r in distinct_records if r['is_duplicate_at_default'])
    print(f"  Distinct events (turns 7-8): {distinct_dup_count}/2 flagged as duplicates")
    print(f"    Expected: 0 (storm and Mara shop are unrelated to king health)")

    same_topic_records = [r for r in records if r['turn'] in (9, 10)]
    same_topic_dup_count = sum(1 for r in same_topic_records if r['is_duplicate_at_default'])
    print(f"  Same-topic different-event (turns 9-10): {same_topic_dup_count}/2 flagged")
    print(f"    Expected: 0 (queen dancing / king's son tournament are not duplicates of king sick)")

    print()
    if drift_dup_count >= 3 and distinct_dup_count == 0 and same_topic_dup_count == 0:
        print("PASS — drift detected, false-positive-free on distinct/same-topic")
        sys.exit(0)
    else:
        print("PARTIAL — needs threshold review or wider corpus")
        sys.exit(1)


if __name__ == "__main__":
    main()
