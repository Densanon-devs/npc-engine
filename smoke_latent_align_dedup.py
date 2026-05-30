"""
Smoke test for active latent-align gossip dedup.

Same drift cascade as smoke_latent_align_observer.py, but verifies that
when NPC_ENGINE_LATENT_ALIGN_DEDUP=1, the propagator actually SUPPRESSES
duplicate gossip from reaching the target NPC's event list — not just
logs the duplication.

Run:
    python smoke_latent_align_dedup.py

Asserts:
  - With dedup OFF: target NPC receives ALL 10 gossip facts (baseline)
  - With dedup ON: drift-chain entries 2-6 suppressed; distinct events
    and same-topic-different-event entries still propagate; suppression
    counter matches expected
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

NPC_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(NPC_ROOT))
sys.path.insert(0, str(NPC_ROOT.parent / "plug-in-intelligence-engine"))
sys.path.insert(0, str(NPC_ROOT.parent / "densanon-core"))


class _StubKnowledge:
    """Captures every inject_event call so we can count suppressions."""

    def __init__(self):
        self.events_by_npc: dict[str, list[str]] = {}

    def inject_event(self, npc_id: str, text: str) -> None:
        self.events_by_npc.setdefault(npc_id, []).append(text)


# Same drift cascade + controls used by the observer smoke test.
# 6 paraphrases of "the king is sick" + 2 distinct events + 2
# same-topic-different-events = 10 gossip facts total.
DRIFT_CHAIN = [
    ("finn", "The king is unwell.", "personal", 0.7),
    ("reva", "The king is feeling sick.", "personal", 0.7),
    ("varro", "The king is ill.", "personal", 0.7),
    ("elara", "The king has fallen sick.", "personal", 0.7),
    ("kael", "The king's health is failing.", "personal", 0.7),
    ("bess", "The king is gravely sick.", "personal", 0.7),
]
DISTINCT = [
    ("finn", "A storm hit the eastern berth last night.", "lore", 0.6),
    ("reva", "Mara opened her shop late this morning.", "trade", 0.5),
]
SAME_TOPIC_DIFFERENT_EVENT = [
    ("varro", "The queen has been dancing at the harvest festival.", "personal", 0.6),
    ("elara", "The king's son announced a tournament.", "personal", 0.6),
]
ALL_FACTS = DRIFT_CHAIN + DISTINCT + SAME_TOPIC_DIFFERENT_EVENT


def _run(env: dict[str, str]) -> dict:
    """Run the scenario under specific env vars and report counts."""
    # Set env BEFORE importing the propagator so the GossipPropagator
    # __init__ picks up the flags. Re-import inside the function so each
    # call gets a fresh instance with current env.
    for k, v in env.items():
        os.environ[k] = v
    for k in ("NPC_ENGINE_LATENT_ALIGN_OBSERVE", "NPC_ENGINE_LATENT_ALIGN_DEDUP"):
        if k not in env and k in os.environ:
            del os.environ[k]

    # Fresh import each run so module-level state doesn't bleed across.
    import importlib
    import npc_engine.config as _cfg
    import npc_engine.social.network as _net
    import npc_engine.social.propagation as _prop
    importlib.reload(_prop)
    GossipRules = _cfg.GossipRules
    SocialGraph = _net.SocialGraph
    GossipPropagator = _prop.GossipPropagator
    GossipFact = _prop.GossipFact

    graph = SocialGraph()
    npcs = ["finn", "reva", "varro", "elara", "kael", "bess"]
    for npc_id in npcs:
        graph.add_connection({
            "from": npc_id, "to": "old_bones",
            "relationship": "friend", "closeness": 0.9,
            "gossip_filter": "all",
        })
    rules = GossipRules(max_hops=1, decay_per_hop=0.0,
                       min_significance=0.1, propagation_delay=0)
    prop = GossipPropagator(graph, rules)
    knowledge = _StubKnowledge()

    for source, text, category, significance in ALL_FACTS:
        prop._turn += 1
        fact = GossipFact(
            text=text, category=category, significance=significance,
            source_npc=source, source_turn=prop._turn,
        )
        prop._inject_gossip("old_bones", fact, knowledge, {})

    received = knowledge.events_by_npc.get("old_bones", [])
    return {
        "received_count": len(received),
        "received_texts": received,
        "suppressed_count": getattr(prop, "_latent_dedup_suppressed_count", 0),
    }


def main():
    print("=" * 70)
    print("Run 1: DEDUP OFF (baseline)")
    print("=" * 70)
    baseline = _run({})
    print(f"  Events received: {baseline['received_count']}/{len(ALL_FACTS)}")
    print(f"  Suppressed:      {baseline['suppressed_count']}")
    if baseline["received_count"] != len(ALL_FACTS):
        print(f"  FAIL — baseline should pass through all {len(ALL_FACTS)} events")
        sys.exit(1)
    print("  PASS — baseline behavior unchanged when dedup OFF")
    print()

    print("=" * 70)
    print("Run 2: DEDUP ON")
    print("=" * 70)
    dedup = _run({"NPC_ENGINE_LATENT_ALIGN_DEDUP": "1"})
    print(f"  Events received: {dedup['received_count']}/{len(ALL_FACTS)}")
    print(f"  Suppressed:      {dedup['suppressed_count']}")
    print()
    print("  Received texts (post-dedup):")
    for i, text in enumerate(dedup["received_texts"], 1):
        print(f"    {i}. {text[:75]}")

    # Expected: first king-sick paraphrase passes, the 5 subsequent
    # duplicates are suppressed. Distinct events (2) and same-topic
    # different-events (2) pass through. So:
    #   received = 1 (first king-sick) + 2 (distinct) + 2 (same-topic) = 5
    #   suppressed = 5 (king-sick paraphrases 2-6)
    expected_received = 5
    expected_suppressed = 5
    print()
    print(f"  Expected received: {expected_received}  Got: {dedup['received_count']}")
    print(f"  Expected suppressed: {expected_suppressed}  Got: {dedup['suppressed_count']}")

    ok = (dedup["received_count"] == expected_received
          and dedup["suppressed_count"] == expected_suppressed)
    if ok:
        print()
        print("PASS — drift chain collapsed to 1 entry, distinct events preserved")
        sys.exit(0)
    else:
        print()
        print("FAIL — dedup behavior does not match expected")
        sys.exit(1)


if __name__ == "__main__":
    main()
