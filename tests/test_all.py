"""
NPC Engine — Full test suite.

Covers:
  1. Bridge: PIE imports work from sibling directory
  2. Capabilities: All 6 capabilities (5 extracted + gossip) work in isolation
  3. Few-Shot Loader: YAML loading, 3-layer merging, category dedup
  4. Social Graph: Adjacency, reachability, filtering
  5. Gossip Propagation: Fact extraction, hop walking, decay, filtering
  6. Reputation Ripple: Trust changes propagate to connected NPCs
  7. NPCEngine Config: YAML loading, path derivation
  8. Profile Validation: All 7 Ashenvale NPCs load with capabilities
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure npc-engine is on path
NPC_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(NPC_ROOT))


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  PASS  {name}")

    def fail(self, name, msg):
        self.failed += 1
        self.errors.append((name, msg))
        print(f"  FAIL  {name}: {msg}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("\nFailures:")
            for name, msg in self.errors:
                print(f"  - {name}: {msg}")
        return self.failed == 0


results = TestResults()


def assert_true(name, cond, msg=""):
    if cond:
        results.ok(name)
    else:
        results.fail(name, msg or "condition was False")


def assert_eq(name, actual, expected):
    if actual == expected:
        results.ok(name)
    else:
        results.fail(name, f"expected {expected!r}, got {actual!r}")


def assert_in(name, sub, text):
    if sub in str(text):
        results.ok(name)
    else:
        results.fail(name, f"'{sub}' not found in '{str(text)[:100]}...'")


# ══════════════════════════════════════════════════════════════
# 1. Bridge Tests
# ══════════════════════════════════════════════════════════════

print("\n--- Bridge Tests ---")

try:
    from npc_engine.bridge import (
        BaseModel, Expert, ExpertRouter, SolvedExample,
        PIE_ROOT, NPC_ENGINE_ROOT, PluginIntelligenceEngine,
    )
    assert_true("bridge imports PIE components", True)
    assert_true("PIE_ROOT exists", PIE_ROOT.exists(), f"PIE_ROOT: {PIE_ROOT}")
    assert_true("NPC_ENGINE_ROOT exists", NPC_ENGINE_ROOT.exists())
except Exception as e:
    results.fail("bridge imports", str(e))
    print(f"  FATAL: Cannot continue without bridge. Error: {e}")
    results.summary()
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
# 2. Capability Tests
# ══════════════════════════════════════════════════════════════

print("\n--- Capability Tests ---")

from npc_engine.capabilities.base import (
    Capability, CapabilityContext, CapabilityManager,
    CapabilityRegistry, CapabilityUpdate,
)
import npc_engine.capabilities  # auto-register

registered = CapabilityRegistry.list_all()
assert_true("scratchpad registered", "scratchpad" in registered)
assert_true("trust registered", "trust" in registered)
assert_true("emotional_state registered", "emotional_state" in registered)
assert_true("goals registered", "goals" in registered)
assert_true("knowledge_gate registered", "knowledge_gate" in registered)
assert_true("gossip registered", "gossip" in registered)
assert_eq("6 capabilities registered", len(registered), 6)

# Quick capability lifecycle test
with tempfile.TemporaryDirectory() as tmpdir:
    mgr = CapabilityManager(
        npc_id="test",
        capability_configs={
            "scratchpad": {"enabled": True, "max_entries": 5},
            "trust": {"enabled": True, "initial_level": 40},
            "gossip": {"enabled": True, "max_rumors": 3},
        },
        state_dir=tmpdir,
    )
    assert_eq("manager loaded 3 capabilities", len(mgr.capabilities), 3)

    ctx = mgr.build_all_contexts("hello", token_budget=200)
    assert_true("manager builds context", isinstance(ctx, str))

    mgr.process_all_responses(
        '{"dialogue":"Hi.","emotion":"neutral","action":null}',
        "My name is Aldric",
    )
    assert_true("trust in shared_state", "trust" in mgr.shared_state)

    mgr.save_state()
    assert_true("state file created", (Path(tmpdir) / "test.json").exists())


# ══════════════════════════════════════════════════════════════
# 3. Few-Shot Loader Tests
# ══════════════════════════════════════════════════════════════

print("\n--- Few-Shot Loader Tests ---")

from npc_engine.experts.examples import FewShotLoader, _STRUCTURAL_EXAMPLES

# Structural examples
assert_true("structural examples exist", len(_STRUCTURAL_EXAMPLES) >= 5,
            f"Count: {len(_STRUCTURAL_EXAMPLES)}")

# World-level examples
loader = FewShotLoader("data/worlds/ashenvale/examples")
assert_true("world examples loaded", len(loader._world_examples) > 0,
            f"Count: {len(loader._world_examples)}")

# Merge: world + structural
world_examples = loader.get_world_examples()
assert_true("merged has structural + world",
            len(world_examples) > len(_STRUCTURAL_EXAMPLES),
            f"Merged: {len(world_examples)}, Structural: {len(_STRUCTURAL_EXAMPLES)}")

# Per-NPC examples
npc_data = {
    "examples": [
        {"query": "Custom NPC question?", "solution": '{"dialogue":"Custom answer.","emotion":"calm","action":null}', "category": "custom"},
    ],
}
npc_examples = loader.get_examples_for_npc("test_npc", npc_data)
assert_true("NPC examples merged",
            any(ex.category == "custom" for ex in npc_examples),
            f"Categories: {[ex.category for ex in npc_examples]}")

# Dedup: NPC category overrides world
npc_data_override = {
    "examples": [
        {"query": "Who are you?", "solution": '{"dialogue":"I am custom.","emotion":"stern","action":null}', "category": "greeting"},
    ],
}
override_examples = loader.get_examples_for_npc("test", npc_data_override)
greeting_examples = [ex for ex in override_examples if ex.category == "greeting"]
assert_eq("greeting deduped to 1", len(greeting_examples), 1)
assert_in("NPC greeting takes priority", "custom", greeting_examples[0].solution)

# Conversion to SolvedExample
solved = loader.to_solved_examples(world_examples[:3])
assert_eq("conversion count", len(solved), 3)
assert_true("converted type is SolvedExample",
            type(solved[0]).__name__ == "SolvedExample")


# ══════════════════════════════════════════════════════════════
# 4. Social Graph Tests
# ══════════════════════════════════════════════════════════════

print("\n--- Social Graph Tests ---")

from npc_engine.social.network import SocialGraph

graph = SocialGraph("data/worlds/ashenvale/world.yaml")
assert_true("graph loaded connections", len(graph.connections) > 0,
            f"Connections: {len(graph.connections)}")

# Adjacency
noah_conns = graph.get_connections("noah")
assert_true("noah has connections", len(noah_conns) >= 3,
            f"Noah connections: {len(noah_conns)}")

noah_targets = [c.to_id for c in noah_conns]
assert_in("noah connected to kael", "kael", noah_targets)
assert_in("noah connected to elara", "elara", noah_targets)

# Reachability
reachable = graph.get_reachable("noah", max_hops=2)
assert_true("noah reaches multiple NPCs", len(reachable) >= 4,
            f"Reachable: {reachable}")

# Closeness
closeness = graph.get_closeness("noah", "elara")
assert_true("noah-elara closeness high", closeness >= 0.8,
            f"Closeness: {closeness}")

closeness_none = graph.get_closeness("noah", "nonexistent")
assert_eq("nonexistent closeness is 0", closeness_none, 0.0)

# Gossip filter
filter_val = graph.get_gossip_filter("noah", "kael")
assert_eq("noah-kael gossip filter", filter_val, "all")

filter_val2 = graph.get_gossip_filter("guard_roderick", "noah")
assert_eq("roderick-noah gossip filter", filter_val2, "military")

# All NPCs
all_npcs = graph.get_all_npcs()
assert_true("graph has 7 NPCs", len(all_npcs) >= 7,
            f"NPCs: {all_npcs}")


# ══════════════════════════════════════════════════════════════
# 5. Gossip Propagation Tests
# ══════════════════════════════════════════════════════════════

print("\n--- Gossip Propagation Tests ---")

from npc_engine.social.propagation import GossipPropagator, classify_fact
from npc_engine.config import GossipRules

# Fact classification
assert_eq("classify personal", classify_fact("My name is Aldric"), "personal")
assert_eq("classify trade", classify_fact("I want to buy some gold"), "trade")
assert_eq("classify military", classify_fact("The guards are on patrol"), "military")
assert_eq("classify lore", classify_fact("The ancient forest is cursed"), "lore")
assert_eq("classify quest", classify_fact("I completed the quest"), "quest")

# Propagation with instant delivery
rules = GossipRules(max_hops=2, decay_per_hop=0.5, min_significance=0.1,
                    propagation_delay=0)
propagator = GossipPropagator(graph, rules)

# Mock knowledge manager and capability managers
class MockKnowledgeManager:
    def __init__(self):
        self.injected_events = []
    def inject_event(self, npc_id, description):
        self.injected_events.append((npc_id, description))
    def inject_global_event(self, description):
        pass

mock_km = MockKnowledgeManager()
mock_caps = {
    "noah": CapabilityManager("noah", {"trust": {"initial_level": 55}}, state_dir=tmpdir),
}

# Simulate: player tells Noah their name
delivered = propagator.propagate(
    source_npc="noah",
    player_input="My name is Aldric and I come from the Eastern Kingdoms",
    npc_response='{"dialogue":"Welcome.","emotion":"warm","action":null}',
    knowledge_manager=mock_km,
    capability_managers=mock_caps,
)

assert_true("gossip delivered to connected NPCs",
            len(mock_km.injected_events) > 0,
            f"Delivered: {delivered}, Events: {mock_km.injected_events}")

# Check that gossip reached NPCs connected to Noah
event_targets = [e[0] for e in mock_km.injected_events]
assert_true("gossip reached at least 1 NPC", len(event_targets) >= 1,
            f"Targets: {event_targets}")

# Propagation delay test
delayed_rules = GossipRules(propagation_delay=2)
delayed_prop = GossipPropagator(graph, delayed_rules)

mock_km2 = MockKnowledgeManager()
delayed_prop.propagate("noah", "I found treasure", '{"dialogue":"...","emotion":"neutral","action":null}',
                       mock_km2, mock_caps)
assert_true("delayed gossip not delivered yet",
            len(mock_km2.injected_events) == 0 or delayed_prop.pending_count > 0,
            f"Pending: {delayed_prop.pending_count}")


# ══════════════════════════════════════════════════════════════
# 6. Reputation Ripple Tests
# ══════════════════════════════════════════════════════════════

print("\n--- Reputation Ripple Tests ---")

from npc_engine.social.reputation import ReputationRipple
from npc_engine.config import TrustRippleConfig

with tempfile.TemporaryDirectory() as tmpdir:
    # Create capability managers for Noah and Kael
    noah_mgr = CapabilityManager("noah", {
        "trust": {"initial_level": 50},
    }, state_dir=tmpdir)
    kael_mgr = CapabilityManager("kael", {
        "trust": {"initial_level": 30},
    }, state_dir=tmpdir)

    cap_mgrs = {"noah": noah_mgr, "kael": kael_mgr}

    ripple = ReputationRipple(graph, TrustRippleConfig(
        enabled=True, positive_factor=0.5, negative_factor=0.2, max_ripple=15,
    ))

    # Simulate Noah's trust increasing significantly
    noah_trust = noah_mgr.capabilities["trust"]
    noah_trust.level = 70  # +20 from initial 50
    noah_mgr.shared_state["trust"]["level"] = 70

    adjusted = ripple.process(cap_mgrs)

    # Kael should have received a trust ripple from Noah
    kael_trust = kael_mgr.capabilities["trust"]
    if "kael" in adjusted:
        assert_true("kael trust rippled up", kael_trust.level > 30,
                    f"Kael trust: {kael_trust.level}")
    else:
        # Kael might not have been adjusted if the delta wasn't detected
        # (first call has no baseline). Trigger a second round.
        noah_trust.level = 85
        noah_mgr.shared_state["trust"]["level"] = 85
        adjusted2 = ripple.process(cap_mgrs)
        assert_true("kael trust rippled on second change",
                    kael_trust.level > 30 or "kael" in adjusted2,
                    f"Kael trust: {kael_trust.level}, adjusted: {adjusted2}")


# ══════════════════════════════════════════════════════════════
# 7. Config Tests
# ══════════════════════════════════════════════════════════════

print("\n--- Config Tests ---")

from npc_engine.config import NPCEngineConfig
config = NPCEngineConfig.load("config.yaml")
assert_eq("config world_name", config.world_name, "Ashenvale")
assert_in("config profiles_dir", "npc_profiles", config.profiles_dir)
assert_in("config state_dir", "npc_state", config.state_dir)
assert_in("config examples_dir", "examples", config.examples_dir)
assert_eq("config gossip max_hops", config.gossip.max_hops, 2)
assert_eq("config trust_ripple enabled", config.trust_ripple.enabled, True)


# ══════════════════════════════════════════════════════════════
# 8. Profile Validation
# ══════════════════════════════════════════════════════════════

print("\n--- Ashenvale Profile Validation ---")

from npc_engine.knowledge import NPCKnowledgeManager

npc_mgr = NPCKnowledgeManager("data/worlds/ashenvale/npc_profiles")
assert_eq("7 profiles loaded", len(npc_mgr.profiles), 7)

for npc_id, npc in npc_mgr.profiles.items():
    name = npc.identity.get("name", "")
    assert_true(f"NPC {npc_id}: has name", len(name) > 0)
    assert_true(f"NPC {npc_id}: has capabilities",
                len(npc.capability_configs) >= 2,
                f"Caps: {list(npc.capability_configs.keys())}")

    # Create capability manager
    with tempfile.TemporaryDirectory() as td:
        mgr = CapabilityManager(npc_id, npc.capability_configs, state_dir=td)
        assert_true(f"NPC {npc_id}: manager OK", len(mgr.capabilities) >= 2)


# ══════════════════════════════════════════════════════════════
# 9. Game State Mutation Tests
# ══════════════════════════════════════════════════════════════

print("\n--- Game State Mutation Tests ---")

# Use a CapabilityManager to test mutation helpers directly
# (testing engine methods without loading a full model)

with tempfile.TemporaryDirectory() as tmpdir:
    # Set up a mock NPC with all capabilities
    mgr = CapabilityManager(
        npc_id="noah",
        capability_configs={
            "scratchpad": {"enabled": True, "max_entries": 10},
            "trust": {"enabled": True, "initial_level": 40},
            "emotional_state": {"enabled": True, "baseline_mood": "calm"},
            "goals": {"enabled": True, "active_goals": [
                {"id": "protect", "description": "Protect village", "priority": 10},
            ]},
            "knowledge_gate": {"enabled": True, "gated_facts": [
                {"id": "secret1", "fact": "Hidden info", "requires": {"trust": 60}},
            ]},
            "gossip": {"enabled": True, "max_rumors": 3},
        },
        state_dir=tmpdir,
    )

    # Test: adjust trust directly
    trust_cap = mgr.capabilities["trust"]
    old_trust = trust_cap.level
    trust_cap.level = min(100, trust_cap.level + 15)
    mgr.shared_state.setdefault("trust", {})["level"] = trust_cap.level
    assert_true("direct trust adjust", trust_cap.level == old_trust + 15,
                f"Expected {old_trust + 15}, got {trust_cap.level}")

    # Test: add scratchpad entry directly
    from npc_engine.capabilities.scratchpad import ScratchpadEntry
    scratchpad = mgr.capabilities["scratchpad"]
    scratchpad._add_entry(ScratchpadEntry(text="Player killed the dragon", turn=1, importance=0.9))
    assert_true("direct scratchpad add", len(scratchpad.entries) == 1)
    assert_in("scratchpad entry text", "dragon", scratchpad.entries[0].text)

    # Test: set mood directly
    emo = mgr.capabilities["emotional_state"]
    emo.mood = "fearful"
    emo.intensity = 0.8
    assert_eq("direct mood set", emo.mood, "fearful")

    # Test: unlock knowledge gate directly
    gate = mgr.capabilities["knowledge_gate"]
    gate.unlocked_ids.add("secret1")
    assert_true("direct gate unlock", "secret1" in gate.unlocked_ids)

    # Test: gossip add rumor directly
    gossip = mgr.capabilities["gossip"]
    from npc_engine.social.propagation import GossipFact
    gossip.add_rumor(GossipFact(
        text="Kael heard the stranger is trustworthy",
        category="personal",
        significance=0.7,
        source_npc="kael",
    ))
    assert_true("direct gossip add", len(gossip.rumors) == 1)
    assert_in("gossip rumor text", "trustworthy", gossip.rumors[0].text)

    # Test: context reflects mutations
    ctx = mgr.build_all_contexts("tell me everything", token_budget=400)
    assert_in("mutated trust in context", "Trust:", ctx)
    assert_in("mutated scratchpad in context", "dragon", ctx)
    assert_in("mutated mood in context", "fearful", ctx)

    # Test: state persists after mutations
    mgr.save_state()
    mgr2 = CapabilityManager(
        npc_id="noah",
        capability_configs={
            "scratchpad": {"enabled": True, "max_entries": 10},
            "trust": {"enabled": True, "initial_level": 40},
            "emotional_state": {"enabled": True, "baseline_mood": "calm"},
            "gossip": {"enabled": True, "max_rumors": 3},
        },
        state_dir=tmpdir,
    )
    restored_trust = mgr2.capabilities["trust"]
    assert_eq("trust survives persistence", restored_trust.level, trust_cap.level)

    restored_scratchpad = mgr2.capabilities["scratchpad"]
    assert_true("scratchpad survives persistence", len(restored_scratchpad.entries) >= 1,
                f"Entries: {len(restored_scratchpad.entries)}")

    restored_emo = mgr2.capabilities["emotional_state"]
    assert_eq("mood survives persistence", restored_emo.mood, "fearful")

    restored_gossip = mgr2.capabilities["gossip"]
    assert_true("gossip survives persistence", len(restored_gossip.rumors) >= 1)


# ══════════════════════════════════════════════════════════════
# 10. Quest State Mutation Tests
# ══════════════════════════════════════════════════════════════

print("\n--- Quest Mutation Tests ---")

from npc_engine.knowledge import PlayerQuestTracker

with tempfile.TemporaryDirectory() as tmpdir:
    tracker = PlayerQuestTracker(os.path.join(tmpdir, "quests.yaml"))

    # Accept quest
    tracker.accept_quest("bitter_well", "The Bitter Well", "noah")
    assert_true("quest accepted", tracker.has_quest("bitter_well"))
    assert_eq("1 active quest", len(tracker.active_quests), 1)

    # Complete quest
    tracker.complete_quest("bitter_well")
    assert_true("quest completed", tracker.has_completed("bitter_well"))
    assert_true("quest no longer active", not tracker.has_quest("bitter_well"))
    assert_eq("1 completed quest", len(tracker.completed_quests), 1)

    # Persistence
    tracker.save()
    tracker2 = PlayerQuestTracker(os.path.join(tmpdir, "quests.yaml"))
    assert_true("quest persists", tracker2.has_completed("bitter_well"))


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════

success = results.summary()
sys.exit(0 if success else 1)
