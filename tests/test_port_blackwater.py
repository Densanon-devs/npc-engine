"""
Port Blackwater — Example world validation.

Tests the complete developer flow:
  1. World loads from config
  2. All 3 NPCs load with capabilities
  3. Social graph connects them correctly
  4. Custom few-shot examples load (world + per-NPC)
  5. Gossip propagates through the network
  6. Trust ripple works across connections
  7. Knowledge gates unlock at thresholds
  8. Game state mutations work (quest accept/complete, trust adjust, etc.)
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


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
    results.ok(name) if cond else results.fail(name, msg or "False")


def assert_eq(name, actual, expected):
    results.ok(name) if actual == expected else results.fail(name, f"expected {expected!r}, got {actual!r}")


def assert_in(name, sub, text):
    results.ok(name) if sub in str(text) else results.fail(name, f"'{sub}' not in '{str(text)[:100]}'")


# ── 1. Config loads ─────────────────────────────────────────

print("\n--- Config ---")

from npc_engine.config import NPCEngineConfig

config = NPCEngineConfig.load("examples/port_blackwater_config.yaml")
assert_eq("world_name", config.world_name, "Port Blackwater")
assert_in("profiles_dir", "port_blackwater", config.profiles_dir)
assert_eq("active_npc", config.active_npc, "captain_reva")


# ── 2. NPC profiles load ────────────────────────────────────

print("\n--- NPC Profiles ---")

from npc_engine.knowledge import NPCKnowledgeManager

npc_mgr = NPCKnowledgeManager(config.profiles_dir)
assert_eq("3 NPCs loaded", len(npc_mgr.profiles), 3)

for npc_id in ["captain_reva", "old_bones", "finn"]:
    npc = npc_mgr.get(npc_id)
    assert_true(f"{npc_id} exists", npc is not None)
    assert_true(f"{npc_id} has name", len(npc.identity.get("name", "")) > 0)
    assert_true(f"{npc_id} has capabilities", len(npc.capability_configs) >= 4,
                f"Caps: {list(npc.capability_configs.keys())}")
    assert_true(f"{npc_id} has quests", len(npc.quests) >= 1)


# ── 3. Capabilities initialize ──────────────────────────────

print("\n--- Capabilities ---")

from npc_engine.capabilities.base import CapabilityManager
import npc_engine.capabilities  # auto-register

with tempfile.TemporaryDirectory() as tmpdir:
    for npc_id in ["captain_reva", "old_bones", "finn"]:
        npc = npc_mgr.get(npc_id)
        mgr = CapabilityManager(npc_id, npc.capability_configs, state_dir=tmpdir)
        assert_true(f"{npc_id}: manager created", len(mgr.capabilities) >= 4)
        assert_true(f"{npc_id}: has gossip", "gossip" in mgr.capabilities)

        # Build context
        ctx = mgr.build_all_contexts("hello", token_budget=300)
        assert_true(f"{npc_id}: context builds", isinstance(ctx, str))


# ── 4. Social graph ─────────────────────────────────────────

print("\n--- Social Graph ---")

from npc_engine.social.network import SocialGraph

graph = SocialGraph(config.world_yaml)
assert_true("graph has connections", len(graph.connections) >= 5,
            f"Connections: {len(graph.connections)}")

# Reva -> Old Bones
reva_conns = graph.get_connections("captain_reva")
reva_targets = [c.to_id for c in reva_conns]
assert_in("reva -> old_bones", "old_bones", reva_targets)
assert_in("reva -> finn", "finn", reva_targets)

# Finn -> Old Bones (friend)
finn_conns = graph.get_connections("finn")
finn_targets = [c.to_id for c in finn_conns]
assert_in("finn -> old_bones", "old_bones", finn_targets)

# Gossip filter: Old Bones -> Reva is trade only
filter_val = graph.get_gossip_filter("old_bones", "captain_reva")
assert_eq("bones->reva filter is trade", filter_val, "trade")

# Reachability: from Finn, can reach Reva via Old Bones (2 hops)
reachable = graph.get_reachable("finn", max_hops=2)
assert_in("finn reaches reva via bones", "captain_reva", reachable)


# ── 5. Custom few-shot examples ─────────────────────────────

print("\n--- Custom Examples ---")

from npc_engine.experts.examples import FewShotLoader

loader = FewShotLoader(config.examples_dir)
assert_true("world examples loaded", len(loader._world_examples) > 0,
            f"Count: {len(loader._world_examples)}")

# Check pirate tone in world examples
world_ex = loader.get_world_examples()
solutions = " ".join(ex.solution for ex in world_ex)
assert_in("pirate tone: Port Blackwater", "Port Blackwater", solutions)

# Per-NPC examples for Finn
finn_npc = npc_mgr.get("finn")
import yaml
with open(finn_npc.profile_path) as f:
    finn_data = yaml.safe_load(f)

finn_examples = loader.get_examples_for_npc("finn", finn_data)
assert_true("finn has custom examples",
            any("Finn" in ex.solution for ex in finn_examples),
            f"Finn solutions: {[ex.solution[:40] for ex in finn_examples]}")

# Verify per-NPC greeting overrides world greeting
finn_greetings = [ex for ex in finn_examples if ex.category == "greeting"]
assert_eq("finn greeting deduped", len(finn_greetings), 1)
assert_in("finn greeting is custom", "Finn", finn_greetings[0].solution)


# ── 6. Gossip propagation ───────────────────────────────────

print("\n--- Gossip Propagation ---")

from npc_engine.social.propagation import GossipPropagator
from npc_engine.config import GossipRules

with tempfile.TemporaryDirectory() as tmpdir:
    # Set up capability managers for all NPCs
    cap_mgrs = {}
    for npc_id in ["captain_reva", "old_bones", "finn"]:
        npc = npc_mgr.get(npc_id)
        cap_mgrs[npc_id] = CapabilityManager(npc_id, npc.capability_configs, state_dir=tmpdir)

    # Mock knowledge manager
    class MockKM:
        def __init__(self):
            self.events = []
        def inject_event(self, npc_id, desc):
            self.events.append((npc_id, desc))
        def inject_global_event(self, desc):
            pass

    mock_km = MockKM()
    propagator = GossipPropagator(graph, GossipRules(propagation_delay=0))

    # Player tells Finn their name
    delivered = propagator.propagate(
        source_npc="finn",
        player_input="My name is Drake, I am a sailor looking for work",
        npc_response='{"dialogue":"...","emotion":"eager","action":null}',
        knowledge_manager=mock_km,
        capability_managers=cap_mgrs,
    )

    # Old Bones should receive gossip (Finn -> Old Bones, closeness 0.7, filter "all")
    event_targets = [e[0] for e in mock_km.events]
    assert_true("gossip reaches old_bones from finn",
                "old_bones" in event_targets,
                f"Targets: {event_targets}")

    # Captain Reva might receive via 2-hop (Finn -> Old Bones -> Reva, filter "trade")
    # Personal info might not pass trade filter, which is correct behavior


# ── 7. Trust ripple ─────────────────────────────────────────

print("\n--- Trust Ripple ---")

from npc_engine.social.reputation import ReputationRipple
from npc_engine.config import TrustRippleConfig

with tempfile.TemporaryDirectory() as tmpdir:
    cap_mgrs = {}
    for npc_id in ["captain_reva", "old_bones", "finn"]:
        npc = npc_mgr.get(npc_id)
        cap_mgrs[npc_id] = CapabilityManager(npc_id, npc.capability_configs, state_dir=tmpdir)

    ripple = ReputationRipple(graph, TrustRippleConfig(
        positive_factor=0.5, max_ripple=15,
    ))

    # Boost Old Bones trust significantly
    bones_trust = cap_mgrs["old_bones"].capabilities["trust"]
    bones_trust.level = 70  # big jump from initial 35
    cap_mgrs["old_bones"].shared_state["trust"]["level"] = 70

    # First call establishes baseline
    ripple.process(cap_mgrs)

    # Second jump
    bones_trust.level = 85
    cap_mgrs["old_bones"].shared_state["trust"]["level"] = 85
    adjusted = ripple.process(cap_mgrs)

    # Finn and Reva should get trust ripple from Old Bones
    finn_trust = cap_mgrs["finn"].capabilities["trust"].level
    assert_true("finn trust rippled from bones",
                finn_trust > 25 or "finn" in adjusted,
                f"Finn trust: {finn_trust}, adjusted: {adjusted}")


# ── 8. Knowledge gates ──────────────────────────────────────

print("\n--- Knowledge Gates ---")

with tempfile.TemporaryDirectory() as tmpdir:
    npc = npc_mgr.get("captain_reva")
    mgr = CapabilityManager("captain_reva", npc.capability_configs, state_dir=tmpdir)

    # At initial trust (15), lighthouse_clue requires trust 40 + quest active
    # So nothing should be unlocked yet
    gate = mgr.capabilities["knowledge_gate"]
    ctx = gate.build_context("tell me about the lighthouse", mgr.shared_state)
    assert_true("gates locked at low trust",
                "black glass" not in ctx.context_fragment,
                f"Context: {ctx.context_fragment}")

    # Raise trust to 45 and activate quest
    mgr.capabilities["trust"].level = 45
    mgr.shared_state["trust"]["level"] = 45
    mgr.shared_state["player_quests"] = [{"id": "lighthouse_mystery", "status": "active"}]

    ctx = gate.build_context("tell me about the lighthouse", mgr.shared_state)
    assert_in("lighthouse_clue unlocks at trust 45 + quest",
              "black glass", ctx.context_fragment)

    # Shoals secret still locked (needs trust 70)
    assert_true("shoals_secret still locked",
                "pulls ships" not in ctx.context_fragment)

    # Raise trust to 75
    mgr.capabilities["trust"].level = 75
    mgr.shared_state["trust"]["level"] = 75
    ctx = gate.build_context("what about the Shoals?", mgr.shared_state)
    assert_in("shoals_secret unlocks at trust 75", "pulls ships", ctx.context_fragment)


# ── 9. Game state mutations ─────────────────────────────────

print("\n--- Game Mutations ---")

with tempfile.TemporaryDirectory() as tmpdir:
    npc = npc_mgr.get("finn")
    mgr = CapabilityManager("finn", npc.capability_configs, state_dir=tmpdir)

    # Inject a scratchpad memory from game event
    from npc_engine.capabilities.scratchpad import ScratchpadEntry
    scratchpad = mgr.capabilities["scratchpad"]
    scratchpad._add_entry(ScratchpadEntry(text="Player defended Finn from bullies", turn=1, importance=0.9))

    # Set mood from cutscene
    emo = mgr.capabilities["emotional_state"]
    emo.mood = "grateful"
    emo.intensity = 0.7

    # Adjust trust (player was kind)
    trust = mgr.capabilities["trust"]
    trust.level = min(100, trust.level + 20)

    # Build context — all mutations should appear
    mgr.shared_state["trust"] = {"level": trust.level}
    mgr.shared_state["emotional_state"] = {"mood": emo.mood, "intensity": emo.intensity}
    ctx = mgr.build_all_contexts("Thanks for helping me", token_budget=400)

    assert_in("mutation: scratchpad visible", "bullies", ctx)
    assert_in("mutation: mood visible", "grateful", ctx)
    assert_in("mutation: trust visible", "Trust:", ctx)

    # Persist and reload
    mgr.save_state()
    mgr2 = CapabilityManager("finn", npc.capability_configs, state_dir=tmpdir)
    assert_eq("trust persists", mgr2.capabilities["trust"].level, trust.level)
    assert_eq("mood persists", mgr2.capabilities["emotional_state"].mood, "grateful")
    assert_true("scratchpad persists", len(mgr2.capabilities["scratchpad"].entries) >= 1)


# ── Summary ──────────────────────────────────────────────────

success = results.summary()
sys.exit(0 if success else 1)
