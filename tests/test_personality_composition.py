#!/usr/bin/env python3
"""
Personality-Composition Audit tests.

Pure-Python (no model). Covers:
  1. The composition metric (audit_composition) over mixed groups.
  2. Gossip-path wiring (GossipPropagator.audit_cluster + the env-gated
     flag stored on the propagator).
  3. Story-Director-path wiring (_maybe_audit_composition surfaces a
     composition_audit block on multi-NPC ticks only when flagged).
  4. Default behavior unchanged when the audit is disabled / healthy.

Style mirrors tests/test_story_director.py: plain def test_*() + assert +
a trailing PASS print, lightweight stubs instead of a live engine.

Usage:
    python tests/test_personality_composition.py
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

NPC_ROOT = Path(__file__).parent.parent.resolve()
PIE_ROOT = (NPC_ROOT.parent / "plug-in-intelligence-engine").resolve()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NPC_ENGINE_DEV_MODE", "1")
logging.basicConfig(level=logging.ERROR)

sys.path.insert(0, str(NPC_ROOT))
sys.path.insert(0, str(NPC_ROOT / "tests"))
sys.path.insert(0, str(PIE_ROOT))

# Import the Story Director test module for its stub engine machinery.
# Side effect: it rebinds sys.stdout to a UTF-8 TextIOWrapper at import
# time, which is exactly the encoding we want — so we reuse that wrapper
# rather than double-wrapping the same buffer (double-wrap closes it).
import test_story_director as _tsd  # noqa: E402

from npc_engine.social.personality_composition import (  # noqa: E402
    DEFAULT_LOW_AGREEABLENESS_THRESHOLD,
    audit_composition,
)


# ── Lightweight stubs ───────────────────────────────────────────

class _StubTrustCap:
    def __init__(self, level):
        self.level = level


class _StubCapManager:
    def __init__(self, trust_level=None):
        caps = {}
        if trust_level is not None:
            caps["trust"] = _StubTrustCap(trust_level)
        self.capabilities = caps
        # Mirror the production manager shape the propagator's fact
        # extractor reads (mgr.shared_state["trust"]).
        self.shared_state = {
            "trust": {"level": trust_level or 0, "trend": "stable"},
        }


def _managers(levels: dict) -> dict:
    """levels: {npc_id: trust_level_or_None}. None => manager with no
    trust capability (no readable disposition)."""
    return {npc: _StubCapManager(lvl) for npc, lvl in levels.items()}


# ── 1. Composition metric ───────────────────────────────────────

def test_metric_counts_low_agreeableness_correctly():
    # 3 low (<25), 1 high — among 4 readable participants.
    mgrs = _managers({"a": 5, "b": 10, "c": 20, "d": 80})
    audit = audit_composition(["a", "b", "c", "d"], mgrs)
    assert audit.n_participants == 4, audit.as_dict()
    assert audit.n_with_signal == 4, audit.as_dict()
    assert audit.n_low == 3, audit.as_dict()
    assert abs(audit.low_fraction - 0.75) < 1e-9, audit.as_dict()
    # mean agreeableness normalized to 0-1: (5+10+20+80)/4/100 = 0.2875
    assert abs(audit.mean_agreeableness - 0.2875) < 1e-9, audit.as_dict()
    print("  [PASS] metric_counts_low_agreeableness_correctly")


def test_metric_excludes_unreadable_from_denominator():
    # 2 readable (one low, one high) + 2 with no trust cap.
    mgrs = _managers({"a": 5, "b": 80, "c": None, "d": None})
    audit = audit_composition(["a", "b", "c", "d"], mgrs)
    assert audit.n_participants == 4, audit.as_dict()
    assert audit.n_with_signal == 2, audit.as_dict()
    assert audit.n_low == 1, audit.as_dict()
    # low_fraction is over readable signals only: 1/2 = 0.5 (not a majority)
    assert abs(audit.low_fraction - 0.5) < 1e-9, audit.as_dict()
    assert audit.flagged is False, audit.as_dict()
    print("  [PASS] metric_excludes_unreadable_from_denominator")


def test_metric_dedupes_repeated_ids():
    mgrs = _managers({"a": 5, "b": 80})
    audit = audit_composition(["a", "a", "b", "a"], mgrs)
    assert audit.n_participants == 2, audit.as_dict()
    assert audit.participants == ["a", "b"], audit.as_dict()
    print("  [PASS] metric_dedupes_repeated_ids")


# ── 2. Flag firing ──────────────────────────────────────────────

def test_low_majority_group_is_flagged():
    # 3 of 4 below the threshold -> 0.75 > 0.5 majority.
    mgrs = _managers({"a": 5, "b": 10, "c": 20, "d": 90})
    audit = audit_composition(["a", "b", "c", "d"], mgrs)
    assert audit.flagged is True, audit.as_dict()
    assert set(audit.low_participants) == {"a", "b", "c"}, audit.as_dict()
    print("  [PASS] low_majority_group_is_flagged")


def test_healthy_group_is_not_flagged():
    # Only 1 of 4 below threshold -> 0.25, not a majority.
    mgrs = _managers({"a": 10, "b": 60, "c": 70, "d": 90})
    audit = audit_composition(["a", "b", "c", "d"], mgrs)
    assert audit.flagged is False, audit.as_dict()
    print("  [PASS] healthy_group_is_not_flagged")


def test_exact_half_is_not_a_majority():
    # 2 of 4 low = 0.5, NOT strictly greater than 0.5 -> not flagged.
    mgrs = _managers({"a": 5, "b": 10, "c": 60, "d": 90})
    audit = audit_composition(["a", "b", "c", "d"], mgrs)
    assert abs(audit.low_fraction - 0.5) < 1e-9, audit.as_dict()
    assert audit.flagged is False, audit.as_dict()
    print("  [PASS] exact_half_is_not_a_majority")


def test_single_npc_group_never_flagged():
    mgrs = _managers({"a": 0})
    audit = audit_composition(["a"], mgrs)
    assert audit.flagged is False, audit.as_dict()
    print("  [PASS] single_npc_group_never_flagged")


def test_no_readable_signal_never_flagged():
    mgrs = _managers({"a": None, "b": None, "c": None})
    audit = audit_composition(["a", "b", "c"], mgrs)
    assert audit.n_with_signal == 0, audit.as_dict()
    assert audit.flagged is False, audit.as_dict()
    print("  [PASS] no_readable_signal_never_flagged")


def test_threshold_is_strict_less_than():
    # Level exactly at the threshold is NOT low.
    t = DEFAULT_LOW_AGREEABLENESS_THRESHOLD
    mgrs = _managers({"a": t, "b": t, "c": t})
    audit = audit_composition(["a", "b", "c"], mgrs)
    assert audit.n_low == 0, audit.as_dict()
    assert audit.flagged is False, audit.as_dict()
    print("  [PASS] threshold_is_strict_less_than")


# ── 3. Gossip-path wiring ───────────────────────────────────────

def _make_propagator(connections):
    """Build a GossipPropagator over an in-memory social graph.
    ``connections`` is a list of (from, to) directed edges."""
    from npc_engine.social.network import SocialGraph
    from npc_engine.social.propagation import GossipPropagator
    graph = SocialGraph()
    for frm, to in connections:
        graph.add_connection({"from": frm, "to": to, "gossip_filter": "all"})
    return GossipPropagator(graph)


def test_gossip_cluster_is_source_plus_reachable():
    prop = _make_propagator([("a", "b"), ("b", "c")])
    cluster = prop.gossip_cluster("a")
    # source + 2-hop reachable (default max_hops=2)
    assert cluster[0] == "a", cluster
    assert set(cluster) == {"a", "b", "c"}, cluster
    print("  [PASS] gossip_cluster_is_source_plus_reachable")


def test_audit_cluster_flags_low_trust_cluster():
    prop = _make_propagator([("a", "b"), ("a", "c")])
    mgrs = _managers({"a": 5, "b": 10, "c": 90})  # 2/3 low
    audit = prop.audit_cluster("a", mgrs)
    assert audit.flagged is True, audit.as_dict()
    print("  [PASS] audit_cluster_flags_low_trust_cluster")


def test_audit_cluster_healthy_not_flagged():
    prop = _make_propagator([("a", "b"), ("a", "c")])
    mgrs = _managers({"a": 60, "b": 70, "c": 5})  # 1/3 low
    audit = prop.audit_cluster("a", mgrs)
    assert audit.flagged is False, audit.as_dict()
    print("  [PASS] audit_cluster_healthy_not_flagged")


def test_propagate_does_not_audit_when_env_disabled():
    # Env var unset (cleared) -> propagator never records an audit.
    os.environ.pop("NPC_ENGINE_PERSONALITY_AUDIT", None)
    prop = _make_propagator([("a", "b"), ("a", "c")])
    assert prop._composition_audit_enabled is False
    assert prop.last_composition_audit is None
    print("  [PASS] propagate_does_not_audit_when_env_disabled")


def test_propagate_records_audit_when_env_enabled():
    # With the gate on, propagate() audits the cluster and stores the
    # result. Use the no-delay path so injection runs in the same call.
    from npc_engine.config import GossipRules
    from npc_engine.social.network import SocialGraph
    from npc_engine.social.propagation import GossipPropagator

    os.environ["NPC_ENGINE_PERSONALITY_AUDIT"] = "1"
    try:
        graph = SocialGraph()
        for frm, to in [("a", "b"), ("a", "c")]:
            graph.add_connection({"from": frm, "to": to, "gossip_filter": "all"})
        rules = GossipRules(max_hops=2, decay_per_hop=0.5,
                            min_significance=0.2, propagation_delay=0)
        prop = GossipPropagator(graph, rules)
        assert prop._composition_audit_enabled is True

        # Stub knowledge manager: just records injected events.
        injected = []

        class _KM:
            def inject_event(self, npc_id, text):
                injected.append((npc_id, text))

        mgrs = _managers({"a": 5, "b": 10, "c": 90})  # low-trust majority
        # A player_input that the pattern extractor turns into a fact.
        prop.propagate("a", "my name is Roland", "Greetings.", _KM(), mgrs)

        assert prop.last_composition_audit is not None
        assert prop.last_composition_audit.flagged is True, \
            prop.last_composition_audit.as_dict()
        # Behavior unchanged: facts still injected to reachable NPCs.
        assert any(t[0] in ("b", "c") for t in injected), injected
    finally:
        os.environ.pop("NPC_ENGINE_PERSONALITY_AUDIT", None)
    print("  [PASS] propagate_records_audit_when_env_enabled")


# ── 4. Story-Director-path wiring ───────────────────────────────

def _make_director_with_trust(profile_levels: dict):
    """Build a stub StoryDirector whose engine exposes capability_managers
    with the given per-NPC trust levels."""
    # Reuse the existing stub engine machinery from test_story_director
    # (imported at module top to control stdout-rebind ordering).
    tsd = _tsd
    specs = [(npc, "role") for npc in profile_levels]
    engine = tsd._make_stub_engine(profile_specs=specs)
    engine.pie.capability_managers = _managers(profile_levels)
    from npc_engine.story_director import StoryDirector
    return StoryDirector(engine)


def test_director_audit_disabled_returns_none():
    os.environ.pop("NPC_ENGINE_PERSONALITY_AUDIT", None)
    director = _make_director_with_trust({"a": 5, "b": 10, "c": 90})
    block = director._maybe_audit_composition(["a", "b", "c"])
    assert block is None, block
    print("  [PASS] director_audit_disabled_returns_none")


def test_director_audit_enabled_flags_low_majority():
    os.environ["NPC_ENGINE_PERSONALITY_AUDIT"] = "1"
    try:
        director = _make_director_with_trust({"a": 5, "b": 10, "c": 90})
        block = director._maybe_audit_composition(["a", "b", "c"])
        assert block is not None, "expected a composition_audit block"
        assert block["flagged"] is True, block
        assert "warning" in block, block
        assert block["n_low"] == 2, block
    finally:
        os.environ.pop("NPC_ENGINE_PERSONALITY_AUDIT", None)
    print("  [PASS] director_audit_enabled_flags_low_majority")


def test_director_audit_enabled_healthy_returns_none():
    os.environ["NPC_ENGINE_PERSONALITY_AUDIT"] = "1"
    try:
        director = _make_director_with_trust({"a": 60, "b": 70, "c": 5})
        block = director._maybe_audit_composition(["a", "b", "c"])
        assert block is None, block
    finally:
        os.environ.pop("NPC_ENGINE_PERSONALITY_AUDIT", None)
    print("  [PASS] director_audit_enabled_healthy_returns_none")


# ── Runner ──────────────────────────────────────────────────────

def main():
    print("Personality-Composition Audit — metric tests")
    test_metric_counts_low_agreeableness_correctly()
    test_metric_excludes_unreadable_from_denominator()
    test_metric_dedupes_repeated_ids()

    print("\nPersonality-Composition Audit — flag-firing tests")
    test_low_majority_group_is_flagged()
    test_healthy_group_is_not_flagged()
    test_exact_half_is_not_a_majority()
    test_single_npc_group_never_flagged()
    test_no_readable_signal_never_flagged()
    test_threshold_is_strict_less_than()

    print("\nPersonality-Composition Audit — gossip-path wiring tests")
    test_gossip_cluster_is_source_plus_reachable()
    test_audit_cluster_flags_low_trust_cluster()
    test_audit_cluster_healthy_not_flagged()
    test_propagate_does_not_audit_when_env_disabled()
    test_propagate_records_audit_when_env_enabled()

    print("\nPersonality-Composition Audit — Story-Director-path wiring tests")
    test_director_audit_disabled_returns_none()
    test_director_audit_enabled_flags_low_majority()
    test_director_audit_enabled_healthy_returns_none()

    print("\nAll personality-composition tests passed.")


if __name__ == "__main__":
    main()
