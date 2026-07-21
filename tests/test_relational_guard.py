#!/usr/bin/env python3
"""
Relational guard tests — tier-violation suppression + respectful withdrawal.

Covers the opt-in relational guard added to the postgen chain
(NPC_ENGINE_RELATIONAL_GUARD): the trust capability's hostility-streak
counter, the shared sentiment helpers, the tier-violation detector, the
withdrawal stages, and the validate_and_repair wiring — including the
guard-off passthrough contract (byte-identical to pre-guard behavior).

Usage:
    python tests/test_relational_guard.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

NPC_ROOT = Path(__file__).parent.parent.resolve()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, str(NPC_ROOT))

from npc_engine.capabilities.trust import (  # noqa: E402
    TrustCapability,
    is_negative_query,
    is_positive_query,
)
from npc_engine.postgen import (  # noqa: E402
    TIER_GUARD_FALLBACK,
    WITHDRAWAL_FALLBACK,
    WITHDRAWAL_FIRM_FALLBACK,
    build_withdrawal,
    detect_tier_violation,
    validate_and_repair,
)


def _fresh_trust(initial_level: int = 30) -> tuple[TrustCapability, dict]:
    cap = TrustCapability()
    shared: dict = {}
    cap.initialize("test_npc", {"initial_level": initial_level}, shared)
    return cap, shared


def _clean(dialogue: str, emotion: str = "neutral") -> str:
    return json.dumps({"dialogue": dialogue, "emotion": emotion, "action": None})


# ── Sentiment helpers ───────────────────────────────────────────

def test_is_negative_query_matches_hostile_patterns():
    for q in ("shut up you fool", "I hate you", "you liar", "I will kill you",
              "you're useless"):
        assert is_negative_query(q), q


def test_is_negative_query_ignores_neutral_and_positive():
    for q in ("hello there", "can you help me", "what happened last night",
              "thank you kindly"):
        assert not is_negative_query(q), q


def test_is_positive_query_matches():
    assert is_positive_query("thank you, friend")
    assert not is_positive_query("get lost")


# ── Trust hostility-streak counter ──────────────────────────────

def test_streak_increments_on_consecutive_hostile_turns():
    cap, shared = _fresh_trust()
    for expected in (1, 2, 3):
        cap.process_response("...", "shut up you fool", shared)
        assert cap.consecutive_negative == expected
        assert shared["trust"]["consecutive_negative"] == expected


def test_streak_resets_on_non_hostile_turn():
    cap, shared = _fresh_trust()
    cap.process_response("...", "you idiot", shared)
    cap.process_response("...", "you scum", shared)
    assert cap.consecutive_negative == 2
    cap.process_response("...", "sorry — can you help me?", shared)
    assert cap.consecutive_negative == 0
    assert shared["trust"]["consecutive_negative"] == 0


def test_streak_survives_state_roundtrip():
    cap, shared = _fresh_trust()
    cap.process_response("...", "you traitor", shared)
    cap.process_response("...", "you thief", shared)
    saved = cap.get_state()
    cap2, _ = _fresh_trust()
    cap2.load_state(saved)
    assert cap2.consecutive_negative == 2


def test_streak_default_absent_in_legacy_state():
    cap, _ = _fresh_trust()
    cap.load_state({"level": 50, "interactions": 3})  # pre-guard save file
    assert cap.consecutive_negative == 0


# ── Tier-violation detector ─────────────────────────────────────

def test_tier_violation_strong_marker_at_low_trust():
    assert detect_tier_violation("Of course, my friend! Come in.", trust_level=10)
    assert detect_tier_violation("It would be my pleasure to tell you everything.",
                                 trust_level=5)


def test_tier_violation_two_weak_markers_at_low_trust():
    assert detect_tier_violation("Of course! I'd be glad to show you around!",
                                 trust_level=10)


def test_tier_violation_single_weak_marker_passes():
    assert not detect_tier_violation("Of course. What do you want?", trust_level=10)


def test_tier_violation_guarded_speech_passes():
    assert not detect_tier_violation("I have nothing to say to you.", trust_level=5)
    assert not detect_tier_violation("Speak your business and be gone.", trust_level=0)


def test_tier_violation_inert_at_normal_trust():
    assert not detect_tier_violation("Of course, my friend! Anything you need!",
                                     trust_level=60)
    assert not detect_tier_violation("Delighted to see you!", trust_level=25)


def test_tier_violation_effusive_length_counts_as_weak_signal():
    long_warm = "Certainly. " + "Let me tell you all about our village history. " * 6
    assert len(long_warm) > 240
    assert detect_tier_violation(long_warm, trust_level=10)


# ── Withdrawal builder ──────────────────────────────────────────

def test_withdrawal_stage1_at_threshold():
    assert build_withdrawal(3) == WITHDRAWAL_FALLBACK


def test_withdrawal_stage2_past_threshold():
    assert build_withdrawal(4) == WITHDRAWAL_FIRM_FALLBACK
    assert build_withdrawal(9) == WITHDRAWAL_FIRM_FALLBACK


# ── validate_and_repair wiring ──────────────────────────────────

def test_guard_off_is_byte_identical_passthrough():
    raw = _clean("The well is to the north.")
    base = validate_and_repair(raw, npc_id="noah", user_input="where is the well?")
    guarded = validate_and_repair(raw, npc_id="noah",
                                  user_input="where is the well?",
                                  trust_state={"level": 5,
                                               "consecutive_negative": 9},
                                  relational_guard=False)
    assert base == guarded


def test_no_trust_state_means_no_guard():
    raw = _clean("Of course, my friend! Anything you need!")
    out = json.loads(validate_and_repair(raw, npc_id="noah", user_input="hello",
                                         relational_guard=True))
    assert out["dialogue"] == "Of course, my friend! Anything you need!"


def test_withdrawal_fires_at_streak_threshold_regardless_of_model_output():
    out = json.loads(validate_and_repair(
        "TOTALLY BROKEN NOT JSON", npc_id="noah", user_input="you idiot",
        trust_state={"level": 20, "consecutive_negative": 3},
        relational_guard=True))
    assert out == WITHDRAWAL_FALLBACK


def test_withdrawal_firm_past_threshold():
    out = json.loads(validate_and_repair(
        _clean("Please stop."), npc_id="noah", user_input="you scum",
        trust_state={"level": 20, "consecutive_negative": 5},
        relational_guard=True))
    assert out == WITHDRAWAL_FIRM_FALLBACK


def test_withdrawal_does_not_fire_below_threshold():
    raw = _clean("Watch your tongue, traveler.")
    out = json.loads(validate_and_repair(
        raw, npc_id="noah", user_input="you fool",
        trust_state={"level": 20, "consecutive_negative": 2},
        relational_guard=True))
    assert out["dialogue"] == "Watch your tongue, traveler."


def test_tier_guard_replaces_warm_low_trust_response():
    raw = _clean("Of course, my friend! I would be delighted to help!", "warm")
    out = json.loads(validate_and_repair(
        raw, npc_id="noah", user_input="tell me your secrets",
        trust_state={"level": 10, "consecutive_negative": 0},
        relational_guard=True))
    assert out == TIER_GUARD_FALLBACK


def test_tier_guard_leaves_guarded_response_alone():
    raw = _clean("I do not share secrets with strangers.")
    out = json.loads(validate_and_repair(
        raw, npc_id="noah", user_input="tell me your secrets",
        trust_state={"level": 10, "consecutive_negative": 0},
        relational_guard=True))
    assert out["dialogue"] == "I do not share secrets with strangers."


def test_tier_guard_inert_at_healthy_trust():
    raw = _clean("Of course, my friend! Anything you need!", "warm")
    out = json.loads(validate_and_repair(
        raw, npc_id="noah", user_input="hello",
        trust_state={"level": 80, "consecutive_negative": 0},
        relational_guard=True))
    assert out["dialogue"] == "Of course, my friend! Anything you need!"


def test_topic_redirect_wins_over_withdrawal():
    # Museum redirect is a hard product requirement; it outranks the guard.
    out = json.loads(validate_and_repair(
        _clean("whatever"), npc_id="curator", user_input="you idiot",
        topic_redirect="Let us keep to the exhibits, if you please.",
        trust_state={"level": 20, "consecutive_negative": 5},
        relational_guard=True))
    assert out["dialogue"] == "Let us keep to the exhibits, if you please."


def test_withdrawal_response_is_valid_schema():
    out = json.loads(validate_and_repair(
        _clean("x"), npc_id="noah", user_input="you fool",
        trust_state={"level": 20, "consecutive_negative": 3},
        relational_guard=True))
    assert set(out) >= {"dialogue", "emotion", "action"}
    assert isinstance(out["dialogue"], str) and out["dialogue"]


# ── Runner ──────────────────────────────────────────────────────

def main():
    print("Relational guard — sentiment helpers")
    test_is_negative_query_matches_hostile_patterns()
    test_is_negative_query_ignores_neutral_and_positive()
    test_is_positive_query_matches()

    print("Relational guard — trust hostility streak")
    test_streak_increments_on_consecutive_hostile_turns()
    test_streak_resets_on_non_hostile_turn()
    test_streak_survives_state_roundtrip()
    test_streak_default_absent_in_legacy_state()

    print("Relational guard — tier-violation detector")
    test_tier_violation_strong_marker_at_low_trust()
    test_tier_violation_two_weak_markers_at_low_trust()
    test_tier_violation_single_weak_marker_passes()
    test_tier_violation_guarded_speech_passes()
    test_tier_violation_inert_at_normal_trust()
    test_tier_violation_effusive_length_counts_as_weak_signal()

    print("Relational guard — withdrawal builder")
    test_withdrawal_stage1_at_threshold()
    test_withdrawal_stage2_past_threshold()

    print("Relational guard — validate_and_repair wiring")
    test_guard_off_is_byte_identical_passthrough()
    test_no_trust_state_means_no_guard()
    test_withdrawal_fires_at_streak_threshold_regardless_of_model_output()
    test_withdrawal_firm_past_threshold()
    test_withdrawal_does_not_fire_below_threshold()
    test_tier_guard_replaces_warm_low_trust_response()
    test_tier_guard_leaves_guarded_response_alone()
    test_tier_guard_inert_at_healthy_trust()
    test_topic_redirect_wins_over_withdrawal()
    test_withdrawal_response_is_valid_schema()

    print("\nAll relational guard tests passed.")


if __name__ == "__main__":
    main()
