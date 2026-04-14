#!/usr/bin/env python3
"""
Post-generation validator tests.

Focused unit tests for ``npc_engine/postgen.py``. The original module
shipped without unit tests — this file backfills coverage for the
validators that mattered most empirically and adds the new
wrong-addressee detection + repair added for the NPC dialogue
identity bleed fix.

Usage:
    python tests/test_postgen.py
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

NPC_ROOT = Path(__file__).parent.parent.resolve()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, str(NPC_ROOT))

from npc_engine.postgen import (  # noqa: E402
    detect_wrong_identity,
    detect_wrong_addressee,
    repair_wrong_addressee,
    validate_and_repair,
)


def _noah_profile() -> dict:
    return {
        "identity": {"name": "Noah", "role": "village elder"},
        "world_facts": ["The village has a well."],
        "personal_knowledge": ["I have a sealed letter from the old king."],
        "active_quests": [],
    }


def _mara_profile() -> dict:
    return {
        "identity": {"name": "Mara", "role": "merchant"},
        "world_facts": [],
        "personal_knowledge": [],
        "active_quests": [],
    }


# ── detect_wrong_identity (backfilled coverage) ─────────────────

def test_detect_wrong_identity_catches_i_am_other_name():
    """If Noah says 'I am Mara' the self-identity bleed detector fires."""
    profile = _noah_profile()
    assert detect_wrong_identity("I am Mara, the merchant.", profile) is True
    print("  [PASS] detect_wrong_identity_catches_i_am_other_name")


def test_detect_wrong_identity_ignores_correct_identity():
    """'I am Noah' from Noah is fine — no bleed."""
    profile = _noah_profile()
    assert detect_wrong_identity("I am Noah, the elder.", profile) is False
    print("  [PASS] detect_wrong_identity_ignores_correct_identity")


def test_detect_wrong_identity_ignores_reference_to_other_npc():
    """'Mara is a tricky one' from Noah is REFERENCING Mara, not
    claiming to be her. No bleed."""
    profile = _noah_profile()
    assert detect_wrong_identity("Mara is a tricky one, I warn you.", profile) is False
    print("  [PASS] detect_wrong_identity_ignores_reference_to_other_npc")


def test_detect_wrong_identity_needs_profile():
    """No profile means no ground truth to compare against — return
    False defensively."""
    assert detect_wrong_identity("I am Mara.", None) is False
    print("  [PASS] detect_wrong_identity_needs_profile")


# ── detect_wrong_addressee (the NEW fix) ────────────────────────

def test_detect_wrong_addressee_catches_greeting_with_other_npc():
    """Noah greeting the player as 'Mara' is the core bug: model
    picked up another NPC name from few-shot bleed."""
    profile = _noah_profile()
    hit, name = detect_wrong_addressee("Greetings, Mara! How can I help you?", profile)
    assert hit is True
    assert name == "mara"
    print("  [PASS] detect_wrong_addressee_catches_greeting_with_other_npc")


def test_detect_wrong_addressee_catches_various_greetings():
    """Multiple greeting forms all trigger the detector."""
    profile = _noah_profile()
    for phrase in [
        "Hello Kael, what brings you here?",
        "Hi Mara! Nice to see you.",
        "Hey Elara, come in.",
        "Well met, Bess.",
        "Good day, Pip.",
        "Good morning, Kael.",
    ]:
        hit, name = detect_wrong_addressee(phrase, profile)
        assert hit is True, f"missed: {phrase}"
        assert name != "noah", f"speaker's own name matched: {phrase}"
    print("  [PASS] detect_wrong_addressee_catches_various_greetings")


def test_detect_wrong_addressee_catches_polite_prefix():
    """'My dear Mara' and 'dear Kael' forms should also trigger."""
    profile = _noah_profile()
    hit, name = detect_wrong_addressee("My dear Mara, sit down.", profile)
    assert hit is True and name == "mara"
    hit, name = detect_wrong_addressee("dear Kael, please listen.", profile)
    assert hit is True and name == "kael"
    print("  [PASS] detect_wrong_addressee_catches_polite_prefix")


def test_detect_wrong_addressee_catches_trailing_comma_address():
    """Trailing comma-address forms like 'Be careful, Mara.' should
    trigger — the model is finishing by addressing a named person."""
    profile = _noah_profile()
    hit, name = detect_wrong_addressee("The village is unsafe, Mara.", profile)
    assert hit is True and name == "mara"
    print("  [PASS] detect_wrong_addressee_catches_trailing_comma_address")


def test_detect_wrong_addressee_catches_start_of_sentence_name():
    """'Mara, come here' with Mara at sentence start is also a direct
    address — the speaker is calling Mara by name."""
    profile = _noah_profile()
    hit, name = detect_wrong_addressee("Mara, come here at once.", profile)
    assert hit is True and name == "mara"
    print("  [PASS] detect_wrong_addressee_catches_start_of_sentence_name")


def test_detect_wrong_addressee_ignores_non_address_mention():
    """Referencing another NPC in narrative text (no address pattern)
    should not fire. 'Mara is suspicious' is commentary, not address."""
    profile = _noah_profile()
    hit, name = detect_wrong_addressee(
        "Mara is a suspicious woman and you should watch her.",
        profile,
    )
    # Start-of-sentence pattern would catch "Mara," but not "Mara is"
    # because the pattern requires a comma after the name. So this
    # MUST return False.
    assert hit is False, f"false positive: captured {name}"
    print("  [PASS] detect_wrong_addressee_ignores_non_address_mention")


def test_detect_wrong_addressee_ignores_speakers_own_name():
    """Noah speaking his own name ('I am Noah' or 'Greetings, Noah.')
    is a separate concern — wrong-addressee must not fire on the
    speaker's own name."""
    profile = _noah_profile()
    hit, _name = detect_wrong_addressee(
        "I stand as Noah, the elder of this village.",
        profile,
    )
    assert hit is False
    print("  [PASS] detect_wrong_addressee_ignores_speakers_own_name")


def test_detect_wrong_addressee_needs_profile():
    """No profile means no speaker → can't detect bleed."""
    hit, _ = detect_wrong_addressee("Hello Mara, welcome!", None)
    assert hit is False
    print("  [PASS] detect_wrong_addressee_needs_profile")


# ── repair_wrong_addressee ──────────────────────────────────────

def test_repair_wrong_addressee_replaces_name():
    """Basic replacement — 'Greetings, Mara!' becomes 'Greetings, traveler!'"""
    out = repair_wrong_addressee("Greetings, Mara! How can I help you?", "mara")
    assert "Mara" not in out and "mara" not in out, out
    assert "traveler" in out.lower(), out
    print("  [PASS] repair_wrong_addressee_replaces_name")


def test_repair_wrong_addressee_preserves_leading_capital():
    """Sentence-initial capital should survive — 'Mara, come here'
    becomes 'Traveler, come here'."""
    out = repair_wrong_addressee("Mara, come here at once.", "mara")
    assert out.startswith("Traveler"), out
    print("  [PASS] repair_wrong_addressee_preserves_leading_capital")


def test_repair_wrong_addressee_lowercase_midstring():
    """Mid-sentence name should become lowercase 'traveler'."""
    out = repair_wrong_addressee("The village is unsafe, Mara.", "mara")
    assert "Traveler" not in out, out
    assert "traveler" in out, out
    print("  [PASS] repair_wrong_addressee_lowercase_midstring")


def test_repair_wrong_addressee_respects_word_boundary():
    """'Mara' in 'Maralynn' (hypothetical, wider word) should NOT be
    replaced — only whole-word matches."""
    out = repair_wrong_addressee("Maralynn is a legend around here.", "mara")
    assert "Maralynn" in out, f"word boundary broken: {out}"
    print("  [PASS] repair_wrong_addressee_respects_word_boundary")


# ── validate_and_repair integration ─────────────────────────────

def test_validate_and_repair_fixes_wrong_addressee():
    """End-to-end: a Noah response that greets 'Mara' should be
    repaired in the returned JSON, with the rest of the response
    preserved."""
    profile = _noah_profile()
    raw = json.dumps({
        "dialogue": "Greetings, Mara! What brings you to my study?",
        "emotion": "warm",
        "action": None,
    })
    out = json.loads(validate_and_repair(raw, npc_id="noah", profile=profile,
                                          user_input="hello elder"))
    assert "Mara" not in out["dialogue"], out
    assert "traveler" in out["dialogue"].lower(), out
    # The rest of the dialogue (study, etc.) should survive
    assert "study" in out["dialogue"], out
    print("  [PASS] validate_and_repair_fixes_wrong_addressee")


def test_validate_and_repair_passes_clean_response_through():
    """A clean Noah response with no bleed should pass through
    largely unchanged."""
    profile = _noah_profile()
    raw = json.dumps({
        "dialogue": "The village is quiet today, traveler. How may I help?",
        "emotion": "calm",
        "action": None,
    })
    out = json.loads(validate_and_repair(raw, npc_id="noah", profile=profile,
                                          user_input="hello"))
    assert "village is quiet" in out["dialogue"], out
    print("  [PASS] validate_and_repair_passes_clean_response_through")


# ── Runner ──────────────────────────────────────────────────────

def main():
    print("Postgen — wrong-identity tests (backfill)")
    test_detect_wrong_identity_catches_i_am_other_name()
    test_detect_wrong_identity_ignores_correct_identity()
    test_detect_wrong_identity_ignores_reference_to_other_npc()
    test_detect_wrong_identity_needs_profile()

    print("\nPostgen — wrong-addressee detection")
    test_detect_wrong_addressee_catches_greeting_with_other_npc()
    test_detect_wrong_addressee_catches_various_greetings()
    test_detect_wrong_addressee_catches_polite_prefix()
    test_detect_wrong_addressee_catches_trailing_comma_address()
    test_detect_wrong_addressee_catches_start_of_sentence_name()
    test_detect_wrong_addressee_ignores_non_address_mention()
    test_detect_wrong_addressee_ignores_speakers_own_name()
    test_detect_wrong_addressee_needs_profile()

    print("\nPostgen — wrong-addressee repair")
    test_repair_wrong_addressee_replaces_name()
    test_repair_wrong_addressee_preserves_leading_capital()
    test_repair_wrong_addressee_lowercase_midstring()
    test_repair_wrong_addressee_respects_word_boundary()

    print("\nPostgen — validate_and_repair integration")
    test_validate_and_repair_fixes_wrong_addressee()
    test_validate_and_repair_passes_clean_response_through()

    print("\nAll postgen tests passed.")


if __name__ == "__main__":
    main()
