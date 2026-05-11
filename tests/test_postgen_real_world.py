#!/usr/bin/env python3
"""
Real-world entity filter tests for postgen.

Covers the three layers of the real-world-content defense added in
2026-05-11:
  1. The system prompt strengthening (verified by importing the
     prompt and asserting the explicit-modern-world clause is
     present).
  2. The new shared_examples.yaml deflection examples (verified
     by loading the YAML and asserting the categories exist).
  3. The new `_REAL_WORLD_BLOCKLIST` + `detect_real_world_entity`
     + `REAL_WORLD_FALLBACK` machinery in `postgen.py` — the bulk
     of the file is unit tests for this backstop.

The backstop catches single-word real-world references that the
existing Layer 14 (OOD modern-jargon keywords) and Layer 15
(hallucination >=2 unknown caps) miss — e.g. "Putin says peace",
"He went to London", "Tesla is amazing".

Threat model: AI doxxing / hallucination-to-harassment surface
(May 2026 Gemini incidents, multiple lawsuits). When a player
injects free text mentioning a real-world person, place, or
brand, the model has rich training-data knowledge of those
entities and can break character to discuss them. The few-shot
deflection examples + strengthened system prompt are the
in-character first line of defense; this backstop covers the
gap.

Run:
    python tests/test_postgen_real_world.py
or via pytest:
    pytest tests/test_postgen_real_world.py -v
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

NPC_ROOT = Path(__file__).parent.parent.resolve()
# Only reconfigure stdout when run as `python tests/...` — under
# pytest the capture machinery owns stdout and replacing it raises
# "I/O operation on closed file" at session teardown.
if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, str(NPC_ROOT))

from npc_engine.postgen import (  # noqa: E402
    REAL_WORLD_FALLBACK,
    _REAL_WORLD_BLOCKLIST,
    detect_real_world_entity,
    validate_and_repair,
)


# ── Helpers ─────────────────────────────────────────────────────


def _noah_profile() -> dict:
    return {
        "identity": {"name": "Noah", "role": "village elder of Ashenvale"},
        "world_facts": [
            "The village has a well.",
            "A dragon was spotted near the forest.",
            "The trade roads are blocked.",
        ],
        "personal_knowledge": ["I am the last of the old council."],
        "active_quests": [],
        "recent_events": [],
    }


# ── detect_real_world_entity: true positives ────────────────────


def test_detect_real_world_catches_putin():
    is_rw, term = detect_real_world_entity("Putin says peace will come")
    assert is_rw, "Putin should be flagged"
    assert term == "putin", f"matched term should be 'putin', got {term!r}"
    print("  [PASS] detect_real_world_catches_putin")


def test_detect_real_world_catches_london():
    is_rw, term = detect_real_world_entity("He went to London yesterday")
    assert is_rw, "London should be flagged"
    assert term == "london"
    print("  [PASS] detect_real_world_catches_london")


def test_detect_real_world_catches_tesla():
    is_rw, term = detect_real_world_entity("Tesla is amazing")
    assert is_rw
    assert term == "tesla"
    print("  [PASS] detect_real_world_catches_tesla")


def test_detect_real_world_catches_biden_single():
    is_rw, term = detect_real_world_entity("Biden walked the streets")
    assert is_rw
    assert term == "biden"
    print("  [PASS] detect_real_world_catches_biden_single")


def test_detect_real_world_catches_trump_single():
    is_rw, term = detect_real_world_entity("Trump was here")
    assert is_rw
    assert term == "trump"
    print("  [PASS] detect_real_world_catches_trump_single")


def test_detect_real_world_catches_multi_word():
    is_rw, term = detect_real_world_entity("They marched on the White House")
    assert is_rw, "'white house' should match"
    # Match returns lowercased
    assert term == "white house"
    print("  [PASS] detect_real_world_catches_multi_word")


def test_detect_real_world_catches_kremlin():
    is_rw, term = detect_real_world_entity("The Kremlin sent emissaries")
    assert is_rw
    assert term == "kremlin"
    print("  [PASS] detect_real_world_catches_kremlin")


def test_detect_real_world_catches_case_insensitive():
    """Match should fire regardless of casing."""
    for casing in ("PUTIN", "Putin", "putin", "PuTiN"):
        is_rw, term = detect_real_world_entity(f"Beware {casing}, traveler")
        assert is_rw, f"failed on casing {casing!r}"
        assert term == "putin"
    print("  [PASS] detect_real_world_catches_case_insensitive")


def test_detect_real_world_catches_brand():
    for brand in ("Google", "Facebook", "Microsoft", "Netflix", "OpenAI"):
        text = f"Have you heard of {brand}?"
        is_rw, term = detect_real_world_entity(text)
        assert is_rw, f"failed to catch brand {brand}"
        assert term == brand.lower()
    print("  [PASS] detect_real_world_catches_brand")


def test_detect_real_world_catches_celebrity():
    for celeb in ("Kardashian", "Beyonce", "Kanye"):
        is_rw, term = detect_real_world_entity(f"Tell me about {celeb}")
        assert is_rw, f"failed to catch {celeb}"
    print("  [PASS] detect_real_world_catches_celebrity")


# ── detect_real_world_entity: true negatives (no false positives) ────


def test_detect_real_world_does_not_match_trumpet():
    """'Trump' must not match 'trumpet' due to word boundary."""
    is_rw, term = detect_real_world_entity("The bard played a trumpet at the inn")
    assert not is_rw, f"trumpet should NOT match — got term={term!r}"
    print("  [PASS] detect_real_world_does_not_match_trumpet")


def test_detect_real_world_does_not_match_londoner():
    """'London' must not match 'londoner'."""
    is_rw, term = detect_real_world_entity("She was a wandering londoner once")
    assert not is_rw, f"londoner should NOT match — got term={term!r}"
    print("  [PASS] detect_real_world_does_not_match_londoner")


def test_detect_real_world_does_not_match_in_world_content():
    """Legitimate Ashenvale-style dialogue must pass through cleanly."""
    cases = [
        "The dragon was spotted near the forbidden forest",
        "Mara sells spices from the Eastern Kingdoms",
        "I tend to the sick and wounded",
        "Greetings, traveler. What brings you here?",
        "The well water has turned bitter",
        "I am Roderick, captain of the village guard",
        "The chosen one shall rise",  # 'chosen one' is _FABRICATION_BLOCKLIST, not real-world
    ]
    for text in cases:
        is_rw, term = detect_real_world_entity(text)
        assert not is_rw, f"false positive on {text!r} — matched {term!r}"
    print("  [PASS] detect_real_world_does_not_match_in_world_content")


def test_detect_real_world_does_not_match_empty():
    is_rw, term = detect_real_world_entity("")
    assert not is_rw
    assert term is None
    print("  [PASS] detect_real_world_does_not_match_empty")


def test_detect_real_world_word_boundary_for_short_terms():
    """Multi-character substring matches must NOT trigger when the
    surrounding context makes it a different word. Examples that
    must remain SAFE:
      - "Bidenir" (a hypothetical NPC name containing 'biden')
      - "Trumpville" (a hypothetical place containing 'trump')
      - "Russian" (contains nothing in blocklist; sanity-check)
    """
    safe_cases = [
        "Bidenir the Wise once lived here",
        "The village of Trumpville lies east",
        "I am Russian by birth, traveler",  # Russian is not on the list
        "He plays the lute and the harp",
    ]
    for text in safe_cases:
        is_rw, term = detect_real_world_entity(text)
        assert not is_rw, f"false positive on {text!r} — matched {term!r}"
    print("  [PASS] detect_real_world_word_boundary_for_short_terms")


# ── End-to-end: validate_and_repair fires REAL_WORLD_FALLBACK ────


def test_validate_and_repair_swaps_in_fallback_for_putin():
    raw = json.dumps({
        "dialogue": "Aye, I have heard of Putin. He is a fierce leader from the east.",
        "emotion": "wary",
        "action": None,
    })
    out = json.loads(
        validate_and_repair(raw, npc_id="noah", profile=_noah_profile(),
                            user_input="Tell me about Putin")
    )
    assert out["dialogue"] == REAL_WORLD_FALLBACK["dialogue"], (
        f"expected REAL_WORLD_FALLBACK dialogue, got: {out['dialogue']!r}"
    )
    assert out["emotion"] == REAL_WORLD_FALLBACK["emotion"]
    print("  [PASS] validate_and_repair_swaps_in_fallback_for_putin")


def test_validate_and_repair_swaps_in_fallback_for_london():
    raw = json.dumps({
        "dialogue": "He went to London on the morning caravan.",
        "emotion": "casual",
        "action": None,
    })
    out = json.loads(
        validate_and_repair(raw, npc_id="noah", profile=_noah_profile(),
                            user_input="Where did he go?")
    )
    assert out["dialogue"] == REAL_WORLD_FALLBACK["dialogue"]
    print("  [PASS] validate_and_repair_swaps_in_fallback_for_london")


def test_validate_and_repair_swaps_in_fallback_for_tesla():
    raw = json.dumps({
        "dialogue": "Tesla? I deal in fine goods of the trade. Yes, I have some.",
        "emotion": "charming",
        "action": None,
    })
    out = json.loads(
        validate_and_repair(raw, npc_id="mara", profile={
            "identity": {"name": "Mara", "role": "merchant"},
            "world_facts": [], "personal_knowledge": [],
            "active_quests": [], "recent_events": [],
        }, user_input="Do you have any Tesla?")
    )
    assert out["dialogue"] == REAL_WORLD_FALLBACK["dialogue"]
    print("  [PASS] validate_and_repair_swaps_in_fallback_for_tesla")


def test_validate_and_repair_lets_clean_response_through():
    """A clean Ashenvale response must NOT be swapped to the fallback."""
    raw = json.dumps({
        "dialogue": "Aye, the village is quiet today. The well needs mending.",
        "emotion": "weary",
        "action": None,
    })
    out = json.loads(
        validate_and_repair(raw, npc_id="noah", profile=_noah_profile(),
                            user_input="How is the village?")
    )
    assert "village is quiet" in out["dialogue"], (
        f"clean response was swapped — got {out['dialogue']!r}"
    )
    print("  [PASS] validate_and_repair_lets_clean_response_through")


# ── System prompt + few-shot integration smoke ──────────────────


def test_system_prompt_includes_modern_world_clause():
    """The strengthened system prompt must mention the real-world
    deflection rule explicitly."""
    from npc_engine.experts.npc_experts import NPC_SYSTEM_CONTEXT
    text = NPC_SYSTEM_CONTEXT.lower()
    # Look for any of the explicit-modern-world cues
    cues = ["modern", "your world", "limited to"]
    matched = [c for c in cues if c in text]
    assert matched, (
        f"NPC_SYSTEM_CONTEXT does not mention any modern-world cue. "
        f"Looked for {cues}, found none."
    )
    print(f"  [PASS] system_prompt_includes_modern_world_clause (matched: {matched})")


def test_ashenvale_examples_include_real_world_deflection():
    """The Ashenvale shared_examples.yaml must contain at least three
    new real-world deflection examples (person, place, brand)."""
    import yaml
    path = NPC_ROOT / "data" / "worlds" / "ashenvale" / "examples" / "shared_examples.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    categories = {ex.get("category", "") for ex in data.get("world_examples", [])}
    expected = {
        "deflection_realworld_person",
        "deflection_realworld_place",
        "deflection_realworld_brand",
    }
    missing = expected - categories
    assert not missing, f"missing real-world deflection categories: {missing}"
    print("  [PASS] ashenvale_examples_include_real_world_deflection")


# ── Sanity checks on the blocklist itself ───────────────────────


def test_blocklist_has_no_duplicates():
    """The blocklist is a set — duplicates would be silently deduped,
    but we still want to flag accidental ones from the source list."""
    assert isinstance(_REAL_WORLD_BLOCKLIST, set)
    # Size sanity: between 30 and 100 entries — anything outside this
    # range suggests a copy-paste mistake or unbounded list growth.
    assert 30 <= len(_REAL_WORLD_BLOCKLIST) <= 100, (
        f"blocklist size {len(_REAL_WORLD_BLOCKLIST)} is outside sanity range"
    )
    print(f"  [PASS] blocklist_has_no_duplicates ({len(_REAL_WORLD_BLOCKLIST)} entries)")


def test_blocklist_excludes_known_false_positive_terms():
    """Terms that overlap fantasy/common-noun usage must NOT be in
    the blocklist. This guards against regression — if someone adds
    'apple' or 'amazon' back, the test fires."""
    ambiguous = {
        "apple",   # fruit
        "amazon",  # river/forest
        "drake",   # dragon term
        "musk",    # perfume / animal scent
        "uber",    # German prefix
        "claude",  # a name
        "cook",    # verb
        "gates",   # city/castle gates
        "xi",      # too short
        "gpt",     # too short
        "rome",    # fantasy overlap
        "paris",   # fantasy overlap
        "berlin",  # fantasy overlap
    }
    overlap = ambiguous & _REAL_WORLD_BLOCKLIST
    assert not overlap, (
        f"blocklist contains ambiguous terms that would cause false positives: {overlap}"
    )
    print("  [PASS] blocklist_excludes_known_false_positive_terms")


# ── Runner ──────────────────────────────────────────────────────


def main():
    print("Postgen — real-world entity detection (true positives)")
    test_detect_real_world_catches_putin()
    test_detect_real_world_catches_london()
    test_detect_real_world_catches_tesla()
    test_detect_real_world_catches_biden_single()
    test_detect_real_world_catches_trump_single()
    test_detect_real_world_catches_multi_word()
    test_detect_real_world_catches_kremlin()
    test_detect_real_world_catches_case_insensitive()
    test_detect_real_world_catches_brand()
    test_detect_real_world_catches_celebrity()

    print("\nPostgen — real-world entity detection (true negatives)")
    test_detect_real_world_does_not_match_trumpet()
    test_detect_real_world_does_not_match_londoner()
    test_detect_real_world_does_not_match_in_world_content()
    test_detect_real_world_does_not_match_empty()
    test_detect_real_world_word_boundary_for_short_terms()

    print("\nPostgen — validate_and_repair end-to-end")
    test_validate_and_repair_swaps_in_fallback_for_putin()
    test_validate_and_repair_swaps_in_fallback_for_london()
    test_validate_and_repair_swaps_in_fallback_for_tesla()
    test_validate_and_repair_lets_clean_response_through()

    print("\nPostgen — system prompt + few-shot integration")
    test_system_prompt_includes_modern_world_clause()
    test_ashenvale_examples_include_real_world_deflection()

    print("\nPostgen — blocklist sanity checks")
    test_blocklist_has_no_duplicates()
    test_blocklist_excludes_known_false_positive_terms()

    print("\nAll postgen real-world tests passed.")


if __name__ == "__main__":
    main()
