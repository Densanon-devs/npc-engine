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
    HALLUCINATION_FALLBACK,
    META_FALLBACK,
    OOD_FALLBACK,
    REAL_WORLD_FALLBACK,
    SAFE_FALLBACK,
    WORLD_KNOWN_TERMS,
    WORLD_NPC_NAMES,
    _FABRICATION_BLOCKLIST,
    _FABRICATION_BLOCKLIST_DEFAULT,
    _GENERIC_KNOWN_TERMS,
    _REAL_WORLD_BLOCKLIST,
    _REAL_WORLD_BLOCKLIST_DEFAULT,
    detect_fabrication,
    detect_real_world_entity,
    detect_wrong_addressee,
    detect_wrong_identity,
    load_world_fabrication_blocklist,
    load_world_fallbacks,
    load_world_known_terms,
    load_world_real_world_blocklist,
    rebuild_fabrication_pattern,
    rebuild_real_world_pattern,
    register_world_npcs,
    reset_fabrication_blocklist,
    reset_real_world_blocklist,
    reset_world_fallbacks,
    reset_world_known_terms,
    reset_world_npcs,
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


def test_detect_real_world_catches_multi_word():
    is_rw, term = detect_real_world_entity("Reports came from the Silicon Valley")
    assert is_rw, "'silicon valley' should match"
    # Match returns lowercased
    assert term == "silicon valley"
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
    """Word-boundary check: a hypothetical 'trump' entry must not match
    'trumpet'. Currently 'trump' has been DROPPED from the blocklist
    because it's a real English card term and verb (see the blocklist
    comment in postgen.py), so this is now testing the general
    word-boundary invariant — even after future re-addition, 'trumpet'
    must remain safe."""
    # Add 'trump' temporarily to verify the word boundary works
    _REAL_WORLD_BLOCKLIST.add("trump")
    rebuild_real_world_pattern()
    try:
        is_rw, term = detect_real_world_entity("The bard played a trumpet at the inn")
        assert not is_rw, f"trumpet should NOT match — got term={term!r}"
        # And confirm 'Trump' alone would match if it were on the list
        is_rw2, _ = detect_real_world_entity("Trump arrived at dawn")
        assert is_rw2, "with 'trump' in blocklist, 'Trump arrived' should match"
    finally:
        _REAL_WORLD_BLOCKLIST.discard("trump")
        rebuild_real_world_pattern()
    print("  [PASS] detect_real_world_does_not_match_trumpet")


def test_card_term_trump_is_NOT_in_blocklist():
    """'trump' was deliberately dropped because of the card-term
    overlap. Card-game NPCs using 'play your trump card' must NOT
    fire the real-world fallback."""
    assert "trump" not in _REAL_WORLD_BLOCKLIST
    is_rw, term = detect_real_world_entity("Play your trump card now, friend!")
    assert not is_rw, (
        f"'trump card' must not match — would false-positive in "
        f"card-game worlds. Got term={term!r}"
    )
    print("  [PASS] card_term_trump_is_NOT_in_blocklist")


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


def test_validate_and_repair_lets_clean_deflection_through():
    """A correct in-character deflection (model successfully refusing
    the real-world topic WITHOUT echoing the trigger word back) must
    pass through cleanly. This is the desired output that the few-shot
    deflection examples teach the model to produce. If the few-shots
    correctly echo-free, the model's output reaches the player intact;
    only the postgen backstop fires when the model fails."""
    raw = json.dumps({
        "dialogue": "I have not heard of this person you mention. Such a name is unknown in these lands.",
        "emotion": "puzzled",
        "action": None,
    })
    out = json.loads(
        validate_and_repair(raw, npc_id="noah", profile=_noah_profile(),
                            user_input="What do you think of Putin?")
    )
    assert "have not heard" in out["dialogue"], (
        f"clean deflection was swapped — got {out['dialogue']!r}"
    )
    # Specifically: the user's trigger word ("Putin") is NOT in the output,
    # so the backstop should NOT fire. The deflection passes through.
    assert "putin" not in out["dialogue"].lower(), (
        "deflection should not echo the trigger word"
    )
    print("  [PASS] validate_and_repair_lets_clean_deflection_through")


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


def test_blocklist_size_sanity():
    """Size check on the blocklist. The blocklist is a set, so
    duplicate entries in the source literal would be silently deduped
    by Python — this test cannot detect them. The size range below is
    a copy-paste-mistake / runaway-growth guard, not a duplicate
    detector."""
    assert isinstance(_REAL_WORLD_BLOCKLIST, set)
    assert 30 <= len(_REAL_WORLD_BLOCKLIST) <= 100, (
        f"blocklist size {len(_REAL_WORLD_BLOCKLIST)} is outside sanity range"
    )
    print(f"  [PASS] blocklist_size_sanity ({len(_REAL_WORLD_BLOCKLIST)} entries)")


def test_blocklist_excludes_known_false_positive_terms():
    """Terms that overlap fantasy/common-noun usage must NOT be in
    the blocklist. This guards against regression — if someone adds
    one of these back, the test fires."""
    ambiguous = {
        "apple",       # fruit
        "amazon",      # river/forest
        "drake",       # dragon term
        "musk",        # perfume / animal scent
        "uber",        # German prefix
        "claude",      # a name
        "cook",        # verb
        "gates",       # city/castle gates
        "xi",          # too short
        "gpt",         # too short
        "rome",        # fantasy overlap
        "paris",       # fantasy overlap
        "berlin",      # fantasy overlap
        "trump",       # card term ("trump card"), verb ("to trump")
        "pentagon",    # geometric shape used in fantasy magic
        "white house", # generic phrase ("a white house at the edge of the village")
        "altman",      # German "old man" — common in Germanic-themed worlds
    }
    overlap = ambiguous & _REAL_WORLD_BLOCKLIST
    assert not overlap, (
        f"blocklist contains ambiguous terms that would cause false positives: {overlap}"
    )
    print("  [PASS] blocklist_excludes_known_false_positive_terms")


def test_per_world_override_via_rebuild():
    """Per-world override: mutating the set + calling
    rebuild_real_world_pattern() must take effect for detection."""
    # Sanity check: 'biden' is on the list, so it should match
    is_rw, _ = detect_real_world_entity("Biden arrived at the village")
    assert is_rw, "Biden should match before override"

    # Remove 'biden' and rebuild
    _REAL_WORLD_BLOCKLIST.discard("biden")
    rebuild_real_world_pattern()
    try:
        is_rw_after, _ = detect_real_world_entity("Biden arrived at the village")
        assert not is_rw_after, "Biden should NOT match after override + rebuild"
    finally:
        # Restore for subsequent tests
        _REAL_WORLD_BLOCKLIST.add("biden")
        rebuild_real_world_pattern()

    # Confirm restored
    is_rw_restored, _ = detect_real_world_entity("Biden arrived at the village")
    assert is_rw_restored, "Biden should match again after restore"
    print("  [PASS] per_world_override_via_rebuild")


def test_rebuild_without_mutation_still_works():
    """Calling rebuild_real_world_pattern() on an unchanged set must
    not break detection."""
    rebuild_real_world_pattern()
    is_rw, _ = detect_real_world_entity("Putin says peace")
    assert is_rw, "detection broken after no-op rebuild"
    print("  [PASS] rebuild_without_mutation_still_works")


def test_rebuild_with_empty_blocklist_disables_detection():
    """Edge case: a game can disable real-world detection entirely
    by clearing the blocklist."""
    saved = set(_REAL_WORLD_BLOCKLIST)
    _REAL_WORLD_BLOCKLIST.clear()
    rebuild_real_world_pattern()
    try:
        is_rw, _ = detect_real_world_entity("Biden arrived with Tesla and Putin")
        assert not is_rw, "empty blocklist should match nothing"
    finally:
        _REAL_WORLD_BLOCKLIST.update(saved)
        rebuild_real_world_pattern()
    print("  [PASS] rebuild_with_empty_blocklist_disables_detection")


# ── Fabrication blocklist (promoted from inline list) ───────────


def test_detect_fabrication_catches_canonical_terms():
    """Each entry in _FABRICATION_BLOCKLIST should be detected when
    it appears in dialogue."""
    cases = {
        "vexnoria": "I have seen Vexnoria fall to the dragon",
        "drath'nul": "Drath'nul rises from the shadow",
        "shadow council": "the Shadow Council watches us all",
        "underdark": "we ventured into the underdark",
        "lor'anath": "Lor'anath was the first to speak",
        "seven kingdoms": "across the Seven Kingdoms",
        "chosen one": "thou art the chosen one",
        "prophecy of the": "fulfill the prophecy of the ancients",
    }
    for term, text in cases.items():
        is_fab, matched = detect_fabrication(text)
        assert is_fab, f"failed to catch {term!r} in {text!r}"
        assert matched == term, f"matched {matched!r}, expected {term!r}"
    print("  [PASS] detect_fabrication_catches_canonical_terms")


def test_detect_fabrication_word_boundary_safety():
    """Word boundary prevents fabrication terms from substring-matching
    legitimate longer words. Tests against future-expansion concerns —
    none of the current entries have known substring collisions."""
    safe_cases = [
        "fortunate son of the king",  # 'rune' is not on the list, but a guard
        "underdarken sky at dusk",    # underdarken should NOT match underdark
        "vexnorian artifacts",        # 'vexnorian' should NOT match 'vexnoria'
    ]
    for text in safe_cases:
        is_fab, matched = detect_fabrication(text)
        assert not is_fab, f"false positive on {text!r} — matched {matched!r}"
    print("  [PASS] detect_fabrication_word_boundary_safety")


def test_detect_fabrication_safe_content_passes():
    """Legitimate in-world dialogue should not trip fabrication."""
    safe_cases = [
        "Aye, the village is quiet today",
        "I tend to the sick and wounded",
        "Greetings, traveler",
        "The well water has turned bitter",
    ]
    for text in safe_cases:
        is_fab, _ = detect_fabrication(text)
        assert not is_fab, f"false positive on {text!r}"
    print("  [PASS] detect_fabrication_safe_content_passes")


def test_fabrication_rebuild_works():
    """Mutating the fabrication blocklist + rebuild takes effect."""
    is_fab_before, _ = detect_fabrication("the chosen one rises")
    assert is_fab_before, "'chosen one' should match before override"

    _FABRICATION_BLOCKLIST.discard("chosen one")
    rebuild_fabrication_pattern()
    try:
        is_fab_after, _ = detect_fabrication("the chosen one rises")
        assert not is_fab_after, "'chosen one' should NOT match after discard"
    finally:
        _FABRICATION_BLOCKLIST.add("chosen one")
        rebuild_fabrication_pattern()
    print("  [PASS] fabrication_rebuild_works")


def test_validate_and_repair_uses_new_fabrication_path():
    """End-to-end: a fabricated-fantasy response should still fire the
    hallucination fallback (preserving the prior behavior after the
    inline→module refactor)."""
    raw = json.dumps({
        "dialogue": "I have heard tales of the Shadow Council in distant lands.",
        "emotion": "wary", "action": None,
    })
    out = json.loads(validate_and_repair(
        raw, npc_id="noah", profile=_noah_profile(),
        user_input="Tell me of distant lands",
    ))
    # Fabrication path uses HALLUCINATION_FALLBACK
    assert "have not heard of such things" in out["dialogue"].lower(), (
        f"expected HALLUCINATION_FALLBACK, got {out['dialogue']!r}"
    )
    print("  [PASS] validate_and_repair_uses_new_fabrication_path")


# ── Per-world WORLD_KNOWN_TERMS loader ──────────────────────────


def test_generic_known_terms_excludes_world_specific():
    """The generic baseline must NOT contain world-specific NPC names
    or place names — those belong in per-world YAML."""
    world_specific = {"noah", "kael", "mara", "ashenvale", "moonpetal", "elara"}
    leaked = world_specific & _GENERIC_KNOWN_TERMS
    assert not leaked, (
        f"world-specific terms leaked into generic baseline: {leaked}"
    )
    print("  [PASS] generic_known_terms_excludes_world_specific")


def test_generic_known_terms_includes_universal_medieval_vocab():
    """The generic baseline must include common medieval-fantasy
    vocabulary that any world will use."""
    must_have = {"merchant", "guard", "elder", "traveler", "village",
                 "forest", "sword", "dragon", "stranger"}
    missing = must_have - _GENERIC_KNOWN_TERMS
    assert not missing, f"generic baseline missing universal terms: {missing}"
    print("  [PASS] generic_known_terms_includes_universal_medieval_vocab")


def test_load_world_known_terms_from_ashenvale():
    """Loading Ashenvale's YAML should add NPC names + place names
    to WORLD_KNOWN_TERMS."""
    reset_world_known_terms()
    # Pre-condition: Ashenvale-specific names not in the set yet
    assert "noah" not in WORLD_KNOWN_TERMS
    assert "ashenvale" not in WORLD_KNOWN_TERMS
    assert "moonpetal" not in WORLD_KNOWN_TERMS

    ashenvale_dir = NPC_ROOT / "data" / "worlds" / "ashenvale"
    added = load_world_known_terms(ashenvale_dir)
    assert added > 0, "Ashenvale YAML should add at least 1 term"

    # Post-condition: NPC names + places now present
    assert "noah" in WORLD_KNOWN_TERMS
    assert "kael" in WORLD_KNOWN_TERMS
    assert "ashenvale" in WORLD_KNOWN_TERMS
    assert "moonpetal" in WORLD_KNOWN_TERMS
    print(f"  [PASS] load_world_known_terms_from_ashenvale ({added} added)")


def test_load_world_known_terms_missing_yaml_is_graceful():
    """A directory without a known_terms.yaml falls back to the
    generic baseline without raising."""
    import tempfile
    reset_world_known_terms()
    with tempfile.TemporaryDirectory() as td:
        added = load_world_known_terms(Path(td))
        assert added == 0, "missing YAML should add 0 terms"
        # Generic baseline still intact
        assert "merchant" in WORLD_KNOWN_TERMS
    # Cleanup: restore Ashenvale terms for downstream tests
    load_world_known_terms(NPC_ROOT / "data" / "worlds" / "ashenvale")
    print("  [PASS] load_world_known_terms_missing_yaml_is_graceful")


def test_load_world_known_terms_nonexistent_path_is_graceful():
    """Loading from a nonexistent path is a no-op, not an error."""
    reset_world_known_terms()
    added = load_world_known_terms(NPC_ROOT / "definitely-does-not-exist")
    assert added == 0
    # Generic baseline intact
    assert "merchant" in WORLD_KNOWN_TERMS
    # Restore Ashenvale for downstream tests
    load_world_known_terms(NPC_ROOT / "data" / "worlds" / "ashenvale")
    print("  [PASS] load_world_known_terms_nonexistent_path_is_graceful")


def test_reset_world_known_terms_clears_world_specific():
    """Resetting should remove world-specific entries but keep
    the generic baseline."""
    # Make sure Ashenvale is loaded
    load_world_known_terms(NPC_ROOT / "data" / "worlds" / "ashenvale")
    assert "noah" in WORLD_KNOWN_TERMS

    reset_world_known_terms()
    assert "noah" not in WORLD_KNOWN_TERMS, "world-specific should be cleared"
    assert "merchant" in WORLD_KNOWN_TERMS, "generic baseline should remain"

    # Restore Ashenvale for downstream tests
    load_world_known_terms(NPC_ROOT / "data" / "worlds" / "ashenvale")
    print("  [PASS] reset_world_known_terms_clears_world_specific")


def test_load_idempotent():
    """Loading the same world twice should not duplicate or break."""
    reset_world_known_terms()
    first_added = load_world_known_terms(NPC_ROOT / "data" / "worlds" / "ashenvale")
    size_after_first = len(WORLD_KNOWN_TERMS)
    second_added = load_world_known_terms(NPC_ROOT / "data" / "worlds" / "ashenvale")
    size_after_second = len(WORLD_KNOWN_TERMS)
    assert size_after_first == size_after_second, (
        "second load should not change set size"
    )
    assert second_added == 0, "second load should report 0 new entries"
    print(f"  [PASS] load_idempotent ({first_added} first, {second_added} second)")


# ── WORLD_NPC_NAMES auto-derivation ─────────────────────────────


def test_register_world_npcs_populates_registry():
    """register_world_npcs() should populate WORLD_NPC_NAMES and
    also extend WORLD_KNOWN_TERMS so the hallucination layer
    doesn't flag legitimate cross-NPC references."""
    reset_world_npcs()
    reset_world_known_terms()
    added = register_world_npcs(["Noah", "Kael", "Captain Reva"])
    assert added == 3
    assert "noah" in WORLD_NPC_NAMES
    assert "kael" in WORLD_NPC_NAMES
    assert "captain reva" in WORLD_NPC_NAMES
    # Also added to known-terms so hallucination doesn't flag
    assert "noah" in WORLD_KNOWN_TERMS
    print("  [PASS] register_world_npcs_populates_registry")


def test_register_world_npcs_idempotent():
    reset_world_npcs()
    first = register_world_npcs(["Noah", "Kael"])
    second = register_world_npcs(["Noah", "Kael"])
    assert first == 2
    assert second == 0, "re-registering same names should report 0 new"
    print("  [PASS] register_world_npcs_idempotent")


def test_register_world_npcs_handles_garbage_input():
    reset_world_npcs()
    added = register_world_npcs(["Noah", "", None, 42, "  ", "  Kael  "])  # type: ignore[list-item]
    assert added == 2  # Noah + Kael (whitespace stripped)
    assert "noah" in WORLD_NPC_NAMES
    assert "kael" in WORLD_NPC_NAMES
    print("  [PASS] register_world_npcs_handles_garbage_input")


def test_detect_wrong_identity_uses_world_npc_names():
    """Wrong-identity detection must work for the active world's
    NPCs only — not a hardcoded Ashenvale set."""
    reset_world_npcs()
    register_world_npcs(["Captain Reva", "Finn", "Old Bones"])
    pirate_profile = {
        "identity": {"name": "Finn", "role": "deck hand"},
        "world_facts": [], "personal_knowledge": [],
        "active_quests": [], "recent_events": [],
    }
    # Finn says "I am Captain Reva" — wrong identity
    assert detect_wrong_identity("I am Captain Reva, master of this ship.",
                                  pirate_profile) is True
    # Finn says "I am Finn" — correct
    assert detect_wrong_identity("I am Finn, deck hand.", pirate_profile) is False
    print("  [PASS] detect_wrong_identity_uses_world_npc_names")


def test_detect_wrong_identity_skips_unregistered_world():
    """If no NPCs are registered, detect_wrong_identity returns False
    (no false positives)."""
    reset_world_npcs()
    profile = {"identity": {"name": "Anyone", "role": "merchant"}}
    assert detect_wrong_identity("I am Noah, the elder.", profile) is False, (
        "with empty WORLD_NPC_NAMES, no detection should fire"
    )
    print("  [PASS] detect_wrong_identity_skips_unregistered_world")


def test_reset_world_npcs_clears_registry():
    reset_world_npcs()
    register_world_npcs(["Noah", "Kael"])
    assert len(WORLD_NPC_NAMES) > 0
    reset_world_npcs()
    assert len(WORLD_NPC_NAMES) == 0
    print("  [PASS] reset_world_npcs_clears_registry")


# ── Per-world fallback dialogues ────────────────────────────────


def test_load_world_fallbacks_from_port_blackwater():
    """Port Blackwater ships a pirate-themed fallbacks.yaml that
    overrides all 5 sections."""
    reset_world_fallbacks()
    pb_dir = NPC_ROOT / "data" / "worlds" / "port_blackwater"
    updated = load_world_fallbacks(pb_dir)
    assert updated == 5, f"expected 5 sections updated, got {updated}"
    # Pirate flavor leaks into every fallback
    assert "landlubber" in META_FALLBACK["dialogue"].lower()
    assert "sail" in SAFE_FALLBACK["dialogue"].lower()
    assert "sea" in HALLUCINATION_FALLBACK["dialogue"].lower()
    assert "port blackwater" in OOD_FALLBACK["dialogue"].lower()
    assert "ship" in REAL_WORLD_FALLBACK["dialogue"].lower()
    # Cleanup
    reset_world_fallbacks()
    print("  [PASS] load_world_fallbacks_from_port_blackwater")


def test_load_world_fallbacks_missing_yaml_is_graceful():
    """A world without a fallbacks.yaml keeps default fallbacks."""
    reset_world_fallbacks()
    default_meta = META_FALLBACK["dialogue"]
    # Creation Museum doesn't ship fallbacks.yaml — should be no-op
    cm_dir = NPC_ROOT / "data" / "worlds" / "creation_museum"
    updated = load_world_fallbacks(cm_dir)
    assert updated == 0
    assert META_FALLBACK["dialogue"] == default_meta, (
        "missing YAML should not modify fallbacks"
    )
    print("  [PASS] load_world_fallbacks_missing_yaml_is_graceful")


def test_reset_world_fallbacks_restores_defaults():
    """After loading Port Blackwater fallbacks, reset should restore
    the medieval-fantasy defaults."""
    reset_world_fallbacks()
    default_meta = META_FALLBACK["dialogue"]
    load_world_fallbacks(NPC_ROOT / "data" / "worlds" / "port_blackwater")
    assert META_FALLBACK["dialogue"] != default_meta, "load should have changed it"
    reset_world_fallbacks()
    assert META_FALLBACK["dialogue"] == default_meta, "reset should restore default"
    print("  [PASS] reset_world_fallbacks_restores_defaults")


def test_fallback_dict_identity_preserved_after_reset():
    """The fallback constants are the SAME dict object before and
    after reset/load — only their contents change. Existing callers
    holding the reference (e.g. `META_FALLBACK = postgen.META_FALLBACK`
    at module import) still see the updated values."""
    saved_id = id(META_FALLBACK)
    reset_world_fallbacks()
    load_world_fallbacks(NPC_ROOT / "data" / "worlds" / "port_blackwater")
    assert id(META_FALLBACK) == saved_id, (
        "load_world_fallbacks must mutate META_FALLBACK in place, not rebind"
    )
    reset_world_fallbacks()
    print("  [PASS] fallback_dict_identity_preserved_after_reset")


def test_port_blackwater_real_world_fallback_when_pirate_says_putin():
    """End-to-end: with Port Blackwater fallbacks loaded, a model that
    leaks 'Putin' gets the pirate-themed fallback, not the medieval
    default."""
    reset_world_fallbacks()
    load_world_fallbacks(NPC_ROOT / "data" / "worlds" / "port_blackwater")
    raw = json.dumps({
        "dialogue": "Putin sails the eastern seas, they say.",
        "emotion": "wary", "action": None,
    })
    out = json.loads(validate_and_repair(
        raw, npc_id="finn", profile={
            "identity": {"name": "Finn", "role": "deck hand"},
            "world_facts": [], "personal_knowledge": [],
            "active_quests": [], "recent_events": [],
        }, user_input="Who sails these waters?",
    ))
    # Pirate flavor in the fallback
    assert "ship" in out["dialogue"].lower() or "waters" in out["dialogue"].lower(), (
        f"expected pirate-flavored fallback, got {out['dialogue']!r}"
    )
    # NOT the medieval default
    assert "wandered" not in out["dialogue"].lower()
    reset_world_fallbacks()
    print("  [PASS] port_blackwater_real_world_fallback_when_pirate_says_putin")


# ── Per-world blocklist YAML overrides ──────────────────────────


def test_load_world_real_world_blocklist_remove():
    """A YAML with `remove: [biden]` should drop biden from the
    blocklist + recompile pattern."""
    reset_real_world_blocklist()
    assert "biden" in _REAL_WORLD_BLOCKLIST

    # Write a temp YAML
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "real_world_blocklist.yaml").write_text(
            "remove:\n  - biden\n", encoding="utf-8"
        )
        mutations = load_world_real_world_blocklist(td_path)
        assert mutations == 1
        assert "biden" not in _REAL_WORLD_BLOCKLIST
        # Pattern recompiled — detection no longer fires on Biden
        is_rw, _ = detect_real_world_entity("Biden arrived at the dock")
        assert not is_rw, "after remove + rebuild, biden should not match"

    reset_real_world_blocklist()
    print("  [PASS] load_world_real_world_blocklist_remove")


def test_load_world_real_world_blocklist_add():
    """A YAML with `add: [acmecorp]` should add custom entries."""
    reset_real_world_blocklist()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "real_world_blocklist.yaml").write_text(
            "add:\n  - acmecorp\n", encoding="utf-8"
        )
        mutations = load_world_real_world_blocklist(td_path)
        assert mutations == 1
        assert "acmecorp" in _REAL_WORLD_BLOCKLIST
        is_rw, _ = detect_real_world_entity("AcmeCorp will save us all")
        assert is_rw, "after add + rebuild, acmecorp should match"

    reset_real_world_blocklist()
    print("  [PASS] load_world_real_world_blocklist_add")


def test_reset_real_world_blocklist_restores_default():
    """After loading a YAML override, reset should restore the
    original default set."""
    reset_real_world_blocklist()
    _REAL_WORLD_BLOCKLIST.discard("biden")
    rebuild_real_world_pattern()
    assert "biden" not in _REAL_WORLD_BLOCKLIST
    reset_real_world_blocklist()
    assert "biden" in _REAL_WORLD_BLOCKLIST
    # Default set is the same as what we shipped
    assert _REAL_WORLD_BLOCKLIST == set(_REAL_WORLD_BLOCKLIST_DEFAULT)
    print("  [PASS] reset_real_world_blocklist_restores_default")


def test_load_world_fabrication_blocklist_remove_chosen_one():
    """A high-fantasy game can remove 'chosen one' from the
    fabrication blocklist when it's canonical lore."""
    reset_fabrication_blocklist()
    assert "chosen one" in _FABRICATION_BLOCKLIST

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "fabrication_blocklist.yaml").write_text(
            "remove:\n  - chosen one\n", encoding="utf-8"
        )
        mutations = load_world_fabrication_blocklist(td_path)
        assert mutations == 1
        is_fab, _ = detect_fabrication("thou art the chosen one")
        assert not is_fab, "after removal, 'chosen one' should pass"

    reset_fabrication_blocklist()
    is_fab_restored, _ = detect_fabrication("thou art the chosen one")
    assert is_fab_restored, "reset should restore default behavior"
    print("  [PASS] load_world_fabrication_blocklist_remove_chosen_one")


def test_world_yaml_loaders_all_graceful_on_missing():
    """All four loaders should be no-ops when the YAML file is missing."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        assert load_world_known_terms(td_path) == 0
        assert load_world_fallbacks(td_path) == 0
        assert load_world_real_world_blocklist(td_path) == 0
        assert load_world_fabrication_blocklist(td_path) == 0
    print("  [PASS] world_yaml_loaders_all_graceful_on_missing")


# ── Other-worlds integration smoke ──────────────────────────────


def test_creation_museum_known_terms_load():
    """Creation Museum's known_terms.yaml should register the biblical
    figures + places."""
    reset_world_known_terms()
    added = load_world_known_terms(NPC_ROOT / "data" / "worlds" / "creation_museum")
    assert added > 0
    # Biblical NPCs
    for name in ("adam", "moses", "noah", "paul"):
        assert name in WORLD_KNOWN_TERMS, f"{name} missing"
    # Biblical geography
    for place in ("eden", "egypt", "jerusalem", "canaan"):
        assert place in WORLD_KNOWN_TERMS, f"{place} missing"
    # Cleanup
    reset_world_known_terms()
    print(f"  [PASS] creation_museum_known_terms_load ({added} added)")


def test_port_blackwater_known_terms_load():
    reset_world_known_terms()
    added = load_world_known_terms(NPC_ROOT / "data" / "worlds" / "port_blackwater")
    assert added > 0
    # Pirate NPCs
    for name in ("reva", "finn", "bones"):
        assert name in WORLD_KNOWN_TERMS, f"{name} missing"
    # Pirate places + items
    for term in ("tortuga", "cutlass", "doubloon", "rum"):
        assert term in WORLD_KNOWN_TERMS, f"{term} missing"
    reset_world_known_terms()
    print(f"  [PASS] port_blackwater_known_terms_load ({added} added)")


# ── Runner ──────────────────────────────────────────────────────


def main():
    print("Postgen — real-world entity detection (true positives)")
    test_detect_real_world_catches_putin()
    test_detect_real_world_catches_london()
    test_detect_real_world_catches_tesla()
    test_detect_real_world_catches_biden_single()
    test_detect_real_world_catches_multi_word()
    test_detect_real_world_catches_kremlin()
    test_detect_real_world_catches_case_insensitive()
    test_detect_real_world_catches_brand()
    test_detect_real_world_catches_celebrity()

    print("\nPostgen — real-world entity detection (true negatives)")
    test_detect_real_world_does_not_match_trumpet()
    test_card_term_trump_is_NOT_in_blocklist()
    test_detect_real_world_does_not_match_londoner()
    test_detect_real_world_does_not_match_in_world_content()
    test_detect_real_world_does_not_match_empty()
    test_detect_real_world_word_boundary_for_short_terms()

    print("\nPostgen — validate_and_repair end-to-end")
    test_validate_and_repair_swaps_in_fallback_for_putin()
    test_validate_and_repair_swaps_in_fallback_for_london()
    test_validate_and_repair_swaps_in_fallback_for_tesla()
    test_validate_and_repair_lets_clean_response_through()
    test_validate_and_repair_lets_clean_deflection_through()

    print("\nPostgen — system prompt + few-shot integration")
    test_system_prompt_includes_modern_world_clause()
    test_ashenvale_examples_include_real_world_deflection()

    print("\nPostgen — blocklist sanity checks")
    test_blocklist_size_sanity()
    test_blocklist_excludes_known_false_positive_terms()

    print("\nPostgen — per-world override mechanism")
    test_per_world_override_via_rebuild()
    test_rebuild_without_mutation_still_works()
    test_rebuild_with_empty_blocklist_disables_detection()

    print("\nPostgen — fabrication blocklist (module-level)")
    test_detect_fabrication_catches_canonical_terms()
    test_detect_fabrication_word_boundary_safety()
    test_detect_fabrication_safe_content_passes()
    test_fabrication_rebuild_works()
    test_validate_and_repair_uses_new_fabrication_path()

    print("\nPostgen — per-world WORLD_KNOWN_TERMS loader")
    test_generic_known_terms_excludes_world_specific()
    test_generic_known_terms_includes_universal_medieval_vocab()
    test_load_world_known_terms_from_ashenvale()
    test_load_world_known_terms_missing_yaml_is_graceful()
    test_load_world_known_terms_nonexistent_path_is_graceful()
    test_reset_world_known_terms_clears_world_specific()
    test_load_idempotent()

    print("\nPostgen — WORLD_NPC_NAMES auto-derivation")
    test_register_world_npcs_populates_registry()
    test_register_world_npcs_idempotent()
    test_register_world_npcs_handles_garbage_input()
    test_detect_wrong_identity_uses_world_npc_names()
    test_detect_wrong_identity_skips_unregistered_world()
    test_reset_world_npcs_clears_registry()

    print("\nPostgen — per-world fallback dialogues")
    test_load_world_fallbacks_from_port_blackwater()
    test_load_world_fallbacks_missing_yaml_is_graceful()
    test_reset_world_fallbacks_restores_defaults()
    test_fallback_dict_identity_preserved_after_reset()
    test_port_blackwater_real_world_fallback_when_pirate_says_putin()

    print("\nPostgen — per-world blocklist YAML overrides")
    test_load_world_real_world_blocklist_remove()
    test_load_world_real_world_blocklist_add()
    test_reset_real_world_blocklist_restores_default()
    test_load_world_fabrication_blocklist_remove_chosen_one()
    test_world_yaml_loaders_all_graceful_on_missing()

    print("\nPostgen — other-worlds integration smoke")
    test_creation_museum_known_terms_load()
    test_port_blackwater_known_terms_load()

    print("\nAll postgen real-world tests passed.")


if __name__ == "__main__":
    main()
