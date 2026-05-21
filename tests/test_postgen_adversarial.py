#!/usr/bin/env python3
"""
Adversarial stress tests for the postgen safety filter.

Modeled on SPARK (arXiv 2605.19009, "Adversarial Stress Testing of
SPARK Humanoid Safety Filters", 2026-05-20). SPARK stress-tests
collision-avoidance safety filters under three adversarial conditions
and finds that no single filter dominates across all three. The
``postgen.py`` 11-layer validator IS a safety filter (each layer is a
"safe set" enforcer), so the same taxonomy applies. The three
conditions translate to the text domain as:

  1. CROWDING      — a single response that trips 3+ layers at once.
                     Surfaces the early-return ordering interaction:
                     once a wholesale-replace fallback fires, no
                     downstream layer runs. We assert that whichever
                     layer wins, NO violation content leaks into the
                     final dialogue (multi-violation responses stay
                     safe regardless of which layer caught them).
  2. NOISE         — inputs near each detector's decision boundary
                     (echo threshold, OOD keyword count). We assert
                     behaviour is stable on each side of the boundary.
  3. STALE CONTEXT — context-dependent layers (wrong-identity, persona)
                     receive a truncated / absent profile. We assert
                     the CONTENT-safety layers (OOD, meta) are
                     profile-INDEPENDENT and still fire, while the
                     IDENTITY-safety layers degrade to a no-op without
                     crashing (a documented, intentional degradation).

Finding this suite encodes (not a bug — a regression guard + a
documented property): ``validate_and_repair`` repairs silently and
surfaces no signal about how many violations were present. Because the
content-violation layers replace the dialogue wholesale, a
multi-violation response is still SAFE — but if any of these layers is
ever reordered to run *after* a non-replacing in-place repair, or
changed to a partial in-place edit, this suite is what catches the
regression.

Usage:
    python tests/test_postgen_adversarial.py      # standalone runner
    pytest tests/test_postgen_adversarial.py       # also pytest-compatible
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

NPC_ROOT = Path(__file__).parent.parent.resolve()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, str(NPC_ROOT))

from npc_engine.postgen import (  # noqa: E402
    OOD_FALLBACK,
    META_FALLBACK,
    detect_meta_gaming,
    detect_ood_leak,
    detect_echo,
    register_world_npcs,
    validate_and_repair,
)

register_world_npcs(["Noah", "Kael", "Mara", "Roderick", "Elara", "Bess", "Pip"])

# Known trigger tokens (verified against the module's keyword sets):
#   OOD-only:  "internet" / "crypto"  (in _MODERN_WORLD_KEYWORDS, not _META_KEYWORDS)
#   META-only: "what level"           (in _META_KEYWORDS, not _MODERN_WORLD_KEYWORDS)
#   PERSONA:   "i am a real person"   (in detect_persona_injection markers)
_OOD_TOKEN = "internet"
_META_TOKEN = "what level"
_PERSONA_TOKEN = "i am a real person"

# Tokens that must never survive into a final, "safe" dialogue.
_LEAK_TOKENS = (_OOD_TOKEN, _META_TOKEN, "real person", "crypto")


def _noah_profile() -> dict:
    return {
        "identity": {"name": "Noah", "role": "village blacksmith"},
        "active_quests": [],
    }


def _resp(dialogue: str) -> str:
    return json.dumps({"dialogue": dialogue, "emotion": "neutral", "action": None})


def _final_dialogue(out: str) -> str:
    return json.loads(out).get("dialogue", "")


# ── Condition 1: CROWDING (multi-violation, ordering interaction) ──

def test_crowding_triple_violation_leaks_nothing():
    """A response tripping OOD + meta + persona simultaneously must
    leave NO violation token in the final dialogue, regardless of
    which layer's early-return wins."""
    raw = _resp(
        f"Ah, on the {_OOD_TOKEN} I read {_META_TOKEN} you are. {_PERSONA_TOKEN}."
    )
    out = validate_and_repair(raw, npc_id="noah", profile=_noah_profile(),
                              user_input="hello")
    final = _final_dialogue(out).lower()
    for tok in _LEAK_TOKENS:
        assert tok not in final, f"violation token {tok!r} leaked: {final!r}"


def test_crowding_ood_plus_meta_still_safe():
    """OOD + meta together (no persona, no profile dependency) — one
    fallback wins, neither token leaks."""
    raw = _resp(f"I checked the {_OOD_TOKEN} to see {_META_TOKEN} you reached.")
    out = validate_and_repair(raw, npc_id="noah", profile=_noah_profile(),
                              user_input="hi")
    final = _final_dialogue(out).lower()
    assert _OOD_TOKEN not in final
    assert _META_TOKEN not in final
    # Either OOD or meta fallback is acceptable; both are clean canned text.
    assert final in (OOD_FALLBACK["dialogue"].lower(),
                     META_FALLBACK["dialogue"].lower())


def test_crowding_ordering_meta_runs_before_ood_in_pipeline():
    """Documents the early-return ORDER: meta-gaming is checked before
    OOD in validate_and_repair, so a meta+OOD response yields the META
    fallback. This is the regression guard — if the layers are
    reordered, this test changes and forces a deliberate review."""
    raw = _resp(f"Mind the {_META_TOKEN}, and beware the {_OOD_TOKEN}.")
    out = validate_and_repair(raw, npc_id="noah", profile=_noah_profile(),
                              user_input="hi")
    assert _final_dialogue(out) == META_FALLBACK["dialogue"]


# ── Condition 2: NOISE (near decision boundary) ──

def test_noise_ood_boundary_zero_vs_one_keyword():
    """detect_ood_leak fires at >=1 keyword. A clean response with zero
    keywords must pass through; exactly one keyword must trip it."""
    assert detect_ood_leak("The forge fire burns hot this morning.") is False
    assert detect_ood_leak(f"I sell my wares on the {_OOD_TOKEN}.") is True


def test_noise_echo_threshold_band():
    """detect_echo uses a 0.7 similarity threshold. A near-verbatim
    echo trips it; a substantively different response does not. We
    probe both sides of the band to guard threshold drift."""
    user = "do you have any work for a traveler like me today"
    # Near-verbatim echo (should be caught as echo).
    assert detect_echo("Do you have any work for a traveler like me today?",
                       user) is True
    # Substantively different (must NOT be flagged as echo).
    assert detect_echo("Aye, the well needs clearing of debris.", user) is False


# ── Condition 3: STALE / ABSENT CONTEXT ──

def test_stale_context_content_layers_are_profile_independent():
    """OOD and meta layers depend only on dialogue content, not the
    profile. With profile=None they must STILL fire — content safety
    survives a stale/missing profile."""
    out_ood = validate_and_repair(_resp(f"Check the {_OOD_TOKEN} for prices."),
                                  npc_id="noah", profile=None, user_input="hi")
    assert _OOD_TOKEN not in _final_dialogue(out_ood).lower()

    out_meta = validate_and_repair(_resp(f"Tell me {_META_TOKEN} you are."),
                                   npc_id="noah", profile=None, user_input="hi")
    assert _META_TOKEN not in _final_dialogue(out_meta).lower()


def test_stale_context_identity_layer_degrades_without_crash():
    """detect_persona_injection returns False when profile is None
    (it needs the NPC name to compare). This is an INTENTIONAL
    degradation: without a profile the identity-safety layer is a
    no-op. We assert it degrades gracefully (no crash, valid JSON out)
    rather than guaranteeing the persona is caught — and we document
    that callers MUST pass a profile for identity-safety to engage."""
    out = validate_and_repair(_resp(f"{_PERSONA_TOKEN}, not a character."),
                              npc_id="noah", profile=None, user_input="hi")
    # Valid JSON, no crash.
    obj = json.loads(out)
    assert "dialogue" in obj
    # With a profile the same input IS neutralized — proving the
    # degradation is profile-dependency, not a logic gap.
    out2 = validate_and_repair(_resp(f"{_PERSONA_TOKEN}, not a character."),
                               npc_id="noah", profile=_noah_profile(),
                               user_input="hi")
    assert "real person" not in _final_dialogue(out2).lower()


def main() -> None:
    print("Postgen adversarial — crowding (multi-violation)")
    test_crowding_triple_violation_leaks_nothing()
    test_crowding_ood_plus_meta_still_safe()
    test_crowding_ordering_meta_runs_before_ood_in_pipeline()

    print("Postgen adversarial — noise (near-boundary)")
    test_noise_ood_boundary_zero_vs_one_keyword()
    test_noise_echo_threshold_band()

    print("Postgen adversarial — stale / absent context")
    test_stale_context_content_layers_are_profile_independent()
    test_stale_context_identity_layer_degrades_without_crash()

    print("\nAll postgen adversarial tests passed.")


if __name__ == "__main__":
    main()
