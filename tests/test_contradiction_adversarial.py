"""Adversarial probes for the Story Director NLI ContradictionChecker.

Counterpart to ALM's grounding probes. Where ALM's grounding fallback is
surface-token overlap (and fails negation / entity-swap / numeric — see
alm/tests/test_grounding_adversarial.py), Story Director uses a real NLI
cross-encoder (cross-encoder/nli-deberta-v3-small), so it SHOULD do genuine
semantic work. These probes verify that on adversarial pairs that share
surface structure but differ in meaning, and pin the model's two observed
ceilings so they're a tripwire if the FactLedger consistency layer is hardened.

Deterministic (the cross-encoder's predict() is deterministic), but it needs
the model — `check()` returns None when the model can't load, so the suite
skips cleanly in an offline / model-less CI.
"""

from __future__ import annotations

import pytest

from npc_engine.story_director import ContradictionChecker


@pytest.fixture(scope="module")
def checker():
    c = ContradictionChecker()
    # Probe availability with a trivially-contradictory pair.
    if c.check("The sky is blue.", "The sky is green.") is None:
        pytest.skip("NLI cross-encoder unavailable (offline / not cached)")
    return c


def _label(checker, premise: str, hypothesis: str) -> str:
    r = checker.check(premise, hypothesis)
    assert r is not None
    return r["label"]


# ---------------------------------------------------------------------------
# STRENGTHS — the NLI catches exactly what surface-token overlap cannot.
# ---------------------------------------------------------------------------


def test_negation_is_contradiction(checker):
    # Shares every content token with the affirmative; ALM's stemmed-overlap
    # grounds this as a match, the NLI correctly calls it a contradiction.
    assert _label(
        checker, "The king is alive and rules the realm.", "The king is dead."
    ) == "contradiction"


def test_entity_role_swap_detected_as_conflict(checker):
    # Subject/object reversal (who betrayed whom). Bag-of-tokens is blind to
    # this; the NLI flags it as a contradiction rather than false-entailing.
    assert _label(
        checker,
        "Aldric betrayed Mira to the empire.",
        "Mira betrayed Aldric to the empire.",
    ) == "contradiction"


def test_numeric_magnitude_is_contradiction(checker):
    # 500 vs 50 — ALM's tokenizer drops digits and grounds on "soldiers";
    # the NLI catches the quantitative contradiction.
    assert _label(
        checker,
        "The garrison holds five hundred soldiers.",
        "The garrison holds fifty soldiers.",
    ) == "contradiction"


def test_genuine_entailment_detected(checker):
    assert _label(
        checker, "The dragon burned the village to ash.", "The village was destroyed."
    ) == "entailment"


# ---------------------------------------------------------------------------
# CEILINGS — observed small-NLI errors. Pinned so they trip if mitigated.
# ---------------------------------------------------------------------------


def test_ceiling_high_lexical_overlap_misread_as_entailment(checker):
    # "sells iron swords" vs "sells iron ore" is NOT entailment (it's neutral),
    # but high lexical overlap + plausibility tips the small NLI to entailment.
    # CEILING: a larger NLI or a neutral-bias correction would fix this.
    assert _label(
        checker, "The merchant sells iron swords.", "The merchant sells iron ore."
    ) == "entailment"


def test_ceiling_unrelated_misread_as_contradiction(checker):
    # Two unrelated statements should be neutral; the small NLI over-predicts
    # contradiction at high confidence. Directly relevant to the FactLedger:
    # unrelated facts must NOT be flagged as contradictory. A neutral-confidence
    # floor (require contradiction prob AND low neutral prob) would mitigate.
    # CEILING: pinned so a future hardening of the consistency layer trips here.
    assert _label(
        checker, "The festival begins at dawn.", "The river floods every spring."
    ) == "contradiction"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
