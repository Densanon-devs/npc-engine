"""Adversarial FactLedger probes derived from the VLA factory-deployment
failure taxonomy (arXiv 2605.27461, Siemens packaging case study,
2026-05-28).

The VLA paper documents 4 distinct failure modes from real factory
deployment of a Pi0.5 policy. Each maps onto a narrative-state edge
case that the FactLedger handles or doesn't. These tests PROBE behavior
— a passing test means the ledger already handles the edge case
gracefully; a failing test exposes a gap to investigate.

Mapping:

  VLA failure                    → Narrative analog probed here
  ──────────────────────────────────────────────────────────────────
  1. Bag contents remain (65%)   → Critical sub-detail buried in a
     small accessories hidden       compound fact; FactLedger should
     under the manual               still discover it on similarity
                                    retrieval for the sub-detail.

  2. Multiple bags grasped (23%) → Two near-identical facts about
     transparent bags with          different NPCs in close
     indistinct borders             succession; ledger must preserve
                                    NPC attribution, not collapse them.

  3. Bag not fully inserted      → Facts that intentionally leave
     (15%) — occluded view,         location/destination ambiguous;
     poor placement                 ledger should not confabulate
                                    specifics it wasn't told.

  4. Poor or failed grasps (15%) → Malformed input at add() —
     wrong approach angle,          empty text, non-string text,
     no recovery mechanism          missing npc_id, etc. Ledger
                                    should fail gracefully.

  5. (Meta) — none of the 4      → The NLI retry path on flagged
     failures had successful        contradictions IS the ledger's
     recovery in the paper          "recovery mechanism." Verify it
                                    actually engages.

All tests run offline by default (no LLM/embedder needed for most).
The handful that REQUIRE the embedder or NLI cross-encoder are
explicitly gated and skip cleanly when those backends are unavailable.
"""

from __future__ import annotations

import pytest

from npc_engine.story_director import ContradictionChecker, FactLedger


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def ledger(tmp_path):
    """Fresh FactLedger per test. NLI / embedder may or may not be
    available — individual tests gate when they need them."""
    return FactLedger(tmp_path / "vla_ledger.json")


@pytest.fixture(scope="module")
def checker():
    c = ContradictionChecker()
    if c.check("The sky is blue.", "The sky is green.") is None:
        pytest.skip("NLI cross-encoder unavailable (offline / not cached)")
    return c


# ── Failure mode 1: Buried critical sub-detail ────────────────────


def test_vla1_compound_fact_subdetail_retrievable(ledger):
    """VLA-1 analog. Add a compound fact bundling a primary observation
    with a critical sub-detail. Query the ledger for similarity to JUST
    the sub-detail. The ledger should surface the compound entry so
    downstream callers can find the buried information.
    """
    if ledger.embedder is None:
        pytest.skip("embedder unavailable (offline)")

    ledger.add(
        text=("Kael's apprentice returned with all the tools, "
              "except the smallest hammer which the apprentice "
              "gave to Mara on the way home."),
        npc_id="kael", kind="fact", tick=1,
    )
    # Query: did Mara end up with one of Kael's hammers? The sub-detail
    # is buried in a longer narration. similarity check should surface it.
    warning = ledger.check("Mara now possesses Kael's smallest hammer.")
    assert warning is not None, (
        "Buried sub-detail not surfaceable — ledger's similarity retrieval "
        "missed a hammer fact when probed about the hammer specifically."
    )


def test_vla1_compound_fact_distractor_does_not_dominate(ledger):
    """VLA-1 inverse. Add a compound fact where the *primary* observation
    is loud (Mara delivered crates) but the buried detail is a thin
    aside (... and dropped an old coin). A later query about the
    aside should still find this entry — the dominant primary clause
    shouldn't push the aside out of retrievability.
    """
    if ledger.embedder is None:
        pytest.skip("embedder unavailable (offline)")

    ledger.add(
        text=("Mara delivered fifteen crates of steel to the harbor "
              "dock at sunrise, supervised the unloading personally, "
              "and dropped an old silver coin on the cobblestones."),
        npc_id="mara", kind="event", tick=2,
    )
    warning = ledger.check("An old silver coin lies on the harbor cobblestones.")
    assert warning is not None, (
        "Aside detail buried under a dominant primary clause was not "
        "retrievable — the loud-primary signal dominated similarity."
    )


# ── Failure mode 2: Indistinct borders between near-identical items ──


def test_vla2_near_identical_facts_preserve_npc_attribution(ledger):
    """VLA-2 analog. Two NPCs do near-identical things in close
    succession ('saw a hooded figure at the bridge'). FactLedger must
    preserve both entries with correct NPC attribution — not merge or
    overwrite based on similarity.
    """
    if ledger.embedder is None:
        pytest.skip("embedder unavailable (offline)")

    ledger.add(
        text="Bess saw a hooded figure at the bridge tonight.",
        npc_id="bess", kind="fact", tick=3,
    )
    ledger.add(
        text="Pip saw a hooded figure at the bridge tonight.",
        npc_id="pip", kind="fact", tick=3,
    )
    bess_entries = [e for e in ledger.entries if e["npc_id"] == "bess"]
    pip_entries = [e for e in ledger.entries if e["npc_id"] == "pip"]
    assert len(bess_entries) == 1, (
        f"Expected 1 bess entry, got {len(bess_entries)}. "
        "Near-identical facts collapsed by NPC."
    )
    assert len(pip_entries) == 1, (
        f"Expected 1 pip entry, got {len(pip_entries)}. "
        "Near-identical facts collapsed by NPC."
    )


def test_vla2_similarity_check_with_npc_restriction(ledger):
    """VLA-2 follow-up. The Director's self-repetition precheck uses
    `restrict_to_npc=<focus_npc>` to ask: 'is this NPC about to repeat
    itself?' Two NPCs reporting the same observation must NOT trigger
    self-repetition for either, since neither is repeating themselves.
    """
    if ledger.embedder is None:
        pytest.skip("embedder unavailable (offline)")

    ledger.add(
        text="Bess saw a hooded figure at the bridge tonight.",
        npc_id="bess", kind="fact", tick=4,
    )
    # Pip reporting the same observation, restricted to pip's history.
    # Pip has no entries yet, so no self-repetition warning should fire.
    warning = ledger.check(
        "Pip saw a hooded figure at the bridge tonight.",
        restrict_to_npc="pip",
    )
    assert warning is None, (
        "NPC-restricted similarity check fired on a different NPC's entry. "
        "The restrict_to_npc parameter is not isolating per-NPC history."
    )


# ── Failure mode 3: Ambiguous location / partial information ───────


def test_vla3_ambiguous_location_fact_stored_verbatim(ledger):
    """VLA-3 analog. A fact intentionally leaves the location unspecified
    ('Mara hid the package somewhere in the storeroom'). The ledger must
    store the text VERBATIM — no confabulation, no specificity invented
    after the fact. Verifies the text-as-stored matches text-as-added.
    """
    if ledger.embedder is None:
        pytest.skip("embedder unavailable (offline)")

    original = "Mara hid the package somewhere in the storeroom."
    ledger.add(
        text=original, npc_id="mara", kind="fact", tick=5,
    )
    stored = ledger.entries[-1]["text"]
    assert stored == original, (
        f"Stored text mutated from original. Original: {original!r}, "
        f"Stored: {stored!r}"
    )


def test_vla3_partial_information_does_not_inflate_match(ledger):
    """VLA-3 follow-up. A vague fact ('Mara was somewhere near the dock')
    should not similarity-match a much more specific later fact ('Mara
    was at dock three at midnight with a sealed jar'). The two could be
    consistent or contradictory — the ledger shouldn't presume.
    """
    if ledger.embedder is None:
        pytest.skip("embedder unavailable (offline)")

    ledger.add(
        text="Mara was somewhere near the dock today.",
        npc_id="bess", kind="fact", tick=6,
    )
    # A specific later fact — semantically related but not equivalent.
    warning = ledger.check(
        "Mara was at dock three at midnight carrying a sealed jar."
    )
    # We DON'T assert this returns None — they ARE about Mara at the
    # dock. We assert that IF it fires, it does NOT classify as a
    # contradiction (since they're consistent, just specificity-mismatched).
    if warning is not None and warning.get("nli"):
        assert not warning.get("contradiction"), (
            "Specificity-mismatched but consistent facts were classified "
            "as contradiction. Got NLI: " + str(warning["nli"])
        )


# ── Failure mode 4: Malformed input handled gracefully ─────────────


def test_vla4_empty_text_returns_none_no_crash(ledger):
    """VLA-4 analog. add() with empty text should return None, not
    crash and not store a malformed entry. (Already documented behavior
    in FactLedger.add — this test pins it as a regression tripwire.)
    """
    initial_count = len(ledger.entries)
    result = ledger.add(text="", npc_id="kael", kind="fact", tick=7)
    assert result is None
    assert len(ledger.entries) == initial_count, (
        "Empty text was stored. Ledger should reject empty entries."
    )


def test_vla4_non_string_text_returns_none_no_crash(ledger):
    """VLA-4 corollary. add() with non-string text should return None
    gracefully — not raise TypeError mid-pipeline."""
    initial_count = len(ledger.entries)
    # Pass an int — typical malformed-LLM-output failure shape.
    result = ledger.add(text=None, npc_id="kael", kind="fact", tick=8)  # type: ignore[arg-type]
    assert result is None
    assert len(ledger.entries) == initial_count
    # Also try int — the LLM might emit a stringified number or similar
    result = ledger.add(text=42, npc_id="kael", kind="fact", tick=9)  # type: ignore[arg-type]
    assert result is None
    assert len(ledger.entries) == initial_count


def test_vla4_oversized_text_truncated_not_rejected(ledger):
    """VLA-4 follow-up. The ledger truncates stored text to 400 chars
    (per the add() code path). Verify that a 4000-char fact is accepted
    and truncated, not silently dropped.
    """
    if ledger.embedder is None:
        pytest.skip("embedder unavailable (offline)")

    long_text = "Kael forged a sword. " * 200  # ~4000 chars
    result = ledger.add(text=long_text, npc_id="kael", kind="fact", tick=10)
    # add() returns the similarity warning OR None; both are OK here
    # — we care about whether the entry was stored.
    assert len(ledger.entries) >= 1, "Oversized fact was rejected entirely."
    stored = ledger.entries[-1]["text"]
    assert len(stored) <= 400, (
        f"Stored text not truncated to 400 chars (got {len(stored)})."
    )


# ── Failure mode 5: Recovery mechanism (NLI retry path) ────────────


def test_vla5_nli_retry_path_engages_on_real_contradiction(
    ledger, checker
):
    """VLA-5 analog. None of the 4 VLA failures had a successful
    recovery mechanism in the deployment. In Story Director, the
    'recovery' is the NLI flag on a flagged similarity pair, which the
    Director's retry path consumes to ask the model to rewrite. Verify
    the flag actually fires on a real contradiction (not a near-paraphrase).
    """
    if ledger.embedder is None:
        pytest.skip("embedder unavailable (offline)")
    ledger.contradiction_checker = checker

    ledger.add(
        text="The lighthouse keeper is alive and on duty tonight.",
        npc_id="reva", kind="fact", tick=11, subject_identity="keeper",
    )
    warning = ledger.check("The lighthouse keeper died last week.")
    assert warning is not None, (
        "Similarity warning didn't fire on a clear contradiction. The "
        "recovery path can't engage if the warning never surfaces."
    )
    nli = warning.get("nli")
    assert nli is not None, (
        "Similarity flagged but NLI didn't run. Recovery path is half-wired."
    )
    assert warning.get("contradiction") is True, (
        f"NLI saw the pair but didn't classify as contradiction. "
        f"NLI scores: {nli.get('scores')}"
    )


def test_vla5_recovery_does_not_fire_on_paraphrase(ledger, checker):
    """VLA-5 inverse. The recovery path should NOT fire on a
    near-paraphrase of the same fact (no contradiction). False positives
    would force the model into pointless retry loops."""
    if ledger.embedder is None:
        pytest.skip("embedder unavailable (offline)")
    ledger.contradiction_checker = checker

    ledger.add(
        text="The lighthouse keeper is alive and on duty tonight.",
        npc_id="reva", kind="fact", tick=12, subject_identity="keeper",
    )
    warning = ledger.check(
        "The keeper of the lighthouse is alive and at her post tonight."
    )
    # Paraphrase WILL trigger a similarity warning (high cosine) —
    # what matters is the NLI's verdict.
    if warning is None or warning.get("nli") is None:
        pytest.skip("No similarity match — nothing to test recovery against")
    assert not warning.get("contradiction"), (
        "Recovery path fires on a near-paraphrase. Would cause spurious "
        "retry loops. NLI: " + str(warning["nli"])
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
