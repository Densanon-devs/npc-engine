"""
Personality-Composition Audit — group-level disposition risk signal.

Motivated by "When Does Personality Composition Matter for Multi-Agent
LLM Teams?" (arXiv:2606.27443). The headline finding: agreeableness
effects are *task-structure-dependent*. In open-ended collaboration and
bargaining, a low-agreeableness **majority** substantially degrades group
outcomes — even when each individual agent's local facts/checks pass.
Per-agent verification therefore cannot catch the failure mode; it is a
property of the group's *composition*.

Applied here: when a set of interacting NPCs skews uncooperative, the
high-stakes social machinery that assumes good-faith consistency —
gossip propagation fidelity, multi-NPC negotiation/alliance ticks —
becomes unreliable and should be flagged (and optionally handled
differently) rather than trusted blindly.

Trait mapping
-------------
The NPC schema has no OCEAN / Big-Five "agreeableness" field. The
closest ranged dispositional signal is ``TrustCapability.level``
(int 0-100; low trust = "speak evasively, withhold information, be
suspicious" — the uncooperative pole the paper measures). We treat
trust as a proxy for agreeableness. If a richer trait ever lands, only
``_agreeableness_for`` below needs to change.

This module is pure / behavior-neutral: it *computes and reports*. It
never mutates NPC state. Callers decide whether to log, warn, or route
to a conflict-resolution branch. Disabled-by-default at the call sites
(env-gated), so default engine behavior is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Default boundary (on the 0-100 trust scale) below which a participant
# is counted as "low-agreeableness". 25 is the TrustCapability "neutral"
# threshold — anyone in the wary band counts as uncooperative.
DEFAULT_LOW_AGREEABLENESS_THRESHOLD: float = 25.0

# Trust scale is 0-100; agreeableness is normalized to 0-1 for the report.
_TRUST_SCALE_MAX: float = 100.0

# Fraction of low-agreeableness participants above which the group is a
# "low-agreeableness majority". Default >50%.
DEFAULT_MAJORITY_FRACTION: float = 0.5

# Minimum group size before the audit is meaningful. A 1-NPC "group"
# has no composition to speak of; a 2-NPC group is the smallest case
# where a majority is well-defined.
_MIN_GROUP_SIZE: int = 2


@dataclass
class CompositionAudit:
    """Result of a personality-composition audit over a group of NPCs.

    ``flagged`` is the single load-bearing field: True iff a
    low-agreeableness majority was detected over a group large enough to
    matter. Everything else is diagnostic detail for logging / routing.
    """

    n_participants: int
    n_with_signal: int            # how many had a readable disposition
    n_low: int                    # how many fell below the threshold
    low_fraction: float           # n_low / n_with_signal (0.0 if none readable)
    mean_agreeableness: Optional[float]  # 0-1, over participants with a signal
    threshold: float
    majority_fraction: float
    flagged: bool
    participants: list[str] = field(default_factory=list)
    low_participants: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_participants": self.n_participants,
            "n_with_signal": self.n_with_signal,
            "n_low": self.n_low,
            "low_fraction": round(self.low_fraction, 3),
            "mean_agreeableness": (
                round(self.mean_agreeableness, 3)
                if self.mean_agreeableness is not None else None
            ),
            "threshold": self.threshold,
            "majority_fraction": self.majority_fraction,
            "flagged": self.flagged,
            "participants": list(self.participants),
            "low_participants": list(self.low_participants),
        }


def _agreeableness_for(npc_id: str, capability_managers: dict) -> Optional[float]:
    """Read an agreeableness proxy (trust level, 0-100) for one NPC.

    Returns ``None`` when no disposition signal is available (no manager,
    no trust capability, or no level attribute) so the caller can exclude
    that participant from the denominator rather than guessing.
    """
    if not capability_managers:
        return None
    mgr = capability_managers.get(npc_id)
    if mgr is None:
        return None
    caps = getattr(mgr, "capabilities", None)
    if not caps:
        return None
    trust_cap = caps.get("trust")
    if trust_cap is None:
        return None
    level = getattr(trust_cap, "level", None)
    if level is None:
        return None
    try:
        return float(level)
    except (TypeError, ValueError):
        return None


def audit_composition(
    participants,
    capability_managers: dict,
    *,
    threshold: float = DEFAULT_LOW_AGREEABLENESS_THRESHOLD,
    majority_fraction: float = DEFAULT_MAJORITY_FRACTION,
    min_group_size: int = _MIN_GROUP_SIZE,
) -> CompositionAudit:
    """Compute the agreeableness composition of a group of interacting NPCs.

    ``participants`` is any iterable of NPC ids (a gossip cluster, a
    Story Director multi-NPC plan, etc.). ``capability_managers`` is the
    engine's ``pie.capability_managers`` dict (or any mapping exposing
    ``.capabilities['trust'].level``).

    The audit is **flagged** iff:
      * the group has at least ``min_group_size`` participants, AND
      * at least one participant has a readable disposition, AND
      * the fraction of low-agreeableness participants (below
        ``threshold``) STRICTLY exceeds ``majority_fraction``.

    Participants with no readable disposition are excluded from the
    low-fraction denominator (we don't assume cooperativeness for an
    unknown), but still count toward ``n_participants`` for diagnostics.
    A group with no readable signals is never flagged.
    """
    ids = [p for p in participants if p]
    # De-duplicate while preserving order — a cluster dict or repeated
    # plan slot must not double-count the same NPC.
    seen: set = set()
    unique_ids: list[str] = []
    for npc_id in ids:
        if npc_id not in seen:
            seen.add(npc_id)
            unique_ids.append(npc_id)

    n_participants = len(unique_ids)
    levels: list[tuple[str, float]] = []
    for npc_id in unique_ids:
        lvl = _agreeableness_for(npc_id, capability_managers)
        if lvl is not None:
            levels.append((npc_id, lvl))

    n_with_signal = len(levels)
    low = [(npc_id, lvl) for npc_id, lvl in levels if lvl < threshold]
    n_low = len(low)
    low_fraction = (n_low / n_with_signal) if n_with_signal else 0.0
    mean_agreeableness = (
        sum(lvl for _, lvl in levels) / n_with_signal / _TRUST_SCALE_MAX
        if n_with_signal else None
    )

    flagged = (
        n_participants >= min_group_size
        and n_with_signal > 0
        and low_fraction > majority_fraction
    )

    return CompositionAudit(
        n_participants=n_participants,
        n_with_signal=n_with_signal,
        n_low=n_low,
        low_fraction=low_fraction,
        mean_agreeableness=mean_agreeableness,
        threshold=threshold,
        majority_fraction=majority_fraction,
        flagged=flagged,
        participants=unique_ids,
        low_participants=[npc_id for npc_id, _ in low],
    )
