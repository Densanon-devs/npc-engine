"""
Gossip Propagation — Spreads information through the NPC social network.

After each player-NPC conversation, the propagator:
1. Extracts gossip-worthy facts from the conversation
2. Classifies each fact (personal, trade, military, lore, quest)
3. Walks the social graph from the source NPC
4. At each hop, checks gossip_filter and injects matching facts as events
5. Applies decay and significance thresholds

Passive latent-alignment observer (off by default, env-gated):
Set NPC_ENGINE_LATENT_ALIGN_OBSERVE=1 to log per-injection latent
similarity scores against the last N facts already delivered to the
same NPC. Uses densanon.core.latent_align (frozen MiniLM + cosine).
Logging only — never changes propagation behavior. Output goes to
<world>/story/latent_align_observations.jsonl (or the equivalent
runtime dir if story_director already chose one).

Adapts SCALE-COMM (arXiv 2605.27532, May 2026) — cross-agent semantic
consistency for cheap via a shared frozen encoder. See
project_npc_engine.md 2026-05-28 for the design rationale.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from npc_engine.bridge import NPC_ENGINE_ROOT
from npc_engine.config import GossipRules
from npc_engine.social.network import SocialGraph

logger = logging.getLogger(__name__)

# Latent-align observer config
_LATENT_OBSERVE_ENV = "NPC_ENGINE_LATENT_ALIGN_OBSERVE"
# Active dedup config — orthogonal to the observer. When this env var is
# set to "1", the propagator SUPPRESSES inject_event calls for facts
# that are semantically duplicate of the target NPC's recent history.
# Behavior change: gossip duplicates are dropped, not just logged.
_LATENT_DEDUP_ENV = "NPC_ENGINE_LATENT_ALIGN_DEDUP"
# Per-NPC ring buffer cap. 30 recent gossip facts is enough to catch
# drift cascades over a typical play session without unbounded memory.
_LATENT_OBSERVE_BUFFER_PER_NPC = 30
_LATENT_OBSERVE_LOG_DEFAULT = (
    NPC_ENGINE_ROOT / "data" / "story_director" / "latent_align_observations.jsonl"
)


@dataclass
class GossipFact:
    """A piece of information that can spread through the social network."""
    text: str
    category: str        # personal, trade, military, lore, quest
    significance: float  # 0.0-1.0
    source_npc: str
    source_turn: int = 0


# ── Fact classification keywords ─────────────────────────────

_CATEGORY_KEYWORDS = {
    "personal": ["name", "family", "wife", "husband", "child", "father", "mother",
                  "brother", "sister", "friend", "home", "born", "grew up", "lost"],
    "trade": ["gold", "coin", "price", "buy", "sell", "merchant", "goods", "trade",
              "shipment", "cargo", "market", "shop", "steal", "thief"],
    "military": ["guard", "patrol", "soldier", "attack", "defend", "weapon", "sword",
                 "creature", "bandit", "threat", "danger", "war", "army", "fight"],
    "lore": ["ancient", "history", "legend", "forest", "magic", "curse", "blessing",
             "well", "ruins", "cave", "stone", "glow", "light"],
    "quest": ["quest", "task", "mission", "help", "investigate", "find", "retrieve",
              "completed", "finished", "reward", "objective"],
}


def classify_fact(text: str) -> str:
    """Classify a gossip fact into a category using keyword matching."""
    text_lower = text.lower()
    scores: dict[str, int] = {}

    for category, keywords in _CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[category] = score

    if not scores:
        return "personal"  # default

    return max(scores, key=scores.get)


# ── Notable patterns for gossip extraction ───────────────────

_GOSSIP_PATTERNS = [
    (r"\b(?:my name is|i'm|i am)\s+(\w+)", "personal", 0.6),
    (r"\b(?:i come from|i'm from|i hail from)\s+(.+?)(?:\.|$|,)", "personal", 0.5),
    (r"\b(?:i found|i discovered|i saw)\s+(.+?)(?:\.|$|,)", "lore", 0.7),
    (r"\b(?:i killed|i defeated|i fought)\s+(.+?)(?:\.|$|,)", "military", 0.7),
    (r"\b(?:i bought|i sold|i traded)\s+(.+?)(?:\.|$|,)", "trade", 0.5),
]


class GossipPropagator:
    """
    Propagates gossip through the social network after each conversation.
    """

    def __init__(self, social_graph: SocialGraph, rules: GossipRules = None):
        self.graph = social_graph
        self.rules = rules or GossipRules()
        self._pending: list[tuple[str, GossipFact, int]] = []  # (target_npc, fact, deliver_at_turn)
        self._turn: int = 0

        # Latent-align observer state (env-gated). Per-NPC ring buffer of
        # recently-delivered gossip texts so a new injection can be
        # scored for semantic duplication against the NPC's actual
        # received history. Behavior-neutral — only logging.
        self._latent_observe_enabled: bool = (
            os.environ.get(_LATENT_OBSERVE_ENV) == "1"
        )
        self._latent_dedup_enabled: bool = (
            os.environ.get(_LATENT_DEDUP_ENV) == "1"
        )
        # The ring buffer is maintained whenever EITHER observer or dedup
        # is enabled — both paths read from the same per-NPC history.
        self._latent_recent_per_npc: dict[str, list[str]] = {}
        self._latent_log_path: Optional[Path] = (
            _LATENT_OBSERVE_LOG_DEFAULT if self._latent_observe_enabled else None
        )
        # Telemetry: how many inject_event calls did dedup suppress?
        # Reset on each propagate() call; cumulative counter on the instance.
        self._latent_dedup_suppressed_count: int = 0
        if self._latent_observe_enabled:
            logger.info(
                f"latent_align observer enabled. Logging to {self._latent_log_path}"
            )
        if self._latent_dedup_enabled:
            logger.info(
                "latent_align ACTIVE DEDUP enabled — duplicate gossip "
                "facts will be suppressed before inject_event"
            )

    def propagate(self, source_npc: str, player_input: str, npc_response: str,
                  knowledge_manager, capability_managers: dict) -> list[str]:
        """
        Extract gossip from a conversation and propagate through the network.
        Returns list of NPCs that received gossip.
        """
        self._turn += 1

        # Deliver any pending gossip from previous turns
        delivered = self._deliver_pending(knowledge_manager, capability_managers)

        # Extract gossip-worthy facts
        facts = self._extract_facts(source_npc, player_input, npc_response,
                                     capability_managers)

        if not facts:
            return delivered

        # Propagate each fact through the social graph
        for fact in facts:
            targets = self._walk_graph(source_npc, fact)
            for target_npc, adjusted_fact in targets:
                if self.rules.propagation_delay > 0:
                    deliver_turn = self._turn + self.rules.propagation_delay
                    self._pending.append((target_npc, adjusted_fact, deliver_turn))
                else:
                    self._inject_gossip(target_npc, adjusted_fact,
                                        knowledge_manager, capability_managers)
                    delivered.append(target_npc)

        return delivered

    def _extract_facts(self, source_npc: str, player_input: str, npc_response: str,
                       capability_managers: dict) -> list[GossipFact]:
        """Extract gossip-worthy facts from the conversation."""
        facts = []
        input_lower = player_input.lower()

        # Pattern-based extraction from player input
        for pattern, category, significance in _GOSSIP_PATTERNS:
            matches = re.findall(pattern, input_lower, re.IGNORECASE)
            for match in matches:
                match_text = match.strip() if isinstance(match, str) else match[0].strip()
                if len(match_text) > 2:
                    facts.append(GossipFact(
                        text=f"A stranger told {source_npc}: {match_text}",
                        category=category,
                        significance=significance,
                        source_npc=source_npc,
                        source_turn=self._turn,
                    ))

        # Check for trust threshold crossings (high-significance gossip)
        mgr = capability_managers.get(source_npc)
        if mgr:
            trust_state = mgr.shared_state.get("trust", {})
            if trust_state.get("trend") == "rising" and trust_state.get("level", 0) >= 50:
                facts.append(GossipFact(
                    text=f"{source_npc} speaks well of the stranger",
                    category="personal",
                    significance=0.7,
                    source_npc=source_npc,
                    source_turn=self._turn,
                ))

        # Check for quest events
        if any(kw in input_lower for kw in ["finished", "completed", "done", "succeeded"]):
            facts.append(GossipFact(
                text=f"The stranger completed a task for {source_npc}",
                category="quest",
                significance=0.8,
                source_npc=source_npc,
                source_turn=self._turn,
            ))

        return facts

    def _walk_graph(self, source_npc: str,
                    fact: GossipFact) -> list[tuple[str, GossipFact]]:
        """Walk the social graph and find valid targets for this fact."""
        targets = []
        reachable = self.graph.get_reachable(source_npc, self.rules.max_hops)

        for target_npc, hops in reachable.items():
            # Check gossip filter on the path
            if not self._passes_filter(source_npc, target_npc, fact.category):
                continue

            # Apply decay
            decayed_significance = fact.significance * (self.rules.decay_per_hop ** hops)
            if decayed_significance < self.rules.min_significance:
                continue

            adjusted = GossipFact(
                text=fact.text,
                category=fact.category,
                significance=decayed_significance,
                source_npc=fact.source_npc,
                source_turn=fact.source_turn,
            )
            targets.append((target_npc, adjusted))

        return targets

    def _passes_filter(self, from_npc: str, to_npc: str, category: str) -> bool:
        """Check if a gossip category passes the connection's filter.
        For direct connections, checks the edge filter.
        For multi-hop, checks that at least one path from source to target
        has all edges allowing the category."""
        gossip_filter = self.graph.get_gossip_filter(from_npc, to_npc)

        # Direct connection exists
        if gossip_filter is not None:
            return gossip_filter in ("all", category)

        # No direct connection — check 1-hop intermediaries
        for conn in self.graph.get_connections(from_npc):
            if conn.gossip_filter not in ("all", category):
                continue  # first hop blocks this category
            # Check if intermediary connects to target
            hop2_filter = self.graph.get_gossip_filter(conn.to_id, to_npc)
            if hop2_filter is not None and hop2_filter in ("all", category):
                return True  # valid 2-hop path found

        return False

    def _inject_gossip(self, target_npc: str, fact: GossipFact,
                       knowledge_manager, capability_managers: dict) -> None:
        """Inject a gossip fact into a target NPC's knowledge.

        Pipeline:
          1. (observer-on) score against per-NPC history + log to JSONL
          2. (dedup-on)    score against per-NPC history; if duplicate,
                           SUPPRESS the inject and skip the rest. Update
                           the buffer with the suppressed fact so the next
                           one is also evaluated against it.
          3. inject_event  + capability update
          4. buffer append (skipped if dedup suppressed the inject)
        """
        # Step 1 — passive observer (logging only, behavior-neutral)
        if self._latent_observe_enabled:
            try:
                self._latent_observe(target_npc, fact)
            except Exception as e:
                logger.warning(f"latent_align observer failed (non-fatal): {e}")

        # Step 2 — active dedup (behavior change; opt-in)
        if self._latent_dedup_enabled:
            try:
                if self._latent_should_dedup(target_npc, fact):
                    self._latent_dedup_suppressed_count += 1
                    self._latent_append_to_history(target_npc, fact.text)
                    logger.info(
                        f"latent_align dedup: SUPPRESSED gossip "
                        f"{fact.source_npc} → {target_npc}: "
                        f"{fact.text[:60]}... (semantic duplicate)"
                    )
                    return
            except Exception as e:
                logger.warning(f"latent_align dedup failed (non-fatal): {e}")

        # Step 3 — actual inject
        knowledge_manager.inject_event(target_npc, fact.text)

        # Update gossip capability state if available
        mgr = capability_managers.get(target_npc)
        if mgr and "gossip" in mgr.capabilities:
            gossip_cap = mgr.capabilities["gossip"]
            if hasattr(gossip_cap, "add_rumor"):
                gossip_cap.add_rumor(fact)

        # Step 4 — append to buffer so future facts can be compared
        # against this one. Centralized here (NOT in _latent_observe)
        # so that when both observer + dedup are enabled, dedup sees
        # history WITHOUT the current fact, avoiding self-match.
        if self._latent_observe_enabled or self._latent_dedup_enabled:
            self._latent_append_to_history(target_npc, fact.text)

        logger.debug(f"Gossip: {fact.source_npc} → {target_npc}: {fact.text[:60]}...")

    def _latent_should_dedup(self, target_npc: str, fact: GossipFact) -> bool:
        """Return True iff the new fact is a semantic duplicate of any
        entry in the target NPC's recent history. Uses
        densanon.core.latent_align with its default threshold.

        Returns False (don't dedup) when the latent_align backend is
        unavailable — fail-open, preserve the original gossip behavior.
        """
        try:
            from densanon.core.latent_align import is_duplicate
        except ImportError:
            return False
        history = self._latent_recent_per_npc.get(target_npc, [])
        if not history:
            return False
        for prev_text in history:
            if is_duplicate(prev_text, fact.text):
                return True
        return False

    def _latent_append_to_history(self, target_npc: str, text: str) -> None:
        """Append text to the per-NPC ring buffer, respecting the cap."""
        history = self._latent_recent_per_npc.setdefault(target_npc, [])
        history.append(text)
        if len(history) > _LATENT_OBSERVE_BUFFER_PER_NPC:
            history.pop(0)

    def _latent_observe(self, target_npc: str, fact: GossipFact) -> None:
        """Score the new fact against the target NPC's recent gossip
        history using densanon.core.latent_align cosine similarity.
        Log + maintain the per-NPC ring buffer."""
        try:
            from densanon.core.latent_align import (
                DEFAULT_DUPLICATE_THRESHOLD, is_duplicate, similarity,
            )
        except ImportError:
            return

        history = self._latent_recent_per_npc.setdefault(target_npc, [])

        # Score new fact vs each existing history entry
        scores = []
        for prev_text in history:
            s = similarity(prev_text, fact.text)
            scores.append({"prev_text": prev_text[:120], "similarity": round(s, 3)})
        if scores:
            scores.sort(key=lambda x: -x["similarity"])
            best = scores[0]
            best_sim = best["similarity"]
            is_dup = best_sim >= DEFAULT_DUPLICATE_THRESHOLD
        else:
            best, best_sim, is_dup = None, 0.0, False

        record = {
            "turn": self._turn,
            "source_npc": fact.source_npc,
            "target_npc": target_npc,
            "category": fact.category,
            "significance": fact.significance,
            "new_fact": fact.text[:240],
            "history_size_before": len(history),
            "best_similarity": best_sim,
            "best_match_text": (best or {}).get("prev_text"),
            "is_duplicate_at_default": is_dup,
            "top_3_history_scores": scores[:3],
        }
        if self._latent_log_path is not None:
            try:
                self._latent_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._latent_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                logger.warning(f"latent_align log write failed: {e}")

        # NOTE: buffer maintenance happens in _inject_gossip's final step,
        # NOT here. Centralizing it there keeps the ordering correct when
        # both observer and dedup are enabled (dedup must see history WITHOUT
        # the current fact, else it self-matches).

    def _deliver_pending(self, knowledge_manager,
                         capability_managers: dict) -> list[str]:
        """Deliver gossip that was delayed by propagation_delay."""
        delivered = []
        remaining = []

        for target_npc, fact, deliver_turn in self._pending:
            if self._turn >= deliver_turn:
                self._inject_gossip(target_npc, fact,
                                    knowledge_manager, capability_managers)
                delivered.append(target_npc)
            else:
                remaining.append((target_npc, fact, deliver_turn))

        self._pending = remaining
        return delivered

    @property
    def pending_count(self) -> int:
        return len(self._pending)
