"""
Gossip Propagation — Spreads information through the NPC social network.

After each player-NPC conversation, the propagator:
1. Extracts gossip-worthy facts from the conversation
2. Classifies each fact (personal, trade, military, lore, quest)
3. Walks the social graph from the source NPC
4. At each hop, checks gossip_filter and injects matching facts as events
5. Applies decay and significance thresholds
"""

import json
import logging
import re
from dataclasses import dataclass, field

from npc_engine.config import GossipRules
from npc_engine.social.network import SocialGraph

logger = logging.getLogger(__name__)


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
        """Check if a gossip category passes the connection's filter."""
        gossip_filter = self.graph.get_gossip_filter(from_npc, to_npc)

        # Try direct connection first
        if gossip_filter == "none":
            # Check if there's any indirect path with a valid filter
            for conn in self.graph.get_connections(from_npc):
                if conn.to_id == to_npc:
                    return False
                # For multi-hop, check the first hop's filter
                if conn.gossip_filter in ("all", category):
                    return True
            return False

        return gossip_filter in ("all", category)

    def _inject_gossip(self, target_npc: str, fact: GossipFact,
                       knowledge_manager, capability_managers: dict) -> None:
        """Inject a gossip fact into a target NPC's knowledge."""
        # Inject as event into NPC knowledge
        knowledge_manager.inject_event(target_npc, fact.text)

        # Update gossip capability state if available
        mgr = capability_managers.get(target_npc)
        if mgr and "gossip" in mgr.capabilities:
            gossip_cap = mgr.capabilities["gossip"]
            if hasattr(gossip_cap, "add_rumor"):
                gossip_cap.add_rumor(fact)

        logger.debug(f"Gossip: {fact.source_npc} → {target_npc}: {fact.text[:60]}...")

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
