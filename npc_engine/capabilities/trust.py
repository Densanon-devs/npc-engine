"""
Trust Capability — NPC-player relationship tracking.

Inspired by YC-Bench finding: top-performing agents focused on 1-2 clients
to build trust, triggering a "trust snowball" — each successful interaction
reduced future workload and unlocked higher-tier opportunities.

Trust is adjusted PROGRAMMATICALLY using heuristics:
  - Quest completion: +10
  - Return visit: +2
  - Polite greeting: +1
  - Hostile language: -5
  - Quest failure: -5
  - Gradual decay toward neutral if no interaction

The model receives the trust tier and its behavioral effect as read-only context.
"""

import logging
import re
import time

from npc_engine.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityRegistry,
    CapabilityUpdate,
)

logger = logging.getLogger(__name__)

# ── Sentiment heuristics ─────────────────────────────────────

_POSITIVE_PATTERNS = [
    r"\b(?:thank|thanks|grateful|appreciate|pleased)\b",
    r"\b(?:hello|greetings|good (?:morning|day|evening)|well met)\b",
    r"\b(?:friend|ally|trust you|count on you)\b",
    r"\b(?:please|kindly|would you)\b",
    r"\b(?:help me|can you help|i need your)\b",
]

_NEGATIVE_PATTERNS = [
    r"\b(?:shut up|go away|leave me|get lost|fool|idiot)\b",
    r"\b(?:hate|despise|loathe|detest)\b",
    r"\b(?:liar|thief|cheat|traitor|scum)\b",
    r"\b(?:die|kill you|attack|fight|threaten)\b",
    r"\b(?:useless|worthless|pathetic|waste)\b",
]

_DEFAULT_THRESHOLDS = {
    "wary": 0,
    "neutral": 25,
    "friendly": 50,
    "trusted": 75,
}

_DEFAULT_EFFECTS = {
    "below_wary": "Speak evasively, withhold information, be suspicious",
    "wary": "Be cautious and guarded in conversation",
    "neutral": "Answer questions but do not volunteer secrets",
    "friendly": "Be warm and helpful, share useful information",
    "trusted": "Share secrets, offer better rewards, speak openly",
}


@CapabilityRegistry.register
class TrustCapability(Capability):
    """Tracks NPC-player trust level and adjusts NPC behavior accordingly."""

    name = "trust"
    version = "1.0"
    dependencies = []
    default_token_budget = 30

    def initialize(self, npc_id: str, yaml_config: dict, shared_state: dict) -> None:
        self.npc_id = npc_id
        self.level: int = yaml_config.get("initial_level", 30)
        self.interactions: int = 0
        self.trend: str = "stable"  # "rising", "stable", "falling"
        self.last_interaction: float = time.time()
        self._previous_level: int = self.level

        # Configuration
        self.thresholds = yaml_config.get("thresholds", _DEFAULT_THRESHOLDS)
        self.effects = {**_DEFAULT_EFFECTS, **yaml_config.get("effects", {})}

        # Publish initial state
        shared_state["trust"] = self._state_snapshot()

    def build_context(self, query: str, shared_state: dict) -> CapabilityContext:
        tier = self._get_tier()
        effect = self._get_effect(tier)

        fragment = f"[Trust: {tier} ({self.level}/100). {effect}]"
        token_est = len(fragment) // 4 + 1

        return CapabilityContext(
            context_fragment=fragment,
            token_estimate=min(token_est, self.default_token_budget),
            priority=90,
            section="state",
        )

    def process_response(self, response: str, query: str,
                         shared_state: dict) -> CapabilityUpdate:
        self._previous_level = self.level
        self.interactions += 1
        self.last_interaction = time.time()

        # Calculate trust delta from player query
        delta = self._calculate_delta(query, shared_state)
        self.level = max(0, min(100, self.level + delta))

        # Update trend
        if self.level > self._previous_level:
            self.trend = "rising"
        elif self.level < self._previous_level:
            self.trend = "falling"
        else:
            self.trend = "stable"

        # Check for threshold crossings
        events = []
        old_tier = self._tier_for_level(self._previous_level)
        new_tier = self._tier_for_level(self.level)
        if old_tier != new_tier:
            events.append(f"trust_crossed:{new_tier}")
            logger.info(f"NPC '{self.npc_id}': trust changed {old_tier} -> {new_tier} "
                        f"(level: {self._previous_level} -> {self.level})")

        # Publish to shared_state
        snapshot = self._state_snapshot()
        shared_state["trust"] = snapshot

        return CapabilityUpdate(
            state_patch=snapshot,
            events=events,
        )

    def _calculate_delta(self, query: str, shared_state: dict) -> int:
        """Calculate trust change from the player's query and game state."""
        delta = 0
        query_lower = query.lower()

        # Return visit bonus (player came back to talk)
        delta += 2

        # Positive sentiment
        for pattern in _POSITIVE_PATTERNS:
            if re.search(pattern, query_lower):
                delta += 1
                break  # Max +1 from sentiment per turn

        # Negative sentiment
        for pattern in _NEGATIVE_PATTERNS:
            if re.search(pattern, query_lower):
                delta -= 5
                break  # Max -5 from sentiment per turn

        # Quest-related trust changes (from shared_state or player_quests)
        player_quests = shared_state.get("player_quests", [])
        npc_name = self.npc_id.lower()
        for quest in player_quests:
            given_by = quest.get("given_by", "").lower()
            if given_by == npc_name:
                if quest.get("status") == "completed" and not quest.get("_trust_credited"):
                    delta += 10
                    quest["_trust_credited"] = True
                elif quest.get("status") == "failed" and not quest.get("_trust_debited"):
                    delta -= 5
                    quest["_trust_debited"] = True

        return delta

    def _get_tier(self) -> str:
        """Get the current trust tier name."""
        return self._tier_for_level(self.level)

    def _tier_for_level(self, level: int) -> str:
        """Determine tier for a given level."""
        tier = "wary"
        for tier_name, threshold in sorted(self.thresholds.items(),
                                           key=lambda x: x[1]):
            if level >= threshold:
                tier = tier_name
        return tier

    def _get_effect(self, tier: str) -> str:
        """Get the behavioral effect text for a trust tier."""
        # Check for specific tier effect
        if tier in self.effects:
            return self.effects[tier]
        # Check for below_X effects
        thresholds_sorted = sorted(self.thresholds.items(), key=lambda x: x[1])
        for tier_name, threshold in thresholds_sorted:
            if self.level < threshold:
                below_key = f"below_{tier_name}"
                if below_key in self.effects:
                    return self.effects[below_key]
        return self.effects.get("neutral", "")

    def _state_snapshot(self) -> dict:
        return {
            "level": self.level,
            "tier": self._get_tier(),
            "interactions": self.interactions,
            "trend": self.trend,
        }

    def on_event(self, event: str, shared_state: dict) -> None:
        """React to events from other capabilities."""
        # Example: world event could affect trust
        if event.startswith("world_threat:"):
            # NPC trusts player more if they're helping during a crisis
            self.level = min(100, self.level + 3)
            shared_state["trust"] = self._state_snapshot()

    def get_state(self) -> dict:
        return {
            "level": self.level,
            "interactions": self.interactions,
            "trend": self.trend,
            "last_interaction": self.last_interaction,
        }

    def load_state(self, state: dict) -> None:
        self.level = state.get("level", self.level)
        self.interactions = state.get("interactions", 0)
        self.trend = state.get("trend", "stable")
        self.last_interaction = state.get("last_interaction", time.time())
        self._previous_level = self.level
