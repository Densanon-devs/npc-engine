"""
Reputation Ripple — Trust changes for one NPC influence connected NPCs.

When a player builds trust with Noah, Kael (connected to Noah) gets a
proportional trust nudge based on closeness and the ripple factor.
"""

import logging

from npc_engine.config import TrustRippleConfig
from npc_engine.social.network import SocialGraph

logger = logging.getLogger(__name__)


class ReputationRipple:
    """Propagates trust changes through the social network."""

    def __init__(self, social_graph: SocialGraph, config: TrustRippleConfig = None):
        self.graph = social_graph
        self.config = config or TrustRippleConfig()
        self._last_trust: dict[str, int] = {}  # npc_id -> last known trust level

    def process(self, capability_managers: dict) -> list[str]:
        """
        Check all NPC trust levels for changes since last call.
        Ripple any significant changes to connected NPCs.
        Returns list of NPCs that had their trust adjusted.
        """
        if not self.config.enabled:
            return []

        adjusted = []

        for npc_id, mgr in capability_managers.items():
            trust_state = mgr.shared_state.get("trust", {})
            current_level = trust_state.get("level", 0)
            previous_level = self._last_trust.get(npc_id, current_level)

            delta = current_level - previous_level
            self._last_trust[npc_id] = current_level

            # Only ripple significant changes
            if abs(delta) < 3:
                continue

            factor = self.config.positive_factor if delta > 0 else self.config.negative_factor
            ripple_amount = min(abs(delta) * factor, self.config.max_ripple)
            ripple_sign = 1 if delta > 0 else -1

            # Apply to connected NPCs
            for conn in self.graph.get_connections(npc_id):
                target_mgr = capability_managers.get(conn.to_id)
                if not target_mgr:
                    continue

                trust_cap = target_mgr.capabilities.get("trust")
                if not trust_cap:
                    continue

                adjustment = int(ripple_amount * conn.closeness * ripple_sign)
                if adjustment == 0:
                    continue

                old_level = trust_cap.level
                trust_cap.level = max(0, min(100, trust_cap.level + adjustment))
                target_mgr.shared_state.setdefault("trust", {})["level"] = trust_cap.level

                logger.debug(
                    f"Trust ripple: {npc_id} ({delta:+d}) → {conn.to_id} ({adjustment:+d}) "
                    f"[{old_level} → {trust_cap.level}]"
                )
                adjusted.append(conn.to_id)

        return adjusted
