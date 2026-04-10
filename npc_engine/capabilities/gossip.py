"""
Gossip Capability — NPC knows what they've heard through the social network.

Injects "you heard from X..." context into the NPC's prompt so they
can reference gossip in conversation. Rumors are injected by the
GossipPropagator (not by this capability — this is read-only context).
"""

import logging
import time
from dataclasses import dataclass

from npc_engine.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityRegistry,
    CapabilityUpdate,
)

logger = logging.getLogger(__name__)


@dataclass
class Rumor:
    text: str
    source_npc: str
    significance: float
    turn_received: int
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@CapabilityRegistry.register
class GossipCapability(Capability):
    """NPC context about rumors heard through the social network."""

    name = "gossip"
    version = "1.0"
    dependencies = []
    default_token_budget = 40

    def initialize(self, npc_id: str, yaml_config: dict, shared_state: dict) -> None:
        self.npc_id = npc_id
        self.max_rumors: int = yaml_config.get("max_rumors", 3)
        self.interests: list[str] = yaml_config.get("interests", ["all"])
        self.rumors: list[Rumor] = []

    def build_context(self, query: str, shared_state: dict) -> CapabilityContext:
        if not self.rumors:
            return CapabilityContext("", 0, 45, "knowledge")

        # Select top rumors by significance, limited to max_rumors
        top = sorted(self.rumors, key=lambda r: r.significance, reverse=True)[:self.max_rumors]

        parts = []
        for rumor in top:
            parts.append(f"{rumor.source_npc} said: {rumor.text}")

        fragment = f"[You heard: {'; '.join(parts)}]"
        token_est = len(fragment) // 4 + 1

        return CapabilityContext(
            context_fragment=fragment,
            token_estimate=min(token_est, self.default_token_budget),
            priority=45,
            section="knowledge",
        )

    def process_response(self, response: str, query: str,
                         shared_state: dict) -> CapabilityUpdate:
        # Gossip capability is passive — rumors are injected externally
        # by GossipPropagator. Nothing to do here.
        return CapabilityUpdate(
            state_patch={"rumor_count": len(self.rumors)},
            events=[],
        )

    def add_rumor(self, fact) -> None:
        """Add a rumor from the gossip propagator."""
        rumor = Rumor(
            text=fact.text,
            source_npc=fact.source_npc,
            significance=fact.significance,
            turn_received=fact.source_turn,
        )
        self.rumors.append(rumor)

        # Prune old/low-significance rumors
        if len(self.rumors) > self.max_rumors * 2:
            self.rumors.sort(key=lambda r: r.significance, reverse=True)
            self.rumors = self.rumors[:self.max_rumors * 2]

    def get_state(self) -> dict:
        return {
            "rumors": [
                {"text": r.text, "source_npc": r.source_npc,
                 "significance": r.significance, "turn_received": r.turn_received}
                for r in self.rumors
            ],
        }

    def load_state(self, state: dict) -> None:
        self.rumors = []
        for r in state.get("rumors", []):
            self.rumors.append(Rumor(
                text=r["text"],
                source_npc=r["source_npc"],
                significance=r.get("significance", 0.5),
                turn_received=r.get("turn_received", 0),
            ))
