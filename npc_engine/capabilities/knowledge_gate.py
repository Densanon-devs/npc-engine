"""
Knowledge Gate Capability — Conditional knowledge sharing.

NPCs only reveal certain facts when conditions are met:
  - Trust level threshold (e.g., trust >= 60)
  - Quest completion (e.g., player completed "bitter_well")
  - Combination of both

Gated facts are injected as additional context entries when unlocked,
making the NPC literally "learn" new things to say as the player
builds trust and progresses through quests.

Depends on: trust capability (reads shared_state["trust"]["level"])
"""

import logging

from npc_engine.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityRegistry,
    CapabilityUpdate,
)

logger = logging.getLogger(__name__)


@CapabilityRegistry.register
class KnowledgeGateCapability(Capability):
    """Gated knowledge that unlocks based on trust and quest progress."""

    name = "knowledge_gate"
    version = "1.0"
    dependencies = ["trust"]
    default_token_budget = 40

    def initialize(self, npc_id: str, yaml_config: dict, shared_state: dict) -> None:
        self.npc_id = npc_id
        self.gated_facts: list[dict] = []
        self.unlocked_ids: set[str] = set()

        for i, entry in enumerate(yaml_config.get("gated_facts", [])):
            fact_id = entry.get("id", f"gate_{i}")
            self.gated_facts.append({
                "id": fact_id,
                "fact": entry.get("fact", ""),
                "requires": entry.get("requires", {}),
                "reveal_style": entry.get("reveal_style", ""),
            })

        shared_state["knowledge_gate"] = {"unlocked": list(self.unlocked_ids)}

    def build_context(self, query: str, shared_state: dict) -> CapabilityContext:
        unlocked_facts = self._get_unlocked_facts(shared_state)

        if not unlocked_facts:
            return CapabilityContext("", 0, 50, "knowledge")

        # Format unlocked facts as secrets the NPC can now share
        secrets = "; ".join(f["fact"] for f in unlocked_facts)
        fragment = f"[Secrets you can share: {secrets}]"

        token_est = len(fragment) // 4 + 1
        return CapabilityContext(
            context_fragment=fragment,
            token_estimate=min(token_est, self.default_token_budget),
            priority=50,
            section="knowledge",
        )

    def process_response(self, response: str, query: str,
                         shared_state: dict) -> CapabilityUpdate:
        # Check for newly unlocked facts
        events = []
        newly_unlocked = []

        for fact in self.gated_facts:
            if fact["id"] in self.unlocked_ids:
                continue

            if self._check_requirements(fact["requires"], shared_state):
                self.unlocked_ids.add(fact["id"])
                newly_unlocked.append(fact["id"])
                logger.info(f"NPC '{self.npc_id}': unlocked gated fact '{fact['id']}'")

        if newly_unlocked:
            events.append(f"knowledge_unlocked:{','.join(newly_unlocked)}")

        shared_state["knowledge_gate"] = {"unlocked": list(self.unlocked_ids)}
        return CapabilityUpdate(
            state_patch={"unlocked": list(self.unlocked_ids)},
            events=events,
        )

    def on_event(self, event: str, shared_state: dict) -> None:
        """React to trust threshold crossings by re-checking gates."""
        if event.startswith("trust_crossed:"):
            # Trust changed — some facts may now be unlockable
            # The actual unlock happens in process_response next turn
            pass

    def _get_unlocked_facts(self, shared_state: dict) -> list[dict]:
        """Get all currently unlocked facts."""
        unlocked = []
        for fact in self.gated_facts:
            if fact["id"] in self.unlocked_ids:
                unlocked.append(fact)
            elif self._check_requirements(fact["requires"], shared_state):
                # Unlock it now (context-time unlock for immediate availability)
                self.unlocked_ids.add(fact["id"])
                unlocked.append(fact)
        return unlocked

    def _check_requirements(self, requires: dict, shared_state: dict) -> bool:
        """Check if all requirements for a gated fact are met."""
        if not requires:
            return True

        # Trust level requirement
        trust_req = requires.get("trust")
        if trust_req is not None:
            trust_state = shared_state.get("trust", {})
            current_trust = trust_state.get("level", 0)
            if current_trust < trust_req:
                return False

        # Quest completion requirement
        quest_req = requires.get("quest")
        if quest_req is not None:
            player_quests = shared_state.get("player_quests", [])
            completed = any(
                q.get("id") == quest_req and q.get("status") == "completed"
                for q in player_quests
            )
            if not completed:
                return False

        # Quest active requirement (player is working on it)
        quest_active_req = requires.get("quest_active")
        if quest_active_req is not None:
            player_quests = shared_state.get("player_quests", [])
            active = any(
                q.get("id") == quest_active_req and q.get("status") == "active"
                for q in player_quests
            )
            if not active:
                return False

        # Mood requirement (NPC must be in a certain mood)
        mood_req = requires.get("mood")
        if mood_req is not None:
            emo_state = shared_state.get("emotional_state", {})
            if emo_state.get("mood") != mood_req:
                return False

        return True

    def get_state(self) -> dict:
        return {"unlocked": list(self.unlocked_ids)}

    def load_state(self, state: dict) -> None:
        self.unlocked_ids = set(state.get("unlocked", []))
