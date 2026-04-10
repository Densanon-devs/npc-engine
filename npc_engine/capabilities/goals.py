"""
Goals Capability — NPC personal goals that influence conversation.

Each NPC has 1-3 personal goals with priorities. The top goal is
injected as a behavioral directive, steering the NPC to bring up
relevant topics when natural. Goals can also be used by other
capabilities (dynamic_quests uses goals to select quest templates).

Goals are static by default (defined in YAML) but can be updated
dynamically via shared_state if a game engine pushes new goals.
"""

import logging
import re

from npc_engine.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityRegistry,
    CapabilityUpdate,
)

logger = logging.getLogger(__name__)


@CapabilityRegistry.register
class GoalsCapability(Capability):
    """NPC personal goals that influence conversation direction."""

    name = "goals"
    version = "1.0"
    dependencies = []
    default_token_budget = 40

    def initialize(self, npc_id: str, yaml_config: dict, shared_state: dict) -> None:
        self.npc_id = npc_id
        self.goals: list[dict] = []

        for g in yaml_config.get("active_goals", []):
            self.goals.append({
                "id": g.get("id", ""),
                "description": g.get("description", ""),
                "priority": g.get("priority", 5),
                "keywords": g.get("keywords", []),
                "progress": g.get("progress", 0),  # 0-100
                "mentions": 0,
            })

        # Sort by priority descending
        self.goals.sort(key=lambda g: g["priority"], reverse=True)

        shared_state["goals"] = self._state_snapshot()

    def build_context(self, query: str, shared_state: dict) -> CapabilityContext:
        if not self.goals:
            return CapabilityContext("", 0, 70, "directive")

        # Inject top 1-2 goals as behavioral directives
        parts = []
        for goal in self.goals[:2]:
            desc = goal["description"]
            if goal["progress"] > 0:
                parts.append(f"{desc} (progress: {goal['progress']}%)")
            else:
                parts.append(desc)

        if len(parts) == 1:
            fragment = f"[Your priority: {parts[0]}. When relevant, steer conversation toward this.]"
        else:
            fragment = f"[Your priorities: 1) {parts[0]} 2) {parts[1]}. Weave these into conversation when natural.]"

        token_est = len(fragment) // 4 + 1
        return CapabilityContext(
            context_fragment=fragment,
            token_estimate=min(token_est, self.default_token_budget),
            priority=70,
            section="directive",
        )

    def process_response(self, response: str, query: str,
                         shared_state: dict) -> CapabilityUpdate:
        query_lower = query.lower()
        events = []

        for goal in self.goals:
            # Check if the player's query touches on goal topics
            keywords = goal.get("keywords", [])
            if not keywords:
                # Auto-derive keywords from description
                keywords = [w.lower() for w in goal["description"].split()
                            if len(w) > 3 and w.lower() not in
                            {"when", "that", "this", "from", "with", "about", "their", "them", "they"}]

            mentioned = any(kw in query_lower for kw in keywords)
            if mentioned:
                goal["mentions"] += 1
                if goal["mentions"] % 3 == 0:
                    events.append(f"goal_discussed:{goal['id']}:{goal['mentions']}")

        snapshot = self._state_snapshot()
        shared_state["goals"] = snapshot
        return CapabilityUpdate(state_patch=snapshot, events=events)

    def _state_snapshot(self) -> dict:
        active = self.goals[0]["id"] if self.goals else None
        return {
            "active": active,
            "goals": [
                {"id": g["id"], "priority": g["priority"],
                 "progress": g["progress"], "mentions": g["mentions"]}
                for g in self.goals
            ],
        }

    def get_state(self) -> dict:
        return {
            "goals": [
                {"id": g["id"], "priority": g["priority"],
                 "progress": g["progress"], "mentions": g["mentions"]}
                for g in self.goals
            ],
        }

    def load_state(self, state: dict) -> None:
        saved_goals = {g["id"]: g for g in state.get("goals", [])}
        for goal in self.goals:
            if goal["id"] in saved_goals:
                saved = saved_goals[goal["id"]]
                goal["progress"] = saved.get("progress", goal["progress"])
                goal["mentions"] = saved.get("mentions", 0)
