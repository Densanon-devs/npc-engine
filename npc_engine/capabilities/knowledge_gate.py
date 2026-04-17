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

        # Topic gate (museum mode) — when enabled, the NPC only
        # discusses topics in the allowed list. Everything else gets
        # a polite redirect. The gate injects a system directive
        # into the prompt AND provides a redirect message the
        # postgen layer can use to replace off-topic model output.
        topic_cfg = yaml_config.get("topic_gate", {}) or {}
        self.topic_gate_enabled: bool = bool(topic_cfg.get("enabled", False))
        self.allowed_topics: list[str] = [
            str(t).lower().strip()
            for t in (topic_cfg.get("allowed_topics") or [])
            if isinstance(t, str) and t.strip()
        ]
        self.redirect_message: str = str(
            topic_cfg.get("redirect_message", "")
        ).strip()
        # Optional cross-references for the redirect template.
        self.suggested_curator: str = str(
            topic_cfg.get("suggested_curator", "another curator")
        ).strip()
        self.suggested_floor: str = str(
            topic_cfg.get("suggested_floor", "another floor")
        ).strip()

        shared_state["knowledge_gate"] = {"unlocked": list(self.unlocked_ids)}

    def build_context(self, query: str, shared_state: dict) -> CapabilityContext:
        parts: list[str] = []

        # Topic gate directive — highest-priority instruction that
        # tells the model to stay on-topic. Placed BEFORE any
        # unlocked secrets so the fence is the first thing the
        # model reads in the knowledge section.
        if self.topic_gate_enabled and self.allowed_topics:
            redirect = self._format_redirect()
            topics_str = ", ".join(self.allowed_topics[:20])
            parts.append(
                f"[IMPORTANT — TOPIC GATE: You may ONLY discuss "
                f"the following topics: {topics_str}. "
                f"If the visitor asks about ANYTHING outside these "
                f"topics, respond EXACTLY with: \"{redirect}\" "
                f"Do NOT answer off-topic questions. Do NOT "
                f"improvise. Stay in character and redirect.]"
            )

            # Quick on-topic check — if the query has zero overlap
            # with allowed topics, flag it in shared_state so the
            # postgen layer can enforce the redirect even if the
            # model ignores the directive.
            query_lower = query.lower() if query else ""
            on_topic = any(
                topic in query_lower for topic in self.allowed_topics
            )
            # Also count general museum/conversation openers as on-topic
            if any(w in query_lower for w in (
                "hello", "hi", "hey", "tell me", "what", "how",
                "who are you", "your name", "this floor", "this gallery",
                "thank", "goodbye", "bye",
            )):
                on_topic = True
            shared_state.setdefault("knowledge_gate", {})["query_on_topic"] = on_topic
            if not on_topic:
                shared_state["knowledge_gate"]["redirect_message"] = redirect

        # Unlocked gated facts (original knowledge_gate behavior).
        unlocked_facts = self._get_unlocked_facts(shared_state)
        if unlocked_facts:
            secrets = "; ".join(f["fact"] for f in unlocked_facts)
            parts.append(f"[Secrets you can share: {secrets}]")

        if not parts:
            return CapabilityContext("", 0, 50, "knowledge")

        fragment = "\n".join(parts)
        token_est = len(fragment) // 4 + 1
        return CapabilityContext(
            context_fragment=fragment,
            token_estimate=min(token_est, max(self.default_token_budget, 80)),
            priority=50,
            section="knowledge",
        )

    def _format_redirect(self) -> str:
        """Build the redirect message with curator/floor substitution."""
        msg = self.redirect_message or (
            "That is a wonderful question, but it falls outside my "
            "gallery. Is there anything else about my topic I can "
            "help you with?"
        )
        msg = msg.replace("{suggested_curator}", self.suggested_curator)
        msg = msg.replace("{suggested_floor}", self.suggested_floor)
        return msg

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
        """React to trust threshold crossings by re-checking gates immediately."""
        if event.startswith("trust_crossed:"):
            # Trust changed — check if any new facts are now unlockable
            for fact in self.gated_facts:
                if fact["id"] not in self.unlocked_ids:
                    if self._check_requirements(fact["requires"], shared_state):
                        self.unlocked_ids.add(fact["id"])
                        logger.info(f"Knowledge gate unlocked via trust event: {fact['id']}")

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
