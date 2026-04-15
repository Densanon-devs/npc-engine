"""
NPC Engine — The main orchestrator that wraps PIE with NPC intelligence.

Composition pattern:
  - Owns a PIE PluginIntelligenceEngine instance
  - Registers NPC capabilities into PIE's CapabilityRegistry
  - Injects custom few-shot examples into PIE's ExpertRouter
  - Wraps process() to add gossip propagation and trust ripple
  - PIE is NOT modified — all extension via its public API
"""

import logging
from pathlib import Path

from npc_engine.bridge import PluginIntelligenceEngine, NPC_ENGINE_ROOT
from npc_engine.config import NPCEngineConfig
from npc_engine.experts.examples import FewShotLoader
from npc_engine.experts.npc_experts import register_npc_experts
from npc_engine.postgen import validate_and_repair
from npc_engine.social.network import SocialGraph
from npc_engine.social.propagation import GossipPropagator
from npc_engine.social.reputation import ReputationRipple

logger = logging.getLogger("NPCEngine")


class NPCEngine:
    """
    NPC intelligence layer built on PIE.

    Usage:
        engine = NPCEngine("config.yaml")
        engine.initialize()
        response = engine.process("Hello, who are you?", npc_id="noah")
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = NPCEngineConfig.load(config_path)

        # Resolve PIE config path
        pie_config = self.config.pie_config
        if not pie_config:
            pie_config = str(NPC_ENGINE_ROOT.parent / "plug-in-intelligence-engine" / "config.yaml")

        # Create PIE engine instance
        self.pie = PluginIntelligenceEngine(config_path=pie_config, dry_run=False)

        # Override PIE's NPC paths with our world directory
        self.pie.config.npc.enabled = True
        self.pie.config.npc.profiles_dir = self.config.profiles_dir
        self.pie.config.npc.state_dir = self.config.state_dir
        self.pie.config.npc.world_name = self.config.world_name

        if self.config.active_npc:
            self.pie.config.npc.active_profile = self.config.active_npc

        # Post-generation validation & repair (on by default)
        self.postgen_enabled = True

        # Social systems (initialized after PIE loads)
        self.social_graph: SocialGraph | None = None
        self.gossip_propagator: GossipPropagator | None = None
        self.reputation_ripple: ReputationRipple | None = None

        # Few-shot loader
        self.few_shot_loader: FewShotLoader | None = None

        # Story Director (world-level narrative overseer, set up in initialize())
        self.story_director = None

    def initialize(self) -> None:
        """Initialize PIE engine, register capabilities, inject examples, load social graph."""
        # Register npc-engine's capabilities into PIE's registry
        # (importing the package auto-registers via decorators)
        import npc_engine.capabilities  # noqa: F401

        # Initialize PIE (loads model, router, modules, memory, etc.)
        self.pie.initialize()

        # Load custom few-shot examples and inject into PIE's expert system
        self.few_shot_loader = FewShotLoader(self.config.examples_dir)
        expert_count = register_npc_experts(
            self.pie.expert_router,
            self.few_shot_loader,
            self.pie.npc_knowledge.profiles,
        )
        logger.info(f"Registered/updated {expert_count} NPC experts with custom examples")

        # Load social graph
        self.social_graph = SocialGraph(self.config.world_yaml)
        if self.social_graph.connections:
            logger.info(f"Social graph: {len(self.social_graph.connections)} connections")

            self.gossip_propagator = GossipPropagator(
                self.social_graph, self.config.gossip,
            )
            self.reputation_ripple = ReputationRipple(
                self.social_graph, self.config.trust_ripple,
            )
        else:
            logger.info("No social graph configured — gossip disabled")

        # Story Director — world-level narrative overseer.
        # Must be last: needs base_model loaded (via pie.initialize()) and
        # NPC profiles available for world snapshots.
        from npc_engine.story_director import StoryDirector
        self.story_director = StoryDirector(self)
        logger.info(
            f"Story Director ready "
            f"(examples={len(self.story_director._examples)}, "
            f"lore={'yes' if self.story_director._lore_text else 'no'})"
        )

    def process(self, user_input: str, npc_id: str = None) -> str:
        """
        Process player input and return NPC response.

        Pipeline:
          1. Switch NPC if specified
          2. Delegate to PIE (routing, experts, capabilities, generation)
          3. Post-generation validation & repair (identity, hallucination, events, etc.)
          4. Gossip propagation + trust ripple

        Set self.postgen_enabled = False to skip step 3 (raw model output).
        """
        # Switch NPC if specified
        if npc_id:
            self.pie.config.npc.active_profile = npc_id
            self.pie.config.npc.enabled = True

        active_npc = self.pie.config.npc.active_profile

        # Delegate to PIE's full pipeline (handles routing, experts,
        # capabilities, quest injection, caching, memory — everything)
        response = self.pie.process(user_input)

        # Post-generation validation & repair — catches common model failures
        # (identity bleed, hallucination, echo, meta-gaming, OOD, etc.)
        if self.postgen_enabled and active_npc:
            try:
                npc = self.pie.npc_knowledge.get(active_npc)
                profile = None
                events = []
                if npc:
                    # Build a profile dict for the post-processor
                    profile = {
                        "identity": npc.identity,
                        "world_facts": npc.world_facts,
                        "personal_knowledge": npc.personal_knowledge,
                        "active_quests": [
                            {"id": q.id, "name": q.name, "description": q.description,
                             "status": q.status, "reward": q.reward,
                             "objectives": q.objectives}
                            for q in npc.quests
                        ],
                    }
                    events = [e.description for e in npc.events[-3:]]
                response = validate_and_repair(
                    response, npc_id=active_npc, profile=profile,
                    user_input=user_input, events=events,
                )
            except Exception as e:
                logger.debug(f"Postgen error (using raw response): {e}")

        # Post-generation: gossip propagation
        if active_npc and self.gossip_propagator:
            self.gossip_propagator.propagate(
                source_npc=active_npc,
                player_input=user_input,
                npc_response=response,
                knowledge_manager=self.pie.npc_knowledge,
                capability_managers=self.pie.capability_managers,
            )

        # Post-generation: trust ripple
        if self.reputation_ripple and self.pie.capability_managers:
            self.reputation_ripple.process(self.pie.capability_managers)

        # Dialogue auto-feed — let the Story Director observe player
        # dialogue turns so it reacts on the next tick. No-op if the
        # director isn't initialized or active_npc is unknown. We swallow
        # any errors here — the dialogue path must not fail because the
        # overseer can't record.
        if self.story_director is not None and active_npc and user_input:
            try:
                self.story_director.record_player_action(
                    text=f"Player said to {active_npc}: {user_input}",
                    target=active_npc,
                )
            except Exception as e:
                logger.debug(f"Story Director dialogue auto-feed failed: {e}")

        return response

    def switch_npc(self, npc_id: str) -> dict:
        """Switch the active NPC. Returns NPC info dict."""
        if npc_id not in self.pie.npc_knowledge.profiles:
            available = list(self.pie.npc_knowledge.profiles.keys())
            raise ValueError(f"NPC '{npc_id}' not found. Available: {available}")

        self.pie.config.npc.active_profile = npc_id
        self.pie.config.npc.enabled = True

        npc = self.pie.npc_knowledge.get(npc_id)
        return {
            "id": npc_id,
            "name": npc.identity.get("name", npc_id),
            "role": npc.identity.get("role", ""),
            "capabilities": list(npc.capability_configs.keys()),
        }

    def list_npcs(self) -> list[dict]:
        """List all available NPCs."""
        npcs = []
        for npc_id, npc in self.pie.npc_knowledge.profiles.items():
            npcs.append({
                "id": npc_id,
                "name": npc.identity.get("name", npc_id),
                "role": npc.identity.get("role", ""),
                "capabilities": list(npc.capability_configs.keys()),
            })
        return npcs

    def get_npc_state(self, npc_id: str) -> dict:
        """Get an NPC's current capability state.
        Lazy-creates the CapabilityManager if it doesn't exist yet,
        which triggers loading persisted state from disk."""
        mgr = self.pie.capability_managers.get(npc_id)
        if not mgr:
            # Try to create the manager (which loads saved state from disk)
            npc = self.pie.npc_knowledge.get(npc_id)
            if npc:
                mgr = self._ensure_capability_manager(npc_id)
            if not mgr:
                return {"npc_id": npc_id, "capabilities": {}, "shared_state": {}}

        cap_states = {}
        for name, cap in mgr.capabilities.items():
            cap_states[name] = cap.get_state()

        return {
            "npc_id": npc_id,
            "turn_count": mgr.turn_count,
            "capabilities": cap_states,
            "shared_state": mgr.shared_state,
        }

    def inject_event(self, description: str, npc_id: str = None) -> None:
        """Inject a world event. If npc_id is set, targets one NPC; otherwise all."""
        if npc_id:
            self.pie.npc_knowledge.inject_event(npc_id, description)
        else:
            self.pie.npc_knowledge.inject_global_event(description)

    # ── Game Engine State Mutation API ─────────────────────────

    def _ensure_capability_manager(self, npc_id: str):
        """Ensure a capability manager exists for this NPC (lazy-create if needed)."""
        if npc_id not in self.pie.capability_managers:
            npc = self.pie.npc_knowledge.get(npc_id)
            if npc and npc.capability_configs:
                self.pie._get_capability_manager(npc_id, npc)
        return self.pie.capability_managers.get(npc_id)

    def accept_quest(self, quest_id: str, quest_name: str, given_by: str) -> dict:
        """Player accepts a quest. Updates quest tracker and NPC quest status."""
        self.pie.player_quests.accept_quest(quest_id, quest_name, given_by)

        # Update NPC's quest status
        npc = self.pie.npc_knowledge.get(given_by)
        if npc:
            npc.update_quest(quest_id, "active")

        return {"accepted": quest_id, "given_by": given_by}

    def complete_quest(self, quest_id: str) -> dict:
        """Player completes a quest. Triggers trust boost and gossip."""
        # Find which NPC gave this quest
        given_by = None
        for q in self.pie.player_quests.active_quests:
            if q["id"] == quest_id:
                given_by = q.get("given_by", "")
                break

        success = self.pie.player_quests.complete_quest(quest_id)
        if not success:
            return {"error": f"Quest '{quest_id}' not found in active quests"}

        # Update NPC's quest status
        if given_by:
            npc = self.pie.npc_knowledge.get(given_by)
            if npc:
                npc.update_quest(quest_id, "completed")

            # Trust boost for completing a quest
            mgr = self._ensure_capability_manager(given_by)
            if mgr and "trust" in mgr.capabilities:
                trust_cap = mgr.capabilities["trust"]
                trust_cap.level = min(100, trust_cap.level + 10)
                mgr.shared_state.setdefault("trust", {})["level"] = trust_cap.level
                mgr.save_state()

            # Propagate quest completion gossip
            if self.gossip_propagator:
                from npc_engine.social.propagation import GossipFact
                fact = GossipFact(
                    text=f"The stranger completed a quest for {given_by}",
                    category="quest",
                    significance=0.8,
                    source_npc=given_by,
                )
                targets = self.gossip_propagator._walk_graph(given_by, fact)
                for target_npc, adjusted_fact in targets:
                    self.gossip_propagator._inject_gossip(
                        target_npc, adjusted_fact,
                        self.pie.npc_knowledge, self.pie.capability_managers,
                    )

            # Trust ripple
            if self.reputation_ripple:
                self.reputation_ripple.process(self.pie.capability_managers)

        return {"completed": quest_id, "given_by": given_by or "unknown"}

    def adjust_trust(self, npc_id: str, delta: int, reason: str = "") -> dict:
        """Directly adjust an NPC's trust level. For game-triggered events."""
        mgr = self._ensure_capability_manager(npc_id)
        if not mgr or "trust" not in mgr.capabilities:
            return {"error": f"NPC '{npc_id}' has no trust capability"}

        trust_cap = mgr.capabilities["trust"]
        old_level = trust_cap.level
        trust_cap.level = max(0, min(100, trust_cap.level + delta))
        mgr.shared_state.setdefault("trust", {})["level"] = trust_cap.level
        mgr.save_state()

        # Ripple if significant
        if abs(delta) >= 3 and self.reputation_ripple:
            self.reputation_ripple.process(self.pie.capability_managers)

        return {
            "npc_id": npc_id,
            "old_level": old_level,
            "new_level": trust_cap.level,
            "delta": delta,
            "reason": reason,
        }

    def add_scratchpad_entry(self, npc_id: str, text: str,
                             importance: float = 0.7) -> dict:
        """Add a fact to an NPC's scratchpad. For game-triggered memories."""
        mgr = self._ensure_capability_manager(npc_id)
        if not mgr or "scratchpad" not in mgr.capabilities:
            return {"error": f"NPC '{npc_id}' has no scratchpad capability"}

        scratchpad = mgr.capabilities["scratchpad"]
        from npc_engine.capabilities.scratchpad import ScratchpadEntry
        entry = ScratchpadEntry(
            text=text,
            turn=scratchpad._turn,
            importance=importance,
        )
        scratchpad._add_entry(entry)
        mgr.save_state()

        return {
            "npc_id": npc_id,
            "added": text,
            "total_entries": len(scratchpad.entries),
        }

    def set_mood(self, npc_id: str, mood: str, intensity: float = 0.5,
                 pin_turns: int = 3) -> dict:
        """Set an NPC's mood directly. For cutscenes, world events, etc.
        The mood is 'pinned' for pin_turns — it resists being overridden by
        the next model response's emotion extraction."""
        mgr = self._ensure_capability_manager(npc_id)
        if not mgr or "emotional_state" not in mgr.capabilities:
            return {"error": f"NPC '{npc_id}' has no emotional_state capability"}

        emo = mgr.capabilities["emotional_state"]
        old_mood = emo.mood
        emo.pin_mood(mood, intensity, turns=pin_turns)
        mgr.shared_state.setdefault("emotional_state", {}).update({
            "mood": mood,
            "intensity": emo.intensity,
        })
        mgr.save_state()

        return {
            "npc_id": npc_id,
            "old_mood": old_mood,
            "new_mood": mood,
            "intensity": emo.intensity,
        }

    def add_knowledge(self, npc_id: str, fact: str,
                      fact_type: str = "world") -> dict:
        """
        Add a fact to an NPC's knowledge at runtime.
        fact_type: "world" (dynamic_world_facts) or "personal"
        (dynamic_personal_knowledge).

        Runtime injections route into the dynamic lane on NPCKnowledge,
        not the static profile lists. ``build_context`` interleaves
        the two — the Director's newest injections reach the dialogue
        prompt without erasing identity-grounding profile lore. See
        ``NPCKnowledge._combine_static_and_dynamic`` for the rule.
        """
        npc = self.pie.npc_knowledge.get(npc_id)
        if not npc:
            return {"error": f"NPC '{npc_id}' not found"}

        if fact_type == "personal":
            npc.dynamic_personal_knowledge.append(fact)
        else:
            npc.dynamic_world_facts.append(fact)

        return {
            "npc_id": npc_id,
            "added": fact,
            "type": fact_type,
            "total_world_facts": (
                len(npc.world_facts) + len(npc.dynamic_world_facts)
            ),
            "total_personal": (
                len(npc.personal_knowledge) + len(npc.dynamic_personal_knowledge)
            ),
        }

    def unlock_knowledge_gate(self, npc_id: str, gate_id: str) -> dict:
        """Force-unlock a gated fact. For quest rewards, story progression, etc."""
        mgr = self._ensure_capability_manager(npc_id)
        if not mgr or "knowledge_gate" not in mgr.capabilities:
            return {"error": f"NPC '{npc_id}' has no knowledge_gate capability"}

        gate = mgr.capabilities["knowledge_gate"]
        if gate_id in gate.unlocked_ids:
            return {"npc_id": npc_id, "gate_id": gate_id, "status": "already_unlocked"}

        gate.unlocked_ids.add(gate_id)
        mgr.save_state()

        return {"npc_id": npc_id, "gate_id": gate_id, "status": "unlocked"}

    def get_social_graph(self) -> dict:
        """Get social graph data for visualization."""
        if not self.social_graph:
            return {"connections": [], "npcs": []}

        return {
            "npcs": list(self.social_graph.get_all_npcs()),
            "connections": [
                {
                    "from": c.from_id, "to": c.to_id,
                    "relationship": c.relationship,
                    "closeness": c.closeness,
                    "gossip_filter": c.gossip_filter,
                }
                for c in self.social_graph.connections
            ],
            "pending_gossip": self.gossip_propagator.pending_count if self.gossip_propagator else 0,
        }

    @property
    def active_npc(self) -> str:
        return self.pie.config.npc.active_profile or ""

    def shutdown(self) -> None:
        """Clean shutdown."""
        try:
            self.pie.shutdown()
        except Exception:
            pass
