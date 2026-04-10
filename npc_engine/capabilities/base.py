"""
NPC Capabilities System — Modular, opt-in intelligence for NPCs.

Each capability is an independent plugin that:
  - Reads config from the NPC's YAML knowledge sheet
  - Produces a context fragment for prompt injection (pre-generation)
  - Updates its persistent state based on model output (post-generation)
  - Communicates with other capabilities via shared_state

Capabilities are programmatic (heuristic/rule-based), NOT LLM-generated.
The model never maintains scratchpads, calculates trust, or manages moods.
The engine does this and injects results as read-only context.

Inspired by YC-Bench finding: persistent structured memory is the
#1 predictor of long-horizon agent success.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────

@dataclass
class CapabilityContext:
    """What a capability contributes to the prompt."""
    context_fragment: str       # Text to inject into the prompt
    token_estimate: int         # Approximate token count (~4 chars per token)
    priority: int               # Higher = allocated space first
    section: str                # "state" | "knowledge" | "directive"


@dataclass
class CapabilityUpdate:
    """State changes a capability wants to persist after generation."""
    state_patch: dict = field(default_factory=dict)
    events: list[str] = field(default_factory=list)


# ── Base capability ──────────────────────────────────────────

class Capability(ABC):
    """
    Base class for all NPC capabilities.

    Subclasses must set `name` as a class attribute and implement
    initialize(), build_context(), and process_response().
    """

    name: str = ""
    version: str = "1.0"
    dependencies: list[str] = []
    default_token_budget: int = 30

    @abstractmethod
    def initialize(self, npc_id: str, yaml_config: dict, shared_state: dict) -> None:
        """
        Called once when the NPC profile loads.
        yaml_config is this capability's section from the NPC YAML.
        """

    @abstractmethod
    def build_context(self, query: str, shared_state: dict) -> CapabilityContext:
        """
        Pre-generation: produce a context fragment for prompt injection.
        Called every turn before the expert generates a response.
        """

    @abstractmethod
    def process_response(self, response: str, query: str,
                         shared_state: dict) -> CapabilityUpdate:
        """
        Post-generation: update internal state based on the model's response
        and the player's query. Return state changes and events to broadcast.
        """

    def on_event(self, event: str, shared_state: dict) -> None:
        """React to events from other capabilities. Default: no-op."""

    def get_state(self) -> dict:
        """Return serializable state for persistence."""
        return {}

    def load_state(self, state: dict) -> None:
        """Restore from persisted state."""


# ── Capability registry ──────────────────────────────────────

class CapabilityRegistry:
    """
    Global registry of available capability types.
    Built-in capabilities auto-register on import.
    """

    _registry: dict[str, type[Capability]] = {}

    @classmethod
    def register(cls, cap_class: type[Capability]) -> type[Capability]:
        """Register a capability class. Can be used as a decorator."""
        if not cap_class.name:
            raise ValueError(f"{cap_class.__name__} must set a 'name' class attribute")
        cls._registry[cap_class.name] = cap_class
        logger.debug(f"Registered capability: {cap_class.name}")
        return cap_class

    @classmethod
    def get(cls, name: str) -> Optional[type[Capability]]:
        return cls._registry.get(name)

    @classmethod
    def list_all(cls) -> list[str]:
        return list(cls._registry.keys())


# ── Capability manager (per-NPC orchestrator) ────────────────

class CapabilityManager:
    """
    Manages the capability lifecycle for a single NPC.
    Created lazily when an NPC is first activated.
    """

    def __init__(self, npc_id: str, capability_configs: dict,
                 state_dir: str = "data/npc_state"):
        self.npc_id = npc_id
        self.capabilities: dict[str, Capability] = {}
        self.shared_state: dict = {}
        self._state_dir = Path(state_dir)
        self._state_path = self._state_dir / f"{npc_id}.json"
        self._turn_count = 0

        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._load_capabilities(capability_configs)
        self._load_state()

    def _load_capabilities(self, configs: dict) -> None:
        """Instantiate capabilities from YAML config and validate dependencies."""
        for cap_name, cap_config in configs.items():
            if isinstance(cap_config, dict) and not cap_config.get("enabled", True):
                continue

            cap_class = CapabilityRegistry.get(cap_name)
            if cap_class is None:
                logger.warning(f"NPC '{self.npc_id}': unknown capability '{cap_name}', skipping")
                continue

            cap = cap_class()
            cap.initialize(self.npc_id, cap_config if isinstance(cap_config, dict) else {}, self.shared_state)
            self.capabilities[cap_name] = cap
            logger.info(f"NPC '{self.npc_id}': loaded capability '{cap_name}'")

        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        """Check that all capability dependencies are satisfied."""
        for name, cap in self.capabilities.items():
            for dep in cap.dependencies:
                if dep not in self.capabilities:
                    raise ValueError(
                        f"NPC '{self.npc_id}': capability '{name}' requires '{dep}', "
                        f"but '{dep}' is not enabled. Add it to the capabilities section."
                    )

    def build_all_contexts(self, query: str, token_budget: int = 200) -> str:
        """
        Call build_context on all active capabilities, compose within token budget.
        Higher-priority capabilities get space first; lowest-priority truncated if over.
        """
        if not self.capabilities:
            return ""

        contexts: list[CapabilityContext] = []
        for cap in self.capabilities.values():
            try:
                ctx = cap.build_context(query, self.shared_state)
                if ctx and ctx.context_fragment.strip():
                    contexts.append(ctx)
            except Exception as e:
                logger.error(f"Capability '{cap.name}' build_context failed: {e}")

        if not contexts:
            return ""

        # Sort by priority descending — highest priority gets budget first
        contexts.sort(key=lambda c: c.priority, reverse=True)

        fragments = []
        used_tokens = 0
        for ctx in contexts:
            if used_tokens + ctx.token_estimate <= token_budget:
                fragments.append(ctx.context_fragment)
                used_tokens += ctx.token_estimate
            else:
                # Try to fit a truncated version
                remaining = token_budget - used_tokens
                if remaining > 10:  # Only worth including if >10 tokens left
                    # Rough truncation: ~4 chars per token
                    max_chars = remaining * 4
                    truncated = ctx.context_fragment[:max_chars].rsplit(" ", 1)[0]
                    if truncated.strip():
                        fragments.append(truncated + "]")
                        used_tokens += remaining
                break  # Budget exhausted

        return "\n".join(fragments)

    def process_all_responses(self, response: str, query: str) -> None:
        """
        Post-generation: update all capability states, emit cross-events, persist.
        Execution order follows priority (highest first).
        """
        self._turn_count += 1
        all_events: list[str] = []

        # Process in priority order
        ordered = sorted(self.capabilities.values(),
                         key=lambda c: c.default_token_budget, reverse=True)
        # Actually sort by the class-level priority hint (use default_token_budget as proxy
        # since priority is per-context; for ordering, we use a stable order)
        ordered = sorted(self.capabilities.values(),
                         key=lambda c: _capability_priority(c), reverse=True)

        for cap in ordered:
            try:
                update = cap.process_response(response, query, self.shared_state)
                if update:
                    # Apply state patch to shared_state
                    if update.state_patch:
                        self.shared_state.setdefault(cap.name, {}).update(update.state_patch)
                    all_events.extend(update.events)
            except Exception as e:
                logger.error(f"Capability '{cap.name}' process_response failed: {e}")

        # Broadcast events to all capabilities
        for event in all_events:
            for cap in self.capabilities.values():
                try:
                    cap.on_event(event, self.shared_state)
                except Exception as e:
                    logger.error(f"Capability '{cap.name}' on_event failed for '{event}': {e}")

        self.save_state()

    def save_state(self) -> None:
        """Persist all capability states to JSON."""
        state = {
            "npc_id": self.npc_id,
            "last_updated": time.time(),
            "turn_count": self._turn_count,
            "shared_state": self.shared_state,
            "capabilities": {},
        }
        for name, cap in self.capabilities.items():
            try:
                state["capabilities"][name] = cap.get_state()
            except Exception as e:
                logger.error(f"Capability '{name}' get_state failed: {e}")

        try:
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save NPC state for '{self.npc_id}': {e}")

    def _load_state(self) -> None:
        """Restore capability states from persisted JSON."""
        if not self._state_path.exists():
            return

        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load NPC state for '{self.npc_id}': {e}")
            return

        self._turn_count = state.get("turn_count", 0)
        self.shared_state = state.get("shared_state", {})

        cap_states = state.get("capabilities", {})
        for name, cap in self.capabilities.items():
            if name in cap_states:
                try:
                    cap.load_state(cap_states[name])
                except Exception as e:
                    logger.warning(f"Capability '{name}' load_state failed: {e}")

    @property
    def turn_count(self) -> int:
        return self._turn_count


# ── Priority helper ──────────────────────────────────────────

# Default processing priorities for known capability types.
# Higher = processed first and gets token budget first.
_PRIORITY_MAP = {
    "trust": 90,
    "emotional_state": 80,
    "goals": 70,
    "scratchpad": 60,
    "dynamic_quests": 55,
    "knowledge_gate": 50,
    "world_awareness": 40,
    "faction": 40,
    "conversation_threading": 30,
    "personality_drift": 20,
}


def _capability_priority(cap: Capability) -> int:
    """Get processing priority for a capability instance."""
    return _PRIORITY_MAP.get(cap.name, 50)
