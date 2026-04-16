"""
NPC Knowledge System — Dynamic knowledge sheets, event injection, quest management.

NPCs have knowledge sheets (YAML) with:
  - identity: name, role, location
  - world_facts: things they know about the world
  - personal_knowledge: private memories and skills
  - active_quests: quests they can assign
  - recent_events: dynamically injected events

The knowledge system builds a context string that gets prepended to
the expert's user input, giving the NPC access to their knowledge
without modifying the expert architecture.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Quest:
    id: str
    name: str
    description: str
    status: str = "available"  # available, active, completed, failed
    reward: str = ""
    objectives: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        return (f"{self.name}: {self.description} "
                f"Objective: {self.objectives[0] if self.objectives else self.description}. "
                f"Reward: {self.reward}")

    def to_prompt_short(self) -> str:
        return f"{self.name} ({self.status})"

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "status": self.status, "reward": self.reward, "objectives": self.objectives,
        }


@dataclass
class Event:
    description: str
    timestamp: float = 0.0
    source: str = "system"

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


def _combine_static_and_dynamic(
    static_list: list,
    dynamic_list: list,
    total_cap: int,
    dynamic_reserve_min: int = 0,
) -> list:
    """
    Pick at most ``total_cap`` items to surface in a prompt block,
    combining a profile-authored static list with a runtime-injected
    dynamic list.

    Static items come from the profile YAML and represent
    identity-grounding lore (relationships, secrets, backstory).
    They're authored in priority order, so we take from the front.
    Dynamic items are appended at runtime by the Story Director and
    represent situational facts (recent events, new rumours). They're
    appended in chronological order, so we take from the back to keep
    newest. When dynamic_list is empty the result is exactly
    ``static_list[:total_cap]`` — identical to the pre-dynamic-lane
    behaviour, so tests and worlds that don't exercise runtime
    injection see no shape change.
    """
    if not dynamic_list:
        return list(static_list[:total_cap])
    dyn_slots = max(dynamic_reserve_min, total_cap - len(static_list))
    dyn_slots = min(dyn_slots, len(dynamic_list), total_cap)
    static_slots = total_cap - dyn_slots
    return list(static_list[:static_slots]) + list(dynamic_list[-dyn_slots:])


class NPCKnowledge:
    """
    Manages a single NPC's knowledge sheet.
    Loads from YAML, supports dynamic event injection and quest state updates.
    """

    def __init__(self, profile_path: str):
        self.profile_path = Path(profile_path)
        self.identity: dict = {}
        self.world_facts: list[str] = []
        self.personal_knowledge: list[str] = []
        # Runtime-only lanes for Story Director / engine.add_knowledge
        # injections. Kept separate from the YAML-loaded static lists
        # so identity-grounding profile lore can't be silently pushed
        # out of the dialogue prompt by accumulated runtime facts.
        # Not persisted: re-derived from the FactLedger on restart.
        self.dynamic_world_facts: list[str] = []
        self.dynamic_personal_knowledge: list[str] = []
        self.quests: list[Quest] = []
        self.events: list[Event] = []
        self.capability_configs: dict = {}  # Raw YAML for capability system
        # Zone fields for Story Director locality scoping. "global" is
        # the wildcard — always considered in every active zone, so
        # worlds without zone config (no `zone:` in profile YAML) run
        # in world-wide mode and existing behaviour is preserved.
        # `home_zone` is immutable profile data; `current_zone` is
        # runtime-mutable for mobile NPCs (traveling merchants,
        # wandering assassins, etc.); `mobile` gates whether
        # /story/npc_zone calls are allowed to change current_zone.
        self.home_zone: str = "global"
        self.current_zone: str = "global"
        self.mobile: bool = False
        # Lifecycle fields for Phase 2 of the Story Director zone +
        # lifecycle work. Every NPC starts alive; deaths flip `status`
        # to "deceased" and stamp the other fields. Fully persisted
        # back into the profile YAML so a game restart honors deaths.
        # `inheritor` optionally points to a successor NPC that
        # receives any open quests when this NPC dies.
        self.status: str = "alive"
        self.death_tick: Optional[int] = None
        self.death_cause: Optional[str] = None
        self.inheritor: Optional[str] = None

        if self.profile_path.exists():
            self._load()

    def _load(self):
        with open(self.profile_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.identity = data.get("identity", {})
        self.world_facts = data.get("world_facts", [])
        self.personal_knowledge = data.get("personal_knowledge", [])
        # Zone + mobility from profile YAML. Default "global" means
        # the NPC is always in every active zone; most non-mobile
        # stationary NPCs living in a specific town should set
        # `zone: "<town_name>"` explicitly in their YAML.
        self.home_zone = str(data.get("zone", "global"))
        self.current_zone = self.home_zone
        self.mobile = bool(data.get("mobile", False))
        # Lifecycle fields. Default "alive" so profiles without a
        # status key keep working. A deceased NPC profile looks like:
        #   status: deceased
        #   death_tick: 42
        #   death_cause: "bandit ambush on the north road"
        #   inheritor: kael_apprentice_pip    # optional
        self.status = str(data.get("status", "alive"))
        dt = data.get("death_tick")
        self.death_tick = int(dt) if isinstance(dt, (int, float)) else None
        dc = data.get("death_cause")
        self.death_cause = str(dc) if dc else None
        ih = data.get("inheritor")
        self.inheritor = str(ih) if ih else None

        for q in data.get("active_quests", []):
            self.quests.append(Quest(
                id=q["id"], name=q["name"], description=q["description"],
                status=q.get("status", "available"), reward=q.get("reward", ""),
                objectives=q.get("objectives", []),
            ))

        for e in data.get("recent_events", []):
            if isinstance(e, str):
                self.events.append(Event(description=e))
            elif isinstance(e, dict):
                self.events.append(Event(
                    description=e["description"],
                    source=e.get("source", "system"),
                ))

        # Phase 8: Store raw capabilities config for the capability system
        self.capability_configs = data.get("capabilities", {})

        caps_str = f", {len(self.capability_configs)} capabilities" if self.capability_configs else ""
        logger.info(f"Loaded NPC profile: {self.identity.get('name', '?')} "
                    f"({len(self.world_facts)} facts, {len(self.quests)} quests{caps_str})")

    def inject_event(self, description: str, source: str = "world"):
        """Inject a new event into the NPC's knowledge mid-conversation."""
        self.events.append(Event(description=description, source=source))
        logger.info(f"Event injected for {self.identity.get('name', '?')}: {description}")

    def update_quest(self, quest_id: str, new_status: str):
        """Update a quest's status."""
        for q in self.quests:
            if q.id == quest_id:
                q.status = new_status
                logger.info(f"Quest '{q.name}' updated to {new_status}")
                return True
        return False

    def add_quest(self, quest: Quest):
        """Add a new quest to this NPC."""
        self.quests.append(quest)
        logger.info(f"Quest '{quest.name}' added for {self.identity.get('name', '?')}")

    def build_context(self, include_quests: bool = True, include_events: bool = True,
                      player_quests: list = None, capability_context: str = None,
                      world_name: str = "") -> str:
        """
        Build a knowledge context string to prepend to the expert prompt.
        Compact format — every token counts for small models.
        """
        parts = []

        # Identity + personality (critical for generic expert)
        name = self.identity.get("name", "NPC")
        role = self.identity.get("role", "")
        personality = self.identity.get("personality", "")
        speech = self.identity.get("speech_style", "")
        # Use world from identity, fallback to passed world_name, or omit
        world = self.identity.get("world", world_name)
        if world:
            parts.append(f"[You are {name}, {role} of {world}. {personality}]")
        else:
            parts.append(f"[You are {name}, {role}. {personality}]")
        if speech:
            parts.append(f"[Speech: {speech}]")

        # World facts: profile-authored static lore + Director-injected
        # dynamic facts. Reserve up to 2 slots for newest dynamic items
        # when any exist so the Director's recent injections always
        # reach the dialogue prompt; static items 1-4 are preserved.
        if self.world_facts or self.dynamic_world_facts:
            fact_items = _combine_static_and_dynamic(
                self.world_facts, self.dynamic_world_facts,
                total_cap=6, dynamic_reserve_min=2,
            )
            if fact_items:
                facts = "; ".join(fact_items)
                parts.append(f"[Facts: {facts}]")

        # Personal knowledge: same shape, smaller cap. Reserve up to 2
        # slots for newest dynamic items; static items 1-2 (the most
        # identity-critical ones in profile order) are always preserved.
        if self.personal_knowledge or self.dynamic_personal_knowledge:
            personal_items = _combine_static_and_dynamic(
                self.personal_knowledge, self.dynamic_personal_knowledge,
                total_cap=4, dynamic_reserve_min=2,
            )
            if personal_items:
                personal = "; ".join(personal_items)
                parts.append(f"[Personal: {personal}]")

        # Active quests — EXPLICIT format so model copies the details
        if include_quests and self.quests:
            available = [q for q in self.quests if q.status in ("available", "active")]
            if available:
                for q in available:
                    parts.append(f"[YOUR QUEST: {q.to_prompt()}]")

        # Player quest state (if provided)
        if player_quests:
            my_name = name.lower()
            player_active = [q for q in player_quests if q.get("given_by") == my_name and q.get("status") == "active"]
            player_done = [q for q in player_quests if q.get("given_by") == my_name and q.get("status") == "completed"]
            if player_active:
                names = ", ".join(q["name"] for q in player_active)
                parts.append(f"[PLAYER IS WORKING ON: {names}]")
            if player_done:
                names = ", ".join(q["name"] for q in player_done)
                parts.append(f"[PLAYER COMPLETED: {names} -- thank them and give reward]")

        # Capability context (Phase 8: injected by CapabilityManager)
        if capability_context:
            parts.append(capability_context)

        # Recent events (highest priority — these are NEW info the model MUST mention)
        if include_events and self.events:
            event_strs = [e.description for e in self.events[-3:]]
            parts.append(f"[RECENT NEWS you just heard: {'; '.join(event_strs)}]")

        return "\n".join(parts)

    def build_quest_system_prompt(self) -> str:
        """
        Approach B: Build quest details as a SYSTEM PROMPT addition
        instead of knowledge context. Puts quest info where the model
        pays most attention.
        """
        available = [q for q in self.quests if q.status in ("available", "active")]
        if not available:
            return ""
        q = available[0]
        return (
            f"When the player asks for work or help, give them this quest: "
            f"Type: {q.id}. "
            f"Say: \"{q.description}\" "
            f"Objective: \"{q.objectives[0] if q.objectives else q.description}\" "
            f"Reward: \"{q.reward}\". "
            f"Output the quest field in your JSON response."
        )

    def extract_quest_for_injection(self) -> dict | None:
        """
        Approach C: Extract quest data for programmatic injection.
        The model generates dialogue only; the game engine appends
        the quest JSON from the knowledge sheet.
        """
        available = [q for q in self.quests if q.status in ("available", "active")]
        if not available:
            return None
        q = available[0]
        return {
            "type": q.id,
            "objective": q.objectives[0] if q.objectives else q.description,
            "reward": q.reward,
        }

    def save(self):
        """Save current state back to YAML."""
        data = {
            "identity": self.identity,
            "world_facts": self.world_facts,
            "personal_knowledge": self.personal_knowledge,
            "active_quests": [q.to_dict() for q in self.quests],
            "recent_events": [{"description": e.description, "source": e.source}
                              for e in self.events],
        }
        with open(self.profile_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


class NPCKnowledgeManager:
    """Manages knowledge sheets for all NPCs."""

    def __init__(self, profiles_dir: str = "data/npc_profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles: dict[str, NPCKnowledge] = {}
        self._load_all()

    def _load_all(self):
        if not self.profiles_dir.exists():
            return
        for path in self.profiles_dir.glob("*.yaml"):
            npc_id = path.stem
            self.profiles[npc_id] = NPCKnowledge(str(path))

    def get(self, npc_id: str) -> Optional[NPCKnowledge]:
        return self.profiles.get(npc_id)

    def inject_global_event(self, description: str):
        """Inject an event into ALL NPCs' knowledge."""
        for npc in self.profiles.values():
            npc.inject_event(description, source="world")

    def inject_event(self, npc_id: str, description: str):
        """Inject an event into a specific NPC's knowledge."""
        npc = self.profiles.get(npc_id)
        if npc:
            npc.inject_event(description, source="targeted")


class PlayerQuestTracker:
    """
    Tracks the player's quest state across all NPCs.
    NPCs use this to know if the player already has their quest,
    has completed it, or needs a new one.
    """

    def __init__(self, path: str = "data/player_quests.yaml", world_name: str = ""):
        self.path = Path(path)
        self.player_name = "Traveler"
        self.active_quests: list[dict] = []
        self.completed_quests: list[dict] = []
        self.reputation: dict[str, int] = {world_name: 0} if world_name else {}

        if self.path.exists():
            self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.player_name = data.get("player_name", "Traveler")
        self.active_quests = data.get("active_quests", []) or []
        self.completed_quests = data.get("completed_quests", []) or []
        self.reputation = data.get("reputation", self.reputation)

    def accept_quest(self, quest_id: str, quest_name: str, given_by: str):
        """Player accepts a quest from an NPC."""
        self.active_quests.append({
            "id": quest_id, "name": quest_name,
            "given_by": given_by, "status": "active",
            "objectives_completed": [],
        })
        logger.info(f"Quest accepted: {quest_name} from {given_by}")

    def complete_quest(self, quest_id: str):
        """Player completes a quest."""
        for i, q in enumerate(self.active_quests):
            if q["id"] == quest_id:
                q["status"] = "completed"
                self.completed_quests.append(self.active_quests.pop(i))
                logger.info(f"Quest completed: {q['name']}")
                return True
        return False

    def has_quest(self, quest_id: str) -> bool:
        return any(q["id"] == quest_id for q in self.active_quests)

    def has_completed(self, quest_id: str) -> bool:
        return any(q["id"] == quest_id for q in self.completed_quests)

    def get_all_quests(self) -> list[dict]:
        """Get all quests (active + completed) for context injection."""
        return self.active_quests + self.completed_quests

    def save(self):
        data = {
            "player_name": self.player_name,
            "active_quests": self.active_quests,
            "completed_quests": self.completed_quests,
            "reputation": self.reputation,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


# ── Quest-aware JSON grammar ────────────────────────────────

# Extended grammar that supports optional quest field
QUEST_JSON_GRAMMAR = r'''
root   ::= "{" ws dialogue "," ws emotion "," ws action ("," ws quest)? ws "}"
dialogue ::= "\"dialogue\"" ws ":" ws string
emotion  ::= "\"emotion\"" ws ":" ws string
action   ::= "\"action\"" ws ":" ws (string | "null")
quest    ::= "\"quest\"" ws ":" ws (questobj | "null")
questobj ::= "{" ws "\"type\"" ws ":" ws string "," ws "\"objective\"" ws ":" ws string "," ws "\"reward\"" ws ":" ws string ws "}"
string ::= "\"" ([^"\\] | "\\" .)* "\""
ws     ::= [ \t\n]*
'''
