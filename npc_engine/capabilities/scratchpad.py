"""
Scratchpad Capability — NPC remembers notable facts from past interactions.

Inspired by YC-Bench finding: scratchpad discipline is the #1 predictor
of long-horizon agent success. Top-performing agents (Claude Opus, GLM-5)
rewrote their scratchpad ~34 times per run, evolving through phases.

This capability is PROGRAMMATIC — we extract facts from the player's input
using keyword heuristics, NOT by asking the model to write notes.
The scratchpad is injected as read-only context every turn.
"""

import logging
import re
import time
from dataclasses import dataclass, field

from npc_engine.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityRegistry,
    CapabilityUpdate,
)

logger = logging.getLogger(__name__)


@dataclass
class ScratchpadEntry:
    text: str
    turn: int
    importance: float = 0.5
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# Words that indicate the player is sharing personal/notable information
_NOTABLE_PATTERNS = [
    # Self-identification
    (r"\b(?:i am|i'm|my name is|call me|they call me)\s+(\w+)", 0.8),
    # Origin/location
    (r"\b(?:i come from|i'm from|i came from|i traveled from|i hail from)\s+(.+?)(?:\.|$|,)", 0.7),
    # Possession of notable items
    (r"\b(?:i have|i carry|i found|i got|i bring|i brought)\s+(?:a |an |the )?(.+?)(?:\.|$|,)", 0.5),
    # Seeking something specific
    (r"\b(?:i'm looking for|i seek|i need to find|i'm searching for)\s+(.+?)(?:\.|$|,)", 0.6),
    # Profession/role
    (r"\b(?:i'm a|i am a|i work as|i serve as)\s+(\w+(?:\s+\w+)?)", 0.7),
    # Past events
    (r"\b(?:i saw|i heard|i witnessed|i discovered|i learned)\s+(.+?)(?:\.|$|,)", 0.6),
    # Relationships
    (r"\b(?:my (?:father|mother|brother|sister|friend|master|lord))\s+(\w+)", 0.7),
]

# Quest-related keywords that should be tracked
_QUEST_KEYWORDS = {"quest", "task", "mission", "help", "problem", "trouble",
                   "investigate", "find", "retrieve", "deliver", "protect",
                   "defeat", "slay", "rescue", "escort"}

# Locations and proper nouns (capitalized words, 3+ chars, not sentence starters)
_PROPER_NOUN_RE = re.compile(r'(?<!\. )(?<!^)\b([A-Z][a-z]{2,})\b')


@CapabilityRegistry.register
class ScratchpadCapability(Capability):
    """NPC remembers notable facts from past player interactions."""

    name = "scratchpad"
    version = "1.0"
    dependencies = []
    default_token_budget = 60

    def initialize(self, npc_id: str, yaml_config: dict, shared_state: dict) -> None:
        self.npc_id = npc_id
        self.max_entries = yaml_config.get("max_entries", 10)
        self.entries: list[ScratchpadEntry] = []
        self._turn = 0

    def build_context(self, query: str, shared_state: dict) -> CapabilityContext:
        if not self.entries:
            return CapabilityContext(
                context_fragment="",
                token_estimate=0,
                priority=60,
                section="knowledge",
            )

        # Select top entries by importance, limited to fit budget
        sorted_entries = sorted(self.entries, key=lambda e: e.importance, reverse=True)
        top = sorted_entries[:5]
        # Re-sort by turn order for natural reading
        top.sort(key=lambda e: e.turn)

        memories = "; ".join(e.text for e in top)
        fragment = f"[You remember: {memories}]"
        token_est = len(fragment) // 4 + 1

        return CapabilityContext(
            context_fragment=fragment,
            token_estimate=min(token_est, self.default_token_budget),
            priority=60,
            section="knowledge",
        )

    def process_response(self, response: str, query: str,
                         shared_state: dict) -> CapabilityUpdate:
        self._turn += 1
        new_entries = self._extract_notable_facts(query)

        for entry in new_entries:
            self._add_entry(entry)

        # Publish scratchpad to shared_state for other capabilities
        return CapabilityUpdate(
            state_patch={"entries_count": len(self.entries)},
            events=[],
        )

    def _extract_notable_facts(self, query: str) -> list[ScratchpadEntry]:
        """Extract notable facts from the player's input using heuristics."""
        entries = []
        query_lower = query.lower().strip()

        # Pattern-based extraction
        for pattern, importance in _NOTABLE_PATTERNS:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                match_text = match.strip() if isinstance(match, str) else match[0].strip()
                if len(match_text) > 2:
                    # Build a readable summary
                    summary = self._build_summary(pattern, match_text, query_lower)
                    if summary and not self._is_duplicate(summary):
                        entries.append(ScratchpadEntry(
                            text=summary,
                            turn=self._turn,
                            importance=importance,
                        ))

        # Extract proper nouns from original (non-lowered) query
        proper_nouns = _PROPER_NOUN_RE.findall(query)
        # Filter out common English words that happen to be capitalized
        common = {"The", "This", "That", "What", "Where", "When", "Who", "How",
                  "Yes", "No", "Hello", "Hey", "Please", "Thank", "Thanks",
                  "Can", "Could", "Would", "Should", "Will", "Have", "Has",
                  "Are", "Were", "Was", "Not", "But", "And", "For", "Its"}
        notable_nouns = [n for n in proper_nouns if n not in common]
        if notable_nouns:
            nouns_text = "Player mentioned: " + ", ".join(set(notable_nouns))
            if not self._is_duplicate(nouns_text):
                entries.append(ScratchpadEntry(
                    text=nouns_text,
                    turn=self._turn,
                    importance=0.4,
                ))

        # Detect quest-related topics
        words = set(query_lower.split())
        quest_words = words & _QUEST_KEYWORDS
        if quest_words:
            quest_text = f"Player asked about: {', '.join(quest_words)}"
            if not self._is_duplicate(quest_text):
                entries.append(ScratchpadEntry(
                    text=quest_text,
                    turn=self._turn,
                    importance=0.6,
                ))

        return entries

    def _build_summary(self, pattern: str, match: str, query: str) -> str:
        """Build a human-readable scratchpad entry from a regex match."""
        if "name is" in pattern or "call me" in pattern or "i am" in pattern:
            return f"Player's name is {match.title()}"
        if "come from" in pattern or "from" in pattern or "hail from" in pattern:
            return f"Player came from {match.strip().rstrip('.')}"
        if "looking for" in pattern or "seek" in pattern or "searching" in pattern:
            return f"Player is looking for {match.strip().rstrip('.')}"
        if "i'm a" in pattern or "work as" in pattern:
            return f"Player is a {match.strip()}"
        if "saw" in pattern or "heard" in pattern or "witnessed" in pattern:
            return f"Player reported: {match.strip().rstrip('.')}"
        if "father" in pattern or "mother" in pattern or "friend" in pattern:
            return f"Player mentioned family/friend: {match.strip()}"
        if "have" in pattern or "carry" in pattern or "found" in pattern:
            return f"Player has {match.strip().rstrip('.')}"
        return ""

    def _is_duplicate(self, text: str) -> bool:
        """Check if a similar entry already exists."""
        text_lower = text.lower()
        for entry in self.entries:
            # Simple overlap check — if >60% of words match, it's a duplicate
            existing = set(entry.text.lower().split())
            new = set(text_lower.split())
            if not existing or not new:
                continue
            overlap = len(existing & new) / max(len(existing), len(new))
            if overlap > 0.6:
                return True
        return False

    def _add_entry(self, entry: ScratchpadEntry) -> None:
        """Add an entry, pruning oldest/least important if at max."""
        self.entries.append(entry)

        if len(self.entries) > self.max_entries:
            # Remove the least important entry (break ties by oldest)
            self.entries.sort(key=lambda e: (e.importance, e.turn))
            self.entries.pop(0)

    def get_state(self) -> dict:
        return {
            "turn": self._turn,
            "entries": [
                {"text": e.text, "turn": e.turn,
                 "importance": e.importance, "timestamp": e.timestamp}
                for e in self.entries
            ],
        }

    def load_state(self, state: dict) -> None:
        self._turn = state.get("turn", 0)
        self.entries = []
        for e in state.get("entries", []):
            self.entries.append(ScratchpadEntry(
                text=e["text"],
                turn=e["turn"],
                importance=e.get("importance", 0.5),
                timestamp=e.get("timestamp", 0.0),
            ))
