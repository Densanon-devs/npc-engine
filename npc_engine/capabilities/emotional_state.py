"""
Emotional State Capability — Persistent mood state machine.

The model already outputs an `emotion` field in every JSON response.
This capability reads that field and uses it to update a persistent mood
that carries across conversations. The persistent mood then feeds back
as context, creating a feedback loop where the model's own emotional
outputs influence its future emotional state.

Mood decays toward the NPC's baseline over turns (configurable decay_rate).
Volatility controls how quickly mood shifts from new emotional signals.

This is NOT per-response emotion — it's persistent state.
"""

import json
import logging

from npc_engine.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityRegistry,
    CapabilityUpdate,
)

logger = logging.getLogger(__name__)

# Mood intensity descriptors
_INTENSITY_LABELS = {
    (0.0, 0.3): "slightly",
    (0.3, 0.6): "",       # No modifier at medium intensity
    (0.6, 0.8): "very",
    (0.8, 1.0): "deeply",
}

# Map common emotion words to mood categories for consolidation
_EMOTION_TO_MOOD = {
    # Positive
    "happy": "happy", "joyful": "happy", "cheerful": "happy", "pleased": "happy",
    "warm": "warm", "friendly": "warm", "welcoming": "warm", "kind": "warm",
    "grateful": "grateful", "thankful": "grateful",
    "amused": "amused", "laughing": "amused",
    "proud": "proud", "satisfied": "proud",
    "calm": "calm", "peaceful": "calm", "serene": "calm", "relaxed": "calm",
    "hopeful": "hopeful", "optimistic": "hopeful",
    # Negative
    "sad": "sad", "sorrowful": "sad", "melancholy": "sad", "grieving": "sad",
    "angry": "angry", "furious": "angry", "irritated": "angry", "annoyed": "angry",
    "worried": "worried", "anxious": "worried", "concerned": "worried", "nervous": "worried",
    "fearful": "fearful", "afraid": "fearful", "scared": "fearful", "terrified": "fearful",
    "suspicious": "suspicious", "wary": "suspicious", "distrustful": "suspicious",
    # Neutral
    "neutral": "neutral", "thoughtful": "contemplative",
    "contemplative": "contemplative", "pensive": "contemplative", "reflective": "contemplative",
    "serious": "serious", "stern": "serious", "grave": "serious",
    "curious": "curious", "intrigued": "curious", "interested": "curious",
}

# Behavioral directives per mood
_MOOD_DIRECTIVES = {
    "happy": "Speak with warmth and enthusiasm",
    "warm": "Be approachable and kind",
    "grateful": "Express thanks, be generous",
    "amused": "Be lighthearted, add humor",
    "proud": "Speak with confidence",
    "calm": "Speak evenly and peacefully",
    "hopeful": "Be encouraging and forward-looking",
    "sad": "Speak with a heavy heart, mention loss",
    "angry": "Be curt, speak with tension",
    "worried": "Speak with urgency about concerns",
    "fearful": "Be on edge, glance around nervously",
    "suspicious": "Question motives, guard words carefully",
    "contemplative": "Pause thoughtfully before speaking",
    "serious": "No small talk, get to the point",
    "curious": "Ask questions, lean forward",
    "neutral": "",
}


@CapabilityRegistry.register
class EmotionalStateCapability(Capability):
    """Persistent mood state machine driven by the model's own emotion output."""

    name = "emotional_state"
    version = "1.0"
    dependencies = []
    default_token_budget = 25

    def initialize(self, npc_id: str, yaml_config: dict, shared_state: dict) -> None:
        self.npc_id = npc_id
        self.baseline_mood: str = yaml_config.get("baseline_mood", "neutral")
        self.volatility: float = max(0.0, min(1.0, yaml_config.get("volatility", 0.3)))
        self.decay_rate: float = max(0.0, min(1.0, yaml_config.get("decay_rate", 0.1)))

        self.mood: str = self.baseline_mood
        self.intensity: float = 0.3
        self.turns_in_mood: int = 0
        self._previous_mood: str = self.mood

        shared_state["emotional_state"] = self._state_snapshot()

    def build_context(self, query: str, shared_state: dict) -> CapabilityContext:
        if self.mood == "neutral" and self.intensity < 0.3:
            return CapabilityContext("", 0, 80, "state")

        intensity_label = self._intensity_label()
        directive = _MOOD_DIRECTIVES.get(self.mood, "")

        if intensity_label:
            fragment = f"[Current mood: {intensity_label} {self.mood}. {directive}]"
        else:
            fragment = f"[Current mood: {self.mood}. {directive}]"

        token_est = len(fragment) // 4 + 1
        return CapabilityContext(
            context_fragment=fragment,
            token_estimate=min(token_est, self.default_token_budget),
            priority=80,
            section="state",
        )

    def process_response(self, response: str, query: str,
                         shared_state: dict) -> CapabilityUpdate:
        self._previous_mood = self.mood

        # Extract emotion from model's JSON response
        detected_emotion = self._extract_emotion(response)

        if detected_emotion:
            # Normalize to mood category
            new_mood = _EMOTION_TO_MOOD.get(detected_emotion.lower(), detected_emotion.lower())

            if new_mood == self.mood:
                # Same mood — reinforce (increase intensity, increment counter)
                self.turns_in_mood += 1
                self.intensity = min(1.0, self.intensity + 0.1 * self.volatility)
            else:
                # Different mood — shift based on volatility
                if self.volatility > 0.5 or self.turns_in_mood < 2:
                    # High volatility or short-lived mood → switch immediately
                    self.mood = new_mood
                    self.intensity = 0.3 + 0.2 * self.volatility
                    self.turns_in_mood = 1
                else:
                    # Low volatility, established mood → resist change, just nudge intensity down
                    self.intensity = max(0.1, self.intensity - 0.1)
                    self.turns_in_mood += 1
        else:
            # No emotion detected — decay toward baseline
            self._decay_toward_baseline()

        # Emit event if mood changed
        events = []
        if self.mood != self._previous_mood:
            events.append(f"mood_changed:{self._previous_mood}:{self.mood}")
            logger.debug(f"NPC '{self.npc_id}' mood: {self._previous_mood} -> {self.mood}")

        # Emit sustained mood event (useful for personality_drift)
        if self.turns_in_mood >= 5 and self.turns_in_mood % 5 == 0:
            events.append(f"mood_sustained:{self.mood}:{self.turns_in_mood}")

        snapshot = self._state_snapshot()
        shared_state["emotional_state"] = snapshot
        return CapabilityUpdate(state_patch=snapshot, events=events)

    def _extract_emotion(self, response: str) -> str | None:
        """Extract the emotion field from the model's JSON response."""
        try:
            obj = json.loads(response.strip())
            if isinstance(obj, dict):
                return obj.get("emotion", None)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try to find JSON substring
        r = response.strip()
        start = r.find("{")
        if start >= 0:
            end = r.rfind("}")
            if end > start:
                try:
                    obj = json.loads(r[start:end + 1])
                    if isinstance(obj, dict):
                        return obj.get("emotion", None)
                except (json.JSONDecodeError, ValueError):
                    pass
        return None

    def _decay_toward_baseline(self) -> None:
        """Gradually return mood to baseline."""
        if self.mood == self.baseline_mood:
            self.intensity = max(0.1, self.intensity - self.decay_rate * 0.5)
            return

        self.intensity -= self.decay_rate
        if self.intensity <= 0.1:
            self.mood = self.baseline_mood
            self.intensity = 0.3
            self.turns_in_mood = 0

    def _intensity_label(self) -> str:
        for (low, high), label in _INTENSITY_LABELS.items():
            if low <= self.intensity < high:
                return label
        return "deeply"

    def _state_snapshot(self) -> dict:
        return {
            "mood": self.mood,
            "intensity": round(self.intensity, 2),
            "turns_in_mood": self.turns_in_mood,
            "baseline": self.baseline_mood,
        }

    def get_state(self) -> dict:
        return self._state_snapshot()

    def load_state(self, state: dict) -> None:
        self.mood = state.get("mood", self.baseline_mood)
        self.intensity = state.get("intensity", 0.3)
        self.turns_in_mood = state.get("turns_in_mood", 0)
        self._previous_mood = self.mood
