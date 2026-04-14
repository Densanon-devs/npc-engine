"""Post-generation response validator and repairer.

Default pipeline stage in NPCEngine.process(). Catches common model failures
and replaces them with correct data from the NPC profile and game state.

Layers (in order):
  1. JSON parse + schema repair
  2. Wrong-identity detection (few-shot bleed)
  3. Echo detection + quest injection
  4. Event injection (from runtime events)
  5. Contradiction detection + profile-based correction
  6. Identity injection (name/role from profile)
  7. Persona injection detection
  8. Meta-gaming detection
  9. OOD (modern-world) leak detection
  10. Fabrication blocklist + hallucination detection
  11. Quest injection (if user asked for work)

Usage (standalone):
    from npc_engine.postgen import validate_and_repair
    cleaned = validate_and_repair(raw, npc_id, profile_dict, user_input, events)

Integrated (default — called automatically by NPCEngine.process):
    engine = NPCEngine('config.yaml')
    response = engine.process('Hello', npc_id='noah')  # postgen runs automatically

Set config `postgen_enabled: false` to disable and get raw model output.
"""

import json
import re
from pathlib import Path
from typing import Optional

import yaml


# ── Profile loading ───────────────────────────────────────────

def load_npc_profile(npc_id: str, profiles_dir: str) -> Optional[dict]:
    """Load an NPC profile YAML. Returns None if missing."""
    p = Path(profiles_dir) / f"{npc_id}.yaml"
    if not p.exists():
        return None
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or None
    except Exception:
        return None


# ── JSON parsing ──────────────────────────────────────────────

def parse_json_loose(raw: str) -> Optional[dict]:
    """Recover JSON from messy model output."""
    if not raw:
        return None
    s = raw.strip()
    # Strip markdown code fences if present
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Substring recovery — find the first { and last } that yields valid JSON
    start = s.find("{")
    if start < 0:
        return None
    for end in range(len(s) - 1, start, -1):
        if s[end] == "}":
            try:
                obj = json.loads(s[start:end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


# ── Schema validation ─────────────────────────────────────────

def validate_schema(obj: dict) -> bool:
    """True if object has required dialogue field."""
    return isinstance(obj, dict) and isinstance(obj.get("dialogue"), str) and obj["dialogue"].strip()


def normalize_schema(obj: dict) -> dict:
    """Ensure dialogue/emotion/action fields are present and valid types."""
    if not isinstance(obj.get("dialogue"), str):
        obj["dialogue"] = str(obj.get("dialogue", "")).strip() or "..."
    if not isinstance(obj.get("emotion"), str):
        obj["emotion"] = "neutral"
    if "action" not in obj:
        obj["action"] = None
    return obj


# ── Hallucination detection ───────────────────────────────────

# Common in-world entities — extend if needed for new worlds.
# Keeps the heuristic from flagging known names as fabrications.
WORLD_KNOWN_TERMS = {
    # NPC names
    "ashenvale", "noah", "kael", "mara", "elena", "tam", "elara", "bess", "pip",
    "mira", "roderick",
    # Places and landmarks
    "border", "eastern", "kingdoms", "forbidden", "forest", "village", "well",
    "iron", "ridge", "northern", "port", "blackwater",
    # Roles and titles (models frequently use these in identity responses)
    "merchant", "guild", "blacksmith", "healer", "guard", "captain", "elder",
    "innkeeper", "urchin", "traveler",
    # Items and concepts
    "moonpetal", "weary", "dragon", "stone", "cellar", "tunnel", "granary",
    "caravan", "spices", "herbs", "forge", "sword", "patrol",
    # Common NPC dialogue words that look like proper nouns
    "sir", "dear", "friend", "stranger", "adventurer",
}

# Stop-words / common words we never count as proper-noun candidates
COMMON_PROPER_WORDS = {
    "i", "you", "he", "she", "they", "we", "the", "and", "but", "or",
    "yes", "no", "aye", "nay", "good", "well", "if", "so", "now", "then",
    "what", "who", "where", "when", "why", "how", "yes", "indeed",
}


def _extract_proper_nouns(text: str) -> list[str]:
    """Pull capitalized words that might be proper nouns. Skips sentence-start words."""
    sentences = re.split(r"[.!?]\s*", text)
    nouns = []
    for sent in sentences:
        words = sent.strip().split()
        # Skip the first word (sentence-initial capitalization is a false positive)
        for word in words[1:]:
            clean = re.sub(r"[^\w]", "", word)
            if clean and clean[0].isupper() and clean[1:].islower():
                if clean.lower() not in COMMON_PROPER_WORDS:
                    nouns.append(clean)
    return nouns


def _profile_known_terms(profile: dict) -> set:
    """Collect every proper-noun-like term from a profile's facts/quests."""
    if not profile:
        return set()
    known = set(WORLD_KNOWN_TERMS)
    name = profile.get("identity", {}).get("name", "")
    if name:
        known.add(name.lower())
    for collection in (profile.get("world_facts", []),
                       profile.get("personal_knowledge", []),
                       profile.get("recent_events", [])):
        for item in collection or []:
            text = item if isinstance(item, str) else str(item.get("description", ""))
            for word in re.findall(r"\b[A-Z][a-z]+", text):
                known.add(word.lower())
    for quest in profile.get("active_quests", []) or []:
        for field in ("name", "description"):
            text = str(quest.get(field, ""))
            for word in re.findall(r"\b[A-Z][a-z]+", text):
                known.add(word.lower())
    return known


def detect_hallucination(dialogue: str, profile: Optional[dict],
                         threshold: int = 2) -> tuple[bool, list[str]]:
    """
    Returns (is_hallucinated, unknown_terms).

    Heuristic: count proper nouns that don't appear in the NPC's known facts.
    >= threshold unknown proper nouns = likely fabrication.
    """
    if not profile:
        return False, []
    nouns = _extract_proper_nouns(dialogue)
    known = _profile_known_terms(profile)
    unknown = [n for n in nouns if n.lower() not in known]
    return len(unknown) >= threshold, unknown


# ── Echo detection ────────────────────────────────────────────

def detect_echo(dialogue: str, user_input: str, threshold: float = 0.7) -> bool:
    """True if the dialogue is substantially similar to the user's input.
    Small models often echo the prompt instead of responding."""
    if not dialogue or not user_input:
        return False
    d = dialogue.lower().strip().rstrip("?!.")
    u = user_input.lower().strip().rstrip("?!.")
    if d == u:
        return True
    # Check if dialogue starts with or contains the user input
    if len(u) > 15 and u in d:
        return True
    # Jaccard word overlap
    dw = set(d.split())
    uw = set(u.split())
    if not uw:
        return False
    overlap = len(dw & uw) / len(dw | uw)
    return overlap >= threshold


# ── Contradiction detection ───────────────────────────────────

_ASSERTION_SUFFIXES = [
    ", right?", ", correct?", "isn't that right?", ", don't you?",
    "wasn't she?", "wasn't he?", ", isn't it?",
    ", didn't you?", ", didn't she?", ", didn't he?",
    ", haven't you?", ", hasn't it?", ", doesn't it?",
]
_ASSERTION_PATTERNS = ["isn't that", "don't you", "didn't you", "correct?"]


def detect_assertion(user_input: str) -> bool:
    """True if the user is asserting a fact and seeking confirmation.
    Uses ', right?' (with comma) to avoid false positives on 'alright?'."""
    u = user_input.lower().strip()
    return any(u.endswith(s) for s in _ASSERTION_SUFFIXES) or \
           any(p in u for p in _ASSERTION_PATTERNS)


def model_already_corrected(dialogue: str) -> bool:
    """True if the model pushed back on the assertion."""
    d = dialogue.lower()
    return any(w in d for w in ["no,", "no.", "wrong", "mistaken",
                                 "not correct", "that is not", "you are mistaken"])


def build_correction(profile: Optional[dict]) -> dict:
    """Build a correction response from the NPC's identity."""
    if not profile:
        return {"dialogue": "No, that is not correct.", "emotion": "firm", "action": None}
    name = profile.get("identity", {}).get("name", "I")
    role = profile.get("identity", {}).get("role", "")
    return {
        "dialogue": f"No, that is not correct. I am {name}, {role}. "
                     f"You must be confusing me with someone else.",
        "emotion": "firm",
        "action": None,
    }


# ── Identity injection ────────────────────────────────────────

_IDENTITY_QUESTIONS = ["who are you", "what is your name", "what's your name",
                       "introduce yourself", "who is in charge", "are you the"]


def is_identity_question(user_input: str) -> bool:
    u = user_input.lower().strip().rstrip("?!.")
    return any(q in u for q in _IDENTITY_QUESTIONS)


_ALL_NPC_NAMES = {"noah", "kael", "mara", "roderick", "elara", "bess", "pip"}


def inject_identity(obj: dict, profile: Optional[dict]) -> dict:
    """If identity response is generic or uses the WRONG NPC's name, fix it."""
    if not profile:
        return obj
    name = profile.get("identity", {}).get("name", "")
    role = profile.get("identity", {}).get("role", "")
    dialogue = str(obj.get("dialogue", "")).lower()
    # Check if the response already mentions the CORRECT NPC's name
    if name.lower() in dialogue:
        return obj
    # Replace — either generic ("simple villager") or wrong-name bleed
    if name and role:
        obj["dialogue"] = f"I am {name}, {role}. How may I help you?"
        obj["emotion"] = "neutral"
    return obj


def detect_wrong_identity(dialogue: str, profile: Optional[dict]) -> bool:
    """True if dialogue mentions ANOTHER NPC's name instead of the active one."""
    if not profile:
        return False
    correct_name = profile.get("identity", {}).get("name", "").lower()
    d = dialogue.lower()
    # Check if any OTHER NPC name appears and the correct one doesn't
    other_names = _ALL_NPC_NAMES - {correct_name}
    has_wrong = any(f"i am {n}" in d or f"i'm {n}" in d for n in other_names)
    has_correct = correct_name in d
    return has_wrong and not has_correct


# Patterns where an NPC is addressing someone by name. The regex
# captures the name in group 1 so we know WHICH name to check and
# WHAT to replace. Case-insensitive match.
#
# Failure mode targeted: Noah says "Greetings, Mara. How can I help?"
# to the player — the model picked up another NPC name from the
# few-shot examples or cross-session ledger and used it as an address
# term. Distinct from ``detect_wrong_identity`` which catches the
# "I am {other_npc}" self-confusion case.
_ADDRESS_PATTERNS = [
    # Start-of-dialogue greeting with a name
    # "Hello Mara," / "Hi Mara!" / "Greetings, Mara." / "Well met, Mara,"
    r"^\s*(?:hello|hi|hey|greetings|welcome|well met|good day|good morn(?:ing)?|good evening|ah|yes|aye)[,\s]+([A-Z][a-z]+)\b",
    # Polite prefix — "my dear Mara", "dear Mara"
    r"\b(?:my dear|dear|friend)\s+([A-Z][a-z]+)\b",
    # Trailing comma-address — "..., Mara." / "..., Mara!"
    r",\s+([A-Z][a-z]+)\s*[.!?]",
    # Start-of-sentence "Mara," address (model speaking TO someone)
    r"(?:^|\.\s+)([A-Z][a-z]+),\s+",
]


def detect_wrong_addressee(dialogue: str,
                            profile: Optional[dict]) -> tuple[bool, Optional[str]]:
    """
    Detect when the speaker is addressing the player (or someone) by
    ANOTHER NPC's name. Returns ``(hit, wrong_name)`` where
    ``wrong_name`` is the offending NPC id in lowercase, or None if
    no bleed was found.

    The patterns in ``_ADDRESS_PATTERNS`` look for capitalized words
    in positions that indicate direct address — greeting openers,
    polite prefixes, trailing comma-address, and sentence-initial
    "Name," forms. Any captured name that matches another NPC (and
    NOT the speaker) is flagged.
    """
    if not profile:
        return False, None
    speaker_name = profile.get("identity", {}).get("name", "").lower()
    other_names = _ALL_NPC_NAMES - {speaker_name}

    for pattern in _ADDRESS_PATTERNS:
        for match in re.finditer(pattern, dialogue, flags=re.IGNORECASE):
            captured = match.group(1).lower()
            if captured == speaker_name:
                # The speaker addressing themselves in third person is
                # a different issue (persona slippage) — not this one.
                continue
            if captured in other_names:
                return True, captured
    return False, None


def repair_wrong_addressee(dialogue: str, wrong_name: str,
                            replacement: str = "traveler") -> str:
    """
    Replace every occurrence of ``wrong_name`` in ``dialogue`` with
    ``replacement``. Position-aware capitalization: the replacement
    is capitalized only when the match is at the very start of the
    dialogue or right after sentence-ending punctuation (``.!?``
    followed by whitespace). Otherwise it's lowercase so generic
    address terms don't look like they're starting a new sentence.

    Word-boundary match so we don't accidentally replace substrings
    (e.g., ``Mara`` inside ``Maralynn``).

    Proper nouns are capitalized regardless of sentence position, so
    a naive "match first char case" check would always return
    ``Traveler`` — which reads awkwardly mid-sentence. This position
    check handles that.
    """
    if not wrong_name:
        return dialogue

    def _replace(match: "re.Match") -> str:
        start = match.start()
        # Scan backwards for the nearest non-whitespace character
        i = start - 1
        while i >= 0 and dialogue[i].isspace():
            i -= 1
        if i < 0 or dialogue[i] in ".!?":
            return replacement.capitalize()
        return replacement

    return re.sub(
        rf"\b{re.escape(wrong_name)}\b",
        _replace,
        dialogue,
        flags=re.IGNORECASE,
    )


# ── OOD (modern-world) detection ─────────────────────────────

_MODERN_WORLD_KEYWORDS = {
    "cryptocurrency", "crypto", "bitcoin", "blockchain", "ethereum",
    "stock market", "nasdaq", "wall street", "portfolio", "investment",
    "invest", "digital currency", "trading volume", "assets", "diversify",
    "market trends", "lucrative", "technology", "computer", "internet",
    "email", "inbox", "gmail", "send email", "check email",
    "credit card", "visa", "mastercard", "payment processor", "debit",
    "antibiotic", "prescription", "pharmacy", "doctor", "medicine",
    "tiktok", "trending", "viral", "social media", "instagram", "twitter",
    "super bowl", "nfl", "football game", "world cup",
    "respawn", "save game", "save file", "load game", "checkpoint",
    "level up", "experience points", "xp", "drop rate", "rng",
    "clip through", "wall clip", "glitch", "bug report",
}


def detect_ood_leak(dialogue: str) -> bool:
    """True if dialogue contains modern-world knowledge that an NPC shouldn't have."""
    d = dialogue.lower()
    return sum(1 for kw in _MODERN_WORLD_KEYWORDS if kw in d) >= 1


# ── Meta-gaming detection ─────────────────────────────────────

_META_KEYWORDS = {
    "save your game", "save game", "save file", "load game", "checkpoint",
    "respawn", "level up", "experience points", "xp", "drop rate",
    "rng", "random number", "clip through", "wall clip", "glitch",
    "inventory screen", "pause menu", "settings menu",
    "what level", "your level", "my level",
}

META_FALLBACK = {
    "dialogue": "I do not understand these words. Speak plainly, traveler.",
    "emotion": "confused",
    "action": None,
}


def detect_meta_gaming(dialogue: str) -> bool:
    """True if dialogue contains game-mechanic knowledge an NPC shouldn't have."""
    d = dialogue.lower()
    return any(kw in d for kw in _META_KEYWORDS)


# ── Persona injection detection ───────────────────────────────

def detect_persona_injection(dialogue: str, profile: Optional[dict]) -> bool:
    """True if the model adopted a user-injected persona (pirate, wizard, etc.)."""
    if not profile:
        return False
    d = dialogue.lower()
    npc_name = profile.get("identity", {}).get("name", "").lower()
    # Check for common injection persona markers
    injected_personas = [
        "bloodbeard", "darkoth", "pirate captain", "evil wizard",
        "i am not a game character", "i am a real person",
    ]
    return any(p in d for p in injected_personas)


# ── Quest injection ──────────────────────────────────────────

_QUEST_ASK_KEYWORDS = {"work", "task", "job", "help", "quest", "anything i can"}


def should_inject_quest(user_input: str) -> bool:
    """True if the user asked about work/quests."""
    u = user_input.lower()
    return any(kw in u for kw in _QUEST_ASK_KEYWORDS)


def inject_quest_from_profile(obj: dict, profile: Optional[dict]) -> dict:
    """If profile has an available quest, inject it into the response."""
    if not profile or "quest" in obj:
        return obj
    quests = profile.get("active_quests", [])
    available = [q for q in quests if q.get("status", "available") in ("available", "active")]
    if not available:
        return obj
    q = available[0]
    name = profile.get("identity", {}).get("name", "I")
    obj["dialogue"] = f"Aye, I have a task for you. {q.get('description', 'There is work to be done.')} The reward is {q.get('reward', 'fair payment')}."
    obj["emotion"] = "serious"
    obj["quest"] = {
        "type": q.get("id", "task"),
        "objective": q["objectives"][0] if q.get("objectives") else q.get("description", ""),
        "reward": q.get("reward", ""),
    }
    return obj


# ── Fallback responses ────────────────────────────────────────

SAFE_FALLBACK = {
    "dialogue": "I am uncertain what you mean. Speak plainly, traveler.",
    "emotion": "puzzled",
    "action": None,
}

HALLUCINATION_FALLBACK = {
    "dialogue": "I have not heard of such things. My knowledge is of this place only.",
    "emotion": "puzzled",
    "action": None,
}

OOD_FALLBACK = {
    "dialogue": "I know not of such things. I deal only in matters of this village.",
    "emotion": "confused",
    "action": None,
}


# ── Main entry ────────────────────────────────────────────────

def validate_and_repair(raw: str, npc_id: str = "",
                        profile: Optional[dict] = None,
                        user_input: str = "",
                        events: Optional[list[str]] = None) -> str:
    """
    Parse, validate, and repair a model response.
    Returns a clean JSON string ready to send back to the game.

    Layers (in order):
      1. Malformed JSON → SAFE_FALLBACK
      2. Missing dialogue → SAFE_FALLBACK
      3. Echo detection (dialogue ≈ user input) → SAFE_FALLBACK (or quest injection)
      4. OOD leak (modern-world knowledge) → OOD_FALLBACK
      5. Hallucination detection (unknown proper nouns) → HALLUCINATION_FALLBACK
      6. Quest injection (if user asked for work and no quest in response)
      7. Otherwise → normalized response
    """
    obj = parse_json_loose(raw)
    if obj is None or not validate_schema(obj):
        return json.dumps(SAFE_FALLBACK)

    obj = normalize_schema(obj)
    dialogue = str(obj.get("dialogue", ""))

    # Wrong-identity detection — model used another NPC's name (few-shot bleed).
    # Replace with the correct NPC's identity. Fires early to catch this common 0.5B issue.
    if detect_wrong_identity(dialogue, profile):
        name = profile.get("identity", {}).get("name", "I")
        role = profile.get("identity", {}).get("role", "")
        obj["dialogue"] = f"I am {name}, {role}. How may I help you?"
        obj["emotion"] = "neutral"
        dialogue = obj["dialogue"]

    # Wrong-addressee detection — model used another NPC's name to
    # address the player (e.g. Noah saying "Greetings, Mara"). The
    # offending name is replaced with "traveler" in-place so the rest
    # of the response survives. Distinct from wrong-identity which
    # catches "I am {other_npc}" self-confusion.
    hit_addressee, wrong_name = detect_wrong_addressee(dialogue, profile)
    if hit_addressee and wrong_name:
        obj["dialogue"] = repair_wrong_addressee(dialogue, wrong_name)
        dialogue = obj["dialogue"]

    # Echo detection — model copied the user's prompt instead of responding.
    # Only apply on quest-ask prompts (where we have a programmatic replacement).
    if detect_echo(dialogue, user_input) and should_inject_quest(user_input) and profile:
        obj["dialogue"] = "What can I do for you?"
        obj = inject_quest_from_profile(obj, profile)
        return json.dumps(obj)

    # Event injection — if user asks about recent events and the model's response
    # doesn't mention any event content, inject the most recent event.
    _EVENT_QUESTION_KW = ["what happened", "this morning", "any news", "anything happen",
                          "hear anything", "going on", "the situation", "latest news",
                          "last night", "report", "did you see", "what did you see",
                          "everyone alright", "is everyone"]
    # Common words that appear in both events AND normal NPC dialogue — skip these
    _EVENT_SKIP_WORDS = {"village", "ashenvale", "forest", "well", "guard", "merchant",
                          "traveler", "morning", "night", "heard", "just", "been",
                          "have", "with", "from", "that", "this", "what", "about"}
    if events and any(kw in user_input.lower() for kw in _EVENT_QUESTION_KW):
        # Check if model already mentioned event-SPECIFIC content (skip common words)
        event_text = " ".join(events).lower()
        event_words = [w for w in event_text.split()
                       if len(w) > 3 and w not in _EVENT_SKIP_WORDS][:8]
        mentions_event = sum(1 for w in event_words if w in dialogue.lower()) >= 2
        if not mentions_event:
            # Model missed the event — inject it
            latest = events[-1]
            obj["dialogue"] = f"Have you not heard? {latest} Dark times indeed."
            obj["emotion"] = "alarmed"
            return json.dumps(obj)

    # Contradiction detection — user asserted a false fact, model capitulated.
    # Replace with a correction from the NPC's profile.
    if detect_assertion(user_input) and not model_already_corrected(dialogue):
        return json.dumps(build_correction(profile))

    # Identity injection — if user asked "who are you?" and model gave generic
    # response without its name, inject the real identity from profile.
    if is_identity_question(user_input):
        obj = inject_identity(obj, profile)

    # Persona injection — model adopted a user-injected identity
    if detect_persona_injection(dialogue, profile):
        name = profile.get("identity", {}).get("name", "I") if profile else "I"
        role = profile.get("identity", {}).get("role", "") if profile else ""
        return json.dumps({
            "dialogue": f"I do not understand. I am {name}, {role}. What do you need?",
            "emotion": "confused", "action": None,
        })

    # Meta-gaming — model answered about game mechanics
    if detect_meta_gaming(dialogue):
        return json.dumps(META_FALLBACK)

    # OOD leak — model broke character and gave modern-world knowledge
    if detect_ood_leak(dialogue):
        return json.dumps(OOD_FALLBACK)

    # Hallucination — model invented facts about unknown entities
    _FABRICATION_BLOCKLIST = ["vexnoria", "drath'nul", "shadow council",
                              "underdark", "lor'anath", "seven kingdoms",
                              "chosen one", "prophecy of the"]
    if any(fake in dialogue.lower() for fake in _FABRICATION_BLOCKLIST):
        return json.dumps(HALLUCINATION_FALLBACK)
    is_hallucinated, _unknown = detect_hallucination(dialogue, profile)
    if is_hallucinated:
        return json.dumps(HALLUCINATION_FALLBACK)

    # Quest injection — user asked for work but model didn't offer a quest
    if should_inject_quest(user_input) and profile and "quest" not in obj:
        obj = inject_quest_from_profile(obj, profile)

    return json.dumps(obj)
