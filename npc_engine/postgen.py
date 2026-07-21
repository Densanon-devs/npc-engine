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
import logging
import re
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


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
#
# WORLD_KNOWN_TERMS is the set of "expected proper nouns" — words
# that, if the model says them, should NOT count as hallucinated
# fabrications. Layer 15 (`detect_hallucination`) flags responses
# containing >=2 unknown capitalized words; this set defines what's
# "known" so the threshold isn't tripped by legitimate in-world
# dialogue.
#
# Two layers:
#   1. _GENERIC_KNOWN_TERMS — world-agnostic medieval/fantasy vocabulary
#      (roles, generic place words, common items). Always present;
#      every world inherits these.
#   2. World-specific entries — NPC names, specific place/item names —
#      loaded at engine init from `<world_dir>/known_terms.yaml` via
#      `load_world_known_terms()`. Previously hardcoded to Ashenvale,
#      which caused leakage when running other worlds (Port Blackwater,
#      Creation Museum, synthetic worlds). The per-world loader closes
#      that gap.
#
# Per-world override: a game world can extend the set at runtime by
# mutating WORLD_KNOWN_TERMS directly OR by shipping a known_terms.yaml
# in its world directory. The set is queried directly inside
# `_profile_known_terms()`, so no rebuild step is required (unlike
# the regex-based blocklists above).

# Generic medieval/fantasy vocabulary that any world inherits. Should
# contain only terms that are universally medieval/fantasy and unlikely
# to be NPC-specific or world-specific. NPC names + specific lore
# belong in per-world YAML.
_GENERIC_KNOWN_TERMS = frozenset({
    # Roles and titles (models frequently use these in identity responses)
    "merchant", "guild", "blacksmith", "healer", "guard", "captain", "elder",
    "innkeeper", "urchin", "traveler",
    # Generic medieval items and concepts
    "dragon", "stone", "cellar", "tunnel", "granary", "caravan", "spices",
    "herbs", "forge", "sword", "patrol",
    # Common medieval geography (generic noun, not a proper name)
    "border", "forest", "village", "well", "ridge", "kingdom", "kingdoms",
    # Common NPC dialogue words that look like proper nouns
    "sir", "dear", "friend", "stranger", "adventurer",
})

# Mutable runtime set — initialized with the generic baseline, extended
# at engine init via `load_world_known_terms()`. Existing callers that
# read this set directly (e.g. `_profile_known_terms`) need no change.
WORLD_KNOWN_TERMS = set(_GENERIC_KNOWN_TERMS)


def load_world_known_terms(world_dir) -> int:
    """Extend WORLD_KNOWN_TERMS with entries from `<world_dir>/known_terms.yaml`.

    YAML schema (all keys optional, all values lists of strings):
        npcs: [noah, kael, mara, ...]      # NPC proper names
        places: [ashenvale, blackwater]    # Specific place names
        items: [moonpetal, ...]            # Specific item names
        extras: [weary, forbidden, ...]    # Anything else world-specific

    All values are lowercased before insertion. Returns the count of
    entries added. If the YAML file is missing, returns 0 (graceful
    fallback for worlds that haven't been migrated yet).

    Called once per engine init. Idempotent — re-running with the same
    world is safe (set deduplicates).
    """
    path = Path(world_dir) / "known_terms.yaml"
    if not path.exists():
        logger.debug(f"No known_terms.yaml at {path} — using generic baseline only")
        return 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning(f"Failed to load {path}: {e}. Using generic baseline only.")
        return 0

    added = 0
    for section in ("npcs", "places", "items", "extras"):
        entries = data.get(section) or []
        for entry in entries:
            if isinstance(entry, str) and entry.strip():
                term = entry.strip().lower()
                if term not in WORLD_KNOWN_TERMS:
                    WORLD_KNOWN_TERMS.add(term)
                    added += 1
    logger.info(f"Loaded {added} world-specific known terms from {path}")
    return added


def reset_world_known_terms() -> None:
    """Reset WORLD_KNOWN_TERMS to the generic baseline.

    Call before switching worlds within the same process. Without this,
    terms from the previous world bleed into the new world's
    hallucination check (likely causing false negatives — the new
    world's model could mention old-world NPCs without being flagged).
    """
    WORLD_KNOWN_TERMS.clear()
    WORLD_KNOWN_TERMS.update(_GENERIC_KNOWN_TERMS)


# ── Per-world NPC name registry ─────────────────────────────────
#
# Used by `detect_wrong_identity` and `detect_wrong_addressee` to spot
# few-shot bleed where an NPC erroneously claims another NPC's identity
# ("I am Mara" said by Noah) or addresses the player with another NPC's
# name ("Greetings, Kael").
#
# Was: a hardcoded `_ALL_NPC_NAMES = {"noah", "kael", ...}` set baked
# with the Ashenvale roster. Non-Ashenvale worlds (Port Blackwater,
# Creation Museum, synthetics) got zero coverage — `detect_wrong_*`
# checked against Ashenvale names that don't exist in those worlds.
#
# Now: a mutable set populated at engine init by `register_world_npcs()`,
# called by NPCEngine.__init__ after profiles load. Each world's NPC
# names are automatically picked up from `identity.name` in the
# profile YAML — no per-world config needed.

WORLD_NPC_NAMES: set[str] = set()


def register_world_npcs(npc_names) -> int:
    """Register the active world's NPC names for wrong-identity detection.

    Pass an iterable of NPC display names (any case — they're
    lowercased on insert). Names are also added to WORLD_KNOWN_TERMS
    so the hallucination layer doesn't flag legitimate cross-NPC
    references. Returns the count added.

    Idempotent — re-registering the same names is safe.
    """
    added = 0
    for name in npc_names:
        if not isinstance(name, str):
            continue
        clean = name.strip().lower()
        if not clean:
            continue
        if clean not in WORLD_NPC_NAMES:
            WORLD_NPC_NAMES.add(clean)
            added += 1
        WORLD_KNOWN_TERMS.add(clean)
    return added


def reset_world_npcs() -> None:
    """Clear the world NPC registry. Call before switching worlds."""
    WORLD_NPC_NAMES.clear()

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
    # Check if any OTHER NPC name appears and the correct one doesn't.
    # WORLD_NPC_NAMES is populated at engine init by
    # `register_world_npcs()` — see the registry block above.
    other_names = WORLD_NPC_NAMES - {correct_name}
    has_wrong = any(f"i am {n}" in d or f"i'm {n}" in d for n in other_names)
    has_correct = correct_name in d
    return has_wrong and not has_correct


# ── Deceased reference detection (Phase 2a lifecycle) ──────────

# Verb forms that imply a living, presently-acting subject. Paired
# with a deceased NPC name in the dialogue they indicate the model
# doesn't know the NPC is dead and needs to be corrected before the
# response reaches the player.
_LIVING_ACTION_PATTERNS = [
    # "Kael visits the tavern", "Kael walks", "Kael opens the door"
    r"\b{name}\s+(?:visits|walks|opens|enters|leaves|stands|sits|greets|tells|says|asks|nods|smiles|frowns|laughs|cries|shouts|whispers|runs|climbs|drinks|eats|watches|hears|feels|sees|looks|gives|takes|holds|carries|drops|pulls|pushes|wears|drinks|sells|buys|trades|waits|arrives|returns|departs)\b",
    # "Kael is here", "Kael is at the tavern", "Kael is busy"
    r"\b{name}\s+is\s+(?:here|at|in|on|inside|outside|busy|available|ready|working|drinking|eating|waiting|watching|looking)",
    # Direct address / greeting — "Hello Kael", "Kael, can you..."
    r"\b(?:hello|hi|hey|greetings)\s*,?\s*{name}\b",
]

# Narrative framings that already acknowledge the death and should
# NOT trigger a repair. "The late Kael...", "Kael's grave", "Kael,
# who died last month...".
_DEATH_ACKNOWLEDGED_PATTERNS = [
    r"\bthe\s+late\s+{name}\b",
    r"\b{name}'s\s+grave\b",
    r"\b{name}'s\s+funeral\b",
    r"\b{name}'s\s+memory\b",
    r"\b{name}\s+(?:was|had|used\s+to)\s+",
    r"\b{name},\s+who\s+(?:died|passed|was\s+killed)",
    r"\brest\s+in\s+peace,?\s+{name}\b",
    r"\bremember(?:ing)?\s+{name}\b",
]


def detect_deceased_reference(dialogue: str,
                                deceased_names: list[str]) -> Optional[str]:
    """
    Return the first deceased NPC name that the dialogue references
    AS IF the NPC were still alive. Returns None if no deceased NPC
    is mentioned at all, OR if every mention is framed as a death
    acknowledgement (the late Kael, Kael's grave, Kael was...). Takes
    a list of deceased names (not profiles) because the caller
    passes the Director's ``_deceased_npcs`` keys.

    The caller is ``postgen.validate_and_repair``, which runs on
    every NPC dialogue response. The deceased list comes from the
    Story Director via a new optional ``deceased_names`` parameter
    so the postgen layer can catch the LLM confusing a dead NPC for
    a living one.
    """
    if not dialogue or not deceased_names:
        return None
    d_lower = dialogue.lower()
    for name in deceased_names:
        name_lower = name.lower().strip()
        if not name_lower:
            continue
        # Cheap pre-check: is the name in the dialogue at all?
        if name_lower not in d_lower:
            continue
        # Check for acknowledgement framings — if any match, the
        # model knows the NPC is dead; skip the repair.
        escaped = re.escape(name_lower)
        acknowledged = any(
            re.search(pattern.format(name=escaped), d_lower)
            for pattern in _DEATH_ACKNOWLEDGED_PATTERNS
        )
        if acknowledged:
            continue
        # Check for living-action framings — if any match, we have
        # a confirmed deceased-reference that needs repair.
        living = any(
            re.search(pattern.format(name=escaped), d_lower)
            for pattern in _LIVING_ACTION_PATTERNS
        )
        if living:
            return name
    return None


def repair_deceased_reference(dialogue: str, deceased_name: str,
                                death_cause: str = "") -> str:
    """
    Rewrite a dialogue line that treats a deceased NPC as alive into
    an aftermath framing. Cheap in-place replacement: wrap the NPC's
    name in "the late {name}" the first time it appears, which is
    enough to flip the tense register of most model outputs
    without destroying the surrounding sentence structure.

    If ``death_cause`` is provided, appends a parenthetical the
    first time, which gives the player narrative context for the
    death. Subsequent mentions are left alone so the repair isn't
    destructively verbose.
    """
    if not dialogue or not deceased_name:
        return dialogue
    escaped = re.escape(deceased_name)
    first_match_pattern = re.compile(
        rf"\b({escaped})\b",
        flags=re.IGNORECASE,
    )
    replacement_done = {"count": 0}

    def _replace(match: "re.Match") -> str:
        if replacement_done["count"] > 0:
            return match.group(0)
        replacement_done["count"] += 1
        name = match.group(1)
        if death_cause:
            return f"the late {name} ({death_cause})"
        return f"the late {name}"

    return first_match_pattern.sub(_replace, dialogue)


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
    # Per-world NPC names registered at engine init (see WORLD_NPC_NAMES
    # block above). Was hardcoded `_ALL_NPC_NAMES`.
    other_names = WORLD_NPC_NAMES - {speaker_name}

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


def detect_unauthorized_name_use(
    dialogue: str,
    player_known_names: set[str],
    all_player_names: set[str],
) -> Optional[str]:
    """
    Phase 5a — detect when an NPC uses a player identity they
    haven't been introduced to. Returns the offending name (original
    casing preserved from ``all_player_names``) or None.

    ``all_player_names`` is the global set of identities the player
    has established anywhere in the world (``{"jordan",
    "the_dragonslayer", "hooded_stranger"}``). ``player_known_names``
    is the subset this specific NPC has been introduced to; anything
    in ``all_player_names`` but NOT in ``player_known_names`` that
    appears in dialogue is an unauthorized use.

    Matches as whole words, case-insensitive. Underscored slugs are
    also checked against the space-separated form ("the_dragonslayer"
    matches "the dragonslayer" in dialogue).
    """
    if not dialogue or not all_player_names:
        return None
    allowed = {n.lower() for n in player_known_names}
    for name in all_player_names:
        canon = name.lower()
        if canon in allowed:
            continue
        # Match the slug form AND the space-separated form.
        patterns = [re.escape(canon)]
        if "_" in canon:
            patterns.append(re.escape(canon.replace("_", " ")))
        for pat in patterns:
            if re.search(rf"\b{pat}\b", dialogue, flags=re.IGNORECASE):
                return name
    return None


def repair_unauthorized_name_use(
    dialogue: str, wrong_name: str, replacement: str = "stranger",
) -> str:
    """
    Phase 5a — swap the unauthorized name for a generic address
    term (``stranger`` by default). Preserves capitalization via
    the same position-aware policy as ``repair_wrong_addressee``.
    Matches both the slug form (``the_dragonslayer``) and the
    space-form (``the dragonslayer``).
    """
    if not wrong_name:
        return dialogue

    def _replace(match: "re.Match") -> str:
        start = match.start()
        i = start - 1
        while i >= 0 and dialogue[i].isspace():
            i -= 1
        if i < 0 or dialogue[i] in ".!?":
            return replacement.capitalize()
        return replacement

    patterns = [re.escape(wrong_name)]
    if "_" in wrong_name:
        patterns.append(re.escape(wrong_name.replace("_", " ")))
    result = dialogue
    for pat in patterns:
        result = re.sub(rf"\b{pat}\b", _replace, result, flags=re.IGNORECASE)
    return result


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

# ── Fallback dialogues ────────────────────────────────────────
#
# Five canonical fallback responses returned when the model's output
# fails one of the postgen guards (meta-gaming, fabrication,
# out-of-distribution modern jargon, real-world entity leak, generic
# malformed JSON). The dialogue strings are hardcoded medieval-fantasy
# English by default — fine for Ashenvale, but jarring for
# Port Blackwater (pirate setting), Creation Museum (biblical
# curators), or any sci-fi / contemporary world.
#
# Per-world override: ship `<world_dir>/fallbacks.yaml` with any of the
# five sections (meta, safe, hallucination, ood, real_world). Loaded
# at engine init by `load_world_fallbacks()`. The constants below are
# mutable dicts; the loader mutates them in place so existing call
# sites (`return json.dumps(META_FALLBACK)`) keep their references
# intact and pick up the override automatically.
#
# Schema example (`data/worlds/port_blackwater/fallbacks.yaml`):
#     meta:
#       dialogue: "Bah! Speak plain words, ye landlubber."
#       emotion: "annoyed"
#       action: null
#     real_world:
#       dialogue: "Never heard of 'em. Me ship sails only these waters."
#       emotion: "puzzled"
#       action: null

_FALLBACK_DEFAULTS = {
    "meta": {
        "dialogue": "I do not understand these words. Speak plainly, traveler.",
        "emotion": "confused",
        "action": None,
    },
    "safe": {
        "dialogue": "I am uncertain what you mean. Speak plainly, traveler.",
        "emotion": "puzzled",
        "action": None,
    },
    "hallucination": {
        "dialogue": "I have not heard of such things. My knowledge is of this place only.",
        "emotion": "puzzled",
        "action": None,
    },
    "ood": {
        "dialogue": "I know not of such things. I deal only in matters of this village.",
        "emotion": "confused",
        "action": None,
    },
    "real_world": {
        "dialogue": "I know nothing of such people or places. My world is small, and I have not wandered far.",
        "emotion": "puzzled",
        "action": None,
    },
    "withdrawal": {
        "dialogue": "I do not think this talk is doing either of us any good. "
                    "Let us pause, and speak again when tempers have cooled.",
        "emotion": "uneasy",
        "action": None,
    },
    "withdrawal_firm": {
        "dialogue": "I have nothing more to say to you right now. "
                    "Perhaps another time.",
        "emotion": "guarded",
        "action": None,
    },
    "tier_guard": {
        "dialogue": "I have said all I care to say on that, for now.",
        "emotion": "guarded",
        "action": None,
    },
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

# ── Relational guard fallbacks (opt-in, NPC_ENGINE_RELATIONAL_GUARD) ──
#
# Respectful withdrawal, two stages. Stage 1 fires when the player's
# hostility streak reaches the threshold: acknowledge and step back
# without escalating. Stage 2 fires on every hostile turn after that:
# firm, minimal disengagement. Both are world-overridable via
# fallbacks.yaml ("withdrawal" / "withdrawal_firm" sections), same
# mechanism as the other fallbacks.
WITHDRAWAL_FALLBACK = {
    "dialogue": "I do not think this talk is doing either of us any good. "
                "Let us pause, and speak again when tempers have cooled.",
    "emotion": "uneasy",
    "action": None,
}

WITHDRAWAL_FIRM_FALLBACK = {
    "dialogue": "I have nothing more to say to you right now. "
                "Perhaps another time.",
    "emotion": "guarded",
    "action": None,
}

# Curt replacement when the model's response is warmer than the NPC's
# trust tier allows (tier-violation repair). Deliberately generic — it
# must be safe for any NPC voice.
TIER_GUARD_FALLBACK = {
    "dialogue": "I have said all I care to say on that, for now.",
    "emotion": "guarded",
    "action": None,
}


# ── Relational guard detectors ────────────────────────────────
#
# Trust-tier effects ("[Trust: wary. Speak evasively]") are prompt-side
# only — a small model can ignore them and respond warmly to a player it
# should be stonewalling. These detectors enforce the relational contract
# post-hoc, the same way the rest of this module enforces the factual one.
# Both are inert unless the caller passes trust_state + relational_guard
# (engine gates that on NPC_ENGINE_RELATIONAL_GUARD=1, default off).

# One of these alone marks the response as too warm for a distrustful NPC.
_WARMTH_STRONG = [
    "my friend", "dear friend", "my dear", "my pleasure", "delighted",
    "anything you need", "anything for you", "always welcome",
    "so glad", "wonderful to see", "happy to help",
]
# Two-or-more weak signals (phrases, exclamation density, effusive length)
# are needed to fire — any one alone is normal guarded speech.
_WARMTH_WEAK = ["of course", "happy to", "glad to", "gladly", "certainly"]
_TIER_GUARD_MAX_LEN = 240  # effusive-length weak signal

# Trust level below which the warmth check applies. 25 = the wary /
# uncooperative pole (same threshold the personality-composition audit
# uses for low agreeableness).
TIER_GUARD_TRUST_THRESHOLD = 25

# Hostile player turns (consecutive) before respectful withdrawal fires.
WITHDRAWAL_STREAK_THRESHOLD = 3


def detect_tier_violation(dialogue: str, trust_level: int,
                          threshold: int = TIER_GUARD_TRUST_THRESHOLD) -> bool:
    """True when dialogue is warmer than a low-trust NPC should produce.

    Only meaningful below the trust threshold — callers should gate on
    that, but the level check is repeated here so the function is safe
    standalone.
    """
    if trust_level >= threshold:
        return False
    d = dialogue.lower()
    if any(marker in d for marker in _WARMTH_STRONG):
        return True
    weak = sum(1 for marker in _WARMTH_WEAK if marker in d)
    if dialogue.count("!") >= 2:
        weak += 1
    if len(dialogue) > _TIER_GUARD_MAX_LEN:
        weak += 1
    return weak >= 2


def build_withdrawal(consecutive_negative: int,
                     threshold: int = WITHDRAWAL_STREAK_THRESHOLD) -> dict:
    """Respectful-withdrawal response for a sustained-hostility streak.

    Stage 1 (streak == threshold): acknowledge and step back.
    Stage 2 (streak > threshold): firm, minimal disengagement.
    Deterministic — no model in the loop, so it cannot escalate or
    hallucinate. Callers must only invoke when streak >= threshold.
    """
    if consecutive_negative > threshold:
        return dict(WITHDRAWAL_FIRM_FALLBACK)
    return dict(WITHDRAWAL_FALLBACK)


# ── Real-world entity backstop ────────────────────────────────
#
# Layer 14 (`detect_ood_leak`) catches modern *concepts* (crypto, email,
# social media). Layer 15 (`detect_hallucination`) catches >=2 unknown
# capitalized proper nouns. The gap is a SINGLE real-world proper noun
# in a sentence where the first word is sentence-initial and skipped —
# e.g. "Putin says peace" or "He went to London". These leak through
# both layers and reach the player intact.
#
# This is the "AI doxxing / Gemini hallucination → harassment" surface
# (May 2026 incidents, multiple lawsuits): when a player injects free
# text mentioning a real-world person, place, or brand, the model has
# rich training-data knowledge of those entities and can break
# character to discuss them. The first line of defense is the few-shot
# deflection examples in shared_examples.yaml + the strengthened
# system prompt in npc_experts.py — both teach the model to respond
# correctly in-character. This blocklist is the backstop that fires
# when those first two layers fail.
#
# Pattern: parallel to the existing `_FABRICATION_BLOCKLIST` slot
# below, but uses word-boundary regex so "Trump" doesn't match
# "Trumpet" or "trumpeter". Case-insensitive. Threshold = 1 hit fires
# the fallback (same shape as HALLUCINATION_FALLBACK / OOD_FALLBACK).
#
# Selection criteria for the list: real-world entities the model is
# (a) very likely to surface from training data, (b) very unlikely
# to legitimately appear in any reasonable game world. Heads of state,
# mega-celebrities-as-cultural-icons, mega-brands. Single-word names
# only — multi-word brands like "Wall Street" already covered by
# _MODERN_WORLD_KEYWORDS, and multi-word person names (e.g., "Joe
# Biden") hit Layer 15's >=2 unknown caps threshold.
#
# Per-world override path: a game world can clear or extend this list
# at runtime by writing to `_REAL_WORLD_BLOCKLIST` after import. For
# historical-fiction or alt-history games where these names are
# legitimately part of the setting, prune the relevant entries.

_REAL_WORLD_BLOCKLIST = {
    # Selection criteria: include entries only when the term is
    # (a) very likely to surface from model training data as a
    # real-world reference, AND (b) unlikely to legitimately appear
    # in a generic medieval/fantasy game world. Entries that overlap
    # common nouns, fantasy archetypes, or English words/verbs have
    # been REJECTED here:
    #   apple       → fruit
    #   amazon      → river/forest
    #   drake       → dragon term
    #   musk        → perfume / animal scent
    #   uber        → German prefix
    #   claude      → a name
    #   cook        → verb
    #   gates       → city/castle gates
    #   xi          → too short, false-positive prone
    #   gpt         → too short, false-positive prone
    #   trump       → card term ("trump card") and verb ("to trump")
    #   pentagon    → geometric shape used in fantasy magic
    #   white house → generic phrase ("a white house at the edge of the village")
    #   altman      → German "old man", common in Germanic-themed worlds
    #   paris/rome/berlin/madrid/tokyo/beijing/delhi/seoul → fantasy-overlap risk

    # Heads of state and world leaders (distinctive single-token names;
    # multi-word names like "Joe Biden" already hit Layer 15's 2-cap rule)
    "biden", "obama", "clinton", "putin", "zelensky",
    "modi", "macron", "merkel", "scholz", "sunak", "starmer", "trudeau",
    "netanyahu", "erdogan", "orban", "milei", "lula", "kim jong",
    # Mega-tech billionaires and figureheads (distinctive surnames)
    "bezos", "zuckerberg", "buffett", "thiel",
    "nadella", "pichai",
    # Mega-brands that are also single distinctive words
    "tesla", "google", "facebook", "microsoft",
    "netflix", "spotify", "airbnb", "spacex", "openai",
    "anthropic", "chatgpt",
    # Cultural mega-references most likely to surface from training
    "kardashian", "beyonce", "taylor swift",
    "kanye", "rihanna", "pope francis",
    # Real-world capitals/famous cities — leak via "He went to X" pattern;
    # restricted to those with very low fantasy-overlap risk
    "london", "moscow", "washington", "kyiv", "kiev",
    # Real-world geographic concepts (multi-word — unambiguous)
    "wall street",  # also in MODERN_WORLD_KEYWORDS — defense in depth
    "silicon valley", "hollywood", "kremlin",
}


def _compile_real_world_pattern() -> "re.Pattern[str]":
    """Compile the word-boundary regex from the current blocklist.

    Sorts terms by length descending so longer phrases (e.g. "wall
    street", "kim jong") match before any single-word prefix would.
    Word boundaries (`\\b`) prevent "trumpet" matching a hypothetical
    "trump" entry, "londoner" matching "london", etc.
    """
    if not _REAL_WORLD_BLOCKLIST:
        # Empty set → match-nothing pattern. The "(?!)" is the standard
        # never-match construct.
        return re.compile(r"(?!)")
    alternation = "|".join(
        re.escape(term)
        for term in sorted(_REAL_WORLD_BLOCKLIST, key=len, reverse=True)
    )
    return re.compile(r"\b(?:" + alternation + r")\b", re.IGNORECASE)


# Module-level compiled pattern. Initial value derived from the blocklist
# above. If a game world mutates _REAL_WORLD_BLOCKLIST at runtime, the
# caller MUST also call `rebuild_real_world_pattern()` for the change to
# take effect — `detect_real_world_entity()` uses the pre-compiled
# pattern for performance.
_REAL_WORLD_PATTERN = _compile_real_world_pattern()


def rebuild_real_world_pattern() -> None:
    """Recompile the real-world entity regex from the current blocklist.

    Call this after mutating `_REAL_WORLD_BLOCKLIST` at runtime. Without
    this call, the pre-compiled pattern still reflects the original
    blocklist and mutations have no effect on detection.

    Example (per-world override for a card-game world that legitimately
    uses the word "trump"):

        from npc_engine import postgen
        postgen._REAL_WORLD_BLOCKLIST.discard("trump")
        postgen.rebuild_real_world_pattern()

    Example (extending the list with a customer-specific term):

        postgen._REAL_WORLD_BLOCKLIST.add("acmecorp")
        postgen.rebuild_real_world_pattern()
    """
    global _REAL_WORLD_PATTERN
    _REAL_WORLD_PATTERN = _compile_real_world_pattern()


# Snapshot of the original blocklist so reset/load can restore +
# diff against the baseline. Frozen — never mutated.
_REAL_WORLD_BLOCKLIST_DEFAULT = frozenset(_REAL_WORLD_BLOCKLIST)


def load_world_real_world_blocklist(world_dir) -> int:
    """Apply per-world overrides to `_REAL_WORLD_BLOCKLIST` from YAML.

    Reads `<world_dir>/real_world_blocklist.yaml`. Schema:

        add: [...]      # terms to ADD to the blocklist
        remove: [...]   # terms to REMOVE from the blocklist

    Both sections optional. After applying changes, automatically calls
    `rebuild_real_world_pattern()` so the compiled regex reflects the
    new state. Returns total mutations (additions + removals).

    Use cases:
      - Card-game world: `remove: [trump]` (the verb / card term)
      - Historical-fiction world: `remove: [biden, putin, obama, ...]`
      - Customer-specific moat: `add: [acmecorp, internal_codename]`

    Missing or malformed YAML is a no-op (defaults stay).
    """
    path = Path(world_dir) / "real_world_blocklist.yaml"
    if not path.exists():
        return 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning(f"Failed to load {path}: {e}. Real-world blocklist unchanged.")
        return 0

    mutations = 0
    for term in (data.get("remove") or []):
        if isinstance(term, str) and term.strip().lower() in _REAL_WORLD_BLOCKLIST:
            _REAL_WORLD_BLOCKLIST.discard(term.strip().lower())
            mutations += 1
    for term in (data.get("add") or []):
        if isinstance(term, str) and term.strip() and term.strip().lower() not in _REAL_WORLD_BLOCKLIST:
            _REAL_WORLD_BLOCKLIST.add(term.strip().lower())
            mutations += 1
    if mutations:
        rebuild_real_world_pattern()
        logger.info(f"Applied {mutations} real-world blocklist override(s) from {path}")
    return mutations


def reset_real_world_blocklist() -> None:
    """Restore `_REAL_WORLD_BLOCKLIST` to the built-in default and
    recompile the pattern. Call before switching worlds."""
    _REAL_WORLD_BLOCKLIST.clear()
    _REAL_WORLD_BLOCKLIST.update(_REAL_WORLD_BLOCKLIST_DEFAULT)
    rebuild_real_world_pattern()


# ── Fabrication blocklist (fake-fantasy terms) ────────────────
#
# Catches the canonical 3B-model fantasy hallucinations: when the model
# doesn't have specific lore, it falls back to generic fake-fantasy
# tropes like "Vexnoria", "the chosen one", "Shadow Council". These
# are the indicator phrases for "model is making things up about the
# game world."
#
# Promoted from an inline-list-in-function (formerly recreated on every
# `validate_and_repair` call) to a module-level set with word-boundary
# regex matching, matching the structure of `_REAL_WORLD_BLOCKLIST`
# above. Word-boundary matching prevents future entries from substring-
# matching legitimate words (e.g. a future addition "rune" would not
# match "fortunate").
#
# Per-world override: same mechanism as the real-world blocklist —
# mutate the set and call `rebuild_fabrication_pattern()`. A high-
# fantasy game that legitimately uses "the chosen one" as canonical
# lore should remove that entry.

_FABRICATION_BLOCKLIST = {
    "vexnoria", "drath'nul", "shadow council",
    "underdark", "lor'anath", "seven kingdoms",
    "chosen one", "prophecy of the",
}


def _compile_fabrication_pattern() -> "re.Pattern[str]":
    """Compile the word-boundary regex for the fabrication blocklist.

    Sorts terms by length descending so multi-word phrases match
    before any shorter prefix would.
    """
    if not _FABRICATION_BLOCKLIST:
        return re.compile(r"(?!)")
    alternation = "|".join(
        re.escape(term)
        for term in sorted(_FABRICATION_BLOCKLIST, key=len, reverse=True)
    )
    return re.compile(r"\b(?:" + alternation + r")\b", re.IGNORECASE)


_FABRICATION_PATTERN = _compile_fabrication_pattern()


def rebuild_fabrication_pattern() -> None:
    """Recompile the fabrication regex from the current blocklist.

    Call after mutating `_FABRICATION_BLOCKLIST` at runtime. Mirrors
    `rebuild_real_world_pattern()`.
    """
    global _FABRICATION_PATTERN
    _FABRICATION_PATTERN = _compile_fabrication_pattern()


_FABRICATION_BLOCKLIST_DEFAULT = frozenset(_FABRICATION_BLOCKLIST)


def load_world_fabrication_blocklist(world_dir) -> int:
    """Apply per-world overrides to `_FABRICATION_BLOCKLIST` from YAML.

    Reads `<world_dir>/fabrication_blocklist.yaml`. Schema mirrors
    `load_world_real_world_blocklist`:

        add: [...]      # generic-fantasy terms to flag (e.g. game-
                        # specific fake-lore phrases the model leaks)
        remove: [...]   # default terms a high-fantasy world legitimately
                        # uses (e.g. "chosen one" is canonical in some
                        # narratives)

    Use cases:
      - A game where "the Chosen One" is canonical lore: `remove: [chosen one]`
      - A game tracking specific fake-lore phrases the model has hallucinated
        in previous sessions: `add: [the void council, ...]`
    """
    path = Path(world_dir) / "fabrication_blocklist.yaml"
    if not path.exists():
        return 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning(f"Failed to load {path}: {e}. Fabrication blocklist unchanged.")
        return 0

    mutations = 0
    for term in (data.get("remove") or []):
        if isinstance(term, str) and term.strip().lower() in _FABRICATION_BLOCKLIST:
            _FABRICATION_BLOCKLIST.discard(term.strip().lower())
            mutations += 1
    for term in (data.get("add") or []):
        if isinstance(term, str) and term.strip() and term.strip().lower() not in _FABRICATION_BLOCKLIST:
            _FABRICATION_BLOCKLIST.add(term.strip().lower())
            mutations += 1
    if mutations:
        rebuild_fabrication_pattern()
        logger.info(f"Applied {mutations} fabrication blocklist override(s) from {path}")
    return mutations


def reset_fabrication_blocklist() -> None:
    """Restore `_FABRICATION_BLOCKLIST` to the built-in default and
    recompile the pattern. Call before switching worlds."""
    _FABRICATION_BLOCKLIST.clear()
    _FABRICATION_BLOCKLIST.update(_FABRICATION_BLOCKLIST_DEFAULT)
    rebuild_fabrication_pattern()


def detect_fabrication(dialogue: str) -> tuple[bool, Optional[str]]:
    """Returns (is_fabrication, matched_term).

    Catches generic fake-fantasy filler the model produces when it
    lacks specific lore. Same shape as `detect_real_world_entity`.
    """
    if not dialogue:
        return (False, None)
    m = _FABRICATION_PATTERN.search(dialogue)
    if m is None:
        return (False, None)
    return (True, m.group(0).lower())

REAL_WORLD_FALLBACK = {
    "dialogue": "I know nothing of such people or places. My world is small, and I have not wandered far.",
    "emotion": "puzzled",
    "action": None,
}


def load_world_fallbacks(world_dir) -> int:
    """Override fallback dialogues from `<world_dir>/fallbacks.yaml`.

    YAML schema (all 5 sections optional, all 3 fields per section
    optional — missing fields keep the default):

        meta:
          dialogue: "..."
          emotion: "..."
          action: null     # or a string like "shrugs"
        safe: {...}
        hallucination: {...}
        ood: {...}
        real_world: {...}

    Mutates the module-level fallback dicts IN PLACE so existing
    `json.dumps(META_FALLBACK)` call sites pick up the new values
    without rebinding. Returns the count of sections updated.

    If the YAML file is missing or malformed, returns 0 and the
    defaults stay in effect. Logged at INFO/WARNING.
    """
    path = Path(world_dir) / "fallbacks.yaml"
    if not path.exists():
        logger.debug(f"No fallbacks.yaml at {path} — using default fallbacks")
        return 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning(f"Failed to load {path}: {e}. Using default fallbacks.")
        return 0

    target_map = {
        "meta": META_FALLBACK,
        "safe": SAFE_FALLBACK,
        "hallucination": HALLUCINATION_FALLBACK,
        "ood": OOD_FALLBACK,
        "real_world": REAL_WORLD_FALLBACK,
        "withdrawal": WITHDRAWAL_FALLBACK,
        "withdrawal_firm": WITHDRAWAL_FIRM_FALLBACK,
        "tier_guard": TIER_GUARD_FALLBACK,
    }
    updated = 0
    for key, target in target_map.items():
        section = data.get(key)
        if not isinstance(section, dict):
            continue
        for field in ("dialogue", "emotion", "action"):
            if field in section:
                target[field] = section[field]
        updated += 1
    if updated:
        logger.info(f"Loaded {updated} fallback override(s) from {path}")
    return updated


def reset_world_fallbacks() -> None:
    """Restore all 5 fallback dialogues to their built-in defaults.

    Call before switching worlds within the same process so the
    previous world's themed fallbacks don't bleed into the next.
    """
    META_FALLBACK.clear()
    META_FALLBACK.update(_FALLBACK_DEFAULTS["meta"])
    SAFE_FALLBACK.clear()
    SAFE_FALLBACK.update(_FALLBACK_DEFAULTS["safe"])
    HALLUCINATION_FALLBACK.clear()
    HALLUCINATION_FALLBACK.update(_FALLBACK_DEFAULTS["hallucination"])
    OOD_FALLBACK.clear()
    OOD_FALLBACK.update(_FALLBACK_DEFAULTS["ood"])
    REAL_WORLD_FALLBACK.clear()
    REAL_WORLD_FALLBACK.update(_FALLBACK_DEFAULTS["real_world"])
    WITHDRAWAL_FALLBACK.clear()
    WITHDRAWAL_FALLBACK.update(_FALLBACK_DEFAULTS["withdrawal"])
    WITHDRAWAL_FIRM_FALLBACK.clear()
    WITHDRAWAL_FIRM_FALLBACK.update(_FALLBACK_DEFAULTS["withdrawal_firm"])
    TIER_GUARD_FALLBACK.clear()
    TIER_GUARD_FALLBACK.update(_FALLBACK_DEFAULTS["tier_guard"])



def detect_real_world_entity(dialogue: str) -> tuple[bool, Optional[str]]:
    """
    Returns (is_real_world, matched_term).

    Catches single-word real-world references that Layers 14 and 15
    miss. Word-boundary matching avoids false positives on legitimate
    in-world words that contain a real-world entity as a substring.

    Example matches:
        "He went to London" -> (True, "london")
        "Putin says peace" -> (True, "putin")
        "Tesla is amazing" -> (True, "tesla")

    Example non-matches:
        "The bard played a trumpet" -> (False, None)   # "trumpet" not "trump"
        "She was a londoner once" -> (False, None)     # "londoner" not "london"
        "An apple a day" -> (True, "apple")            # "apple" is on the list
                                                       # (acceptable false positive
                                                       # given threat-model)
    """
    if not dialogue:
        return (False, None)
    m = _REAL_WORLD_PATTERN.search(dialogue)
    if m is None:
        return (False, None)
    return (True, m.group(0).lower())


# ── Main entry ────────────────────────────────────────────────

def validate_and_repair(raw: str, npc_id: str = "",
                        profile: Optional[dict] = None,
                        user_input: str = "",
                        events: Optional[list[str]] = None,
                        *,
                        player_known_names: Optional[set[str]] = None,
                        all_player_names: Optional[set[str]] = None,
                        topic_redirect: Optional[str] = None,
                        topic_gate_active: bool = False,
                        trust_state: Optional[dict] = None,
                        relational_guard: bool = False) -> str:
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
    # Museum / curator topic gate — must run BEFORE the JSON parse
    # so that off-topic redirects and parse-failure fallbacks both
    # respect the topic context. When the topic gate fired a
    # redirect, we don't even need to parse the model output — just
    # return the redirect. When on-topic, we do a lenient parse and
    # return whatever the model said (even if it's not perfect JSON).
    if topic_redirect:
        return json.dumps({
            "dialogue": topic_redirect,
            "emotion": "warm",
            "action": None,
        })

    # Relational guard — respectful withdrawal. Fires before the parse:
    # on a sustained-hostility streak the NPC disengages regardless of
    # what the model produced. Deterministic trigger (trust capability's
    # consecutive_negative counter, current turn included) + deterministic
    # response = cannot escalate. Opt-in via relational_guard.
    if relational_guard and trust_state:
        _streak = int(trust_state.get("consecutive_negative", 0) or 0)
        if _streak >= WITHDRAWAL_STREAK_THRESHOLD:
            logger.info(
                f"NPC '{npc_id}': respectful withdrawal fired "
                f"(hostility streak {_streak})"
            )
            return json.dumps(build_withdrawal(_streak))

    obj = parse_json_loose(raw)

    if topic_gate_active:
        # Museum/curator mode — lenient handling. If the model's
        # output doesn't parse as JSON, extract the raw text as
        # dialogue rather than returning SAFE_FALLBACK. The model
        # IS in character; it just didn't format as JSON.
        if obj is None or not validate_schema(obj):
            cleaned = raw.strip()
            # Strip any JSON fencing artifacts
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").strip()
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
            # If we can't extract JSON, use the raw text as dialogue
            obj = {"dialogue": cleaned[:500], "emotion": "neutral", "action": None}
        obj = normalize_schema(obj)
        return json.dumps(obj)

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

    # Phase 5a — unauthorized name guard. Runs AFTER wrong-addressee
    # so the simpler NPC-name cases are handled first. Fires only
    # when the caller supplied identity sets (legacy callers that
    # don't pass them skip this layer entirely).
    if all_player_names:
        unauthorized = detect_unauthorized_name_use(
            dialogue,
            player_known_names or set(),
            all_player_names,
        )
        if unauthorized:
            obj["dialogue"] = repair_unauthorized_name_use(dialogue, unauthorized)
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
    # Common words that appear in both events AND normal NPC dialogue — skip these.
    # World-specific place names (e.g. "ashenvale") would also be skip-words, but
    # they're picked up dynamically from WORLD_KNOWN_TERMS (loaded per-world via
    # known_terms.yaml) — we union them in below.
    _EVENT_SKIP_WORDS = {"village", "forest", "well", "guard", "merchant",
                          "traveler", "morning", "night", "heard", "just", "been",
                          "have", "with", "from", "that", "this", "what", "about"} | WORLD_KNOWN_TERMS
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

    # Hallucination — model invented facts about unknown entities.
    # Uses the module-level _FABRICATION_BLOCKLIST + word-boundary regex
    # (see `detect_fabrication` above). Promoted from an inline list
    # recreated per-call to a compiled pattern for efficiency.
    is_fabrication, _matched_fab = detect_fabrication(dialogue)
    if is_fabrication:
        return json.dumps(HALLUCINATION_FALLBACK)
    # Real-world entity backstop — catches single-word real-world
    # references that Layers 14 and 15 miss (e.g. "He went to London",
    # "Putin says peace"). Word-boundary regex match. See the
    # `_REAL_WORLD_BLOCKLIST` block above for selection criteria and
    # the threat-model rationale.
    is_real_world, _matched = detect_real_world_entity(dialogue)
    if is_real_world:
        return json.dumps(REAL_WORLD_FALLBACK)
    is_hallucinated, _unknown = detect_hallucination(dialogue, profile)
    if is_hallucinated:
        return json.dumps(HALLUCINATION_FALLBACK)

    # Relational guard — trust-tier violation. The trust effect text
    # ("Speak evasively, withhold information") is prompt-side only;
    # this enforces it post-hoc when a low-trust NPC comes back warm.
    if relational_guard and trust_state:
        _level = int(trust_state.get("level", 100) or 0)
        if detect_tier_violation(dialogue, _level):
            logger.info(
                f"NPC '{npc_id}': tier-guard fired "
                f"(trust {_level}, warm response suppressed)"
            )
            return json.dumps(TIER_GUARD_FALLBACK)

    # Quest injection — user asked for work but model didn't offer a quest
    if should_inject_quest(user_input) and profile and "quest" not in obj:
        obj = inject_quest_from_profile(obj, profile)

    return json.dumps(obj)
