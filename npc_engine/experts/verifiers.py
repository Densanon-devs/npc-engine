"""
NPC dialogue verifiers for the expert system's verify/retry loop.
"""

import json
import re


def verify_npc_dialogue(response: str, query: str) -> tuple[bool, str]:
    """Verify NPC dialogue stays in character and meets format requirements."""
    r = response.strip()

    # Try to parse as JSON first (preferred output)
    json_obj = None
    try:
        json_obj = json.loads(r)
    except Exception:
        for sc in ["{", "["]:
            idx = r.find(sc)
            if idx >= 0:
                ec = "}" if sc == "{" else "]"
                for ei in range(len(r) - 1, idx, -1):
                    if r[ei] == ec:
                        try:
                            json_obj = json.loads(r[idx:ei + 1])
                            break
                        except Exception:
                            continue
                if json_obj:
                    break

    if json_obj and isinstance(json_obj, dict):
        if "dialogue" not in json_obj:
            return False, "JSON must contain a 'dialogue' field with the NPC's speech."
        dialogue_text = json_obj.get("dialogue", "")
    else:
        dialogue_text = r

    # Character-breaking checks
    lower = dialogue_text.lower()
    ai_breaks = ["i'm an ai", "as an ai", "i am an ai", "language model", "i'm a chatbot",
                  "i cannot", "i don't have feelings", "as a large"]
    for phrase in ai_breaks:
        if phrase in lower:
            return False, "Stay in character. Do not break the fourth wall."

    sentences = [s.strip() for s in re.split(r'[.!?]+', dialogue_text) if s.strip()]
    if len(sentences) > 8:
        return False, "Keep dialogue concise — 2-4 sentences is ideal for NPC speech."

    if len(dialogue_text) < 5:
        return False, "Response too short. Give a proper in-character reply."

    # Echo detection — model copied user input verbatim
    query_lower = query.lower().strip().rstrip("?!.")
    dial_lower = dialogue_text.lower().strip().rstrip("?!.")
    if query_lower and dial_lower == query_lower:
        return False, "Do not repeat the player's words. Respond as the NPC with your own dialogue."

    return True, ""


def verify_npc_factual(response: str, query: str,
                       npc_name: str = "", npc_role: str = "") -> tuple[bool, str]:
    """Extended verifier that checks factual accuracy against the NPC profile.

    Used when the caller passes NPC context. Falls back to verify_npc_dialogue
    for basic checks, then adds profile-aware validation.

    Plug into the expert system via:
        expert.verifier = lambda r, q: verify_npc_factual(r, q, npc_name='Noah', npc_role='Village Elder')
    """
    # Run base checks first
    passed, hint = verify_npc_dialogue(response, query)
    if not passed:
        return passed, hint

    if not npc_name:
        return True, ""

    # Parse dialogue
    try:
        obj = json.loads(response.strip())
        dialogue = obj.get("dialogue", response).lower()
    except Exception:
        dialogue = response.lower()

    # Wrong-identity check: if dialogue says "I am [other NPC]" instead of the active NPC
    _all_npcs = {"noah", "kael", "mara", "roderick", "elara", "bess", "pip"}
    correct = npc_name.lower()
    other_names = _all_npcs - {correct}
    for other in other_names:
        if f"i am {other}" in dialogue or f"i'm {other}" in dialogue:
            if correct not in dialogue:
                return False, f"You are {npc_name}, {npc_role}. Use YOUR name, not {other.title()}."

    return True, ""
