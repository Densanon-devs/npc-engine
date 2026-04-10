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

    return True, ""
