"""
NPC Expert Builders — Creates expert instances for the PIE expert system.

These experts use the FewShotLoader to get examples from YAML instead of
hardcoding them. The GBNF grammar and system context remain the same.
"""

import logging

from npc_engine.bridge import Expert, SolvedExample
from npc_engine.experts.verifiers import verify_npc_dialogue, verify_npc_factual

logger = logging.getLogger(__name__)

# ── GBNF Grammar for NPC JSON output ────────────────────────

NPC_JSON_GRAMMAR = r'''
root     ::= "{" ws dialogue "," ws emotion "," ws action ("," ws quest)? ws "}"
dialogue ::= "\"dialogue\"" ws ":" ws string
emotion  ::= "\"emotion\"" ws ":" ws string
action   ::= "\"action\"" ws ":" ws (string | "null")
quest    ::= "\"quest\"" ws ":" ws (questobj | "null")
questobj ::= "{" ws "\"type\"" ws ":" ws string "," ws "\"objective\"" ws ":" ws string "," ws "\"reward\"" ws ":" ws string ws "}"
string   ::= "\"" ([^"\\] | "\\" .)* "\""
ws       ::= [ \t\n]*
'''

NPC_SYSTEM_CONTEXT = (
    "You are the NPC described above. "
    "IMPORTANT: Use YOUR name and role from the [You are...] context — never copy names from examples. "
    "If asked about something you do not know, say so. Never invent facts about unknown places or people. "
    "If the player says something false about you, correct them. "
    "If there is RECENT NEWS, mention it. If you have quests, offer them when asked for work. "
    "Respond with valid JSON. Stay in character. 2-3 sentences."
)


def build_npc_expert(name: str, examples: list[SolvedExample],
                     system_context: str = None,
                     npc_name: str = "", npc_role: str = "") -> Expert:
    """Build an NPC expert with the given examples.
    When npc_name/npc_role are provided, uses the factual verifier that
    checks for wrong-identity (few-shot bleed) during the verify/retry loop.
    """
    if npc_name:
        verifier = lambda r, q: verify_npc_factual(r, q, npc_name=npc_name, npc_role=npc_role)
    else:
        verifier = verify_npc_dialogue
    return Expert(
        name=name,
        system_context=system_context or NPC_SYSTEM_CONTEXT,
        examples=examples,
        scaffolding=None,
        verifier=verifier,
        grammar_str=NPC_JSON_GRAMMAR,
        max_examples=3,
        max_retries=2,
    )


def register_npc_experts(expert_router, few_shot_loader, npc_profiles: dict) -> int:
    """
    Register NPC experts into PIE's ExpertRouter using custom examples.

    1. Replace npc_generic's examples with world-level custom examples
    2. For each NPC with per-NPC examples, register a dedicated npc_{id} expert

    Returns the number of experts registered/updated.
    """
    count = 0

    # Get world-level examples (structural + world YAML)
    world_examples = few_shot_loader.get_world_examples()
    world_solved = few_shot_loader.to_solved_examples(world_examples)

    # Update npc_generic with custom world examples
    if "npc_generic" in expert_router.experts and world_solved:
        expert = expert_router.experts["npc_generic"]
        expert.examples = world_solved
        # Reset cached FAISS embeddings — they index the OLD examples list and
        # will cause IndexError in retrieve_examples() if not invalidated.
        if hasattr(expert, "_example_embeddings"):
            expert._example_embeddings = None
        logger.info(f"Updated npc_generic with {len(world_solved)} custom examples")
        count += 1

    # Register per-NPC experts
    for npc_id, npc_profile in npc_profiles.items():
        # Load NPC's raw YAML data for examples
        npc_data = {}
        if hasattr(npc_profile, 'profile_path'):
            try:
                import yaml
                with open(npc_profile.profile_path, "r", encoding="utf-8") as f:
                    npc_data = yaml.safe_load(f) or {}
            except Exception:
                pass

        npc_examples_raw = few_shot_loader.get_examples_for_npc(npc_id, npc_data)
        npc_solved = few_shot_loader.to_solved_examples(npc_examples_raw)

        # Only register dedicated expert if NPC has custom examples beyond structural
        has_custom = any(
            ex.category and ex.category not in {"greeting", "identity", "adversarial", "quest_ask", "lore"}
            for ex in npc_examples_raw
            if ex not in few_shot_loader.get_world_examples()
        )

        if npc_data.get("examples") or has_custom:
            expert_name = f"npc_{npc_id}"
            npc_name = npc_profile.identity.get("name", "") if hasattr(npc_profile, "identity") else ""
            npc_role = npc_profile.identity.get("role", "") if hasattr(npc_profile, "identity") else ""
            expert_router.experts[expert_name] = build_npc_expert(
                name=expert_name,
                examples=npc_solved,
                npc_name=npc_name,
                npc_role=npc_role,
            )
            logger.info(f"Registered expert '{expert_name}' with {len(npc_solved)} examples")
            count += 1

    return count
