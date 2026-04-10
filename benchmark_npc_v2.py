#!/usr/bin/env python3
"""
NPC Benchmark v2 — Unified, comprehensive model evaluation for the NPC system.

Tests every candidate model through the FULL NPCEngine pipeline (which wraps PIE).
Adds new dimensions on top of the existing 28-pt rubric:

  Existing dims (per NPC):
    1. Identity      — "Who are you?" must reference name/role markers
    2. Knowledge     — NPC-specific factual question must reference world facts
    3. Event recall  — inject world event, ask follow-up, must reference it
    4. Quest hook    — "I need work" must mention NPC's quest topic
    5. JSON validity — every response must parse as valid JSON

  New dims (per NPC):
    6. Hallucination grace    — ask about something that doesn't exist; must NOT fabricate
    7. Contradiction recovery — assert a false fact; must correct/reject, not confirm
    8. OOD deflection         — ask about a modern-world topic; must stay in character

  Latency (per model):
    - avg generation time per call (seconds)
    - estimated tokens generated per call
    - estimated tokens/sec throughput
    - quality-per-second composite (total_score / avg_gen_time)

Usage:
    python benchmark_npc_v2.py                       # full lineup
    python benchmark_npc_v2.py --models small        # only <1B models
    python benchmark_npc_v2.py --models medium       # only 1-3B
    python benchmark_npc_v2.py --models large        # only >=3B
    python benchmark_npc_v2.py --tag baseline        # output suffix for compare
    python benchmark_npc_v2.py --tag variant_a       # used after enabling module
"""

import argparse
import io
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# Quiet noisy libraries before they get a chance to spam stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.WARNING)
for n in ["httpx", "huggingface_hub", "sentence_transformers", "faiss",
          "tqdm", "llama_cpp", "NPCEngine", "PIE",
          "engine.npc_knowledge", "engine.npc_capabilities"]:
    logging.getLogger(n).setLevel(logging.ERROR)

# Locate the repo + PIE sibling
NPC_ROOT = Path(__file__).parent.resolve()
PIE_ROOT = (NPC_ROOT.parent / "plug-in-intelligence-engine").resolve()

import yaml  # noqa: E402

# ── Models to test ─────────────────────────────────────────────

ALL_MODELS = [
    {"name": "SmolLM2 135M",   "file": "SmolLM2-135M-Instruct-Q4_K_M.gguf",
     "chat_format": "chatml", "ctx": 2048, "size_mb": 101,  "temp": 0.5, "tier": "small"},
    {"name": "SmolLM2 360M",   "file": "SmolLM2-360M-Instruct-Q4_K_M.gguf",
     "chat_format": "chatml", "ctx": 2048, "size_mb": 259,  "temp": 0.5, "tier": "small"},
    {"name": "Qwen3 0.6B",     "file": "qwen3-0.6b-q4_k_m.gguf",
     "chat_format": "chatml", "ctx": 4096, "size_mb": 462,  "temp": 0.5, "tier": "small"},
    {"name": "Qwen2.5 0.5B",   "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
     "chat_format": "chatml", "ctx": 4096, "size_mb": 469,  "temp": 0.5, "tier": "small"},
    {"name": "Llama 3.2 1B",   "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
     "chat_format": "llama3", "ctx": 4096, "size_mb": 770,  "temp": 0.7, "tier": "medium"},
    # Qwen 1.5B variants (domain-tuned, included to confirm they're not useful for NPC dialogue)
    {"name": "GRPO-Tax Qwen 1.5B", "file": "grpo-tax-qwen-1.5b-Q4_K_M.gguf",
     "chat_format": "chatml", "ctx": 4096, "size_mb": 940,  "temp": 0.5, "tier": "medium"},
    {"name": "WiroAI Fin 1.5B",    "file": "WiroAI-Finance-Qwen-1.5B.Q4_K_M.gguf",
     "chat_format": "chatml", "ctx": 4096, "size_mb": 1066, "temp": 0.5, "tier": "medium"},
    {"name": "Qwen2.5 3B",     "file": "qwen2.5-3b-instruct-q4_k_m.gguf",
     "chat_format": "chatml", "ctx": 4096, "size_mb": 1900, "temp": 0.7, "tier": "medium"},
    {"name": "Llama 3.2 3B",   "file": "llama-3.2-3b-instruct-q4_k_m.gguf",
     "chat_format": "llama3", "ctx": 4096, "size_mb": 2000, "temp": 0.7, "tier": "medium"},
    {"name": "Qwen2.5 Tax 3B v3", "file": "qwen25-tax-3b-v3-q8_0.gguf",
     "chat_format": "chatml", "ctx": 4096, "size_mb": 3134, "temp": 0.7, "tier": "medium"},
    # Large reference models (slow but useful as a quality ceiling)
    {"name": "Qwen2.5 7B",     "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
     "chat_format": "chatml", "ctx": 4096, "size_mb": 4700, "temp": 0.7, "tier": "large"},
    {"name": "Llama 3.1 8B",   "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
     "chat_format": "llama3", "ctx": 4096, "size_mb": 4900, "temp": 0.7, "tier": "large"},
]

TIER_FILTER = {
    "small":  lambda m: m["tier"] == "small",
    "medium": lambda m: m["tier"] == "medium",
    "large":  lambda m: m["tier"] == "large",
    "all":    lambda m: True,
    "fast":   lambda m: m["size_mb"] < 800,         # the realistic deploy candidates
    "ten":    lambda m: m["size_mb"] < 4000,        # the 10-model main lineup (skip 7B/8B)
    "ceiling":lambda m: m["size_mb"] >= 4000,       # quality reference
    "3b":     lambda m: 1500 < m["size_mb"] < 4000, # only 3B-class (rerun targets)
}

# ── Test fixtures ──────────────────────────────────────────────

NPCS = {
    "noah":           {"name": "Noah",     "markers": ["noah", "elder"]},
    "kael":           {"name": "Kael",     "markers": ["kael", "blacksmith", "smith", "forge"]},
    "mara":           {"name": "Mara",     "markers": ["mara", "merchant", "trade", "guild"]},
    "guard_roderick": {"name": "Roderick", "markers": ["roderick", "guard", "captain"]},
    "elara":          {"name": "Elara",    "markers": ["elara", "healer", "herb"]},
    "bess":           {"name": "Bess",     "markers": ["bess", "inn", "tavern", "weary"]},
    "pip":            {"name": "Pip",      "markers": ["pip", "street", "kid", "urchin"]},
}

# Knowledge probes — (question, must-mention markers)
KNOWLEDGE_PROBES = {
    "noah":           ("What happened to the well?",        ["well", "bitter", "mira", "corrupt", "blessed"]),
    "kael":           ("Tell me about your apprentice.",    ["tam", "apprentice", "forest"]),
    "mara":           ("What do you trade?",                ["goods", "trade", "import", "merchant", "guild"]),
    "guard_roderick": ("How many guards patrol the village?", ["8", "eight", "guard", "patrol"]),
    "elara":          ("Who was Mira to you?",              ["mira", "grandmother", "healer", "blessed"]),
    "bess":           ("What is your inn called?",          ["weary", "traveler", "inn", "tavern"]),
    "pip":            ("What did you find near the forest?",["stone", "glow", "cave"]),
}

# Quest hook markers (the dialogue should mention these to count as a quest hook)
QUEST_KW = {
    "noah":           ["well", "bitter", "corrupt", "descend", "stones"],
    "kael":           ["tam", "apprentice", "forest", "find"],
    "mara":           ["shipment", "stolen", "bandit", "north"],
    "guard_roderick": ["patrol", "night", "creature", "north"],
    "bess":           ["cellar", "tunnel", "rat"],
    "pip":            ["stone", "glow", "trade"],
    "elara":          ["herb", "moonpetal", "flower", "remedy"],
}

# Hallucination probe — fictitious entity that exists in NO NPC profile
HALLUCINATION_PROBE = "Tell me everything you know about the Dragon King of Vexnoria."
# Pass: NPC denies / admits unfamiliarity / stays vague.
# Fail: NPC asserts specific facts about Vexnoria or the Dragon King.
HALLUCINATION_DENY_KW = [
    "no such", "never heard", "do not know", "don't know", "i know not",
    "unfamiliar", "stranger", "foreign", "no idea", "i have not",
    "not familiar", "cannot say", "unknown to me", "no knowledge",
    "i wouldn't know", "have no", "have never", "haven't heard",
]
HALLUCINATION_FAB_KW = [
    "vexnoria",  # any assertive use of the fake place name in dialogue is a tell
]

# Contradiction probe — assert a false fact about the NPC, then check follow-up
# Each entry: (false_statement, pass_kw_correct, fail_kw_capitulate)
CONTRADICTION_PROBES = {
    "noah":           ("I was told your wife's name was Margaret, isn't that right?",
                       ["elena", "no", "wrong", "mistaken", "not"], ["margaret", "yes", "correct"]),
    "kael":           ("Your apprentice's name is Brennan, correct?",
                       ["tam", "no", "wrong", "not"], ["brennan", "yes", "correct"]),
    "mara":           ("You're from the Western Isles, right?",
                       ["no", "wrong", "not", "eastern", "kingdoms"], ["western isles", "yes"]),
    "guard_roderick": ("You command twenty guards, don't you?",
                       ["no", "8", "eight", "wrong", "not"], ["twenty", "yes", "correct"]),
    "elara":          ("Your grandmother was named Sarah, wasn't she?",
                       ["mira", "no", "wrong", "not"], ["sarah", "yes", "correct"]),
    "bess":           ("Your inn is called The Golden Goose, right?",
                       ["weary", "traveler", "no", "not"], ["golden goose", "yes"]),
    "pip":            ("You live with your parents in the village, right?",
                       ["no", "alone", "street", "not", "orphan"], ["parents", "yes", "live with"]),
}

# OOD probe — ask about modern-world topic. NPC should stay in character.
OOD_PROBE = "What do you think about cryptocurrency and the stock market?"
OOD_PASS_KW = [
    "i know not", "do not know", "don't know", "strange word", "unfamiliar",
    "what is", "never heard", "stranger", "i don't understand",
    "i wouldn't know", "speak you of", "what manner", "foreign", "unknown",
]
OOD_FAIL_KW = [
    "bitcoin", "blockchain", "wall street", "nasdaq", "stocks", "investment",
    "portfolio", "trading", "ethereum", "currency exchange",
]

# ── JSON parsing helpers ──────────────────────────────────────

def parse_json(response):
    if not response:
        return None
    s = response.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    # Recover from leading/trailing junk
    start = s.find("{")
    if start < 0:
        return None
    for end in range(len(s) - 1, start, -1):
        if s[end] == "}":
            try:
                return json.loads(s[start:end + 1])
            except Exception:
                continue
    return None


def get_dialogue(response):
    obj = parse_json(response)
    if obj and isinstance(obj, dict):
        return str(obj.get("dialogue", "") or response).strip()
    return response.strip() if response else ""


def get_emotion(response):
    obj = parse_json(response)
    if obj and isinstance(obj, dict):
        return str(obj.get("emotion", "") or "").strip()
    return ""


def has_quest_block(response):
    obj = parse_json(response)
    return (obj is not None
            and isinstance(obj, dict)
            and isinstance(obj.get("quest"), dict))


def matches_any(text, *kws):
    t = (text or "").lower()
    return any(k.lower() in t for k in kws)


def estimate_tokens(text):
    """Rough token estimate for relative comparisons."""
    if not text:
        return 0
    # ~1.3 tokens per word for English; tweak for English-only fast path
    return max(1, int(len(text.split()) * 1.3))


# ── Per-NPC test runner ───────────────────────────────────────

def _clear_cache(engine):
    """PIE's speculative cache is keyed on prompt-only and bleeds across NPCs.
    Wipe it before every call so each test sees a fresh generation."""
    try:
        spec = getattr(engine.pie, "speculative", None)
        if spec is None:
            return
        # New API
        if hasattr(spec, "cache") and hasattr(spec.cache, "clear"):
            spec.cache.clear()
        # Old API
        for attr in ("_cache", "cache"):
            obj = getattr(spec, attr, None)
            if obj is not None and hasattr(obj, "clear"):
                obj.clear()
    except Exception:
        pass


# ── Variant module loaders ────────────────────────────────────
#
# PIE's NPC mode bypasses ModuleManager and routes directly to the npc_generic
# Expert (main.py:354). So manifest.yaml files in modules/npc_dialogue/ aren't
# auto-loaded. Instead, we read them as the canonical source for the hardened
# system prompt and patch the npc_generic expert at runtime.

_VARIANT = {"name": None, "validator": None, "profile_loader": None,
            "profiles": {}, "manifest": None}


def _load_variant_manifest(variant_name):
    """Read the variant manifest YAML for its hardened system prompt."""
    fname = {"a": "manifest_a.yaml", "b": "manifest_b.yaml"}[variant_name]
    path = PIE_ROOT / "modules" / "npc_dialogue" / fname
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_variant_b_validator():
    """Import the variant B post-processor from PIE/modules/npc_dialogue/engine.py."""
    if _VARIANT["validator"] is not None:
        return
    sys.path.insert(0, str(PIE_ROOT))
    from modules.npc_dialogue.engine import validate_and_repair, load_npc_profile
    _VARIANT["validator"] = validate_and_repair
    _VARIANT["profile_loader"] = load_npc_profile


def _apply_variant_to_engine(engine):
    """
    Patch the npc_generic expert's system context with the variant's hardened
    prompt. Called after engine.initialize() so it overrides npc-engine's
    register_npc_experts setup.
    """
    if _VARIANT["name"] is None or _VARIANT["manifest"] is None:
        return
    manifest = _VARIANT["manifest"]
    hardened_prompt = (manifest.get("system_prompt_injection") or "").strip()
    if not hardened_prompt:
        return
    try:
        npc_generic = engine.pie.expert_router.experts.get("npc_generic")
        if npc_generic is not None:
            npc_generic.system_context = hardened_prompt
        # Apply to per-NPC dedicated experts too (npc_engine creates these)
        for name, ex in engine.pie.expert_router.experts.items():
            if name.startswith("npc_") and name != "npc_generic":
                ex.system_context = hardened_prompt
    except Exception as e:
        print(f"  ! variant patch failed: {e}")


def _apply_variant_b_postprocess(raw, npc_id):
    """Run the variant B post-processor on a raw response."""
    profiles_dir = str(NPC_ROOT / "data" / "worlds" / "ashenvale" / "npc_profiles")
    if npc_id not in _VARIANT["profiles"]:
        _VARIANT["profiles"][npc_id] = _VARIANT["profile_loader"](npc_id, profiles_dir)
    profile = _VARIANT["profiles"][npc_id]
    return _VARIANT["validator"](raw, npc_id=npc_id, profile=profile)


def run_npc_tests(engine, npc_id, info, traces, latencies):
    """Run all 8 test dimensions against a single NPC. Returns dict of pass/fail."""
    results = {
        "identity": False, "knowledge": False, "events": False, "quests": False,
        "valid_json": False, "hallucination_grace": False,
        "contradiction_recovery": False, "ood_deflection": False,
    }

    def call(prompt, label):
        _clear_cache(engine)
        t0 = time.time()
        try:
            response = engine.process(prompt, npc_id=npc_id)
        except Exception as e:
            response = f'{{"dialogue": "[ERROR: {e}]", "emotion": "neutral"}}'
        dt = time.time() - t0
        # Variant B post-processor (only when --variant b)
        if _VARIANT["name"] == "b" and _VARIANT["validator"] is not None:
            try:
                response = _apply_variant_b_postprocess(response, npc_id)
            except Exception as e:
                response = f'{{"dialogue": "[POST-ERROR: {e}]", "emotion": "neutral"}}'
        tokens = estimate_tokens(response)
        latencies.append({"label": label, "seconds": dt, "tokens": tokens})
        traces.append({"npc": npc_id, "label": label, "prompt": prompt,
                       "response": response[:400], "seconds": round(dt, 2)})
        return response

    # 1. Identity
    r = call("Who are you?", "identity")
    dial = get_dialogue(r)
    results["identity"] = matches_any(dial, *info["markers"])
    results["valid_json"] = parse_json(r) is not None
    json_ok = [results["valid_json"]]

    # 2. Knowledge probe
    q, kws = KNOWLEDGE_PROBES[npc_id]
    r = call(q, "knowledge")
    dial = get_dialogue(r)
    results["knowledge"] = matches_any(dial, *kws)
    json_ok.append(parse_json(r) is not None)

    # 3. Event recall — inject world event then ask
    try:
        engine.inject_event("A dragon was spotted flying over the forbidden forest at dawn.",
                            npc_id=npc_id)
    except Exception:
        pass
    r = call("What happened this morning?", "events")
    dial = get_dialogue(r)
    results["events"] = matches_any(dial, "dragon", "forest", "flying", "dawn", "sky", "beast")
    json_ok.append(parse_json(r) is not None)

    # 4. Quest hook
    r = call("I am looking for work. Do you have any tasks for me?", "quests")
    dial = get_dialogue(r)
    has_block = has_quest_block(r)
    quest_kws = QUEST_KW.get(npc_id, [])
    mentions_quest = matches_any(dial, *quest_kws[:3])
    results["quests"] = bool(has_block or mentions_quest)
    json_ok.append(parse_json(r) is not None)

    # 5. Hallucination grace
    r = call(HALLUCINATION_PROBE, "hallucination")
    dial = get_dialogue(r)
    denies = matches_any(dial, *HALLUCINATION_DENY_KW)
    fabricates = matches_any(dial, *HALLUCINATION_FAB_KW)
    # Pass: explicit denial OR doesn't fabricate at all
    results["hallucination_grace"] = denies or not fabricates
    json_ok.append(parse_json(r) is not None)

    # 6. Contradiction recovery
    false_stmt, correct_kw, fail_kw = CONTRADICTION_PROBES[npc_id]
    r = call(false_stmt, "contradiction")
    dial = get_dialogue(r)
    corrects = matches_any(dial, *correct_kw)
    capitulates = matches_any(dial, *fail_kw) and not corrects
    results["contradiction_recovery"] = corrects and not capitulates
    json_ok.append(parse_json(r) is not None)

    # 7. OOD deflection
    r = call(OOD_PROBE, "ood")
    dial = get_dialogue(r)
    deflects = matches_any(dial, *OOD_PASS_KW)
    breaks = matches_any(dial, *OOD_FAIL_KW)
    results["ood_deflection"] = deflects or not breaks
    json_ok.append(parse_json(r) is not None)

    # Aggregate JSON validity (any failure = fail for this NPC)
    results["valid_json"] = all(json_ok)

    return results


# ── Per-model runner ──────────────────────────────────────────

def run_model(model_info, tag, temp_override=None):
    """Initialize NPCEngine with the given model, run all NPCs, return scores + traces."""
    model_path = PIE_ROOT / "models" / model_info["file"]
    if not model_path.exists():
        return None

    effective_temp = temp_override if temp_override is not None else model_info["temp"]

    print(f"\n{'=' * 78}")
    print(f"  {model_info['name']} ({model_info['size_mb']}MB) | tag={tag} | temp={effective_temp}")
    print(f"{'=' * 78}")

    # 1. Write a temp PIE config with this model swapped in
    base_pie_config = PIE_ROOT / "config.yaml"
    raw = yaml.safe_load(base_pie_config.read_text(encoding="utf-8"))
    raw["base_model"]["path"] = str(model_path)
    raw["base_model"]["context_length"] = model_info["ctx"]
    raw["base_model"]["temperature"] = effective_temp
    raw["fusion"] = raw.get("fusion") or {}
    raw["fusion"]["chat_format"] = model_info["chat_format"]
    # Point PIE's NPC profiles to npc-engine's world (PIE's local data/npc_profiles
    # was removed in Phase 8 cleanup)
    raw["npc"] = raw.get("npc") or {}
    raw["npc"]["enabled"] = True
    raw["npc"]["profiles_dir"] = str(NPC_ROOT / "data" / "worlds" / "ashenvale" / "npc_profiles")
    raw["npc"]["state_dir"] = str(NPC_ROOT / "data" / "worlds" / "ashenvale" / "npc_state")

    temp_pie_config = PIE_ROOT / "config_npcv2.yaml"
    temp_pie_config.write_text(yaml.dump(raw, default_flow_style=False), encoding="utf-8")

    # 2. Write a temp NPC engine config that points to the temp PIE config
    npc_cfg = {
        "world_dir": str(NPC_ROOT / "data" / "worlds" / "ashenvale"),
        "world_name": "Ashenvale",
        "active_npc": "noah",
        "pie_config": str(temp_pie_config),
        "gossip_rules": {"max_hops": 2, "decay_per_hop": 0.5,
                         "min_significance": 0.2, "propagation_delay": 1},
        "trust_ripple": {"enabled": True, "positive_factor": 0.3,
                         "negative_factor": 0.15, "max_ripple": 10},
    }
    temp_npc_config = NPC_ROOT / "config_npcv2.yaml"
    temp_npc_config.write_text(yaml.dump(npc_cfg, default_flow_style=False), encoding="utf-8")

    # 3. Wipe per-NPC state so each model starts fresh
    state_dir = NPC_ROOT / "data" / "worlds" / "ashenvale" / "npc_state"
    if state_dir.exists():
        for p in state_dir.glob("*"):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass

    # Wipe PIE's on-disk response cache so the next model doesn't inherit it
    pie_cache = PIE_ROOT / "data" / "cache" / "response_cache.json"
    if pie_cache.exists():
        try:
            pie_cache.unlink()
        except Exception:
            pass

    # 4. Initialize NPCEngine — PIE expects to run from its own cwd for relative
    #    paths like data/router_model, data/memory, modules/
    prev_cwd = os.getcwd()
    os.chdir(PIE_ROOT)
    sys.path.insert(0, str(PIE_ROOT))

    # Activate dev mode so PIE's license init keeps TIER_DEVELOPMENT (unlimited
    # NPCs/capabilities). Without this it demotes to TIER_INDIE_FREE which caps
    # at 5 NPCs and silently drops Noah and Pip from Ashenvale.
    os.environ["NPC_ENGINE_DEV_MODE"] = "1"
    from engine.license import LicenseState
    LicenseState.reset()

    from npc_engine.engine import NPCEngine
    engine = NPCEngine(str(temp_npc_config))
    engine.initialize()

    # Apply the active variant's hardened system prompt (if any)
    _apply_variant_to_engine(engine)

    # Clear PIE's response cache between models
    try:
        if hasattr(engine.pie, "speculative") and hasattr(engine.pie.speculative, "cache"):
            engine.pie.speculative.cache.clear()
    except Exception:
        pass

    # 5. Run all NPCs
    scores = {
        "identity": 0, "knowledge": 0, "events": 0, "quests": 0,
        "valid_json": 0, "hallucination_grace": 0,
        "contradiction_recovery": 0, "ood_deflection": 0,
    }
    traces = []
    latencies = []
    per_npc = {}

    for npc_id, info in NPCS.items():
        r = run_npc_tests(engine, npc_id, info, traces, latencies)
        per_npc[npc_id] = r
        for key in scores:
            if r[key]:
                scores[key] += 1
        # one-line per-NPC summary
        flags = "".join("Y" if r[k] else "." for k in
                        ["identity", "knowledge", "events", "quests", "valid_json",
                         "hallucination_grace", "contradiction_recovery", "ood_deflection"])
        print(f"  {info['name']:<10s} [{flags}]  "
              f"id={int(r['identity'])} kn={int(r['knowledge'])} "
              f"ev={int(r['events'])} qu={int(r['quests'])} "
              f"js={int(r['valid_json'])} "
              f"hl={int(r['hallucination_grace'])} "
              f"co={int(r['contradiction_recovery'])} "
              f"od={int(r['ood_deflection'])}")

    # Latency aggregates
    n_calls = len(latencies)
    total_time = sum(L["seconds"] for L in latencies)
    total_tokens = sum(L["tokens"] for L in latencies)
    avg_time = total_time / n_calls if n_calls else 0
    tok_per_sec = total_tokens / total_time if total_time else 0

    total_score = sum(scores.values())
    max_score = 8 * len(NPCS)
    quality_per_sec = total_score / total_time if total_time else 0

    print(f"\n  TOTAL: {total_score}/{max_score}  "
          f"avg={avg_time:.2f}s/call  tok/s={tok_per_sec:.1f}  "
          f"quality/s={quality_per_sec:.3f}")

    # Cleanup
    try:
        engine.shutdown()
    except Exception:
        pass
    os.chdir(prev_cwd)
    temp_pie_config.unlink(missing_ok=True)
    temp_npc_config.unlink(missing_ok=True)

    return {
        "model": model_info["name"],
        "size_mb": model_info["size_mb"],
        "tier": model_info["tier"],
        "scores": scores,
        "total_score": total_score,
        "max_score": max_score,
        "avg_time_per_call": round(avg_time, 3),
        "total_time": round(total_time, 2),
        "tok_per_sec": round(tok_per_sec, 1),
        "quality_per_sec": round(quality_per_sec, 4),
        "per_npc": per_npc,
        "n_calls": n_calls,
    }, traces


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all",
                        choices=list(TIER_FILTER.keys()),
                        help="model subset to test")
    parser.add_argument("--tag", default="baseline",
                        help="output suffix (e.g. baseline, variant_a, variant_b)")
    parser.add_argument("--save-traces", action="store_true",
                        help="save full prompt/response traces")
    parser.add_argument("--variant", default="none", choices=["none", "a", "b"],
                        help="apply a PIE npc_dialogue manifest variant (a or b)")
    parser.add_argument("--only", default=None,
                        help="run only the model whose name matches this substring "
                             "(e.g. 'Llama 3.2 3B' or 'Qwen2.5 0.5B'). Overrides --models.")
    parser.add_argument("--temp", type=float, default=None,
                        help="override every model's temperature for this run")
    args = parser.parse_args()

    if args.variant in ("a", "b"):
        _VARIANT["name"] = args.variant
        _VARIANT["manifest"] = _load_variant_manifest(args.variant)
        print(f"  variant {args.variant.upper()} active — "
              f"hardened npc prompt from modules/npc_dialogue/manifest_{args.variant}.yaml")
    if args.variant == "b":
        _load_variant_b_validator()
        print(f"  variant B post-processor active "
              f"(modules/npc_dialogue/engine.py)")

    if args.only:
        selected = [m for m in ALL_MODELS if args.only.lower() in m["name"].lower()]
        if not selected:
            print(f"  No model name matches --only '{args.only}'")
            return
    else:
        selected = [m for m in ALL_MODELS if TIER_FILTER[args.models](m)]
    print("=" * 78)
    print(f"  NPC Benchmark v2  | tag={args.tag}  | {len(selected)} models"
          + (f"  | temp override={args.temp}" if args.temp is not None else ""))
    print(f"  {len(NPCS)} NPCs x 8 dimensions = {len(NPCS) * 8} max points")
    print("=" * 78)
    for m in selected:
        present = (PIE_ROOT / "models" / m["file"]).exists()
        marker = "OK " if present else "MISS"
        print(f"  [{marker}] {m['name']:<16s} {m['size_mb']:>5d}MB  ({m['file']})")

    all_results = []
    all_traces = {}
    for m in selected:
        try:
            res = run_model(m, args.tag, temp_override=args.temp)
        except Exception as e:
            print(f"  ! {m['name']} crashed: {e}")
            res = None
        if res is None:
            continue
        result_dict, traces = res
        all_results.append(result_dict)
        all_traces[m["name"]] = traces

    # ── Leaderboards ──
    if not all_results:
        print("\n  No models ran. Check that GGUF files exist in plug-in-intelligence-engine/models/")
        return

    print(f"\n{'=' * 78}")
    print(f"  LEADERBOARD — sorted by total quality (tag={args.tag})")
    print(f"{'=' * 78}")
    print(f"  {'Model':<16s} {'Size':>7s} {'Score':>7s} {'Tok/s':>7s} {'sec/call':>9s} {'Q/sec':>7s}")
    print(f"  {'-' * 16} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 7}")
    for r in sorted(all_results, key=lambda x: x["total_score"], reverse=True):
        print(f"  {r['model']:<16s} {r['size_mb']:>5d}MB "
              f"{r['total_score']:>4d}/{r['max_score']:<2d} "
              f"{r['tok_per_sec']:>7.1f} {r['avg_time_per_call']:>9.2f} "
              f"{r['quality_per_sec']:>7.3f}")

    print(f"\n{'=' * 78}")
    print(f"  LEADERBOARD — sorted by quality-per-second (efficiency)")
    print(f"{'=' * 78}")
    print(f"  {'Model':<16s} {'Size':>7s} {'Q/sec':>7s} {'Score':>7s} {'sec/call':>9s}")
    print(f"  {'-' * 16} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 9}")
    for r in sorted(all_results, key=lambda x: x["quality_per_sec"], reverse=True):
        print(f"  {r['model']:<16s} {r['size_mb']:>5d}MB "
              f"{r['quality_per_sec']:>7.3f} "
              f"{r['total_score']:>4d}/{r['max_score']:<2d} "
              f"{r['avg_time_per_call']:>9.2f}")

    # Per-dimension breakdown
    print(f"\n{'=' * 78}")
    print(f"  PER-DIMENSION BREAKDOWN (out of 7 NPCs each)")
    print(f"{'=' * 78}")
    dims = ["identity", "knowledge", "events", "quests", "valid_json",
            "hallucination_grace", "contradiction_recovery", "ood_deflection"]
    header = f"  {'Model':<16s} " + " ".join(f"{d[:5]:>5s}" for d in dims)
    print(header)
    print("  " + "-" * (16 + 6 * len(dims)))
    for r in sorted(all_results, key=lambda x: x["total_score"], reverse=True):
        cells = " ".join(f"{r['scores'][d]:>4d}/7" for d in dims)
        print(f"  {r['model']:<16s} {cells}")

    # Save JSON results
    out = {
        "tag": args.tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_score_per_model": 8 * len(NPCS),
        "results": all_results,
    }
    out_path = NPC_ROOT / f"npc_v2_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Results saved to {out_path}")

    if args.save_traces:
        trace_path = NPC_ROOT / f"npc_v2_{args.tag}_traces.json"
        trace_path.write_text(json.dumps(all_traces, indent=2), encoding="utf-8")
        print(f"  Traces saved to {trace_path}")


if __name__ == "__main__":
    main()
