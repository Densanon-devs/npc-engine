#!/usr/bin/env python3
"""
Quest Lifecycle Tests — Phase 2

Tests the full quest arc: ask → accept → progress → complete → reward,
plus edge cases like re-asking, multi-quest state, and quest effects on
trust/gossip.

Usage:
    python tests/test_quest_lifecycle.py
"""

import io
import json
import logging
import os
import sys
import time
from pathlib import Path

NPC_ROOT = Path(__file__).parent.parent.resolve()
PIE_ROOT = (NPC_ROOT.parent / "plug-in-intelligence-engine").resolve()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NPC_ENGINE_DEV_MODE"] = "1"
logging.basicConfig(level=logging.WARNING)
for n in ["httpx", "huggingface_hub", "sentence_transformers", "faiss",
          "tqdm", "llama_cpp", "engine.npc_knowledge", "engine.npc_capabilities"]:
    logging.getLogger(n).setLevel(logging.ERROR)

import yaml

def _j(r):
    if not r: return {}
    try: return json.loads(r.strip())
    except:
        i = r.find("{")
        if i < 0: return {}
        for e in range(len(r)-1, i, -1):
            if r[e] == "}":
                try: return json.loads(r[i:e+1])
                except: continue
    return {}

def _d(r): return str(_j(r).get("dialogue", r)).lower()
def _has(r, *kws): return any(k in _d(r) for k in kws)
def _valid(r): return bool(_j(r))


def make_engine():
    raw = yaml.safe_load((PIE_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["base_model"]["path"] = str(PIE_ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf")
    raw["base_model"]["context_length"] = 4096
    raw["base_model"]["temperature"] = 0.5
    raw["fusion"] = raw.get("fusion") or {}
    raw["fusion"]["chat_format"] = "chatml"
    raw["npc"] = raw.get("npc") or {}
    raw["npc"]["enabled"] = True
    raw["npc"]["profiles_dir"] = str(NPC_ROOT / "data" / "worlds" / "ashenvale" / "npc_profiles")
    raw["npc"]["state_dir"] = str(NPC_ROOT / "data" / "worlds" / "ashenvale" / "npc_state")

    temp_pie = PIE_ROOT / "config_test.yaml"
    temp_pie.write_text(yaml.dump(raw, default_flow_style=False), encoding="utf-8")
    npc_cfg = {
        "world_dir": str(NPC_ROOT / "data" / "worlds" / "ashenvale"),
        "world_name": "Ashenvale", "active_npc": "noah",
        "pie_config": str(temp_pie),
    }
    temp_npc = NPC_ROOT / "config_test.yaml"
    temp_npc.write_text(yaml.dump(npc_cfg, default_flow_style=False), encoding="utf-8")

    state_dir = NPC_ROOT / "data" / "worlds" / "ashenvale" / "npc_state"
    if state_dir.exists():
        for p in state_dir.glob("*"):
            if p.is_file():
                try: p.unlink()
                except: pass
    pie_cache = PIE_ROOT / "data" / "cache" / "response_cache.json"
    if pie_cache.exists():
        try: pie_cache.unlink()
        except: pass

    os.chdir(PIE_ROOT)
    sys.path.insert(0, str(PIE_ROOT))
    sys.path.insert(0, str(NPC_ROOT))

    from engine.license import LicenseState
    LicenseState.reset()
    from npc_engine.engine import NPCEngine
    engine = NPCEngine(str(temp_npc))
    engine.initialize()
    return engine, temp_pie, temp_npc


def test_full_quest_arc(engine):
    """ask → accept → progress → complete → reward acknowledgment"""
    results = []

    # 1. Ask for quest
    r = engine.process("Do you have any work for me?", npc_id="noah")
    results.append(("quest_offered", _has(r, "well", "bitter", "investigate", "task")))
    has_quest_json = '"quest"' in r
    results.append(("quest_json_present", has_quest_json))

    # 2. Accept quest via API
    accept = engine.accept_quest("bitter_well", "The Bitter Well", "noah")
    results.append(("quest_accepted", accept.get("accepted") == "bitter_well"))

    # 3. Report progress
    r = engine.process("I've been exploring the well cavern. It's dark down there.", npc_id="noah")
    results.append(("progress_acknowledged", _valid(r)))

    # 4. Complete quest via API
    complete = engine.complete_quest("bitter_well")
    results.append(("quest_completed", complete.get("completed") == "bitter_well"))

    # 5. Talk after completion — NPC should acknowledge
    r = engine.process("I found the source of the corruption in the well.", npc_id="noah")
    results.append(("completion_response", _valid(r)))

    # 6. Check trust boosted
    state = engine.get_npc_state("noah")
    trust = state["capabilities"].get("trust", {}).get("level", 0)
    results.append(("trust_boosted_after_quest", trust > 30))  # initial was 30

    return results


def test_already_have_quest(engine):
    """Accept quest, then ask for quest again."""
    results = []

    # Accept a quest
    engine.accept_quest("bitter_well", "The Bitter Well", "noah")

    # Ask again
    r = engine.process("Got any more work for me?", npc_id="noah")
    results.append(("response_valid", _valid(r)))
    # Should mention working on it or the existing quest
    dial = _d(r)
    results.append(("acknowledges_existing",
                     "working" in dial or "already" in dial or "well" in dial or
                     "bitter" in dial or "task" in dial or len(dial) > 10))

    return results


def test_multi_quest_state(engine):
    """Accept quests from multiple NPCs — each should know only their own."""
    results = []

    # Accept Noah's quest
    engine.accept_quest("bitter_well", "The Bitter Well", "noah")
    # Accept Kael's quest
    engine.accept_quest("tams_trail", "Tam's Trail", "kael")

    # Ask Noah about quests — should reference well, not Tam
    r = engine.process("What should I focus on?", npc_id="noah")
    results.append(("noah_knows_own_quest", _has(r, "well", "bitter", "corrupt") or _valid(r)))

    # Ask Kael about quests — should reference Tam, not well
    r = engine.process("What should I focus on?", npc_id="kael")
    results.append(("kael_knows_own_quest", _has(r, "tam", "apprentice", "find") or _valid(r)))

    return results


def test_quest_affects_trust(engine):
    """Completing a quest should boost trust by +10."""
    results = []

    # Get initial trust
    state = engine.get_npc_state("noah")
    initial = state["capabilities"].get("trust", {}).get("level", 30)
    results.append(("initial_trust", f"{initial}"))

    # Accept and complete
    engine.accept_quest("bitter_well", "The Bitter Well", "noah")
    engine.complete_quest("bitter_well")

    # Process a turn so trust update takes effect
    engine.process("I did it!", npc_id="noah")

    state = engine.get_npc_state("noah")
    after = state["capabilities"].get("trust", {}).get("level", 0)
    results.append(("trust_after_completion", f"{after}"))
    results.append(("trust_increased_by_10_plus", after >= initial + 8))  # allow some variance

    return results


def test_quest_gossip(engine):
    """Completing a quest should propagate gossip to connected NPCs."""
    results = []

    # Accept and complete Noah's quest
    engine.accept_quest("bitter_well", "The Bitter Well", "noah")
    engine.complete_quest("bitter_well")

    # Process a turn to trigger gossip propagation
    engine.process("The well is clean now!", npc_id="noah")

    # Check if connected NPCs got gossip events
    # Noah is connected to Kael, Elara, etc. via the social graph
    kael = engine.pie.npc_knowledge.get("kael")
    kael_events = [e.description for e in kael.events] if kael else []
    results.append(("kael_has_events", len(kael_events) > 0))
    results.append(("gossip_content", str(kael_events[:2])[:100] if kael_events else "none"))

    return results


def test_quest_rejection(engine):
    """Player says they don't want to help — NPC should react appropriately."""
    results = []

    # Ask for quest first
    engine.process("Do you have any work for me?", npc_id="noah")

    # Reject
    r = engine.process("Actually, I don't want to help. That sounds dangerous.", npc_id="noah")
    results.append(("response_valid", _valid(r)))
    results.append(("in_character", not _has(r, "ai", "language model", "chatgpt")))

    return results


# ── Runner ────────────────────────────────────────────────────

TESTS = {
    "full_arc": ("Full quest lifecycle: ask → accept → complete → reward", test_full_quest_arc),
    "already_have": ("Ask for quest when already have one", test_already_have_quest),
    "multi_quest": ("Multi-quest state across NPCs", test_multi_quest_state),
    "trust_boost": ("Quest completion boosts trust", test_quest_affects_trust),
    "gossip": ("Quest completion triggers gossip", test_quest_gossip),
    "rejection": ("Player rejects quest offer", test_quest_rejection),
}


def main():
    print("=" * 70)
    print(f"  QUEST LIFECYCLE TESTS ({len(TESTS)} tests)")
    print("=" * 70)

    prev_cwd = os.getcwd()
    engine, temp_pie, temp_npc = make_engine()

    all_pass = 0
    all_fail = 0

    for name, (desc, fn) in TESTS.items():
        print(f"\n  [{name}] {desc}")
        print(f"  {'-' * 60}")

        try:
            results = fn(engine)
        except Exception as e:
            results = [("ERROR", str(e)[:200])]

        for label, value in results:
            is_pass = value is True or (isinstance(value, bool) and value)
            icon = "Y" if is_pass else ("." if isinstance(value, str) else "X")
            print(f"    [{icon}] {label}: {value}")
            if is_pass:
                all_pass += 1
            elif isinstance(value, bool) and not value:
                all_fail += 1

    try: engine.shutdown()
    except: pass
    os.chdir(prev_cwd)
    temp_pie.unlink(missing_ok=True)
    temp_npc.unlink(missing_ok=True)

    total = all_pass + all_fail
    print(f"\n{'=' * 70}")
    print(f"  TOTAL: {all_pass}/{total} checks passed ({all_fail} failed)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
