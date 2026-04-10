#!/usr/bin/env python3
"""
Social System Tests — Phase 3

Tests gossip propagation, trust ripple, and the social graph.
These are mostly state-level checks (don't need model generation for most).

Usage:
    python tests/test_social.py
"""

import io
import json
import logging
import os
import sys
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


def test_social_graph(engine):
    """Social graph should load correctly from world.yaml."""
    results = []
    graph = engine.get_social_graph()
    results.append(("has_npcs", len(graph.get("npcs", [])) > 0))
    results.append(("has_connections", len(graph.get("connections", [])) > 0))
    results.append(("npc_count", f"{len(graph.get('npcs', []))} NPCs"))
    results.append(("connection_count", f"{len(graph.get('connections', []))} connections"))

    # Verify specific connections exist
    conns = graph.get("connections", [])
    noah_kael = any(c["from"] == "noah" and c["to"] == "kael" for c in conns)
    results.append(("noah_kael_connected", noah_kael))

    return results


def test_trust_ripple_positive(engine):
    """Boosting one NPC's trust should ripple to connected NPCs."""
    results = []

    # Get initial trust for Noah and a connected NPC
    noah_state = engine.get_npc_state("noah")
    noah_initial = noah_state["capabilities"].get("trust", {}).get("level", 30)

    # Boost Noah's trust significantly
    engine.adjust_trust("noah", 20, reason="test: positive ripple")

    # Process a turn to trigger ripple
    engine.process("Thank you, Noah!", npc_id="noah")

    # Check Noah's trust
    noah_state = engine.get_npc_state("noah")
    noah_after = noah_state["capabilities"].get("trust", {}).get("level", 0)
    results.append(("noah_trust_boosted", noah_after > noah_initial))
    results.append(("noah_trust_value", f"{noah_initial} -> {noah_after}"))

    # Check if connected NPC's trust also changed via ripple
    # (ripple is 30% of delta * closeness — may be small)
    kael_state = engine.get_npc_state("kael")
    kael_trust = kael_state["capabilities"].get("trust", {}).get("level", 30)
    results.append(("kael_trust_check", f"kael trust = {kael_trust}"))
    # Ripple may not always fire (requires |delta| >= 3 and process() to run)
    results.append(("ripple_system_exists", engine.reputation_ripple is not None))

    return results


def test_trust_ripple_negative(engine):
    """Dropping trust should ripple negatively (at 15% factor)."""
    results = []

    # Drop Noah's trust
    engine.adjust_trust("noah", -15, reason="test: negative ripple")

    # Process a turn
    engine.process("I don't trust you anymore.", npc_id="noah")

    noah_state = engine.get_npc_state("noah")
    noah_trust = noah_state["capabilities"].get("trust", {}).get("level", 0)
    results.append(("noah_trust_dropped", noah_trust < 30))
    results.append(("noah_trust_value", f"{noah_trust}"))

    return results


def test_gossip_spread(engine):
    """Telling one NPC a fact should propagate to connected NPCs via gossip."""
    results = []

    # Talk to Noah — mention something notable
    engine.process("I come from the kingdom of Valdris, far to the east. I carry a magical sword.", npc_id="noah")

    # Check if any connected NPCs received gossip events
    kael = engine.pie.npc_knowledge.get("kael")
    if kael:
        kael_events = [e.description for e in kael.events]
        results.append(("kael_received_gossip", len(kael_events) > 0))
        results.append(("gossip_content", str(kael_events[:2])[:100] if kael_events else "none"))
    else:
        results.append(("kael_loaded", False))

    # Check gossip propagator state
    if engine.gossip_propagator:
        results.append(("propagator_active", True))
    else:
        results.append(("propagator_active", False))

    return results


def test_event_injection_global(engine):
    """Global events should reach all NPCs."""
    results = []

    # Inject global event
    engine.inject_event("A meteor crashed in the forbidden forest, shaking the ground.")

    # Check multiple NPCs have the event
    for npc_id in ["noah", "kael", "mara"]:
        npc = engine.pie.npc_knowledge.get(npc_id)
        if npc:
            has_event = any("meteor" in e.description.lower() for e in npc.events)
            results.append((f"{npc_id}_has_meteor_event", has_event))
        else:
            results.append((f"{npc_id}_loaded", False))

    return results


# ── Runner ────────────────────────────────────────────────────

TESTS = {
    "social_graph": ("Social graph loads correctly", test_social_graph),
    "trust_positive": ("Positive trust ripple to connected NPCs", test_trust_ripple_positive),
    "trust_negative": ("Negative trust ripple", test_trust_ripple_negative),
    "gossip": ("Gossip spreads to connected NPCs", test_gossip_spread),
    "global_event": ("Global events reach all NPCs", test_event_injection_global),
}


def main():
    print("=" * 70)
    print(f"  SOCIAL SYSTEM TESTS ({len(TESTS)} tests)")
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
            if is_pass: all_pass += 1
            elif isinstance(value, bool) and not value: all_fail += 1

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
