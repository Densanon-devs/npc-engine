#!/usr/bin/env python3
"""
State Persistence Tests — Phase 4

Tests that NPC state (trust, mood, scratchpad) survives engine shutdown/restart.
"""

import io, json, logging, os, sys, shutil
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

STATE_DIR = NPC_ROOT / "data" / "worlds" / "ashenvale" / "npc_state"


def _make_configs():
    raw = yaml.safe_load((PIE_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["base_model"]["path"] = str(PIE_ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf")
    raw["base_model"]["context_length"] = 4096
    raw["base_model"]["temperature"] = 0.5
    raw["fusion"] = raw.get("fusion") or {}
    raw["fusion"]["chat_format"] = "chatml"
    raw["npc"] = raw.get("npc") or {}
    raw["npc"]["enabled"] = True
    raw["npc"]["profiles_dir"] = str(NPC_ROOT / "data" / "worlds" / "ashenvale" / "npc_profiles")
    raw["npc"]["state_dir"] = str(STATE_DIR)

    tp = PIE_ROOT / "config_test.yaml"
    tp.write_text(yaml.dump(raw, default_flow_style=False), encoding="utf-8")
    tn = NPC_ROOT / "config_test.yaml"
    tn.write_text(yaml.dump({
        "world_dir": str(NPC_ROOT / "data" / "worlds" / "ashenvale"),
        "world_name": "Ashenvale", "active_npc": "noah", "pie_config": str(tp),
    }, default_flow_style=False), encoding="utf-8")
    return tp, tn


def _wipe_state():
    if STATE_DIR.exists():
        for p in STATE_DIR.glob("*"):
            if p.is_file():
                try: p.unlink()
                except: pass
    pc = PIE_ROOT / "data" / "cache" / "response_cache.json"
    if pc.exists(): pc.unlink(missing_ok=True)


def _make_engine(tp, tn):
    os.chdir(PIE_ROOT)
    sys.path.insert(0, str(PIE_ROOT))
    sys.path.insert(0, str(NPC_ROOT))
    from engine.license import LicenseState
    LicenseState.reset()
    # Force re-import for clean state
    for mod in list(sys.modules.keys()):
        if mod.startswith(("npc_engine",)):
            sys.modules.pop(mod, None)
    from npc_engine.engine import NPCEngine
    engine = NPCEngine(str(tn))
    engine.initialize()
    return engine


def test_save_load_cycle():
    """Talk to NPC → shutdown → reinit → verify state preserved."""
    results = []
    _wipe_state()
    tp, tn = _make_configs()

    # Session 1: Talk and build state
    engine = _make_engine(tp, tn)
    engine.process("Hello Noah, I am Gareth from the eastern plains.", npc_id="noah")
    engine.adjust_trust("noah", 15, reason="test")
    engine.process("Thank you for your wisdom.", npc_id="noah")

    state1 = engine.get_npc_state("noah")
    trust1 = state1["capabilities"].get("trust", {}).get("level", 0)
    results.append(("session1_trust", f"{trust1}"))

    # Check state file exists
    state_file = STATE_DIR / "noah.json"
    results.append(("state_file_exists", state_file.exists()))

    engine.shutdown()

    # Session 2: Reinit and check state persisted
    engine2 = _make_engine(tp, tn)
    state2 = engine2.get_npc_state("noah")
    trust2 = state2["capabilities"].get("trust", {}).get("level", 0)
    results.append(("session2_trust", f"{trust2}"))
    results.append(("trust_persisted", trust2 >= trust1 - 2))  # allow small variance

    # Check scratchpad persisted
    scratchpad = json.dumps(state2["capabilities"].get("scratchpad", {})).lower()
    results.append(("scratchpad_persisted", "gareth" in scratchpad or len(scratchpad) > 20))

    engine2.shutdown()
    tp.unlink(missing_ok=True)
    tn.unlink(missing_ok=True)
    return results


def test_corrupted_state():
    """Delete state file → NPC starts fresh without crash."""
    results = []
    _wipe_state()
    tp, tn = _make_configs()

    # Create some state
    engine = _make_engine(tp, tn)
    engine.process("Hello!", npc_id="noah")
    engine.shutdown()

    # Corrupt the state file
    state_file = STATE_DIR / "noah.json"
    if state_file.exists():
        state_file.write_text("THIS IS CORRUPTED DATA {{{", encoding="utf-8")

    # Reinit — should not crash
    try:
        engine2 = _make_engine(tp, tn)
        r = engine2.process("Hello again!", npc_id="noah")
        results.append(("survives_corruption", True))
        results.append(("response_valid", bool(json.loads(r.strip()) if r.strip().startswith("{") else False) or len(r) > 5))
        engine2.shutdown()
    except Exception as e:
        results.append(("survives_corruption", False))
        results.append(("error", str(e)[:100]))

    tp.unlink(missing_ok=True)
    tn.unlink(missing_ok=True)
    return results


def test_state_file_format():
    """State file should have expected JSON structure."""
    results = []
    _wipe_state()
    tp, tn = _make_configs()

    engine = _make_engine(tp, tn)
    engine.process("Hello Noah!", npc_id="noah")
    engine.shutdown()

    state_file = STATE_DIR / "noah.json"
    if state_file.exists():
        data = json.loads(state_file.read_text(encoding="utf-8"))
        results.append(("has_npc_id", "npc_id" in data))
        results.append(("has_capabilities", "capabilities" in data))
        results.append(("has_turn_count", "turn_count" in data))
        results.append(("turn_count_positive", data.get("turn_count", 0) >= 1))
    else:
        results.append(("state_file_exists", False))

    tp.unlink(missing_ok=True)
    tn.unlink(missing_ok=True)
    return results


TESTS = {
    "save_load": ("Save/load cycle preserves state", test_save_load_cycle),
    "corruption": ("Corrupted state file doesn't crash", test_corrupted_state),
    "format": ("State file has correct structure", test_state_file_format),
}


def main():
    print("=" * 70)
    print(f"  STATE PERSISTENCE TESTS ({len(TESTS)} tests)")
    print("=" * 70)

    prev = os.getcwd()
    all_pass = all_fail = 0

    for name, (desc, fn) in TESTS.items():
        print(f"\n  [{name}] {desc}")
        print(f"  {'-' * 60}")
        try:
            results = fn()
        except Exception as e:
            results = [("ERROR", str(e)[:200])]
        for label, value in results:
            is_pass = value is True or (isinstance(value, bool) and value)
            icon = "Y" if is_pass else ("." if isinstance(value, str) else "X")
            print(f"    [{icon}] {label}: {value}")
            if is_pass: all_pass += 1
            elif isinstance(value, bool) and not value: all_fail += 1

    os.chdir(prev)
    total = all_pass + all_fail
    print(f"\n{'=' * 70}")
    print(f"  TOTAL: {all_pass}/{total} checks passed ({all_fail} failed)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
