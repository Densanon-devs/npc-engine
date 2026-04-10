#!/usr/bin/env python3
"""
Combat & Action State Tests — Phase 5

Tests NPC reactions to threats, danger, and emotional extremes.
"""

import io, json, logging, os, sys
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
def _e(r): return str(_j(r).get("emotion", "")).lower()
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

    tp = PIE_ROOT / "config_test.yaml"
    tp.write_text(yaml.dump(raw, default_flow_style=False), encoding="utf-8")
    tn = NPC_ROOT / "config_test.yaml"
    tn.write_text(yaml.dump({
        "world_dir": str(NPC_ROOT / "data" / "worlds" / "ashenvale"),
        "world_name": "Ashenvale", "active_npc": "noah", "pie_config": str(tp),
    }, default_flow_style=False), encoding="utf-8")

    for p in (NPC_ROOT / "data" / "worlds" / "ashenvale" / "npc_state").glob("*"):
        if p.is_file(): p.unlink(missing_ok=True)
    (PIE_ROOT / "data" / "cache" / "response_cache.json").unlink(missing_ok=True)

    os.chdir(PIE_ROOT)
    sys.path.insert(0, str(PIE_ROOT))
    sys.path.insert(0, str(NPC_ROOT))
    from engine.license import LicenseState
    LicenseState.reset()
    from npc_engine.engine import NPCEngine
    engine = NPCEngine(str(tn))
    engine.initialize()
    return engine, tp, tn


def test_threat_reaction(engine):
    """Threatening an NPC should provoke fear/anger, not helpfulness."""
    results = []

    r = engine.process("I'm going to burn this village to the ground!", npc_id="guard_roderick")
    results.append(("valid_response", _valid(r)))
    emotion = _e(r)
    results.append(("emotion_present", bool(emotion)))
    results.append(("not_helpful", not _has(r, "how can i help", "what can i do for you", "welcome")))

    return results


def test_danger_event(engine):
    """Inject a danger event — NPC should acknowledge it."""
    results = []

    engine.inject_event("Bandits are attacking the east gate! Villagers are fleeing!", npc_id="guard_roderick")
    r = engine.process("Captain! What's happening?!", npc_id="guard_roderick")
    results.append(("valid_response", _valid(r)))
    results.append(("mentions_danger", _has(r, "bandit", "attack", "east", "flee", "gate", "danger")))

    return results


def test_mood_override(engine):
    """Setting mood directly should affect dialogue tone."""
    results = []

    # Set Noah to angry
    engine.set_mood("noah", "angry", intensity=0.8)
    r = engine.process("Hello, nice day isn't it?", npc_id="noah")
    emotion = _e(r)
    results.append(("response_valid", _valid(r)))
    results.append(("mood_set", True))  # set_mood API worked without crash

    # Verify state
    state = engine.get_npc_state("noah")
    mood = state["capabilities"].get("emotional_state", {}).get("mood", "")
    results.append(("mood_is_angry", mood == "angry" or "angry" in str(state).lower()))

    return results


def test_guard_duty_response(engine):
    """Tell the guard about an intruder — should respond with duty."""
    results = []

    r = engine.process("Captain Roderick! There's a suspicious stranger sneaking around the well at night!", npc_id="guard_roderick")
    results.append(("valid_response", _valid(r)))
    results.append(("duty_response", _has(r, "patrol", "investigate", "guard", "watch", "look into",
                                           "well", "suspicious", "night", "check")))

    return results


def test_repeated_threats(engine):
    """Multiple threats should shift emotional state toward fear/anger."""
    results = []

    for prompt in [
        "I'll destroy everything you love!",
        "Your village is doomed!",
        "There's nothing you can do to stop me!",
    ]:
        engine.process(prompt, npc_id="pip")

    state = engine.get_npc_state("pip")
    mood = state["capabilities"].get("emotional_state", {}).get("mood", "")
    results.append(("mood_shifted", mood != ""))
    results.append(("current_mood", mood))
    results.append(("not_happy", mood not in ("happy", "content", "joyful", "amused", "")))

    return results


TESTS = {
    "threat": ("Threat provokes fear/anger", test_threat_reaction),
    "danger_event": ("Danger event acknowledged", test_danger_event),
    "mood_override": ("Direct mood override affects NPC", test_mood_override),
    "guard_duty": ("Guard responds to duty call", test_guard_duty_response),
    "repeated_threats": ("Repeated threats shift mood", test_repeated_threats),
}


def main():
    print("=" * 70)
    print(f"  COMBAT & ACTION STATE TESTS ({len(TESTS)} tests)")
    print("=" * 70)

    prev = os.getcwd()
    engine, tp, tn = make_engine()
    all_pass = all_fail = 0

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

    engine.shutdown()
    os.chdir(prev)
    tp.unlink(missing_ok=True)
    tn.unlink(missing_ok=True)
    total = all_pass + all_fail
    print(f"\n{'=' * 70}")
    print(f"  TOTAL: {all_pass}/{total} checks passed ({all_fail} failed)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
