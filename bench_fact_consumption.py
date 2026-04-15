#!/usr/bin/env python3
"""
Story Director — NPC dialogue fact-consumption bench.

Step 2 of the "Next steps — pick up here" direction in FINDINGS.md.

For a given world + narration mode, this script:
  1. Boots the engine
  2. Captures each NPC's build_context() output BEFORE any Director activity
  3. Runs N Director ticks (no player scripting, just the overseer)
  4. Captures each NPC's build_context() output AFTER
  5. Reports per-NPC and world-wide:
     - char / token delta
     - section-level breakdown (Facts / Personal / Events / Quests)
     - how many Director-injected facts actually reach the prompt vs how
       many are sliced off by the ``world_facts[:6]`` / ``personal_knowledge[:4]``
       caps in ``NPCKnowledge.build_context``.

The measurement exists because the Director adds facts via
``engine.add_knowledge(npc_id, fact, fact_type)``, which appends to the
end of the list. ``build_context`` then slices the FIRST 6 world_facts
and FIRST 4 personal_knowledge items — meaning new Director content at
the tail can be invisible to NPC dialogue. Events are taken as
``events[-3:]`` and DO surface new content. This bench quantifies all
three effects.

Usage:
    python bench_fact_consumption.py                    # 10 ticks, ashenvale, prose, qwen_3b
    python bench_fact_consumption.py --world port_blackwater --narration-mode terse
    python bench_fact_consumption.py --ticks 15 --actions-per-tick 3 --log logs/facts.json
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

NPC_ROOT = Path(__file__).parent.resolve()
PIE_ROOT = (NPC_ROOT.parent / "plug-in-intelligence-engine").resolve()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NPC_ENGINE_DEV_MODE", "1")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for n in ["httpx", "huggingface_hub", "sentence_transformers", "faiss",
          "tqdm", "llama_cpp", "engine.npc_knowledge", "engine.npc_capabilities"]:
    logging.getLogger(n).setLevel(logging.ERROR)

sys.path.insert(0, str(NPC_ROOT))
sys.path.insert(0, str(PIE_ROOT))


# Reuse the same MODELS / WORLDS tables + boot logic shape as
# bench_story_director.py, but stripped to what this bench needs.
MODELS = {
    "qwen_05b": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "qwen_3b":  "qwen2.5-3b-instruct-q4_k_m.gguf",
    "llama_1b": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "llama_3b": "llama-3.2-3b-instruct-q4_k_m.gguf",
}

WORLDS = {
    "ashenvale": {
        "world_dir_rel":    "data/worlds/ashenvale",
        "world_name":       "Ashenvale",
        "active_npc":       "noah",
        "story_runtime_rel": "data/story_director",
    },
    "port_blackwater": {
        "world_dir_rel":    "data/worlds/port_blackwater",
        "world_name":       "Port Blackwater",
        "active_npc":       "captain_reva",
        "story_runtime_rel": "data/worlds/port_blackwater/story",
    },
}


def find_model(choice: str) -> Path:
    models_dir = PIE_ROOT / "models"
    preferred = models_dir / MODELS.get(choice, MODELS["qwen_3b"])
    if preferred.exists():
        return preferred
    for name in MODELS.values():
        candidate = models_dir / name
        if candidate.exists():
            print(f"  [info] preferred model '{preferred.name}' not found, falling back to '{candidate.name}'")
            return candidate
    raise FileNotFoundError(f"No GGUF models found in {models_dir}")


def boot_engine(model_path: Path, world_spec: dict):
    import yaml

    world_dir = NPC_ROOT / world_spec["world_dir_rel"]
    story_runtime_dir = NPC_ROOT / world_spec["story_runtime_rel"]

    # Always reset — fact-consumption measurement is meaningless if the
    # baseline state carries over from a prior session.
    npc_state_dir = world_dir / "npc_state"
    if npc_state_dir.exists():
        for p in npc_state_dir.glob("*"):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass
    for runtime_file in (
        story_runtime_dir / "state.json",
        story_runtime_dir / "fact_ledger.json",
        story_runtime_dir / "fact_ledger.embeddings.npy",
        story_runtime_dir / "arcs.json",
    ):
        if runtime_file.exists():
            try:
                runtime_file.unlink()
            except Exception:
                pass
    pie_cache = PIE_ROOT / "data" / "cache" / "response_cache.json"
    if pie_cache.exists():
        try:
            pie_cache.unlink()
        except Exception:
            pass
    player_quests = world_dir / "player_quests.yaml"
    if player_quests.exists():
        try:
            shutil.copy(player_quests, player_quests.with_suffix(".yaml.bak"))
        except Exception:
            pass

    raw = yaml.safe_load((PIE_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["base_model"]["path"] = str(model_path)
    raw["base_model"]["context_length"] = 4096
    raw["base_model"]["temperature"] = 0.6
    raw["fusion"] = raw.get("fusion") or {}
    raw["fusion"]["chat_format"] = "chatml"
    raw["npc"] = raw.get("npc") or {}
    raw["npc"]["enabled"] = True
    raw["npc"]["profiles_dir"] = str(world_dir / "npc_profiles")
    raw["npc"]["state_dir"] = str(world_dir / "npc_state")

    temp_pie = PIE_ROOT / "config_bench_facts.yaml"
    temp_pie.write_text(yaml.dump(raw, default_flow_style=False), encoding="utf-8")

    npc_cfg = {
        "world_dir": str(world_dir),
        "world_name": world_spec["world_name"],
        "active_npc": world_spec["active_npc"],
        "pie_config": str(temp_pie),
    }
    temp_npc = NPC_ROOT / "config_bench_facts.yaml"
    temp_npc.write_text(yaml.dump(npc_cfg, default_flow_style=False), encoding="utf-8")

    os.chdir(PIE_ROOT)
    try:
        from engine.license import LicenseState
        LicenseState.reset()
    except Exception:
        pass

    from npc_engine.engine import NPCEngine
    engine = NPCEngine(str(temp_npc))
    engine.initialize()
    return engine, temp_pie, temp_npc


def _count_tokens(engine, text: str) -> int:
    """Prefer the actual GGUF tokenizer; fall back to ~4 chars/token."""
    try:
        bm = engine.pie.base_model
        if hasattr(bm, "count_tokens"):
            return bm.count_tokens(text)
    except Exception:
        pass
    return max(1, len(text) // 4)


def _section_breakdown(context: str) -> dict[str, int]:
    """
    Break a ``build_context`` output string into named sections, by char
    count. The format is line-delimited ``[Section: ...]`` blocks, where
    ``Section`` is one of ``You are`` / ``Speech`` / ``Facts`` /
    ``Personal`` / ``YOUR QUEST`` / ``PLAYER IS WORKING ON`` / ``PLAYER
    COMPLETED`` / ``RECENT NEWS``, plus capability blocks. We bucket
    anything that doesn't start with ``[`` under ``other``.
    """
    sections = {
        "identity":       0,  # [You are ...]  + [Speech: ...]
        "facts":          0,  # [Facts: ...]
        "personal":       0,  # [Personal: ...]
        "quests":         0,  # [YOUR QUEST: ...]
        "player_quests":  0,  # [PLAYER IS WORKING ON ...] + [PLAYER COMPLETED ...]
        "events":         0,  # [RECENT NEWS ...]
        "capabilities":   0,  # capability-injected blocks
        "other":          0,
    }
    for line in context.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        cost = len(line)
        if stripped.startswith("[You are") or stripped.startswith("[Speech:"):
            sections["identity"] += cost
        elif stripped.startswith("[Facts:"):
            sections["facts"] += cost
        elif stripped.startswith("[Personal:"):
            sections["personal"] += cost
        elif stripped.startswith("[YOUR QUEST:"):
            sections["quests"] += cost
        elif stripped.startswith("[PLAYER IS WORKING ON") or stripped.startswith("[PLAYER COMPLETED"):
            sections["player_quests"] += cost
        elif stripped.startswith("[RECENT NEWS"):
            sections["events"] += cost
        elif stripped.startswith("["):
            # Heuristic: anything else in a single-line [Label: ...] form
            # is probably a capability block (trust/emotion/goals/etc).
            sections["capabilities"] += cost
        else:
            sections["other"] += cost
    return sections


def _world_name(engine) -> str:
    try:
        return engine.config.world_name or ""
    except Exception:
        return ""


def snapshot_npcs(engine) -> dict[str, dict]:
    """
    Build a per-NPC record of the current ``build_context`` output plus
    the underlying counts that drive it. The same shape is captured
    both before the Director runs and after, so deltas are trivially
    computed.

    Director-injected facts route into ``dynamic_world_facts`` /
    ``dynamic_personal_knowledge``; the static profile lists don't
    grow at runtime. The bench tracks both, plus the combined view,
    so the slice-survival math knows the difference.
    """
    pie = engine.pie
    out = {}
    for npc_id, npc in pie.npc_knowledge.profiles.items():
        context = npc.build_context(
            include_quests=True,
            include_events=True,
            world_name=_world_name(engine),
        )
        token_count = _count_tokens(engine, context)
        record = {
            "npc_id": npc_id,
            "context": context,
            "char_count": len(context),
            "token_count": token_count,
            "world_facts_total": len(npc.world_facts),
            "personal_total":    len(npc.personal_knowledge),
            "dynamic_world_total":    len(getattr(npc, "dynamic_world_facts", []) or []),
            "dynamic_personal_total": len(getattr(npc, "dynamic_personal_knowledge", []) or []),
            "events_total":      len(npc.events),
            "quests_total":      len(npc.quests),
            # Snapshot list contents so the delta pass can tell which
            # items were added by the Director vs pre-existing.
            "world_facts":       list(npc.world_facts),
            "personal":          list(npc.personal_knowledge),
            "dynamic_world":     list(getattr(npc, "dynamic_world_facts", []) or []),
            "dynamic_personal":  list(getattr(npc, "dynamic_personal_knowledge", []) or []),
            "events":            [e.description for e in npc.events],
            "sections":          _section_breakdown(context),
        }
        out[npc_id] = record
    return out


# Mirror NPCKnowledge.build_context's interleave parameters so the
# bench's slice-survival accounting matches the production rule.
_WORLD_TOTAL_CAP = 6
_WORLD_DYNAMIC_RESERVE_MIN = 2
_PERSONAL_TOTAL_CAP = 4
_PERSONAL_DYNAMIC_RESERVE_MIN = 2


def _dynamic_in_prompt(static_count: int, dynamic_count: int,
                       total_cap: int, reserve_min: int) -> int:
    """How many dynamic items reach the prompt under the interleave rule?"""
    if dynamic_count <= 0:
        return 0
    dyn_slots = max(reserve_min, total_cap - static_count)
    return min(dyn_slots, dynamic_count, total_cap)


def compute_delta(before: dict, after: dict) -> dict:
    """For one NPC, compute what changed between the two snapshots."""
    # Director writes only to the dynamic lanes now, so growth in the
    # static lists is unexpected (would mean a profile re-load or a
    # caller bypassing engine.add_knowledge). Track both for safety.
    added_static_world    = after["world_facts"][len(before["world_facts"]):]
    added_static_personal = after["personal"][len(before["personal"]):]
    added_dynamic_world =       after["dynamic_world"][len(before["dynamic_world"]):]
    added_dynamic_personal =    after["dynamic_personal"][len(before["dynamic_personal"]):]
    added_world_facts = added_static_world + added_dynamic_world
    added_personal    = added_static_personal + added_dynamic_personal
    added_events      = after["events"][len(before["events"]):]

    # Slice-survival under the interleave rule. Static items in the
    # baseline don't move; dynamic items reach the prompt only if they
    # sit in the dynamic-reserved tail slots.
    static_world_count    = after["world_facts_total"]
    static_personal_count = after["personal_total"]
    dynamic_world_in_prompt = _dynamic_in_prompt(
        static_world_count, len(added_dynamic_world),
        _WORLD_TOTAL_CAP, _WORLD_DYNAMIC_RESERVE_MIN,
    )
    dynamic_personal_in_prompt = _dynamic_in_prompt(
        static_personal_count, len(added_dynamic_personal),
        _PERSONAL_TOTAL_CAP, _PERSONAL_DYNAMIC_RESERVE_MIN,
    )
    # Static appends (rare/unexpected) get the same forward-slice
    # treatment they used to: the first N items reach the prompt up
    # to the cap, anything past the cap is sliced.
    static_world_survives = max(0, _WORLD_TOTAL_CAP - len(before["world_facts"]))
    static_personal_survives = max(0, _PERSONAL_TOTAL_CAP - len(before["personal"]))
    static_world_in_prompt    = min(len(added_static_world),    static_world_survives)
    static_personal_in_prompt = min(len(added_static_personal), static_personal_survives)

    added_world_in_prompt    = dynamic_world_in_prompt    + static_world_in_prompt
    added_personal_in_prompt = dynamic_personal_in_prompt + static_personal_in_prompt
    # Events use [-3:] so the last 3 always reach the prompt.
    added_events_in_prompt = min(len(added_events), 3)

    section_delta = {
        k: after["sections"].get(k, 0) - before["sections"].get(k, 0)
        for k in set(after["sections"]) | set(before["sections"])
    }
    return {
        "char_delta":  after["char_count"]  - before["char_count"],
        "token_delta": after["token_count"] - before["token_count"],
        "added_world_facts": len(added_world_facts),
        "added_personal":    len(added_personal),
        "added_dynamic_world":    len(added_dynamic_world),
        "added_dynamic_personal": len(added_dynamic_personal),
        "added_events":      len(added_events),
        "added_quests":      after["quests_total"] - before["quests_total"],
        "added_world_in_prompt":    added_world_in_prompt,
        "added_personal_in_prompt": added_personal_in_prompt,
        "added_events_in_prompt":   added_events_in_prompt,
        "world_facts_sliced_off": max(0, len(added_world_facts) - added_world_in_prompt),
        "personal_sliced_off":    max(0, len(added_personal)    - added_personal_in_prompt),
        "section_delta": section_delta,
        "added_world_list":    added_world_facts,
        "added_personal_list": added_personal,
        "added_events_list":   added_events,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=10)
    parser.add_argument("--model", choices=list(MODELS.keys()), default="qwen_3b")
    parser.add_argument("--world", choices=list(WORLDS.keys()), default="ashenvale")
    parser.add_argument("--narration-mode", choices=["prose", "terse"], default="prose")
    parser.add_argument("--actions-per-tick", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override the Director's per-tick token cap. "
                             "Defaults to 120 (terse) or 400 (prose).")
    parser.add_argument("--log", type=str, default=None)
    args = parser.parse_args()

    model_path = find_model(args.model)
    world_spec = WORLDS[args.world]
    print(f"Model:           {model_path.name}")
    print(f"World:           {args.world} ({world_spec['world_name']})")
    print(f"Narration mode:  {args.narration_mode}")
    print(f"Director ticks:  {args.ticks}  (actions per tick: {args.actions_per_tick})")
    print("-" * 72)

    engine, tmp_pie, tmp_npc = boot_engine(model_path, world_spec)
    try:
        engine.story_director.set_narration_mode(args.narration_mode)
        print(f"Lore file:       {engine.story_director._lore_file}")
        print(f"Examples file:   {engine.story_director._examples_file} "
              f"({len(engine.story_director._examples)} entries)")

        # ── BEFORE snapshot ───────────────────────────────────────
        before = snapshot_npcs(engine)
        print("\n=== BEFORE (Director inactive) ===")
        for npc_id, rec in before.items():
            print(f"  {npc_id:20} "
                  f"chars={rec['char_count']:5d}  "
                  f"tokens={rec['token_count']:4d}  "
                  f"facts={rec['world_facts_total']:2d}  "
                  f"personal={rec['personal_total']:2d}  "
                  f"events={rec['events_total']:2d}  "
                  f"quests={rec['quests_total']:2d}")

        # ── Run ticks ─────────────────────────────────────────────
        print(f"\n=== Running {args.ticks} Director ticks ===")
        t0 = time.monotonic()
        tick_results = []
        for i in range(args.ticks):
            tick_num = i + 1
            tick_t0 = time.monotonic()
            result = engine.story_director.tick(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                actions_per_tick=args.actions_per_tick,
            )
            elapsed = time.monotonic() - tick_t0
            sub_actions = result.get("sub_actions", [])
            if not sub_actions:
                sub_actions = [{
                    "action": result.get("action") or {},
                    "dispatch": result.get("dispatch") or {},
                }]
            outcomes = []
            for s in sub_actions:
                a = s.get("action") or {}
                d = s.get("dispatch") or {}
                kind = a.get("action", "?")
                ok = bool(d.get("ok"))
                outcomes.append(f"{kind}{'✓' if ok else '✗'}")
            print(f"  T{tick_num:2d}  {elapsed:5.2f}s   {'  '.join(outcomes)}")
            tick_results.append({
                "tick": tick_num,
                "elapsed": round(elapsed, 3),
                "outcomes": outcomes,
            })
        total_elapsed = time.monotonic() - t0
        print(f"  total: {total_elapsed:.2f}s  "
              f"mean: {total_elapsed / args.ticks:.2f}s/tick")

        # ── AFTER snapshot ────────────────────────────────────────
        after = snapshot_npcs(engine)
        print("\n=== AFTER ===")
        for npc_id, rec in after.items():
            print(f"  {npc_id:20} "
                  f"chars={rec['char_count']:5d}  "
                  f"tokens={rec['token_count']:4d}  "
                  f"facts={rec['world_facts_total']:2d}  "
                  f"personal={rec['personal_total']:2d}  "
                  f"events={rec['events_total']:2d}  "
                  f"quests={rec['quests_total']:2d}")

        # ── Deltas ────────────────────────────────────────────────
        deltas = {
            npc_id: compute_delta(before[npc_id], after[npc_id])
            for npc_id in before
        }

        print("\n=== DELTAS PER NPC ===")
        for npc_id, d in deltas.items():
            print(f"  {npc_id}")
            print(f"    chars:  {d['char_delta']:+5d}   tokens: {d['token_delta']:+4d}")
            print(f"    added:  world={d['added_world_facts']}  "
                  f"personal={d['added_personal']}  "
                  f"events={d['added_events']}  "
                  f"quests={d['added_quests']}")
            print(f"    reached prompt: "
                  f"world={d['added_world_in_prompt']}/{d['added_world_facts']}  "
                  f"personal={d['added_personal_in_prompt']}/{d['added_personal']}  "
                  f"events={d['added_events_in_prompt']}/{d['added_events']}")
            if d["world_facts_sliced_off"] or d["personal_sliced_off"]:
                print(f"    sliced off: "
                      f"world={d['world_facts_sliced_off']}  "
                      f"personal={d['personal_sliced_off']}")
            nonzero_sections = {k: v for k, v in d["section_delta"].items() if v}
            if nonzero_sections:
                parts = [f"{k}{v:+d}" for k, v in nonzero_sections.items()]
                print(f"    sections: {'  '.join(parts)}")

        # ── Aggregates ────────────────────────────────────────────
        total_char_delta = sum(d["char_delta"] for d in deltas.values())
        total_token_delta = sum(d["token_delta"] for d in deltas.values())
        num_npcs = len(deltas)
        mean_char_delta = total_char_delta / num_npcs if num_npcs else 0
        mean_token_delta = total_token_delta / num_npcs if num_npcs else 0
        total_added_world = sum(d["added_world_facts"] for d in deltas.values())
        total_added_personal = sum(d["added_personal"] for d in deltas.values())
        total_added_events = sum(d["added_events"] for d in deltas.values())
        total_added_quests = sum(d["added_quests"] for d in deltas.values())
        total_sliced_world = sum(d["world_facts_sliced_off"] for d in deltas.values())
        total_sliced_personal = sum(d["personal_sliced_off"] for d in deltas.values())

        print("\n" + "=" * 72)
        print("SUMMARY")
        print("=" * 72)
        print(f"World:             {args.world}  ({num_npcs} NPCs)")
        print(f"Narration mode:    {args.narration_mode}")
        print(f"Director ticks:    {args.ticks} x {args.actions_per_tick} actions")
        print(f"Director wall:     {total_elapsed:.2f}s  "
              f"({total_elapsed / args.ticks:.2f}s/tick)")
        print(f"Added (total):     "
              f"world_facts={total_added_world}  "
              f"personal={total_added_personal}  "
              f"events={total_added_events}  "
              f"quests={total_added_quests}")
        print(f"Sliced off cap:    "
              f"world_facts={total_sliced_world}  "
              f"personal={total_sliced_personal}  "
              f"(these never reach NPC dialogue prompts)")
        print(f"Per-NPC mean:      "
              f"chars {mean_char_delta:+.0f}  tokens {mean_token_delta:+.0f}")
        print(f"Per-NPC max:       "
              f"chars {max(d['char_delta'] for d in deltas.values()):+d}  "
              f"tokens {max(d['token_delta'] for d in deltas.values()):+d}")

        # Pass/fail on the "<150 tokens" goal from FINDINGS.md
        max_token = max(d["token_delta"] for d in deltas.values())
        if args.narration_mode == "terse":
            verdict = "PASS" if max_token < 150 else "FAIL"
            print(f"Terse goal <150:   {verdict}  (max delta {max_token})")

        if args.log:
            log_path = Path(args.log)
            if not log_path.is_absolute():
                log_path = NPC_ROOT / log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps({
                "world": args.world,
                "narration_mode": args.narration_mode,
                "model": args.model,
                "ticks": args.ticks,
                "actions_per_tick": args.actions_per_tick,
                "director_wall": total_elapsed,
                "tick_results": tick_results,
                "before": before,
                "after": after,
                "deltas": deltas,
                "summary": {
                    "num_npcs": num_npcs,
                    "total_char_delta": total_char_delta,
                    "total_token_delta": total_token_delta,
                    "mean_char_delta": mean_char_delta,
                    "mean_token_delta": mean_token_delta,
                    "total_added_world": total_added_world,
                    "total_added_personal": total_added_personal,
                    "total_added_events": total_added_events,
                    "total_added_quests": total_added_quests,
                    "total_sliced_world": total_sliced_world,
                    "total_sliced_personal": total_sliced_personal,
                },
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nFull trace written to: {log_path}")

    finally:
        try:
            engine.shutdown()
        except Exception:
            pass
        try:
            tmp_pie.unlink()
            tmp_npc.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
