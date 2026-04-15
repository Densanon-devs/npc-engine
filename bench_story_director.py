#!/usr/bin/env python3
"""
Story Director tick harness — run N ticks against a real model and observe
how the world mutates over a session.

Prints per-tick: snapshot preview, raw LLM response, parsed action, coercion
flag, dispatch result. Ends with a summary of outcome categories.

Usage:
    python bench_story_director.py                 # 10 ticks, 0.5B Qwen
    python bench_story_director.py --ticks 20
    python bench_story_director.py --model qwen_1b
    python bench_story_director.py --reset         # wipe NPC state + director state first

Purpose: expose the failure modes that only emerge over a session (repetition,
schema drift, contradictions, token-budget overflow, etc.) so we can fix them.
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
from collections import Counter
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


MODELS = {
    "qwen_05b": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "qwen_3b":  "qwen2.5-3b-instruct-q4_k_m.gguf",
    "llama_1b": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "llama_3b": "llama-3.2-3b-instruct-q4_k_m.gguf",
    "llama_8b": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "qwen_7b":  "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
}

# Scripted player events: (tick_to_fire_before, text, target_npc, trust_delta).
# Each entry is applied immediately before the given tick number fires, so
# the Director sees it in its world snapshot for that tick. Deliberately
# provocative — different NPCs, mix of positive/negative, distinct themes —
# to test whether the Director reacts to player behavior or keeps running
# its own arc. One script per supported world.
PLAYER_SCRIPTS = {
    "ashenvale": [
        (3, "Player solved Kael's missing-hammers investigation and returned the stolen tools",
         "kael", +20),
        (5, "Player gifted Elara a rare bundle of moonwort they found in the Silverwood",
         "elara", +15),
        (7, "Player publicly accused Mara of selling counterfeit steel in the tavern",
         "mara", -30),
        (9, "Player began asking Noah about the sealed letter from the old king",
         "noah", +5),
    ],
    "port_blackwater": [
        (3, "Player climbed the dark lighthouse at low tide and brought back a melted lens fragment",
         "captain_reva", +20),
        (5, "Player bought Finn supper at the Drowned Rat and listened to his stories",
         "finn", +15),
        (7, "Player publicly accused Old Bones of running contraband through the cellar tunnels",
         "old_bones", -30),
        (9, "Player began asking Captain Reva about the sealed letter from the governor of Meridia",
         "captain_reva", +5),
    ],
}

# Real dialogue turns the player utters mid-session. Each entry is
# (before_tick, player_text, npc_id) — the bench calls engine.process()
# which generates an NPC dialogue response AND auto-feeds the Director
# via the wiring in NPCEngine.process. This exercises the FULL Cardinal
# loop: player → NPC dialogue → director observation → director reaction.
DIALOGUE_SCRIPTS = {
    "ashenvale": [
        (3, "What does that sealed letter from the old king say, Noah? I can be trusted.", "noah"),
        (5, "I know you're hiding something under your floorboards, Mara. Don't lie to me.", "mara"),
        (7, "Tell me about the elven ruins in the Silverwood, Elara. Did your grandmother find them?", "elara"),
        (9, "Have you seen anything strange on the north road, Captain Roderick?", "guard_roderick"),
    ],
    "port_blackwater": [
        (3, "Captain Reva, what does the sealed letter from the governor of Meridia say? I can be trusted.", "captain_reva"),
        (5, "I know you're hiding something under that loose plank on dock three, Finn. Show me.", "finn"),
        (7, "Tell me about the singing you've heard from the Shoals at midnight, Old Bones.", "old_bones"),
        (9, "Have you seen any strange cargo come off the Red Tide ships, Finn?", "finn"),
    ],
}

# Per-world runtime defaults. ``profile_dir``/``state_dir``/``world_yaml``
# all live under ``data/worlds/<world>/``; ``story_runtime_dir`` is where
# the Director persists state + ledger + arcs. For Ashenvale, the Director
# still writes to the legacy ``data/story_director/`` directory (so existing
# runs and tests stay stable). For Port Blackwater, the per-world
# ``story/`` pack takes over and state is isolated there.
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
    # Two-zone PB variant for the zone layer + lifecycle benches.
    # Same lore as the base PB but split across dock_district and
    # lighthouse_bluffs with one mobile NPC (varro). Used for Phase 1
    # zone tests and all Phase 2 lifecycle benches.
    "port_blackwater_zoned": {
        "world_dir_rel":    "data/worlds/port_blackwater_zoned",
        "world_name":       "Port Blackwater",
        "active_npc":       "captain_reva",
        "story_runtime_rel": "data/worlds/port_blackwater_zoned/story",
    },
    # Synthetic worlds for scaling tests. Generate with
    # ``python generate_synthetic_world.py --towns N --npcs-per-town M
    # --output synthetic_<total>``. The active_npc is auto-resolved at
    # boot to the first profile in the directory if the named one
    # doesn't exist (see boot_engine for the fallback).
    "synthetic_25": {
        "world_dir_rel":    "data/worlds/synthetic_25",
        "world_name":       "Synthetic 25",
        "active_npc":       "auto",
        "story_runtime_rel": "data/worlds/synthetic_25/story",
    },
    "synthetic_100": {
        "world_dir_rel":    "data/worlds/synthetic_100",
        "world_name":       "Synthetic 100",
        "active_npc":       "auto",
        "story_runtime_rel": "data/worlds/synthetic_100/story",
    },
    "synthetic_500": {
        "world_dir_rel":    "data/worlds/synthetic_500",
        "world_name":       "Synthetic 500",
        "active_npc":       "auto",
        "story_runtime_rel": "data/worlds/synthetic_500/story",
    },
}


def find_model(choice: str) -> Path:
    models_dir = PIE_ROOT / "models"
    preferred = models_dir / MODELS.get(choice, MODELS["qwen_05b"])
    if preferred.exists():
        return preferred
    # Fall back: any gguf that exists
    for name in MODELS.values():
        candidate = models_dir / name
        if candidate.exists():
            print(f"  [info] preferred model '{preferred.name}' not found, falling back to '{candidate.name}'")
            return candidate
    raise FileNotFoundError(f"No GGUF models found in {models_dir}")


def boot_engine(model_path: Path, reset: bool, world_spec: dict):
    import yaml

    world_dir = NPC_ROOT / world_spec["world_dir_rel"]
    story_runtime_dir = NPC_ROOT / world_spec["story_runtime_rel"]

    # Reset state if asked
    if reset:
        state_dir = world_dir / "npc_state"
        if state_dir.exists():
            for p in state_dir.glob("*"):
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
        # Clear PIE's response cache so dialogue calls actually hit the
        # LLM instead of returning identical replies from the prior run.
        pie_cache = PIE_ROOT / "data" / "cache" / "response_cache.json"
        if pie_cache.exists():
            try:
                pie_cache.unlink()
            except Exception:
                pass
        # Back up the world's player quest file in case prior runs
        # mutated it — player_quests.yaml is re-loaded at init.
        player_quests = world_dir / "player_quests.yaml"
        if player_quests.exists():
            try:
                shutil.copy(player_quests, player_quests.with_suffix(".yaml.bak"))
            except Exception:
                pass

    raw = yaml.safe_load((PIE_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["base_model"]["path"] = str(model_path)
    # Default 4096 fits Ashenvale and Port Blackwater. Synthetic
    # large-world stress runs need more — the world snapshot grows
    # roughly linearly with cast size and a 500-NPC unbounded snapshot
    # is ~9300 tokens before any examples or focus block. 16K is
    # comfortable headroom while still inside Qwen 2.5's 32K training
    # window. Bumping context here costs RAM (KV cache scales with
    # context*layers) but doesn't affect prompt length.
    raw["base_model"]["context_length"] = 16384
    raw["base_model"]["temperature"] = 0.6
    raw["fusion"] = raw.get("fusion") or {}
    raw["fusion"]["chat_format"] = "chatml"
    raw["npc"] = raw.get("npc") or {}
    raw["npc"]["enabled"] = True
    raw["npc"]["profiles_dir"] = str(world_dir / "npc_profiles")
    raw["npc"]["state_dir"] = str(world_dir / "npc_state")

    temp_pie = PIE_ROOT / "config_bench_story.yaml"
    temp_pie.write_text(yaml.dump(raw, default_flow_style=False), encoding="utf-8")

    # Synthetic worlds use procedural NPC ids; resolve "auto" to the
    # first profile yaml the loader will find. The active_npc only
    # matters for dialogue scripts (which the synthetic stress runs
    # don't use), so any valid id works.
    active_npc = world_spec["active_npc"]
    if active_npc == "auto":
        profiles = sorted((world_dir / "npc_profiles").glob("*.yaml"))
        if not profiles:
            raise RuntimeError(f"No NPC profiles found in {world_dir / 'npc_profiles'}")
        active_npc = profiles[0].stem

    npc_cfg = {
        "world_dir": str(world_dir),
        "world_name": world_spec["world_name"],
        "active_npc": active_npc,
        "pie_config": str(temp_pie),
    }
    temp_npc = NPC_ROOT / "config_bench_story.yaml"
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


def classify(action: dict, dispatch: dict) -> str:
    """Categorize a tick outcome for the summary."""
    if not dispatch.get("ok"):
        kind = action.get("action", "?")
        reason = dispatch.get("reason", "unknown")
        return f"FAIL[{kind}:{reason.split(':')[0][:24]}]"
    kind = dispatch.get("kind", action.get("action", "?"))
    return f"OK[{kind}]"


def _was_coerced(raw: str, action: dict) -> bool:
    """Rough check: did parser coercion change the action label?"""
    if not raw:
        return False
    try:
        import re
        match = re.search(r'"action"\s*:\s*"([^"]+)"', raw)
        if not match:
            return False
        return match.group(1) != action.get("action")
    except Exception:
        return False


def _snapshot_world(engine) -> dict:
    """Cheap world fingerprint for change tracking."""
    pie = engine.pie
    snap = {
        "quests": {npc_id: [q.id for q in npc.quests]
                   for npc_id, npc in pie.npc_knowledge.profiles.items()},
        "events": {npc_id: len(npc.events)
                   for npc_id, npc in pie.npc_knowledge.profiles.items()},
        "facts": {npc_id: len(npc.world_facts) + len(npc.personal_knowledge)
                  for npc_id, npc in pie.npc_knowledge.profiles.items()},
    }
    return snap


def _diff_world(before: dict, after: dict) -> list[str]:
    changes = []
    for npc_id in after["quests"]:
        new_quests = set(after["quests"][npc_id]) - set(before["quests"].get(npc_id, []))
        if new_quests:
            changes.append(f"+quest({npc_id}): {', '.join(new_quests)}")
        added_events = after["events"][npc_id] - before["events"].get(npc_id, 0)
        if added_events:
            changes.append(f"+events({npc_id}): {added_events}")
        added_facts = after["facts"][npc_id] - before["facts"].get(npc_id, 0)
        if added_facts:
            changes.append(f"+facts({npc_id}): {added_facts}")
    return changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=10)
    parser.add_argument("--model", choices=list(MODELS.keys()), default="qwen_05b")
    parser.add_argument("--world", choices=list(WORLDS.keys()), default="ashenvale",
                        help="World to run the Director on. Each world has its own "
                             "NPC profiles, lore pack, and example library. Defaults "
                             "to Ashenvale (the original pack).")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe NPC state + story director state before starting")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--log", type=str, default=None,
                        help="Optional path to write full JSON trace")
    parser.add_argument("--player-script", action="store_true",
                        help="Interleave the built-in player script between ticks")
    parser.add_argument("--dialogue-script", action="store_true",
                        help="Interleave real engine.process() dialogue turns between "
                             "ticks. Tests the full player→NPC→director loop via "
                             "auto-feed instead of synthetic record_player_action calls.")
    parser.add_argument("--actions-per-tick", type=int, default=1,
                        help="Number of parallel sub-actions per tick (architect/worker). "
                             "Default 1 = single-action mode. >=2 = each tick plans N "
                             "distinct (focus, kind) slots and runs one LLM call per slot.")
    parser.add_argument("--narration-mode", choices=["prose", "terse"], default="prose",
                        help="Output style. 'prose' = cinematic novel narration (current "
                             "default). 'terse' = short third-person facts under 25 words, "
                             "optimized for downstream NPC dialogue context injection.")
    parser.add_argument("--tick-budget-seconds", type=float, default=0.0,
                        help="Parallel-to-game pacing. After each tick, sleep until at "
                             "least this many seconds have elapsed since the tick started. "
                             "Lets the bench simulate a Director that cooperates with a "
                             "main game loop instead of monopolizing the CPU. 0 = no "
                             "budget (back-to-back ticks, the default).")
    parser.add_argument("--active-zones", type=str, default=None,
                        help="Comma-separated list of zone names to set as "
                             "active for the bench run. Empty/omitted = "
                             "world-wide mode (current behaviour). Only "
                             "meaningful for worlds with NPCs that have "
                             "zone: fields in their profile YAML.")
    args = parser.parse_args()

    model_path = find_model(args.model)
    world_spec = WORLDS[args.world]
    # Synthetic worlds don't ship player/dialogue scripts (procedural NPC
    # ids would make the scripts brittle); they fall back to empty lists,
    # which means --player-script / --dialogue-script become no-ops.
    player_script = PLAYER_SCRIPTS.get(args.world, [])
    dialogue_script = DIALOGUE_SCRIPTS.get(args.world, [])
    print(f"Model: {model_path.name}")
    print(f"World: {args.world} ({world_spec['world_name']})")
    print(f"Ticks: {args.ticks}")
    print(f"Reset: {args.reset}")
    print("-" * 72)

    engine, tmp_pie, tmp_npc = boot_engine(model_path, args.reset, world_spec)
    engine.story_director.set_narration_mode(args.narration_mode)
    print(f"Narration mode: {args.narration_mode}")
    if args.active_zones:
        zone_list = [z.strip() for z in args.active_zones.split(",") if z.strip()]
        engine.story_director.set_active_zones(zone_list)
        print(f"Active zones: {zone_list}")
    print(f"Lore file: {engine.story_director._lore_file}")
    print(f"Examples:  {engine.story_director._examples_file} "
          f"({len(engine.story_director._examples)} entries)")
    print(f"State dir: {engine.story_director._runtime_dir}")
    print(f"Profiles loaded: {len(engine.pie.npc_knowledge.profiles)}")

    # Optional psutil for peak-RSS instrumentation. Bench still runs
    # without it; the parallel-to-game scaling reports just lose the
    # memory column.
    try:
        import psutil
        _proc = psutil.Process()
    except Exception:
        _proc = None

    def _sample_rss_mb() -> float:
        if _proc is None:
            return 0.0
        try:
            return _proc.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    rss_baseline_mb = _sample_rss_mb()
    print(f"RSS baseline: {rss_baseline_mb:.0f} MB"
          + (" (psutil unavailable, all RSS metrics will read 0)" if _proc is None else ""))

    try:
        outcomes = Counter()
        timings = []
        trace = []
        coerce_count = 0
        action_kinds = Counter()
        action_targets = Counter()
        player_events_log: list[dict] = []
        dialogue_events_log: list[dict] = []
        dialogue_total_time = 0.0
        similarity_warnings: list[dict] = []
        arc_events: list[dict] = []  # (tick, event, arc_id, detail)
        snapshot_lengths: list[int] = []
        rss_samples_mb: list[float] = []

        for i in range(args.ticks):
            tick_num = i + 1

            # Apply any scripted player actions that fire before this tick
            if args.player_script:
                for entry in player_script:
                    if entry[0] != tick_num:
                        continue
                    before_tick, text, target, trust_delta = entry
                    print(f"\n[PLAYER before T{tick_num}] {text}  "
                          f"(target={target}, trust {trust_delta:+d})")
                    engine.story_director.record_player_action(
                        text, target=target, trust_delta=trust_delta,
                    )
                    player_events_log.append({
                        "before_tick": tick_num,
                        "text": text, "target": target, "trust_delta": trust_delta,
                    })

            # Real dialogue turns — call engine.process which generates an
            # NPC reply AND auto-feeds the Director through the wiring in
            # NPCEngine.process. This is the FULL player→NPC→director loop.
            if args.dialogue_script:
                for entry in dialogue_script:
                    if entry[0] != tick_num:
                        continue
                    before_tick, player_text, npc_id = entry
                    print(f"\n[DIALOGUE before T{tick_num}] -> {npc_id}: {player_text}")
                    d_t0 = time.monotonic()
                    try:
                        npc_response = engine.process(player_text, npc_id=npc_id)
                    except Exception as e:
                        npc_response = f"[error: {e}]"
                    d_elapsed = time.monotonic() - d_t0
                    dialogue_total_time += d_elapsed
                    print(f"  npc reply ({d_elapsed:.2f}s): {str(npc_response)[:160]}")
                    dialogue_events_log.append({
                        "before_tick": tick_num,
                        "player_text": player_text,
                        "npc_id": npc_id,
                        "npc_response": str(npc_response)[:400],
                        "elapsed": round(d_elapsed, 3),
                    })

            print(f"\n=== TICK {tick_num} ===")
            before = _snapshot_world(engine)
            # Sample the world snapshot length BEFORE the tick — this
            # is the prompt-bound input the LLM gets each tick. Grows
            # roughly linearly with cast size on the unbounded snapshot
            # path, which is the bottleneck for 100+ NPC worlds.
            try:
                snap_text = engine.story_director._world_snapshot()
                snap_chars = len(snap_text)
                snap_tokens_est = snap_chars // 4
                snapshot_lengths.append(snap_chars)
                if tick_num == 1 or tick_num % 10 == 0 or args.ticks <= 5:
                    print(f"  snapshot: {snap_chars} chars (~{snap_tokens_est} tok)")
            except Exception as e:
                print(f"  (snapshot length sample failed: {e})")
            # Snapshot all active arcs before the tick so we can diff
            # the set and show proposals / beat advances / resolutions
            # per arc. With multi-arc support, we may see multiple arcs
            # in flight and multiple events per tick.
            arc_before_snap = {
                a.id: (a.current_beat, a.status)
                for a in engine.story_director.arc_planner.active_arcs()
            }
            rss_samples_mb.append(_sample_rss_mb())
            t0 = time.monotonic()
            result = engine.story_director.tick(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                actions_per_tick=args.actions_per_tick,
            )
            elapsed = time.monotonic() - t0
            timings.append(elapsed)
            # Tick budget for parallel-to-game simulation. Sleep so the
            # tick-to-tick interval respects the budget, but never sleep
            # negative time if the tick already overran.
            if args.tick_budget_seconds > 0 and elapsed < args.tick_budget_seconds:
                time.sleep(args.tick_budget_seconds - elapsed)

            arc_after = {
                a.id: a
                for a in engine.story_director.arc_planner.active_arcs()
            }

            # New arcs — proposed this tick
            for arc_id, arc in arc_after.items():
                if arc_id not in arc_before_snap:
                    print(f"  [ARC] PROPOSED {arc.id}")
                    print(f"        theme: {arc.theme[:100]}")
                    print(f"        cast:  {arc.focus_npcs}")
                    print(f"        beat:  {arc.current_beat + 1}/"
                          f"{len(arc.beat_goals)} ({arc.current_beat_label})")
                    arc_events.append({
                        "tick": tick_num, "event": "proposed",
                        "arc_id": arc.id, "theme": arc.theme,
                        "focus_npcs": arc.focus_npcs,
                    })

            # Resolved arcs — in before snapshot, not in after
            for arc_id in arc_before_snap:
                if arc_id not in arc_after:
                    resolved = next(
                        (a for a in engine.story_director.arc_planner.arcs
                         if a.id == arc_id),
                        None,
                    )
                    status = resolved.status if resolved else "unknown"
                    print(f"  [ARC] RESOLVED {arc_id} (status={status})")
                    arc_events.append({
                        "tick": tick_num, "event": "resolved",
                        "arc_id": arc_id, "status": status,
                    })

            # Beat advances on still-active arcs
            for arc_id, arc in arc_after.items():
                if arc_id in arc_before_snap:
                    prior_beat = arc_before_snap[arc_id][0]
                    if prior_beat != arc.current_beat:
                        print(f"  [ARC] {arc_id} BEAT {prior_beat + 1} -> "
                              f"{arc.current_beat + 1} "
                              f"({arc.current_beat_label})")
                        arc_events.append({
                            "tick": tick_num, "event": "beat_advanced",
                            "arc_id": arc.id,
                            "from_beat": prior_beat,
                            "to_beat": arc.current_beat,
                        })

            # Quiet status line for every currently-active arc
            if arc_after:
                status_parts = [
                    f"{a.id} beat {a.current_beat + 1}/{len(a.beat_goals)} "
                    f"({a.current_beat_label})"
                    for a in arc_after.values()
                ]
                print(f"  [ARC] {' | '.join(status_parts)}")

            # Multi-action ticks return a sub_actions list; single-action
            # ticks return action/dispatch at the top level. Normalize.
            if "sub_actions" in result:
                sub_results = result["sub_actions"]
            else:
                sub_results = [{
                    "focus_npc": result.get("action", {}).get("npc_id")
                                  or result.get("action", {}).get("target"),
                    "action_kind": result.get("action", {}).get("action"),
                    "action": result.get("action") or {},
                    "dispatch": result.get("dispatch") or {},
                    "raw_response": result.get("raw_response") or "",
                }]

            after = _snapshot_world(engine)
            world_changes = _diff_world(before, after)

            print(f"  elapsed: {elapsed:.2f}s   sub_actions: {len(sub_results)}")
            tick_actions: list[dict] = []
            for sub_idx, sub in enumerate(sub_results):
                raw = sub.get("raw_response") or ""
                action = sub.get("action") or {}
                dispatch = sub.get("dispatch") or {}

                tag = classify(action, dispatch)
                outcomes[tag] += 1
                action_kinds[action.get("action", "?")] += 1
                target = action.get("npc_id") or action.get("target") or "-"
                action_targets[target] += 1
                coerced = _was_coerced(raw, action)
                if coerced:
                    coerce_count += 1

                indent = "    " if len(sub_results) > 1 else "  "
                marker = f"[{sub_idx + 1}/{len(sub_results)}] " if len(sub_results) > 1 else ""
                print(f"{indent}{marker}{tag}  coerced={coerced}")
                print(f"{indent}  action: {json.dumps(action, ensure_ascii=False)[:280]}")
                print(f"{indent}  dispatch: {dispatch}")
                warning = dispatch.get("similarity_warning")
                if warning:
                    nli = warning.get("nli") or {}
                    nli_label = nli.get("label", "?")
                    nli_conf = nli.get("confidence", 0.0)
                    wmark = "⚠⚠ CONTRADICTION" if warning.get("contradiction") else "⚠ ledger"
                    print(f"{indent}  {wmark} sim={warning['similarity']:.2f}  "
                          f"nli={nli_label}({nli_conf:.2f})  "
                          f"matches T{warning['matches_tick']} {warning['matches_kind']}/{warning['matches_npc']}")
                    similarity_warnings.append({
                        "tick": tick_num, **warning,
                    })
                tick_actions.append({
                    "focus_npc": sub.get("focus_npc"),
                    "action_kind": sub.get("action_kind"),
                    "outcome": tag,
                    "coerced": coerced,
                    "raw": raw,
                    "action": action,
                    "dispatch": dispatch,
                })

            if world_changes:
                print(f"  world: {'; '.join(world_changes)}")
            else:
                print(f"  world: (no change)")

            trace.append({
                "tick": tick_num,
                "elapsed": round(elapsed, 3),
                "sub_actions": tick_actions,
                "world_changes": world_changes,
                # Legacy fields for the player-reactivity scanner below
                "action": tick_actions[0]["action"],
                "dispatch": tick_actions[0]["dispatch"],
                "outcome": tick_actions[0]["outcome"],
            })

        # Summary
        print("\n" + "=" * 72)
        print("SUMMARY")
        print("=" * 72)
        print(f"Total ticks: {args.ticks}")
        total_time = sum(timings)
        print(f"Total time: {total_time:.2f}s   mean: {total_time / args.ticks:.2f}s/tick")
        print(f"Coercions:   {coerce_count}/{args.ticks}")
        print(f"Outcomes:")
        for tag, count in outcomes.most_common():
            bar = "#" * count
            print(f"  {tag:36} {count:3d} {bar}")
        print(f"Action kinds:")
        for k, c in action_kinds.most_common():
            print(f"  {k:12} {c}")
        print(f"Targets touched:")
        for t, c in action_targets.most_common():
            print(f"  {t:20} {c}")

        # ── Scaling report — cast coverage, snapshot growth, RSS ─────
        # Only meaningful on big-cast worlds, but cheap enough to print
        # for every run.
        n_profiles = len(engine.pie.npc_knowledge.profiles)
        touched_ids = set(action_targets.keys()) - {None, "", "-"}
        touched_count = len(touched_ids)
        untouched_count = max(0, n_profiles - touched_count)
        coverage_pct = 100.0 * touched_count / n_profiles if n_profiles else 0.0
        if action_targets:
            touch_counts = sorted(action_targets.values(), reverse=True)
            max_touch = touch_counts[0]
            median_touch = touch_counts[len(touch_counts) // 2]
        else:
            max_touch = 0
            median_touch = 0
        if snapshot_lengths:
            snap_min = min(snapshot_lengths)
            snap_max = max(snapshot_lengths)
            snap_first = snapshot_lengths[0]
            snap_last = snapshot_lengths[-1]
        else:
            snap_min = snap_max = snap_first = snap_last = 0
        if rss_samples_mb:
            rss_peak = max(rss_samples_mb)
            rss_growth = rss_peak - rss_baseline_mb
        else:
            rss_peak = rss_baseline_mb
            rss_growth = 0.0

        print("\nScaling report")
        print(f"  cast size:          {n_profiles}")
        print(f"  NPCs touched:       {touched_count} / {n_profiles} "
              f"({coverage_pct:.1f}%)")
        print(f"  NPCs untouched:     {untouched_count}")
        print(f"  max touches/NPC:    {max_touch}")
        print(f"  median touches/NPC: {median_touch}")
        print(f"  snapshot chars:     first={snap_first}  last={snap_last}  "
              f"min={snap_min}  max={snap_max}")
        print(f"  snapshot tokens (~chars/4): first={snap_first//4}  "
              f"last={snap_last//4}  max={snap_max//4}")
        print(f"  RSS baseline:       {rss_baseline_mb:.0f} MB")
        print(f"  RSS peak:           {rss_peak:.0f} MB  (growth {rss_growth:+.0f} MB)")
        if args.tick_budget_seconds > 0:
            wall_total = sum(max(t, args.tick_budget_seconds) for t in timings)
            print(f"  tick budget:        {args.tick_budget_seconds:.1f}s/tick "
                  f"(simulated wall: {wall_total:.0f}s)")

        # Drift signal: how often did consecutive sub-actions (across the
        # whole flat sequence, including within and across ticks) target
        # the same NPC? This is a stronger metric for multi-action ticks
        # than per-tick repeats.
        flat_targets: list[Optional[str]] = []
        for t in trace:
            for sub in t.get("sub_actions", []):
                a = sub.get("action") or {}
                flat_targets.append(a.get("npc_id") or a.get("target"))
        consecutive_repeats = sum(
            1 for i in range(1, len(flat_targets))
            if flat_targets[i] and flat_targets[i] == flat_targets[i - 1]
        )
        print(f"Consecutive target repeats (sub-action level): "
              f"{consecutive_repeats}/{max(len(flat_targets) - 1, 0)}")

        if similarity_warnings:
            contradictions = [w for w in similarity_warnings if w.get("contradiction")]
            print(f"FactLedger warnings: {len(similarity_warnings)}/{args.ticks}  "
                  f"(contradictions: {len(contradictions)})")
            for w in similarity_warnings:
                nli = w.get("nli") or {}
                marker = "CONTRA" if w.get("contradiction") else "sim   "
                print(f"  T{w['tick']} {marker} sim={w['similarity']:.2f} "
                      f"nli={nli.get('label', '?')}({nli.get('confidence', 0.0):.2f}) "
                      f"-> T{w['matches_tick']} {w['matches_kind']}/{w['matches_npc']}: "
                      f"{w['matches_text'][:70]}")
        try:
            ledger_stats = engine.story_director.ledger.stats()
            print(f"Ledger entries: {ledger_stats['entry_count']}  "
                  f"threshold: {ledger_stats['threshold']}")
        except Exception:
            pass

        # Arc planner summary
        try:
            planner = engine.story_director.arc_planner
            active_count = len(planner.active_arcs())
            print(f"Arcs proposed: {len(planner.arcs)}  "
                  f"active at end: {active_count}")
            for a in planner.arcs:
                beat_label = a.current_beat_label
                total = len(a.beat_goals)
                if a.is_complete:
                    beat_str = f"{total}/{total} (done)"
                else:
                    beat_str = f"{a.current_beat + 1}/{total} ({beat_label})"
                print(f"  {a.id}  status={a.status}  beat={beat_str}")
                print(f"    theme: {a.theme[:120]}")
                print(f"    cast:  {a.focus_npcs}")
            if arc_events:
                print(f"Arc events ({len(arc_events)}):")
                for ev in arc_events:
                    if ev["event"] == "proposed":
                        print(f"  T{ev['tick']} PROPOSED {ev['arc_id']} "
                              f"cast={ev['focus_npcs']} theme={ev['theme'][:60]}")
                    elif ev["event"] == "beat_advanced":
                        print(f"  T{ev['tick']} BEAT {ev['from_beat'] + 1} -> "
                              f"{ev['to_beat'] + 1}  arc={ev['arc_id']}")
                    elif ev["event"] == "resolved":
                        print(f"  T{ev['tick']} RESOLVED {ev['arc_id']} "
                              f"status={ev['status']}")
        except Exception as e:
            print(f"  (arc planner reporting failed: {e})")

        if dialogue_events_log:
            print(f"Dialogue turns: {len(dialogue_events_log)}  "
                  f"total dialogue gen time: {dialogue_total_time:.2f}s")
            for de in dialogue_events_log:
                before_tick = de["before_tick"]
                target = de["npc_id"]
                next_tick = trace[before_tick - 1] if before_tick - 1 < len(trace) else None
                if not next_tick:
                    print(f"  T{before_tick} [{target}] -- no following tick --")
                    continue
                action = next_tick.get("action", {})
                action_target = action.get("npc_id") or action.get("target") or ""
                action_text = (action.get("event") or action.get("fact") or
                               (action.get("quest") or {}).get("description") or "")
                reacted = (target == action_target) or (target in str(action_text).lower())
                marker = "REACT" if reacted else "miss "
                print(f"  T{before_tick} [{target}] {marker}: {str(action_text)[:65]}")

        # Player-reactivity signal: for each scripted player event, did the
        # *next* director tick mention that target or reference the action?
        if player_events_log:
            print(f"Player events: {len(player_events_log)}")
            for pe in player_events_log:
                before_tick = pe["before_tick"]
                target = pe["target"]
                # Locate the tick that fired right after the player event
                next_tick_trace = trace[before_tick - 1] if before_tick - 1 < len(trace) else None
                if next_tick_trace is None:
                    print(f"  T{before_tick} [{target}]: -- no matching tick --")
                    continue
                action = next_tick_trace.get("action", {})
                reacted = False
                action_target = action.get("npc_id") or action.get("target") or ""
                if target and target == action_target:
                    reacted = True
                action_text = (action.get("event") or action.get("fact") or
                               (action.get("quest") or {}).get("description") or "")
                if target and target in str(action_text).lower():
                    reacted = True
                reason = action.get("reason", "")
                if target and target in str(reason).lower():
                    reacted = True
                marker = "REACT" if reacted else "miss "
                print(f"  T{before_tick} [{target}] {marker}: "
                      f"{str(action_text)[:70]}")

        if args.log:
            log_path = Path(args.log)
            if not log_path.is_absolute():
                log_path = NPC_ROOT / log_path
            log_path.write_text(
                json.dumps(
                    {
                        "player_events": player_events_log,
                        "dialogue_events": dialogue_events_log,
                        "ticks": trace,
                    },
                    indent=2, ensure_ascii=False,
                ),
                encoding="utf-8",
            )
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
