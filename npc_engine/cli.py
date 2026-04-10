#!/usr/bin/env python3
"""
NPC Engine — Interactive CLI.

Usage:
    python -m npc_engine.cli                    # Default config
    python -m npc_engine.cli --config my.yaml   # Custom config
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from npc_engine.engine import NPCEngine


def main():
    parser = argparse.ArgumentParser(description="NPC Engine — Interactive CLI")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n  NPC Engine")
    print("  Built on Plug-in Intelligence Engine")
    print("  Type /help for commands, /quit to exit.\n")

    engine = NPCEngine(args.config)
    engine.initialize()

    active = engine.active_npc
    if active:
        npc = engine.pie.npc_knowledge.get(active)
        name = npc.identity.get("name", active) if npc else active
        print(f"  Active NPC: {name}")

    print()

    while True:
        try:
            npc_label = engine.active_npc or "none"
            user_input = input(f"[{npc_label}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down...")
            engine.shutdown()
            break

        if not user_input:
            continue

        # Handle commands
        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            print("Shutting down...")
            engine.shutdown()
            break

        if cmd == "/help":
            print(
                "Commands:\n"
                "  /npc             — List all NPCs\n"
                "  /npc <name>      — Switch to an NPC\n"
                "  /caps            — Show active NPC's capability states\n"
                "  /gossip          — Show social graph and pending gossip\n"
                "  /gossip <name>   — Show what an NPC has heard\n"
                "  /event <text>    — Inject a world event to all NPCs\n"
                "  /event <npc> <text> — Inject event to specific NPC\n"
                "  /graph           — Show social connections\n"
                "  /help            — Show this help\n"
                "  /quit            — Exit"
            )
            continue

        if cmd == "/npc" or cmd == "/npcs":
            npcs = engine.list_npcs()
            active = engine.active_npc
            print(f"NPCs ({len(npcs)} loaded, active: {active or 'none'}):")
            for npc in npcs:
                marker = " <" if npc["id"] == active else ""
                caps = ", ".join(npc["capabilities"][:3])
                print(f"  {npc['id']}: {npc['name']}, {npc['role']} [{caps}]{marker}")
            continue

        if cmd.startswith("/npc "):
            npc_id = user_input[5:].strip().lower()
            try:
                info = engine.switch_npc(npc_id)
                caps = ", ".join(info["capabilities"])
                print(f"Switched to {info['name']} ({info['role']}). [{caps}]")
            except ValueError as e:
                print(str(e))
            continue

        if cmd in ("/caps", "/capabilities"):
            if not engine.active_npc:
                print("No active NPC. Use /npc <name> first.")
                continue
            state = engine.get_npc_state(engine.active_npc)
            print(f"Capabilities for {state['npc_id']} (turn {state['turn_count']}):")
            for cap_name, cap_state in state["capabilities"].items():
                summary = ", ".join(f"{k}={v}" for k, v in list(cap_state.items())[:3])
                print(f"  {cap_name}: {summary}")
            continue

        if cmd == "/gossip" or cmd == "/graph":
            graph = engine.get_social_graph()
            if not graph["connections"]:
                print("No social graph configured.")
                continue
            print(f"Social Graph ({len(graph['connections'])} connections, "
                  f"{graph['pending_gossip']} pending gossip):")
            for conn in graph["connections"]:
                print(f"  {conn['from']} --[{conn['relationship']}]--> {conn['to']} "
                      f"(closeness: {conn['closeness']}, filter: {conn['gossip_filter']})")
            continue

        if cmd.startswith("/gossip "):
            npc_id = user_input[8:].strip().lower()
            state = engine.get_npc_state(npc_id)
            gossip_state = state["capabilities"].get("gossip", {})
            rumors = gossip_state.get("rumors", [])
            if not rumors:
                print(f"{npc_id} hasn't heard any gossip yet.")
            else:
                print(f"{npc_id} has heard {len(rumors)} rumors:")
                for r in rumors:
                    print(f"  - [{r['source_npc']}] {r['text']}")
            continue

        if cmd.startswith("/event "):
            parts = user_input[7:].strip()
            # Check if first word is an NPC ID
            words = parts.split(" ", 1)
            if len(words) >= 2 and words[0].lower() in [n["id"] for n in engine.list_npcs()]:
                engine.inject_event(words[1], npc_id=words[0].lower())
                print(f"Event injected for {words[0]}: {words[1]}")
            else:
                engine.inject_event(parts)
                print(f"Event injected for all NPCs: {parts}")
            continue

        if cmd.startswith("/"):
            # Try passing to PIE's command handler
            result = engine.pie._handle_commands(user_input)
            if result:
                print(result)
            else:
                print(f"Unknown command: {cmd.split()[0]}")
            continue

        # Regular dialogue — process through NPC engine
        response = engine.process(user_input)

        # Pretty-print JSON response
        try:
            obj = json.loads(response.strip())
            if isinstance(obj, dict) and "dialogue" in obj:
                name = engine.active_npc
                npc = engine.pie.npc_knowledge.get(name)
                display_name = npc.identity.get("name", name) if npc else name

                emotion = obj.get("emotion", "")
                action = obj.get("action", "")
                quest = obj.get("quest")

                print(f"\n  {display_name} [{emotion}]: {obj['dialogue']}")
                if action:
                    print(f"  * {action} *")
                if quest:
                    print(f"  [QUEST: {quest.get('type', '?')} — {quest.get('objective', '')}]")
                print()
            else:
                print(response)
        except (json.JSONDecodeError, ValueError):
            print(response)


if __name__ == "__main__":
    main()
