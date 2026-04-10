# NPC Engine - Godot SDK

GDScript HTTP client for the NPC Engine REST API. Provides AI-powered NPC dialogue with trust, gossip, emotions, and quests for Godot 4.2+.

## Installation

1. Copy the `addons/npc_engine/` folder into your Godot project's `addons/` directory.
2. Open **Project > Project Settings > Plugins** and enable **NPC Engine**.

This registers a global autoload singleton named `NPCEngine` (an `NPCEngineClient` instance).

## Configuration

The default server URL is `http://127.0.0.1:8000`. Change it at runtime:

```gdscript
NPCEngine.server_url = "http://192.168.1.10:8000"
```

## Usage with await

The async methods return typed results directly. Use `await` in any function:

```gdscript
func talk_to_blacksmith() -> void:
    var result := await NPCEngine.generate_async("I need a new sword.", "blacksmith")
    if result == null:
        print("Request failed")
        return

    print(result.parsed.dialogue)   # "I can forge one for 50 gold."
    print(result.parsed.emotion)    # "helpful"
    print(result.parsed.action)     # "*reaches for hammer*"

    if result.parsed.quest:
        print("Quest: ", result.parsed.quest.objective)
```

### Listing NPCs

```gdscript
func show_npcs() -> void:
    var list := await NPCEngine.list_npcs_async()
    for npc in list.npcs:
        print("%s (%s) - %s" % [npc.name, npc.id, npc.role])
```

## Usage with signals

Connect to signals for a fire-and-forget pattern:

```gdscript
func _ready() -> void:
    NPCEngine.npc_response_received.connect(_on_dialogue)
    NPCEngine.trust_adjusted.connect(_on_trust)
    NPCEngine.request_failed.connect(_on_error)

func _on_dialogue(result: NPCModels.GenerateResult) -> void:
    $DialogueLabel.text = result.parsed.dialogue

func _on_trust(result: NPCModels.TrustResult) -> void:
    print("Trust with %s: %d -> %d" % [result.npc_id, result.old_level, result.new_level])

func _on_error(endpoint: String, error: String) -> void:
    push_warning("API error on %s: %s" % [endpoint, error])
```

Then call the fire-and-forget methods:

```gdscript
NPCEngine.generate("Hello!", "blacksmith")
NPCEngine.adjust_trust("blacksmith", 5, "helped with quest")
NPCEngine.set_mood("blacksmith", "happy", 0.8)
NPCEngine.inject_event("A dragon has been spotted nearby!")
```

## Server Manager (optional)

If you want the plugin to start/stop the NPC Engine server binary automatically, add an `NPCEngineServerManager` node to your scene:

```gdscript
var server := NPCEngineServerManager.new()
server.server_binary_path = "res://bin/npc_engine_server"
server.port = 8000
add_child(server)
await server.server_ready
print("Server is up!")
```

## API Reference

### Fire-and-forget methods

| Method | Signal |
|--------|--------|
| `generate(prompt, npc_id)` | `npc_response_received` |
| `list_npcs()` | `npc_list_received` |
| `switch_npc(npc_id)` | `npc_switched` |
| `inject_event(description, npc_id)` | `event_injected` |
| `adjust_trust(npc_id, delta, reason)` | `trust_adjusted` |
| `set_mood(npc_id, mood, intensity)` | `mood_set` |
| `add_scratchpad(npc_id, text, importance)` | -- |
| `accept_quest(quest_id, quest_name, given_by)` | -- |
| `complete_quest(quest_id)` | -- |
| `check_health()` | `server_connected` |

### Async methods

| Method | Returns |
|--------|---------|
| `generate_async(prompt, npc_id)` | `NPCModels.GenerateResult` |
| `list_npcs_async()` | `NPCModels.NPCListResult` |
| `check_health_async()` | `NPCModels.HealthResult` |
