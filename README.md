# NPC Engine

Modular NPC intelligence system built on [PIE](https://github.com/densanon-devs/plug-in-intelligence-engine). One small model + YAML knowledge sheets = unlimited characters with memory, trust, moods, gossip, and gated knowledge.

Runs on 135M-3B parameter models **locally on your machine**. No cloud API, no subscription, no internet required after model download.

## Benchmark Results

| Model | Size | Quality | Speed | Best for |
|---|---|---|---|---|
| **Llama 3.2 3B** | 2GB | 56/56 (raw) | 11s/call | Turn-based games, quality-first |
| **Qwen2.5 0.5B** | 469MB | 56/56 (with postgen) | 3s/call | Real-time games, speed-first |

100-scenario stress test (adversarial, hallucination, meta-gaming, combat, edge cases): Qwen 87/100, Llama 78/100.

System tests: **88/88 (100%)** across multi-turn conversations, quest lifecycle, social gossip, state persistence, combat reactions, and edge cases.

## Quick Start — Unity (no terminal needed)

1. **Import the SDK** — Copy `sdks/unity/NPCEngine/` into your Unity project's `Assets/` folder (or import from the [Releases](https://github.com/Densanon-devs/npc-engine/releases) page)
2. **Setup wizard opens automatically** — Go to `Window > NPC Engine > Setup Wizard` if it doesn't
3. **Click "Download Everything"** — Downloads the server binary + AI model (~1GB total)
4. **Add components to a GameObject** — `NPCEngineServer` (auto-launches the engine) + `NPCEngineClient` (all your API calls)
5. **Press Play** — The server starts in the background, NPCs are ready

```csharp
// Talk to an NPC
var response = await client.GenerateAsync("Hello, who are you?", "noah");
dialogueText.text = response.parsed.dialogue;   // "I am Noah, elder of Ashenvale."
animController.SetEmotion(response.parsed.emotion); // "warm"

// Inject a world event (dragon attack, building destroyed, etc.)
await client.InjectEventAsync("A dragon was spotted over the forbidden forest!");

// Complete a quest (triggers trust boost + gossip propagation)
await client.CompleteQuestAsync("bitter_well");
```

See `Samples~/BasicDialogue.cs` for a complete copy-paste example.

### Shipping your game

When you build your Unity project, `StreamingAssets/NPCEngine/` ships with the game automatically. The `NPCEngineServer` component launches `npc-engine.exe` as a background process — players never see it. No internet needed, no cloud, no subscription.

## Quick Start — Godot

1. Copy `sdks/godot/addons/npc_engine/` into your project's `addons/` folder
2. Download the server binary from [Releases](https://github.com/Densanon-devs/npc-engine/releases) into your project
3. Enable the plugin in Project Settings
4. Add `NPCEngineServerManager` + `NPCEngineClient` nodes to your scene
5. Set the `server_binary_path` to where you placed the binary

## Quick Start — Unreal

1. Copy `sdks/unreal/NPCEngine/` into your project's `Plugins/` folder
2. Download the server binary from [Releases](https://github.com/Densanon-devs/npc-engine/releases) into `Binaries/`
3. Enable the plugin in Edit > Plugins
4. Use `UNPCEngineClient` in C++ or Blueprint nodes for dialogue

## Quick Start — Python developers

If you're building your own integration or running the server directly:

```bash
git clone https://github.com/densanon-devs/plug-in-intelligence-engine
git clone https://github.com/densanon-devs/npc-engine
cd npc-engine && pip install -r requirements.txt
cd ../plug-in-intelligence-engine && pip install -r requirements.txt
cd ../npc-engine

python download_model.py              # Qwen2.5 0.5B (469MB, fastest)
python -m npc_engine.server           # REST API at localhost:8000
```

Interactive CLI for testing:
```bash
python -m npc_engine.cli
[noah] > Hello, who are you?
  Noah [warm]: I am Noah, elder of Ashenvale. Welcome, traveler.
```

## What NPC Engine Does

- **Zero-config for game devs** — Import SDK, click download, press play. No Python, no terminal, no cloud setup.
- **Post-generation validator** — Built-in 11-layer safety net catches hallucination, identity bleed, meta-gaming, modern-world leakage, and more. A 469MB model matches a 2GB model's quality.
- **Cross-NPC gossip network** — Tell Pip a secret, Bess hears about it. Help Noah, Kael respects you more.
- **6 modular capabilities** — Scratchpad (remembers player facts), trust (relationship tracking), emotional state (mood system), goals (personal motivations), knowledge gates (secrets unlocked by trust/quests), gossip (rumors from social network). All opt-in per NPC via YAML.
- **Social graph + trust ripple** — Reputation propagates through connections.
- **Quest system** — NPCs offer quests, track progress, boost trust on completion, gossip about it.
- **State persistence** — Trust, mood, scratchpad survive between play sessions.
- **Emotion labels for animation** — calm, compassionate, confused, excited, friendly, laughing, neutral, serious, warm. Consistent enough for animation state machines.
- **Ships with your game** — Standalone binary runs as a background process. No internet, no subscription, no player-visible setup.

## Game Engine SDKs

| Engine | Location | Setup | Integration |
|---|---|---|---|
| **Unity** | `sdks/unity/` | Import + Setup Wizard (auto-download) | C# async/await, `NPCEngineClient.cs` |
| **Godot** | `sdks/godot/` | Copy addon + download binary | GDScript signals, autoload singleton |
| **Unreal** | `sdks/unreal/` | Copy plugin + download binary | C++ BlueprintCallable, FHttpModule |

All SDKs launch the NPC Engine as a local background process and connect via HTTP. The engine runs on the player's machine — no server hosting required.

## CLI Commands (for testing)

```
/npc              List all NPCs
/npc <name>       Switch to an NPC
/caps             Show capability states (trust, mood, scratchpad)
/gossip           Show social graph
/gossip <name>    Show what an NPC has heard
/event <text>     Inject world event to all NPCs
/graph            Show social connections
```

### Commands

```
/npc              List all NPCs
/npc <name>       Switch to an NPC
/caps             Show capability states (trust, mood, scratchpad)
/gossip           Show social graph
/gossip <name>    Show what an NPC has heard
/event <text>     Inject world event to all NPCs
/graph            Show social connections
```

## Create Your Own World

### 1. Set up world directory

```
data/worlds/your_world/
  world.yaml              # Social graph + gossip rules
  npc_profiles/           # NPC YAML files
  examples/               # Custom few-shot dialogue examples
    shared_examples.yaml
  npc_state/              # Auto-managed capability state
  player_quests.yaml
```

### 2. Define NPCs

Each NPC is a YAML file. See `templates/npc_template.yaml` for the full schema.

```yaml
identity:
  name: "Commander Vex"
  role: "Station Commander"
  personality: "Stern, disciplined, protective of her crew"

world_facts:
  - "Station Nine is the last outpost before the Drift"

capabilities:
  scratchpad: { enabled: true, max_entries: 8 }
  trust:
    enabled: true
    initial_level: 20
    effects:
      below_wary: "Deny access, question motives"
      trusted: "Share classified intel, grant clearance"
  gossip: { enabled: true, max_rumors: 3 }
```

### 3. Define social connections

`world.yaml`:

```yaml
world_name: "Station Nine"
social_graph:
  connections:
    - from: "commander_vex"
      to: "engineer_kira"
      relationship: "duty"
      closeness: 0.7
      gossip_filter: "military"
```

### 4. Provide dialogue examples

`examples/shared_examples.yaml`:

```yaml
world_examples:
  - query: "Who are you?"
    solution: '{"dialogue": "Commander Vex, Station Nine. State your business.", "emotion": "stern", "action": null}'
    category: "greeting"
```

### 5. Update config

```yaml
world_dir: "data/worlds/station_nine"
world_name: "Station Nine"
active_npc: "commander_vex"
```

## Full Walkthrough: Port Blackwater

A complete example world ships with this repo at `data/worlds/port_blackwater/`. Here's how it was built from scratch — follow the same steps for your own world.

### The World

Port Blackwater is a pirate free port. 3 NPCs:

| NPC | Role | Personality | Key Secret |
|-----|------|------------|------------|
| Captain Reva | Harbor Master | Stern, clipped commands | The Shoals aren't natural — something pulls ships down |
| Old Bones | Tavern Owner | Cackling gossip, smuggler | Runs tunnels under the tavern, has maps of every cave |
| Finn | Dock Worker | Eager kid, dreams of sailing | Found a melted lighthouse lens piece with glowing markings |

### Step 1: Create world directory

```
data/worlds/port_blackwater/
  world.yaml                     # Social graph
  npc_profiles/                  # One YAML per NPC
    captain_reva.yaml
    old_bones.yaml
    finn.yaml
  examples/
    shared_examples.yaml         # World-tone dialogue examples
  npc_state/                     # Auto-managed
  player_quests.yaml
```

### Step 2: Define the social graph (`world.yaml`)

```yaml
world_name: "Port Blackwater"
social_graph:
  connections:
    - from: "captain_reva"
      to: "old_bones"
      relationship: "business"
      closeness: 0.6
      gossip_filter: "all"

    - from: "old_bones"
      to: "captain_reva"
      relationship: "business"
      closeness: 0.6
      gossip_filter: "trade"       # Only shares trade info with Reva

    - from: "finn"
      to: "old_bones"
      relationship: "friend"
      closeness: 0.7
      gossip_filter: "all"         # Tells Old Bones everything
```

### Step 3: Create NPCs with capabilities

Each NPC YAML has identity, knowledge, quests, and opt-in capabilities:

```yaml
# captain_reva.yaml
identity:
  name: "Captain Reva"
  role: "Harbor Master"
  personality: "Sharp-tongued, fair but ruthless with rule-breakers."
  speech_style: "Clipped naval commands. Uses 'sailor' and 'stranger'."

world_facts:
  - "Port Blackwater is a free port -- no kingdom claims it"
  - "The old lighthouse has been dark for two months"

personal_knowledge:
  - "Lost her ship to the Shoals five years ago -- sole survivor"
  - "Suspects Old Bones is smuggling through the tavern cellar"

active_quests:
  - id: "lighthouse_mystery"
    name: "The Dark Lighthouse"
    description: "The lighthouse has been dark for two months. Find out why."
    status: "available"
    reward: "200 gold and a permanent berth"
    objectives:
      - "Climb to the lighthouse and investigate"

capabilities:
  scratchpad: { enabled: true, max_entries: 8 }
  trust:
    enabled: true
    initial_level: 15
    effects:
      below_wary: "Deny docking, threaten arrest"
      trusted: "Reveal Shoals secret, share sealed letter"
  emotional_state:
    enabled: true
    baseline_mood: "serious"
    volatility: 0.2
  knowledge_gate:
    enabled: true
    gated_facts:
      - id: "lighthouse_clue"
        fact: "The last ship before the light went dark carried black glass jars"
        requires:
          trust: 40
          quest_active: "lighthouse_mystery"
      - id: "shoals_secret"
        fact: "The Shoals aren't natural -- something pulls ships down"
        requires:
          trust: 70
  gossip: { enabled: true, max_rumors: 3, interests: ["military", "trade"] }
```

### Step 4: Write world-tone dialogue examples

```yaml
# examples/shared_examples.yaml
world_examples:
  - query: "Hello there."
    solution: '{"dialogue": "Another fresh face off the boats. Welcome to Port Blackwater. Watch your purse.", "emotion": "wary", "action": "looks you up and down"}'
    category: "greeting"

  - query: "What is electricity?"
    solution: '{"dialogue": "Elec-what? We use lanterns and honest fire here.", "emotion": "confused", "action": null}'
    category: "deflection"
```

Per-NPC examples go directly in the NPC YAML under `examples:`:

```yaml
# In finn.yaml
examples:
  - query: "Hey kid, who are you?"
    solution: '{"dialogue": "I am Finn, sir! I work the docks. Are you with one of the crews?", "emotion": "eager", "action": "wipes hands on trousers"}'
    category: "greeting"
```

### Step 5: Configure and run

```yaml
# config.yaml
world_dir: "data/worlds/port_blackwater"
world_name: "Port Blackwater"
active_npc: "captain_reva"
```

```bash
python -m npc_engine.cli --config examples/port_blackwater_config.yaml
```

### Step 6: Play through it

```
[captain_reva] > Who are you?

  Captain Reva [stern]: Harbor Master Reva. State your business, sailor.

[captain_reva] > /npc finn

  Switched to Finn (Dock Worker). [scratchpad, trust, emotional_state, goals, knowledge_gate, gossip]

[finn] > Hey kid, what do you know about the lighthouse?

  Finn [nervous]: The lighthouse? It went dark two months ago, sir. Ships keep
  wrecking on the rocks. I... I found something on the beach near there.
  * glances around nervously *

[finn] > /caps

  Capabilities for finn (turn 1):
    trust: level=27, interactions=1, trend=rising
    scratchpad: turn=1, entries=[...]
    emotional_state: mood=nervous, intensity=0.34
    gossip: rumor_count=0

[finn] > /gossip

  Social Graph (5 connections, 0 pending gossip):
    captain_reva --[business]--> old_bones (closeness: 0.6, filter: all)
    finn --[friend]--> old_bones (closeness: 0.7, filter: all)
    ...
```

## Game Engine Integration

### REST API

```bash
python -m npc_engine.server --port 8000
```

Docs at `http://localhost:8000/docs`.

**Dialogue:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | NPC dialogue (optionally specify `npc_id`) |
| `/npc/list` | GET | List all NPCs |
| `/npc/switch` | POST | Switch active NPC |
| `/npc/{id}/state` | GET | Capability state (trust, mood, scratchpad, etc.) |

**World Events:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/events/inject` | POST | Push world events to one or all NPCs |
| `/gossip/graph` | GET | Social network data |
| `/gossip/{id}` | GET | Rumors an NPC heard |

**Game State Mutations:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/quests/accept` | POST | Player accepts a quest |
| `/quests/complete` | POST | Player completes a quest (triggers trust +10, gossip, ripple) |
| `/npc/trust` | POST | Adjust trust directly (gift, betrayal, story event) |
| `/npc/scratchpad` | POST | Inject a memory (game event the NPC should know) |
| `/npc/mood` | POST | Set mood from cutscene/world event |
| `/npc/knowledge` | POST | Add world or personal fact at runtime |
| `/npc/unlock-gate` | POST | Force-unlock a gated secret |
| `/quests` | GET | Player quest state |

### Example: Full Game Loop

```bash
# 1. Player arrives — talk to Reva
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I need to dock my ship.", "npc_id": "captain_reva"}'

# 2. Player accepts the lighthouse quest
curl -X POST http://localhost:8000/quests/accept \
  -d '{"quest_id": "lighthouse_mystery", "quest_name": "The Dark Lighthouse", "given_by": "captain_reva"}'

# 3. Player buys Finn a meal (game event → trust boost)
curl -X POST http://localhost:8000/npc/trust \
  -d '{"npc_id": "finn", "delta": 15, "reason": "bought him a meal"}'

# 4. Finn now trusts player enough to reveal the shard location (gate unlocks at trust 45)
curl http://localhost:8000/npc/finn/state
# → capabilities.knowledge_gate.unlocked: ["shard_location"]

# 5. Player finds the lighthouse sabotaged (game injects event to all NPCs)
curl -X POST http://localhost:8000/events/inject \
  -d '{"description": "The lighthouse was sabotaged -- black glass jars found shattered inside"}'

# 6. Player completes the quest
curl -X POST http://localhost:8000/quests/complete \
  -d '{"quest_id": "lighthouse_mystery"}'
# → Reva trust +10, gossip propagates to Old Bones and Finn, trust ripples

# 7. Check what Old Bones heard through gossip
curl http://localhost:8000/gossip/old_bones
# → rumors: ["The stranger completed a quest for captain_reva"]

# 8. Game cutscene makes Reva angry (player discovered her sealed letter)
curl -X POST http://localhost:8000/npc/mood \
  -d '{"npc_id": "captain_reva", "mood": "angry", "intensity": 0.8}'

# 9. Inject a memory from a game event Reva witnessed
curl -X POST http://localhost:8000/npc/scratchpad \
  -d '{"npc_id": "captain_reva", "text": "Player read the sealed letter from Meridia", "importance": 0.95}'
```

Every mutation persists to disk, propagates through the gossip network when relevant, and immediately affects the NPC's next response.

## Architecture

```
Player Input
    |
    v
NPCEngine.process()
    |
    v
PIE Pipeline (routing, experts, capabilities, generation)
    |
    v
Post-Generation
    ├── Gossip Propagator (spread facts through social graph)
    ├── Reputation Ripple (trust changes influence connected NPCs)
    └── Capability Updates (scratchpad, mood, trust, gates)
```

NPC Engine does NOT modify PIE. It composes around it:
- Registers capabilities into PIE's `CapabilityRegistry` (class-level, auto-registers on import)
- Injects custom examples into PIE's `ExpertRouter.experts` dict
- Wraps `PIE.process()` with gossip propagation

## Capabilities

| Capability | Priority | Budget | Description |
|-----------|----------|--------|-------------|
| trust | 90 | ~30 tokens | Relationship tracking with behavioral tiers |
| emotional_state | 80 | ~25 tokens | Persistent mood from model emotion output |
| goals | 70 | ~40 tokens | Personal motivations as directives |
| scratchpad | 60 | ~60 tokens | Remembers facts about the player |
| knowledge_gate | 50 | ~40 tokens | Facts unlocked by trust/quest conditions |
| gossip | 45 | ~40 tokens | Rumors heard through social network |

All are opt-in per NPC. Higher priority = allocated token budget first.

## Custom Few-Shot Examples

Three layers, highest priority wins:

1. **Structural** (built-in) — 5 world-agnostic format examples
2. **World** (`examples/shared_examples.yaml`) — your world's tone
3. **Per-NPC** (`examples:` in NPC YAML) — character-specific samples

Game devs never touch Python. Everything is YAML.
