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

## Quick Start

### 1. Install

```bash
# Clone both repos as siblings
git clone https://github.com/densanon-devs/plug-in-intelligence-engine
git clone https://github.com/densanon-devs/npc-engine

# Install dependencies
cd npc-engine
pip install -r requirements.txt
cd ../plug-in-intelligence-engine
pip install -r requirements.txt
```

### 2. Download a model

```bash
cd npc-engine
python download_model.py              # Qwen2.5 0.5B (469MB, fastest)
python download_model.py --quality    # Llama 3.2 3B (2GB, best quality)
python download_model.py --both       # Both
```

### 3. Run

**CLI (interactive)**:
```bash
python -m npc_engine.cli

[noah] > Hello, who are you?
  Noah [warm]: I am Noah, elder of Ashenvale. Welcome, traveler.
```

**REST API (for game engine integration)**:
```bash
python -m npc_engine.server
# API docs at http://localhost:8000/docs
```

**From your game engine** (Unity, Godot, Unreal):
```
POST http://localhost:8000/generate
{"prompt": "Hello!", "npc_id": "noah"}
```

Response includes dialogue, emotion (for animation), capability state (trust, mood), and quest data.

## What NPC Engine Does

- **Post-generation validator** — Built-in 11-layer safety net catches hallucination, identity bleed, meta-gaming, modern-world leakage, and more. A 469MB model matches a 2GB model's quality through smart post-processing.
- **Cross-NPC gossip network** — Tell Pip a secret, Bess hears about it. Help Noah, Kael respects you more.
- **6 modular capabilities** — Scratchpad (remembers player facts), trust (relationship tracking), emotional state (mood system), goals (personal motivations), knowledge gates (secrets unlocked by trust/quests), gossip (rumors from social network). All opt-in per NPC.
- **Social graph** — Define who knows whom, relationship types, and what gossip passes between them.
- **Trust ripple** — Reputation propagates through the social network.
- **Quest system** — NPCs offer quests from their profiles, track player progress, boost trust on completion.
- **State persistence** — Trust, mood, scratchpad survive between play sessions.
- **9 emotion labels** — calm, compassionate, confused, excited, friendly, laughing, neutral, serious, warm. Stable enough for animation mapping.

## Game Engine SDKs

| Engine | Location | Integration |
|---|---|---|
| **Unity** | `sdks/unity/` | C# async/await, `NPCEngineClient.cs` |
| **Godot** | `sdks/godot/` | GDScript signals, autoload singleton |
| **Unreal** | `sdks/unreal/` | C++ BlueprintCallable, FHttpModule |

All SDKs connect to the local REST API server.

## CLI Commands

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
