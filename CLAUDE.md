# NPC Engine

Game NPC dialogue engine that wraps PIE via composition. Adds capabilities (trust, gossip, emotions, goals), social networks, and game engine SDKs (Unity/Godot/Unreal).

**Business model:** PIE is private + Cython-compiled (moat). NPC Engine is MIT (growth). SDKs are free on asset stores — every install needs the compiled server.

## Quick Reference

| Layer | Stack | Entry Point |
|-------|-------|-------------|
| Engine | Python 3.11+, wraps PIE via composition | `npc_engine/engine.py` |
| CLI | Interactive terminal | `python -m npc_engine.cli` |
| API | FastAPI, Uvicorn | `python -m npc_engine.server` |
| SDKs | C# (Unity), GDScript (Godot), C++ (Unreal) | `sdks/` |
| Deploy | Docker + docker-compose | `dist/` |

## Project Structure

```
npc-engine/
├── npc_engine/                  # Main Python package
│   ├── engine.py                # NPCEngine orchestrator (wraps PIE, manages capabilities)
│   ├── bridge.py                # PIE/densanon-core sibling directory imports
│   ├── config.py                # NPC config + YAML loading
│   ├── cli.py                   # Interactive CLI
│   ├── server.py                # FastAPI REST server
│   ├── knowledge.py             # NPC knowledge sheets, quests, events
│   ├── capabilities/            # 6 modular NPC capabilities
│   │   ├── base.py              # Capability ABC + Registry decorator
│   │   ├── scratchpad.py        # Remember facts about player (~60 tokens)
│   │   ├── trust.py             # Relationship tracking (~30 tokens)
│   │   ├── emotional_state.py   # Persistent mood (~25 tokens)
│   │   ├── goals.py             # Personal motivations (~40 tokens)
│   │   ├── knowledge_gate.py    # Gated secrets by trust/quest (~40 tokens)
│   │   └── gossip.py            # Rumors via social graph (~40 tokens)
│   ├── experts/                 # PIE expert system customization
│   │   ├── examples.py          # 3-layer few-shot loader (structural/world/per-NPC)
│   │   ├── npc_experts.py       # NPC expert builder + registration
│   │   └── verifiers.py         # JSON output validation
│   └── social/                  # Cross-NPC systems
│       ├── network.py           # Social graph (connections, reachability)
│       ├── propagation.py       # Gossip spreading with hop decay
│       └── reputation.py        # Trust ripple effects
├── data/worlds/                 # Game world definitions
│   ├── ashenvale/               # Demo world (7 NPCs, medieval)
│   │   ├── world.yaml           # Social graph
│   │   ├── npc_profiles/        # One YAML per NPC
│   │   └── examples/            # Dialogue examples (shared + per-NPC)
│   └── port_blackwater/         # Demo world (3 NPCs, pirate port)
├── sdks/                        # Game engine integrations
│   ├── unity/                   # C# async/await, NPCEngineClient/Server (9 files)
│   ├── godot/                   # GDScript signals+await, autoload singleton (6 files)
│   └── unreal/                  # C++ BlueprintCallable, FHttpModule (10 files)
├── templates/                   # YAML templates for game devs
│   ├── npc_template.yaml
│   ├── world_template.yaml
│   ├── config_template.yaml
│   ├── examples_template.yaml
│   └── player_quests_template.yaml
├── modules/                     # Legacy NPC modules (npc_kael, npc_noah)
├── tests/
│   ├── test_all.py              # Full integration tests (8 test groups)
│   └── test_port_blackwater.py  # Port Blackwater scenario tests
├── dist/
│   ├── Dockerfile               # Python 3.11 + llama-cpp-python
│   └── docker-compose.yml       # Mounts PIE models + world data
├── config.yaml                  # Default config (points to ashenvale)
├── requirements.txt             # PyYAML, FastAPI, uvicorn
└── README.md                    # User guide (13.5KB)
```

## Commands

```bash
# CLI
python -m npc_engine.cli                                    # Interactive (default: ashenvale)
python -m npc_engine.cli --config examples/port_blackwater_config.yaml

# REST API
python -m npc_engine.server                                 # Port 8000
python -m npc_engine.server --port 9000
# Docs at http://localhost:8000/docs

# Tests
python tests/test_all.py                                    # Full suite (8 test groups)
python tests/test_port_blackwater.py                        # Port Blackwater tests

# Docker
docker-compose -f dist/docker-compose.yml up                # Requires PIE as sibling dir
```

## Architecture

NPCEngine wraps PIE via **composition, NOT modification**. PIE is a black box.

```
Player Input
  → NPCEngine.process(user_input, npc_id)
    → PIE.process()  [unchanged black box]
      - Routing, expert selection, capabilities context injection, generation
    → Post-generation:
      → GossipPropagator.propagate()  — extract facts, walk social graph, inject to NPCs
      → ReputationRipple.process()    — propagate trust changes through connections
  → Return Response
```

### How it extends PIE (without modifying it)

1. **Capabilities Registry** — decorator pattern, auto-registers on import. PIE's CapabilityRegistry populated at init.
2. **Custom Few-Shot Examples** — FewShotLoader merges 3 layers (structural + world + per-NPC) into PIE's expert system.
3. **Process Wrapping** — calls `pie.process()` unchanged, then runs gossip + trust ripple post-generation.
4. **Configuration Override** — sets PIE's NPC config (profiles_dir, state_dir, world_name, active_profile).

## Capabilities (6 Modular, Opt-In)

| Capability | Priority | Budget | Purpose |
|------------|----------|--------|---------|
| trust | 90 | 30 tok | Relationship 0-100, thresholds (wary/neutral/friendly/trusted), behavioral effects |
| emotional_state | 80 | 25 tok | Persistent mood, baseline + volatility + decay |
| goals | 70 | 40 tok | Personal motivations, priority 1-10, keyword detection |
| scratchpad | 60 | 60 tok | Remember facts about player, max 10 entries |
| knowledge_gate | 50 | 40 tok | Gated secrets unlocked by trust/quest conditions |
| gossip | 45 | 40 tok | Rumors from social network, filtered by interests |

All capabilities are **programmatic** (rule-based, not model-generated). They inject context pre-generation and update state post-generation via heuristics.

## Social Systems

### Gossip Propagation
- Extract facts from NPC responses → classify (personal/trade/military/lore/quest)
- Walk social graph from source NPC with configurable `max_hops` (default 2)
- Decay significance per hop (default 0.5), filter by `gossip_filter` per connection
- Inject as events to target NPCs

### Trust Ripple
- Trust changes propagate to connected NPCs by `closeness` factor
- Positive factor: 30%, negative factor: 15%, max ripple: 10 points

## Configuration

**config.yaml** — root level:
```yaml
world_dir: "data/worlds/ashenvale"
world_name: "Ashenvale"
active_npc: "noah"
gossip_rules:
  max_hops: 2
  decay_per_hop: 0.5
  min_significance: 0.2
trust_ripple:
  enabled: true
  positive_factor: 0.3
  negative_factor: 0.15
  max_ripple: 10
```

**world.yaml** — per-world social graph with connections (from, to, relationship, closeness, gossip_filter).

**NPC profiles** — YAML with identity, personality, world_facts, personal_knowledge, active_quests, capabilities config, per-NPC dialogue examples.

## Key Files by Task

### Adding a new NPC
1. Create `data/worlds/{world}/npc_profiles/{npc_id}.yaml` (use `templates/npc_template.yaml`)
2. Add connections in `world.yaml`
3. Optionally add per-NPC examples in `examples/`

### Adding a new world
1. Create `data/worlds/{world_name}/` with `world.yaml`, `npc_profiles/`, `examples/`
2. Create a config YAML pointing to the new world
3. Run: `python -m npc_engine.cli --config your_config.yaml`

### Adding a new capability
1. Create `npc_engine/capabilities/{name}.py` implementing `Capability` ABC
2. Use `@CapabilityRegistry.register` decorator
3. Add config section to NPC profiles under `capabilities.{name}`

### Modifying gossip/trust
- Gossip: `npc_engine/social/propagation.py` + config `gossip_rules`
- Trust ripple: `npc_engine/social/reputation.py` + config `trust_ripple`
- Social graph: `npc_engine/social/network.py`

### SDK development
- Unity: `sdks/unity/` — C# async/await pattern, `NPCEngineClient.cs`
- Godot: `sdks/godot/` — GDScript signals, autoload singleton
- Unreal: `sdks/unreal/` — C++ BlueprintCallable, FHttpModule

## CLI Commands

| Command | Purpose |
|---------|---------|
| `/npc {id}` | Switch active NPC |
| `/caps` | Show active NPC capabilities |
| `/gossip` | Show current gossip |
| `/event {text}` | Inject world event |
| `/graph` | Show social connections |
| `/help` | List commands |
| `/quit` | Exit |

## License System

- Key format: `NPCE-{TIER}-{CUSTOMER_ID}-{EXPIRY}-{SIGNATURE}`
- Tiers: `IND` (indie — 5 NPCs, 3 caps/NPC, log watermark), `COM` (commercial — unlimited)
- Development mode: unlimited (no enforcement without explicit `initialize_license()`)
- Keygen: `tools/keygen.py` (private, never distributed)
- Validation: HMAC-SHA256 in PIE's compiled `engine/license.py`

## densanon-core Dependency

NPC Engine depends on PIE, which imports from `densanon.core.*`. The NPC-specific modules (capabilities, npc_knowledge) are in [densanon-core](https://github.com/densanon-devs/densanon-core) and consumed via PIE's bridge.

## GitHub

- Org: densanon-devs
- Repo: densanon-devs/npc-engine (MIT)
