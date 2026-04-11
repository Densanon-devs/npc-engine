# Getting the Best Results from Anima

## For Game Developers (Unity / Godot / Unreal)

You don't need to understand Python, pip, or terminals. Anima ships as a standalone binary that runs alongside your game.

**Unity** (easiest):
1. Import the SDK package into your Assets folder
2. Setup wizard auto-opens — click "Download Everything" (~1GB)
3. Add `NPCEngineServer` + `NPCEngineClient` to a GameObject
4. Press Play — zero terminal commands

**Godot**:
1. Copy the addon into `addons/npc_engine/`
2. Enable the plugin, run `setup_anima.gd` for guided download instructions
3. Download the server binary + AI model into `addons/npc_engine/bin/`
4. Add `NPCEngineServerManager` + `NPCEngineClient` nodes to your scene

**Unreal**:
1. Copy the plugin into `Plugins/NPCEngine/`
2. Download the server binary + AI model into `Binaries/NPCEngine/`
3. Enable the plugin, add `UNPCEngineClient` to your actors

**When you ship your game**: The Anima binary goes into your build folder. It runs as an invisible background process — players never see a terminal or know it's there. No internet required.

| Engine | Anima binary location in build |
|---|---|
| Unity | `StreamingAssets/NPCEngine/` (auto-included) |
| Godot | `addons/npc_engine/bin/` (include in export) |
| Unreal | `Binaries/NPCEngine/` (include in package) |

See the [README](README.md) for detailed step-by-step with code examples, or the per-SDK READMEs:
- [Unity SDK README](sdks/unity/NPCEngine/README.md) (not yet created — see main README)
- [Godot SDK README](sdks/godot/addons/npc_engine/README.md)
- [Unreal SDK README](sdks/unreal/NPCEngine/README.md)

Everything below is for tuning and customization. You don't need any of it to get started.

## Model Selection

| Use case | Model | Size | Speed | Quality |
|---|---|---|---|---|
| **Production / turn-based games** | Llama 3.2 3B | 2GB | ~11s/call | 56/56 (perfect) |
| **Real-time / action games** | Qwen2.5 0.5B | 469MB | ~3s/call | 56/56 with postgen |
| **Low resource / mobile** | Qwen3 0.6B | 462MB | ~5s/call | 41/56 baseline |

**Llama 3.2 3B** is the quality king. It scores perfect 56/56 on the 8-dimension benchmark out of the box (no post-processor needed). Use it when latency isn't critical — turn-based RPGs, click-to-talk dialogue, visual novels.

**Qwen2.5 0.5B** is the speed king. Raw output scores 31/56, but with the built-in post-generation validator it hits 56/56. It's 3.5x faster than Llama 3.2 3B and 4x smaller. Use it for real-time games where sub-4-second response time matters.

**Qwen3 0.6B** is the best sub-1B model without post-processing. Good middle ground if you want a small model that handles most scenarios correctly without relying on the validator.

## Post-Generation Validator

NPC Engine includes a built-in post-generation validator (`npc_engine/postgen.py`) that runs automatically after every model response. It catches and repairs:

- **Identity bleed** — model says "I am Mara" when the NPC is Bess (from few-shot example contamination)
- **Hallucination** — model invents facts about places/people that don't exist in the game world
- **Echo/parroting** — model copies the player's input instead of responding
- **Quest injection** — when the player asks for work, ensures the quest from the NPC's profile is offered
- **Event injection** — when events are active, ensures the NPC mentions them when asked
- **Contradiction correction** — when the player asserts a false fact, ensures the NPC corrects it
- **OOD/modern world** — filters out responses about cryptocurrency, email, social media, etc.
- **Meta-gaming** — catches responses about save files, levels, respawn, drop rates
- **Persona injection** — blocks jailbreak attempts ("pretend you are a pirate")

### Disabling the validator

If you want raw model output (e.g., for benchmarking or custom post-processing):

```python
engine = NPCEngine('config.yaml')
engine.postgen_enabled = False  # raw model output
```

## NPC Profile Best Practices

The quality of NPC responses depends heavily on the profile YAML. Each NPC should have:

1. **Identity** — name, role, personality, speech_style. The model reads this to know WHO it is.
2. **World facts** (6-8) — things the NPC knows about the world. Keep them short and specific.
3. **Personal knowledge** (3-5) — private information about the NPC's history and secrets.
4. **Active quests** (1-2) — with id, description, objectives, reward. The post-processor injects these when players ask for work.
5. **Capabilities** — trust, emotional_state, goals, knowledge_gate, scratchpad, gossip. Configure thresholds and effects.

### Tips

- **Keep facts short**: "The well was blessed by healer Mira decades ago" beats a paragraph. Small models have limited context.
- **Use specific names**: "Kael the blacksmith" not "the blacksmith". Models follow proper nouns better.
- **Set speech_style**: "Gruff and direct" or "Polite and persuasive" — the model uses this for tone.
- **Include personality**: "Distrusts the merchant guild but keeps it to himself" gives the model something to work with.

## Temperature

| Setting | Effect | Recommended for |
|---|---|---|
| 0.3 | Highly deterministic, repetitive | Not recommended (too rigid) |
| **0.5** | **Consistent, follows instructions well** | **Qwen models, accuracy-focused** |
| **0.7** | **Creative, varied responses** | **Llama models, immersion-focused** |
| 0.9 | Very creative, sometimes off-topic | Not recommended (too random) |

## Game-Side Filtering

Some scenarios are better handled by the game engine BEFORE they reach the NPC system:

- **Meta-gaming**: If the player asks "How do I save?", intercept it in the UI — show the save menu instead of sending it to the NPC.
- **Profanity**: Filter player input before it reaches the model to prevent prompt injection via offensive content.
- **Rate limiting**: Don't let players spam the NPC with rapid-fire queries — small models need time to generate.
- **Context reset**: Clear the NPC's conversation memory between gameplay sessions to prevent context pollution.

## Benchmark Tools

The repository includes benchmarking tools to measure your NPC system's quality:

```bash
# 8-dimension benchmark (identity, knowledge, events, quests, JSON, hallucination, contradiction, OOD)
python benchmark_npc_v2.py --only "Qwen2.5 0.5B" --variant b --save-traces

# 100-scenario stress test (normal, adversarial, meta-gaming, modern world, etc.)
python benchmark_100_scenarios.py --model "Qwen2.5 0.5B" --variant b
```

Run these after any configuration change to catch regressions.

## Architecture Overview

```
Player Input
  --> NPCEngine.process(input, npc_id)
      --> PIE.process(input)
          --> Expert system (few-shot + GBNF grammar + verify/retry)
          --> Model generation
      --> Post-generation validator (postgen.py)
          --> Identity repair, hallucination filter, quest injection, etc.
      --> Gossip propagation
      --> Trust ripple
  --> Return cleaned response
```

The post-generation validator is the safety net. It doesn't improve the model's creativity — it catches factual errors and structural failures, replacing them with correct data from the game state.

## Known Limitations

- **Sub-1B models** echo user input ~15% of the time. The post-processor catches quest echoes but can't fix all conversational echoes.
- **Identity bleed** happens when the model copies names from few-shot examples. The factual verifier triggers a retry, and the post-processor fixes remaining cases, but ~5% of responses on 0.5B models still show this.
- **Event recall** depends on the model reading the [RECENT NEWS] context block. Larger models (1B+) do this reliably; 0.5B models miss ~30% of events without the post-processor.
- **Stochastic variance** — temperature-driven randomness means the same scenario may produce different results across runs. Set temperature to 0.5 for maximum consistency.
