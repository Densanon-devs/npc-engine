# Story Director — Empirical Findings

**Date:** 2026-04-13
**Branch:** `story-director`
**Hardware target:** local GGUF on consumer CPU
**Goal:** A SAO/Cardinal-style narrative overseer that watches a fantasy
village (Ashenvale), generates new quests, drops world facts, and reacts
to player behavior — all on small local models, in real-time.

This document captures what we learned building the prototype: the
architectural insight that made it work, the model comparison that
motivated it, the failure modes encountered, and where the system stops
being scaffolding and starts being a system.

---

## The central architectural insight

> **Python plans, the LLM writes.**
> Deterministic scaffolding owns every decision the model can't make
> reliably; the model owns only what it's actually good at.

After working through ~9 iterations against four model sizes, this is
the one finding that matters most. Every model we tested — from Qwen 0.5B
to Llama 3.2 3B — fails at one or more of:

| Decision | Model failure mode |
|---|---|
| Which NPC to focus on next | Fixates on one NPC for 7+ ticks |
| Whether to fire `event` / `quest` / `fact` | Defaults to `event` 100% of the time |
| Whether to react to a player action | Continues its own arc, ignores player |
| Maintaining schema across ticks | Mislabels actions, leaks fields, emits prose |

Every decision in this list moved into Python. The model is left with
exactly one job: given a focus NPC and an action kind, *write the
content of the story beat*. That's a creative-writing task, which is
what these models can actually do.

The result: on Qwen 2.5 3B, **0 coercions, 0 noops, 0 parse errors,
0 consecutive target repeats, balanced action kinds, and 4/4 player
reactivity in a 10-tick session** — at 1.4–3.8s per tick.

---

## Architecture overview

```
                  POST /story/tick
                          │
                          ▼
                ┌─────────────────────┐
                │   StoryDirector     │
                │       .tick()       │
                └──────────┬──────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
 _world_snapshot   _pick_focus_npc      _pick_action_kind
 (NPCs, quests,   (1. pending player    (round-robin
  events, lore,    target wins;          event/quest/fact;
  player block,    2. else least-        skips quest if NPC
  ALREADY DONE)    recently-touched)     has 2+ open quests)
       │                   │                   │
       └─────────┬─────────┴─────────┬─────────┘
                 │                   │
                 ▼                   ▼
            _build_prompt    (lore + few-shot + snapshot
                              + FOCUS NPC + ACTION KIND)
                 │
                 ▼
            base_model.generate    ← ONLY job: creative content
                 │
                 ▼
        _llm_call_with_repair      ← one repair retry on parse fail
                 │
                 ▼
             _parse_action          ← brace-matching extractor
                 │                    + coercion (label → field shape)
                 ▼
        _enforce_focus_npc          ← override if model deviated
                 │
                 ▼
        _enforce_action_kind        ← override if model deviated
                 │
                 ▼
              _dispatch              ← NPCEngine APIs
        (add_quest / inject_event /
         add_knowledge — tagged with
         source="director" so they
         don't echo into next snapshot)
                 │
                 ▼
            FactLedger.add          ← embed + similarity check
                 │                    against all prior injections
                 ▼
            _save_state              ← persist tick + decisions
                                       + player actions + ledger
```

### The four input channels

1. **World state** — `pie.npc_knowledge.profiles`, capability managers,
   player quest tracker
2. **Director's own history** — `recent_decisions` (ALREADY DONE block,
   prevents text repeat) + `_kind_rotation_index`
3. **Player events** — `recent_player_actions`, populated by either
   the explicit `record_player_action()` API or the dialogue auto-feed
   in `NPCEngine.process()`
4. **Lore bible** — `data/story_director/ashenvale_lore.md` + few-shot
   examples in `examples.yaml`

### The dialogue auto-feed loop

```
Player → engine.process(text, npc_id="noah")
            │
            ▼
       PIE generates NPC dialogue (existing path)
            │
            ▼
       gossip + trust ripple (existing path)
            │
            ▼
       story_director.record_player_action(   ← NEW
           f"Player said to noah: {text}",
           target="noah",
       )
            │
       (next /story/tick reacts)
```

One line of glue in `NPCEngine.process()` closes the loop. No client
code changes needed.

---

## Model comparison

Same prompt, same world state, same 10-tick session. All times include
the GGUF load.

| Model | Size | Dispatch | Schema | Narrative | Repeats | Latency | Noops |
|---|---|---|---|---|---|---|---|
| Qwen 2.5 0.5B Instruct | 469 MB | 10/10 | fair (via coercion) | literal, fair | 1/9 | 1.3s | 2 |
| Llama 3.2 1B Instruct | 770 MB | 10/10 | **broken** (no JSON) | — all noops | 0/9 | 0.5s | 10 |
| **Qwen 2.5 3B Instruct** | **1.9 GB** | **10/10** | **excellent** | **coherent arcs** | **0/9** | **1.4-3.8s** | **0** |
| Llama 3.2 3B Instruct | 1.9 GB | 9/10 | fair (meta-prose) | best creativity | 7/9 | 5.2s | 1 |

### The Llama-1B finding

Llama 3.2 1B is actively *worse* than Qwen 0.5B on this task despite
being 60% larger. It refuses JSON, returns prose descriptions, or just
emits the bare word `noop`. This contradicts the intuition that "bigger =
better" and matches the ultralight-coder observation that **Qwen family
handles structured output better than Llama at small sizes, even outside
Coder variants.** Llama is chat-tuned and wants to converse, not emit
schemas.

**Practical implication:** Don't reach for Llama under 3B for any
schema-driven task. Qwen is the small-model winner.

### The Qwen-3B vs Llama-3B finding

Llama 3B has the **best raw creativity** — invented "Village Square
Riot" out of nothing, directly invoked the sealed letter from the lore.
But it also tries to *teach the user the schema* mid-response (`(Note:
You can use the format shown above, or a simplified version like…)`),
emits two JSON blocks back-to-back, and runs at 2x the latency.

**Practical implication:** Qwen 3B is the production pick. Llama 3B is
useful for one-off content generation where you can review by hand.

### The Qwen-0.5B finding (the cheap path)

With the full architectural scaffolding in place — coercion, brace
matcher, focus NPC override, action kind override — Qwen 0.5B produces
**structurally identical output to Qwen 3B at 1.3s/tick (half the
latency).** The only difference is content quality: 0.5B copies few-shot
examples more literally, 3B invents.

This is the ultralight-coder thesis playing out exactly: *engineering
compensates for model-side weakness until it stops paying off*. For a
latency-critical use case (real-time game director), 0.5B + heavy
scaffolding is viable. The remaining content gap is a curation problem,
not a model size problem.

---

## The empirical journey, by iteration

| # | Configuration | Repeat rate | Noops | Key failure |
|---|---|---|---|---|
| 1 | 0.5B, no anti-repetition | 7/9 | 1 | Death-spiral repetition + greedy JSON regex |
| 2 | 0.5B + ALREADY DONE block + repair retry | 2/9 | 5 | Repair preamble echoed as prose |
| 3 | 0.5B + brace-matching extractor + short repair | 1/9 | 2 | Returns to earlier themes when block scrolls out |
| 4 | Llama 1B (untouched) | 0/9 | 10 | Refuses JSON entirely |
| 5 | Qwen 3B (untouched) | 7/9 | 0 | Thematic fixation on one plot thread |
| 6 | Llama 3B (untouched) | 7/9 | 1 | Most creative + meta-prose breakdowns |
| 7 | Qwen 3B + COOLDOWN block | 8/9 | 0 | Hides behind `target: all` |
| 8 | Qwen 3B + hardened cooldown | 8/9 | 0 | Ignores cooldown, keeps targeting `all` |
| 9 | **Qwen 3B + forced focus NPC** | **0/9** | **0** | **—** |
| 10 | Qwen 3B + forced focus + action rotation | 0/9 | 0 | Player reactivity 0/4 |
| 11 | Qwen 3B + reactive focus + rotation | **1/9** | **0** | **Player reactivity 4/4** |
| 12 | + dialogue auto-feed + FactLedger | **0/9** | **0** | **4/4 reactivity, 1 ledger warning (correct)** |

---

## Failure modes encountered (and what fixed each)

### 1. Repetition death spiral
**Symptom:** Ticks 5–10 of the very first run all said *"Roderick
finds out a tax collector's body was found on the north road."* Six
near-identical events.

**Root cause:** Director-emitted events showed up in the *next* tick's
snapshot under "Recent events", so the Director saw its own outputs as
world state and copied them. Classic feedback loop.

**Fix:** Tag events injected by the Director with `source="director"`.
Filter them out of `_world_snapshot()`. List the Director's own past
actions in a separate `ALREADY DONE` block so the model knows what it's
already fired but doesn't see them as world state to react to.

### 2. JSON parse errors from greedy regex
**Symptom:** When the model emitted two JSON blocks back-to-back, the
parser tried to consume both and failed.

**Root cause:** `re.compile(r"\{.*\}", re.DOTALL)` is greedy.

**Fix:** Replaced with a brace-matching scanner that respects string
escapes and returns the *first* balanced object. See
`_extract_first_json_object()`.

### 3. Repair preamble echoed as prose
**Symptom:** Tick 1 of the second iteration produced raw prose: *`"noop"
JSON object and nothing else. The only action is to return to the
setting...`*. The model regurgitated the repair instructions instead of
producing JSON.

**Root cause:** The verbose repair preamble I added on parse failure
confused the 0.5B — it can't distinguish instruction from content.

**Fix:** Shortened to one line + `ACTION:` anchor:
```
(respond with JSON only)
ACTION:
```

### 4. Schema-drift noops
**Symptom:** Model emitted `{"action": "noop", "quest": {"id": "none"}, "npc_id": "none"}`
— hallucinated schema fields tacked onto a noop.

**Fix:** Strip everything except `action` and `reason` from noops in
`_coerce_action()`.

### 5. Mislabeled action shapes
**Symptom:** First-ever real LLM run emitted `{"action": "event", ...,
"npc_id": "guard_roderick", "fact": "I am watching Mara...", "fact_type":
"personal"}`. The label said "event" but the field shape was "fact".

**Fix:** Coercion in `_parse_action`. If the label doesn't match its
required fields, infer the correct label from the fields present. This
is the most common 0.5B failure mode and the cheapest scaffolding —
~40 lines salvaged hours of model fighting.

### 6. Thematic fixation (Qwen 3B)
**Symptom:** Qwen 3B with raw scaffolding picked one plot thread (Mara's
hidden package) and spent 9 of 10 ticks elaborating it. Text varied each
tick so the ALREADY DONE block didn't catch it.

**Failed fix attempt:** Added a COOLDOWN block — *"do NOT target these
NPCs"*. The model hid behind `target: "all"` to bypass single-NPC
cooldowns.

**Failed fix attempt #2:** Hardened cooldown to also ban `"all"` after
two uses. Model ignored the rule and kept targeting `"all"` anyway —
schema defaults and few-shot examples were more salient than the rule.

**Real fix:** Stop asking the model to pick the target. **Python picks
the focus NPC** via least-recently-touched rotation, injects it as a
`FOCUS NPC` block at the very end of the prompt (recency bias), and
overrides at dispatch time if the model deviated. Result: 0/9
consecutive repeats, 6 unique targets in 10 ticks.

### 7. Action-kind monoculture
**Symptom:** Qwen 3B with focus NPC defaults to `event` 100% of the
time. Quests and facts never get airtime.

**Fix:** Same pattern — Python rotates `event/quest/fact` each tick,
skipping `quest` when the focus NPC has 2+ open quests. Override at
dispatch via `_enforce_action_kind` which can salvage content from a
mislabeled action by wrapping it in the target schema.

### 8. Player reactivity = 0/4
**Symptom:** After scripting player actions between ticks, the Director
saw them in the snapshot but completely ignored them — kept running its
own arc.

**Root cause:** `_pick_focus_npc` only looked at `recent_decisions`,
not `recent_player_actions`. Rotation was *fighting* reactivity.

**Fix:** Two-layer focus selection:
1. If a player action targeting a known NPC was recorded *since the
   last tick*, prioritize that NPC.
2. Otherwise, least-recently-touched rotation.

This preserves variety when the player is passive and breaks rotation
when the player does something. Result: 4/4 reactivity.

### 9. PIE response cache contaminating dialogue tests
**Symptom:** Dialogue calls in the bench reported `0.00s` generation
time on the second run. The bench was hitting PIE's response cache and
returning identical replies to the prior run.

**Fix:** Bench `--reset` now also clears
`PIE_ROOT/data/cache/response_cache.json`.

---

## What the FactLedger is and isn't

The FactLedger:
- Embeds every successful injection with sentence-transformers
- Computes cosine similarity against every prior entry
- Flags pairs above 0.6 with a `similarity_warning` on the dispatch result

The 0.6 threshold was tuned empirically. all-MiniLM-L6-v2 is paraphrase-
trained, so:
- ≥0.85 → near-paraphrase (probably duplicate)
- 0.6–0.85 → same topic, different specifics (worth surfacing)
- <0.6 → unrelated

In the final 10-tick session, exactly **one** warning fired:
- T2 quest *"The Disappeared Healer"* matched T1 event *"Bess heard
  whispers that Elara is not speaking of her grandmother's
  disappearance"* at 0.70.
- These ARE about the same plot thread. T2 escalated T1.

**What the ledger is NOT (yet):** contradiction *detection*. It's
similarity *surfacing*. To turn a similarity warning into a contradiction
flag you need an NLI model that classifies the pair as
entailment / neutral / contradiction. The ledger gives that future layer
the candidate pairs to check.

This is the first piece of *semantic awareness* in the stack — Python
now knows when two things the Director said are about the same subject.

---

## Latency profile

Measured on the production-pick configuration (Qwen 2.5 3B, all
scaffolding enabled), 10-tick session with 4 dialogue interruptions:

| Stage | Time |
|---|---|
| Boot + GGUF load | ~3s (one-time) |
| Sentence-transformer load (lazy, first ledger call) | ~3s (one-time) |
| Director tick (no dialogue) | 1.4–3.8s |
| Dialogue turn (engine.process LLM call) | 4.5–6.5s |
| FactLedger encode + similarity check | <50ms |
| Mean tick time | 2.5s |
| **Total session (10 ticks + 4 dialogues)** | **~38s** |

For a real-time game loop targeting one tick every 30 seconds with
occasional player dialogue, this fits comfortably. For a turn-based game
where ticks fire on demand, it's instantaneous from the player's
perspective.

---

## What works

- **Long-range plot continuity.** In the final session, the player asked
  Noah about the sealed letter on T3. Tick 10 (no player input) produced
  *"Noah the elder is found unconscious in his study, his eyes glazed
  over and a strange, glowing crystal on his desk."* Seven ticks later,
  with no scaffolding telling the model to remember.

- **Direct lore engagement.** The Director references the lore bible
  spontaneously: Noah's sealed letter, the Silverwood elven ruins,
  Elara's grandmother Mira, the wolf bounty, the missing tax collector.
  No prompting beyond the lore file in the prompt.

- **Cross-NPC plot weaving.** Pip → Kael linkage, Pip → Mara
  observation, Roderick watching Kael and Mara, Bess overhearing
  everyone. The Director makes connections across the social graph.

- **Player intent extraction from dialogue.** Player asks *"What's the
  letter say?"* → Director generates a fact about Noah hiding secrets.
  Player accuses Mara → Director gives Mara a quest to prove innocence.
  The Director reads the *subtext* of dialogue, not just the surface.

- **Action-shape variety.** Equal mix of events, quests, and facts —
  not because the model chose them, because Python rotated them and the
  model wrote the content for whatever kind it was given.

- **Real-time NPC rotation.** Six of seven NPCs touched in 10 ticks, no
  consecutive repeats, deterministic rotation when the player is quiet.

## What doesn't work yet

- **Contradiction *detection*.** The FactLedger surfaces semantically
  similar content. It doesn't classify entailment vs contradiction. A
  small NLI model (~`cross-encoder/nli-deberta-v3-base` or smaller) over
  flagged pairs would close this gap.

- **Player quest tracking integration.** `record_player_action("Player
  completed Kael's quest")` records the *text* but doesn't update
  `pie.player_quests`. A real game loop would parse intent and call
  `engine.complete_quest("missing_hammers")`.

- **Multi-action ticks.** One action per tick. Real Cardinal would plan
  arcs of 2–3 simultaneous moves. The forced-focus + rotation discipline
  is now safe enough that an architect/worker decomposition could plan
  multiple moves without thrashing the rotation layer.

- **NPC identity bleed in dialogue.** Noah called the player *"Mara"*
  and Mara called the player *"Kael"* in test dialogues. This is in the
  npc-engine *dialogue path*, not the StoryDirector — separate concern.

- **0.5B re-bench with the full stack** is unverified. Probably fine
  given 0.5B passed the structural tests on every prior iteration, but
  not formally re-run after the player hook + ledger landed.

---

## Where this stops being scaffolding

When the session started, the Director was a single LLM call with a
prompt template and three action types. By the end it's:

- **A deterministic planner** (focus NPC + action kind + reactive
  override + cool-down rotation)
- **A creative LLM call** (one job: write content)
- **A coercing parser** (brace-matching JSON + label/shape inference +
  schema-strip on noops)
- **A tagged action dispatcher** (events tagged so they don't echo back)
- **A persistent tick state** (decisions, player actions, ledger)
- **A semantic ledger** (every injection embedded + similarity-checked)
- **A dialogue auto-feed loop** (every player turn observed)
- **A REST surface** (`/story/tick`, `/story/state`, `/story/player_action`)

Every single piece exists because an empirical run produced a failure
that pure prompting couldn't fix. Nothing in this list is speculative
engineering — every component has a corresponding "this ran, this broke,
here's why" story.

The Cardinal pattern from SAO is real and shippable on local hardware,
**but only if you stop asking the model to make decisions it can't
make.** That's the whole insight in one sentence.

---

## File map

```
npc-engine/
├── npc_engine/
│   ├── story_director.py     # The service class + FactLedger
│   ├── engine.py             # +14 LOC: instantiate + dialogue auto-feed
│   └── server.py             # +43 LOC: 3 new REST endpoints
├── tests/
│   └── test_story_director.py  # 35 tests (33 offline + integration)
├── data/
│   └── story_director/
│       ├── ashenvale_lore.md     # ~30 lines of setting bible
│       ├── examples.yaml         # 5 few-shot world-state→action pairs
│       ├── FINDINGS.md           # this file
│       ├── state.json            # gitignored runtime state
│       └── fact_ledger.json      # gitignored ledger (200-entry cap)
├── bench_story_director.py   # Model comparison + scripted sessions
└── .gitignore                # +9 LOC: runtime files, traces, backups
```

## Reproduction

```bash
cd D:/LLCWork/npc-engine

# Unit tests (no model needed)
python tests/test_story_director.py

# Best production session: Qwen 2.5 3B with dialogue + ledger
python bench_story_director.py --ticks 10 --reset \
    --model qwen_3b --dialogue-script --log session.json

# Cheapest path: Qwen 0.5B with full scaffolding
python bench_story_director.py --ticks 10 --reset --model qwen_05b

# REST server
python -m npc_engine.server
# POST /story/tick
# POST /story/player_action {"text": "...", "target": "noah", "trust_delta": 5}
# GET  /story/state
```
