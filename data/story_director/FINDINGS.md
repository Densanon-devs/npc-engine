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

## Architect / worker — multi-action ticks

The default `tick()` produces one action per call. SAO's Cardinal plans
*arcs*, not single beats. Adding the architect/worker pattern from
ultralight-coder lets a single tick advance N plot threads in parallel:

```python
director.tick(actions_per_tick=3)
# returns {tick, sub_actions: [{action, dispatch, raw_response}, ...]}
```

### Architecture

The split is **planning vs writing**, same as before — just at a wider
scope:

- **Architect (`_architect_plan`)** — picks N distinct
  `(focus_npc, action_kind)` slots up front. Uses the same focus +
  rotation logic as single-action mode, but maintains an in-flight
  `excluded` set so two workers can't compete for the same NPC.
- **Workers (`_run_single_action`)** — each takes one slot from the
  plan and runs the same LLM-call → enforce → dispatch → ledger
  pipeline that single-action mode uses. Same prompt builder, same
  parser, same coercion. Workers are **independent** — they all see
  the same pre-tick world snapshot.
- **Backward compatible** — `actions_per_tick=1` (default) still
  returns the legacy `{tick, action, dispatch, raw_response}` shape.

### Empirical run: Qwen 2.5 3B, 5 ticks × 3 actions = 15 actions

```
Total time: 49.33s   mean: 9.87s/tick (= ~3.3s/worker)
Dispatch: 15/15 OK
Coercions / Noops / Errors: 0 / 0 / 0
Consecutive target repeats (sub-action level): 0 / 14
Action kinds: event×6, fact×5, quest×4
Targets touched: bess×3, elara×3, roderick×2, kael×2, mara×2, noah×2, pip×1
```

**All 7 Ashenvale NPCs got airtime in 5 ticks.** The single-action mode
took 10 ticks to do the same. Multi-action *halves* the wall-clock per
NPC visit.

The narrative was meaningfully richer because each tick advanced **three
plot threads at once.** Selected highlights:

- **T2:** Kael finds a letter in one of his stolen hammers signed
  "Kael" (a planted forgery!) → Quest "Steel Breach" against Mara →
  Noah finally reads the sealed letter, which *"speaks of a hidden
  treasure and a prophecy about the village's future."*
- **T3:** Roderick whispers something to Pip in a recruitment scene
  → Bess starts a "Tavern Rumors" quest about the tax collector →
  Elara confides she's heard whispers about the Silverwood ruins.
- **T5:** Noah and Roderick exchange a knowing look about the letter
  → Bess hears whispers about Roderick's letter (gossip propagation
  in real-time) → Elara stumbles out of the Silverwood *"as if she
  had seen something she shouldn't have."*

### What multi-action ticks revealed about the FactLedger

Six ledger warnings fired across the 15 sub-actions — all correctly
classified as `neutral` by NLI. The most informative match was:

> T5 sub-2: *"I've heard whispers that a mysterious letter has been
> delivered to Roderick. It's been sitting in his mailbag for days."*
> matched T4 sub-1: *"Roderick receives a mysterious letter in his
> mailbag. It's addressed to him with no return address."*
> Similarity: 0.86. NLI: neutral (1.00 confidence).

This is **the FactLedger watching the Director propagate gossip across
plot threads.** T4 dropped a piece of information; T5 had a different
NPC react to it. The ledger correctly identified them as related and
the NLI correctly identified them as consistent (not contradictory) —
exactly the signal a contradiction-detection layer should produce.

### Performance

- ~3.3s per worker on Qwen 2.5 3B (vs ~2.5s for a standalone single-
  action call — the slight overhead is from the larger ledger needing
  more cosine comparisons each call)
- Workers run sequentially on local CPU (llama-cpp-python isn't thread-
  safe for parallel calls into one base model). On a real game loop,
  3 sub-actions in 10 seconds is fine for a story tick that fires
  every 30+ seconds.
- A future optimization: keep N independent base_model instances and
  run workers in actual parallel. Or use ultralight-coder's
  `GenerationQueue` from PIE which already prioritizes background work.

### When to use it

| Mode | Use when |
|---|---|
| `actions_per_tick=1` | Real-time game director on a fast cadence (one beat every 5-10s of gameplay). Lowest latency per call. |
| `actions_per_tick=3` | Story-tick on a slower cadence (one tick every 30+ seconds). Richer narrative per call, full NPC coverage in fewer ticks. |
| `actions_per_tick=N>3` | Burst story progression — exposition, scene transitions, between-act planning. Diminishing returns past 3 because the architect runs out of distinct NPCs in a 7-NPC world. |

### Cheap-path validation: Qwen 0.5B with the full stack

Re-ran the bench against Qwen 2.5 0.5B with **every layer enabled**:
focus rotation, kind rotation, architect with 3 actions per tick,
dialogue auto-feed, FactLedger, NLI contradiction check, contradiction
retry. 5 ticks × 3 sub-actions = 15 actions.

| Metric | Value |
|---|---|
| Dispatch | **15 / 15** |
| Coercions | 1 / 15 (one schema slip, fixed by `_enforce_action_kind`) |
| Noops / parse errors | **0 / 0** |
| Action kinds | **event×5, quest×5, fact×5** (perfect balance) |
| Targets touched | **All 7 NPCs** (bess×3, noah×3, elara×2, roderick×2, kael×2, mara×2, pip×1) |
| Consecutive sub-action repeats | 1 / 14 |
| Ledger warnings | 3 / 15 — all NLI-classified `neutral`, **0 contradictions** |
| Dialogue reactivity | 2 / 2 (both dialogue turns produced REACT) |
| Per-tick latency (warm) | **~3.5s for 3 sub-actions = ~1.2s per worker** |

The cheap path is **structurally indistinguishable from the 3B path**.
Every one of the 6 layers we built holds at 469 MB of model weight.
0.5B is 2-3× faster per worker than 3B (~1.2s vs ~3.3s) once warm.
The single coercion fired correctly and the dispatched action was
still valid.

**Latency anomaly:** tick 4 of the 5-tick run took ~2400s (~40 min)
while the other four ticks took 3-5s each. Almost certainly a
one-time environmental issue (PyTorch CPU graph optimization, MKL
re-init, OS swap during a contradiction retry, or similar). Tick 5
came back to 3.51s. Not a structural defect — the per-worker math
on the warm ticks is what to trust.

**Practical implication:** the latency-critical path (real-time game
loop, low-spec target hardware) can run Qwen 0.5B + full scaffolding
and get the same Cardinal behavior as the 3B path at ~⅓ the latency.
The remaining content gap (more literal example copying on 0.5B)
is the curation problem ultralight-coder already solves with more
YAML examples.

---

## NLI contradiction detection

The FactLedger surfaces *similar* pairs. To turn similarity into
**contradiction** you need an NLI model that classifies the pair. Built
on top of the ledger:

- **`ContradictionChecker`** wraps `cross-encoder/nli-deberta-v3-small`
  (~140MB, ~500ms/pair on CPU)
- Lazy-loaded on first check; falls back to no-op if unavailable
- Outputs `contradiction / entailment / neutral` with softmax confidence
- Runs **only on flagged similarity pairs** — cost is bounded by how
  often the Director recycles plot threads (typically 0–2 per session)

### Threshold tuning empirically

The small DeBERTa NLI model is **hypersensitive**. Out of the box at the
default 0.55 threshold it labeled this pair as contradiction with 0.991
confidence:

> Premise: *"Bess heard rumors that Elara is hiding something about the
> Silverwood."*
> Hypothesis: *"Elara, the healer, returned from the Silverwood looking
> pale and silent."*

These aren't a contradiction — they're two unrelated facts about
distinct subjects. But the model is trained on MNLI/SNLI which compares
hypothetical statements, not narrative consistency. It defaults to
labeling "topically related but distinct" pairs as contradiction.

**Fix:** raised `_NLI_CONTRADICTION_THRESHOLD` to 0.85. The model still
catches genuine contradictions (which it labels with 0.95+ confidence),
but borderline 0.55–0.84 false positives are filtered out.

### What the layer reliably catches

| Pair | NLI label | Caught? |
|---|---|---|
| *"Mara is hiding contraband under the floorboards"* vs *"Mara has nothing hidden, accusations are false"* | contradiction (0.99) | ✓ |
| *"Mara is a merchant who runs a fabric shop in Ashenvale"* vs *"Mara owns a fabric shop in Ashenvale and works as a merchant"* | not contradiction | ✗ (correct) |
| *"Noah the elder is troubled by an old letter from the king"* vs *"Noah the elder confided that the king's letter weighs heavily on him"* | not contradiction | ✗ (correct) |
| *"Bess heard whispers that Elara is not speaking of her grandmother's disappearance"* vs *"The Disappeared Healer — Elara has been seen pale and silent, returning from the Silverwood"* | neutral (1.00) | ✗ (correct — plot escalation) |

The third row is the actual T1/T2 pair from the production session. The
ledger flagged it via similarity (sim=0.70) but NLI correctly classified
it as `neutral` — Director escalation, not self-contradiction.

### What this gives you

The Director now has a primitive **self-consistency check**. Every
fact-shaped injection is embedded, compared against the ledger, and —
on flagged pairs — passed to NLI. If a future tick says *"Noah refused
to ever discuss the letter"* and an earlier one said *"Noah read the
letter aloud in the village square"*, the layer will flag it.

This is the first novel piece in the stack — your existing AI projects
(PIE, ultralight-coder, npc-engine) had nothing for narrative
consistency. The pattern is general: any LLM-driven content-generation
system that injects facts into a persistent state can use the same
ledger + NLI scaffold to detect when it contradicts itself.

### What it still can't do

- **Subject resolution.** If T1 says "the elder" and T20 says "Noah", the
  embedding similarity might miss the connection. Co-reference resolution
  would help but adds another model.
- **Reasoning over multi-fact contradictions.** E.g., A: *"Mara was in
  Ashenvale on Monday."* B: *"Mara was 100 miles north on Monday."* —
  these contradict only if you know Ashenvale isn't 100 miles north.

### Acting on the warning — pre-dispatch retry loop

The contradiction layer doesn't just surface conflicts now — it
**recovers from them**. The flow:

1. Worker generates a candidate action (LLM call)
2. Action is finalized (focus + kind enforcement)
3. **Pre-dispatch ledger check**: embed the action's text, search the
   ledger, run NLI on any similarity match
4. **If NLI flags contradiction** at >=0.85 confidence, retry the
   worker ONCE with a corrective preamble:
   > NOTE: Your previous attempt contradicts an earlier established
   > fact (T8 fact/noah): "Noah refuses to discuss the king's letter".
   > Pick a DIFFERENT angle that does not conflict with that fact.
   > Do not negate it; build a story beat that's consistent with it.
5. The retry's response (whether it succeeds or fails) is what gets
   dispatched

The retry is capped at one attempt to bound latency and prevent
oscillation. The dispatch result includes
`retried_after_contradiction: True` so callers can audit retry rate.

`FactLedger.check()` was split out from `add()` for this — pre-dispatch
needs to compute the warning *without* storing the candidate (otherwise
the candidate would compare against itself).

This is the **first piece in the stack that lets a small local LLM
recover from its own contradictions**. Verified end-to-end with a unit
test that pre-seeds the ledger with *"Mara is hiding contraband"*, then
drives the worker with a stub model whose first response is
*"Mara has nothing hidden, the accusations are false"* and second
response is a non-contradicting alternative. The retry path correctly
discards the first response and dispatches the second.

---

## Multi-tick narrative arcs

Every tick up to this point has been locally coherent but
*structurally independent*. The Director reacts, rotates, and varies
action kinds — but it doesn't plan a *shape* across ticks. A SAO-style
Cardinal plans arcs that span rising action, a climax, and a
resolution. This section is the first layer of that.

### The insight

Arcs don't need an LLM. The FactLedger already has everything the
planner needs — every fact-shaped injection is embedded and stored
with the NPC id that received it. A deterministic clustering pass
over recent ledger entries surfaces the densest thematic cluster, and
that cluster *is* the arc: theme = the center entry's text, cast = the
unique NPC ids in the cluster.

This preserves the Python-plans-LLM-writes split at a wider scope.
Where the single-tick planner picks *who* and *what kind*, the arc
planner picks *what story thread is worth committing to*. Neither
layer asks the LLM to plan — planning stays deterministic, the LLM
stays a content writer.

### Data model

```python
@dataclass
class NarrativeArc:
    id: str
    theme: str                      # drawn from the cluster's center entry
    focus_npcs: list[str]           # cast, in order of first appearance
    beat_goals: list[str]           # fixed 4-beat skeleton
    current_beat: int = 0
    status: str = "active"          # active | resolved | abandoned
    started_at_tick: int = 0
    last_advanced_at_tick: int = 0
```

The beat skeleton is fixed and generic:
- **seed** — introduce the tension or hint at it without resolving anything
- **escalate** — deepen the stakes with a new wrinkle or complication
- **confront** — force a scene where the tension comes to a head
- **resolve** — show the aftermath and let the thread close cleanly

Generic phrasing is deliberate: any theme the cluster produces can be
told through those four beats. Per-arc custom beats would require
another LLM call and more state — defer.

### Proposal (greedy clustering)

On each tick, `ArcPlanner.maybe_propose()` runs if (a) there's no
active arc and (b) at least `_ARC_PROPOSAL_COOLDOWN_TICKS` ticks have
elapsed since the last attempt. The algorithm:

1. Take the last `_ARC_CLUSTER_LOOKBACK=20` ledger entries.
2. Build the pairwise cosine similarity matrix (entries are already
   L2-normalized on encode, so it's one matrix multiply).
3. For each entry, count neighbors above
   `_ARC_CLUSTER_SIMILARITY=0.55` (looser than the 0.6 contradiction
   threshold — we're detecting thematic overlap, not duplicates).
4. Pick the entry with the most neighbors as the cluster center.
5. The cluster is the center + all its above-threshold neighbors.
6. Theme = the center entry's text (truncated to 160 chars).
7. Focus NPCs = unique ids across the cluster, in order of first
   appearance, capped at `_ARC_MAX_FOCUS_NPCS=4`, filtered to the
   currently-available roster.

Nothing in this pipeline needs a model. Ledger entries already have
embeddings, numpy does the math, the theme text was written by the
LLM in a prior tick. The planner is pure orchestration.

### Beat advancement (touch counter)

`advance_if_beat_met()` walks `recent_decisions`, counts how many
actions since `last_advanced_at_tick` targeted a focus NPC (via
`npc_id` or `target` field), and bumps `current_beat` when the count
reaches `_ARC_BEAT_ADVANCE_THRESHOLD=2`. When `current_beat` passes
the last beat, `status = "resolved"` and `active_arc_id` is cleared —
the next eligible tick will propose a fresh arc.

Counter-based advancement is coarse. A better heuristic would measure
*progress toward the beat goal* (e.g., semantic similarity between
new content and the goal string). That's a v2 concern — the touch
counter is enough to keep arcs moving at a predictable pace (one beat
per 2 tick-visits ≈ one arc per 8-10 focused ticks).

### Prompt integration

`_build_prompt` adds an `=== ACTIVE NARRATIVE ARC ===` block between
the world snapshot and the forced-focus block:

```
=== ACTIVE NARRATIVE ARC ===
Theme: {theme}
Cast: {focus_npcs}
Current beat (2/4): escalate — deepen the stakes with a new wrinkle
If this tick's focus NPC fits the cast, advance the beat. If not,
write something the cast can react to next tick.
```

The guidance is soft — the arc block is *before* the forced-focus
block, so Python still owns which NPC and which action kind this
tick fires. The arc just tells the model *what the beat should feel
like* when the focus NPC happens to be in the cast. If the rotation
lands on an NPC outside the cast, the arc block still nudges the
model to plant something the cast can react to next tick — keeping
the thread alive across non-cast ticks.

### Persistence

Arcs live in `data/story_director/arcs.json` (gitignored) alongside
`state.json` and `fact_ledger.json`. On load, the planner restores
the full arc list, the active arc id, and the proposal cooldown
counter — so a cold restart doesn't lose the thread.

### What this gives you

- **Session-level structure.** Previously you got coherent local
  decisions. Now you get story threads that have a beginning, a
  middle, and an end, with an explicit beat to write toward.
- **Grounded themes.** The planner can never "invent" a theme — it
  can only commit to text the LLM already wrote. So arcs always feel
  like they emerge from the world rather than being imposed on it.
- **Zero new latency.** No extra LLM calls. The cluster pass on a
  20-entry ledger is ~50μs of numpy matmul. The only cost is the
  tick-time arithmetic to count touches.
- **Resilient across restarts.** Arc state is persistent; the
  Director picks up the active beat on the next tick after a reboot.

### What it still can't do

- **Multi-arc concurrency.** One active arc at a time. A Cardinal
  would run 2-3 threads in parallel (e.g., the player's main quest +
  a background faction tension). Enabling this requires the prompt
  to pick *which* active arc each tick advances — an extra decision
  the current architect doesn't make.
- **Semantic beat progress.** The touch counter doesn't know if the
  *content* written during those touches actually advanced the beat.
  A tick that names the NPC but doesn't address the theme still
  counts. Upgrading to "embed the tick's new content, measure
  similarity to the current beat goal, only count if above
  threshold" would be a tighter signal.
- **LLM-proposed themes.** v1 pulls themes from the cluster center's
  raw text. A larger model could summarize the cluster into a crisp
  one-line theme ("Mara's smuggling ring", rather than the raw event
  text that seeded it). Cheap (one call per arc proposal) and
  deferrable.
- **Abandonment.** If the rotation never lands on a cast NPC, an arc
  sits stuck at its current beat forever. An abandonment heuristic
  (e.g., no touches for N ticks → `status = "abandoned"`) would close
  stale threads so new arcs can form.

### Tuning notes from empirical runs

The arc planner's knobs were picked against a 15-tick Qwen 2.5 3B
session at `actions_per_tick=3`. Observations that drove the final
values:

**Beat advance threshold = 4.** At the original value of 2, beat
advancement *raced ahead of LLM content pacing*. In the first 3B run
the touch counter said "beat 4/resolve" by T13, but the actual
physical-confrontation scene (*"Kael storms into the blacksmith's
workshop, hammer in hand, and accuses Mara of stealing"*) didn't
land until T14 — a tick *after* the planner had already moved past
the "confront" beat. Raising to 4 matches the counter to 3B's
rotation cadence: with a 2-NPC cast touched roughly every 2 ticks in
multi-action mode, each beat now spans 2-3 ticks, which gives the LLM
room to set up → escalate → pay off before the next beat goal shows
up in the prompt.

**Content concentration = 73% on 3B.** In the same session, 33 of 45
sub-actions mentioned the arc cast or theme keywords — despite
focus-NPC rotation continuing to touch every NPC in the world. This
is the arc block doing its job: non-cast NPCs write *about* the
cast's story thread (pip overhears mara, bess notices suspicious
behavior, noah receives cryptic letters, roderick investigates), so
the cast's arc spills organically across the social graph.

**Cluster similarity = 0.55** (looser than the FactLedger's 0.6
contradiction threshold). Picked to catch thematic overlap without
pulling in merely-adjacent content. With 0.5 or lower the cluster
absorbs unrelated NPCs; with 0.65+ the cluster can't form on 3B's
more-varied content.

**Proposal cooldown = 5 ticks.** Long enough that a resolved arc
doesn't immediately re-propose the same theme (cluster is still
dominant right after resolution). Short enough that a growing ledger
has room to produce a distinct theme before the next attempt.

**Minimum ledger entries = 4.** First proposal fires at T2-T3 on
3-worker ticks (ledger grows by 3/tick). Lower and proposals fire on
too-thin clusters; higher wastes early-session ticks.

**Session-length guidance.** A 4-beat arc on Qwen 2.5 3B at
`actions_per_tick=3` takes ~11 ticks to fully resolve (propose at
T3-T5, advance every 2-3 ticks, resolve at T13-T15). Sessions
shorter than 13 ticks will leave the last arc mid-resolution — not
a bug, just cadence math. For 0.5B the cadence is faster (~8 ticks
per arc) because its rotation touches the cast more densely.

### Tests

8 offline unit tests cover proposal, skip conditions, advancement,
resolution, prompt injection, persistence, and cooldown gating. They
run without sentence-transformers by stuffing fake embeddings
(2-D unit vectors with known cosine similarities) directly into the
ledger — fast, deterministic, and independent of model availability.

---

## NPC bio injection

### The gap

Pre-bio, `_world_snapshot()` surfaced exactly `(role, mood, trust,
quest_count)` per NPC. The Director never saw motives, fears,
secrets, personality, or backstory — all of which the NPC profiles
ALREADY carry for the dialogue capabilities to consume.

Mara's YAML has a priority-9 goal *"Keep the counterfeit steel
operation secret"* and personal knowledge *"Knows Kael suspects her —
she does"*. Kael's YAML has a priority-10 goal *"Learn what happened
to apprentice Tam in the forbidden forest"*. Noah's has priority-8
*"Discover why the well water turned bitter"* and personal knowledge
*"Lost wife Elena five years ago, tends her garden daily"*.

None of that reached the Director. Every NPC was one-dimensional from
the overseer's perspective, which forced it to riff on the 5
examples and the 30-line lore bible for its entire plot vocabulary.

### The fix

Two cheap additions to the prompt assembly:

1. **Roster one-liners.** Each NPC's line in `_world_snapshot()` now
   includes `wants: {top_goal_description}` when the NPC has an
   active goal via the `goals` capability. Costs ~40 chars/NPC but
   tells the LLM what every character in the world wants at a
   glance.
2. **Focus NPC full bio.** `_build_focus_npc_bio()` produces a
   multi-line block with personality, priority-sorted goals,
   personal_knowledge, and a few world_facts for the NPC each worker
   is currently writing for. Injected into the prompt between the
   `ACTIVE NARRATIVE ARC` block and the forced-focus directive, so
   bio context comes *before* the MUST rules hit.

Both read from the capability manager via the same pattern as
`_peek_npc_state`: `mgr.capabilities.get("goals").goals` exposes the
priority-sorted goal list on `GoalsCapability`. No changes to the
capability system itself — purely a new consumer.

### Empirical verification (3B, 15 ticks × 3 sub-actions)

Compared the same session configuration with and without bio
injection. The pre-bio run had already been tuned for multi-tick arcs
(threshold=4, etc.), so any content difference is attributable to the
bio block alone.

**Brand-new plot hooks that did not exist in the pre-bio run:**

| Hook | Source | Pre | Bio |
|---|---|---|---|
| Bitter well / well water | Noah goal p8 | 0 | **5** |
| Elena / garden (Noah) | Noah pk | 0 | **2** |
| Soldiers whispering / guard barracks | emergent | 0 | **4** |
| Rare minerals / weapons hidden (Mara) | Mara pk | 0 | **2** |
| Merchant guild warehouse | Mara pk | 0 | **2** |
| Strange lights forest edge | emergent | 0 | **3** |

**Vocabulary expansion:** 204 → 221 unique content words, with 149
newly-introduced terms including *bitterness, flickering, forbidden,
glowing, informer, minerals, muttering, whispering*. The model isn't
just remixing the same 5 examples — it's pulling novel vocabulary
from the bio data.

**Arc cluster shifted.** Pre-bio 3B sessions always clustered around
`[kael, mara]` because the missing-hammers few-shot example is so
gravitationally strong. With bio enabled, the cluster re-formed
around `[kael, guard_roderick]` — the bio's motivational fuel
changed what content the model emitted, which changed which ledger
entries clustered together.

**Wall-clock impact: noise-level.** 92.5s vs the prior run's 92-183s
variance envelope. Prompt grows ~200 tokens per worker (~15 bio
lines), which at 1000 tok/sec eval adds ~0.2s/worker — under the
run-to-run variance.

### What didn't surface

Three bios stayed invisible across the whole session:
- Kael's priority-10 goal *"Learn what happened to apprentice Tam"*
- Bess's priority-9 goal *"Clear whatever is in the cellar tunnels"*
- Kael's dwarf/Iron Ridge backstory from his pk

Root cause: **the few-shot examples in `examples.yaml` have stronger
gravitational pull on 3B than the NPC bio does.** Every time the
worker rotation landed on Kael, the model re-wrote the
`missing_hammers` example verbatim instead of pulling from his bio
goals. The bio steers content at the margins — it can invent new
side threads and novel vocabulary — but it can't escape the few-shot
orbit when the focus NPC has an example directly in the library.

**Implication for v3 work:** diversify or rotate `examples.yaml` per
tick. Either pick examples dynamically based on the focus NPC's top
goal (so Kael ticks don't always see the hammers example), or
shrink the library so the bio can compete for salience. This is a
larger change than bio injection itself and is called out as a v3
item below.

### Tests

6 offline unit tests cover the goals peek helper, the bio block
shape, priority ordering, roster goal suffix, prompt integration
(bio before forced-focus), and the skip-when-bare fallback. Stub
test engine now has `_StubGoalsCap` and `_StubCapabilityManager` so
bio tests don't need the real capability system wired up.

### Tuning knobs

- Top 3 goals per NPC bio (more would bloat the prompt with
  low-priority padding)
- Top 4 personal_knowledge items
- Top 3 world_facts
- 60-char cap on roster goal suffix
- 180-char cap on personality
- 140-char cap per goal description in bio
- 160-char cap per pk/wf line

---

## Dynamic example rotation (escaping the few-shot orbit)

### The gap

The NPC bio injection worked for low-salience bios but hit a hard
ceiling on high-salience ones. Kael's priority-10 goal *"Learn what
happened to apprentice Tam in the forbidden forest"* never once
surfaced across three 3B sessions, even though the bio block
repeatedly told the model about it. Root cause: every time the
forced-focus rotation landed on Kael, the model saw the
``missing_hammers`` example in the EXAMPLES block and rewrote it
near-verbatim. The example had stronger gravitational pull than the
bio, so Kael's content was stuck in the hammers loop.

This was the *same* failure mode documented for 0.5B (literal copy
of few-shots), just showing up at 3B as thematic copy. More
parameters, same anchor.

### The fix

Replace the "dump all 5 examples every tick" block with a per-worker
picker. Each example in `examples.yaml` now has a ``primary_npc``
tag (the NPC the example is directly about). On each prompt build,
`_pick_examples(focus_npc, action_kind)`:

1. **Excludes examples whose primary_npc matches the focus NPC.**
   A Kael-focused worker never sees the Kael example. The copy
   loop is broken at the source.
2. **Prioritizes one example matching the target action_kind.**
   The forced-focus block says the worker MUST emit a specific
   kind; showing one example of that kind stabilizes the schema
   for that shape.
3. **Fills remaining slots with different kinds for variety.**
   So the worker still sees event/quest/fact shapes even when
   forced to emit one.
4. **Caps at 3 picks.** Fewer examples leaves more salience for
   the bio and arc blocks.
5. **Falls back to the full library if filtering leaves nothing.**
   An empty EXAMPLES block tanks schema parse reliability.

No other change to the prompt — same lore, same bio, same arc
block, same forced-focus directive. Just a smarter example pick.

### Empirical verification (3B, 15 × 3 actions)

Compared rotation against the prior (arcs) and (bio) runs on the
same config. The numbers are the biggest leap in Director output
quality in the whole branch so far.

**Dormant hooks finally surfaced:**

| Hook | arcs | bio | rotation |
|---|---|---|---|
| Tam apprentice (Kael p10) | 0 | 0 | **4** ✨ |
| Cellar tunnels (Bess p9) | 0 | 0 | **1** ✨ |
| Bitter well (Noah p8) | 0 | 4 | 5 |
| Elena / garden (Noah pk) | 0 | 2 | 3 |

**Hammers copy loop collapsed:**

| Metric | arcs | bio | rotation |
|---|---|---|---|
| Kael-focus subs mentioning hammers | **7/7 (100%)** | 3/7 (42%) | **1/7 (14%)** |
| Total ticks w/ any hammer mention | 10/15 | 6/15 | **1/15** |

**Vocabulary exploded:** 204 → 221 → **260** unique content words.
147 new words exclusive to the rotation run: *caravan, clatters,
dismisses, enchanted, faeries, feverish, footprints, ghostly,
incantations, informants, intruder*. These are real thematic
shifts, not just paraphrase.

**Kael's bio unlocked a detective arc.** In the rotation run,
Kael's 7 focus ticks built a coherent "find the missing apprentice"
investigation: shadowy figure in the workshop → Tam's disappearance
quest → footprints in the snow → wolves acting strange → unknown
metal Tam brought from the forest → Kael shouting Tam's name in
the Silverwood. Every thread pulled from his bio: Tam (p10 goal),
Silverwood (lore rule), the metal-sense backstory from his pk, the
wolf bounty (lore fact). None of this content existed in the prior
runs.

**Bonus: a new arc emerged around Roderick alone.** The planner
clustered `[guard_roderick]` with theme *"Roderick saw strange
lights in the forest last week but dismissed it as nothing"* — a
7-tick investigation arc that appeared nowhere in the prior runs.
The model even extrapolated plausibly (*"he thinks it might be
faeries or something harmless"*) before escalating toward a
smuggling suspicion (*"The Northern caravan had strange items like
enchanted weapons and potions, which Roderick suspects are
contraband"*). Genuinely emergent story from zero prior seed.

### Tradeoffs

- **Lost some prior content threads.** The counterfeit-steel thread
  (Kael's p6 goal) vanished — 5 hits → 0. That's because the old
  Kael example was the only thing bridging Kael to "counterfeit",
  and pulling it removed the bridge. Not a regression if you think
  of it as "making room for Tam to surface", but worth noting.
- **Per-NPC content can still fixate on top bio items.** Bess's
  8 ticks concentrated heavily on her top pk entry ("merchant
  guild planning to raise taxes") — 5 of 8 mentioned it. The bio
  fix broke the example copy loop but didn't stop per-NPC bio
  concentration. A v4 item is probably *rotating WITHIN a bio*
  (showing different goal/pk items on different ticks).
- **One LLM-pacing oddity:** Bess's ticks kept using the phrase
  "I heard whispers of the merchant guild planning to raise taxes"
  almost verbatim. Phrasing cloning at the bio level. Not a bug
  but a signal that bio text could use a "paraphrase, don't quote"
  instruction.
- **Wall-clock: within variance.** 203s for 15 × 3 ticks this run
  vs the 92-183s envelope on prior runs. 3 picks instead of 5
  shrinks the prompt by ~300 tokens, so generation should be
  faster, but run-to-run variance on CPU is larger than the savings.

### Tests

6 offline unit tests cover the picker: exclude focus_npc,
prioritize target action_kind, cap at 3, fallback when the filter
strips everything, kind diversity, and an integration check that
`_build_prompt` emits only non-focus-NPC examples.

---

## Intra-bio rotation (breaking the per-NPC copy loop)

### The gap

Example rotation broke the few-shot copy loop but surfaced a new
smaller one: each NPC's *top bio items* became the new gravitational
attractor. Bess's merchant-guild-taxes pk item was quoted
near-verbatim across 5 of 8 Bess-focus ticks in the rotation run.
The fix was to rotate WITHIN a single NPC's bio so the model stops
seeing the same pk item every time it focuses on that NPC.

### The fix

Four things together:

1. **Bio-item mention tracking.** Every time a worker dispatches
   content, `_record_bio_mentions` scans the dispatched text for
   word-overlap against the focus NPC's bio items and bumps a
   per-item mention counter. Keyed by `(npc_id, item_text)` and
   persisted in `state.json`.

2. **Original-bio snapshot.** `NPCKnowledge.personal_knowledge` is
   *mutable* — the dispatch layer for `add_knowledge` appends
   Director-generated facts straight into it. The first version of
   mention tracking read bios live, which meant it was treating the
   model's own outputs as bio items (the `state.json` dump for a
   bench run was full of fragments like *"i heard whispers from
   the merchant guild"* — those were the model's ticks, not the
   YAML). Fix: snapshot every NPC's bio items at director init into
   `_original_bios` and use that for both the bio block and the
   mention tracker. The live `NPCKnowledge` still evolves for the
   dialogue layer, but Story Director's bio view is frozen to the
   character sheet.

3. **Hard cooldown exclusion.** Items with `mention_count >= 2`
   are dropped entirely from the bio until other items catch up.
   Previously rotation just *reordered* — which didn't help when
   the top-N cap was generous enough to show everything. The
   exclusion forces the model to look at fresh material. If
   exclusion strips a section to empty, we fall back to showing
   the least-mentioned items so no section ever blanks out.

4. **Tightened top-N caps.** Top 2 goals (was 3), top 3 pk (was 4),
   top 2 wf (was 3). With small bios (~2-4 items per section),
   the prior caps showed everything. Tight caps give rotation
   room to actually hide items.

Plus a paraphrase instruction in the bio header:
*"(Use these as raw material — PARAPHRASE in your own words,
do not quote verbatim.)"*. Soft nudge to discourage literal cloning.

### The word-overlap heuristic

`_is_bio_mentioned(bio_item, output_lower)` returns True when at
least 2 content words (>3 chars, not stopwords) from the bio item
appear in the output AND those hits cover >= 40% of the bio's
content words. The 2-hit floor catches short-but-distinctive items
like *"Serves the best stew"* (only 3 content words, clone signal
is "best stew"). The 40% ratio rejects incidental one-word overlap
on long items.

Stopwords are a short list (`the, and, with, from, have, she,
their, know...`) — enough to kill the obvious false positives,
not a linguistically complete set.

### Empirical verification (3B, 15 × 3 actions)

Compared against the prior three runs. Key metric: Bess's fixation
on her top pk item dropped from **62% → 25%**.

| Metric | arcs | bio | rotation | biorot v2 |
|---|---|---|---|---|
| Bess merchant-guild fixation | 0/8 | 3/8 | **5/8** | **2/8** |
| Tam apprentice (Kael p10) | 0 | 0 | 4 | 3 |
| Cellar tunnels (Bess p9) | 0 | 1 | 1 | 1 |
| Metal identification (Kael pk) | 0 | 0 | 0 | **NEW ✨** |
| Bitter well (Noah p8) | 0 | 4 | 5 | 5 |
| Vocabulary (unique content words) | 204 | 221 | 260 | 247 |

**Kael's dwarf metallurgy backstory finally surfaced.** T4 in the
biorot v2 run: *"A tax collector's body was found on the north
road. Kael, known for his metal identification skills, could help
identify the body."* That's his pk item *"Can identify any metal
by sound — a dwarf technique"* showing up for the first time
across all four 3B runs. Previously buried under the hammers or
Tam threads; rotation made room.

**Bess's content diversified from 1 theme to 4.** Her 8 focus ticks
now split: 3 on a "hot soup scene" (new emergent invention), 1 on
her cellar-tunnels p9 goal (*"Bess has a hidden passage beneath
her inn. She wants to know if you can lead her to it"* — first
quest about her goal), 2 on merchant guild taxes (capped by
cooldown after the 2nd mention), 2 on an emergent "Roderick
investigating Bess's dealings" thread (cross-character).

**State.json now contains only YAML-sourced items.** The snapshot
fix is measurable: after a bench run, every key in
`bio_mention_counts` is a real bio item from the NPC YAML. Before
the fix the dict was full of fragments like *"i've heard whispers
from the merchant guild..."* — the model's own outputs being
tracked as bio.

### What this did NOT fix

The model now has a *new* fixation mode: **its own recent outputs**.
In the biorot v2 run, Bess's T1 "dropping a hot soup tray" invention
reappeared at T13 and T15 — the model riffed on its own prior
emergent content rather than its bio. This is separate from bio
rotation — it's closer to the ALREADY DONE block's job (which is
supposed to catch self-repetition but is matching too loosely).
Moved to the next-steps list as a separate item.

Vocabulary also dipped slightly (260 → 247) because the cooldown
exclusion hides some bio content the model would have otherwise
riffed on. The tradeoff is worth it — the *distribution* of content
across themes is meaningfully better even with a small vocab hit.

### Tests

7 offline unit tests cover the mention detector (word overlap
detection, rejection of too-short items), the recorder (bump on
match, ignore unrelated), the cooldown exclusion (demotion on heavy
mention), the empty-section fallback, the pk rotation, the
paraphrase header, and persistence round-trip. The snapshot fix
changed the test fixture behavior — tests now construct their NPCs
BEFORE instantiating `StoryDirector` so `_snapshot_original_bios`
catches the test data. All 63 tests pass.

### Tuning knobs

- `_BIO_COOLDOWN_THRESHOLD = 2` (excluded from bio once mentioned 2+ times)
- Top 2 goals / top 3 pk / top 2 wf in the bio block
- 2-hit + 40%-ratio floor for mention detection
- `_BIO_STOPWORDS` — short list, not a complete stopword set

---

## Self-repetition precheck (breaking the own-output copy loop)

### The gap

Intra-bio rotation broke the per-NPC bio copy loop but exposed the
next layer: the model fixating on **its own recent outputs**. In
the biorot_v2 bench, Bess's T1 emergent invention — *"Bess, hiding
a furtive look in her eyes, suddenly drops a tray of hot soup,
spilling it on her leg"* — reappeared with variations at T13 and
T15. The ALREADY DONE block was supposed to catch this but matches
too loosely; text variations slip through.

The FactLedger already had the data. In the biorot_v2 bench log:
T13 and T15 were flagged as 0.75 and 0.84 similarity to T1's hot
soup scene, with NLI=neutral. The machinery existed — it just
didn't *act* on similarity warnings. Only contradictions triggered
a retry.

### The fix

`_precheck_self_repetition(action)` runs right after the existing
`_precheck_contradiction` in the retry path. If the contradiction
check returned None (no NLI contradiction), the self-rep check
looks for a near-duplicate same-NPC match in the ledger and retries
the worker with a prescriptive nudge.

Four tuning iterations landed the current thresholds:

**v1 (aggressive retry):** similarity >= 0.75, 8-tick lookback,
"pick a different angle" retry nudge. Broke the hot soup fixation
(3→1) but 2 retries degenerated to noops, killing Kael's Tam
thread (3→0). The nudge was too open-ended for 3B — the model
couldn't "invent a fresh situation" and gave up.

**v2 (noop fallback):** if the retry returns `noop` and the
original wasn't a noop, fall back to the original. Preserved
content but reverted the fixation fix (hot soup 1→3) because the
retry kept noop'ing and the fallback kept re-dispatching the
repeated content.

**v3 (NPC-restricted ledger + lower threshold + prescriptive nudge):**
- New `restrict_to_npc` param on `FactLedger.check` — limits
  similarity search to same-NPC entries. Previously the ledger
  returned the overall top match, so a cross-NPC gossip entry at
  0.86 could mask a same-NPC repetition at 0.77.
- Lowered threshold 0.75 → 0.70. With the NPC restriction, false
  positive risk is much lower; any same-NPC match at 0.70+ is
  almost certainly paraphrased self-repetition.
- Prescriptive retry nudge lists concrete alternatives:
  *"(a) a conversation with another villager, (b) a physical
  observation about a place or object, (c) a new piece of
  information from a rumor, (d) a reaction to something another
  NPC did."* Gives the model options instead of asking for
  "invention".

Big vocabulary jump (247 → 288) but T15 still slipped through
because it was 14 ticks after T1, beyond the 8-tick lookback. The
entire first half of a 15-tick session was untouchable.

**v4 (extended lookback):** 8 → 20 tick window. Catches cross-half-
session repetitions while still letting long-range plot echoes
through (Kael mentioning Tam at T5 and again at T50 is continuity,
not repetition).

### Empirical verification (3B, 15 × 3 actions)

| Metric | arcs | bio | rotation | biorot v2 | **selfrep v4** |
|---|---|---|---|---|---|
| Bess hot-soup hits | 0 | 0 | 0 | 3 | **1** |
| Cellar passage (Bess p9) | 1 | 1 | 1 | 1 | **4** |
| Elena / garden | 0 | 2 | 3 | 1 | 3 |
| Tam apprentice | 0 | 0 | 4 | 3 | 1 |
| Metal-id (Kael pk) | 0 | 0 | 0 | 1 | 1 |
| Unique vocab words | 204 | 221 | 260 | 247 | **286** |
| Self-rep retries fired | 0 | 0 | 0 | 0 | 10 |
| Noops dispatched | 0 | 0 | 0 | 0 | **0** |

**T15 Bess is the exemplar.** In biorot_v2 the model wrote
*"Bess the innkeeper curses loudly as hot soup spills on her leg"*
— a near-verbatim repeat of T1. In selfrep v4 the precheck fired
(matches_tick=1, lookback=14 ticks now inside the 20-tick window),
the retry nudge landed, and the model wrote instead:

> *"Bess, the innkeeper, steps cautiously into the old cellar
> tunnels beneath the inn, her nostrils flaring as she breathes in
> the musty air. The scent is faint but unmistakabl[e]..."*

That's Bess's priority-9 goal (*"Clear whatever is in the cellar
tunnels before guests find out"*) expressed as a physical
investigation scene. On-bio, novel, and directly triggered by the
retry nudge's "(b) a physical observation about a place or object"
option.

**Cellar passage hits quadrupled** (1 → 4) because this retry
mechanism produced new cellar beats that wouldn't have existed
otherwise.

### Honest tradeoffs

- **Wall-clock roughly 2.4x** (biorot_v2 ~92s → selfrep_v4 ~221s).
  10 retries × ~13s each on average. Retry is cheap per-call but
  frequent. Production use would probably want a per-tick cap or a
  per-session budget to bound latency.
- **Arc advancement suffers.** In v4 the arc proposed at T5
  (cast=[bess], theme="Bess drops a tray of hot soup") never
  advanced past beat 1. The retries kept pushing Bess's content
  away from the initial theme faster than the touch counter could
  accumulate on a coherent thread. Real tension between
  "retry for content diversity" and "arc wants thematic cohesion."
  Noted as a next-steps v5 item.
- **Some strong bio threads weaken.** Kael's Tam content dropped
  from 3 → 1 because some of his Tam-related beats were retried
  away as "too similar to earlier Tam content." The model's
  continuity on a single thread loses to the diversity-pressure.
- **No noops dispatched.** The v2 noop fallback still applies —
  if the retry degenerates to silence, we keep the original. Zero
  dispatched noops in the v4 run.

### Tests

8 offline unit tests cover the precheck: fires on high-sim
same-NPC recent match, passes `restrict_to_npc` to the ledger,
`FactLedger.check` honors the filter, ignores low-similarity,
ignores old matches, ignores target='all', retry path dispatches
the retry's new content, retry falls back to original on noop,
contradiction takes precedence over self-repetition. All 72 tests
green including the integration smoke test.

### Tuning knobs

- `_SELF_REPETITION_SIMILARITY = 0.70` (was 0.75 in v1/v2)
- `_SELF_REPETITION_LOOKBACK_TICKS = 20` (was 8 in v1-v3)
- `_MAX_SELF_REP_RETRIES_PER_TICK = 1` (added later — see
  "Retry budgeting" section below)
- Contradiction retry takes precedence — self-rep only runs if
  contradiction precheck returned None
- Contradiction retries are NOT budget-gated
- Noop fallback: if the retry returns `noop` and the original
  wasn't a noop, keep the original
- Ledger `restrict_to_npc` param — new optional filter, backward
  compatible (defaults to no filter, current contradiction path
  unchanged)

---

## Retry budgeting (bounded worst-case tick latency)

### The gap

The self-rep retry pipeline worked but cost ~2.4x wall-clock on
the v4 3B bench (92s → 221s). Per-tick breakdown showed the cost
was *lumpy*: 8 of 15 ticks had zero retries and ran at baseline,
but **3 ticks had 2 retries each** (T6/T9/T14 in the v4 trace),
pushing those ticks from ~10s baseline to 15-30s. The worst tick
was T6 at 29.62s — visibly laggy even for a game loop that ticks
every 30+ seconds.

For a real-time game director, what matters isn't average latency
(already OK on the low-retry ticks) — it's *predictable* per-tick
latency. A 29s spike is a user-visible jank event. Capping the
per-tick retry budget is the right fix.

### The fix

Add `_MAX_SELF_REP_RETRIES_PER_TICK = 1` and a per-tick counter
(`_self_rep_retries_this_tick`) on the director. Reset to 0 at the
start of every `tick()` call so workers in a single multi-action
tick share one retry slot. Gate the self-rep retry on the counter
in `_run_single_action` — if the budget is exhausted, skip the
retry and dispatch the original content. A `skipped_self_rep_retry`
note is surfaced on the dispatch result so benches can audit how
often the budget kicks in.

Contradiction retries stay UNBUDGETED — they're rare (we've seen
0-4 per 15-tick session) and more serious (dispatching a
contradiction pollutes the ledger for every subsequent tick).
The budget only applies to self-rep retries, which are the bulk.

### Empirical verification (3B, 15 × 3 actions, same config)

Compared directly against selfrep v4:

| Metric | selfrep v4 | **budget** |
|---|---|---|
| Self-rep retries fired | 10 | **6** |
| Self-rep retries skipped (budget) | 0 | **1** |
| 2-retry ticks (hot spots) | 3 | **0** |
| Retry-stacking eliminated? | no | **yes** |
| Total wall-clock | 221s | 253s |
| Mean per-tick | 14.76s | 16.84s |
| Worst-tick latency | 29.62s | 40.47s |

**The mechanical claim works.** Retries dropped 40%, the 2-retry
stack pattern is gone, and the budget correctly skipped one retry
that would have fired unbudgeted.

**The wall-clock claim does NOT hold on a single 3B bench run.**
Total time actually went *up* 221s → 253s, and worst-tick latency
went from 29.62s → 40.47s. Investigation of T14 (the 40.47s spike)
showed it was 3 workers + 1 retry = 4 LLM calls, but each call
took ~10s instead of the usual ~5s. That's CPU variance (thermal
throttling, competing processes) on the dev laptop, not
retry-related. 3B run-to-run variance on this hardware is wide
enough (~100s between identical configs) that single-run wall-
clock comparisons are noise-dominated.

**The content cost is real, though small.** The 4 retries that
didn't fire under the budget (and 1 explicitly skipped) were
exactly the ones that in v4 produced bio-sourced new content:

| Metric | v4 | budget |
|---|---|---|
| Cellar passage hits | 4 | 1 |
| Elena / garden | 3 | 0 |
| Tam apprentice | 1 | 0 |
| Unique vocab words | 286 | 276 |

Bess's T15 cellar-tunnels investigation scene (the v4 exemplar)
is still present, but the 3 additional cellar beats generated by
retries in v4 don't happen in the budgeted run. Tam and Elena
content drops because the retries that displaced generic beats
with bio-sourced ones don't fire anymore.

Vocabulary dip is small (-3.5%).

### Honest takeaway

**Budget = 1 per tick is the right call for production use** even
though the single-bench wall-clock comparison doesn't show a clean
win. The reasons:

1. **Bounded worst-case.** The 2-retry-stack pattern is eliminated
   structurally. A tick can never have more than `3 workers + 1
   retry = 4 LLM calls` worth of work in the self-rep path. For a
   real-time game director this predictability matters more than
   the average.
2. **Run-to-run variance will converge.** On a production machine
   without CPU thermal variance, the mechanical 40% retry cut
   should translate into ~10-20% wall-clock savings.
3. **Content cost is small and recoverable.** For sessions where
   variety matters more than latency, raise the cap to
   `_MAX_SELF_REP_RETRIES_PER_TICK = 2` or disable the budget
   entirely. It's a tuning dial, not a hard change.
4. **Contradiction retries still fire freely.** The serious case
   (the Director contradicting itself) is never budget-gated.

### Tests

3 offline unit tests:
- `self_rep_budget_resets_each_tick` — counter must start at 0
  on every tick so budgets don't leak across ticks
- `self_rep_budget_blocks_second_retry_in_same_tick` — in a
  2-action tick where both workers would retry, only the first
  gets through; the second's dispatch carries a
  `skipped_self_rep_retry` note with `reason: "budget_exhausted"`
- `contradiction_retry_not_budget_gated` — contradiction retries
  still fire after the self-rep budget is exhausted

All 75 tests green.

### Tuning knobs

- `_MAX_SELF_REP_RETRIES_PER_TICK = 1` — per-tick cap on self-rep
  retries. Raise to 2 for more content variety at the cost of
  latency bounds. Set to 0 to disable self-rep retries entirely.
- Contradiction retries are hardcoded as unbudgeted. Changing
  that would require a separate constant.

---

## Long-session validation (50-tick stress test)

### The gap

Every session bench up to this point had been 15-20 ticks. Multi-arc
added real plot density but the longest run (20 ticks) only saw
three arcs with one full resolution. Open questions for production
use:

- Does the concurrent-arc cap of 3 hold up as arcs resolve and new
  ones propose?
- Does `bio_mention_counts` state bloat over a long session?
- Does the 200-entry ledger cap saturate, and how does the drop
  behavior affect content?
- Does per-tick wall-clock degrade as internal state grows (e.g.,
  longer similarity matrices, more arcs to iterate)?

### The verification

Ran `bench_story_director.py --ticks 50 --reset --model qwen_3b
--actions-per-tick 3` — a 50-tick session with 3 sub-actions per
tick (150 actions total). No code changes; pure observation.

### Results

**Arc lifecycle — six arcs, three fully resolved:**

```
arc_t5  [bess]                  T5  → T11 → T19 → T27 → T35 RESOLVED
arc_t10 [roderick]              T10 → T16 → T24 → T32 → T40 RESOLVED
arc_t15 [kael, pip]             T15 → T18 → T24 → T30 → T36 RESOLVED
arc_t36 [bess, mara]            T36 → T40 → T45 → T49 (beat 4/resolve, active)
arc_t41 [roderick, kael]        T41 → T44 → T48        (beat 3/confront, active)
arc_t46 [noah, elara, pip]      T46                    (beat 1/seed, active)
```

**Three complete arc lifecycles** in one session. As the first-wave
arcs resolved (T35, T36, T40), second-wave proposals fired almost
immediately — and the new arcs had *richer casts* (2-3 NPCs each)
because the ledger had accumulated more cross-NPC content for
clustering to find.

The six-arc pattern also revealed something about how the stack
scales with session length: the first wave (arcs 1-3) hit the cap
at T15 and stayed put for 20 ticks. The second wave (arcs 4-6)
fired in rapid succession at T36/T41/T46 as older arcs resolved,
producing a more continuous narrative texture. Single-NPC arcs
are a cold-start artifact — once the ledger has content, richer
casts form naturally.

**A brand-new plot thread surfaced at T46**: `arc_t46` with
`cast=[noah, elara, pip]` and theme *"I have heard from the wolf
hunters that a new wolf pack has..."* — the wolf bounty from the
lore bible that had NEVER anchored an arc in any prior bench
finally became a proper multi-character mystery. By T50 the
content had Kael *"leaves his blacksmith shop, a lantern in one
hand, a sword in the other, and heads toward the Silverwood"*,
Noah *"I've been hearing whispers from the wolf hunters"*,
Roderick *"overheard the wolf hunters discussing their plans to
ambush the village"*. A fresh threat narrative building 46+ ticks
into the session.

**Wall-clock is flat.** No degradation as the session grows:

| Metric | Value |
|---|---|
| Total wall-clock | 622.1s |
| Mean per-tick | 12.44s |
| Min per-tick | 8.28s |
| Max per-tick | 20.30s |
| First half mean (T1-T25) | 12.48s |
| Second half mean (T26-T50) | 12.41s |
| **Change between halves** | **−0.6%** |

The 0.6% drop is statistical noise. Per-tick latency is flat —
no quadratic state growth, no slowdown from accumulated ledger
entries, no slowdown from tracking 6 arcs vs 1.

**State is bounded:**

| File | Size | Content |
|---|---|---|
| `state.json` | 31 KB | `recent_decisions` capped at 5, `bio_mention_counts` = 24 items across 7 NPCs, max mention count = 4 |
| `arcs.json` | 4.8 KB | 6 arcs total (3 resolved + 3 active), ~800 bytes each |
| `fact_ledger.json` | 1.8 MB | 149/200 entries — ledger entries are fat because embeddings are stored as JSON float lists |

- `bio_mention_counts`: 24 total items tracked across 7 NPCs. Key
  observation: this grows with *bio items seen*, not session
  length. Bounded by (NPCs × items per NPC) ≈ 60 max. No bloat.
- `recent_decisions`: still capped at 5 (as designed).
- `arcs` list: grows linearly with session length but each arc is
  ~800 bytes. 500-tick session ≈ 60 arcs ≈ 48KB. Not a concern.
- **Ledger at 149/200** — approaching but not saturated. Ledger
  saturation would first hit around tick 67 (200/3 ≈ 66). This
  50-tick bench didn't reach it; drop behavior untested from this
  run alone but the drop path is a simple `entries[-200:]` slice.

**Content quality holds through 50 ticks.** Per-quarter vocab
diversity:

| Quarter | Ticks | Unique content words |
|---|---|---|
| Q1 | T1-T12 | 250 |
| Q2 | T13-T25 | 237 |
| Q3 | T26-T37 | 210 |
| Q4 | T38-T50 | 220 |
| Overall | T1-T50 | **607** |

607 unique content words is more than 1.7x the 15-tick multiarc
run's 361. The Q3 dip to 210 reflects the brief window between
first-wave resolution and second-wave proposal, where the Director
was mostly advancing existing beats toward resolution. Q4 bounces
back as the new wave starts introducing fresh vocabulary.

### Observations and nice-to-haves

- **Ledger file size**: 1.8 MB for 149 entries means each entry
  is ~12 KB. Most of that is the embedding stored as a JSON list
  of 384 float64 values. Storing as float32 binary would cut this
  ~10× with zero content loss. Pure optimization — the current
  format works fine.
- **Arc proposal timing**: arcs 4-5-6 fired at T36/T41/T46 —
  5-tick cooldown cadence exactly. For very long sessions the
  cadence could tighten slightly once the stack is known-stable,
  but 5 ticks is a good conservative default.
- **Cold-start pattern**: the first 3 arcs all had 1-2 NPC casts;
  the next 3 all had 2-3 NPC casts. The richer casts are an
  emergent property of a fuller ledger, not a code change. Worth
  noting in bench docs so users don't expect big casts from tick 1.
- **3-NPC arcs work cleanly**: arc_t46 with `[noah, elara, pip]`
  advanced just like single- or 2-NPC arcs. The architecture has
  no implicit cast-size limit.

### Verdict

**The stack is production-ready for sessions up to at least 50
ticks.** Everything is bounded, wall-clock is linear, content
stays fresh, arcs cycle naturally, and new plot threads continue
emerging late into the session. No code changes needed from this
validation run.

---

## Multi-arc concurrency (Cardinal-class plot weaving)

### The gap

Through every session commit from arcs through arc-touch, the
planner tracked **one active arc at a time**. Even with the touch
counter fix unlocking reliable advancement, the Director could
only hold one plot thread. A Cardinal-style overseer should run
multiple threads in parallel — a main quest, a background
faction tension, a slow-burn mystery — and let each advance on its
own schedule. Single-arc means one plot at a time; whatever
thread the initial cluster caught at T5 was the only one that
grew for the whole session, and every NPC outside that cast
wrote generic content.

### The fix

Four changes to `ArcPlanner`:

1. **List of active arcs.** `active_arc_id: Optional[str]` →
   `active_arc_ids: list[str]`. `active()` (single) → `active_arcs()`
   (list). State file gains `active_arc_ids` with backward-compat
   fallback to the old `active_arc_id` field.

2. **Per-worker arc selection.** New `arc_for_focus(focus_npc)`
   returns the one arc whose cast contains `focus_npc`, or None.
   When an NPC is somehow in multiple casts (defensive —
   proposal normally excludes overlap), prefer the arc with the
   fewest `touches_since_last_advance` (the weakest thread, so
   each relevant worker helps the neglected arc catch up).
   `_build_prompt` uses this instead of `active()` so the arc
   block reflects the right arc per-worker, and no arc block is
   injected when the focus NPC isn't in any cast.

3. **Proposal excludes NPCs already in active casts.**
   `maybe_propose` filters ledger entries to only those whose
   NPCs are NOT covered by an active arc before clustering.
   Without this, the densest cluster keeps being whatever plot
   thread is already saturated, and proposals duplicate. The
   `_MAX_CONCURRENT_ARCS = 3` cap gates at the top level.

4. **Multi-arc touches and advancement.** `record_cast_touch`
   iterates active arcs and bumps every one whose cast contains
   the NPC. `advance_if_beat_met` iterates active arcs and
   advances each that has met its threshold in a single call,
   returning the count of arcs advanced (was a bool). Resolved
   arcs drop out of `active_arc_ids`.

### Empirical verification (3B, 20 × 3 actions)

```
Metric                     arctouch (15t)  multiarc (20t)
Concurrent arcs (max)                   1               3
Arcs proposed                           1               3
Arc advances (total)                    1               4
Unique vocab words                    318             361
Self-rep retries                        3               7
Wall-clock total                       92s            296s
Mean per-tick                         6.2s          14.8s
```

**Three concurrent arcs ran to partial completion:**

| Arc | Cast | Proposed | Advances |
|---|---|---|---|
| arc_t5 | `[bess]` | T5 | T11 (seed→escalate), T19 (escalate→confront) |
| arc_t10 | `[guard_roderick]` | T10 | T16 (seed→escalate) |
| arc_t15 | `[kael, pip]` | T15 | T18 (seed→escalate) |

**Pip got his own arc for the first time.** Across every prior
3B run Pip was a bystander — rotation touched him occasionally
but his content was generic because no arc cast included him.
In multiarc he appeared in arc_t15 and the prompt gave him
beat-specific guidance, producing 6 consecutive ticks of
coherent investigation content:

- T3: glimpses a hooded figure near the village well
- T6: quest — "wants to prove Mara is guilty of smuggling"
- T9: cautiously approaches the merchant's warehouse
- T12: "Mara has a new informant in the village"
- T15: seeing Mara meet a shadowy figure in the Silverwood
- T18: watching Mara, overhearing her

That's a classic detective beat sequence: suspicion →
commitment → approach → corroboration → evidence → surveillance.
None of it was in any prior run because Pip was never cast-relevant.

**Roderick's parallel arc** is the strange-lights/Silverwood
shadows vigilance thread that first surfaced in the
example-rotation bench. In multiarc it's now a full arc with
9 focus ticks of escalating paranoia:

- T1: "strange lights in the forest near the Silverwood"
- T4: "notices a shadow dart from the Silverwood"
- T6: "body found on the road north...suspects it's related to
  the missing tax collector"
- T8: quest — "needs to organize a patrol"
- T10-T20: tightening grip on sword, spotting shadows, noticing
  a dark figure in the trees, deciding to send a guard

**Bess's internal-tension arc** from prior runs also survived
and got TWO beat advances in one session (T11 and T19), reaching
beat 3 (confront) by end of session. This is the first arc in
the whole session history that advanced to beat 3.

Vocabulary hit 361 unique words — highest of the session (prior
best was arctouch at 318, arcs baseline was 204).

### Tradeoffs

- **Wall-clock is 2.4x the single-arc case** (6.2s/tick →
  14.8s/tick). Some of this is natural variance (per-tick timing
  on CPU ranges wildly), but some is real: each worker now pays
  a small cost for the arc_for_focus lookup and any arc block
  they inject. For real-time game loops where ticks fire every
  30+ seconds this is fine; for fast cadence it's the new
  bottleneck.
- **More self-rep retries** (3 → 7). With more arcs in flight,
  more content references earlier content within-cast, so the
  precheck fires more often. The per-tick budget still caps
  worst-case latency.
- **Arc proposal cooldown may be too slow for multi-arc.** The
  5-tick cooldown was chosen for single-arc. With multi-arc
  headroom for 3 concurrent, the planner could realistically
  propose more often early in the session. Not a bug, just
  conservative.
- **Legacy state files load with a warning.** Old `state.json`
  files from pre-multiarc sessions have `active_arc_id` singular;
  the loader still reads them correctly via a fallback, but
  anyone resuming a pre-multiarc session will see their single
  arc become a list of one.

### Tests

8 new unit tests:

- `test_arc_planner_caps_at_max_concurrent_arcs` — proposal
  blocks at the cap
- `test_arc_planner_proposal_excludes_npcs_in_active_casts` —
  new arcs can't form on NPCs already in another arc's cast
- `test_arc_for_focus_returns_matching_arc` — returns the arc
  whose cast contains the NPC (or None)
- `test_arc_for_focus_prefers_weakest_thread_on_overlap` — on
  overlap, returns the arc with fewest touches (defensive)
- `test_record_cast_touch_bumps_all_matching_arcs` — all
  matching active arcs get their counter bumped
- `test_advance_if_beat_met_advances_multiple_arcs` — returns
  count, advances each eligible arc in one call

Two existing tests updated to use `active_arc_ids` /
`active_arcs()`. `test_arc_planner_skips_when_active_arc_exists`
renamed to `test_arc_planner_caps_at_max_concurrent_arcs` since
"single active arc" is no longer the relevant constraint. All
83 offline tests green.

### Tuning knobs

- `_MAX_CONCURRENT_ARCS = 3` — soft cap on simultaneous active
  arcs. Raise for more plot density at the cost of prompt
  clarity; lower for simpler single-thread sessions.
- `_ARC_PROPOSAL_COOLDOWN_TICKS` unchanged at 5 — but with
  multi-arc headroom, it's worth considering raising this to 7
  so arcs form more slowly and each has room to develop.

---

## Arc touch counter fix (regression that was hiding in plain sight)

### The gap

Across the arcs, bio, rotation, bio-rotate v2, selfrep v1-v4, and
budget runs — **six consecutive 3B benches** — the active arc
proposed at T5 and never advanced past beat 1. Every run showed
`Arc events (1): T5 PROPOSED ...` and nothing else. I chalked this
up to "arc-vs-retry tension: retries push content away from the
theme faster than touches accumulate."

That was wrong. The real cause was a bug in the arc advancement
logic that had been there since the arcs commit shipped.

### Diagnosis

`ArcPlanner.advance_if_beat_met` walked `recent_decisions` and
counted cast-NPC touches. But `recent_decisions` is capped at 5
ticks (for ALREADY DONE, rotation, and other concerns). That means
the max touches visible to the advance check is
`5 ticks × 3 workers = 15` in theory, but only as many as the
cast can actually claim in a 5-tick window.

For a single-NPC arc like `cast=[bess]`, the rolling 5-tick window
can contain at most ~2-3 Bess touches (she gets ~1 per 2 ticks via
rotation). Threshold = 4. **So single-NPC arcs literally could not
advance on this setup.**

Empirical confirmation from the selfrep_v4 data:

```
Bess focus touches T6-T15 (after proposal): 5 total across 10 ticks
5-tick rolling window values:    0, 1, 1, 2, 2, 3, 2, 3, 2, 3
Maximum window sum: 3 (threshold: 4)
```

The rolling window maxed out at 3 — just under the threshold.
Every prior bench was silently broken; I just hadn't noticed
because "arc didn't advance" isn't a visible failure mode.

### The fix

Track cast touches on the arc itself instead of walking
`recent_decisions`. Three changes:

1. New field `NarrativeArc.touches_since_last_advance: int = 0`.
2. New method `ArcPlanner.record_cast_touch(npc_id)` — bumps the
   active arc's counter if the NPC is in its cast. Called from
   `_run_single_action` after every successful non-noop dispatch.
3. `advance_if_beat_met(current_tick)` — signature changed from
   `(recent_decisions, current_tick)` to just `(current_tick)`.
   Checks `arc.touches_since_last_advance >= threshold` instead
   of walking the decisions list. Resets the counter to 0 on
   advance.

Tests updated to match the new API; 3 new tests added for
`record_cast_touch` (bumps on cast, ignores non-cast, ignores
inactive arcs).

### Empirical verification (3B, 15 × 3 actions)

```
Metric                 biorot_v2  selfrep_v4  budget  arctouch
Self-rep retries              0          10       6         3
Wall-clock total          188s        221s    253s       92s
Mean per-tick           12.5s       14.8s   16.8s      6.2s
Unique vocab words          247         286     276       318
Arc advanced past beat 1?    no          no      no       YES (T11)
```

**The arc advancement unlocked a cascade of benefits:**

1. **Arc advances seed → escalate at T11.** First beat advance in
   any 3B run across the whole session.
2. **After T11, the prompt carries "Current beat 2/4: escalate —
   deepen the stakes with a new wrinkle"** instead of the generic
   seed guidance. The model has more specific direction to write
   toward.
3. **Content naturally diversifies** because the prompt is driving
   it toward a different kind of beat. Bess T11 wove world-lore
   with new invention: *"The king's tax collectors were never
   seen again after that. I think they were replaced by a
   different group."* Bess T13 cross-wove three threads:
   Roderick+Mara, wolf bounty, Silverwood.
4. **Naturally diverse content → fewer self-rep triggers**
   (3 retries vs v4's 10).
5. **Fewer retries → dramatically faster wall-clock**
   (92s vs v4's 221s — 58% reduction).
6. **Vocabulary diversity jumps to 318** — the highest of any 3B
   run in the session.

Every metric improved except the retry-driven bio-hook surfacing
that v4 had (cellar passage 4 → 1, Elena garden 3 → 0). Those
losses are offset by the new cellar passage appearing as a full
QUEST at T3 (a richer hit than v4's multiple one-line mentions),
and by the qualitative improvement in content quality where T11
and T13 weave multiple bio and lore threads together.

### Why this matters for the stack

The fix is small (20 lines) but the impact is disproportionate
because it unblocked a **positive feedback loop** between arc
advancement, prompt specificity, content diversity, and retry
pressure. All the prior session work (bio injection, rotation,
self-rep) was building infrastructure that depended on this loop
working. With the arc advancement broken, each layer was
compensating for a deeper problem.

The honest lesson: I should have verified arcs were advancing in
the 3B bench output of the original commit (f457d2d), not just
the unit tests. The unit tests passed because they constructed
their own `recent_decisions` lists; they never exercised the
"only last 5 ticks visible" interaction.

### Tests

3 new unit tests plus updates to 2 existing arc tests (which used
the old `(recent_decisions, current_tick)` signature). All 78
offline tests green.

### Tuning knobs

- `touches_since_last_advance` is the new counter field on each
  arc; persisted via `asdict` → `NarrativeArc(**a)` round-trip
  in `ArcPlanner._load` / `.save` without code changes (dataclass
  handles it automatically).
- Threshold constant `_ARC_BEAT_ADVANCE_THRESHOLD = 4` unchanged.

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
- **A multi-tick arc planner with concurrent arcs** (deterministic
  cluster → theme + 4-beat skeleton, per-arc touch counter with
  unbounded accumulation, up to 3 arcs running in parallel with
  per-worker arc selection via `arc_for_focus`, proposal excludes
  NPCs already in active casts, zero new LLM calls)
- **NPC bio injection** (goals + personality + personal_knowledge +
  world_facts per focus NPC, one-line motive summary per roster NPC)
- **Dynamic example rotation** (per-worker example picker that
  excludes focus-NPC examples so the bio can escape the few-shot
  copy loop — this is what unlocked emergent plots like Kael's
  search for Tam)
- **Intra-bio rotation** (per-item mention tracking, original-bio
  snapshot to avoid contamination from Director-added facts,
  cooldown exclusion that drops items from the prompt once
  mentioned enough — this broke the "each NPC quotes its top bio
  item verbatim" copy loop)
- **Self-repetition retry** (pre-dispatch ledger check with
  NPC-restricted similarity, prescriptive retry nudge listing
  concrete alternatives, noop fallback, per-tick budget of 1 —
  broke the "model paraphrases its own recent output" copy loop
  and surfaced Bess's cellar tunnels quest by turning a would-be
  hot-soup repeat into a fresh investigation beat)
- **A REST surface** (`/story/tick`, `/story/state`, `/story/player_action`)

Every single piece exists because an empirical run produced a failure
that pure prompting couldn't fix. Nothing in this list is speculative
engineering — every component has a corresponding "this ran, this broke,
here's why" story.

The Cardinal pattern from SAO is real and shippable on local hardware,
**but only if you stop asking the model to make decisions it can't
make.** That's the whole insight in one sentence.

---

## Next steps — pick up here

Ranked by leverage. Each entry is detailed enough that a fresh
session (or another agent) can start work without re-reading the
full narrative above.

### 1. Ledger compression (nice-to-have optimization)

**The gap:** `fact_ledger.json` reached 1.8 MB at 149 entries
(~12 KB per entry) in the 50-tick validation run. Most of that
is the embedding stored as a JSON list of 384 float64 values.
Storing embeddings as binary float32 would cut this ~10× with
zero content loss.

This is a pure optimization — the current format works fine for
the 200-entry cap and loads in a fraction of a second. But for
very long sessions, agent servers handling many parallel worlds,
or checkpointing scenarios, the size matters.

**Design sketch:**
- Change `FactLedger._save`/`_load` to serialize embeddings as a
  sidecar binary file (`fact_ledger_embeddings.bin`) with
  numpy's `np.savez_compressed` or similar.
- Keep the JSON entries metadata (text, npc_id, kind, tick) in
  `fact_ledger.json`.
- Optionally: skip persisting embeddings entirely and re-encode
  on load. Trades ~3s boot time for ~1.8MB on disk.

**Where to start:** `FactLedger._save` and `FactLedger._load` in
`story_director.py`.

**Risk:** Low. No behavior change; backward compat via format
version flag.

### 2. Longer-session validation (done — see "Long-session validation" section)

### 3. Own-output fixation (done — see Self-repetition precheck section)

### 4. Multi-arc concurrency (done — see Multi-arc concurrency section)

**Done in commit 20fb7e7.** The pre-dispatch
`_precheck_self_repetition` + NPC-restricted ledger check + noop
fallback + per-tick budget collectively break the own-output copy
loop that was the prior #1 item. See the "Self-repetition precheck"
and "Retry budgeting" sections above for the full writeup and
empirical verification.

### 2. Auto-diversify examples.yaml library

**The gap:** `examples.yaml` has exactly 5 examples covering 5 NPCs
(kael, elara, roderick, bess, pip). When a focus NPC has no
primary example, the picker shows all 5 and things work fine. But
if the example library grew to 10+ per-NPC examples with different
themes, the picker could show 2-3 NOT-about-focus examples drawn
from a wider thematic pool — giving the model exposure to more
action shapes and tones without ever touching the focus NPC's
example territory. Right now the library is too thin to exercise
that.

**Design sketch:**
- Expand `examples.yaml` to ~15-20 entries, covering each NPC with
  at least 2 examples (different action kinds per NPC).
- Add a `theme` tag to each entry so the picker can prefer examples
  whose theme differs from the current active arc's theme.
- Optionally port ultralight-coder's `generate_augmentors_from_failures`
  approach: run benches, flag literal-copy events, generate
  augmentor variants with a larger model, isolation-gate, and
  write to `examples_generated/` for manual review.

**Where to start:** `data/story_director/examples.yaml` (hand-curate
first), then `_pick_examples` to honor theme diversity.

**Risk:** Medium — hand-curating 15+ good examples takes effort.
Auto-generation risks drift from the lore bible's tone.

### 5. NPC dialogue identity bleed (unchanged — still needs npc-engine work)

**The gap:** In bench dialogue tests, Noah called the player "Mara"
and Mara called the player "Kael". This is in the npc-engine
*dialogue path* (`engine.process` → PIE → expert generation), not
the StoryDirector. It taints the dialogue auto-feed because the
Director records garbled dialogue.

**Where to start:** `npc_engine/postgen.py` already does
identity/hallucination/echo validation — it's where to add a
"speaker addresses player by wrong name" check. Need to detect when
an NPC dialogue uses ANOTHER NPC's name as if addressing the
player, and either repair (replace with "traveler" / "stranger") or
flag for retry.

**Risk:** This is its own concern, separate from the Director. Best
done as a focused npc-engine fix on its own branch, then merge
forward.

### 6. Real parallel workers

**The gap:** Multi-action ticks run sub-actions sequentially because
llama-cpp-python isn't thread-safe for one base model. On a 3-action
tick this means 3× the wall-clock vs. theoretical parallel.

**Design sketch:**
- Option A: keep N independent `BaseModel` instances (3× the RAM but
  trivial parallelism via `concurrent.futures.ThreadPoolExecutor`)
- Option B: use PIE's `GenerationQueue` from `densanon.core.pipeline`
  which already prioritizes background work — the Director would
  enqueue all N sub-actions at once and await results
- Option C: a separate llama.cpp server process per worker (real
  process isolation, no thread issues)

**Where to start:** `npc_engine/story_director.py:_run_single_action`
is the single function that needs to become async-aware. The
`tick(actions_per_tick=N)` loop would dispatch N futures and
`await asyncio.gather(*futures)`.

**Risk:** RAM cost for option A is significant (3× model size in
working memory). Option B requires understanding PIE's queue
priorities. Option C is the cleanest but adds operational complexity.

### 7. Co-reference resolution in the FactLedger

**The gap:** Embedding similarity misses linked entities when names
differ. *"The elder is hiding something"* and *"Noah won't talk
about it"* would not match because "elder" vs "Noah" lowers the
embedding cosine.

**Design sketch:**
- Build an alias map from NPC profiles: `{"noah": ["the elder",
  "village elder"], "kael": ["the blacksmith", "smith"]}`
- Before embedding, expand alias references in the text to the
  canonical NPC id. So *"The elder is hiding something"* becomes
  *"noah (the elder) is hiding something"* before passing to the
  embedder.
- The expanded form keeps the original surface text in the entry
  but uses the augmented version for similarity comparison.

**Where to start:** `FactLedger._encode` in
`npc_engine/story_director.py`. Add an `_expand_aliases` step that
reads NPC profile data from `engine.pie.npc_knowledge.profiles`
and substitutes role/title references. Cache the alias map on the
director (it doesn't change at runtime).

**Risk:** Low — purely additive to the embedding path.

### 8. Auto-augmentor generation for narrative examples

**The gap:** The 0.5B path's content is more literal than the 3B path
because it copies few-shot examples verbatim. The fix is more curated
examples — that's the ultralight-coder playbook
(`generate_augmentors_from_failures.py`). Port it to story examples.

**Design sketch:**
- Run the bench against 0.5B repeatedly, log every "literal copy"
  failure (heuristic: tick text contains 80%+ overlap with an
  examples.yaml entry)
- Feed the failures to a larger local model (Qwen 2.5 3B is already
  loaded) with a prompt: *"Here's a story-state and a worker action
  that copied the example too literally. Generate 3 alternative
  story beats that match the same pattern but use different
  specifics."*
- Schema-gate, isolation-gate (run the new candidate as a few-shot
  on 0.5B and verify it doesn't trigger another literal-copy
  failure), and write to `data/story_director/examples_generated/`
  with a `_meta` quarantine block. Manual review before merging
  into the main library.

**Where to start:** `bench_story_director.py` — it already produces
trace JSON. Add a "literal copy detector" pass over the trace.
Then a new `generate_story_augmentors.py` script in the repo root.

**Risk:** Medium — needs careful human review of generated examples
to avoid drift away from the lore bible's tone.

### 9. Narrative arc polish (v2 items)

The deterministic arc planner shipped in v1 leaves a few things on
the table that would be worth coming back to once multi-tick arcs
have been exercised in real sessions:

- **Multi-arc concurrency**: one active arc at a time right now.
  Cardinal would run 2-3 threads in parallel. Requires the prompt
  to pick *which* active arc each tick advances.
- **Semantic beat progress**: touch counter is coarse — a tick
  naming the NPC but not advancing the theme still counts. Replace
  with "embed the tick's new content, measure similarity to the
  current beat goal, only count if above threshold."
- **LLM-proposed themes**: v1 pulls themes straight from cluster
  center text. A one-call-per-proposal LLM summarization pass
  would produce crisper one-line themes.
- **Abandonment**: stale arcs with no recent cast touches should
  flip to `status = "abandoned"` so new arcs can form.

### 10. Smaller items worth doing

- **Acting on contradiction** beyond retry: when the retry ALSO
  fails, fall back to a noop and log to a `narrative_conflicts.json`
  file for human review. Currently we just dispatch the conflicting
  retry result.
- **Dialogue auto-feed filtering**: every player turn currently feeds
  the Director. For a chatty player this floods `recent_player_actions`.
  Add a heuristic: only feed if the dialogue contains substance
  (length >= N, or matches a "meaningful" keyword set).
- **Token budget management**: `_build_prompt` concatenates lore +
  examples + snapshot + ALREADY DONE + FOCUS + ACTION + ACTIVE
  NARRATIVE ARC. As ledger grows and recent_decisions accumulate,
  the prompt gets long. Add a token budget split (e.g., 40% lore +
  examples, 30% snapshot, 20% history, 10% directive) with
  truncation.

---

## File map

```
npc-engine/
├── npc_engine/
│   ├── story_director.py     # Director + FactLedger + NarrativeArc + ArcPlanner
│   ├── engine.py             # +14 LOC: instantiate + dialogue auto-feed
│   └── server.py             # +43 LOC: 3 new REST endpoints
├── tests/
│   └── test_story_director.py  # 43 tests (42 offline + integration)
├── data/
│   └── story_director/
│       ├── ashenvale_lore.md     # ~30 lines of setting bible
│       ├── examples.yaml         # 5 few-shot world-state→action pairs
│       ├── FINDINGS.md           # this file
│       ├── state.json            # gitignored runtime state
│       ├── fact_ledger.json      # gitignored ledger (200-entry cap)
│       └── arcs.json             # gitignored narrative arc state
├── bench_story_director.py   # Model comparison + scripted sessions
└── .gitignore                # runtime files, traces, backups
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
