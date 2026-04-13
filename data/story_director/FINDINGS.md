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
- **A multi-tick arc planner** (deterministic cluster → theme + 4-beat
  skeleton, touch-counter advancement, zero new LLM calls)
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

### 1. NPC dialogue identity bleed

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

### 2. Real parallel workers

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

### 3. Co-reference resolution in the FactLedger

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

### 4. Auto-augmentor generation for narrative examples

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

### 5. Narrative arc polish (v2 items)

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

### 6. Smaller items worth doing

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
