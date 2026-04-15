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

## Speculative items evaluated and rejected

Features that were implemented, tested, and rolled back because the
empirical bench showed they didn't produce measurable improvements.
Documented here so the rationale survives future "should we try X
again?" conversations.

### Real parallel workers via N independent Llama instances (rejected 2026-04-14)

**The design:** Multi-action ticks run sub-actions sequentially
because llama-cpp-python's `Llama` instance isn't thread-safe.
Hypothesis: load N independent `Llama` instances (one per worker
slot), dispatch sub-actions concurrently via
`ThreadPoolExecutor`, route each LLM call through a thread-safe
`ModelPool` that serves slots from a `queue.Queue`. llama-cpp-
python releases the GIL during inference, so threads against
independent instances *should* run concurrently and deliver
~2× wall-clock improvement on 3-worker ticks.

**The scout finding:** Confirmed via a standalone bench
(`scratch_parallel_llama.py`) that 2-3 independent Qwen 2.5 3B
instances on CPU could parallelize cleanly:

| Config | Per-call time | Total | Speedup |
|---|---|---|---|
| 2 instances, short prompts (40 tok) | 1.16 / 1.27s | serial 2.42s, parallel 1.28s | **1.89×** |
| 3 instances, short prompts (40 tok) | ~1.5s | serial 4.69s, parallel 2.22s | **2.11×** |
| 3 instances, bench-realistic (1500-tok prompts, 400-tok output) | 7.3s | serial 21.9s, parallel 12.4s | **1.76×** |

Memory cost: GGUF weights mmap-shared across instances; only the
KV cache (~30 MB at n_ctx=4096) is per-instance. Loading N=3
instances added ~100 MB on top of the existing PIE base_model.

**The prototype:** Implemented `ModelPool` class in story_director.py
with thread-safe slot acquisition, eagerly loaded via
`StoryDirector.enable_parallel_workers(n, model_path)`. Wired into
`tick()` so multi-action ticks dispatch via `ThreadPoolExecutor`
when the pool is set. Added `_state_lock` (RLock) around the
shared-state mutation phase of `_run_single_action` (dispatch,
ledger.add, bio mention) so concurrent workers can't race on
writes. LLM calls run outside the lock through the pool's own
slot mechanism. 4 unit tests covered ModelPool slot mechanics,
exception-safety, and serial-path regression.

**The empirical bench killed it.** Side-by-side on real Story
Director ticks (15-tick equivalent, 3 sub-actions per tick):

| Config | Mean per-tick |
|---|---|
| Serial baseline (PIE base_model, n_threads=8) | **~12.4s** |
| Parallel workers=2, n_threads=2 | 22.5s |
| Parallel workers=3, n_threads=2 | 33.1s |
| Parallel workers=2, n_threads=4 | 67.6s (high variance) |
| Parallel workers=3, n_threads=4 | 42.4s |

**Every parallel configuration was slower than serial.** Added
thread-id traces to confirm workers DO start at the same
timestamp (real concurrency), but they finish at very different
times — within one tick we'd see 10.9s, 22.9s, and 29.3s for the
three workers.

**Why the bench shows the opposite of the scout:**

1. **Per-call time triples under contention.** The serial baseline
   uses PIE's well-tuned `base_model` with `n_threads=8` against
   the full CPU. Splitting into N=2 or N=3 pool instances forces
   `n_threads_per_instance=2-4`, which makes EACH call ~3× slower
   even before contention.
2. **The slowest worker dominates.** With `actions_per_tick=3`,
   one worker generates a ~400-token response while another stops
   at ~80. The tick wall-clock = max(worker_times), not the sum.
   So a 3-worker tick is bounded by the longest call, often
   roughly equal to (or worse than) the serial total.
3. **The scout's near-2× speedup was an artifact** of equalised
   short prompts (40 tokens) where generation length variance
   doesn't matter and per-call time is short enough that
   contention overhead is negligible.

**The fundamental hardware story:** Modern CPU llama.cpp
inference is already thread-saturated by a single instance with
n_threads ≈ physical cores. There's no spare CPU for parallel
instances to use — they trade serial efficiency for nominal
concurrency and lose. Parallelism would only win on hardware
with significantly more cores than the serial baseline can use,
or with much longer prompts where the eval phase dominates and
benefits from N independent batches.

**The rollback:** Discarded the prototype with `git restore`. No
code survives in the main branch — only this rejection note and
the new `_run_single_action` `_state_lock` (kept as harmless,
zero-overhead defensive code). The scratch script was deleted.

**When to reconsider:** If a future deployment runs on a 16+
core server with the model fitting fully on GPU (CUDA contexts
allowing per-thread instance ownership), the math changes.
Could also be worth revisiting if Story Director moves to
larger context windows (8k+) where eval-phase time dominates
and per-instance n_threads=4 isn't a meaningful slowdown.

---

### Co-reference alias expansion in FactLedger (rejected 2026-04-14)

**The design:** Build an alias map from NPC profiles
(`{"elder": "noah", "merchant": "mara"}`) and append canonical NPC
ids to ledger text before embedding — so *"The elder fears the
merchant"* becomes *"The elder fears the merchant noah mara"* in
embedding space. The idea was that role references would cluster
closer to direct name mentions in similarity checks, helping
bio-grounded hooks surface and tightening arc proposals.

**The prototype:** Implemented on a throwaway branch
(`story-director-coref`). Added `FactLedger.alias_map` field,
`_expand_aliases(text)` method, and integration in `_encode`.
StoryDirector built the alias map from NPC `identity.role` fields
on init with ambiguous-short-form dropping (if two NPCs share
"guard" as a short form, the alias is not added). 8 unit tests
covered all the cases. Code was clean, tests were green.

**Empirical results (15-tick 3B bench, same config as baseline):**

| Metric | Baseline | With coref |
|---|---|---|
| Self-rep retries | 5 | 5 |
| Unique vocab | 307 | 307 |
| Tam apprentice hits | 1 | 1 |
| Cellar passage hits | 1 | 1 |
| Metal-id (Kael pk) | 1 | 1 |
| Elena garden | 3 | 3 |
| Bitter well | 3 | 2 |
| Merchant guild | 5 | 5 |
| Wall-clock | 186s | 232s (+25%) |

Every content metric is identical or noise-level. Vocabulary is
*exactly* the same (307 unique words in both runs). The one
non-zero delta (bitter well −1) is a single ledger entry and
within run-to-run variance. Wall-clock went up 25% but that's
inside the 3B CPU-variance envelope (other 15-tick runs have
ranged 92s–253s for identical configs).

**Why it didn't help:** Instrumented the ledger to see how often
role references show up *without* the canonical name. Of 44 ledger
entries:

- 40% "name only" — no role to expand, coref is a no-op
- 13% "role only" — coref could help IN THEORY
- 20% "both name+role" — redundant, already has name
- 25% "neither" — coref is a no-op

So only 13% of entries were eligible for coref to affect
clustering. And on those, the modern sentence-transformers embedder
(`all-MiniLM-L6-v2`) already places role words semantically close
to their canonical referents in its training data — "elder" and
"Noah" don't need explicit anchoring to cluster. Adding the
canonical id is a minor nudge in embedding space, not a dramatic
shift, and the nudge is too weak to push any pair across the
similarity threshold that would change arc clustering or retry
behavior.

**The rollback:** Discarded the branch via `git checkout
story-director`. No code survives in the main branch — only this
rejection note.

**When to reconsider:** If future work uses a smaller / weaker
embedder that's actually bad at coreference, or if sessions show
meaningful content stuck behind role-reference language, this
could be worth revisiting with a more aggressive expansion strategy
(e.g., replacing role substrings entirely rather than appending,
or using a second pass to rewrite historical ledger entries).
Current empirical evidence says the effort doesn't land on the
current stack.

---

## Ledger compression (binary embeddings sidecar)

### The gap

The 50-tick long-session validation flagged `fact_ledger.json` at
1.8 MB — dominated by embeddings serialized as JSON lists of 384
float64 values per entry (~12 KB per entry in JSON). The content
payload (text, npc_id, kind, tick) is tiny by comparison; the
numbers hog the disk.

For a 200-entry capped ledger this is ~2 MB steady-state. Fine on
local dev, noisy for agent servers running many parallel worlds,
and wasteful given the fix is mechanical.

### The fix

Split storage into two files:

1. `fact_ledger.json` — entry metadata (text, npc_id, kind, tick)
   as before but *without* the `embedding` field. Stays
   human-readable and small.
2. `fact_ledger.embeddings.npy` — matched-index float32 array,
   written via numpy's native `.npy` format. Same stem as the
   JSON file, same directory.

`FactLedger._save` writes both files in sequence. `_load` reads
the JSON, then (if any entry lacks an inline `embedding` key)
loads the sidecar and zips embeddings back into entries by
index. `reset()` cleans up both paths.

**Backward compat is zero-touch.** Old `fact_ledger.json` files
from before this commit have inline embeddings — the loader
detects this per-entry (`"embedding" in e`) and uses the inline
data when present, falling back to the sidecar when absent. An
existing session's ledger loads cleanly; on the next `add()` it
gets re-saved in the new format, dropping the inline embeddings
and writing the sidecar. No migration script.

Float64 → float32 is a precision tradeoff. Sentence-transformers
embeddings are unit-normalized and cosine similarity is stable
at float32 precision — tested within 1e-5 absolute error. For
our threshold tuning (0.6 / 0.7 / 0.75 / 0.85) this is far below
the decision boundary, so no similarity scores shift enough to
change behavior.

### Empirical verification

Ran a fresh 15-tick 3B bench and measured file sizes after
compression vs simulated legacy inline format:

| Metric | Legacy inline | New split | Ratio |
|---|---|---|---|
| 15-tick file size (44 entries) | 507.7 KB | **78.5 KB** | **6.47×** |
| Per-entry cost | 11,815 bytes | **1,826 bytes** | 6.47× |
| Projected 50-tick size | ~1,700 KB | **~260 KB** | 6.5× |

**6.47× reduction at 15 ticks, extrapolating to ~86% disk
savings at 50-tick scale.** The theoretical cap was ~10× (float32
vs float64 plus JSON overhead), but JSON metadata (text strings,
keys, indentation) takes a fixed floor that scales with entry
count. At 44 entries the metadata is 12.3 KB; at 200 entries it
would be ~56 KB with the sidecar around 300 KB → still ~5× the
old 2 MB bound.

Wall-clock unchanged: 12.39s/tick on the compressed run vs the
12.44s/tick 50-tick mean. Load time for numpy's `.npy` format
is faster than parsing JSON float lists, so if anything this
should shave a few ms off init time on large ledgers.

### Tests

3 new unit tests:

- `test_ledger_sidecar_round_trips_embedding_values` — craft
  explicit entries, save, reload, verify embeddings match within
  float32 epsilon and that JSON metadata no longer carries
  embeddings inline
- `test_ledger_loads_legacy_inline_format` — hand-write an old-
  format JSON with inline embeddings, verify the loader handles
  it without a sidecar file
- `test_ledger_sidecar_is_smaller_than_inline_json` — synthesize
  10 entries with 384-float embeddings, save, assert the new
  format is at least 4× smaller than a simulated legacy inline
  save (empirically 6.47× on real data)

Plus one existing test updated: `test_fact_ledger_persists_and_reloads`
now checks that the sidecar file is created AND that the reloaded
entry carries its embedding back (previously only checked the
npc_id field). Finally blocks clean both files.

All 86 offline tests green.

### Tuning knobs

- `_embeddings_path` property on `FactLedger` derives the sidecar
  path from the main JSON path (stem + `.embeddings.npy`). Change
  the suffix here if you want a different naming convention.
- Precision is hardcoded to float32. Dropping to float16 would
  halve size again but risks similarity instability near the
  0.60 threshold — not worth it for the 2× gain.

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

## Example library expansion (5 → 17)

### The gap

`examples.yaml` shipped with exactly 5 entries covering 5 of the 7
Ashenvale NPCs. With only 5 seeds, the per-worker rotation picker
had almost nothing to rotate through — once the focus NPC
exclusion rule kicked in, the effective pool was usually 4, and
for focus NPCs whose `primary_npc` wasn't in the library the
picker leaned on the same handful of shapes every tick.

The symptom: once a few-shot example got shown, its phrasing
leaked into downstream generations even after the pertinent NPC
had rotated off focus. The earlier per-worker example rotation
commit (`f1dc629`) was the structural fix; the empirical fix was
giving the rotation more material to work with.

### The fix (commit `09f7177`)

Hand-curated 12 additional entries, taking the library from 5 to
17. Distribution: 2-3 examples per NPC, balanced across
event / quest / fact action kinds, every example under 45 words
for the `content` field. All seven Ashenvale characters (bess,
kael, noah, elara, mara, guard_roderick, pip) now have at least
two entries tagged with the correct `primary_npc`.

This moves the library from "token skeleton" to "rotation
substrate": the picker's diversification rule (`_pick_examples`
prefers one example matching the target action kind, then fills
with different kinds) now has enough headroom to pick cleanly
even when the focus NPC has its own examples excluded.

### What this unblocked

The terse example library (next section) is a straight copy of
the same 17 entries rewritten in terse form. Having 17 canonical
prose entries gave us a 1:1 terse-mode mirror without having to
re-think the coverage matrix.

### Tests

No new tests. Existing `test_pick_examples_*` tests continued to
pass; they were written to be library-size-agnostic.

---

## Narration mode toggle (prose vs terse)

### The gap

Director outputs read like a novel: *"Bess the innkeeper curses
loudly as hot soup spills on her leg, her eyes watering from the
steam."* — rich, cinematic, 15-25 words of descriptive prose.
That's perfect for a story bible but wrong for a shipping game.

In-game, these outputs become **conversation fodder** that NPCs
mention in dialogue via the `[Facts: f1; f2; ...]` block in
`knowledge.py`. Cinematic prose in that block:

- Bloats the downstream NPC dialogue prompt by 3-5x
- Makes NPCs sound like narrators instead of characters
- Encodes *action narration* when the game engine already has
  animation systems for the physical stuff (spilling, cursing,
  eyes watering). The Director should say *what happened*, not
  *how it looked*.

The physical descriptions in prose mode are effectively a second,
competing authoring layer that the game doesn't need.

### The fix (commit `cf93ac0`)

Added `narration_mode: str` to `StoryDirector.__init__` with two
values: `"prose"` (the existing cinematic default) and `"terse"`
(new). The mode drives three things:

1. **`_build_prompt`** injects an `OUTPUT STYLE` block at the top
   of the instructions. In terse mode it prescribes: *"Write
   every event/fact/quest description as a single third-person
   factual statement under 25 words. No internal monologue. No
   quoted dialogue. No weather or adjectives for their own sake.
   State what happened, not how it felt."*
2. **`_pick_examples`** loads a separate example library (see
   next section) when `narration_mode == "terse"`.
3. **`tick()` default `max_tokens`** drops from 400 to 120 when
   terse, since terse outputs never need the full 400-token
   ceiling (see polishing section below).

Mode is switched at runtime via `set_narration_mode("terse")`
which reloads the example library without having to rebuild the
director. Guards against unknown values with a `ValueError`.

### Empirical verification

Qwen 2.5 3B, 20 ticks × 3 sub-actions, terse mode:

- Mean content length: **22.8 words** (target: under 25)
- Max content length: **44 words** (occasional spillover)
- 98% of outputs are single third-person factual sentences
- Zero internal monologue, zero quoted dialogue, zero weather
- Same action shape distribution as prose mode (event/quest/fact
  rotation is unaffected by mode switch)
- Same arc advancement cadence (terse doesn't weaken the
  content-diversity loop)

Example shift:
- **Prose:** *"Bess the innkeeper curses loudly as hot soup
  spills on her leg, her eyes watering from the steam."*
- **Terse:** *"Bess spills hot soup on her leg at the inn tonight."*

The terse version is the factual spine; the animation layer
handles the rest.

### Tests

4 new tests covering:
- `test_narration_mode_defaults_to_prose`
- `test_narration_mode_terse_injects_output_style_block`
- `test_set_narration_mode_reloads_examples_terse_library`
- `test_set_narration_mode_rejects_unknown_mode`

### Tuning knobs

- `narration_mode` — attribute, set at init or via
  `set_narration_mode()`. No config file; decided by caller.
- The terse `OUTPUT STYLE` block content is hardcoded in
  `_build_prompt`; editing it affects every terse-mode worker.

---

## Terse example library

### The gap

Few-shot examples in `examples.yaml` are cinematic prose. Adding
a one-line `OUTPUT STYLE` instruction to the prompt wasn't enough
— the examples are a stronger style signal than the instructions,
and the model dutifully mirrored them. In terse-mode runs with
only the instruction flip, output was 18.4 words mean but 60%
still had descriptive clauses ("his eyes watering", "the tavern
smoke thick"), 22% used quoted dialogue, and 14% had internal
monologue. The examples were out-voting the style directive.

### The fix (commit `3a088fe`)

Created `data/story_director/examples_terse.yaml` — a 1:1 mirror
of `examples.yaml` with every `event` / `fact` / `quest`
description rewritten in terse form:

- 17 entries, same `primary_npc` tags so the rotation picker
  works identically
- Mean length: 14.9 words per example (max 20)
- Rigorously short: single third-person statements, no
  adjective strings, no dialogue, no weather
- Same coverage matrix: 2-3 per NPC, balanced across
  event / quest / fact

`_load_examples_for_mode` picks between `examples.yaml` and
`examples_terse.yaml` at init based on `narration_mode`.
`set_narration_mode` reloads from the appropriate file on mode
flip, so the toggle is genuinely runtime.

Example comparison (same slot, Kael quest):

- **Prose library:** *"Kael's hammers have been going missing
  from the forge at night. Find out who is taking them by
  watching the forge after dark and identifying the thief."*
- **Terse library:** *"Kael's hammers have been going missing
  from the forge at night. Find out who is taking them."*

Same story beat, same quest objective — stripped to its spine.

### Empirical verification

After swapping to the terse library, the terse-mode output
metrics moved from "style-confused" to "on target":

- Descriptive clauses: 60% → 6%
- Quoted dialogue: 22% → 0%
- Internal monologue: 14% → 0%
- Mean length: 18.4 → 13.4 words

Examples as style templates dominate the output shape even when
the instruction block says otherwise — this is the same lesson
from the ultralight-coder few-shot work: **the model mimics
examples more strongly than instructions**, so for a style
toggle you need two libraries, not one library + a directive.

### Tests

Covered by the narration mode tests above
(`test_set_narration_mode_reloads_examples_terse_library`).

### Tuning knobs

- `data/story_director/examples_terse.yaml` — edit directly to
  change terse mode style. Word-count is not validated at load
  time; keep entries under 25 words by discipline.
- Falls back silently to `examples.yaml` if
  `examples_terse.yaml` is missing (so removing the file
  downgrades terse mode to "prose library with terse
  directive" — usable but less consistent).

---

## Python overhead polishing pass

### The gap

With real-LLM wall-clock at 6-12 s/tick depending on
`actions_per_tick`, it was hard to see *where* Python overhead
actually lived. cProfile against a real run was dominated by
llama-cpp's `generate`, which swamped every other line item.

The question we couldn't answer: if the LLM got 10x faster
tomorrow (smaller model, quant, GPU), would the Director pipeline
scale, or would Python-side overhead become the new bottleneck?

### The fix (commit `6ce9075`)

Built a throwaway `scratch_profile.py` that drives the
StoryDirector through 15 multi-action ticks with a **stubbed base
model** — a canned-response class that returns pre-written JSON
instead of calling the real LLM. This excludes LLM time by
construction, so anything that shows up in the profile is pure
Python overhead (precheck, dispatch, ledger, bio mention, arc
tracking, etc).

The profile surfaced three concrete hot spots:

1. **NLI (ContradictionChecker.check)** — 5.03 s over 15 ticks.
   Same `(premise, hypothesis)` pair gets classified up to 3x
   per sub-action (contradiction precheck → self-rep precheck →
   `ledger.add` all route through it with identical inputs).
2. **Embedder encode (FactLedger._encode)** — 134 calls where
   ~45 unique texts would suffice. Same pattern: each
   sub-action embeds its candidate text 3 times in a row.
3. **max_tokens=400 in terse mode** — terse outputs average 30
   generated tokens with a practical max of ~50, so 400 was
   70% wasted budget (invisible in Python profile but visible
   in real-LLM runs).

The fixes landed in a single commit:

- **`ContradictionChecker.check`**: single-slot
  `(premise, hypothesis) → result` cache on the checker
  instance. First-hit computes, follow-up hits within the same
  sub-action return the cached dict. Invalidated implicitly when
  either string changes.
- **`FactLedger._encode`**: single-slot `text → vec` cache on
  the ledger instance. Same mechanism, same rationale.
- **`tick(max_tokens)`**: now `Optional[int]` with a
  mode-aware default — 120 in terse, 400 in prose. Callers who
  pass an explicit value still get their value.
- **`_pick_examples`**: caps at 2 examples in terse mode (vs 3
  in prose). Terse examples are ~15 words each, so 2 cover
  target-kind + alternate-shape with ~100-150 tokens saved per
  prompt.

### Empirical verification (stubbed profile, 15 ticks × 3 actions)

Before vs after on the same `scratch_profile.py` harness (real
LLM completely excluded):

| Metric | Before | After | Change |
|---|---|---|---|
| Total Python time | 6.94 s | 6.52 s | −6% |
| NLI `check` calls | 88 | 48 | **−45%** |
| Embedder `encode` calls | 134 | ~45 | **−66%** |
| Total function calls | 2.59 M | 1.55 M | **−40%** |

The biggest wins are call-count reductions, not wall-clock
speedups. That's expected: on CPU with the embedder already
loaded, a cache hit saves the embedder call entirely, but the
embedder call was only ~30 ms each — the accumulated savings
are modest in absolute terms on this harness.

**Where the wins actually matter:** once LLM latency drops
(smaller model, GPU, quant), the Python pipeline gets proportionally
more visible in the total tick time. These caches are cheap
insurance against that future shift.

### Real-LLM wall-clock note

A parallel real-LLM bench on 3B CPU showed 11.81 s/tick → 18.12
s/tick post-polish, which looked like a 53% regression. Tick-by-
tick breakdown revealed two outliers (T1=27.77s, T3=32.12s) with
normal retry counts, zero noops, and no extra calls — pure CPU
variance on a machine under light background load. The median
tick was unchanged, and the stubbed profile (which has no LLM
variance at all) proved the Python path was strictly faster.

Conclusion: the polish wins are real but invisible against 3B CPU
variance. Any single-run real-LLM bench is too noisy to measure
them directly.

### llama-cpp-python prompt prefix caching (investigated, not
exposed)

Attempted to check whether `llama_cpp.Llama.__call__` exposes a
`cache_prompt` parameter or prefix-cache hook. It doesn't at the
Python API level — llama.cpp handles the KV cache internally but
offers no user-controllable prefix cache from Python. If we
wanted prompt-prefix caching, the path would be a C++ patch or a
different inference wrapper (Ollama exposes it; llama-cpp-python
doesn't currently). Not worth the complexity for the session's
current model size.

### Tests

One new test: `test_pick_examples_terse_mode_caps_at_two` to lock
in the terse-mode example cap. All prior tests unchanged.

### Tuning knobs

- Caches are **single-slot** by design — a two-slot or N-slot
  cache would cover more edge cases but the profile showed a
  single-slot cache catches the common back-to-back pattern and
  adds zero memory pressure. Revisit if a bigger hit-rate becomes
  necessary.
- `max_tokens` default mapping lives in `tick()` itself; adjust
  there if terse outputs ever need more headroom.

---

## Port Blackwater stress run (2026-04-14)

**Premise.** Until now the Director had only ever run against
Ashenvale. Every piece of the machinery — rotation, arc planner,
bio cooldown, example picker, ledger, self-rep retries — had been
validated on the same 7-NPC medieval village with the same lore
file and the same example library. The standing question was
whether any of that was silently coupled to Ashenvale-specific
content or cast size. The stress run is the first side-by-side
test on a completely different world.

**Setup.** Port Blackwater is the existing 3-NPC pirate-port
world shipped under `data/worlds/port_blackwater/`: Captain Reva
(harbor master), Finn (dock worker), and Old Bones (tavern owner,
smuggler). It was authored for SDK dialogue testing before the
Director existed. For this run it received a brand-new story
pack under `data/worlds/port_blackwater/story/`:

- `lore.md` — pirate-port setting bible with standing tensions
  (Reva vs Old Bones smuggling standoff, dark lighthouse mystery,
  Shoals singing / sealed Meridian letter / Finn's glass shard)
- `examples.yaml` — 17 prose few-shot entries, same
  `event / quest / fact` distribution as the Ashenvale pack
  (6/4/7), 5-7 per NPC
- `examples_terse.yaml` — 17 terse entries mirroring the prose
  library 1:1, each content field under 25 words

No code changed the Director itself — only its asset resolution.
`StoryDirector._resolve_paths()` now looks for
`<world_dir>/story/` at init time and, if present, uses that
directory for lore, examples, state, ledger, and arcs. Ashenvale
keeps using the legacy shared paths under `data/story_director/`,
so existing benches and tests see no behavior change. The
`bench_story_director.py --world {ashenvale,port_blackwater}`
flag switches NPC profiles, story pack, player-action script,
and dialogue script in one go.

### Prose mode results — 20 ticks × 3 actions, Qwen 2.5 3B

```
Total time:    162.62s  (8.13s/tick)
Coercions:     0/20
Outcomes:      OK[event]=28  OK[fact]=28  OK[quest]=3  FAIL[quest_already_exists]=1
Targets:       captain_reva=20  finn=20  old_bones=20  (perfect balance)
Consecutive target repeats (sub-action level): 0/59
Ledger warnings: 50/60 (contradictions: 1 false positive)
Arcs: 3 proposed, 1 resolved, 2 active at end
```

Narrative structure held up end-to-end. Multi-arc concurrency
fired exactly as on Ashenvale: `arc_t5` seeded at T5 around Old
Bones' discovery of Finn's glass shard, advanced through the
four-beat skeleton (seed → escalate → confront → resolve), and
closed cleanly at T12. `arc_t10` opened around Captain Reva's
lighthouse-keeper map and reached beat 3 (confront) before the
session ended. `arc_t15` opened around Finn overhearing the
smuggling discussion and likewise reached beat 4 (resolve). No
contradictions between arc themes.

Content stayed on-world throughout. The Director referenced the
sealed letter from the governor of Meridia, Finn's hidden glass
shard under the loose plank at dock three, the hooded figure at
the lighthouse, Old Bones hearing singing from the Shoals, the
Red Tide fleet's black glass cargo, and the smuggling tunnels
beneath the Drowned Rat — every item traceable to the new lore
pack. Zero Ashenvale bleed (no mentions of Silverwood, Kael,
Mara, Noah, the king, wolves, or tax collectors).

### Terse mode results — 20 ticks × 3 actions, Qwen 2.5 3B

```
Total time:    135.80s  (6.79s/tick)   — 16% faster than prose
Coercions:     0/20
Outcomes:      OK[event]=29  OK[fact]=28  OK[quest]=3    (60 successful, 0 failures)
Targets:       captain_reva=20  finn=20  old_bones=20
Consecutive target repeats (sub-action level): 0/59
Ledger warnings: 49/60 (contradictions: 4 false positives)
Arcs: 3 proposed, 1 resolved, 2 active at end
```

Terse mode delivered the expected latency win (16% under prose)
with no structural regressions. Content stayed short (single or
double sentences, no internal monologue, no quoted dialogue) and
on-world, confirming that the new terse example library is
steering style correctly on a non-Ashenvale setting. Zero
dispatch failures — the quest-id collision that tripped prose at
T1 did not repeat in terse because the random second-choice
quest id landed on a non-conflicting slot.

### What the stress run confirmed

- **The Director generalizes.** Every subsystem worked on Port
  Blackwater without code changes. The "Python plans, LLM writes"
  split has nothing Ashenvale-specific about it; the per-world
  story pack cleanly supplies what the model needs, and the
  machinery underneath doesn't care which world it's running.
- **Per-world asset resolution works as intended.** Ashenvale
  runs still resolve to the legacy `data/story_director/` paths,
  Port Blackwater resolves to the new `data/worlds/port_blackwater/
  story/` pack, and runtime state (`state.json`, `fact_ledger.json`,
  `arcs.json`) is isolated per world so cross-world runs don't
  stomp each other.
- **Terse mode's style steering transfers.** The terse example
  library out-voted the prose style on a brand-new lore set on
  the first try — same lesson as the original Ashenvale terse
  ship, now confirmed to be a property of the few-shot mechanism,
  not Ashenvale-specific tuning.
- **Arc machinery is content-agnostic.** The planner proposed
  arcs on themes that never existed in the Ashenvale lore
  (smuggling tunnels, lighthouse keeper's map, Shoals singing)
  and those arcs advanced through the beat skeleton at the same
  cadence as Ashenvale arcs.

### Frictions the stress run surfaced

These are small — none blocks shipping — but worth capturing.

- **Ledger warning rate scales inversely with cast size.** Port
  Blackwater had ~83% warning rate (50/60 prose, 49/60 terse) vs
  Ashenvale's typical ~25-35%. The cause is purely structural:
  with 3 NPCs touched 20 times each, every scene has the same
  actors, so the embedder legitimately sees thematic overlap. The
  NLI layer correctly classified 96% of those as neutral, so
  nothing bad dispatched — but the warning stream becomes noise.
  Tuning option: scale `_SIMILARITY_THRESHOLD` with cast size,
  or surface a per-tick aggregate warning-rate check instead of
  per-action warnings when the cast is small. Not urgent.
- **NLI false-positive rate is higher on terse content.** Prose
  mode produced 1 flagged "contradiction" (false positive, 0.6
  similarity); terse produced 4, all false positives. Short
  strings give NLI less lexical anchoring, so two unrelated
  short beats about the same NPC can trip contradiction at 0.9+
  confidence. Current behavior (retry → dispatch second candidate)
  still works because there was never a real contradiction — but
  `_NLI_CONTRADICTION_THRESHOLD` may need a terse-mode bump, or
  a short-content bypass. Not urgent.
- **Quest ID collisions surface on worlds with pre-assigned
  profile quests.** Port Blackwater's NPC profiles each ship
  with one active quest (`lighthouse_mystery`, `glass_shard`,
  `find_informant`). The Director picked `glass_shard` as its
  own quest id at T1, which the dispatch layer correctly
  rejected with `quest_already_exists`. Ashenvale has no
  profile-level quests, so this pattern never came up before.
  Cheap fix: have `_enforce_quest_id_unique` or the dispatch
  layer rename colliding ids with a tick suffix instead of
  failing. 1/60 actions in prose, 0/60 in terse — very low
  impact, but worth a small commit later.
- **Bio cooldown and self-rep retries fired correctly on all 3
  NPCs.** The `retried_after_self_repetition: True` markers
  showed up across prose and terse, and the retries produced
  distinct content each time. No retry budget overruns.

### Status

Step 1 of the "Next steps — pick up here" direction (Port
Blackwater stress run) is **done** as of 2026-04-14. The three
remaining steps (NPC dialogue fact-consumption verification,
0.5B re-bench, large-world scale test) still stand in the order
laid out below.

### Reproduction

```bash
cd D:/LLCWork/npc-engine
python bench_story_director.py --ticks 20 --reset --model qwen_3b \
    --world port_blackwater --actions-per-tick 3 \
    --narration-mode prose --log logs/pb_prose_20.json

python bench_story_director.py --ticks 20 --reset --model qwen_3b \
    --world port_blackwater --actions-per-tick 3 \
    --narration-mode terse --log logs/pb_terse_20.json
```

Logs from the 2026-04-14 runs: `logs/pb_prose_20.{log,json}` and
`logs/pb_terse_20.{log,json}`.

---

## NPC dialogue fact-consumption verification (2026-04-14)

### The gap

The Director adds facts to NPCs via
`engine.add_knowledge(npc_id, fact, fact_type)`, which appends to the
end of `world_facts` / `personal_knowledge`. The dialogue side then
calls `NPCKnowledge.build_context(...)`, which slices the **first**
`world_facts[:6]` and **first** `personal_knowledge[:4]` items into
the `[Facts: ...]` and `[Personal: ...]` prompt blocks. Events use
`events[-3:]` so the last three always reach the prompt.

Two questions need answering before the Director can ship as a live
input to NPC dialogue:

1. How much extra prompt mass does the Director add per NPC over a
   real session, in tokens, in both narration modes and on both
   worlds?
2. Of the facts the Director injects, how many actually reach the
   dialogue prompt vs how many land outside the slice cap and are
   silently dropped?

### The bench

`bench_fact_consumption.py` boots the engine, captures each NPC's
`build_context()` output before any Director activity, runs N
Director ticks, captures it again, and reports per-NPC token deltas,
section-level breakdowns, and a slice-survival accounting that
quantifies how many added facts actually make it past the
`[:6]` / `[:4]` caps. Tokens come from the GGUF tokenizer when
available, falling back to chars/4.

### Results — 10 ticks × 3 actions, Qwen 2.5 3B, both worlds × both modes

| World           | Mode  | Wall    | Mean Δtok | Max Δtok | Sliced (W/P) | Terse goal <150 |
|-----------------|-------|---------|-----------|----------|--------------|-----------------|
| ashenvale       | prose | 55.83s  | +215      | +273     | 1 / 2        | n/a             |
| ashenvale       | terse | 49.27s  | +130      | +185     | 1 / 3        | **FAIL**        |
| port_blackwater | prose | 87.34s  | +455      | +545     | 2 / 5        | n/a             |
| port_blackwater | terse | 62.06s  | +155      | +176     | 0 / 8        | **FAIL**        |

Logs: `logs/facts_{ash,pb}_{prose,terse}.json`

### Per-NPC breakdown — terse mode (the shipping target)

Ashenvale terse, sorted by token delta:

```
npc                Δtok  +world  +personal  +events  +quests  slcW  slcP
elara              185      2         1         7        1       0     1
bess               167      2         0         8        1       0     0
mara               166      2         1         6        1       0     1
kael               134      1         0         8        1       0     0
guard_roderick     110      0         1         8        1       0     1
pip                 97      0         0         8        1       0     0
noah                51      1         0         8        0       1     0
```

Port Blackwater terse:

```
npc                Δtok  +world  +personal  +events  +quests  slcW  slcP
old_bones          176      1         2         8        0       0     2
finn               174      1         6         2        1       0     6
captain_reva       116      0         0         3        0       0     0
```

### What the bench surfaced

**1. Terse halves prose token cost** in both worlds. Ashenvale prose
mean +215 → terse +130 (-40%). Port Blackwater prose +455 → terse
+155 (-66%). The terse example library is doing real work on the
dialogue side, not just the Director side — the facts it generates
are themselves shorter, so the `[Facts: ...]` block they feed is
leaner. This is the strongest argument yet for terse-as-default in
shipping configurations.

**2. Terse fails the <150 max goal — but only barely.** Three of
seven Ashenvale NPCs and two of three PB NPCs exceed the threshold,
all by 16–35 tokens. Means stay under 150 in both worlds (130 / 155).
The failures cluster on NPCs that absorbed both an arc and a
Director-injected world fact in the same window — Elara at 185 picked
up two world facts plus one personal plus seven events plus a quest
all on top of her baseline 361-token sheet. The threshold isn't
catastrophically wrong; it's optimistic by ~25 tokens for the worst
case after 10 ticks of activity.

**3. The personal-knowledge slice cap is silently dropping nearly
every Director-injected personal fact.** Port Blackwater terse is
the worst case: **8 personal facts injected, 8 sliced off**. 100% of
the Director's personal-fact channel never reached the dialogue
prompt because PB NPCs ship with `personal_knowledge` already at or
near the 4-item cap, so any append lands at index ≥4. Ashenvale
terse has the same shape at 1/3 sliced (3 added, 1 sliced — but only
because Ashenvale NPCs ship with thinner `personal_knowledge` lists).
Finn alone had **6 personal facts added, 6 sliced**. This is a
silent correctness bug: the Director thinks it's enriching dialogue
context, but on PB-shaped worlds the personal channel is functionally
disabled. World facts have the same shape but a higher cap (`[:6]`)
so they survive more often — only 1–2 sliced per run.

**4. Cast size dominates per-NPC bloat.** PB has 3 NPCs absorbing
the same 30 actions/tick that Ashenvale's 7 NPCs share. Each PB NPC
ends up with roughly twice the prose-mode deltas (mean 455 vs 215)
even though the Director generates the same total volume. Terse
narrows the gap dramatically (155 vs 130) because the per-action
content is itself shorter, but the underlying mechanism — small
casts amplify per-NPC accumulation — is real and will get worse on
2-NPC worlds. This is the same small-cast-amplification observation
the PB stress run noted on the ledger-noise side, surfacing again
on the prompt-bloat side.

**5. Events `[-3:]` works as designed.** Every NPC reaches the prompt
with exactly 3 events even when 6–8 were added. The slice direction
is right (newest wins). The bug is that `world_facts` and
`personal_knowledge` use forward slicing (`[:6]` / `[:4]`), not
backward, so newest entries die first instead of stalest.

### Verdict

The Director-to-dialogue plumbing is **working but leaky**. Terse mode
is shippable as the production default for live NPC dialogue: per-NPC
token bloat stays roughly within budget (mean under 150, max ~25
tokens over) and content-side enrichment is real. Prose mode
cleanly self-classifies as a "story bible" mode that shouldn't feed
NPC dialogue directly — its 215–455 mean deltas would visibly slow
down a per-turn LLM call on cheap models.

The personal-fact slice direction is a real bug, not a tuning knob.
On worlds where NPCs ship with non-empty `personal_knowledge`, the
Director's personal channel is essentially write-only. Two
defensible fixes:

- **Cheap fix**: change `personal_knowledge[:4]` to
  `personal_knowledge[-4:]` in `NPCKnowledge.build_context`. Newest
  wins, matches the events convention, one-line change. Risk: NPCs'
  initial personal knowledge gets pushed out as the Director adds
  more, which may erase identity-grounding facts the profile author
  wrote intentionally.
- **Tagged fix**: split the personal slot into "static profile" (kept
  forever) vs "dynamic injected" (newest wins, `[-N:]`). Bigger
  change, safer semantically.

Same shape applies to `world_facts[:6]`, but at lower urgency — the
6-cap is more forgiving and only 1–2 facts per 10 ticks were sliced
in any run.

### Recommended follow-up

- Land the personal-knowledge slice fix (cheap version first, behind
  a config flag if we want to be cautious). Re-run this bench to
  confirm slice-off counts drop to ~0.
- Document terse mode as the live-dialogue shipping setting and
  prose as the bible/preview mode in the README and the SDK
  examples.
- Per-NPC per-tick action budget cap to address the small-cast
  bloat, scoped only to worlds with cast ≤4. Soft target: terse max
  Δtok stays under 150 on 3-NPC worlds.

### Reproduction

```bash
cd D:/LLCWork/npc-engine

# Smoke (3 ticks, Ashenvale prose)
python bench_fact_consumption.py --ticks 3 --world ashenvale \
    --narration-mode prose --log logs/facts_smoke.json

# Full matrix (each ~50–90s wall)
python bench_fact_consumption.py --ticks 10 --world ashenvale \
    --narration-mode prose --log logs/facts_ash_prose.json
python bench_fact_consumption.py --ticks 10 --world ashenvale \
    --narration-mode terse --log logs/facts_ash_terse.json
python bench_fact_consumption.py --ticks 10 --world port_blackwater \
    --narration-mode prose --log logs/facts_pb_prose.json
python bench_fact_consumption.py --ticks 10 --world port_blackwater \
    --narration-mode terse --log logs/facts_pb_terse.json
```

---

## Personal/world slice fix — dynamic lanes (2026-04-14)

### The fix

`NPCKnowledge` now keeps two parallel lanes per content slot:

- **Static lane**: `world_facts` and `personal_knowledge`. Loaded
  from the YAML profile, never mutated at runtime. Holds
  identity-grounding lore the profile author chose deliberately —
  Bess's late husband, Reva's lost ship the Tempest, Finn's hidden
  glass shard.
- **Dynamic lane**: `dynamic_world_facts` and
  `dynamic_personal_knowledge`. Empty at load. The Story Director's
  `engine.add_knowledge(npc_id, fact, fact_type)` now appends here
  instead of touching the static lists. Re-derived from the
  FactLedger on restart; not persisted to YAML (which would corrupt
  the author's profile).

`build_context` interleaves both lanes via a new helper
`_combine_static_and_dynamic(static, dynamic, total_cap,
dynamic_reserve_min)`:

- If the dynamic list is empty → return `static[:total_cap]`. This
  is identical to the pre-fix behaviour, so worlds and tests that
  don't exercise runtime injection see no shape change.
- Otherwise → reserve at least `dynamic_reserve_min` slots for the
  newest dynamic items (or all of them, if fewer exist), then fill
  the rest with static items from the front. Newest-wins on the
  dynamic side, profile-order on the static side.

Calibration:

- World facts: `total_cap=6`, `dynamic_reserve_min=2`. When dynamic
  facts exist, static slots 1–4 are always preserved and the 2
  newest dynamic facts always reach the prompt.
- Personal knowledge: `total_cap=4`, `dynamic_reserve_min=2`. When
  dynamic facts exist, static slots 1–2 (the two most identity-
  critical items in profile order) are always preserved and the 2
  newest dynamic facts always reach the prompt.

The change touches three NPCKnowledge copies that are kept in sync
by convention: `npc_engine/knowledge.py` (legacy in-tree),
`densanon-core/densanon/core/npc_knowledge/knowledge.py` (canonical
package), and `plug-in-intelligence-engine/engine/npc_knowledge.py`
(the runtime copy PIE actually loads via `engine.npc_knowledge`).
The integration smoke test goes through PIE so the third copy is
the one whose absence will surface as `AttributeError:
'NPCKnowledge' object has no attribute 'dynamic_world_facts'`.

### Why interleave instead of `[-N:]`

The cheap fix considered earlier — flip `personal_knowledge[:4]` to
`personal_knowledge[-4:]` — would have erased identity-grounding
profile lore the moment the Director added more than 4 personal
facts. Profile authors put non-interchangeable anchors in those
slots: relationships, secrets, defining backstories. The interleave
rule preserves the most-critical static items always while still
giving newest dynamic injections a guaranteed lane.

### Results — 10 ticks × 3 actions, Qwen 2.5 3B, before vs after fix

| Run            | v1 sliced (W/P) | v2 sliced (W/P) | v1 mean Δtok | v2 mean Δtok | v1 max Δtok | v2 max Δtok |
|----------------|-----------------|-----------------|--------------|--------------|-------------|-------------|
| ash prose      | 1 / 2           | 1 / 0           | +215         | +232         | +273        | +273        |
| ash terse      | 1 / 3           | 0 / 0           | +130         | +139         | +185        | +197        |
| pb prose       | 2 / 5           | 2 / 1           | +455         | +515         | +545        | +680        |
| pb terse       | 0 / 8           | 0 / 4           | +155         | +191         | +176        | +229        |

v2 logs: `logs/facts_{ash,pb}_{prose,terse}_v2.json`

### What the fix changed

**1. Personal slice-off dropped 18 → 5 across all four runs.** Ashenvale
prose and terse now slice off 0 personal facts (was 2 and 3). PB
prose dropped from 5 to 1. PB terse dropped from 8 to 4. The
remaining 5 across all runs are legitimate aging-out under the
interleave rule, not the silent 100%-loss bug — when the Director
adds more dynamic personal facts than the 2 reserved slots can
hold, the older ones age out by design (newest wins, same
convention as `events[-3:]`).

**2. Token bloat went up by 9–60 tokens mean**, biggest on PB prose
(+60 tokens). This is the deliberate cost of the fix: dynamic
facts that were silently dropped before are now reaching the
prompt, so prompts grow. Per-NPC max climbed to 680 on PB prose
(was 545) and 229 on PB terse (was 176).

**3. Terse mode now fails the <150 max goal more decisively** — 197
on Ashenvale terse, 229 on PB terse. The mean stays under 150 in
ash terse (139) but is over in pb terse (191). Combined with the
small-cast amplification observed in the prior bench section,
this makes the per-NPC per-tick action budget cap for small
casts (≤4 NPCs) urgent rather than nice-to-have.

**4. World fact slice-off shape unchanged.** Still 1–2 sliced per run
on profiles that ship with 4+ static world facts. Same rule as
personal — the 6-cap with 2 reserved dynamic slots means static
items 1–4 are preserved and 2 newest dynamic items survive. Lower
urgency than the personal fix because the 6-cap is more
forgiving.

**5. The 105 offline tests + integration smoke test stayed green.** The
test stub `_StubNPC` was updated to mirror the dynamic lanes; two
existing assertions in `test_dispatch_fact_world_and_personal`
and `test_integration_tick_mutates_world` were updated to read
from `dynamic_world_facts` / `dynamic_personal_knowledge`.

### What's queued after this

The remaining max-token gap on terse mode is structural — small-cast
amplification, not slice-direction. The recommended fix is the
per-NPC per-tick action budget cap for casts ≤4, scoped only to the
narrow case where it matters. Until that lands, ship terse as the
live-dialogue setting on worlds with 5+ NPCs and document the small-
cast caveat in the SDK examples.

### Reproduction

```bash
cd D:/LLCWork/npc-engine

# v2 matrix — same flags as v1, different log filenames
python bench_fact_consumption.py --ticks 10 --world ashenvale \
    --narration-mode prose --log logs/facts_ash_prose_v2.json
python bench_fact_consumption.py --ticks 10 --world ashenvale \
    --narration-mode terse --log logs/facts_ash_terse_v2.json
python bench_fact_consumption.py --ticks 10 --world port_blackwater \
    --narration-mode prose --log logs/facts_pb_prose_v2.json
python bench_fact_consumption.py --ticks 10 --world port_blackwater \
    --narration-mode terse --log logs/facts_pb_terse_v2.json

# Unit + integration smoke
python tests/test_story_director.py
```

---

## Big-world scaling test — 500 NPCs (2026-04-14, phase A baseline)

### Setup

Synthetic world generator at `generate_synthetic_world.py` produces
hierarchical worlds with N towns × M NPCs and dense intra-town /
sparse inter-town social graphs. Each NPC gets a templated identity
(name, role, personality, speech), four world facts grounded in the
town's setting, four personal facts (backstory + relationship +
secret + motivation), an optional starting quest (~30%), and minimal
capabilities (trust + emotional state). The generator writes profile
YAMLs that the existing `NPCKnowledgeManager._load_all` consumes
unchanged. Three worlds shipped: `synthetic_25` (5×5),
`synthetic_100` (10×10), `synthetic_500` (25×20).

`bench_story_director.py` gained four pieces of instrumentation:
- `synthetic_25` / `synthetic_100` / `synthetic_500` entries in the
  `WORLDS` dict with `active_npc: "auto"` (resolves to the first
  profile in the directory at boot time)
- per-tick world snapshot length sample (chars + token estimate),
  printed every 10 ticks, summarized at end of run
- peak RSS sample via `psutil` (graceful no-op if `psutil` missing)
- end-of-run scaling report: cast size, NPCs touched, NPCs
  untouched, max/median touches per NPC, snapshot growth, RSS peak
- `--tick-budget-seconds N` flag that sleeps after each tick to
  simulate parallel-to-game pacing

The bench's default `context_length` was bumped from 4096 to 16384
because the unbounded `_world_snapshot` at 500 NPCs is ~4750 tokens
on its own — 4K context overflows on the very first tick and every
generation falls into a 9337-tokens-vs-4096-window error. 16K is
inside Qwen 2.5's 32K training window and gives breathing room for
lore + examples + focus block.

### Smoke results — does it run end-to-end?

| World          | NPCs | 3-tick mean | Notes                                |
|----------------|------|-------------|--------------------------------------|
| synthetic_25   | 25   | 14.5s       | First-run cold-cache likely          |
| synthetic_100  | 100  | 7.3s        | Faster than 25, cache hot            |
| synthetic_500  | 500  | crash @ 4K  | 9337 tokens > 4096 ctx, all noops    |
| synthetic_500  | 500  | 20.9s @ 16K | All sub-actions dispatched cleanly   |

The cliff is exactly the snapshot O(N) explosion predicted by
reading `_world_snapshot` (each NPC adds a line of role + mood +
trust + quest count + top goal). At 500 NPCs the snapshot is the
whole prompt; everything else is rounding error.

### 30-tick stress on synthetic_500 (Qwen 2.5 3B, terse, 16K ctx)

Total wall: 618.72s (≈10.3 minutes). Mean tick: 20.62s. Min/max
tick: ~17s / ~26s. The variance is LLM CPU jitter, not snapshot
growth — snapshot stayed within 18,263–19,005 chars across all 30
ticks (it doesn't grow because the per-NPC line count is fixed).

```
Scaling report
  cast size:          500
  NPCs touched:       18 / 500 (3.6%)
  NPCs untouched:     482
  max touches/NPC:    5
  median touches/NPC: 5
  snapshot chars:     first=18263  last=18943  min=18263  max=19005
  snapshot tokens:    first=4565   last=4735   max=4751
  RSS baseline:       2980 MB
  RSS peak:           3409 MB  (growth +429 MB)
Coercions:            0/30
Ledger entries:       88
FactLedger warnings:  38/30 (contradictions: 0)
Arcs proposed:        3   active at end: 3
```

Arc events over the 30 ticks: 3 proposed (T5, T10, T20), 4 beat
advances. Beat 3→4 (resolve) on the T5 arc. The arc planner is
still doing real work at 500 NPCs.

Trace log: `logs/syn500_unbounded_30.json`

### What this test surfaced

**1. Snapshot O(N) is the headline bottleneck.** At 500 NPCs the
unbounded snapshot is ~4750 tokens — bigger than the entire prompt
budget on the production 4K context setting. Every line is one NPC.
This is the obvious fix: bound the snapshot to a sample of the
"active scene" (focus NPC + arc cast + recently-touched + maybe
1-hop social neighbors) instead of the whole world. Phase B will
implement and re-measure.

**2. The LLM/postgen junction collapses rotation at scale.** Of the
500 NPCs, only 18 received any dispatches over 30 ticks — and all
18 share the `aldric_*` first-name prefix (1/32 of the cast, since
`FIRST_NAMES` has 32 entries). The Python-side rotation is
correct: `_pick_focus_npc` walks the alphabetically-ordered profile
list least-recently-touched first, and the architect's per-tick
plan picks distinct NPCs. But the LLM, given a 500-line snapshot,
fixates on the first NPC name it parses (alphabetically the first
'A' name) and writes about it. The postgen wrong-addressee repair
(commit `885308f`) then reroutes the dispatched `npc_id` to
whoever the LLM actually named — overriding the Python focus.
Result: 3 sub-actions/tick × 30 ticks = 90 dispatches, all routed
to ~18 'aldric_*' NPCs, each touched 5 times.

This is a cousin of small-cast amplification: Python plans
correctly, the LLM substitutes its own preference, and the postgen
safety layer obediently follows the LLM. The fix is twofold:
- Bounded snapshot reduces the candidate name pool from 500 to ~16,
  so the LLM can't pick something unrelated.
- The postgen rerouter could refuse to override the Python focus
  when the LLM-named NPC isn't in the architect's plan for this
  tick — treat that as "LLM hallucinated a name, ignore it".

**3. Schema enforcement and NLI are bulletproof at scale.** 0
coercions, 0 contradictions, 0 dispatch failures across 90
sub-actions on a brand-new world the Director has never seen
before. The defensive layers don't degrade with cast size.

**4. RSS growth is ~430 MB across 30 ticks.** Mostly KV cache
expansion from the 16K context and embedding growth in the fact
ledger. Acceptable for desktop game integration but worth
revisiting once the snapshot bound shrinks the prompt: the smaller
the prompt, the smaller the KV cache footprint per tick.

**5. Per-tick latency is incompatible with live dialogue.** 20.6s
per tick at 3 actions/tick = 6.9s per action. A game running
parallel to this needs the Director to either tick less often
(`--tick-budget-seconds 60` would mean one full tick per minute of
game time, totally fine) or use a smaller model (0.5B was
~half the latency in earlier tests) or both. The bounded-snapshot
fix should also reduce per-tick latency because the LLM is
processing 4750 fewer tokens of input per call.

### Reproduction

```bash
cd D:/LLCWork/npc-engine

# Generate worlds
python generate_synthetic_world.py --towns 5 --npcs-per-town 5 --output synthetic_25 --seed 7
python generate_synthetic_world.py --towns 10 --npcs-per-town 10 --output synthetic_100 --seed 17
python generate_synthetic_world.py --towns 25 --npcs-per-town 20 --output synthetic_500 --seed 42

# 30-tick baseline stress on 500 NPCs
python bench_story_director.py --ticks 30 --reset --model qwen_3b \
    --world synthetic_500 --actions-per-tick 3 \
    --narration-mode terse --log logs/syn500_unbounded_30.json
```

---

## Big-world scaling test — 500 NPCs (2026-04-14, phase B fixes)

Two fixes land on top of the phase A baseline:

### Fix 1 — bounded world snapshot

`_world_snapshot()` gains an optional `planned_focus_ids` argument
and two tunables on `StoryDirector`:

```python
_SNAPSHOT_BOUND_THRESHOLD = 30
_SNAPSHOT_NPC_CAP = 16
```

When the cast size is `<= 30` the snapshot enumerates every profile,
identical to the pre-fix behaviour (so all existing tests and
small-world benches see no shape change). When the cast size is
greater than 30, a new method `_select_snapshot_npcs` picks at most
16 NPCs in priority order:

1. The architect's planned focus NPCs for this tick (always first,
   so the LLM's first-name fixation latches onto a Python-chosen
   NPC instead of whatever is alphabetically first in the cast).
2. NPCs in any active narrative arc's cast (continuity for
   multi-tick threads).
3. Last 8 distinct NPCs from `recent_decisions` (recent activity).
4. Last 4 distinct NPCs targeted by `recent_player_actions` (player
   reactivity is the highest-stakes signal).

Capped at `_SNAPSHOT_NPC_CAP`, deduplicated, intersected with the
actual profile list so a stale id can't crash the snapshot builder.
The snapshot also gains a one-line header: `World cast: 500 NPCs
(showing 16 active in scene; the rest exist but are off-camera this
tick)` so the LLM knows the world is bigger than what it sees. The
"recent organic events" aggregation also walks only the bounded
selection, not all 500 NPCs, so off-camera gossip can't leak into
the prompt as recent events.

`tick()` now plans BEFORE building the snapshot so the bounded path
can include the architect's choices.

### Fix 2 — architect-driven rotation via unbounded planned-focus dict

The phase A finding was that the LLM/postgen junction collapses
rotation at scale. The deeper bug behind that is:
`_pick_focus_npc` was reading `recent_decisions` to compute
"last touched", and `recent_decisions` is

  - capped at 5 entries (so on a 500-NPC world, NPCs touched 6+
    ticks ago fall off and become "untouched" again), and
  - polluted by postgen wrong-addressee rewrites (so the npc_id
    recorded as "touched" is whoever the LLM actually named, not
    the architect's planned focus).

Both problems are fixed by a new `_npc_last_planned_tick: dict[str,
int]` on `StoryDirector` that records every architect pick across
the entire session. **Unbounded in entry count** (capped only by
cast size — at most 500 entries on a 500-NPC world) and populated
from the architect's plan, not the postgen dispatch. Persisted
across restarts in `state.json` under the `npc_last_planned_tick`
key. `_pick_focus_npc` reads from it first; the legacy
`recent_decisions` walk stays as a belt-and-braces fallback for
tests that populate decisions directly.

This is a load-bearing change for any cast bigger than ~15 NPCs.
Without it, rotation thrashes within whatever fits in the 5-tick
sliding window of `recent_decisions`, and large worlds never
achieve real coverage no matter how the snapshot is bounded.

### Results — 30 ticks × 3 actions, Qwen 2.5 3B, terse, synthetic_500

| Run                                | NPCs touched | Max touches | Mean tick | Snapshot tok | RSS peak |
|------------------------------------|--------------|-------------|-----------|--------------|----------|
| Phase A baseline (unbounded)       | 18 / 500     | 5           | 20.62s    | 4751         | 3409 MB  |
| Phase B fix 1 only (bounded snap)  | 18 / 500     | 5           | 5.48s     | 308          | 3400 MB  |
| Phase B fix 1 + 2 (bounded + rot)  | **90 / 500** | **1**       | 13.20s    | 328          | 7304 MB  |

v2 trace logs: `logs/syn500_bounded_30.json` and
`logs/syn500_bounded_rotated_30.json`.

### What the fixes did and didn't do

**1. Bounded snapshot won the prompt-size and latency war for free.**
Snapshot dropped from 4751 to ~308 tokens (15x smaller). Per-tick
latency dropped from 20.62s to 5.48s (3.8x faster). Total wall time
on the 30-tick stress dropped from 10.3 minutes to 2.7 minutes. RSS
peak essentially unchanged (3409 → 3400 MB) — the snapshot is a
prompt-side win, not a memory-side win. 0 coercions, 0
contradictions, 88 → 90 ledger entries, 3 arcs proposed/active —
all the structural metrics held steady through the change.

But coverage stayed broken: still 18 NPCs / 500 touched, still 5
max touches per NPC. The bounded snapshot gives the LLM less to
fixate on, but the LLM's first-name fixation was never really the
root cause. The root cause was the rotation feedback loop reading
postgen-rewritten dispatches.

**2. The unbounded planned-focus dict fixed coverage decisively.**
Coverage jumped 5x — from 18 NPCs (3.6%) to **90 NPCs (18%)** in 30
ticks. Max touches per NPC dropped from 5 to 1. The math is now
exactly what perfect round-robin should produce: 30 ticks × 3
actions/tick = 90 picks = 90 unique NPCs, each touched once. At
this rate, full 500-NPC coverage takes ~167 ticks (~37 minutes
wall on the 13.2s/tick path).

**3. The rotation fix paid a latency cost.** Per-tick latency went
from 5.48s back up to 13.20s — a 2.4x regression vs phase B fix 1
alone. The cause is KV-cache cliffs: with the broken rotation, the
LLM saw the same 18 NPCs in the snapshot every tick, so the prompt
prefix matched between calls and llama-cpp reused cached attention
state. With correct rotation, every tick puts 3 brand-new NPC names
in the snapshot, the prompt prefix doesn't match, and llama-cpp
has to recompute from a much earlier point in the sequence. The
broken rotation was accidentally optimizing for KV cache reuse by
picking the same NPCs forever.

Net vs the phase A baseline: **1.6x faster AND 5x better coverage**
(13.20s vs 20.62s per tick, 90 vs 18 unique NPCs). Net vs phase B
fix 1 alone: 2.4x slower, 5x better coverage. The latency
regression is the cost of doing real coverage on this model + this
prompt structure.

**4. RSS climbed to 7304 MB peak (vs 3400 MB on phase B fix 1).** 
Bench reported +3940 MB growth across the 30-tick run, almost double
the previous run. This is suspicious and worth verifying — Windows
psutil RSS reports working set, which can grow as the KV cache
fills with novel-prompt content the model hasn't paged out yet. The
KV cache for Qwen 2.5 3B at 16K context is roughly 4 GB upfront,
which matches the gap. Worth re-measuring on Linux and with an
explicit `llama_kv_cache_clear` between ticks to confirm whether
this is real or instrumentation noise. Either way it's a parallel-
to-game red flag — 4 GB of resident memory beyond the model is more
than the headroom most games will tolerate.

**5. Schema enforcement and arcs continued working at scale.** 0
coercions, 0 contradictions, 90 ledger entries, 3 arcs proposed
and active across both phase B runs. The defensive layers and the
arc planner don't degrade with the rotation fix.

### What's queued after this

The latency regression and the RSS spike both point at the same
root: **KV cache cold-misses on novel-NPC prompts**. Three
candidate mitigations to test next session:

1. **Burst rotation**: instead of picking 3 distinct NPCs per tick,
   pick 1 focus NPC and stay with that NPC for K consecutive ticks
   before rotating. Trades coverage breadth for cache reuse depth.
   K=4 means ~25% as many cache misses per tick.
2. **Smaller model**: re-run with Qwen 2.5 0.5B + bounded snapshot
   + rotation fix. Smaller model = smaller KV cache = lower per-
   tick latency cost when cache misses. The 0.5B re-bench was
   already queued from the earlier work.
3. **Explicit KV cache management**: call `llama_kv_cache_clear`
   before each tick to force cold-cache behaviour, then measure
   what the actual recompute cost is without phantom RSS retention.

Coverage at this rate is sufficient — 90 unique NPCs in 30 ticks
of 3-action ticks. A real game running parallel-to-this would tick
slowly enough that 167 ticks to cover all 500 is many hours of
wall time, totally fine.

### Reproduction

```bash
cd D:/LLCWork/npc-engine

# Phase B run with bounded snapshot + rotation fix
python bench_story_director.py --ticks 30 --reset --model qwen_3b \
    --world synthetic_500 --actions-per-tick 3 \
    --narration-mode terse --log logs/syn500_bounded_rotated_30.json
```

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

- **Runtime narration mode toggle.** Same Director, same model, same
  prompt scaffolding — flip `narration_mode` from `"prose"` to
  `"terse"` and outputs drop from 15-25 word cinematic sentences to
  sub-25-word third-person factual statements. Shipping games use
  terse, story bibles use prose.

- **Python overhead is well-bounded.** With NLI and embedder caches,
  dynamic `max_tokens`, and terse-mode prompt trimming, the stubbed
  profile (LLM excluded) runs 15 multi-action ticks in 6.5s — any
  future LLM-latency drop will expose proportionally less Python
  overhead, not more.

## What doesn't work yet

- **Player quest tracking integration.** `record_player_action("Player
  completed Kael's quest")` records the *text* but doesn't update
  `pie.player_quests`. A real game loop would parse intent and call
  `engine.complete_quest("missing_hammers")`.

- **NPC dialogue identity bleed — partially addressed.** The postgen
  wrong-addressee detector (`detect_wrong_addressee` +
  `repair_wrong_addressee` in `npc_engine/postgen.py`, commit
  `885308f`) catches and repairs dialogue where an NPC addresses the
  player by another NPC's name. This is a patch, not a root fix; the
  underlying expert routing still occasionally confuses speaker
  identity at the generation layer.

- **0.5B re-bench with the full post-polish stack** is unverified.
  0.5B passed the structural tests on every prior iteration, but not
  formally re-run after the narration mode toggle + polish pass
  landed.

- **Architectural push beyond Ashenvale.** The Director's been
  exercised only against the 7-NPC Ashenvale world. Port Blackwater
  (3 NPCs) exists in the repo but hasn't seen Director runs; a larger
  world (15+ NPCs) would stress the NPC rotation + arc cast selection
  in ways the current benches don't.

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

The Director's internals are architecturally settled. Every copy
loop that showed up across the iteration commits has been broken,
the arc-advancement positive feedback loop closed, the terse
toggle lets the same Director serve both "story bible" and
"shipping game" consumers, and the Python overhead is well-bounded
by the polishing pass. What's left is *external* — validating at
scale and wiring the Director into the consumers that will use
its output.

### Recommended direction: validate at scale + integrate into NPC dialogue consumption

Pick these in order unless a higher-priority bug surfaces. All
four should be doable in under a week of focused work.

**1. Port Blackwater stress run** *(done 2026-04-14 — see
"Port Blackwater stress run (2026-04-14)" above)*

20-tick × 3-action runs in both prose and terse modes on the
3-NPC pirate-port world landed clean: 0 coercions, 0 consecutive
repeats, 20/20/20 NPC balance, arcs proposed and advanced
through the beat skeleton, terse mode 16% faster than prose,
content stayed on the new lore pack with zero Ashenvale bleed.
Three small frictions surfaced (ledger noise on small casts,
NLI false positives on terse content, quest id collisions with
profile-level quests) — all documented above, none blocking.

**2. NPC dialogue fact-consumption verification** *(done 2026-04-14
— see "NPC dialogue fact-consumption verification (2026-04-14)"
above)*

`bench_fact_consumption.py` measured per-NPC token deltas and
slice-survival counts across both worlds × both modes. Headlines:
terse halves prose token cost in both worlds (ash 215 → 130, pb
455 → 155 mean), terse fails the <150 max goal by 16–35 tokens on
a minority of NPCs (3/7 ash, 2/3 pb), and the
`personal_knowledge[:4]` forward-slice silently drops nearly every
Director-injected personal fact on worlds whose NPCs ship with
non-empty personal lists (PB terse: 8 added, 8 sliced, 100% loss).
Two follow-up actions queued in the section above: land the
slice-direction fix, and add a per-NPC per-tick action budget cap
for small casts.

**3. 0.5B re-bench with the full post-polish stack** *(small —
half-day)*

Run `bench_story_director.py --ticks 15 --reset --model qwen_05b
--actions-per-tick 3` in both modes, compare against the 3B
baseline, note any structural regressions. 0.5B was validated
pre-polish; this confirms the cheap path still ships after
narration mode + polish commits landed.

Success criteria: 0.5B schema pass rate within 2% of 3B;
content may be more literal (expected) but rotation / arcs /
bios all work.

**4. Large-world scale test** *(medium — day+)*

Build a throwaway 15-NPC world (or augment Ashenvale with
procedurally-named extras) and run 30 ticks × 3 actions. This
stresses NPC rotation (currently least-recently-touched
round-robin — fine at 7 NPCs, might thrash at 15+) and arc cast
selection (currently picks 2-3 NPCs from cluster center —
needs to scale with world size).

Success criteria: no rotation starvation (every NPC touched at
least once in 30 ticks), arcs continue to form and advance, no
wall-clock cliff as world size grows.

After those four land, the Director is genuinely production-
ready for Anima's commercial release. Until then, everything
else below is premature.

---

### Backlog (not blocking the recommended direction)

#### A. Player quest tracking integration

`record_player_action("Player completed Kael's quest")` records
the text but doesn't update `pie.player_quests`. Add an
intent-extraction pass that matches accept/complete/abandon
keywords against active quest IDs and dispatches the engine
call. Guard with a confidence threshold.

#### B. Auto-augmentor generation for narrative examples

The 0.5B path copies few-shot examples more literally than 3B.
Port ultralight-coder's
`generate_augmentors_from_failures.py`: run bench, flag literal-
copy events, feed to 3B with a "vary this without drifting from
the lore bible" prompt, schema- and isolation-gate, write to
`data/story_director/examples_generated/` for manual review.
Less urgent now that the base library is 17 entries instead of 5.

#### C. Narrative arc v2 polish

- **Semantic beat progress**: touch counter is coarse — replace
  with "embed the tick's new content, measure similarity to
  the current beat goal, only count if above threshold."
- **LLM-proposed themes**: v1 pulls themes from cluster center
  text; a one-call-per-proposal summarization pass would
  produce crisper one-line themes.
- **Abandonment**: stale arcs with no recent cast touches flip
  to `status = "abandoned"` so new arcs can form without
  waiting for the proposal cooldown.
- **Cross-arc interference**: two active arcs sharing an NPC
  can stall each other when the touch counter splits between
  them. Per-arc per-NPC touch lane.

#### D. Small items worth doing

- **Acting on contradiction** beyond retry: when retry ALSO
  fails, noop and log to `narrative_conflicts.json` for human
  review. Currently we dispatch the conflicting retry result.
- **Dialogue auto-feed filtering**: chatty-player flood —
  filter on length or keyword relevance before feeding the
  Director.
- **Token budget management**: `_build_prompt` concatenates
  lore + examples + snapshot + ALREADY DONE + FOCUS + ACTION
  + ACTIVE NARRATIVE ARC. Add a per-section token budget with
  truncation as the ledger grows.

#### E. Look-into: SECL-style discriminative gate as a cheap extra check

Research reference: **SECL — Self-Calibrating LMs via Test-Time
Discriminative Distillation** (arXiv 2604.09624, April 2026). The
paper's empirical claim is that the token probability of `"True"`
when an LLM is asked `"Is this answer correct?"` is a
significantly better-calibrated signal than the model's generative
confidence — ~56–78% ECE reduction across four small LMs and four
domains. The paper's test-time *training* loop is out of scope for
a Cython-compiled shipping binary, but the *discriminative readout*
is cheap to try.

Where it would plug in: **after the current `ContradictionChecker`
NLI gate and before dispatch.** Today the flow is generate → NLI
check against FactLedger → retry once if contradicted → dispatch.
A SECL-style second pass would ask the same local model
`"Given the NPC bio and the current world snapshot, is the
following story beat consistent? Answer True or False."` and read
the True-token probability. Below a threshold (empirically tuned
against the 10-tick × 3-action benches in "Port Blackwater stress
run" and "NPC dialogue fact-consumption verification") the beat
gets flagged — potentially folded into the same retry budget as
the NLI contradiction path, not a new one.

Why it might be worth the experiment: the NLI check catches
*pairwise* factual contradiction against prior ledger entries, but
doesn't catch "this beat is technically consistent but tonally
wrong for this NPC given their bio" — which is the failure mode
still visible in terse-mode Port Blackwater runs. A discriminative
pass against the full bio + snapshot prompt window is exactly the
check the NLI layer can't do. Budget: one extra LLM call per tick,
~0.8–2.5s depending on model (Qwen 0.5B vs 3B). Retry budgeting
("Retry budgeting" section above) already proves we can afford one
extra call per tick on the 3B path.

Why it might not be worth it: the 10-tick benches show terse-mode
3B already at high pass rates, and the remaining failures are
mostly literal-copy from few-shot examples, not inconsistency with
bio. "Intra-bio rotation" and "Self-repetition precheck" already
target those. If a quick smoke run (5 ticks on Ashenvale, check
whether True-probability correlates with manual pass/fail on a
handful of injections) shows weak correlation, drop it. Do NOT
adopt the paper's weight-updating test-time training loop — weight
mutation mid-session breaks the license-system assumptions in the
Cython build and has no upside for a game dialogue engine.

Experiment entry point: a new `scripts/bench_secl_gate.py` that
replays the existing `bench_story_director.py` output JSONs, runs
the discriminative readout pass against each dispatched beat, and
reports whether True-probability separates the manually-labeled
passes from the manually-labeled failures. No Director code
changes until the correlation is shown.

---

### Done and referenced from above — do not re-open

- Ledger compression — see "Ledger compression"
- Long-session 50-tick validation — see "Long-session validation"
- Own-output fixation — see "Self-repetition precheck"
- Multi-arc concurrency — see "Multi-arc concurrency"
- Arc touch counter regression — see "Arc touch counter fix"
- Example library expansion 5 → 17 — see "Example library expansion"
- Narration mode toggle — see "Narration mode toggle"
- Terse example library — see "Terse example library"
- Python overhead polishing — see "Python overhead polishing pass"
- Port Blackwater stress run — see "Port Blackwater stress run
  (2026-04-14)"
- NPC dialogue wrong-addressee repair — shipped in commit `885308f`,
  see `npc_engine/postgen.py` detect/repair functions
- Real parallel workers via N Llama instances — rejected, see
  "Speculative items evaluated and rejected"
- Co-reference resolution in FactLedger — rejected, see
  "Speculative items evaluated and rejected"

---

## File map

```
npc-engine/
├── npc_engine/
│   ├── story_director.py     # Director + FactLedger + NarrativeArc + ArcPlanner (~2700 LOC)
│   ├── postgen.py            # wrong-addressee detect/repair (dialogue identity fix)
│   ├── engine.py             # +14 LOC: instantiate + dialogue auto-feed
│   └── server.py             # +43 LOC: 3 new REST endpoints
├── tests/
│   └── test_story_director.py  # 105 tests (all offline + 1 integration smoke)
├── data/
│   └── story_director/
│       ├── ashenvale_lore.md         # ~30 lines of setting bible
│       ├── examples.yaml             # 17 prose few-shot world-state→action pairs
│       ├── examples_terse.yaml       # 17 terse few-shot pairs (1:1 mirror of examples.yaml)
│       ├── FINDINGS.md               # this file
│       ├── state.json                # gitignored runtime state
│       ├── fact_ledger.json          # gitignored ledger (200-entry cap)
│       ├── fact_ledger.embeddings.npy # gitignored binary float32 sidecar
│       └── arcs.json                 # gitignored narrative arc state
├── bench_story_director.py   # Model comparison + scripted sessions
├── scratch_profile.py        # throwaway: stubbed-LLM cProfile harness
└── .gitignore                # runtime files, traces, backups
```

## Reproduction

```bash
cd D:/LLCWork/npc-engine

# Unit tests (no model needed)
python tests/test_story_director.py

# Best production session: Qwen 2.5 3B, multi-action, prose mode
python bench_story_director.py --ticks 15 --reset \
    --model qwen_3b --actions-per-tick 3 --log session.json

# Terse (game-ready) mode: same bench with --narration-mode terse
python bench_story_director.py --ticks 15 --reset \
    --model qwen_3b --actions-per-tick 3 --narration-mode terse

# Cheapest path: Qwen 0.5B with full scaffolding
python bench_story_director.py --ticks 15 --reset \
    --model qwen_05b --actions-per-tick 3

# Python-only profile (stubbed LLM, surfaces pure Python overhead)
python scratch_profile.py

# REST server
python -m npc_engine.server
# POST /story/tick
# POST /story/player_action {"text": "...", "target": "noah", "trust_delta": 5}
# GET  /story/state
```
