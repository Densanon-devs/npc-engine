# Predictive FactLedger — Design Spec

**Status:** v1 IMPLEMENTED 2026-07-07 (branch
`feature/predictive-factledger`): `npc_engine/predictive_factledger.py`
+ StoryDirector wiring + `tests/test_predictive_factledger.py` +
`e2e_stress.py --scenario predictive_drift` + `verify_predictive.py`.
Documented deviations from this spec: sidecar filenames derive from
the state-file stem (`state.predictive.npz`, `state.activity_history
.jsonl`) so the existing STATE_FILE test-isolation pattern covers
them; a read-only `GET /story/predictive` observability endpoint was
added; the v1 edge kind is `npc_beat` (NPC receives a Director beat
at a tick-phase bucket) rather than the npc_in_zone/faction_presence
examples, whose observation streams don't exist yet; the arc boost is
a multiplicative nudge capped at 0.10, DEFAULT-OFF: the lane merged to
master observe-only on 2026-07-07 (Story Director frozen), so the nudge
is inert unless `NPC_ENGINE_PREDICTIVE_BOOST=1` is set explicitly
(boosted e2e cycles verified clean before the flip: predictive_drift
13/13 x2 + gameplay 41/41 x2). Original draft
2026-05-04; v2+ roadmap refinement 2026-05-10. v1 (pure-Python,
single linear prior) was the implementation target; v2/v3 (ONNX
sequence model, multi-expert split) remain roadmap, sketched in the
"2026-05-10 design refinement" section near the bottom.

This document specifies an enhancement to the existing Story Director
that turns the FactLedger from a passive **record** into a passive
**record + cheap forward-predictor**. The predictive pass biases arc
proposal and activity-context guesses *before* the game client's next
`POST /story/activity` arrives — closing the round-trip gap between
"world state changed" and "Director knows it changed."

The two underlying ideas are pulled from 2026-05-04 robotics research
(see `~/.claude/projects/D--LLCWork/memory/project_story_director.md`
2026-05-04 entries):

- **Being-H0.7 prior/posterior dual-branch** (arXiv 2605.00078) — a
  posterior branch trained offline aligns future observations into
  latent queries; a prior branch at inference predicts those latents
  from current context alone, with zero runtime overhead beyond a
  matrix multiply. We use the prior branch at tick time only.
- **Predictive Spatio-Temporal Scene Graphs / Perpetua\*** (arXiv
  2605.00121) — Bayesian filters live on the **edges** of a scene
  graph and learn cyclic real-world patterns; predictions are
  edge-local, robust to distributional shift over weeks of data. We
  put one filter on each FactLedger relationship type that is known
  to be cyclic (NPC-in-zone, faction-presence, time-of-day quest
  preferences).

Phase 4a is **already shipped** (`PlayerActivity` enum +
`POST /story/activity`); this spec does NOT rewrite Phase 4a — it
adds a parallel predictive lane that produces a *prior* the existing
machinery can fall back on or override.

## Why this is worth doing

Today's FactLedger is reactive. The Director picks an action kind
(`event` / `quest` / `fact`) on a fixed round-robin and biases arc
selection from the most-recent ledger entries. Two failure modes
this leaves on the table:

1. **Activity-context lag.** When the player transitions from
   `in_town` → `in_dungeon`, the Director keeps proposing tavern
   events for one or two ticks until the client pushes a new
   `/story/activity`. The cycle is short and observable — by tick
   N+1 the client always catches up — but the wasted-tick beat is
   the immersion break.
2. **Predictable-cycle lock-in.** Tavern crowd levels follow
   time-of-day; a market follows a weekly trade pattern. The
   Director can't *anticipate* the shift, so arcs that would land
   well one tick later get proposed one tick too early.

Both failures are exactly the "semi-static environment" pattern the
Perpetua\* paper targets, and exactly the "predict next latent state
without generating it" pattern Being-H0.7 targets.

## What this is NOT

- **Not** a model-training project. No fine-tuning. No GPU training.
  Both the Perpetua\* edge filters and the Being-H0.7 prior
  projection are tiny enough to learn from FactLedger history on the
  CPU at startup or in a background thread.
- **Not** a Phase 4a rewrite. Phase 4a's `PlayerActivity` enum and
  the activity-tick gating in `_pick_action_kind` stay as-is. The
  predictive lane is additive.
- **Not** a public API change. No new endpoints. The prior branch
  fires inside `tick()`, before the existing kind-rotation and arc
  proposal calls.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  StoryDirector.tick()                                          │
│                                                                │
│  1. Existing flow:                                             │
│       _pick_focus_npc()  → focus                               │
│       _pick_action_kind() → kind                               │
│       _propose_arcs(ledger, focus, kind) → arc                 │
│       LLM writes the beat                                      │
│                                                                │
│  2. NEW: predictive lane (parallel to step 1):                 │
│       PredictiveLayer.predict_next(ledger, tick, activity)     │
│         → ActivityPrior  (Being-H0.7 prior latent rollup)      │
│         → EdgePriors     (per-edge Perpetua* posterior at t+1) │
│                                                                │
│  3. Bias step:                                                 │
│       - If ActivityPrior disagrees with the most-recent        │
│         /story/activity by > confidence threshold, log a       │
│         "predicted_drift" warning (no behavior change yet —    │
│         we want to observe before acting). Future iteration:   │
│         bias _pick_action_kind toward the predicted activity.  │
│       - Pass EdgePriors into _propose_arcs as an optional      │
│         scoring boost — arcs that align with a high-confidence │
│         edge prior get a soft preference.                      │
└────────────────────────────────────────────────────────────────┘
```

### Predictive layer module

New file: `npc_engine/predictive_factledger.py` (~250-350 LOC). No
external dependencies beyond what FactLedger already imports
(numpy, json). Three classes:

```python
class EdgeFilter:
    """One Perpetua*-style Bayesian filter per cyclic edge.

    State: per-bucket (time-of-day, day-of-week, etc.) Beta
    distributions over the edge's truth probability. Update is a
    one-line conjugate posterior on each new ledger observation.
    Predict at t+1 returns the bucket's mean.
    """
    edge_kind: str  # "npc_in_zone" | "faction_presence" | ...
    bucket_dim: str  # "tod_quartile" | "weekday" | ...
    counts: dict[tuple, tuple[int, int]]  # (success, fail) per (key, bucket)

    def update(self, key: tuple, bucket: int, observed: bool) -> None: ...
    def predict(self, key: tuple, bucket: int) -> float: ...


class ActivityPrior:
    """Being-H0.7-style prior projection from FactLedger latents to
    the next-tick player activity.

    Learns offline (or background-warm) from
    (recent_ledger_summary_vector, observed_next_activity) pairs.
    At inference: project the current ledger's recent-window summary
    vector through a single linear matrix W to logits over the
    PlayerActivity enum. No model call. No GPU.
    """
    weight_matrix: np.ndarray  # shape (latent_dim, len(PlayerActivity))
    label_map: list[str]  # PlayerActivity values, in column order

    def fit(self, history: list[tuple[np.ndarray, str]]) -> None: ...
    def predict(self, latent: np.ndarray) -> dict[str, float]: ...


class PredictiveLayer:
    """Owns the edge filters and the activity prior. One instance per
    StoryDirector. Hot path is < 1 ms — pure linalg, no I/O."""

    def __init__(self, ledger: FactLedger, profiles_dir: Path): ...
    def predict_next(
        self,
        ledger: FactLedger,
        tick: int,
        current_activity: str,
    ) -> tuple[ActivityPrediction, dict[str, float]]: ...
    def record_observation(self, tick: int, activity: str, ledger_delta: list[FactEntry]) -> None: ...
```

### Integration points (minimal)

```
npc_engine/story_director.py
  StoryDirector.__init__:
    self._predictive = PredictiveLayer(self.ledger, self._story_dir)

  StoryDirector.tick:
    + (just before _pick_focus_npc)
    +   activity_pred, edge_priors = self._predictive.predict_next(
    +       self.ledger, self.tick_count, self._player_activity,
    +   )
    +   self._last_activity_pred = activity_pred  # for /story/state
    + (during arc proposal)
    +   arcs = self._propose_arcs(..., edge_prior_boost=edge_priors)

  StoryDirector._propose_arcs:
    + accept optional edge_prior_boost: dict[str, float]
    + apply as a small additive score (cap < 0.10 of total) so the
      prior never dominates the existing greedy-cluster heuristic
```

State serialization piggybacks on the existing
`fact_ledger.embeddings.npy` sidecar pattern — the edge-filter
counts and the prior weight matrix get a separate `predictive.npz`
sidecar in the same `story/` directory, loaded at init, saved on
graceful shutdown (and on tick if the existing autosave hook fires).

### Data flow during the warm-up window

The first time the predictive layer runs against a fresh
`fact_ledger.json`, neither the edge filters nor the prior matrix
have any data. Three behaviors:

1. **Edge filter cold start.** Beta(1, 1) priors → every prediction
   returns 0.5. The arc-proposal boost is `0.5 - 0.5 = 0` for every
   candidate, so cold edges literally do nothing. The system reduces
   to today's behavior until enough ledger entries accumulate to
   move the posteriors.
2. **Activity prior cold start.** No `weight_matrix` → the predictive
   layer returns a uniform distribution and `_last_activity_pred` is
   marked `cold=True`. The bias step skips the drift check.
3. **Background warm pass.** A `PredictiveLayer.warm_from_history()`
   call replays the existing `fact_ledger.json` + a new
   `activity_history.jsonl` (which we'd need to start writing — see
   "Schema changes" below) and trains both the edge filters and the
   prior matrix. This is the only "training" step; it runs on the
   CPU and finishes in seconds for any realistic ledger size.

### Schema changes

One new file:

- `data/story_director/activity_history.jsonl` — append-only.
  Each line: `{"tick": int, "activity": str, "ts": iso8601}`. Written
  on every `/story/activity` POST. Used as the supervision signal for
  the `ActivityPrior.fit` call. ~80 bytes/entry, negligible.

One new sidecar (mirrors `fact_ledger.embeddings.npy`):

- `data/story_director/predictive.npz` — saved on graceful shutdown.
  Contains `edge_counts` (a dict-as-arrays serialization) and
  `prior_weight_matrix`. Re-derivable from the JSONL + ledger if
  deleted, so it's purely a startup-cost optimization.

Both files are gitignored, same as `state.json` and the existing
ledger files.

## Composition with existing systems

| Existing | Interaction with predictive layer |
|---|---|
| `FactLedger.add_entry` | Predictive layer subscribes; updates edge filters in `record_observation` |
| `FactLedger.surface_similar` | Unchanged — predictive layer reads the ledger but doesn't modify the similarity surface |
| `ContradictionChecker` (NLI) | Unchanged. NLI catches semantic contradictions; predictive layer catches *temporal/cyclic* drift. They're orthogonal. |
| `PlayerActivity` enum (Phase 4a) | Predictive layer outputs distributions over the same enum so labels align |
| `_pick_focus_npc` | No change in v1 (observe drift only, no behavior change) |
| `_pick_action_kind` | No change in v1 (observe drift only) |
| `_propose_arcs` | Optional `edge_prior_boost` kwarg, capped at < 0.10 of total score |
| Quest pacing (Phase 4b) | Unchanged. Pacing is an NPC-side cap; predictive layer is a global prior |
| Game-reset flow | `PredictiveLayer.reset()` zeroes the edge filters but keeps the prior matrix (the prior generalizes across resets — the edge filters don't) |
| FactLedger embeddings sidecar | Same pattern, separate file. No collision. |

## Validation strategy

Three levels of test, all offline (no GPU, no game client):

1. **Unit (pure math)** — `tests/test_predictive_factledger.py`:
   - `EdgeFilter.update + predict`: beta-binomial conjugate update
     produces correct posterior means on synthetic 2-bucket cycles.
   - `ActivityPrior.fit + predict`: linear projection trained on a
     fixture mapping (random latent vector, label) recovers the
     label distribution within ε.
   - `PredictiveLayer.predict_next` with empty ledger returns
     uniform + `cold=True`.
2. **Integration with FactLedger** — extend `test_story_director.py`:
   - `tick()` runs to completion with the predictive layer enabled.
   - The drift-check warning fires when a cyclic synthetic ledger
     diverges from the predicted activity.
   - Graceful shutdown writes `predictive.npz`; restart loads it
     without re-fitting.
3. **End-to-end stress** — extend `e2e_stress.py` with a
   `predictive_drift` scenario:
   - 30-tick scripted cycle that pushes activity transitions on a
     fixed cadence; assert that drift warnings start firing within
     5 ticks of the predictive layer accumulating enough history.
   - Compare arc-selection distribution with vs without the
     `edge_prior_boost` enabled. Expected: no headline pass-rate
     regression on `e2e_stress.py gameplay`, plus a small
     improvement on a new `cyclic_consistency` scenario.

Convergence standard mirrors the rest of npc-engine: two consecutive
clean stress passes before merging. No model bench (Qwen 2.5 3B)
needed in v1 — the predictive layer is pure-Python.

## Risk register

| Risk | Mitigation |
|---|---|
| The learned prior overfits to one player's session and ships in `predictive.npz` | The sidecar is gitignored. The prior matrix is per-engagement, not per-distribution. |
| Edge filter cold start makes early ticks noisy | Beta(1,1) prior + `cold=True` flag + capped boost magnitude (< 0.10) means cold filters are silent, not noisy. |
| The drift-check warning floods logs | v1 uses `logger.debug`. Promote to `logger.info` only after the threshold tuning settles in stress runs. |
| `activity_history.jsonl` grows unbounded over a multi-month engagement | Cap at last 90 days at warm-from-history time; older entries are pruned. ~30k lines/engagement worst case = ~2.5 MB. |
| Predictive layer drifts from FactLedger schema (new entry kinds added) | `EdgeFilter` registers per `edge_kind`; unknown kinds are ignored, not crashed. New kinds need an explicit registration call. |

## Out of scope for this spec

- Per-NPC predictive priors (current spec is global). Could become
  Phase N+1 after the global prior is proven.
- Cross-world prior transfer (Ashenvale → Port Blackwater). Each
  world keeps its own `predictive.npz` until there's a reason not to.
- LLM-side integration (giving the model the predicted activity in
  the prompt). v1 keeps the prior in Python only — model behavior
  stays exactly as today.
- Story Director's `_propose_arcs` deeper integration with edge
  priors beyond the additive boost (e.g. proposing arcs from edge
  predictions directly). Reachable in v2 if v1's drift-check data
  proves the priors are well-calibrated.

## 2026-05-10 design refinement — deployment architecture + multi-expert split

Three robotics-research findings from the 2026-05-10 daily digest
(see `~/.claude/projects/D--LLCWork/memory/project_story_director.md`
2026-05-10 entry) refine — not replace — the v1 design above. They
matter when the `ActivityPrior` graduates from "single linear matrix"
toward something with more capacity, and they give a concrete
deployment path that keeps the hot path cheap.

### Bimo RP2040 ONNX deployment — the serving architecture

**Source:** Bimo open-source bipedal robot
(r/robotics 2026-05-10, https://www.reddit.com/r/robotics/comments/1t968vj/).
A locomotion policy trains in Isaac Lab (2048 parallel envs, <15 min
on a modern GPU), exports as ONNX, and runs natively on a $4 RP2040
microcontroller at ~5.2 ms inference / 20 Hz control. The pattern:
**train the small thing offline, export to ONNX, deploy it next to
(not in place of) the big model.**

Mapped onto Story Director, this is a *two-tier serving design*:

| Tier | Component | Frequency | Latency budget | Decides |
|---|---|---|---|---|
| 1 (cheap) | ONNX sequence predictor — the `ActivityPrior` + edge filters, compiled | every tick | < 50 ms (target; Bimo hits 5 ms on far weaker hardware with a bigger model) | next-tick activity prior + edge-prior boosts for `_propose_arcs` |
| 2 (expensive) | Qwen 2.5 3B (the existing worker LLM) | only on threshold-crossing decisions | ~2.5 s/tick (current) | dialogue text, multi-action arc planning, anything creative |

The v1 spec already keeps the predictive layer at "< 1 ms — pure
linalg, no I/O." Bimo's contribution is the *export path* for when
the `ActivityPrior` outgrows a single `np.ndarray` matmul: instead
of carrying a heavier model in Python, export it to ONNX and run it
through `onnxruntime` (CPU execution provider — no GPU). The
`BimoAPI/` repo (`scripts/` directory) is a directly-readable
reference for the ONNX export + quantization steps. The Story
Director predictor is far smaller than Bimo's locomotion policy and
will fit in a few hundred KB after INT8 quantization — small enough
that the `predictive.npz` sidecar (see "Schema changes" above) could
become a `predictive.onnx` sidecar without changing the deployment
footprint meaningfully.

**Action when v1's `ActivityPrior` proves out:** swap the
`weight_matrix` matmul for an `onnxruntime.InferenceSession` with the
CPU provider. Keep the sidecar pattern. No GPU dependency at runtime.
The fast-tier still finishes well inside the < 50 ms budget.

### Rhoda AI Direct Video Action — the training paradigm

**Source:** Rhoda AI (The Robot Report 2026-05-09,
https://www.therobotreport.com/why-traditional-robotics-data-collection-is-obsolete-and-what-replaces-it/).
DVA models train on publicly-available internet *video* rather than
robot-collected datasets and still transfer zero-shot to real
manipulation. The load-bearing insight: **ordered frames implicitly
encode causal sequences and physics priors** — a model trained on
"video" learns world dynamics without ever touching a robot.

The FactLedger's append-only event stream is *exactly* an ordered
causal sequence — a "video" of the world's narrative state. The v1
spec's `ActivityPrior.fit(history)` already consumes
`(recent_ledger_summary_vector, observed_next_activity)` pairs; the
DVA framing sharpens this: treat the *whole ledger stream* as the
training signal (next-N-entry prediction), not just the summary
vector → next activity mapping. The model predicts the next K
FactLedger deltas given the current state, and the activity prior
falls out as one head on that sequence model. This is the "treat the
log as video" reframe.

**Action:** when the `warm_from_history()` pass is built (step 4 in
the implementation order), have it train a small sequence model
(LSTM or a 2-layer transformer — both ONNX-exportable, both CPU-
trainable in seconds for realistic ledger sizes) on next-K-delta
prediction over the full ledger stream. The activity-prior logits
become one output head; the edge-prior boosts another. Single model,
two heads — replaces the v1 "single linear matrix + separate beta
filters" split if it benches better. (Keep the v1 split as the
fallback if the sequence model doesn't beat it — the v1 design's
whole virtue is that it can't be worse than today.)

### Multi-expert distillation — the fusion architecture

**Source:** Boston Dynamics Spot (IEEE Spectrum Video Friday
2026-05-09, https://spectrum.ieee.org/video-friday-robotic-hand-dexterity).
Spot's controller trains multiple specialist policies independently,
then compresses them into a single deployable network via a fusion
head — multi-expert distillation.

The v1 spec's `ActivityPrior` is one monolithic predictor. As the
predictive lane matures it will want to predict across distinct
narrative domains that have different dynamics:

- **Pacing** — how soon the next arc beat should land (depends on
  recent beat density, player engagement signals).
- **NPC motivation** — which NPC is "due" for a focus turn (depends
  on round-robin state + recent player-target signals + relationship
  trajectory).
- **World economy** — market/resource cycles, faction-presence
  cycles (the Perpetua\* edge filters already cover the cyclic
  subset of this).

Training one model to predict all three is harder to debug than
training three small specialists + a lightweight fusion head that
merges them at inference. The fusion head is cheap (a small linear
layer over the concatenated specialist outputs); each specialist
stays small enough to retrain independently as a world's content
evolves (a game studio adds new NPCs → retrain the NPC-motivation
specialist, leave the other two).

**Action:** allocate the v2 predictive-layer work as N domain
specialists (pacing / NPC-motivation / economy) + 1 fusion head,
rather than one end-to-end predictor. Each specialist is ONNX-
exportable; the fusion head merges them. Composes with the DVA
sequence-model entry above (each specialist can itself be a small
sequence model over its domain's slice of the ledger) and the
2026-05-08 DexSim2Real VLM-critic entry in the project memory (the
closed-loop critic adjusts each specialist's calibration
independently when predicted vs observed deltas diverge).

### Net effect on the spec

None of this changes v1's scope or risk profile. v1 ships first
(pure-Python, single matrix, can't be worse than today). The
refinements above are the *v2+ roadmap*:

- v1: single linear `ActivityPrior` matrix + Perpetua\* beta filters,
  pure NumPy, `predictive.npz` sidecar. (Implementation order below.)
- v2: replace the matrix with an ONNX sequence model trained DVA-style
  on the full ledger stream. `predictive.onnx` sidecar. Still CPU-
  only, still < 50 ms hot path.
- v3: split into pacing / NPC-motivation / economy specialists +
  fusion head, each ONNX-exportable, each independently retrainable.
  Add the DexSim2Real-style closed-loop calibration critic.

The "ledger is a causal-sequence video" reframe is the through-line:
once that lands in v2, v3's specialist split is a natural
decomposition of the sequence-prediction task by narrative domain.

## Implementation order if/when greenlit

1. Land the schema: write `activity_history.jsonl` on every
   `/story/activity` POST. No other behavior change. Ship + observe
   for one playtest cycle to confirm the data is clean.
2. Land `predictive_factledger.py` with `EdgeFilter` + unit tests.
   `ActivityPrior` stub returns uniform.
3. Wire `PredictiveLayer` into `StoryDirector.__init__` + `tick()`,
   drift-check at `logger.debug`. No `edge_prior_boost` yet.
4. Land `ActivityPrior.fit + predict` with the warm-from-history
   pass. Drift-check at `logger.info`.
5. Land `_propose_arcs(edge_prior_boost=...)` integration. Run two
   consecutive `e2e_stress.py` cycles; promote to default-on if
   clean.

Total scope estimate: ~600-800 LOC + ~200-300 LOC tests. No model
training. No new dependencies. No public API changes.

## Design input added 2026-05-20 — KG-ASG primary-support attribution

Source: KG-ASG (arXiv 2605.18895, "Collision-Knowledge-Guided
Closed-Loop Adversarial Scenario Generation With Primary-Support
Attribution"). The paper is autonomous-driving scenario generation,
but its event-descriptor schema and single-collider constraint map
cleanly onto the Story Director tick. Filed here as design input for
the predictive-lane arc-proposal step (not the v1 implementation
target above — this informs how arcs are *shaped* once the prior
biases which arcs to consider).

**The schema mapping.** KG-ASG's "Collision Expert" (a small fine-tuned
LLM) emits a structured descriptor before any scenario is generated:

```
KG-ASG:           (collision_mode, primary_adversary_idx,
                   support_vehicles[≤2], conflict_time_window,
                   behavior_template)

Story Director:   (event_kind,      primary_actor,
                   support_actors[≤N], tick_window,
                   guidance/arc_template)
```

This is the *same shape* as the existing "Python plans, LLM writes"
split — `_pick_focus_npc` already chooses the primary actor and
`_pick_action_kind` chooses `event/quest/fact`. KG-ASG's contribution
is the **explicit support-actor slot + single-primary causal
constraint**, which the current tick descriptor does not carry. When
`_propose_arcs(edge_prior_boost=...)` shapes a multi-action beat, it
should emit this full descriptor, not just (focus_npc, action_kind).

**The single-collider constraint → single-primary-agent FactLedger
rule.** KG-ASG enforces `Σ_j I{Collide(o_j, o_0)} ≤ 1` so every
generated scenario has exactly one cause and unambiguous attribution;
this drove their multi-collision rate from 43.6% → 0.00%. The
narrative analogue is a FactLedger validation rule:

> Each narrative event has exactly ONE primary actor whose action
> *causes* the world-state transition recorded in the ledger. Support
> actors may only *modulate context* (witness, react, color the scene)
> — they may not author additional state transitions in the same beat.

This is a concrete, testable invariant for the predictive lane: when
the prior boosts a multi-NPC arc, the dispatch path must tag exactly
one `primary_actor` on the resulting `FactLedger` entry and mark any
other involved NPCs as `support` (≈ the existing `witness_npcs` field
on `/story/player_action`, generalized to Director-authored events).
Prevents the "who actually did this?" causal tangle that makes
long-run ledgers contradict themselves — and it composes with the
existing `ContradictionChecker` NLI pass (clear single-cause
attribution makes contradiction classification easier, not harder).

**The closed-loop retry → existing `_enforce_*` + NLI-retry path.**
KG-ASG uses 5 retry rounds with *failure-type-specific* retry profiles
(timing / steering / brake-delay variants), lifting valid-primary-attack
from 68.8% → 92.2%. Story Director already has the substrate: the
`_enforce_*` schema-override methods + the NLI contradiction retry.
The KG-ASG refinement is to make the retry **failure-type-specific** —
when an NLI contradiction fires, branch the retry prompt on *what kind*
of failure it was (contradicts a prior fact / wrong primary actor /
support actor authored a transition), rather than a single generic
"try again." Bounded retry count (KG-ASG uses 5; Story Director should
cap lower given the per-tick latency budget — suggest 2-3) then
surface/skip on exhaustion.

**Action when the predictive lane is built:**
1. Extend the tick descriptor to `(event_kind, primary_actor,
   support_actors, tick_window, guidance)`.
2. Add the single-primary-agent invariant as a `FactLedger`
   validation rule + a unit test that rejects a beat tagging two
   primaries.
3. Make the NLI-retry path failure-type-specific (2-3 bounded rounds).

Composes with the SKG-Eval geometric contradiction engine (filed in
`project_story_director.md` 2026-05-19 — the *post-hoc* consistency
scorer; KG-ASG is the *generation-time* attribution constraint) and
the SDOF FSM pre-condition gate (2026-05-18 — pre-hoc dispatchability
check). Pipeline ordering: SDOF gate → KG-ASG attribution → LLM writes
→ SKG-Eval/NLI scores → failure-type-specific retry.

## Design input added 2026-05-21 — SUGAR candidate-state pool + coherence scoring

Source: SUGAR (arXiv 2605.20373, "Scalable Human-Video-Driven
Generalizable Humanoid Loco-Manipulation"). The paper is humanoid
robot skill learning, but two of its architectural devices transfer to
the predictive lane. Filed as design input for *how the predictive
prior is represented and committed*, not the v1 target above.

**Candidate-state pool instead of a single committed prediction.**
SUGAR's middle stage maintains a *progressive state pool* — a set of
candidate states refined against a coherence prior — rather than
greedily committing to one trajectory, which is what lets it convert
noisy human-video priors into physically feasible motion. The
narrative analogue: the predictive layer should hold N candidate
near-future world-states (each a small bundle of likely-next
FactLedger entries) and *score* each against a coherence prior, rather
than emitting a single predicted next beat. This directly hardens the
warm-up window described under "Data flow" above — instead of one
prior that biases arc proposal, carry a small ranked pool (suggest
3-5) so a low-coherence top candidate can lose to a runner-up without a
re-prediction round-trip.

**The coherence prior is already in the tree: the NLI pass.** We do
not need a new scorer. The existing `ContradictionChecker` NLI pass
*is* the coherence prior — score each candidate state by how few
contradictions it introduces against the committed FactLedger (and,
optionally, how well it satisfies open quest/arc obligations). The
candidate with the best coherence score is promoted; the others are
retained as fallbacks for the failure-type-specific retry path (see
the 2026-05-20 KG-ASG entry above — a rejected primary-attribution beat
can fall back to the next candidate in the pool instead of a cold
re-generation).

**Hierarchical command→tracking split = the existing "Python plans,
LLM writes" split.** SUGAR distills to a two-tier policy (high-level
command generation → low-level tracking). Story Director already has
this shape (`_pick_focus_npc`/`_pick_action_kind` → LLM realizes the
beat). No change needed here — noting it only because it confirms the
candidate-pool device sits cleanly at the *command-generation* tier
(pool of candidate beats) and leaves the realization tier untouched.

**Action when the predictive lane is built:**
1. Represent the predictive prior as a ranked pool of 3-5 candidate
   next-states, not a single prediction.
2. Reuse `ContradictionChecker` NLI as the coherence scorer over the
   pool; promote the lowest-contradiction candidate.
3. Wire pool fallback into the KG-ASG failure-type-specific retry: on
   rejection, advance to the next pooled candidate before re-generating.

## Empirical validation added 2026-05-28 — GE-Sim 2.0 Narrative Judge transfer

Built a feasibility prototype validating the GE-Sim 2.0 World Judge
architectural pattern as a *coherence scorer* over candidate world states.
Prototype files at npc-engine repo root (NOT committed to npc_engine/):
`narrative_judge_prototype.py`, `narrative_judge_dataset.py`,
`narrative_judge_dataset_pb.py`, `narrative_judge_trainer.py`,
`narrative_judge_cross_domain.py`.

**Architecture (mirrors GE-Sim 2.0 §World Judge):**
- Frozen backbones: `cross-encoder/nli-deberta-v3-small` (already loaded
  for ContradictionChecker) + `all-MiniLM-L6-v2` (already in FactLedger).
- Trainable head: sklearn LogisticRegression on an 8-dim feature vector
  per `(fact, quest_objective)` pair — NLI three-class scores, embedding
  cosine, token overlap, length, two interaction terms.
- Loss: cross-entropy, 3-class (advance / block / neutral).
- Supervision: 216 hand-labeled tuples across Ashenvale (144) + Port
  Blackwater (72).

**Numbers:**

| Configuration | LR accuracy |
|---|---:|
| Raw NLI baseline (no training) | 0.241 |
| Single-domain Ashenvale (80/20 holdout) | **0.897** |
| Zero-shot Ashenvale → Port Blackwater | 0.708 |
| Zero-shot Port Blackwater → Ashenvale | 0.792 |
| Few-shot Ashenvale + 80% PB → 20% PB held-out | 0.800 |

**Findings:**
1. **+65.5pp lift over raw NLI** — the training-on-labels step is doing
   the work GE-Sim 2.0's MLP head does in robotics.
2. **Generalizes cross-domain.** Zero-shot 70.8% Ashenvale → PB despite
   totally different setting/vocabulary/quests.
3. **LR beats MLP at this dataset size.** Don't reach for deeper heads
   without ≥500 labels.
4. **Block-class is the weak link.** The Cosmos simulate-then-commit
   pattern (filed 2026-05-22) needs reliable contradiction detection;
   pair this Judge's advance-scoring with the *existing*
   ContradictionChecker for block-detection. Two-signal pipeline,
   not single classifier.
5. **Few-shot cheap.** +57 PB labels lifts held-out PB acc from
   70.8% → 80.0% (+9.2pp). **~50 labels per new world** is enough for
   production-quality per-world adaptation. ~25-50 min of authoring.

**How this slots into the spec:**

This Narrative Judge does NOT replace v1's `ActivityPrior` or `EdgeFilter`
— those address *next-tick activity prediction*. The Narrative Judge
addresses *candidate-state coherence scoring*, which the 2026-05-21
SUGAR entry and the 2026-05-22 Cosmos entry both call for. Composes as:

- **v1 (still as-spec'd above):** EdgeFilter + ActivityPrior produce
  prior + edge boosts → biases `_propose_arcs` and flags activity drift.
- **v2 (per the 2026-05-10 refinement):** ONNX sequence model
  generalizes the prior; predictive lane emits candidate pool.
- **v3 (Narrative Judge integration, NEW):** Trained Judge scores each
  candidate against active quest specs. Recommended pipeline per quest:
  `Judge.predict_proba(fact, quest_spec) → P(advance)` AND
  `ContradictionChecker.check(fact, quest_spec) → contradicts/entails/neutral`.
  Promote candidate that maximizes mean advance-prob minus weighted
  contradiction-prob across active quests.

**Implementation order if/when greenlit:**

After v1's `ActivityPrior` and `EdgeFilter` ship, the Narrative Judge
addition is:
1. Add `npc_engine/narrative_judge.py` (~150 LOC: featurize + load
   sklearn model + predict_proba interface). Copy from
   `narrative_judge_trainer.py::featurize` and the LR model export.
2. Train the model offline on the existing 216 tuples plus any new-world
   labels. Pickle to `data/<world>/narrative_judge.pkl`. ~30 sec
   sklearn fit. **Important:** pickle is per-world per the few-shot
   finding — Ashenvale and PB get distinct models, OR a combined model
   with both worlds in the training set.
3. Wire into `StoryDirector._propose_arcs` candidate-scoring path (when
   the SUGAR candidate pool exists). Score each candidate against each
   active quest; aggregate per quest, pick best.
4. Test against the existing 50-tick replay corpus (`narrative_judge_prototype.py::REPLAY_ENTRIES`).
5. Validate per-world separately — 50 hand-labels per new world is the
   target authoring budget.

**Key risks (verified, not theoretical):**
- Block-class accuracy drops in zero-shot (F1 0.49 PB, 0.61 reverse).
  Two-signal pipeline (Judge + ContradictionChecker) mitigates.
- 144 + 72 = 216 labels is small. MLP overfits; LR is the right
  capacity. Watch for over-claim if scaling beyond this.
- Hand-labels from a single author (no inter-rater check). Validate
  with at least one independent re-label of ~30 random tuples before
  shipping commercially.

Premise caveat (why this is "input," not "must-do"): the paper's
"smaller models benefit most from decomposition" framing did NOT hold
in a sibling ALM experiment (the parked PExA decomposed-parallel branch
regressed its 3B gauntlet -0.096 — "3B needs full context"). The pool
device is cheap here because it reuses the NLI scorer and adds no
context-splitting, but validate that maintaining N candidates doesn't
blow the per-tick latency budget on the local Qwen 2.5 3B before
committing to a large pool.
