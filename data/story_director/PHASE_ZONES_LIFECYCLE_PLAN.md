# Story Director — Zone Layer + NPC Lifecycle Plan

Finalized 2026-04-15. Scoped and signed off by Jordan. This doc is
the canonical plan for the zone + lifecycle expansion; progress and
bench results are written back into `FINDINGS.md` as each phase
ships. Feature branch: `story-director-zones` (off `story-director`
at `9df63bd`).

## Design invariants

- **Modular by design.** Every feature is gated so existing tests
  and benches keep working unchanged. Worlds without zone config
  run in world-wide mode. Worlds without lifecycle config never
  emit birth/death actions. Phase-1-only games can ignore Phase 2
  entirely.
- **Game-client-authoritative by default; Director-autonomous as
  opt-in.** The game client owns canonical state (who's alive,
  who's in which zone) unless a config flag says otherwise.
- **Deceased state is simple.** `status: deceased` flag in the
  profile YAML, deceased NPCs skipped in roster iteration, cleanup
  threshold deferred to later.
- **Dev-only system tags.** Players never see
  `generated: true` / `birthed_at_tick` / `arc_promotion_source`.
  These are strictly for bench + audit visibility.
- **Roles are pools, not slots.** A zone's `role_pool` is a source
  of role templates for birth; it doesn't enforce uniqueness. Two
  merchants in one zone is fine. The pool is experimental — we'll
  tune it based on how the bench plays out.

## Phase 1 — zone layer (read-only)

### Data model

**NPC profile YAML (static):**
```yaml
identity:
  name: "Aldric"
  role: "blacksmith"
zone: "dock_district"     # NEW: default zone, "global" if missing
mobile: false              # NEW: can current_zone change at runtime?
```

**NPCKnowledge runtime (dynamic), three in-sync copies:**
```python
home_zone: str = "global"       # from profile, immutable
current_zone: str = "global"    # mutable if mobile=True
mobile: bool = False
```

`current_zone` persists in per-NPC state files (reload after
restart keeps the traveling merchant where the game left them).

**StoryDirector state:**
```python
self._active_zones: set[str] = set()   # empty = world-wide mode
```

Persisted in `state.json`. Empty set = all existing benches and
tests run unchanged.

### Behavior

- **Focus selection soft weighting.** When `_active_zones` is
  non-empty, `_pick_focus_npc` partitions available NPCs into
  in-zone / out-of-zone. Picks from in-zone 6 of 7 times; picks
  from out-of-zone 1 of 7 (tunable via `_OUT_OF_ZONE_RATE`). The
  out-of-zone pick feeds "distant rumor" content that propagates
  via gossip. Stationary zones still surface "the world beyond"
  at a low bandwidth.
- **Quest action kind hard filter.** If `_active_zones` is
  non-empty AND the picked focus NPC is out-of-zone AND the
  kind rotation landed on `quest`, downgrade to `event` or
  `fact`. Out-of-zone NPCs can seed rumors but can't offer
  quests the player can't reach. Dead-end quests are realistic
  (especially when the giver dies post-Phase-2a), but
  unreachable ones are not.
- **Bounded snapshot zone tier.** `_world_snapshot` priority
  order becomes:
  1. Planned focus NPCs
  2. **Active-zone NPCs** (NEW tier)
  3. Arc cast NPCs
  4. Recent decisions
  5. Recent player targets
  On a 500-NPC world with one active zone, the snapshot
  reliably surfaces in-zone NPCs under the 16-NPC cap.
- **NPC membership check.**
  ```python
  def _npc_in_active_zone(self, npc_id) -> bool:
      npc = self.engine.pie.npc_knowledge.profiles.get(npc_id)
      if npc is None:
          return False
      zone = getattr(npc, "current_zone", "global")
      if zone == "global":
          return True   # "global" NPCs always in every active zone
      return zone in self._active_zones
  ```

### REST endpoints

| Endpoint | Body | Purpose |
|---|---|---|
| `POST /story/player_zone` | `{"zones": ["dock_district"]}` | Replace active zones. |
| `POST /story/npc_zone` | `{"npc_id": "varro", "zone": "lighthouse_bluffs"}` | Move a mobile NPC. Rejects if `mobile: false`. |
| `GET /story/zones` | — | Returns `{"active": [...], "npc_zones": {...}}` for debug. |

`record_player_action` gains an optional `zone` field so a single
call handles action + zone change.

### 2-zone reference world: `port_blackwater_zoned`

Five NPCs split across two zones:

```
dock_district:
  - captain_reva (harbor master)
  - finn (dock worker)
  - varro (fence, mobile)

lighthouse_bluffs:
  - old_bones (lighthouse keeper)
  - thessa (hermit)
```

`zones.yaml`:
```yaml
zones:
  dock_district:
    target_population: 4
    min_population: 2
    role_pool: [merchant, dock_worker, fence, harbor_guard, drunk]
    lore_hook: "bustling trade waterfront, smugglers and cargo"
    adjacent_to: [lighthouse_bluffs]
  lighthouse_bluffs:
    target_population: 2
    min_population: 1
    role_pool: [keeper, hermit, pilgrim]
    lore_hook: "windswept cliffs above the harbor where the
                lighthouse watches for storms"
    adjacent_to: [dock_district]
```

Phase 1 only reads `target_population` / `min_population` /
`lore_hook`; they become load-bearing in Phase 2b (generation).

### Unit tests (additive, all offline)

1. `test_zone_soft_weighting_prefers_in_zone`
2. `test_zone_out_of_zone_rate_honored` (statistical over 100 picks)
3. `test_global_zone_npc_always_in_scope`
4. `test_empty_active_zones_preserves_world_wide_mode` (critical backward-compat)
5. `test_quest_hard_filter_on_out_of_zone_focus`
6. `test_bounded_snapshot_surfaces_active_zone_npcs`
7. `test_npc_current_zone_mutation_persists` (save/load roundtrip)
8. `test_immobile_npc_zone_change_rejected`
9. `test_arc_proposal_respects_active_zones`

### Integration tests (new 2-zone PB world)

1. `test_pb_zoned_solo_mode_stays_in_dock`: player in dock_district,
   10 ticks, all quests go to dock NPCs, facts include
   occasional lighthouse rumors.
2. `test_pb_zoned_player_moves_on_tick_5`: starts in dock, moves
   to lighthouse on T5, focus shifts by T7-8.

### Bench

- `bench_story_director.py --world port_blackwater_zoned
  --active-zones dock_district --ticks 20 --terse` — re-run vs
  the non-zoned PB bench.
- `bench_fact_consumption.py` same world — verify per-NPC
  delta drops further than the non-zoned case.

### Finalized Phase 1 open-question answers

| Q | A |
|---|---|
| Zone granularity | Whole town to start; district sub-zones later if games want them. |
| Out-of-zone rate | 85/15 (~1 in 7 focus picks). Tunable via `_OUT_OF_ZONE_RATE`. |
| Adjacent zones | Game client passes the set it wants active. No automatic 1-hop expansion in the Director. |
| Quest hard filter | Hard filter. Out-of-zone NPCs can't offer quests, period. |
| Mobile authority | Game-client-authoritative via `/story/npc_zone`. |
| MP player counts per zone | Phase 3 concern. Not encoded in Phase 1 data structure (yet — flagged). |
| Arc continuity across zones | Arcs are zone-agnostic once proposed. An arc whose cast moves between zones continues to advance via `record_cast_touch`. |

---

## Phase 2a — death plumbing

### Data model

**Profile YAML additions:**
```yaml
status: "alive"                 # or: "deceased"
death_tick: null
death_cause: null
inheritor: null                 # optional successor npc_id
```

**StoryDirector runtime:**
```python
self._deceased_npcs: dict[str, dict] = {}   # id -> {death_tick, death_cause, arc_cleanup, ...}
```

### New action kind: `npc_death`

```json
{
  "action": "npc_death",
  "reason": "The bandits caught Kael on the north road.",
  "npc_id": "kael",
  "cause": "bandit ambush",
  "transfers_quests_to": null
}
```

Dispatch (`_dispatch_npc_death`):
1. Validate NPC exists and is alive.
2. Mark `status: deceased`, `death_tick`, `death_cause`.
3. Move to `_deceased_npcs`.
4. Call `arc_planner.on_cast_death(npc_id)`.
5. Walk active quests; abort or transfer per `transfers_quests_to`.
   **Only open quests transfer** — not goals or personal_knowledge.
6. Emit a FactLedger entry ("Kael was killed in a bandit ambush").
7. Save state.

### Director can kill any NPC

Phase 2c flag enables Director-autonomous death. No gating on
"not in active arc" or "not within 1 zone of player" — **any NPC
can die**. Dead-end quests from deceased givers ARE the realistic
outcome. The world is allowed to be indifferent.

### Deceased propagation

- `_pick_focus_npc`: excludes `deceased` NPCs from the available
  pool. Burst rotation self-heals if its sticky focus dies.
- `_world_snapshot`: excludes deceased from live roster. Adds a
  new "RECENTLY DEPARTED (last 10 ticks)" section when any active
  arc cast contains deceased NPCs.
- `ArcPlanner.on_cast_death`: see below.
- FactLedger: untouched. Gossip about the dead survives.
- Postgen: new `detect_deceased_reference` +
  `repair_deceased_reference` in `postgen.py`. If the LLM writes
  "Kael visits the tavern" after Kael is dead, repair to past
  tense or ghost/memory framing. Lives alongside existing
  wrong-identity and wrong-addressee repairs.

### Arc transitions on death

`ArcPlanner.on_cast_death(npc_id)`:

- For each arc containing `npc_id` in cast:
  - Already past `confront` beat → mark resolved. The death IS
    the resolution.
  - Still at `seed` or `escalate` → transform to **aftermath**
    beat: "the cast deals with {name}'s death".
  - `npc_id` was the sole cast member → collapse to aftermath
    with the social graph's closest connections as new cast.

This preserves story continuity without forcing every death to
abort an arc mechanically.

### Quest cleanup on death

- `transfers_quests_to: pip` → open quests by the deceased NPC get
  `giver_npc = pip`, tagged `inherited: true`.
- No inheritor → open quests marked `aborted`,
  `original_giver_deceased: true` stored for narrative hooks.
- Player-accepted quests in progress → marked `patron_deceased`;
  reward flows through a zone's designated "notary" NPC
  (profile field TBD, probably a `zone_notary: <npc_id>` in
  `zones.yaml`).

### REST endpoints

| Endpoint | Body | Purpose |
|---|---|---|
| `POST /story/npc_death` | `{"npc_id": "...", "cause": "...", "transfers_quests_to": null}` | Game-authoritative death. Queued for next lifecycle tick. |
| `POST /story/npc_revive` | `{"npc_id": "..."}` | Emergency dev-only reverse. Unwinds aftermath. |
| `GET /story/graveyard` | — | Full death history for audit/bench. |

### Lifecycle tick

New `_lifecycle_tick()` method runs at the start of `tick()`,
before `_architect_plan`:

```python
def tick(self, ...):
    if actions_per_tick < 1:
        actions_per_tick = 1
    # ... small-cast cap (unchanged) ...

    # PHASE 0: lifecycle maintenance (NEW)
    self._lifecycle_tick()

    # PHASE 1: architect plan (unchanged)
    plan = self._architect_plan(actions_per_tick)
    # ... rest of tick ...
```

`_lifecycle_tick` priority order:
1. Process any pending death dispatches from `/story/npc_death`.
2. Process any pending birth dispatches (Phase 2b).
3. Autonomous checks (Phase 2c): population gap, arc-referenced
   promotion, optional director death proposals.

At most one lifecycle action per tick (strict cap) so the core
story beat cadence isn't overwhelmed by births/deaths.

### Phase 2a tests

- `test_npc_death_marks_status_and_excludes_from_focus`
- `test_npc_death_cleans_arc_cast`
- `test_npc_death_arc_aftermath_transition`
- `test_npc_death_aborts_open_quests`
- `test_npc_death_transfers_quests_to_inheritor`
- `test_deceased_excluded_from_snapshot`
- `test_burst_focus_breaks_on_death`
- `test_postgen_detects_deceased_reference`
- `test_postgen_repairs_deceased_reference`

Integration: `test_pb_zoned_kill_captain_reva` — kill the harbor
master, verify her open quests abort, her arc transitions, dock
district snapshot updates, finn's dialogue references the death.

---

## Phase 2b — birth generation pipeline

### Data model

**Profile YAML additions:**
```yaml
generated: true                  # dev-only tag — never shown to player UI
generated_at_tick: 47
generated_from_arc: "arc_t42_stranger_at_inn"   # optional promotion source
```

**Zone config** (`data/worlds/<world>/zones.yaml`):
```yaml
zones:
  dock_district:
    target_population: 6
    min_population: 3
    role_pool: [merchant, dock_worker, fence, harbor_guard, drunk]
    lore_hook: "..."
    adjacent_to: [...]
```

### Dispatch: `_dispatch_npc_birth`

1. Generate unique `npc_id` — format `gen_t{tick}_{slug_of_name}`.
2. Python scaffold stage: render a profile template with
   everything except 4-5 narrative fields.
3. LLM generate stage: call with a `generate_npc.yaml` prompt,
   expect a small JSON blob (`name`, `personality_keywords`,
   `world_fact`, `personal_knowledge`, `goal`,
   `initial_connections`).
4. Merge stage 2 + stage 3 into final YAML; validate schema.
5. Write to `data/worlds/<world>/npc_profiles/<new_id>.yaml`.
6. Call new `engine.add_profile(yaml_path)` runtime loader.
7. Seed 1-2 social graph edges from stage 3's
   `initial_connections`.
8. Emit FactLedger entry: "A new {role} arrived in {zone}: {name}".
9. Record in `state.json` under a `birth_history` trail.

### LLM generation prompt

New file `npc-engine/data/story_director/generate_npc.yaml`:
- Takes: zone lore_hook, role template, recent FactLedger, current
  active arcs, 3-5 existing zone NPCs for context, existing names
  to avoid.
- Returns: small JSON with name + 2 flavor traits + 1-2 knowledge
  lines + 1 goal + 1-2 initial connections.
- Schema-validated via existing `verifiers.py` pattern.
- Failure path: retry with repair nudge, or defer to next tick.

### Arc-referenced promotion (killer feature)

Before generating from scratch, scan recent FactLedger for
unnamed-figure references:
- "a stranger at the tavern"
- "an unknown figure"
- "the hooded man"
- "a mysterious traveler"

**Threshold: 3+ mentions AND player currently in the zone.** If
met, promote: generate the new NPC with `name` drawn from the
ledger reference if possible (otherwise LLM names them), and
flag `generated_from_arc: <arc_id>`. The new NPC appears with
pre-seeded lore already tied to an active arc.

Both thresholds are tunable via new constants. We'll play with
them in the stress benches.

### Population management

Every `_lifecycle_tick`:
```python
for zone_name, cfg in self._zone_config.items():
    alive_in_zone = sum(1 for npc in self.engine.pie.npc_knowledge.profiles.values()
                         if getattr(npc, "current_zone", "global") == zone_name
                         and getattr(npc, "status", "alive") == "alive")
    if alive_in_zone < cfg.min_population:
        self._pending_births.append(BirthRequest(zone=zone_name, reason="population_below_minimum"))
        break  # one per tick
```

Population gaps are filled one NPC per tick, max. A depleted zone
gradually recovers over multiple ticks rather than spawning en
masse.

### REST endpoints

| Endpoint | Body | Purpose |
|---|---|---|
| `POST /story/npc_birth_request` | `{"zone": "...", "role": null, "template_from_arc": null}` | Game requests a birth. Queued for next lifecycle tick. |
| `GET /story/population` | — | Per-zone alive count + target + min. Useful for game UI. |

### Phase 2b tests

- `test_birth_generates_valid_profile` (LLM stubbed)
- `test_birth_writes_yaml_and_loads_into_engine`
- `test_birth_seeds_social_graph_edges`
- `test_population_gap_triggers_birth`
- `test_population_at_target_no_birth`
- `test_arc_reference_promotion_fires_at_threshold`
- `test_arc_reference_promotion_requires_player_in_zone`

Integration: `test_pb_zoned_population_recovery` — kill NPCs in
dock_district until under min_population, verify births fill
the gap within 3-5 ticks.

---

## Phase 2c — autonomous lifecycle

### Config flag

```yaml
director:
  lifecycle:
    autonomous: false           # default OFF
    max_deaths_per_session: 3
    max_births_per_session: 10
```

When `autonomous: true`, the Director can emit `npc_death` and
`npc_birth` actions via its rotation. Off by default — game-
authoritative remains the easy path.

### Director-emitted death

New path in `_lifecycle_tick`: propose a death when an arc's
`confront` beat has fired and the death would resolve the arc
narratively. Rare — bounded by `max_deaths_per_session` to prevent
massacres.

**No gating on zones, cast roles, or player proximity.** Director
can kill anyone. The stress bench is how we find out if this is
actually OK or if unbounded mortality causes story collapse.

### Director-emitted birth

Autonomous births fire for:
1. Population gap (as in 2b)
2. Arc-referenced promotion (as in 2b)
3. **NEW** — narrative cohesion: Director-proposed births when a
   zone's lore suggests a role is missing (e.g. dock_district has
   no fence and a smuggling arc is active). Detected via
   ledger scanning + lore_hook semantic matching.

### Phase 2c tests

- `test_autonomous_off_skips_director_lifecycle_actions`
- `test_autonomous_on_director_can_propose_death`
- `test_max_deaths_per_session_bounds_director_kills`
- `test_autonomous_birth_fires_on_narrative_cohesion_gap`

---

## Stress test — push limits

Once 2c lands, a long-session bench that pushes the system beyond
any prior measurement:

```bash
# 200 ticks × 3 actions × autonomous lifecycle on PB zoned
python bench_story_director.py --ticks 200 --reset --model qwen_3b \
    --world port_blackwater_zoned --actions-per-tick 3 \
    --narration-mode terse --lifecycle-autonomous \
    --log logs/stress_pb_zoned_200.json

# Same on synthetic_500 to see if zones + lifecycle scale
python bench_story_director.py --ticks 200 --reset --model qwen_3b \
    --world synthetic_500 --actions-per-tick 3 \
    --narration-mode terse --lifecycle-autonomous \
    --log logs/stress_syn500_lifecycle_200.json
```

### Metrics to capture

- **Death rate**: deaths per 100 ticks. Too high = world depopulation.
- **Birth rate**: births per 100 ticks. Should match death rate + 
  population gap fills. Drift = population trend.
- **Population stability**: zone populations at tick 50, 100, 150, 200.
  Stable = balancing works.
- **Arc continuity**: arcs proposed / resolved / collapsed-from-death.
  Collapsed arcs are cheap narrative costs, not bugs.
- **Performance drift**: mean tick time at T10, T50, T100, T150, T200.
  Flat = no memory leaks or state bloat.
- **Memory growth**: RSS peak at each checkpoint.
- **Dialogue references to deceased NPCs**: postgen repair hit rate.
  High rate = model is confused about who's alive; needs prompt
  hardening.
- **Arc-referenced promotions**: how many births came from arc
  scanning vs. population gap fills. Ratio indicates how "alive"
  the feature actually feels.

### What we're looking for

1. **System stability**: can the Director run 200 ticks with
   lifecycle enabled without state corruption, performance
   degradation, or schema failures?
2. **Narrative coherence**: does the world feel alive, or like a
   chaotic death/birth pump?
3. **Emergent story**: arc-referenced promotions should create
   "characters the world grew into existence" — track qualitatively.
4. **Breaking points**: what's the population floor where the
   world collapses? What's the death rate where the player would
   notice "everyone keeps dying"?

---

## Backlog items not in this plan (reminder)

From the prior session's memory, still open and still valid:
- Events section reserve/cap (3-token captain_reva miss from
  2026-04-15 bench)
- Ledger noise rate scaling for small casts
- NLI false-positive bypass for terse short content
- Quest id collision rename with tick suffix
- Move Ashenvale lore into `data/worlds/ashenvale/story/` for
  consistency with PB

These can land alongside or after this plan, small and
independent, not blocking. Noted here so they don't get lost.
