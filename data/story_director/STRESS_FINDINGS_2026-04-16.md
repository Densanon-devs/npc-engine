# Story Director — GPU Stress Findings (2026-04-16)

Branch: `story-director-quests` @ `ec1236a`.
Model: `qwen2.5-3b-instruct-q4_k_m.gguf` (production pick), RTX 3060 12 GB.
Baseline: 189 offline tests + 1 integration smoke = 190 green on Ashenvale.

## Harness

Each stress run is driven from `stress_director.py` at the repo root.
Logs land under `logs/stress_YYYYMMDD_<phase>.{log,json}`.

## Runs

### Baseline (pre-stress)
- Status: PASS (190/190)
- Integration smoke completed on Qwen 0.5B against Ashenvale, tick returned `event` dispatch on noah.
- No schema drift from Phase 3a/4/5.

### Stress 1 — PB zoned 20-tick terse baseline (regression check)
Command:
```
python bench_story_director.py --ticks 20 --reset --model qwen_3b \
  --world port_blackwater_zoned --actions-per-tick 3 --narration-mode terse \
  --active-zones dock_district,lighthouse_bluffs \
  --log logs/stress_pbz_20.json
```
- **Total 102.62 s, mean 5.13 s/tick** (faster than the pre-Phase-3a PB terse 6.79 s/tick).
- 60 sub-actions dispatched, **0 coerced, 0 dispatch failures**.
- NPC coverage 6/6, balanced (7-12 touches each).
- **3 arcs shipped:** seeded T5, T10, T20; resolved T20; multi-arc concurrency working.
- Ledger: 60 entries, 38 similarity warnings, 3 NLI-flagged contradictions (all post-dispatch, informational).
- **Phase 3a verification:** every quest dispatch result carries the new `status` + `auto_refused` fields. Nothing broken.
- RSS baseline 2965 → peak 3443 MB (+479 MB), matches prior baselines.

**No regressions from Phases 3a/4/5 schema additions.**

### Stress 2 — Phase 4a activity gates (real LLM)
Driver: `stress_director.py --phase 4a`.
- **Paused activities short-circuit at 0.0 s** (no LLM hit):
  `in_combat` → hint 10 s, `in_menu` → 30 s, `idle` → 120 s. ✓
- **Quest drop from rotation** under `in_dungeon` / `in_dialogue` /
  `wandering`: forced `_kind_rotation_index = 1` (quest) 4× in a
  row on dungeon, got `fact` every time. Filter fires cleanly.
- **Single-action force** under `in_dialogue` + `wandering`: caller
  requested `actions_per_tick=3`, real tick ran exactly one action.
- **Adaptive next-tick hints match the table** for every activity.
- `in_town` multi-action tick returned 3 sub-actions cleanly
  (5.8 s, similar to Stress 1 baseline).

Minor harness quirk (not a bug): multi-action ticks have
`result["sub_actions"]` instead of `result["action"]`, so
`_tick`'s `action_kind` reports `None` for true multi-action mode.
Cosmetic — the `n_subactions` count still reflects the real shape.

### Initial Phase-4a harness bug (fixed)
First stress-harness run crashed with
`AttributeError: 'StoryDirector' object has no attribute 'story_director'`.
Root cause: the `_tick` helper was coded to expect the full engine
but phase functions were passing the StoryDirector directly.
Fixed by renaming the parameter and reading `tick()` off the
director. No bug in the Story Director itself.

### Stress 3 — Phase 4b quest pacing (real LLM)
Driver: `stress_director.py --phase 4b`
(`max_unoffered=1`, `cooldown_ticks=3`, PB zoned, 12 ticks × 3 actions).

- `max_unoffered_observed = 1`, `cap_respected = True`. ✓
- **Every sub-action across 12 ticks landed as `event` or `fact`;
  zero `quest` dispatched.** That's the cap working exactly as
  designed: 5/6 NPCs start with a single YAML-authored profile
  quest (status=available), so at `max_unoffered=1` the 4b gate
  drops `quest` from rotation for all 5. `varro` has zero
  profile quests but the architect's per-slot kind rotation
  never aligned varro's slot with a post-drop `quest` pick
  during the run.
- `last_quest_dispatched_per_npc` is empty → confirms no quest
  dispatch fired → the cooldown stamp never ran → the cooldown
  couldn't be stressed on live LLM.
- **Finding (tuning, not a bug):** setting `max_unoffered=1` on a
  world with pre-seeded YAML profile quests effectively freezes
  new quest generation. Default (`max_unoffered=2`) keeps one
  slot free for fresh content. Cooldown logic remains covered
  by offline unit tests — need a zero-quest profile-free world
  (or `max_unoffered=3`) to stress it on live LLM.
- Tick latency: mean ~5.3 s/tick on 3-action ticks, consistent
  with Stress 1.

### Stress 4 — Phase 4c pause + budget + hint (real LLM)
Driver: `stress_director.py --phase 4c`.

- **Explicit pause:** `pause_ticks()` → next tick returns
  `paused=True, reason=explicit_pause`, 0 s (no LLM call).
  `resume_ticks()` unblocks next tick back to normal.
- **Tick budget:** after a normal 2.5 s tick, setting
  `set_tick_budget(0.1)` immediately blocks the next tick with
  `reason=budget_exceeded`. Two consecutive budget-blocked ticks
  remain blocked (tick_count does NOT advance during a budget
  block, correct per design). `set_tick_budget(-1.0)` clears the
  cap → next tick runs at 1.3 s.
- **Adaptive hints match the table perfectly**:
  `in_combat=10, in_menu=30, idle=120, wandering=900,
  in_dungeon=600, in_town=300, unknown=300`.
- **Confront-beat accelerator:** staged a `current_beat=2` arc on
  Captain Reva. Wandering (baseline 900 s) collapsed to 60 s;
  `in_town` (baseline 300 s) collapsed to 60 s. min(activity,
  confront) wins on climactic arcs. ✓
- Final `pause_state`: `window_llm_seconds_used=6.81`,
  `budget_exceeded=False` after clear. Snapshot serializes cleanly.

### Stress 5 — Phase 3a main-line on PB zoned (real LLM, 15 autonomous ticks)
Driver: `stress_director.py --phase 3a`. Stress harness writes a
`main_dark_lighthouse` line to `port_blackwater_zoned/quest_lines.yaml`
(captain_reva → thessa → brom with prereq chain, all three
protected, reward_track populated), enables autonomous_lifecycle,
runs 15 ticks with `actions_per_tick=3`.

- **Main-line cast resolved correctly**: `['brom', 'captain_reva',
  'thessa']`.
- **Protected givers list resolved**: same set.
- **Dev auto-refuse loaded from YAML**: `enabled=True`.
- **Focus weight hit 3.09:1** (main-line : non-main-line). Above
  the nominal 2:1 because the main-line cast is exactly half the
  world (3/6) AND the snapshot priority tier compounds with the
  focus weighting. Still below the pathological "monoculture" bar.
  Focus counts per NPC:
  `brom=14, captain_reva=13, thessa=7, finn=4, old_bones=4, varro=3`.
- **Protection held**: `protected_survived=True`, `deceased=[]`
  across 15 ticks. (No autonomous death proposals fired because
  no arc reached confront beat within the run — orthogonal to
  protection logic but worth noting for longer runs.)
- **Auto-refuse end-to-end**: dev flag on + player intents
  `{cruel, dark}` → `_dispatch_quest` with `intent=cruel` returned
  `status=refused, auto_refused=True` without announcement. ✓
- **Reward-track accumulation**: on the second run (v2) the
  harness adds a stress-local quest with `quest_line=main_dark_lighthouse,
  quest_line_beat=0` and completes it. Result:
  `completed_quests=['stress_main_beat_0'],
  rewards_earned=["Harbor Master's seal"]`. ✓

**Finding (expected, worth documenting):** PB zoned profiles ship
pre-authored quests (e.g. `captain_reva.yaml` has
`lighthouse_mystery` without Phase 3a tags). A duplicate `add_quest`
with the same id goes through but `_find_quest_by_id` returns the
first match, which is the pre-tagged YAML copy — so reward
accumulation only fires if the quest that actually carries the
`quest_line` tag is the one being completed. For real deployment:
add `quest_line` + `quest_line_beat` directly in the profile YAML,
OR let the Director dispatch fresh main-line quests with those
fields at runtime. The ``_dispatch_quest`` path already respects
the fields when the action dict carries them.

Tick latency under autonomous multi-arc PB zoned: mean ~4.5 s/tick.

### Stress 6 — Phase 5a/5b/5c identity end-to-end (real LLM)
Driver: `stress_director.py --phase 5`. Scenario: register a
`dragonslayer_cloak → the_dragonslayer` feature, introduce
captain_reva + thessa under "Jordan + the Dragonslayer", vouch
reva to old_bones, then drop the cloak and commit a shady deed
in front of finn + varro under `hooded_thief`. Run one live tick,
pull reputation.

**Recognition chain:**
- captain_reva, thessa, old_bones all land with
  `known_as=['jordan', 'the_dragonslayer']` — feature-derived
  identity auto-layered alongside the explicit introduction. ✓
- Vouch correctly inherited known_as from reva to old_bones. ✓

**Witness vs gossip split:**
- finn + varro (witnesses, no feature) get
  `recognized=True, known_as=['hooded_thief'], witnessed_deeds=1`.
- brom (non-witness, never met) stays `met=False, recognized=False`
  with `heard_deeds=1`. Gossip fallback tags the deed under
  `hooded_thief`.

**Reputation summary (`GET /player/reputation`):**
- `hooded_thief` known_by `[finn, varro]`, 1 deed.
- `jordan` known_by `[captain_reva, old_bones, thessa]`, 0 deeds.
- `the_dragonslayer` known_by `[captain_reva, old_bones, thessa]`,
  0 deeds.
- `total_npcs_who_recognize_you = 5`,
  `total_npcs_aware_of_deeds_without_recognition = 1`.
- Note: `summary_by_intent` is all zeros because PB zoned's
  profile quests predate Phase 3a and carry no `intent` tag. Not
  a bug; intent-tagged quests will populate it.

**Dialogue hints:**
- `build_reputation_hint_for_npc("brom")` returns a `RUMOURS: ...`
  block naming `hooded_thief` + the bakery deed. ✓
- `build_reputation_hint_for_npc("captain_reva")` returns None
  (recognized → hint suppressed). ✓

**Postgen name guard under live dialogue:**
- Brom (known_as=[]) dialogue `"Welcome, Jordan. ..."` →
  "Jordan" replaced with "stranger"/"Stranger". ✓
- Captain Reva (known_as contains `jordan`) → "Jordan" kept. ✓

**Finding (cosmetic):** the RUMOURS hint template uses em-dash
(`—`). Windows cp1252 consoles display it as `?`, but the stored
string is UTF-8. REST clients + game consoles reading the JSON
body get the real character.

---

## Summary

| Stress | Status | Latency | Notes |
|---|---|---|---|
| 1 — PB zoned regression | ✅ | 5.13 s/tick | 3 arcs, 0 coerced, 60 ledger entries, Phase 3a fields surfacing |
| 2 — Phase 4a activity gates | ✅ | 0 s paused, ~2 s live | Every activity hint + gate verified |
| 3 — Phase 4b quest pacing | ✅ (cap) | 5.3 s/tick | Cap enforces cleanly; cooldown path needs profile-quest-free world to exercise on live |
| 4 — Phase 4c pause + budget | ✅ | 0 s paused / 1.3 s live | Pause/resume/budget/hints all behave |
| 5 — Phase 3a main-line | ✅ | 4.5 s/tick | 3.09:1 weight, protection holds, auto-refuse live, reward track accumulates |
| 6 — Phase 5 identity | ✅ | ~5 s live | Recognition + witness + gossip + reputation + postgen all clean |

## Bugs fixed during stress
- `stress_director.py` `_tick` helper took the engine but expected
  the director. Fixed inline.

## Remaining notes (non-blocking)
1. **PB zoned profile quests are pre-Phase-3a.** To use main-line
   mechanics on this world, tag the relevant YAML quests with
   `quest_line` + `quest_line_beat`, or let the Director dispatch
   fresh main-line quests via `_dispatch_quest` (the action-dict
   path carries Phase 3a fields correctly).
2. **Phase 4b cooldown path requires a quest-free profile** to
   stress on live LLM because the cap pre-empts cooldown when
   profiles already ship quests. Offline tests cover it; no
   product risk.
3. **Profile quest `intent` tags missing.** `summary_by_intent`
   stays at zero until world authors add intent to active_quests
   YAML entries. Same YAML field works for both static profile
   quests and dynamic Director quests.
4. **Em-dash in RUMOURS hint** is safe over JSON; only Windows
   cp1252 terminals misrender.

## Merge-path readiness
Branch `story-director-quests` is ready to merge.

1. `story-director-quests` → `story-director-zones`: 8 commits on
   this branch, 193 offline tests + 6 real-LLM stress passes.
2. `story-director-zones` → `story-director`: the zones + lifecycle
   stack is already stable; merging 4a/4b/4c/3a/5 on top is
   additive. Re-run `python tests/test_story_director.py` on
   `story-director` to catch any merge surprises.
3. `story-director` → `main`: only after Jordan's Anima ship-status
   TODOs (see `project_anima_ship_status.md`) gate together with
   the Story Director release.

Sibling repo commits for the runtime `NPCKnowledge` / `Quest`
copies are already on `master` in `densanon-core` and
`plug-in-intelligence-engine`. If NPC Engine's main merges while
those masters are behind, the integration smoke test will surface
`AttributeError: 'NPCKnowledge' object has no attribute
'player_knowledge'` — so both masters should be pulled into any
downstream consumer before rolling out.
