# Story Director — Phases 3 + 4 + 5 Plan (Quests, Pacing, Reputation)

Finalized 2026-04-16. Scoped and signed off by Jordan. This doc is
the canonical plan for the quest-line system (Phase 3a), pacing &
activity awareness layer (Phase 4), and anonymity / reputation
system (Phase 5). Together these close the gap from "technically
works" to "Skyrim-shippable Director."

**Branches:** will be cut from `story-director-zones` (where
Phases 1 + 2 live). Candidate name: `story-director-quests`.

**Prerequisites**: Phases 1 (zones) and 2 (lifecycle) already
shipped on `story-director-zones` as of `fc7777e`. This plan
builds on top without modifying anything under them.

---

## Design invariants

- **Modular, game-authoritative default.** Every feature is gated
  so existing worlds and benches keep working unchanged. Dev flags
  are `false` by default; autonomous director behaviours remain
  opt-in.
- **Two-layer opt-ins for player-facing features.** Where a feature
  could affect player agency (auto-refuse, reputation visibility,
  witness tracking), the dev enables the capability in config and
  the player configures their personal settings via REST. Player
  settings persist regardless of dev flag state; dev flag gates
  whether the player's settings actually fire.
- **Game code owns identity/faction/combat; Director owns narrative
  state.** Witness tracking, trust propagation to witnesses,
  combat resolution, and quest-log UI are all game-side. Director
  tracks attribution, intent tags, and narrative consequences.
- **Quests are delivered through dialogue, not autonomous Director
  pushes.** The Director's `quest` action kind adds quests to
  NPC state — the player only sees them if they talk to that NPC.
  Tick cadence controls world simulation, not quest flood rate.

---

## Locked-in decisions (from the 2026-04-16 design session)

| Decision | Value |
|---|---|
| Quest intent scale | **5-tag**: `good` / `neutral` / `gray` / `dark` / `cruel`. Game UI can collapse to **3-tag**: `good` / `neutral` / `harmful`. |
| Default `refusal_trust_delta` | Scale with `moral_weight`: `int(moral_weight * -15)`. Explicit YAML values override. |
| Player visibility of intent | **Hidden from quest log.** Player infers from dialogue cues. Game UI may show tags to devs in debug mode. |
| Auto-refuse mode | **Two-layer opt-in.** Dev enables in config (`director.quest_auto_refuse.enabled`). Player configures intents via REST (`POST /player/auto_refuse`). Player settings persist regardless; dev flag gates whether they fire. |
| Refusal persistence | **Both modes supported.** Default `permanent`. Optional `decay_after_ticks: N` per-quest or per-world. |
| Recognition merge rule | On recognition (player introduces themselves to NPC who knows their reputation), **keep max trust** across identities. Being famous helps. |
| Tick cadence in town | **~5 minutes** (not 30-60s). With `~5s/tick × 12 ticks/hour = ~1.7% GPU**. Dialogue-triggered ticks are separate and immediate when `triggers_director_tick: true` is flagged on a dialogue event. |
| Quests auto-offering flood | **Not a Director concern.** The Director adds quests to NPC state; dialogue surfaces them when player chooses to talk. First-town-visit flood is a game designer authorship concern, not a Director pacing concern. |
| Phase protection for main-line NPCs from autonomous death | **Yes, while quest-line is active.** Game client can still kill protected NPCs via REST. Protection expires when line is `completed`. |
| Skyrim-scale feasibility | **Yes**, with 1000-1500 real NPCs. Bounded snapshot caps prompt size at 16 NPCs regardless of total cast. Per-tick latency stays ~5-6s on GPU. |

---

## Phase 4a — Activity context (1-2 days)

### Goal

Let the game tell the Director what the player is doing. The Director
self-pauses during combat, menus, and idle. During dialogue it ticks
responsively. In wilderness it ticks at low cadence.

### Data model

New enum:
```python
class PlayerActivity(str, Enum):
    IN_TOWN = "in_town"
    IN_DIALOGUE = "in_dialogue"
    IN_DUNGEON = "in_dungeon"
    IN_COMBAT = "in_combat"
    IN_MENU = "in_menu"
    WANDERING = "wandering"
    TRAVELING = "traveling"
    IDLE = "idle"
    UNKNOWN = "unknown"   # default
```

StoryDirector gains:
```python
self._player_activity: str = "unknown"
self._activity_set_at_tick: int = 0
```

Both persisted in `state.json` so a game restart honors the activity
state the client sent last.

### Behavior table

| Activity | Behavior |
|---|---|
| `in_town` | Full ticks at normal cadence (5 min recommended) |
| `in_dialogue` | Single-action ticks (no quest-generation), triggered by game on significant dialogue events |
| `in_dungeon` | Reduced cadence (10 min), prefer events over quests |
| `in_combat` | **Skip tick**, return `{paused: true, reason: "in_combat"}` |
| `in_menu` | **Skip tick** |
| `wandering` | Single-action ticks at 15-20 min cadence, prefer fact/event |
| `traveling` | Single summary tick on arrival (Phase 4d enhancement) |
| `idle` | **Skip tick** |
| `unknown` | Full ticks (backward compat) |

### tick() prologue

```python
def tick(self, ...):
    if self._player_activity in ("in_combat", "in_menu", "idle"):
        return {
            "tick": self.tick_count,
            "action": {"action": "noop", "reason": f"paused: {self._player_activity}"},
            "dispatch": {"ok": True, "kind": "noop"},
            "raw_response": "",
            "paused": True,
            "next_tick_recommended_in_seconds": 10,
        }
    # ... existing logic with activity-aware tweaks ...
```

### REST endpoints

| Endpoint | Body | Purpose |
|---|---|---|
| `POST /story/activity` | `{"activity": "in_combat"}` | Game sets player activity |
| `GET /story/activity` | — | Debug / sync |

### Bench flag

`--player-activity {in_town|wandering|...}` — static activity for
the whole bench. For integration tests we can add a per-tick
override table in a follow-up.

### Tests

1. `test_activity_default_is_unknown_and_ticks_normally`
2. `test_activity_in_combat_returns_paused_noop`
3. `test_activity_in_menu_returns_paused_noop`
4. `test_activity_in_dungeon_prefers_events_over_quests`
5. `test_activity_persists_in_state_json`

---

## Phase 4b — Per-NPC quest accumulation (1-2 days)

### Goal

Prevent NPCs from piling up quests faster than a player can digest
them. Per-NPC caps, not global throttling.

### Tunables

```python
_MAX_UNOFFERED_QUESTS_PER_NPC = 2
_NPC_QUEST_COOLDOWN_TICKS = 10
```

Both overridable at runtime via `director.set_quest_pacing(...)`.

### Gate in `_pick_action_kind`

Before allowing `quest` kind:
1. Count NPC's current `available` quests (unaccepted, not yet offered
   or offered-but-declined). If ≥ `_MAX_UNOFFERED_QUESTS_PER_NPC`,
   drop `quest` from allowed.
2. Check `_last_quest_dispatched_per_npc[focus_npc]`. If within
   `_NPC_QUEST_COOLDOWN_TICKS`, drop `quest` from allowed.

State:
```python
self._last_quest_dispatched_per_npc: dict[str, int] = {}
```

Persisted in `state.json`.

### Main-line override

Phase 3a main-line beats **bypass** the per-NPC quest cap. Main-line
progression is authored; if the author scheduled three beats on the
same patron, that's deliberate.

### Tests

1. `test_quest_capped_at_max_unoffered_per_npc`
2. `test_quest_cooldown_skips_quest_within_window`
3. `test_cooldown_resets_after_ticks_elapse`
4. `test_main_line_bypasses_per_npc_cap`

---

## Phase 4c — GPU coordination + next-tick hint (1 day)

### Pause/resume

```python
POST /story/pause
POST /story/resume
GET  /story/pause_state
```

When paused, `tick()` returns `{paused: true, reason: "explicit_pause"}`.
State persisted in `state.json`.

### Tick budget

```python
POST /story/tick_budget
  {"max_seconds_per_minute": 6}
```

Director tracks total LLM time used in a trailing 60s window. If
budget is exceeded, subsequent ticks return
`{paused: true, reason: "budget_exceeded"}` until the window rolls
forward.

```python
self._tick_budget_seconds: float = -1.0   # -1 = unconstrained (default)
self._tick_time_log: list[tuple[float, float]] = []   # [(timestamp, duration), ...]
```

### Adaptive next-tick hint

Every tick result gains:
```python
{
    ...,
    "next_tick_recommended_in_seconds": 30,
}
```

Formula:
```python
def _compute_next_tick_hint(self) -> int:
    if self._player_activity == "in_combat":
        return 10   # check back quickly in case combat ends
    if self._player_activity == "in_menu":
        return 30
    if self._player_activity == "idle":
        return 120
    if self._player_activity == "wandering":
        return 900   # 15 min
    if self._player_activity == "in_dungeon":
        return 600   # 10 min
    # in_town, in_dialogue, unknown — default 5 min
    # Accelerate if an active arc is at confront beat
    for arc in self.arc_planner.active_arcs():
        if arc.current_beat >= 2:
            return 60   # 1 min during climactic beats
    return 300   # 5 min default
```

### Tests

1. `test_pause_state_blocks_ticks`
2. `test_resume_restores_normal_ticks`
3. `test_tick_budget_throttles_when_exceeded`
4. `test_next_tick_hint_reflects_activity`
5. `test_next_tick_hint_accelerates_during_confront_beat`

---

## Phase 3a — Quest-lines + intent + refusal (3-4 days)

### Quest data model additions

All optional, all backward-compatible:
```python
@dataclass
class Quest:
    # existing: id, name, description, status, reward, objectives
    # Phase 3a additions
    quest_line: Optional[str] = None
    quest_line_beat: int = 0
    prerequisite_quests: list[str] = field(default_factory=list)
    # Intent / moral weight (Phase 3a additions)
    intent: Optional[str] = None        # "good" | "neutral" | "gray" | "dark" | "cruel"
    moral_weight: float = 0.0
    refusal_trust_delta: int = 0         # 0 = use default: int(moral_weight * -15)
    # Refusal persistence
    refusal_mode: str = "permanent"      # or "decay"
    refusal_decay_ticks: int = 0         # 0 = N/A; >0 = number of ticks until re-eligible
```

### `quest_lines.yaml` config (optional)

```yaml
quest_lines:
  main_dark_lighthouse:
    type: main
    title: "The Dark Lighthouse"
    description: "Discover why the lighthouse went dark."
    beats:
      - quest_id: "lighthouse_mystery"
        giver: "captain_reva"
      - quest_id: "witness_account"
        giver: "thessa"
        requires: ["lighthouse_mystery"]
      - quest_id: "broken_lantern"
        giver: "brom"
        requires: ["witness_account"]
    protected_givers:
      - captain_reva
      - thessa
      - brom
    reward_track:
      - "Harbor Master's seal"
      - "Thessa's sea-tongue charm"
      - "Brom's pilgrim medallion"

  side_tavern_informant:
    type: side
    title: "The Loose Tongue"
    beats:
      - quest_id: "find_informant_bones"
        giver: "old_bones"
    protected_givers: []
```

### Director behavior

1. **Snapshot priority boost** for main-line cast members (just
   after planned focus).
2. **Focus selection weighting**: 2:1 preference for NPCs in active
   main-line casts (tunable `_MAIN_LINE_WEIGHT_FACTOR = 2`).
3. **Autonomous death protection**: `_propose_autonomous_death`
   skips NPCs in `protected_givers` for any active (non-completed)
   main line. Game client can still kill via REST.
4. **Prompt preference**: quest-action prompt gains ACTIVE MAIN LINE
   block naming current beat + giver, nudging LLM to write
   line-advancing quests over random fetch quests.
5. **Sequential gating**: `_dispatch_quest` checks `prerequisite_quests`
   before allowing `status: available`. If prereqs incomplete, sets
   `status: locked`.
6. **Reward accumulation**: on quest completion, if part of a main
   line, append the corresponding reward to
   `_quest_line_state[line_id].rewards_earned`.

### Intent — author + Director-generated paths

**Author-tagged in YAML** (Path A, takes precedence).

**Director-generated with intent inferred from giver** (Path B). The
quest-action prompt template gets a new block:
```
GIVER CONTEXT:
{focus_npc}'s personality: {personality}
{focus_npc}'s goals: {top_goals}

INTENT GUIDANCE:
- Good-aligned characters (priests, honest craftsfolk) → `intent: good` or `neutral`
- Gray characters (smugglers, fences, opportunists) → `intent: gray`
- Harmful characters (assassins, thieves, zealots) → `intent: dark` or `cruel`

Include `intent` (one of good/neutral/gray/dark/cruel) and
`moral_weight` (0.0-1.0) in the quest JSON you emit.
```

### Refusal — REST + mechanics

```python
POST /quests/refuse
  {
    "quest_id": "...",
    "npc_id": "...",
    "reason": "..."   # optional
  }
```

Director behavior on refusal:
1. Set `quest.status = "refused"`
2. If `refusal_trust_delta == 0`, compute
   `trust_delta = int(moral_weight * -15)`. Apply via
   `engine.adjust_trust()`.
3. Emit FactLedger entry: `"The player refused {giver}'s quest '{name}'"`
4. Future dialogue with giver gets context line: `"The player has
   previously refused your quest '{name}'."`
5. If `refusal_mode == "decay"`, schedule re-eligibility after
   `refusal_decay_ticks` via a new `_refused_quest_timers` dict:
   `{quest_id: unlock_tick}`.
6. Lifecycle tick scans `_refused_quest_timers`; when a timer expires,
   sets `quest.status = "available"`, emits a FactLedger entry
   `"The giver's offer is open again."` (subtle: the game can
   surface this as "the giver brings it up again" in next
   conversation).

### Auto-refuse — two-layer opt-in

**Dev layer** in `config.yaml`:
```yaml
director:
  quest_auto_refuse:
    enabled: false        # default — feature disabled
    player_configurable: true
```

**Player layer** via REST:
```python
POST /player/auto_refuse
  {"intents": ["cruel", "dark"]}
```

Player state:
```python
self._player_auto_refuse_intents: set[str] = set()
```

Persisted in `state.json`. Setting survives dev flag toggles.

Dispatch-time check in `_dispatch_quest` (or wherever the quest is
marked offerable):
```python
if (self._config.quest_auto_refuse.enabled
        and quest.intent in self._player_auto_refuse_intents):
    # Auto-refuse immediately — no dialogue prompt
    self._process_refusal(quest, npc_id, reason="auto_refused_by_player_filter")
```

### Tests (Phase 3a)

1. `test_quest_without_quest_line_is_standalone_backward_compat`
2. `test_prerequisite_gating_locks_until_satisfied`
3. `test_protected_giver_excluded_from_autonomous_death`
4. `test_protected_giver_killable_by_game_client`
5. `test_main_line_npc_prioritized_in_snapshot`
6. `test_main_line_focus_weight_preferred`
7. `test_quest_line_reward_track_persists`
8. `test_empty_quest_lines_config_preserves_existing_behavior`
9. `test_intent_loads_from_yaml`
10. `test_director_generated_quest_includes_intent_tag` (LLM stubbed)
11. `test_refusal_trust_delta_scales_with_moral_weight`
12. `test_refused_quest_surfaces_in_giver_dialogue_context`
13. `test_refusal_decay_mode_reopens_quest_after_ticks`
14. `test_refusal_permanent_mode_stays_refused`
15. `test_auto_refuse_dev_disabled_ignores_player_filter`
16. `test_auto_refuse_dev_enabled_applies_player_filter`
17. `test_main_line_bypasses_per_npc_quest_cap`

### Integration bench

`logs/pb_main_line_20.json` on PB zoned:
- Configure main_dark_lighthouse line
- Enable `autonomous_lifecycle`
- Ensure main-line patrons (captain_reva, thessa, brom) survive
  all 20 ticks
- Non-protected NPCs (old_bones, finn, varro) killable by
  autonomous proposals
- Quest-line beat state persists across ticks
- Reward track accumulates on completion

---

## Phase 5a — Identity split (2 days)

### FactLedger subject tags

```python
@dataclass
class LedgerEntry:
    text: str
    npc_id: str
    kind: str
    tick: int
    embedding: list[float]
    # Phase 5a addition
    subject_identity: Optional[str] = None
```

Default `None` means legacy-compatible (treated as `"player"` for
backward-compat in reputation queries). Director- and
dialogue-generated entries about the player get explicit identity
tags based on context.

### Per-NPC player knowledge

```python
# in NPCKnowledge
player_knowledge: dict = field(default_factory=lambda: {
    "met": False,
    "recognized": False,
    "known_as": [],           # list of identity strings this NPC associates with the player
    "witnessed_deeds": [],    # ledger entry refs for deeds this NPC personally saw
    "heard_deeds": [],         # ledger entry refs for gossip this NPC received
    "first_met_tick": None,
    "last_interaction_tick": None,
})
```

Persisted in NPC state files (same system Phase 2a's `status`
field uses).

### Recognition triggers — REST

```python
POST /player/introduce
  {"to_npc": "old_bones", "name": "Jordan", "titles": ["Dragonslayer"]}

POST /player/visible_feature
  {"feature": "dragonslayer_cloak"}

POST /player/vouched_by
  {"voucher_npc": "captain_reva", "to_npc": "thessa"}
```

Director behavior on recognition:
1. Set `player_knowledge[to_npc].met = True`, stamp `first_met_tick` if unset
2. Set `player_knowledge[to_npc].recognized = True`
3. Merge `known_as` lists — NPC now associates player with named identities
4. Apply Trust merge rule: **keep max trust across identities**
5. Re-scope NPC's dialogue context to include all deeds under any
   `known_as` identity

### Witness recording (auto-recognition)

Existing `record_player_action` gains optional fields:
```python
POST /story/player_action
  {
    "text": "Player killed Old Bones in the tavern",
    "target": "old_bones",
    "witness_npcs": ["finn", "varro"],     # NEW — Phase 5a
    "visible_feature": null,                # NEW — Phase 5a
    "subject_identity": "stranger"         # NEW — how the ACT is attributed in gossip
  }
```

Behavior:
- All `witness_npcs` get instant `met = True`, `recognized = True`,
  and the deed added to their `witnessed_deeds`
- FactLedger entry is tagged with `subject_identity`
- Non-witness NPCs only hear via gossip propagation, receive it as
  `heard_deeds` under the attributed identity

### Postgen name guard

New functions in `postgen.py`:
```python
def detect_unauthorized_name_use(dialogue: str, npc_profile: dict,
                                  player_known_names: set[str]) -> Optional[str]:
    """Return the unauthorized name if NPC uses it but isn't recognized."""

def repair_unauthorized_name_use(dialogue: str, wrong_name: str,
                                  replacement: str = "stranger") -> str:
    """Swap the name for a generic address."""
```

Called in `validate_and_repair` after wrong-addressee check. The
`player_known_names` set is populated from the speaker's
`player_knowledge.known_as` list — if they haven't been introduced,
that list is empty and any player-name in dialogue is unauthorized.

### Gossip propagation by identity

`GossipPropagator.propagate()` already walks the social graph. We
extend it so ledger entries about the player carry
`subject_identity` through the walk, and each receiving NPC appends
the deed to their `heard_deeds` tagged by that identity.

Identity-based recognition happens purely at the postgen/dialogue
layer — gossip doesn't "know" the real player. Two NPCs who both
hear "the hooded stranger killed old_bones" can independently
connect that identity to the player (or never).

### Tests

1. `test_fact_ledger_entry_carries_subject_identity`
2. `test_unmet_npc_has_empty_player_knowledge`
3. `test_player_introduce_flips_met_and_recognized`
4. `test_witness_npc_gets_instant_recognition`
5. `test_gossip_propagates_deed_under_subject_identity`
6. `test_postgen_rewrites_unauthorized_name_to_stranger`
7. `test_postgen_allows_name_use_after_recognition`
8. `test_trust_merges_max_across_identities_on_recognition`

---

## Phase 5b — Deed attribution + witness tracking (1 day)

### Scope

Full integration of witness-based attribution with the ledger,
gossip, and dialogue layers. Most of the data model exists from
Phase 5a; Phase 5b ties it together.

### `record_player_action` extensions

```python
POST /story/player_action
  {
    "text": "...",
    "target": "...",
    "witness_npcs": [...],
    "visible_feature": "dragonslayer_cloak" | null,
    "subject_identity": "stranger" | "jordan" | null
  }
```

If `subject_identity` is None and `visible_feature` is set, look up
any existing identity keyed by that feature and use it. Otherwise
default to `"unknown_figure"` for anonymous actions.

### Visible feature registry

```python
self._visible_feature_to_identity: dict[str, str] = {}
# e.g. {"dragonslayer_cloak": "the_dragonslayer",
#       "bloodied_hand": "hooded_killer"}
```

Populated via `POST /player/register_feature`:
```python
{
  "feature": "dragonslayer_cloak",
  "identity": "the_dragonslayer"
}
```

### Auto-recognition from features

When an NPC meets the player and the player's `visible_feature` is
set AND there's a matching identity registered, auto-promote that
NPC's recognition (`recognized = True`, identity added to
`known_as`).

### Tests

1. `test_anonymous_action_with_no_witnesses_tags_unknown_figure`
2. `test_visible_feature_auto_recognition_on_first_meeting`
3. `test_visible_feature_overrides_explicit_subject_identity`

---

## Phase 5c — Reputation surfacing (1 day)

### `GET /player/reputation`

```json
{
  "known_identities": {
    "jordan": {
      "known_by": ["captain_reva", "finn"],
      "deeds": ["killed the harbor pirate", "restored the lighthouse"]
    },
    "hooded stranger": {
      "known_by": ["old_bones", "thessa"],
      "deeds": ["killed old bones", "visited the bluffs at night"]
    }
  },
  "total_npcs_who_recognize_you": 2,
  "total_npcs_aware_of_deeds_without_recognition": 4,
  "summary_by_intent": {
    "good": 2,
    "neutral": 0,
    "gray": 0,
    "dark": 1,
    "cruel": 0
  }
}
```

### Optional in-dialogue reputation hints

When an NPC has `heard_deeds` but `recognized == False`, their
dialogue prompt gains:
```
RUMOURS: You've heard tales of {identity} — {deed_summary}.
Don't assume the player is them, but you may bring up the
rumour if relevant.
```

This produces natural moments: "There's a rumour of a hooded
stranger at the docks... you've heard anything?" — and the player
decides whether to admit it.

### Tests

1. `test_get_reputation_returns_per_identity_state`
2. `test_reputation_hint_appears_in_unrecognized_npc_dialogue`
3. `test_reputation_hint_absent_after_recognition`

---

## Full implementation order

1. **Phase 4a** — activity context (1-2d)
2. **Phase 4b** — per-NPC quest caps (1-2d)
3. **Phase 4c** — GPU coordination + next-tick hint (1d)
4. **Phase 3a** — quest-lines + intent + refusal (3-4d)
5. **Phase 5a** — identity split (2d)
6. **Phase 5b** — deed attribution (1d)
7. **Phase 5c** — reputation surfacing (1d)

**Total: 10-13 days.** Each phase ships independently on
`story-director-quests` branch with its own commit + FINDINGS
section + bench validation. Final branch merge into
`story-director` when all 7 phases are stable.

---

## Companion changes in other repos

- **densanon-core**: NPCKnowledge gets `player_knowledge` dict,
  Quest gets all new fields
- **plug-in-intelligence-engine**: same (runtime copy)

Three-copy sync discipline as before — enforced by the NPC Engine
integration smoke test.

---

## Backward compatibility guarantees

- All existing profile YAMLs load unchanged. Worlds without
  `quest_lines.yaml` → no main-line behaviour. Quests without
  `intent` tag → treated as neutral.
- All existing benches continue to run at the same per-tick
  latency or better. Phase 4 adds gating, never mandates extra
  work.
- All 144 existing offline tests continue to pass.
- REST contract is additive. Existing endpoints unchanged; all new
  endpoints are new routes.
- State file layout gains new optional keys. Old state files load
  with sensible defaults for missing keys.

---

## Open questions — NONE AT THIS POINT

All design questions answered in the 2026-04-16 session. Ready to
implement. Start with Phase 4a.

---

## Related docs

- `PHASE_ZONES_LIFECYCLE_PLAN.md` — Phases 1 + 2 (zones, deaths,
  births) — shipped
- `FINDINGS.md` — canonical bench results and empirical journey
