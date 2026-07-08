"""
Story Director — the "Cardinal" overseer that watches Ashenvale and decides
what happens next.

Not a Capability — capabilities are per-NPC dialogue hooks. The Story Director
is a world-level service owned by NPCEngine. On each tick() it:

  1. Builds a compact snapshot of the current world state
  2. Calls the base LLM directly (bypassing per-NPC dialogue pipeline)
  3. Parses a structured action from the response
  4. Dispatches the action through existing NPCEngine APIs
     (inject_event, add_quest, add_knowledge)

v0 keeps it deliberately small: one LLM call per tick, one action per tick,
three action types. No architect/worker, no grammar, no lore embedding.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import yaml

from npc_engine.bridge import NPC_ENGINE_ROOT
from npc_engine.knowledge import Quest

if TYPE_CHECKING:
    from npc_engine.engine import NPCEngine

logger = logging.getLogger("NPCEngine.story_director")

# NarrativeJudge passive observer — off by default (zero risk to the
# existing 192 test suite). Set NPC_ENGINE_NARRATIVE_JUDGE_OBSERVE=1 to
# log per-dispatch coherence scores against active quest specs to
# <runtime_dir>/narrative_judge_observations.jsonl. Logging only —
# never affects dispatch behavior. See npc_engine/narrative_judge.py.
_NARRATIVE_JUDGE_OBSERVE_ENV = "NPC_ENGINE_NARRATIVE_JUDGE_OBSERVE"

# Predictive FactLedger lane (PHASE_PREDICTIVE_FACTLEDGER_PLAN v1).
# The layer itself is ON by default because it is behavior-neutral:
# it observes ticks, logs predicted-drift at DEBUG, and writes two
# gitignored sidecars. Set NPC_ENGINE_PREDICTIVE_DISABLE=1 to remove
# it entirely. The one behavior-CHANGING piece — the edge-prior boost
# on arc proposal — stays OFF until explicitly enabled with
# NPC_ENGINE_PREDICTIVE_BOOST=1, per the plan's implementation order
# (step 5: promote to default-on only after two consecutive clean
# e2e_stress cycles).
_PREDICTIVE_DISABLE_ENV = "NPC_ENGINE_PREDICTIVE_DISABLE"
_PREDICTIVE_BOOST_ENV = "NPC_ENGINE_PREDICTIVE_BOOST"


DATA_DIR = NPC_ENGINE_ROOT / "data" / "story_director"
# Default (Ashenvale-compat) asset and runtime paths. StoryDirector uses
# these unless the active world has its own ``<world_dir>/story/`` pack,
# in which case per-world files take over (see ``_resolve_paths``). Tests
# monkey-patch these module-level names directly — the instance captures
# them at ``__init__`` time so patches still take effect.
LORE_FILE = DATA_DIR / "ashenvale_lore.md"
EXAMPLES_FILE = DATA_DIR / "examples.yaml"
EXAMPLES_TERSE_FILE = DATA_DIR / "examples_terse.yaml"
STATE_FILE = DATA_DIR / "state.json"
LEDGER_FILE = DATA_DIR / "fact_ledger.json"
ARCS_FILE = DATA_DIR / "arcs.json"

# Cosine-similarity threshold above which the FactLedger surfaces a
# warning. all-MiniLM-L6-v2 is paraphrase-tuned, so:
#   ≥0.85  near-paraphrase / suspect duplicate
#   0.6-0.85  same topic, different specifics (likely worth surfacing)
#   <0.6  unrelated
# 0.6 picks up thematic recurrence (e.g. two facts about the tax
# collector with different details) without flagging every passing
# mention of a recurring NPC. Tune via observation.
_SIMILARITY_THRESHOLD = 0.6

# NLI confidence above which a flagged pair is reported as a
# contradiction. Empirically the small DeBERTa NLI model is
# hypersensitive — it labels many "topically related but distinct"
# pairs as contradiction with mid-range confidence (0.5-0.8). 0.85
# filters out those false positives while still catching real
# contradictions, which the small model labels with confidence 0.95+.
_NLI_CONTRADICTION_THRESHOLD = 0.85

# Cross-encoder NLI model — small variant runs on CPU at <500ms/pair.
# Lazy-loaded on first contradiction check. Falls back silently if
# sentence-transformers' CrossEncoder isn't available or the model
# can't be downloaded.
_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# Round-robin kind rotation. Python decides the kind so the LLM doesn't
# default to 'event' every tick (which is exactly what 3B models do when
# left to choose). Same split as focus NPC: deterministic planning layer
# + creative writing layer.
_ACTION_KIND_ROTATION = ("event", "quest", "fact")

# Maximum concurrent active/available quests on one NPC — past this, we
# skip 'quest' in rotation so a focus NPC doesn't accumulate unfinished work.
_MAX_QUESTS_PER_NPC = 2

# Phase 4b — per-NPC quest accumulation caps. Distinct from
# ``_MAX_QUESTS_PER_NPC`` (which bounds TOTAL active+available): these
# bound how fast *new* unaccepted quests can pile up on the same NPC.
#
# ``_MAX_UNOFFERED_QUESTS_PER_NPC`` counts only ``available`` (unaccepted)
# quests. Two waiting on the board is already enough; a third quest
# offer feels like noise.
# ``_NPC_QUEST_COOLDOWN_TICKS`` prevents two quest dispatches on the
# same NPC from landing back-to-back even when only one is still
# unaccepted. 10 ticks is roughly 50 real-time minutes at the town
# cadence — long enough that the player has time to engage the first
# quest before a second is offered by the same giver.
#
# Both are overridable at runtime via ``set_quest_pacing(...)`` so a
# game can tighten or loosen pacing without a code change.
_MAX_UNOFFERED_QUESTS_PER_NPC = 2
_NPC_QUEST_COOLDOWN_TICKS = 10

# ── Narrative arc tuning ────────────────────────────────────────
# Minimum ledger entries before the planner will try to cluster. Below
# this, there isn't enough material to find a dense theme.
_ARC_PROPOSAL_MIN_LEDGER_ENTRIES = 4

# Ticks to wait between proposal attempts. Even if the ledger grows, we
# don't want the planner re-scoring every single tick — clustering is
# O(n²) and a stable arc should outlive a few ticks of new content.
_ARC_PROPOSAL_COOLDOWN_TICKS = 5

# Cosine similarity above which two ledger entries are considered part
# of the same theme cluster. Looser than the FactLedger's 0.6 because
# we're detecting thematic overlap, not duplicates.
_ARC_CLUSTER_SIMILARITY = 0.55

# Only look at the most recent N ledger entries when clustering — older
# entries produce stale arcs.
_ARC_CLUSTER_LOOKBACK = 20

# Number of focus-NPC touches required to advance one beat. Tuned
# empirically against Qwen 2.5 3B at actions_per_tick=3 — at the
# original value of 2, the touch counter advanced beats faster than
# the LLM could pace content (T14's physical-confrontation scene
# landed during beat 4/"resolve" instead of beat 3/"confront").
# At 4, beats advance every ~2-3 ticks for 3B rotation cadence,
# which matches the natural arc of set-up → escalation → climax
# → aftermath the model writes. See FINDINGS.md "tuning notes".
_ARC_BEAT_ADVANCE_THRESHOLD = 4

# Maximum focus NPCs per arc — bigger than this and the arc loses
# coherence (every NPC "touches" it).
_ARC_MAX_FOCUS_NPCS = 4

# Maximum concurrent active arcs. A Cardinal-style overseer should run
# 2-3 plot threads in parallel (e.g., main quest + background faction
# tension + slow-burn mystery). At 3 the planner can hold a diverse
# thread set without bloating the prompt with arc blocks — each
# worker only sees the ONE arc relevant to its focus NPC.
_MAX_CONCURRENT_ARCS = 3

# The fixed 4-beat skeleton every arc follows. Descriptive strings so
# they slot directly into the prompt without re-wording.
_ARC_BEAT_SKELETON = (
    "seed — introduce the tension or hint at it without resolving anything",
    "escalate — deepen the stakes with a new wrinkle or complication",
    "confront — force a scene where the tension comes to a head",
    "resolve — show the aftermath and let the thread close cleanly",
)

# Once a bio item has been mentioned this many times by a focus
# worker, it drops out of the bio block entirely until other items
# catch up. Set low enough (2) that cooldown kicks in within 2-3 NPC
# visits even on small bios — large enough that a single sub-action's
# paraphrase doesn't immediately hide the item the model is using.
_BIO_COOLDOWN_THRESHOLD = 2

# Cosine similarity above which a candidate action is considered a
# self-repetition of a recent Director output. Tuned against the 3B
# selfrep_v2 bench where a literal near-duplicate ("hot soup spills
# on her leg" vs "hot soup, spilling it on her leg") sat at 0.73 —
# just under the earlier 0.75 threshold. With the check now restricted
# to same-NPC matches, the false-positive risk is much lower: any
# same-NPC match at 0.70+ is almost certainly the model paraphrasing
# itself rather than gossip propagation or organic continuation.
_SELF_REPETITION_SIMILARITY = 0.70

# Per-tick budget for self-repetition retries. Self-rep retries add
# a full extra LLM call per trigger; on multi-action ticks this can
# stack (the v4 3B bench had 3 ticks with 2 retries each, pushing
# those ticks from ~10s to 20-30s). Budget=1 caps the worst-case
# per-tick latency at "3 workers + 1 retry" worth of LLM calls
# without losing the diversity gains (most retry-eligible ticks
# only produce one retry anyway). Contradiction retries are NOT
# budget-gated — they're rare and more serious.
_MAX_SELF_REP_RETRIES_PER_TICK = 1

# Zone locality rate. When ``_active_zones`` is non-empty,
# ``_pick_focus_npc`` splits available NPCs into in-zone and
# out-of-zone pools and picks from the in-zone pool for (N-1) out
# of N calls. One call in N goes to an out-of-zone NPC so distant
# rumors still seed the FactLedger and propagate via gossip.
# At ``_OUT_OF_ZONE_RATE = 7`` (6 in 7 in-zone, 1 in 7 out) the
# player stays with the locals while the world still breathes.
# Tune down to 4 for "more distant rumors", up to 20 for "tight
# locality" or set to 0 to disable the out-of-zone escape hatch
# entirely. Empty ``_active_zones`` = world-wide mode, this
# constant never fires, existing bench numbers unchanged.
_OUT_OF_ZONE_RATE = 7

# Autonomous lifecycle caps. When autonomous mode is enabled, the
# Director can propose deaths and births on its own. These caps
# prevent runaway mortality/natality over a session. Both count
# from the start of the session (process boot or state reset) and
# are NOT persisted — a restart resets the counters, which is the
# desired behavior for long-running servers.
_MAX_AUTONOMOUS_DEATHS_PER_SESSION = 3
_MAX_AUTONOMOUS_BIRTHS_PER_SESSION = 10

# Small-cast action budget cap. On worlds with at most
# ``_SMALL_CAST_THRESHOLD`` NPCs, the Director forces
# ``actions_per_tick = 1`` regardless of what the caller requested.
# With unrestricted multi-action on a 3-NPC cast, every NPC is
# touched every tick, and the 2026-04-14 fact-consumption bench
# showed each NPC accumulating ~10 Director injections in 10 ticks
# — driving terse-mode per-NPC token delta to 229 vs a 150-token
# shipping budget. Capping to one action per tick on small casts
# lets rotation walk the cast over ``cast_size`` ticks instead of
# piling N injections onto every NPC every tick. Shipping gate for
# PB-style worlds as the live-dialogue default.
_SMALL_CAST_THRESHOLD = 4

# Burst rotation depth: number of consecutive ticks a single NPC
# stays as the architect's first-slot focus before rotation moves on.
# Trades coverage breadth for KV cache reuse depth — keeping the
# planned_focus_ids set stable across K ticks means the world snapshot
# (shared by every sub-action in a tick) hashes to the same prefix so
# llama-cpp's KV cache reuses prior state instead of recomputing from
# scratch. K=4 is the starting guess from the 2026-04-14 syn500 bench
# where the rotation fix jumped latency 5.48s → 13.20s/tick on cold
# misses.
_BURST_ROTATION_DEPTH = 4

# ── Player activity context (Phase 4a) ──────────────────────────
# The game tells the Director what the player is doing via
# ``POST /story/activity``. The Director self-pauses during combat,
# menus, and idle; runs single-action ticks during dialogue and
# wandering; and drops 'quest' from rotation when the player can't
# meaningfully accept a new one (dungeon, dialogue, wandering).
#
# Stored on the instance as a plain string (``self._player_activity``)
# to match ``narration_mode``; the enum is the canonical set of values.


class PlayerActivity(str, Enum):
    IN_TOWN = "in_town"
    IN_DIALOGUE = "in_dialogue"
    IN_DUNGEON = "in_dungeon"
    IN_COMBAT = "in_combat"
    IN_MENU = "in_menu"
    WANDERING = "wandering"
    TRAVELING = "traveling"
    IDLE = "idle"
    UNKNOWN = "unknown"


_PLAYER_ACTIVITY_VALUES: frozenset[str] = frozenset(m.value for m in PlayerActivity)

# Activities that short-circuit tick() to a paused noop — the
# Director has nothing useful to say while the player is in a menu,
# in combat, or idle, and every generated beat would be wasted GPU
# before the player can react to it.
_PAUSED_ACTIVITIES: frozenset[str] = frozenset({
    PlayerActivity.IN_COMBAT.value,
    PlayerActivity.IN_MENU.value,
    PlayerActivity.IDLE.value,
})

# Activities that force actions_per_tick=1 regardless of the caller's
# request. Dialogue fires mid-conversation (one beat is all the player
# can absorb before the next turn) and wandering is low-cadence filler.
_SINGLE_ACTION_ACTIVITIES: frozenset[str] = frozenset({
    PlayerActivity.IN_DIALOGUE.value,
    PlayerActivity.WANDERING.value,
})

# Activities where 'quest' is dropped from the kind rotation. A quest
# offer during dialogue, a dungeon run, or wilderness wander can't be
# meaningfully accepted — the giver isn't present or the context is
# wrong — so we prefer event/fact instead.
_NO_QUEST_ACTIVITIES: frozenset[str] = frozenset({
    PlayerActivity.IN_DIALOGUE.value,
    PlayerActivity.IN_DUNGEON.value,
    PlayerActivity.WANDERING.value,
})

# Suggested delay (seconds) to tell the game client to wait before
# calling tick() again when paused. Small — the game might transition
# out of combat/menu quickly and a polling client should come back
# soon to pick up the first non-paused tick.
_PAUSED_NEXT_TICK_HINT_SECONDS = 10


# ── GPU coordination + next-tick hint (Phase 4c) ────────────────
# Explicit pause state is orthogonal to player activity — the game can
# hold the Director entirely via ``POST /story/pause`` (menu overlay,
# cutscene) and resume with ``POST /story/resume``. Separate from the
# activity-based pause in 4a so the two paths can coexist; the reason
# string lets the client tell them apart (``explicit_pause`` vs
# ``paused: in_combat``).

# Tick budget: rolling-window cap on LLM seconds consumed per real-
# time minute. -1 = no cap (default). The 60-second trailing window
# smooths out bursty sessions — one slow tick doesn't force a pause
# if the prior minute was idle.
_TICK_BUDGET_WINDOW_SECONDS = 60.0

# Cap on the size of the tick-time log. We prune entries older than
# the window anyway; this just prevents pathological growth when a
# caller forgets to consume results.
_TICK_TIME_LOG_CAP = 2048

# Adaptive next-tick hint defaults. These go into the tick response so
# a cooperative game client knows when it's worth calling tick() again.
# Values are in seconds. Calibrated against the 2026-04-16 plan:
#   in_combat:   10   (check back quickly — combat ends fast)
#   in_menu:     30   (menus are brief)
#   idle:        120  (player AFK, stretch it out)
#   wandering:   900  (wilderness, low cadence)
#   in_dungeon:  600  (dungeon runs are longer arcs)
#   in_town / in_dialogue / unknown: 300 (town default)
#   confront-beat accelerator: 60 (climactic pacing)
# Override by editing the ``_NEXT_TICK_HINT_BY_ACTIVITY`` table rather
# than scattering magic numbers in the helper.
_NEXT_TICK_HINT_BY_ACTIVITY: dict[str, int] = {
    "in_combat": 10,
    "in_menu": 30,
    "idle": 120,
    "wandering": 900,
    "in_dungeon": 600,
    "traveling": 300,
    "in_town": 300,
    "in_dialogue": 300,
    "unknown": 300,
}
_NEXT_TICK_HINT_DEFAULT = 300
_NEXT_TICK_HINT_CONFRONT = 60

# Beat index at which an active arc is considered to be in its
# climactic phase. Arc beat skeleton is (seed=0, escalate=1,
# confront=2, resolve=3). ``>= 2`` covers confront + resolve.
_ARC_CONFRONT_BEAT_INDEX = 2

# ── Phase 3a main-line weighting ────────────────────────────────
# Focus-selection preference for main-line givers. FACTOR=2 gives a
# 2:1 preference: out of every (FACTOR + 1) focus picks, FACTOR of
# them go to main-line NPCs (when any are in the pool) and 1 goes
# to a non-main-line NPC. Empty main-line cast = this layer is a
# no-op, focus picks are identical to the pre-3a rotation.
_MAIN_LINE_WEIGHT_FACTOR = 2


# Only consider self-repetition against recent same-NPC entries.
# Older matches are stale and may legitimately be echoed; cross-NPC
# similarity is gossip propagation, not self-repetition.
#
# Tuned against the v3 bench where T15 Bess repeated T1 Bess's
# "hot soup spilling" scene at sim=0.79 but was ignored because
# 14 ticks > 8-tick lookback. With 15-tick sessions as the norm
# and the ledger capped at 200 entries (~60 ticks of 3-worker
# content), 20 ticks covers "most of a session" while still letting
# long-range plot echoes through (Kael mentioning Tam at T5 and
# again at T50 is continuity, not repetition).
_SELF_REPETITION_LOOKBACK_TICKS = 20

# Common English stopwords to strip from bio-mention detection. Short
# list — we want to avoid false positives from generic structural
# words, not build a linguistically accurate stopword set.
_BIO_STOPWORDS = frozenset({
    "the", "and", "that", "this", "with", "from", "have", "his", "her",
    "she", "him", "them", "they", "their", "there", "these", "those",
    "when", "what", "which", "where", "will", "would", "could", "should",
    "been", "being", "into", "onto", "about", "after", "before", "over",
    "under", "some", "such", "just", "only", "also", "than", "then",
    "more", "most", "many", "much", "very", "like", "does", "doing",
    "know", "knows", "knew",
})

class ContradictionChecker:
    """
    Wraps a small NLI cross-encoder to classify a (premise, hypothesis)
    pair as contradiction / entailment / neutral. Used by the FactLedger
    on flagged similarity pairs to elevate "these two are similar" into
    "these two contradict each other".

    First call lazy-loads the model (~140MB download on first run).
    Falls back to no-op if sentence-transformers' CrossEncoder is
    unavailable or the model can't be loaded — the FactLedger still
    works without it.

    Label order: ['contradiction', 'entailment', 'neutral'] per the
    ``cross-encoder/nli-deberta-v3-*`` model cards.
    """

    LABELS = ("contradiction", "entailment", "neutral")

    def __init__(self, model_name: str = _NLI_MODEL_NAME,
                 contradiction_threshold: float = _NLI_CONTRADICTION_THRESHOLD):
        self.model_name = model_name
        self.contradiction_threshold = contradiction_threshold
        self._model = None  # None = not yet attempted; False = unavailable
        # Single-slot (premise, hypothesis) → result cache. Profile
        # showed NLI is the single biggest CPU consumer (5s over 15
        # ticks of stubbed bench). The same (premise, hypothesis)
        # pair gets classified up to 3 times per sub-action —
        # contradiction precheck, self-rep precheck, and ledger.add
        # all route through here with identical inputs. One-slot
        # cache catches the redundant calls with zero API churn.
        self._cached_key: Optional[tuple[str, str]] = None
        self._cached_result: Optional[dict] = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info(f"ContradictionChecker loaded NLI model: {self.model_name}")
            except Exception as e:
                logger.warning(f"ContradictionChecker NLI model unavailable: {e}")
                self._model = False
        return self._model if self._model is not False else None

    def check(self, premise: str, hypothesis: str) -> Optional[dict]:
        """
        Classify the pair. Returns a dict with the predicted label, its
        confidence, and the full score breakdown — or ``None`` if the
        model isn't available. One-slot cache for repeated identical
        pair lookups (see __init__ docstring for the motivation).
        """
        if not premise or not hypothesis:
            return None
        key = (premise, hypothesis)
        if key == self._cached_key and self._cached_result is not None:
            return self._cached_result
        model = self.model
        if model is None:
            return None
        try:
            raw = model.predict([(premise, hypothesis)])[0]
            scores = [float(s) for s in raw]
        except Exception as e:
            logger.error(f"ContradictionChecker.predict failed: {e}")
            return None

        if len(scores) != len(self.LABELS):
            return None

        # Softmax for nicer probabilities (CrossEncoder returns logits)
        try:
            import math
            mx = max(scores)
            exps = [math.exp(s - mx) for s in scores]
            total = sum(exps)
            probs = [e / total for e in exps] if total else scores
        except Exception:
            probs = scores

        label_idx = max(range(len(probs)), key=lambda i: probs[i])
        result = {
            "label": self.LABELS[label_idx],
            "confidence": round(float(probs[label_idx]), 3),
            "is_contradiction": (
                self.LABELS[label_idx] == "contradiction"
                and float(probs[label_idx]) >= self.contradiction_threshold
            ),
            "scores": {
                self.LABELS[i]: round(float(probs[i]), 3)
                for i in range(len(self.LABELS))
            },
        }
        self._cached_key = key
        self._cached_result = result
        return result


class FactLedger:
    """
    Tracks every fact-shaped content the Story Director injects (events,
    quests, facts) with embeddings, and surfaces similarity warnings when
    new content is too close to existing content.

    v0 deliberately does NOT try to classify entailment vs. contradiction
    — that's an NLI problem and the small models in your stack would be
    unreliable at it. Surfacing high-similarity matches is enough to:

      1. Catch the Director recycling the same beat under a different
         npc_id (a common mode on Qwen 3B — see T6→T7 Mara chain).
      2. Give a future contradiction-detection layer something to anchor on.
      3. Let you spot themes drifting before they become contradictions.

    The embedder is lazy-loaded — first call pays the ~4s cost; later
    calls are warm. If sentence-transformers is unavailable the ledger
    silently no-ops so the Director still works.
    """

    def __init__(self, storage_path: Path, threshold: float = _SIMILARITY_THRESHOLD,
                 contradiction_checker: Optional["ContradictionChecker"] = None):
        self.storage_path = storage_path
        self.threshold = threshold
        self.entries: list[dict] = []
        self._embedder = None  # None = not yet attempted; False = unavailable
        self._np = None
        # Single-slot encode cache. Profile showed that each
        # sub-action's text gets encoded 3 times in a row — once in
        # _precheck_contradiction, once in _precheck_self_repetition,
        # and once in ledger.add(). All 3 embed the same string. A
        # one-slot cache keyed on text catches all the follow-ups
        # with zero plumbing through the caller-side API.
        self._cached_encode_text: Optional[str] = None
        self._cached_encode_vec = None
        # NLI checker — runs on flagged similarity pairs to upgrade
        # "similar" to "contradiction". Optional; ledger works without it.
        self.contradiction_checker = contradiction_checker or ContradictionChecker()
        self._load()

    # ── Lazy resource loading ────────────────────────────────────

    @property
    def embedder(self):
        if self._embedder is None:
            try:
                from densanon.core.embeddings.embedder import get_embedder
                self._embedder = get_embedder()
            except Exception as e:
                logger.warning(f"FactLedger embedder unavailable: {e}")
                self._embedder = False
        return self._embedder if self._embedder is not False else None

    @property
    def np(self):
        if self._np is None:
            try:
                import numpy as np  # noqa: WPS433
                self._np = np
            except Exception:
                self._np = False
        return self._np if self._np is not False else None

    # ── Public API ──────────────────────────────────────────────

    def check(self, text: str,
              restrict_to_npc: Optional[str] = None) -> Optional[dict]:
        """
        Compute a similarity + NLI warning for ``text`` against the
        existing ledger, WITHOUT storing it. Used by the Director's
        pre-dispatch retry path: if a candidate action would conflict
        with prior content, retry before mutating the world.

        When ``restrict_to_npc`` is set, the similarity comparison
        only considers ledger entries with a matching ``npc_id``. This
        is how the self-repetition precheck distinguishes "the same
        NPC is saying the same thing again" (a copy loop) from "a
        different NPC is referencing earlier content" (gossip
        propagation, which is legitimate).

        Returns a warning dict (with ``nli`` block and ``contradiction``
        flag if applicable) or ``None`` if no match exceeds the threshold.
        """
        embedding = self._encode(text)
        if embedding is None:
            return None
        warning = self._check_similarity(embedding, restrict_to_npc=restrict_to_npc)
        if warning is None:
            return None
        nli = self.contradiction_checker.check(
            premise=warning["matches_text"],
            hypothesis=text,
        )
        if nli is not None:
            warning["nli"] = nli
            if nli.get("is_contradiction"):
                warning["contradiction"] = True
        return warning

    def add(
        self,
        text: str,
        npc_id: str,
        kind: str,
        tick: int,
        *,
        source_ledger_entries: Optional[list[int]] = None,
        suggested_by: Optional[str] = None,
        subject_identity: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Add a new entry to the ledger and return a similarity warning if
        any prior entry exceeds the threshold. Returns None when no
        warning fires (or when embeddings are unavailable).

        Provenance markers (Phase 12 universal pattern, 2026-04-15):

        - ``source_ledger_entries``: indices of prior ledger entries this
          one was derived from (e.g., an arc-beat fact shaped by two
          earlier events). Empty/None for origin facts. Lets downstream
          tooling reconstruct the causal chain without re-embedding
          every pair.
        - ``suggested_by``: identifier of the subsystem that produced
          this injection — typically the Director sub-action name
          ("arc_advance", "quest_injection", "event_drop",
          "player_reaction"). Empty/None for pre-Phase-12 entries in
          the loaded ledger. Lets contradiction review and FINDINGS
          post-mortems filter "which part of the Director produced
          this?" without heuristic text matching.

        Both fields are optional. Callers that don't pass them produce
        entries identical to pre-Phase-12 behavior. Entries loaded from
        an older ``fact_ledger.json`` that predates the fields simply
        lack the keys — consumers should ``dict.get(...)`` rather than
        ``[...]`` when reading them.

        Phase 5a adds ``subject_identity`` with the same contract: the
        key is only present on entries where the caller attributed the
        deed to a specific identity (``jordan``, ``stranger``,
        ``hooded_figure``, ...). Reputation queries treat missing =
        default ``"player"``.
        """
        if not text or not isinstance(text, str):
            return None
        embedding = self._encode(text)
        if embedding is None:
            return None

        warning = self._check_similarity(embedding)

        # If we have a similarity match, run NLI to see if it's a real
        # contradiction. Only a few hundred ms on CPU per check, and only
        # fires when there's already a flagged pair — so the cost is
        # bounded by how often the Director recycles plot threads.
        if warning is not None:
            nli = self.contradiction_checker.check(
                premise=warning["matches_text"],
                hypothesis=text,
            )
            if nli is not None:
                warning["nli"] = nli
                if nli.get("is_contradiction"):
                    warning["contradiction"] = True

        entry: dict = {
            "text": text[:400],
            "embedding": embedding.tolist(),
            "npc_id": npc_id,
            "kind": kind,
            "tick": tick,
        }
        if source_ledger_entries:
            entry["source_ledger_entries"] = list(source_ledger_entries)
        if suggested_by:
            entry["suggested_by"] = suggested_by
        # Phase 5a — subject_identity tags player-related entries with
        # whichever identity produced the deed/fact (``jordan``,
        # ``stranger``, ``hooded_figure``, ...). Missing key = legacy
        # entry; reputation queries treat that as ``"player"``.
        if subject_identity:
            entry["subject_identity"] = subject_identity

        # Observer-conflict (OSCToM insight, 2026-05-22): when a NEW
        # observation/fact CONTRADICTS an existing belief for the SAME
        # subject, we want to flag the conflict rather than let it slide
        # in silently next to the stale belief. This models the gap
        # between what a character now OBSERVES and what they previously
        # BELIEVED (e.g. saw the player steal, but believed them
        # trustworthy). We do NOT delete or overwrite the prior belief —
        # the ledger is append-only — we just mark the incoming entry so
        # a downstream "belief update" layer (out of scope for v1) can
        # surface it.
        #
        # Gating, deliberately tight:
        #   • the NLI must have fired ``is_contradiction`` (real semantic
        #     conflict, not mere lexical similarity), and
        #   • the prior belief must concern the SAME subject — same
        #     ``subject_identity`` when both carry one, else falling back
        #     to same ``npc_id``. Cross-subject contradictions (two
        #     different NPCs' unrelated facts) are NOT observer-conflicts.
        # Additive: the key is only ever set when both conditions hold,
        # so the non-conflict path and every legacy consumer are
        # untouched.
        if warning is not None and warning.get("contradiction"):
            prior_idx = warning.get("matches_index")
            prior_subject = warning.get("matches_subject_identity")
            prior_npc = warning.get("matches_npc")
            if subject_identity is not None and prior_subject is not None:
                same_subject = (subject_identity == prior_subject)
            else:
                # No explicit identity on one/both sides — fall back to
                # the owning NPC as the subject proxy.
                same_subject = (prior_npc == npc_id)
            if same_subject:
                entry["observer_conflict"] = True
                entry["conflicts_with"] = {
                    "index": prior_idx,
                    "text": warning.get("matches_text"),
                    "tick": warning.get("matches_tick"),
                    "npc_id": prior_npc,
                    "subject_identity": prior_subject,
                }
                # Mirror onto the returned warning so the dispatch-path
                # caller can react in-line without re-reading the entry.
                warning["observer_conflict"] = True

        self.entries.append(entry)
        # Bound memory — keep last 200 entries (largest a typical session reaches)
        if len(self.entries) > 200:
            self.entries = self.entries[-200:]
        self._save()
        return warning

    def _encode(self, text: str):
        """
        Lazy-encode helper shared by ``check`` and ``add``. Uses a
        one-slot cache so back-to-back calls with the same text
        (the common pattern during a sub-action: contradiction
        precheck → self-rep precheck → ledger.add all embed the
        same candidate text) reuse the vector without re-running
        the embedder. Saves ~⅔ of encode work per sub-action —
        the profile showed 134 encodes across 15 ticks dropping
        cleanly to ~45 with this cache.
        """
        if not text or not isinstance(text, str):
            return None
        if text == self._cached_encode_text and self._cached_encode_vec is not None:
            return self._cached_encode_vec
        embedder = self.embedder
        if embedder is None or self.np is None:
            return None
        try:
            vec = embedder.encode(text, normalize_embeddings=True)
        except Exception as e:
            logger.error(f"FactLedger encode failed: {e}")
            return None
        self._cached_encode_text = text
        self._cached_encode_vec = vec
        return vec

    def reset(self) -> None:
        self.entries = []
        for path in (self.storage_path, self._embeddings_path):
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass

    def stats(self) -> dict:
        return {
            "entry_count": len(self.entries),
            "threshold": self.threshold,
            "embedder_loaded": isinstance(self._embedder, object)
                                and self._embedder not in (None, False),
            "nli_loaded": (self.contradiction_checker._model is not None
                            and self.contradiction_checker._model is not False),
        }

    # ── Internals ───────────────────────────────────────────────

    def _check_similarity(self, new_embedding,
                           restrict_to_npc: Optional[str] = None) -> Optional[dict]:
        if not self.entries:
            return None
        np = self.np
        if np is None:
            return None

        # Optionally filter to entries tied to a specific NPC. This
        # is what makes the self-repetition precheck NPC-aware: it
        # looks for the best *same-NPC* match rather than the overall
        # top match, which may be cross-NPC gossip propagation.
        if restrict_to_npc is not None:
            candidates = [
                e for e in self.entries
                if e.get("npc_id") == restrict_to_npc
            ]
        else:
            candidates = self.entries
        if not candidates:
            return None

        try:
            existing = np.array([e["embedding"] for e in candidates])
            sims = existing @ new_embedding  # cosine, vectors normalized
            max_idx = int(sims.argmax())
            max_sim = float(sims[max_idx])
        except Exception as e:
            logger.error(f"FactLedger similarity check failed: {e}")
            return None
        if max_sim < self.threshold:
            return None
        match = candidates[max_idx]
        # ``matches_index`` is the position of the matched entry in the
        # *full* self.entries list (not the filtered candidates list), so
        # downstream consumers can reference the prior belief by stable
        # index. ``matches_subject_identity`` mirrors the entry's optional
        # subject_identity (absent key = legacy "player").
        try:
            matches_index = self.entries.index(match)
        except ValueError:
            matches_index = None
        return {
            "similarity": round(max_sim, 3),
            "matches_text": match["text"][:240],
            "matches_npc": match["npc_id"],
            "matches_kind": match["kind"],
            "matches_tick": match["tick"],
            "matches_index": matches_index,
            "matches_subject_identity": match.get("subject_identity"),
        }

    @property
    def _embeddings_path(self) -> Path:
        """Binary sidecar for embeddings — same stem as the main JSON
        file but with a ``.embeddings.npy`` suffix. Keeps the JSON
        human-readable while pushing the large numeric payload to a
        compact float32 binary that's ~10x smaller on disk."""
        return self.storage_path.with_suffix(".embeddings.npy")

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            entries = data.get("entries", [])[-200:]
        except Exception as e:
            logger.warning(f"FactLedger load failed: {e}")
            return

        # Try to load embeddings from the binary sidecar. Entries that
        # already have an ``embedding`` key inline (old-format saves
        # pre-compression) skip the sidecar lookup and use the inline
        # data directly. This makes the upgrade path zero-touch: an
        # existing fact_ledger.json from before this commit loads
        # cleanly and will be re-saved in the new format on the next
        # ``add``.
        missing_embedding = any("embedding" not in e for e in entries)
        if missing_embedding and self._embeddings_path.exists():
            try:
                import numpy as np  # noqa: WPS433
                embeddings = np.load(self._embeddings_path)
            except Exception as e:
                logger.warning(f"FactLedger embeddings sidecar load failed: {e}")
                embeddings = None
            if embeddings is not None:
                # Match by index — the sidecar was written in the same
                # order as entries and both are trimmed together on save.
                count = min(len(entries), embeddings.shape[0])
                offset = embeddings.shape[0] - count  # align to tail
                for i in range(count):
                    if "embedding" not in entries[i]:
                        entries[i]["embedding"] = embeddings[offset + i].tolist()

        self.entries = entries

    def _save(self) -> None:
        """
        Save in two files:

        - ``fact_ledger.json``: entry metadata (text, npc_id, kind,
          tick) without embeddings — stays human-readable and small.
        - ``fact_ledger.embeddings.npy``: matched-index float32
          embeddings array, written via numpy's native format.

        The pair is always rewritten together so a crash mid-save
        leaves both files matching the same state, or neither.
        Numpy is a hard dependency of the ledger (required by the
        similarity check) so there's no conditional code path.
        """
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Split metadata from embeddings
            metadata_entries: list[dict] = []
            embeddings: list[list[float]] = []
            for e in self.entries:
                meta = {k: v for k, v in e.items() if k != "embedding"}
                metadata_entries.append(meta)
                emb = e.get("embedding")
                if emb is not None:
                    embeddings.append(emb)

            # Write JSON metadata
            self.storage_path.write_text(
                json.dumps({"entries": metadata_entries},
                            indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Write binary sidecar (float32 — cosine similarity is
            # well-approximated at this precision, and it halves the
            # byte count vs float64)
            try:
                import numpy as np  # noqa: WPS433
                if embeddings:
                    arr = np.asarray(embeddings, dtype=np.float32)
                    np.save(self._embeddings_path, arr,
                             allow_pickle=False)
                elif self._embeddings_path.exists():
                    # All entries gone — remove stale sidecar
                    try:
                        self._embeddings_path.unlink()
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"FactLedger embeddings sidecar save failed: {e}")
        except Exception as e:
            logger.error(f"FactLedger save failed: {e}")


@dataclass
class NarrativeArc:
    """
    A multi-tick story thread the Director commits to. Arcs give a session
    shape beyond round-robin local decisions: a theme, a cast, and a fixed
    4-beat progression (seed → escalate → confront → resolve).

    Arcs are proposed deterministically from the FactLedger — the planner
    clusters recent entries by embedding similarity and commits the densest
    cluster as an arc. The theme and focus NPCs are derived from that
    cluster, so the arc is always grounded in content the Director has
    already produced.

    ``touches_since_last_advance`` accumulates every dispatch that
    targets a cast NPC since the last beat advance. It's the input to
    ``ArcPlanner.advance_if_beat_met``, replacing the old approach
    which walked ``recent_decisions`` (capped at 5 ticks) and couldn't
    fit enough touches to advance single-NPC arcs at all. The counter
    has no upper bound and resets on beat advance.
    """

    id: str
    theme: str
    focus_npcs: list[str]
    beat_goals: list[str]
    current_beat: int = 0
    status: str = "active"  # active | resolved | abandoned
    started_at_tick: int = 0
    last_advanced_at_tick: int = 0
    touches_since_last_advance: int = 0

    @property
    def current_beat_label(self) -> str:
        """Short label for the current beat (the word before the em-dash)."""
        goal = self.current_beat_goal
        if goal is None:
            return "done"
        return goal.split(" — ", 1)[0]

    @property
    def current_beat_goal(self) -> Optional[str]:
        if 0 <= self.current_beat < len(self.beat_goals):
            return self.beat_goals[self.current_beat]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_beat >= len(self.beat_goals)


class ArcPlanner:
    """
    Owns the narrative arcs, proposes new ones from the ledger, and
    advances them as the story progresses.

    Tracks up to ``_MAX_CONCURRENT_ARCS`` active arcs in parallel.
    Each worker sees only the ONE arc relevant to its focus NPC (via
    ``arc_for_focus``) — so the prompt never bloats with multiple arc
    blocks, but the session can hold multiple plot threads alive at
    once.

    Deterministic proposal (v1): greedy-cluster the recent FactLedger
    entries by cosine similarity. The densest cluster becomes an arc —
    theme = the center entry's text, focus_npcs = the unique NPC ids
    across the cluster. Fixed 4-beat skeleton, touch-counter advancement.
    NPCs already covered by an active arc's cast are excluded from new
    proposals so the same thread doesn't spawn a duplicate arc.

    No LLM call is made during proposal or advancement — the planner is
    pure Python. A future v2 could add an LLM-theming pass for richer
    theme strings, but the clusters themselves are already grounded in
    Director-written content so themes are never "made up from nothing."
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.arcs: list[NarrativeArc] = []
        self.active_arc_ids: list[str] = []
        self._last_proposal_attempt_tick: int = 0
        self._load()

    # ── Public API ──────────────────────────────────────────────

    def active_arcs(self) -> list[NarrativeArc]:
        """
        Return every currently-active arc. Filters out resolved arcs
        and drops stale ids from ``active_arc_ids`` so the list stays
        consistent.
        """
        result: list[NarrativeArc] = []
        still_active: list[str] = []
        arc_by_id = {a.id: a for a in self.arcs}
        for arc_id in self.active_arc_ids:
            arc = arc_by_id.get(arc_id)
            if arc is not None and arc.status == "active":
                result.append(arc)
                still_active.append(arc_id)
        if still_active != self.active_arc_ids:
            self.active_arc_ids = still_active
        return result

    def arc_for_focus(self, focus_npc: Optional[str]) -> Optional[NarrativeArc]:
        """
        Pick the one arc this worker's prompt should reference, given
        its forced focus NPC. If the NPC is in multiple active arc
        casts, prefer the arc with the fewest touches since its last
        advance (the weakest thread, so each tick can help the
        neglected one).
        """
        if not focus_npc:
            return None
        candidates = [
            a for a in self.active_arcs() if focus_npc in a.focus_npcs
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # Weakest-thread preference — helps starved arcs catch up
        return min(candidates, key=lambda a: a.touches_since_last_advance)

    def maybe_propose(self, ledger: "FactLedger", available_npcs: list[str],
                       current_tick: int,
                       edge_prior_boost: Optional[dict[str, float]] = None) -> Optional[NarrativeArc]:
        """
        Attempt to propose a new arc if there's headroom under the
        concurrent-arc cap and the cooldown has elapsed. Returns the
        new arc (and appends it to ``active_arc_ids``) or ``None`` if
        nothing was proposed.

        ``edge_prior_boost`` (predictive lane, optional): per-NPC
        edge-prior means in [0, 1]. Passed through to
        ``_propose_from_ledger`` where it soft-prefers cluster
        centers whose NPC the filters expect to be active next tick.
        None/empty = scoring identical to pre-predictive behavior.
        """
        active_now = self.active_arcs()
        if len(active_now) >= _MAX_CONCURRENT_ARCS:
            return None
        if current_tick - self._last_proposal_attempt_tick < _ARC_PROPOSAL_COOLDOWN_TICKS:
            return None
        self._last_proposal_attempt_tick = current_tick

        # Exclude NPCs already in active arc casts so a proposal
        # can't spawn a duplicate thread on the same cast.
        used_npcs: set[str] = set()
        for arc in active_now:
            used_npcs.update(arc.focus_npcs)
        eligible_npcs = [nid for nid in available_npcs if nid not in used_npcs]
        if not eligible_npcs:
            return None

        arc = self._propose_from_ledger(ledger, eligible_npcs, current_tick,
                                         exclude_npcs=used_npcs,
                                         edge_prior_boost=edge_prior_boost)
        if arc is not None:
            self.arcs.append(arc)
            self.active_arc_ids.append(arc.id)
            logger.info(
                f"ArcPlanner proposed arc {arc.id}: '{arc.theme[:60]}' "
                f"(focus={arc.focus_npcs}, concurrent={len(self.active_arc_ids)}"
                f"/{_MAX_CONCURRENT_ARCS})"
            )
        self.save()
        return arc

    def record_cast_touch(self, npc_id: Optional[str]) -> None:
        """
        Bump the touch counter on every active arc whose cast
        contains ``npc_id``. Called from
        ``StoryDirector._run_single_action`` after every successful
        non-noop dispatch. Proposal excludes NPCs already in active
        casts, so in normal operation an NPC is in at most one
        active arc's cast — but we iterate defensively in case a
        future change allows cast overlap.
        """
        if not npc_id:
            return
        for arc in self.active_arcs():
            if npc_id in arc.focus_npcs:
                arc.touches_since_last_advance += 1

    def on_cast_death(self, npc_id: str, current_tick: int) -> dict:
        """
        React to an NPC death by transitioning any arcs whose cast
        includes the deceased NPC. Three cases:

        1. Arc is past the ``confront`` beat (current_beat >= 2):
           mark resolved. The death IS the resolution.
        2. Arc is at seed or escalate, and the deceased NPC was the
           SOLE cast member: collapse to aftermath-only with status
           "resolved" and a note in the theme.
        3. Arc is at seed or escalate with multiple cast members:
           drop the deceased NPC from the cast list, keep the arc
           active, and reset touch counter to trigger a new beat's
           worth of non-dead-npc touches before advancing.

        Returns a summary dict so the Director can surface the
        transition in the death dispatch result.
        """
        affected: list[dict] = []
        for arc in list(self.active_arcs()):
            if npc_id not in arc.focus_npcs:
                continue
            original_cast = list(arc.focus_npcs)
            arc.focus_npcs = [n for n in arc.focus_npcs if n != npc_id]
            if arc.current_beat >= 2 or not arc.focus_npcs:
                # Either already past confront OR deceased was sole
                # cast — arc resolves now.
                arc.status = "resolved"
                arc.last_advanced_at_tick = current_tick
                if arc.id in self.active_arc_ids:
                    self.active_arc_ids.remove(arc.id)
                affected.append({
                    "arc_id": arc.id,
                    "transition": "resolved_by_death",
                    "original_cast": original_cast,
                })
                logger.info(
                    f"ArcPlanner resolved {arc.id} after death of {npc_id}"
                )
            else:
                # Active arc with remaining cast — drop the deceased
                # and reset the touch counter so we re-earn the next
                # beat advance with the smaller cast.
                arc.touches_since_last_advance = 0
                affected.append({
                    "arc_id": arc.id,
                    "transition": "cast_trimmed",
                    "original_cast": original_cast,
                    "new_cast": list(arc.focus_npcs),
                })
                logger.info(
                    f"ArcPlanner trimmed {npc_id} from {arc.id} cast "
                    f"(remaining: {arc.focus_npcs})"
                )
        if affected:
            self.save()
        return {"arcs_affected": affected}

    def advance_if_beat_met(self, current_tick: int) -> int:
        """
        Iterate active arcs and advance each one that has met its
        touch threshold. Returns the number of arcs that advanced
        this call (0, 1, or more). Resolved arcs are removed from
        ``active_arc_ids``.
        """
        advanced_count = 0
        # Iterate over a snapshot — the loop mutates active_arc_ids
        # via active_arcs() cleanup and direct removals.
        for arc in list(self.active_arcs()):
            if arc.touches_since_last_advance < _ARC_BEAT_ADVANCE_THRESHOLD:
                continue
            arc.current_beat += 1
            arc.last_advanced_at_tick = current_tick
            arc.touches_since_last_advance = 0
            advanced_count += 1
            logger.info(
                f"ArcPlanner advanced {arc.id} to beat {arc.current_beat} "
                f"({arc.current_beat_label})"
            )
            if arc.is_complete:
                arc.status = "resolved"
                if arc.id in self.active_arc_ids:
                    self.active_arc_ids.remove(arc.id)
                logger.info(f"ArcPlanner resolved {arc.id}")
        if advanced_count > 0:
            self.save()
        return advanced_count

    def stats(self) -> dict:
        active = self.active_arcs()
        return {
            "arc_count": len(self.arcs),
            "active_arc_ids": list(self.active_arc_ids),
            "active_count": len(active),
            "active_summaries": [
                {
                    "id": a.id,
                    "theme": a.theme[:80],
                    "cast": a.focus_npcs,
                    "beat": a.current_beat_label,
                    "beat_index": a.current_beat,
                    "touches": a.touches_since_last_advance,
                }
                for a in active
            ],
            "last_proposal_attempt_tick": self._last_proposal_attempt_tick,
        }

    def reset(self) -> None:
        self.arcs = []
        self.active_arc_ids = []
        self._last_proposal_attempt_tick = 0
        if self.storage_path.exists():
            try:
                self.storage_path.unlink()
            except Exception:
                pass

    # ── Internals ───────────────────────────────────────────────

    def _propose_from_ledger(self, ledger: "FactLedger",
                              available_npcs: list[str],
                              current_tick: int,
                              exclude_npcs: Optional[set[str]] = None,
                              edge_prior_boost: Optional[dict[str, float]] = None) -> Optional[NarrativeArc]:
        """
        Greedy clustering over the most recent ledger entries. Pick the
        entry with the most high-similarity neighbors (among entries
        whose NPC is NOT in ``exclude_npcs``) as the cluster center,
        then collect its neighbors. Theme = center entry's text,
        focus NPCs = unique NPC ids in the cluster filtered against
        ``available_npcs``.

        ``exclude_npcs`` is how multi-arc proposal prevents a new arc
        from forming around NPCs already covered by an active arc's
        cast. Without this filter, the densest cluster would keep
        being whatever plot thread is most saturated, and proposals
        would duplicate each other.

        ``edge_prior_boost`` maps npc_id -> edge-prior mean in [0, 1].
        A candidate center whose NPC the predictive lane expects to be
        active next tick gets its neighbor-count score multiplied by
        at most (1 + EDGE_PRIOR_BOOST_CAP) — a sub-10% nudge that can
        break ties or flip near-equal clusters but never dominates the
        greedy-cluster heuristic. Cold priors (0.5) and missing NPCs
        produce a factor of exactly 1.0, i.e. pre-predictive scoring.
        """
        if len(ledger.entries) < _ARC_PROPOSAL_MIN_LEDGER_ENTRIES:
            return None
        np = ledger.np
        if np is None:
            return None

        excluded = set(exclude_npcs or ())
        # Pull recent entries but skip any whose NPC is already covered
        # by an active arc — clustering then runs only over "unused"
        # content.
        recent = [
            e for e in ledger.entries[-_ARC_CLUSTER_LOOKBACK:]
            if e.get("npc_id") and e.get("npc_id") not in excluded
        ]
        if len(recent) < _ARC_PROPOSAL_MIN_LEDGER_ENTRIES:
            return None

        try:
            embeddings = np.array([e["embedding"] for e in recent])
            # Pairwise cosine — entries are already L2-normalized on encode
            sims = embeddings @ embeddings.T
            # Zero the diagonal so self-similarity doesn't dominate
            for i in range(len(sims)):
                sims[i][i] = 0.0
            neighbor_counts = (sims >= _ARC_CLUSTER_SIMILARITY).sum(axis=1)
            # Predictive-lane boost: multiply each candidate center's
            # score by (1 + min(CAP, max(0, prior - 0.5))). Priors at
            # or below the Beta(1,1) cold value of 0.5 contribute
            # nothing, so a cold predictive layer reproduces the raw
            # neighbor-count argmax bit-for-bit.
            scores = neighbor_counts.astype(float)
            if edge_prior_boost:
                from npc_engine.predictive_factledger import EDGE_PRIOR_BOOST_CAP
                for i, e in enumerate(recent):
                    prior = edge_prior_boost.get(e.get("npc_id") or "")
                    if prior is not None:
                        boost = min(EDGE_PRIOR_BOOST_CAP, max(0.0, float(prior) - 0.5))
                        scores[i] *= (1.0 + boost)
            best_idx = int(scores.argmax())
            if int(neighbor_counts[best_idx]) == 0:
                return None
            cluster_indices = [best_idx] + [
                i for i in range(len(sims))
                if i != best_idx and sims[best_idx][i] >= _ARC_CLUSTER_SIMILARITY
            ]
        except Exception as e:
            logger.error(f"ArcPlanner cluster failed: {e}")
            return None

        cluster_entries = [recent[i] for i in cluster_indices]
        theme = str(cluster_entries[0].get("text") or "")[:160]
        if not theme:
            return None

        # Focus NPCs: unique ids in cluster, in order of first appearance,
        # filtered to the currently-available roster AND not in the
        # excluded set. Dropping "all" and anything not in the world
        # keeps the arc grounded.
        focus_npcs: list[str] = []
        seen: set[str] = set()
        avail = set(available_npcs)
        for e in cluster_entries:
            nid = e.get("npc_id") or ""
            if (nid and nid in avail and nid not in seen
                    and nid not in excluded):
                seen.add(nid)
                focus_npcs.append(nid)
        if not focus_npcs:
            return None
        focus_npcs = focus_npcs[:_ARC_MAX_FOCUS_NPCS]

        arc_id = f"arc_t{current_tick}_{int(time.time())}"
        return NarrativeArc(
            id=arc_id,
            theme=theme,
            focus_npcs=focus_npcs,
            beat_goals=list(_ARC_BEAT_SKELETON),
            current_beat=0,
            status="active",
            started_at_tick=current_tick,
            last_advanced_at_tick=current_tick,
        )

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self.arcs = [NarrativeArc(**a) for a in data.get("arcs", [])]
            # Prefer the new list field; fall back to the legacy
            # single-arc field so state.json files from pre-multiarc
            # sessions still load cleanly.
            if "active_arc_ids" in data:
                self.active_arc_ids = [
                    str(x) for x in (data.get("active_arc_ids") or [])
                ]
            else:
                legacy = data.get("active_arc_id")
                self.active_arc_ids = [str(legacy)] if legacy else []
            self._last_proposal_attempt_tick = int(data.get("last_proposal_attempt_tick", 0))
        except Exception as e:
            logger.warning(f"ArcPlanner load failed: {e}")
            self.arcs = []
            self.active_arc_ids = []

    def save(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(
                json.dumps({
                    "arcs": [asdict(a) for a in self.arcs],
                    "active_arc_ids": list(self.active_arc_ids),
                    "last_proposal_attempt_tick": self._last_proposal_attempt_tick,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"ArcPlanner save failed: {e}")


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Find the first balanced JSON object in ``text`` by scanning braces.

    A regex like ``\\{.*\\}`` is greedy and will glue multiple objects
    together when the model emits more than one — that produces invalid
    JSON. This scans the string char by char, respecting string escapes,
    and returns the first balanced ``{...}`` substring (or None).
    """
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start:i + 1]
    return None


class StoryDirector:
    """World-level narrative overseer."""

    def __init__(self, engine: "NPCEngine"):
        self.engine = engine
        self.tick_count: int = 0
        self.last_tick_at: Optional[str] = None
        self.recent_decisions: list[dict] = []  # last 5 actions
        # Architect's PLANNED focus per NPC, keyed by NPC id, value is
        # the most recent tick number at which that NPC was selected as
        # a focus by the architect. UNBOUNDED in entry count — capped
        # only by the cast size — because the rotation needs to
        # remember which NPCs were touched many ticks ago so it can
        # walk a 500-NPC world in true round-robin order. The 5-tick
        # recent_decisions cap was the bug at scale: after 5 ticks the
        # earliest NPCs fell off and rotation cycled the same ~15
        # NPCs forever.
        #
        # This is a separate trail from recent_decisions so any future
        # code path that mutates action.npc_id after _enforce_focus_npc
        # can't pollute rotation — rotation reads the architect's
        # INTENDED focus, written here before workers run.
        self._npc_last_planned_tick: dict[str, int] = {}
        # Zone locality state. When non-empty, focus selection prefers
        # NPCs whose ``current_zone`` is in the set. Empty = world-wide
        # mode, every existing bench and test sees the same behaviour
        # as before the zone layer existed. The game client owns this
        # state authority via POST /story/player_zone; the Director
        # reads but never writes.
        self._active_zones: set[str] = set()
        # Tick counter used to pace out-of-zone focus picks (one in
        # _OUT_OF_ZONE_RATE picks go to the out-of-zone pool).
        self._zone_escape_counter: int = 0
        # Lifecycle state (Phase 2a). ``_deceased_npcs`` is keyed by
        # npc_id and stores the full death record (death_tick,
        # death_cause, inheritor, affected arcs, quests cleaned). It's
        # persisted in state.json so game restarts honor deaths.
        # ``_pending_death_requests`` is a FIFO queue: REST calls to
        # /story/npc_death push here, and the next lifecycle tick
        # drains the queue and actually dispatches each death.
        self._deceased_npcs: dict[str, dict] = {}
        self._pending_death_requests: list[dict] = []
        # Phase 2b birth pipeline state.
        self._pending_birth_requests: list[dict] = []
        self._birth_history: list[dict] = []
        # Autonomous lifecycle (Phase 2c). When enabled, the Director
        # can propose NPC deaths (rare, arc-driven) and births
        # (population-driven, already built in Phase 2b) without game
        # client input. Off by default — game-authoritative is safer.
        self._autonomous_lifecycle: bool = False
        self._autonomous_deaths_this_session: int = 0
        self._autonomous_births_this_session: int = 0
        # Zone config loaded from zones.yaml (if present).
        self._zone_config: dict = {}  # zone_name -> {target_population, min_population, role_pool, lore_hook, ...}
        # Burst rotation state: the NPC currently held as the first-slot
        # focus across consecutive ticks, plus the remaining number of
        # ticks (AFTER the one that locked it in) the burst will keep
        # that NPC. Both persist in state.json so a bench resume keeps
        # the cache-reuse pattern intact across restarts.
        self._burst_focus_npc: Optional[str] = None
        self._burst_remaining: int = 0
        # Phase 4a — player activity context. Game client owns the
        # authority via POST /story/activity; the Director reads it in
        # tick() to short-circuit paused modes, force single-action
        # ticks, and drop 'quest' from rotation when the player can't
        # meaningfully accept one. "unknown" is the backward-compat
        # default — a caller that never sets an activity sees the same
        # behaviour as before the activity layer existed.
        self._player_activity: str = PlayerActivity.UNKNOWN.value
        self._activity_set_at_tick: int = 0
        # Phase 4b — per-NPC quest accumulation caps. Keyed by npc_id,
        # value is the tick number at which that NPC last had a quest
        # dispatched. _pick_action_kind gates future 'quest' picks on
        # this dict + the open-quest count. Persisted in state.json so
        # the cooldown window survives a Director restart. Unbounded in
        # entry count (capped only by cast size), same shape as
        # _npc_last_planned_tick.
        self._last_quest_dispatched_per_npc: dict[str, int] = {}
        # Runtime override hooks for the two Phase 4b tunables. None =
        # use the module-level default. Persisted so a runtime tweak
        # survives a restart.
        self._max_unoffered_quests_override: Optional[int] = None
        self._quest_cooldown_ticks_override: Optional[int] = None
        # Phase 4c — explicit pause state. Orthogonal to the Phase 4a
        # activity-based pause; game clients use this for menu
        # overlays, cutscenes, save/load, or any global hold. Stamped
        # with the tick at which it was set so the client can tell
        # how long a pause has held. Persisted in state.json.
        self._paused: bool = False
        self._paused_at_tick: int = 0
        # Phase 4c — tick budget. Rolling 60-second cap on LLM time.
        # -1 = unconstrained (default, keeps every existing bench
        # behavior identical). Positive = max LLM seconds allowed in
        # the trailing 60s window; when exceeded, the Director returns
        # a paused dict with reason="budget_exceeded" until the window
        # rolls forward and frees capacity.
        self._tick_budget_seconds: float = -1.0
        # Tick-time log: [(timestamp_wall_clock, llm_seconds), ...].
        # Pruned to the trailing window at every budget check. NOT
        # persisted — the window is short enough that a restart
        # effectively resets the cap, which is the right behavior
        # (a fresh process should start with full capacity).
        self._tick_time_log: list[tuple[float, float]] = []
        # Phase 3a — quest-lines. ``_quest_lines_config`` mirrors the
        # ``quest_lines:`` dict from the world's ``quest_lines.yaml``:
        # {line_id: {type, title, beats, protected_givers,
        # reward_track, ...}}. Empty dict = no main-lines, every
        # existing test and bench sees the same behaviour as before
        # 3a.
        self._quest_lines_config: dict = {}
        # Runtime state per line: tracks which beats have been
        # dispatched, which quests are completed, and the accumulated
        # reward-track items. Keyed by line_id. Persisted in
        # state.json so a game restart keeps the main-line progress.
        self._quest_line_state: dict[str, dict] = {}
        # Counter used by _pick_focus_npc's Phase 3a main-line
        # weighting (modulo _MAIN_LINE_WEIGHT_FACTOR+1). Persisted so
        # the 2:1 ratio is preserved across restarts.
        self._main_line_focus_counter: int = 0
        # Refused-quest decay timers: {(npc_id, quest_id): unlock_tick}.
        # Scanned every lifecycle tick — when tick_count >= unlock_tick
        # the quest flips status back to 'available' and a subtle
        # re-open ledger entry is emitted.
        self._refused_quest_timers: dict[tuple[str, str], int] = {}
        # Auto-refuse (Phase 3a). Two-layer:
        #   Dev layer (config.yaml: director.quest_auto_refuse.enabled)
        #   Player layer (set via /player/auto_refuse REST)
        # The dev flag defaults to False so the feature stays off until
        # explicitly enabled by a game integration.
        self._quest_auto_refuse_enabled: bool = False
        self._quest_auto_refuse_player_configurable: bool = True
        self._player_auto_refuse_intents: set[str] = set()
        # Phase 5a — identity split. The Director tracks trust per
        # (npc_id, identity) pair so two NPCs can know the player
        # under different identities and have independent trust
        # records. introduce_player merges identities for a given
        # NPC and normalizes trust to max across them — keeps the
        # "good deeds under one identity don't stay invisible
        # forever" invariant Jordan specified.
        self._npc_player_identity_trust: dict[str, dict[str, int]] = {}
        # Last player-reported visible feature. Populated by
        # /player/visible_feature — matched against the feature
        # registry below for auto-recognition (5b).
        self._player_visible_feature: Optional[str] = None
        # Phase 5b — feature → identity registry. Populated via
        # ``/player/register_feature``. When the player's current
        # visible feature has a registered identity AND an NPC's
        # first meeting fires (via record_player_action witnessing),
        # the identity is auto-added to the NPC's ``known_as`` list
        # without requiring an explicit /player/introduce call.
        self._visible_feature_to_identity: dict[str, str] = {}
        self._kind_rotation_index: int = 0      # round-robin over _ACTION_KIND_ROTATION
        self.recent_player_actions: list[dict] = []  # last 8 player observations
        self._lore_text: str = ""
        self._examples: list[dict] = []
        # Per-tick self-repetition retry counter. Reset at the start
        # of every tick() call so subsequent workers in a multi-action
        # tick share one budget. Keeps worst-case tick latency bounded.
        self._self_rep_retries_this_tick: int = 0
        # Per-NPC per-bio-item mention counts. The focus NPC bio block
        # rotates by mention count ascending, so items that have already
        # been quoted repeatedly fall to the bottom and fresh items rise
        # to the top. Keyed as {npc_id: {item_key: count}}.
        self._bio_mention_counts: dict[str, dict[str, int]] = {}
        # Cache of each NPC's ORIGINAL (YAML-sourced) bio items at
        # director init time. NPCKnowledge.personal_knowledge is mutable
        # — the dispatch layer appends Director-generated facts to it —
        # so if we read it live, we'd end up treating the model's own
        # outputs as bio items and bumping mention counts on them. The
        # snapshot fixes this: bio tracking always operates on the
        # original character data, while plot continuity lives in the
        # ledger and recent_decisions.
        self._original_bios: dict[str, list[tuple[str, str]]] = {}
        # Output narration style. "prose" (default, backward-compat)
        # produces cinematic novel-style narration with internal
        # monologue, quoted dialogue, and detailed action choreography.
        # "terse" produces short third-person factual statements under
        # 25 words that downstream NPC dialogue generation can cite
        # verbatim without bloating its own prompt. Terse is the right
        # default for game contexts where Director outputs feed NPC
        # conversation state; prose is preserved for storytelling/demo
        # contexts where the novelistic output is the product.
        self.narration_mode: str = "prose"
        self._resolve_paths()
        self.ledger = FactLedger(self._ledger_file)
        self.arc_planner = ArcPlanner(self._arcs_file)
        self._load_assets()
        self._load_zone_config()
        self._load_quest_lines()
        self._load_state()
        self._snapshot_original_bios()
        # Manifest: subtract any NPC ids found in _birth_history
        # (already persisted in state.json) from the current profile
        # set. The remainder is the YAML-authored "shipping cast."
        # Persisted births whose YAML files still exist on disk get
        # loaded by NPCKnowledgeManager at boot; this subtraction
        # keeps them out of the manifest so reset_to_initial_state()
        # can identify and remove them cleanly.
        born_ids = {
            r.get("npc_id") for r in self._birth_history
            if isinstance(r, dict) and isinstance(r.get("npc_id"), str)
        }
        self._initial_cast: frozenset[str] = frozenset(
            npc_id for npc_id in self.engine.pie.npc_knowledge.profiles
            if npc_id not in born_ids
        )

        # NarrativeJudge passive observer (off by default, env-gated).
        # Logs per-dispatch coherence scores to a per-world sidecar JSONL.
        # Behavior-neutral — only logging.
        self._narrative_judge: Optional[Any] = None
        self._narrative_judge_log_path: Optional[Path] = None
        if os.environ.get(_NARRATIVE_JUDGE_OBSERVE_ENV) == "1":
            try:
                from npc_engine.narrative_judge import NarrativeJudge
                self._narrative_judge = NarrativeJudge()
                self._narrative_judge_log_path = (
                    self._runtime_dir / "narrative_judge_observations.jsonl"
                )
                logger.info(
                    f"NarrativeJudge observer enabled. Logging to "
                    f"{self._narrative_judge_log_path}"
                )
            except Exception as e:
                logger.warning(f"NarrativeJudge observer init failed: {e}")
                self._narrative_judge = None
                self._narrative_judge_log_path = None

        # Predictive FactLedger lane (PHASE_PREDICTIVE_FACTLEDGER_PLAN
        # v1). Behavior-neutral by default: predicts next-tick activity
        # + per-NPC edge priors before each tick, logs drift at DEBUG,
        # persists to a gitignored .npz sidecar. The arc-proposal boost
        # is separately gated behind NPC_ENGINE_PREDICTIVE_BOOST=1.
        self._predictive: Optional[Any] = None
        self._predictive_boost_enabled: bool = (
            os.environ.get(_PREDICTIVE_BOOST_ENV) == "1"
        )
        self._last_activity_pred: Optional[dict] = None
        if os.environ.get(_PREDICTIVE_DISABLE_ENV) != "1":
            try:
                from npc_engine.predictive_factledger import PredictiveLayer
                self._predictive = PredictiveLayer(
                    storage_path=self._predictive_file,
                    labels=sorted(_PLAYER_ACTIVITY_VALUES),
                    history_path=self._activity_history_file,
                )
                # Warm pass only when the sidecar didn't already carry
                # a state — replaying history on every boot would be
                # wasted work the sidecar exists to avoid.
                if not self._predictive._loaded_from_sidecar:
                    warm_report = self._predictive.warm_from_history(self.ledger)
                    if warm_report.get("pairs") or warm_report.get("edge_updates"):
                        logger.info(
                            f"PredictiveLayer warm-from-history: {warm_report}"
                        )
            except Exception as e:
                logger.warning(f"PredictiveLayer init failed: {e}")
                self._predictive = None

    def _judge_observe(self, action: dict, dispatch_result: dict) -> None:
        """Passive observer: score the dispatched action against every
        active quest spec via NarrativeJudge and log to sidecar JSONL.

        Behavior-neutral — never raises, never affects dispatch.
        Activated only when NPC_ENGINE_NARRATIVE_JUDGE_OBSERVE=1.
        """
        if self._narrative_judge is None or self._narrative_judge_log_path is None:
            return
        if not self._narrative_judge.available:
            return

        # Extract the dispatched fact text from the action dict
        kind = action.get("action")
        fact_text = ""
        if kind == "quest":
            q = action.get("quest") or {}
            name = str(q.get("name") or "").strip()
            desc = str(q.get("description") or "").strip()
            fact_text = (name + " — " + desc).strip(" —")
        elif kind == "event":
            fact_text = str(action.get("event") or action.get("description") or "").strip()
        elif kind == "fact":
            fact_text = str(action.get("fact") or "").strip()
        if not fact_text:
            return

        # Collect every active/available quest spec across the cast
        try:
            active_specs: dict[str, str] = {}
            for npc_id, npc in self.engine.pie.npc_knowledge.profiles.items():
                for q in getattr(npc, "quests", []):
                    if getattr(q, "status", None) not in ("available", "active"):
                        continue
                    spec = str(getattr(q, "description", "") or "").strip()
                    if not spec:
                        spec = str(getattr(q, "name", "") or "").strip()
                    if not spec:
                        continue
                    key = f"{npc_id}:{getattr(q, 'id', '?')}"
                    active_specs[key] = spec
        except Exception as e:
            logger.warning(f"NarrativeJudge observer: spec collection failed: {e}")
            return

        if not active_specs:
            # No active quests = nothing to score against. Still record so
            # the analysis sees the empty case.
            record = {
                "tick": self.tick_count,
                "action_kind": kind,
                "dispatched_target": (action.get("npc_id")
                                       or action.get("target")
                                       or "*"),
                "dispatch_ok": bool(dispatch_result.get("ok")),
                "fact_text": fact_text[:500],
                "active_quest_count": 0,
                "scores": {},
            }
        else:
            try:
                ranked = self._narrative_judge.score_against_quests_ranked(
                    fact_text, active_specs
                )
            except Exception as e:
                logger.warning(f"NarrativeJudge observer: score failed: {e}")
                return
            if ranked is None:
                return
            # Decision-quality classification (post-hoc analysis on PB
            # gameplay session: margin >= 0.15 -> 2/15 specific,
            # 0.05 <= margin < 0.15 -> 8/15 plurality,
            # margin < 0.05 -> 5/15 ambiguous/multi-thread).
            margin = ranked["margin"]
            if margin >= 0.15:
                decision = "specific"
            elif margin >= 0.05:
                decision = "plurality"
            else:
                decision = "ambiguous"
            score_payload = {}
            for qid, res in ranked["per_quest"].items():
                if res is None:
                    continue
                score_payload[qid] = {
                    "label": res.get("label"),
                    "scores": res.get("scores"),
                }
            record = {
                "tick": self.tick_count,
                "action_kind": kind,
                "dispatched_target": (action.get("npc_id")
                                       or action.get("target")
                                       or "*"),
                "dispatch_ok": bool(dispatch_result.get("ok")),
                "fact_text": fact_text[:500],
                "active_quest_count": len(active_specs),
                "best_quest": ranked["best_quest"],
                "best_advance": ranked["best_advance"],
                "runner_up_quest": ranked["runner_up_quest"],
                "runner_up_advance": ranked["runner_up_advance"],
                "margin": ranked["margin"],
                "softmax_peak": ranked["softmax_peak"],
                "decision": decision,
                "scores": score_payload,
            }

        try:
            self._narrative_judge_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._narrative_judge_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"NarrativeJudge observer: log write failed: {e}")

    def _resolve_paths(self) -> None:
        """
        Pick asset + runtime paths for this Director instance.

        Default: the module-level ``LORE_FILE`` / ``EXAMPLES_FILE`` /
        state / ledger / arcs files in ``data/story_director/`` (the
        Ashenvale pack). Per-world override: if ``<world_dir>/story/``
        exists on disk, every file in that directory overrides its
        default and runtime state is world-scoped into the same
        directory. This keeps Ashenvale runs pointing at the shared
        defaults (so existing tests and benches don't budge) while
        Port Blackwater and any future world get an isolated asset
        pack plus isolated state, ledger, and arcs.
        """
        # Capture module-level paths at init time so tests that
        # monkey-patch ``sd_mod.STATE_FILE`` etc. still win.
        self._lore_file: Path = LORE_FILE
        self._examples_file: Path = EXAMPLES_FILE
        self._examples_terse_file: Path = EXAMPLES_TERSE_FILE
        self._state_file: Path = STATE_FILE
        self._ledger_file: Path = LEDGER_FILE
        self._arcs_file: Path = ARCS_FILE
        self._runtime_dir: Path = DATA_DIR

        world_dir_str = getattr(
            getattr(self.engine, "config", None), "world_dir", None,
        )
        if not world_dir_str:
            return
        story_dir = Path(world_dir_str) / "story"
        if not story_dir.exists():
            return

        custom_lore = story_dir / "lore.md"
        custom_examples = story_dir / "examples.yaml"
        custom_examples_terse = story_dir / "examples_terse.yaml"
        if custom_lore.exists():
            self._lore_file = custom_lore
        if custom_examples.exists():
            self._examples_file = custom_examples
        if custom_examples_terse.exists():
            self._examples_terse_file = custom_examples_terse
        self._state_file = story_dir / "state.json"
        self._ledger_file = story_dir / "fact_ledger.json"
        self._arcs_file = story_dir / "arcs.json"
        self._runtime_dir = story_dir

    @property
    def _predictive_file(self) -> Path:
        """Predictive-layer .npz sidecar. Derived from the state file's
        stem (``state.json`` -> ``state.predictive.npz``) rather than the
        plan's literal ``predictive.npz`` so the existing test-isolation
        pattern — monkey-patching module-level STATE_FILE to a
        ``_tmp_*`` path — isolates this sidecar for free."""
        return self._state_file.parent / (self._state_file.stem + ".predictive.npz")

    @property
    def _activity_history_file(self) -> Path:
        """Append-only activity-history JSONL (the ActivityPrior's
        supervision signal). Same stem-derivation rationale as
        ``_predictive_file``."""
        return self._state_file.parent / (
            self._state_file.stem + ".activity_history.jsonl")

    # ── Lifecycle ───────────────────────────────────────────────

    def _load_assets(self) -> None:
        if self._lore_file.exists():
            self._lore_text = self._lore_file.read_text(encoding="utf-8").strip()
        else:
            logger.warning(f"Story Director lore file not found at {self._lore_file}")
        self._reload_examples()

    def _reload_examples(self) -> None:
        """
        Load the example library that matches the current
        ``narration_mode``. Terse mode prefers
        ``examples_terse.yaml`` — if it's missing, we fall back to
        the prose library so an incomplete terse install still
        works (with the partial-compliance result documented for
        the prose-only toggle commit).

        Called from ``_load_assets`` at init and from
        ``set_narration_mode`` if the caller flips modes after init.
        """
        source = (
            self._examples_terse_file
            if self.narration_mode == "terse" and self._examples_terse_file.exists()
            else self._examples_file
        )
        if source.exists():
            try:
                data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
                self._examples = data.get("examples", []) or []
                logger.info(f"Story Director loaded {len(self._examples)} examples from {source.name}")
            except Exception as e:
                logger.error(f"Story Director failed to load examples: {e}")
                self._examples = []
        else:
            logger.warning(f"Story Director examples file not found at {source}")
            self._examples = []

    def set_narration_mode(self, mode: str) -> None:
        """
        Switch between ``prose`` and ``terse`` output styles at runtime.
        Reloads the matching example library so the picker serves the
        right style templates to the next tick. Use this instead of
        assigning to ``self.narration_mode`` directly — the attribute
        assignment works but won't reload examples.
        """
        if mode not in ("prose", "terse"):
            raise ValueError(f"narration_mode must be 'prose' or 'terse', got {mode!r}")
        if mode == self.narration_mode:
            return
        self.narration_mode = mode
        self._reload_examples()

    def _load_zone_config(self) -> None:
        """
        Load zone configuration from ``zones.yaml`` in the world
        directory (if present). Each zone entry declares
        ``target_population``, ``min_population``, ``role_pool``,
        ``lore_hook``, ``adjacent_to`` — these fields drive the
        Phase 2b birth pipeline's population management and prompt
        context. Worlds without a ``zones.yaml`` get an empty
        ``_zone_config`` dict and no lifecycle population checks
        ever fire — strictly backward-compatible.
        """
        world_dir_str = getattr(
            getattr(self.engine, "config", None), "world_dir", None,
        )
        if not world_dir_str:
            return
        zones_path = Path(world_dir_str) / "zones.yaml"
        if not zones_path.exists():
            return
        try:
            data = yaml.safe_load(zones_path.read_text(encoding="utf-8")) or {}
            raw_zones = data.get("zones", {})
            if isinstance(raw_zones, dict):
                self._zone_config = raw_zones
                logger.info(
                    f"Story Director loaded zone config: "
                    f"{len(self._zone_config)} zones from {zones_path}"
                )
        except Exception as e:
            logger.warning(f"Failed to load zones.yaml: {e}")

    def _load_quest_lines(self) -> None:
        """
        Phase 3a — load ``quest_lines.yaml`` from the world directory
        if present. The file declares one or more main/side lines:

            quest_lines:
              main_dark_lighthouse:
                type: main
                title: "The Dark Lighthouse"
                beats:
                  - {quest_id: lighthouse_mystery, giver: captain_reva}
                  - {quest_id: witness_account,    giver: thessa,
                     requires: [lighthouse_mystery]}
                protected_givers: [captain_reva, thessa]
                reward_track: ["Harbor Master's seal", ...]

        ``_quest_line_state`` picks up any existing state from
        ``state.json`` in ``_load_state`` — this loader only wires
        the static config. Worlds without the file run with empty
        main-line state and the old behaviour.
        """
        world_dir_str = getattr(
            getattr(self.engine, "config", None), "world_dir", None,
        )
        if not world_dir_str:
            return
        qlines_path = Path(world_dir_str) / "quest_lines.yaml"
        if not qlines_path.exists():
            return
        try:
            data = yaml.safe_load(qlines_path.read_text(encoding="utf-8")) or {}
            raw = data.get("quest_lines", {})
            if isinstance(raw, dict):
                self._quest_lines_config = raw
                logger.info(
                    f"Story Director loaded quest_lines: "
                    f"{len(self._quest_lines_config)} lines from {qlines_path}"
                )
                # Seed _quest_line_state skeletons for lines we've
                # never seen before, so code that reads state doesn't
                # need to check for missing keys.
                for line_id in self._quest_lines_config.keys():
                    self._quest_line_state.setdefault(line_id, {
                        "dispatched_beats": [],
                        "completed_quests": [],
                        "rewards_earned": [],
                        "line_status": "active",
                    })
            # Dev-layer auto-refuse config — optional.
            director_block = data.get("director") or {}
            auto_cfg = director_block.get("quest_auto_refuse") or {}
            if isinstance(auto_cfg, dict):
                self._quest_auto_refuse_enabled = bool(auto_cfg.get("enabled", False))
                self._quest_auto_refuse_player_configurable = bool(
                    auto_cfg.get("player_configurable", True)
                )
        except Exception as e:
            logger.warning(f"Failed to load quest_lines.yaml: {e}")

    def _load_state(self) -> None:
        if not self._state_file.exists():
            return
        try:
            state = json.loads(self._state_file.read_text(encoding="utf-8"))
            self.tick_count = state.get("tick_count", 0)
            self.last_tick_at = state.get("last_tick_at")
            self.recent_decisions = state.get("recent_decisions", [])[-5:]
            raw_planned = state.get("npc_last_planned_tick", {}) or {}
            if isinstance(raw_planned, dict):
                self._npc_last_planned_tick = {
                    str(k): int(v) for k, v in raw_planned.items()
                    if isinstance(v, (int, float))
                }
            raw_burst = state.get("burst_focus_npc")
            if isinstance(raw_burst, str) and raw_burst:
                self._burst_focus_npc = raw_burst
            raw_burst_rem = state.get("burst_remaining", 0)
            if isinstance(raw_burst_rem, (int, float)):
                self._burst_remaining = max(0, int(raw_burst_rem))
            raw_zones = state.get("active_zones", [])
            if isinstance(raw_zones, list):
                self._active_zones = {
                    str(z) for z in raw_zones if isinstance(z, str)
                }
            raw_escape = state.get("zone_escape_counter", 0)
            if isinstance(raw_escape, (int, float)):
                self._zone_escape_counter = max(0, int(raw_escape))
            raw_deceased = state.get("deceased_npcs", {})
            if isinstance(raw_deceased, dict):
                self._deceased_npcs = {
                    str(k): v for k, v in raw_deceased.items()
                    if isinstance(v, dict)
                }
            raw_pending_deaths = state.get("pending_death_requests", [])
            if isinstance(raw_pending_deaths, list):
                self._pending_death_requests = [
                    r for r in raw_pending_deaths if isinstance(r, dict)
                ]
            raw_pending_births = state.get("pending_birth_requests", [])
            if isinstance(raw_pending_births, list):
                self._pending_birth_requests = [
                    r for r in raw_pending_births if isinstance(r, dict)
                ]
            raw_birth_history = state.get("birth_history", [])
            if isinstance(raw_birth_history, list):
                self._birth_history = [
                    r for r in raw_birth_history if isinstance(r, dict)
                ]
            self._kind_rotation_index = state.get("kind_rotation_index", 0)
            raw_activity = state.get("player_activity")
            if isinstance(raw_activity, str) and raw_activity in _PLAYER_ACTIVITY_VALUES:
                self._player_activity = raw_activity
            raw_activity_tick = state.get("activity_set_at_tick", 0)
            if isinstance(raw_activity_tick, (int, float)):
                self._activity_set_at_tick = max(0, int(raw_activity_tick))
            raw_last_quest = state.get("last_quest_dispatched_per_npc", {}) or {}
            if isinstance(raw_last_quest, dict):
                self._last_quest_dispatched_per_npc = {
                    str(k): int(v) for k, v in raw_last_quest.items()
                    if isinstance(v, (int, float))
                }
            raw_max_unoffered = state.get("max_unoffered_quests_override")
            if isinstance(raw_max_unoffered, (int, float)):
                self._max_unoffered_quests_override = max(0, int(raw_max_unoffered))
            raw_cooldown = state.get("quest_cooldown_ticks_override")
            if isinstance(raw_cooldown, (int, float)):
                self._quest_cooldown_ticks_override = max(0, int(raw_cooldown))
            raw_paused = state.get("paused", False)
            if isinstance(raw_paused, bool):
                self._paused = raw_paused
            raw_paused_tick = state.get("paused_at_tick", 0)
            if isinstance(raw_paused_tick, (int, float)):
                self._paused_at_tick = max(0, int(raw_paused_tick))
            raw_budget = state.get("tick_budget_seconds")
            if isinstance(raw_budget, (int, float)):
                self._tick_budget_seconds = float(raw_budget)
            raw_ml_counter = state.get("main_line_focus_counter", 0)
            if isinstance(raw_ml_counter, (int, float)):
                self._main_line_focus_counter = max(0, int(raw_ml_counter))
            raw_line_state = state.get("quest_line_state", {}) or {}
            if isinstance(raw_line_state, dict):
                for line_id, st in raw_line_state.items():
                    if not isinstance(st, dict):
                        continue
                    self._quest_line_state[str(line_id)] = {
                        "dispatched_beats": list(st.get("dispatched_beats", [])),
                        "completed_quests": list(st.get("completed_quests", [])),
                        "rewards_earned": list(st.get("rewards_earned", [])),
                        "line_status": str(st.get("line_status", "active")),
                    }
            raw_refuse_timers = state.get("refused_quest_timers", []) or []
            if isinstance(raw_refuse_timers, list):
                for entry in raw_refuse_timers:
                    if (isinstance(entry, (list, tuple))
                            and len(entry) == 3
                            and isinstance(entry[0], str)
                            and isinstance(entry[1], str)
                            and isinstance(entry[2], (int, float))):
                        self._refused_quest_timers[(entry[0], entry[1])] = int(entry[2])
            raw_player_refuse = state.get("player_auto_refuse_intents", []) or []
            if isinstance(raw_player_refuse, list):
                self._player_auto_refuse_intents = {
                    str(v) for v in raw_player_refuse if isinstance(v, str)
                }
            raw_identity_trust = state.get("npc_player_identity_trust", {}) or {}
            if isinstance(raw_identity_trust, dict):
                for npc_id, idmap in raw_identity_trust.items():
                    if not isinstance(idmap, dict):
                        continue
                    self._npc_player_identity_trust[str(npc_id)] = {
                        str(k): int(v) for k, v in idmap.items()
                        if isinstance(v, (int, float))
                    }
            raw_feature = state.get("player_visible_feature")
            if isinstance(raw_feature, str) and raw_feature:
                self._player_visible_feature = raw_feature
            raw_feature_registry = state.get("visible_feature_to_identity", {}) or {}
            if isinstance(raw_feature_registry, dict):
                self._visible_feature_to_identity = {
                    str(k): str(v) for k, v in raw_feature_registry.items()
                    if isinstance(v, str)
                }
            # Phase 5a — restore per-NPC player_knowledge onto the
            # live profiles. The profiles are already loaded by this
            # point (NPCKnowledge __init__ ran during engine boot),
            # so we just splice the saved dict back onto each NPC.
            # NPCs absent from the saved dict keep their fresh
            # defaults — the NPC may have been born via Phase 2b
            # mid-session and never had player interactions yet.
            raw_pk = state.get("per_npc_player_knowledge", {}) or {}
            if isinstance(raw_pk, dict):
                for npc_id, pk in raw_pk.items():
                    if not isinstance(pk, dict):
                        continue
                    npc = self.engine.pie.npc_knowledge.profiles.get(npc_id)
                    if npc is None:
                        continue
                    npc.player_knowledge = {
                        "met": bool(pk.get("met", False)),
                        "recognized": bool(pk.get("recognized", False)),
                        "known_as": list(pk.get("known_as", [])),
                        "witnessed_deeds": list(pk.get("witnessed_deeds", [])),
                        "heard_deeds": list(pk.get("heard_deeds", [])),
                        "first_met_tick": pk.get("first_met_tick"),
                        "last_interaction_tick": pk.get("last_interaction_tick"),
                    }
            self.recent_player_actions = state.get("recent_player_actions", [])[-8:]
            raw_counts = state.get("bio_mention_counts", {}) or {}
            if isinstance(raw_counts, dict):
                self._bio_mention_counts = {
                    str(k): {str(ik): int(iv) for ik, iv in (v or {}).items()}
                    for k, v in raw_counts.items()
                    if isinstance(v, dict)
                }
        except Exception as e:
            logger.warning(f"Story Director failed to load state: {e}")

    def _save_state(self) -> None:
        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "tick_count": self.tick_count,
            "last_tick_at": self.last_tick_at,
            "recent_decisions": self.recent_decisions[-5:],
            "npc_last_planned_tick": self._npc_last_planned_tick,
            "burst_focus_npc": self._burst_focus_npc,
            "burst_remaining": self._burst_remaining,
            "active_zones": sorted(self._active_zones),
            "zone_escape_counter": self._zone_escape_counter,
            "deceased_npcs": self._deceased_npcs,
            "pending_death_requests": self._pending_death_requests,
            "pending_birth_requests": self._pending_birth_requests,
            "birth_history": self._birth_history,
            "kind_rotation_index": self._kind_rotation_index,
            "player_activity": self._player_activity,
            "activity_set_at_tick": self._activity_set_at_tick,
            "last_quest_dispatched_per_npc": self._last_quest_dispatched_per_npc,
            "max_unoffered_quests_override": self._max_unoffered_quests_override,
            "quest_cooldown_ticks_override": self._quest_cooldown_ticks_override,
            "paused": self._paused,
            "paused_at_tick": self._paused_at_tick,
            "tick_budget_seconds": self._tick_budget_seconds,
            "quest_line_state": self._quest_line_state,
            "main_line_focus_counter": self._main_line_focus_counter,
            # dict with tuple keys doesn't JSON-serialize; flatten to
            # a list of [npc_id, quest_id, unlock_tick] triples.
            "refused_quest_timers": [
                [npc_id, quest_id, unlock_tick]
                for (npc_id, quest_id), unlock_tick
                in self._refused_quest_timers.items()
            ],
            "player_auto_refuse_intents": sorted(self._player_auto_refuse_intents),
            "npc_player_identity_trust": self._npc_player_identity_trust,
            "player_visible_feature": self._player_visible_feature,
            "visible_feature_to_identity": self._visible_feature_to_identity,
            # Phase 5a — persist per-NPC player_knowledge so a
            # restart preserves recognition + deeds. Runtime-only
            # default remains the NPCKnowledge in-memory dict; state
            # save captures it alongside identity trust.
            "per_npc_player_knowledge": {
                npc_id: dict(pk) for npc_id, npc in
                self.engine.pie.npc_knowledge.profiles.items()
                for pk in [getattr(npc, "player_knowledge", None)]
                if isinstance(pk, dict)
            },
            "recent_player_actions": self.recent_player_actions[-8:],
            "bio_mention_counts": self._bio_mention_counts,
        }
        self._state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        # Predictive sidecar rides the existing autosave hook (plan:
        # "saved on graceful shutdown (and on tick if the existing
        # autosave hook fires)" — _save_state IS that hook). A few KB
        # of npz; negligible next to the state.json write.
        if self._predictive is not None:
            try:
                self._predictive.save()
            except Exception as e:
                logger.warning(f"Predictive sidecar save failed: {e}")

    # ── Public API ──────────────────────────────────────────────

    def tick(self, max_tokens: Optional[int] = None, temperature: float = 0.7,
              actions_per_tick: int = 1) -> dict:
        """
        Advance the story by one decision (or by N parallel decisions
        when ``actions_per_tick > 1``).

        Single-action mode (``actions_per_tick=1``, default) returns the
        legacy shape ``{tick, action, dispatch, raw_response}`` for
        backward compatibility with existing clients.

        Multi-action mode (``actions_per_tick >= 2``) runs Python's
        architect/worker pattern: the architect picks N distinct
        ``(focus_npc, action_kind)`` slots up front, then each worker
        generates the content for its slot independently. All sub-
        actions share the same pre-tick world snapshot — they are true
        peers, not a sequential pipeline. Returns
        ``{tick, sub_actions: [{action, dispatch, raw_response}, ...]}``.

        Always returns — never raises — so a game loop can call this on
        a timer without guarding every field.
        """
        if actions_per_tick < 1:
            actions_per_tick = 1

        # Phase 4c — explicit pause takes precedence over every other
        # tick path. Game client paused us (menu overlay, cutscene,
        # save/load) and nothing else is safe to do. Don't bump
        # tick_count: explicit pauses are expected to be held for
        # minutes-to-hours and advancing would move every cooldown
        # forward against real time.
        if self._paused:
            return {
                "tick": self.tick_count,
                "action": {"action": "noop", "reason": "explicit_pause"},
                "dispatch": {"ok": True, "kind": "noop"},
                "raw_response": "",
                "paused": True,
                "paused_reason": "explicit_pause",
                "next_tick_recommended_in_seconds": self._compute_next_tick_hint(),
                **({"sub_actions": []} if actions_per_tick > 1 else {}),
            }

        # Phase 4c — tick budget. If LLM time in the rolling window
        # is at or above the configured cap, return a paused dict so
        # the client backs off and the window can free capacity.
        # Also does NOT bump tick_count — otherwise a hot game loop
        # that polls through budget saturation would rotate focus
        # without producing content.
        if self._budget_exceeded():
            return {
                "tick": self.tick_count,
                "action": {"action": "noop", "reason": "budget_exceeded"},
                "dispatch": {"ok": True, "kind": "noop"},
                "raw_response": "",
                "paused": True,
                "paused_reason": "budget_exceeded",
                "next_tick_recommended_in_seconds": self._compute_next_tick_hint(),
                **({"sub_actions": []} if actions_per_tick > 1 else {}),
            }

        # Phase 4a — paused activities short-circuit the tick. During
        # combat, menus, and idle the Director has nothing useful to
        # say; burning a generation for a beat the player can't react
        # to is pure waste. Return a paused noop that still increments
        # tick_count so calling code advances uniformly whether paused
        # or not, and hand the client an activity-adaptive next-tick
        # hint (in_combat=10s, in_menu=30s, idle=120s). State is NOT
        # saved here — the paused path is hit many times in a row and
        # each write would add I/O churn with no new information.
        if self._player_activity in _PAUSED_ACTIVITIES:
            self.tick_count += 1
            self.last_tick_at = datetime.now(timezone.utc).isoformat()
            paused_reason = f"paused: {self._player_activity}"
            result = {
                "tick": self.tick_count,
                "action": {"action": "noop", "reason": paused_reason},
                "dispatch": {"ok": True, "kind": "noop"},
                "raw_response": "",
                "paused": True,
                "paused_reason": f"activity:{self._player_activity}",
                "next_tick_recommended_in_seconds": self._compute_next_tick_hint(),
            }
            if actions_per_tick > 1:
                result["sub_actions"] = []
            return result

        # Phase 4a — dialogue and wandering force single-action ticks.
        # Dialogue fires mid-conversation (one Director beat per turn
        # is all the player can absorb); wandering is low-cadence
        # filler and multi-action would pile content no one will
        # read. Applied before the small-cast cap so the logs below
        # describe the final value.
        if self._player_activity in _SINGLE_ACTION_ACTIVITIES and actions_per_tick > 1:
            logger.info(
                f"Story Director: activity={self._player_activity}, "
                f"capping actions_per_tick {actions_per_tick} -> 1"
            )
            actions_per_tick = 1

        # Small-cast cap: on worlds with at most _SMALL_CAST_THRESHOLD
        # NPCs, force actions_per_tick to 1 so rotation walks the cast
        # over multiple ticks instead of touching every NPC every tick.
        # Prevents the PB-style accumulation that pushed terse-mode
        # token delta past the 150-token shipping budget (2026-04-14
        # fact-consumption bench). Above the threshold, the caller's
        # requested value is honored.
        cast_size = len(self.engine.pie.npc_knowledge.profiles)
        if 0 < cast_size <= _SMALL_CAST_THRESHOLD and actions_per_tick > 1:
            logger.info(
                f"Story Director: small cast ({cast_size} NPCs <= "
                f"{_SMALL_CAST_THRESHOLD}), capping actions_per_tick "
                f"{actions_per_tick} -> 1"
            )
            actions_per_tick = 1

        # Phase 0: lifecycle maintenance. Drain any pending deaths
        # (and later, births) before the architect plans the beat.
        # Deaths mutate the profile set, so running this before the
        # plan ensures the architect picks from the post-death cast.
        lifecycle_actions = self._lifecycle_tick()

        # Default max_tokens depends on narration mode. Prose
        # outputs run 40-50 words (~60-80 tokens) and occasionally
        # spike to 100+ which justifies the 400-token ceiling.
        # Terse outputs average 23 words (~30 tokens) with a
        # practical max around 50; a 120-token cap lets the model
        # finish its sentence cleanly while cutting wasted
        # generation budget by 70% vs the prose default.
        if max_tokens is None:
            max_tokens = 120 if self.narration_mode == "terse" else 400

        # Reset the per-tick self-rep retry budget before any worker
        # runs so multi-action workers share a single retry slot.
        self._self_rep_retries_this_tick = 0

        # Predictive lane (v1) — runs before the architect plans so a
        # future iteration can bias focus/kind selection. Today it only
        # (a) surfaces the activity prior on /story/state, (b) logs
        # predicted-drift at DEBUG, and (c) produces per-NPC edge
        # priors that MAY boost arc proposal when the boost env gate
        # is on. Cold predictions reduce every path to pre-predictive
        # behavior. Never raises — tick() must always return.
        edge_priors: dict[str, float] = {}
        if self._predictive is not None:
            try:
                activity_pred, edge_priors = self._predictive.predict_next(
                    self.ledger, self.tick_count, self._player_activity,
                )
                self._last_activity_pred = activity_pred.to_dict()
            except Exception as e:
                logger.warning(f"Predictive lane failed: {e}")
                edge_priors = {}

        # Plan first, then build the snapshot — the bounded snapshot
        # path needs to know which NPCs the architect picked so it can
        # surface them in the per-tick "active scene". On unbounded
        # casts (≤ _SNAPSHOT_BOUND_THRESHOLD NPCs) the snapshot ignores
        # the planned ids and walks every profile, identical to the
        # pre-bounded behaviour.
        plan = self._architect_plan(actions_per_tick)
        planned_focus_ids = [npc for (npc, _kind) in plan]
        # Record the architect's plan in the unbounded planned-focus
        # trail BEFORE workers run. This is what _pick_focus_npc reads
        # on the NEXT call to compute least-recently-touched rotation.
        # Kept independent of recent_decisions so future code paths
        # that rewrite action.npc_id can't pollute rotation state.
        # The dict is unbounded in entry count
        # (capped only by cast size), so on a 500-NPC world the
        # rotation eventually visits every NPC instead of cycling the
        # same ~15.
        next_tick = self.tick_count + 1
        for npc_id in planned_focus_ids:
            self._npc_last_planned_tick[npc_id] = next_tick
        snapshot = self._world_snapshot(planned_focus_ids=planned_focus_ids)

        # Try to propose a narrative arc BEFORE workers run so their
        # prompts can reference it. No-op if there's already an active
        # arc or the ledger is too thin — proposal is cooldown-gated.
        available_npcs = list(self.engine.pie.npc_knowledge.profiles.keys())
        self.arc_planner.maybe_propose(
            self.ledger, available_npcs, current_tick=self.tick_count + 1,
            edge_prior_boost=(
                edge_priors if self._predictive_boost_enabled else None
            ),
        )

        if not plan:
            # No NPCs to focus on — return a minimal noop response
            self.tick_count += 1
            self.last_tick_at = datetime.now(timezone.utc).isoformat()
            self._save_state()
            empty = {"action": "noop", "reason": "no_focus_npc_available"}
            if actions_per_tick == 1:
                return {
                    "tick": self.tick_count,
                    "action": empty,
                    "dispatch": {"ok": True, "kind": "noop"},
                    "raw_response": "",
                    "next_tick_recommended_in_seconds": self._compute_next_tick_hint(),
                }
            return {
                "tick": self.tick_count,
                "sub_actions": [],
                "next_tick_recommended_in_seconds": self._compute_next_tick_hint(),
            }

        # Phase 4c — time the worker loop so the rolling budget
        # window sees LLM cost. The Director's Python overhead is
        # negligible next to generation, so measuring the full loop
        # (vs just the LLM calls) is close enough and keeps the
        # instrumentation out of _run_single_action.
        tick_start_time = time.monotonic()
        sub_results: list[dict] = []
        for focus_npc, action_kind in plan:
            sub_results.append(self._run_single_action(
                snapshot=snapshot,
                focus_npc=focus_npc,
                action_kind=action_kind,
                max_tokens=max_tokens,
                temperature=temperature,
            ))
        self._record_tick_duration(time.monotonic() - tick_start_time)

        self.tick_count += 1
        self.last_tick_at = datetime.now(timezone.utc).isoformat()

        # Predictive lane — feed this tick's ledger delta into the
        # edge filters. Entries were added with tick = the NEW
        # tick_count, so select by tick number rather than slicing
        # from the pre-tick length: FactLedger.add trims the list to
        # its 200-entry cap, which would make a length-based slice
        # index stale (empty/short delta) once a long session hits
        # the cap. Absence updates (known NPCs that did NOT receive
        # a beat) happen inside record_observation.
        if self._predictive is not None:
            try:
                delta = [
                    e for e in self.ledger.entries
                    if e.get("tick") == self.tick_count
                ]
                self._predictive.record_observation(
                    self.tick_count, self._player_activity, delta,
                )
            except Exception as e:
                logger.warning(f"Predictive record_observation failed: {e}")

        # Record this tick. For multi-action ticks we store ALL sub-action
        # actions in the recent_decisions trail (under their parent tick)
        # so future cooldown calculations see every NPC touched.
        decision_record = {
            "tick": self.tick_count,
            "at": self.last_tick_at,
            "snapshot_preview": snapshot[:200],
        }
        if actions_per_tick == 1:
            decision_record["action"] = sub_results[0]["action"]
            decision_record["dispatch"] = sub_results[0]["dispatch"]
        else:
            # Use the FIRST sub-action's metadata as the canonical "action"
            # so legacy cooldown code that reads decision["action"] still
            # works. Store the full list under sub_actions.
            decision_record["action"] = sub_results[0]["action"]
            decision_record["dispatch"] = sub_results[0]["dispatch"]
            decision_record["sub_actions"] = [
                {"action": r["action"], "dispatch": r["dispatch"]}
                for r in sub_results
            ]
        self.recent_decisions.append(decision_record)
        self.recent_decisions = self.recent_decisions[-5:]

        # After the tick is recorded, check if the active arc's beat has
        # met its touch threshold and should advance (or resolve).
        # Touches are counted on the arc itself via record_cast_touch
        # (called from _run_single_action) — no need to walk the tail
        # of recent_decisions.
        self.arc_planner.advance_if_beat_met(current_tick=self.tick_count)

        self._save_state()

        if actions_per_tick == 1:
            r = sub_results[0]
            return {
                "tick": self.tick_count,
                "action": r["action"],
                "dispatch": r["dispatch"],
                "raw_response": r["raw_response"],
                "next_tick_recommended_in_seconds": self._compute_next_tick_hint(),
            }
        return {
            "tick": self.tick_count,
            "sub_actions": sub_results,
            "next_tick_recommended_in_seconds": self._compute_next_tick_hint(),
        }

    def _run_single_action(self, snapshot: str, focus_npc: Optional[str],
                            action_kind: str, max_tokens: int,
                            temperature: float) -> dict:
        """
        Run one (focus_npc, action_kind) slot through the LLM + enforce
        + (pre-dispatch contradiction check) + dispatch + ledger
        pipeline. Returns a dict the tick caller merges into the response.

        If the pre-dispatch ledger check reports a real contradiction
        (similarity match + NLI contradiction at >=0.85 confidence),
        the worker retries ONCE with a corrective preamble that names
        the conflicting prior fact. Capped at one retry to bound
        latency and prevent oscillation.
        """
        prompt = self._build_prompt(snapshot, focus_npc, action_kind)
        raw, action = self._llm_call_with_repair(prompt, max_tokens, temperature)
        action = self._finalize_action(action, focus_npc, action_kind)

        # Pre-dispatch contradiction check — fires only when the ledger
        # has prior entries AND NLI flags the pair at >=0.85 confidence.
        # Cheap to check (one embed + one NLI inference) and bypassed
        # entirely on the first few ticks before the ledger has anything.
        retried = False
        retry_reason: Optional[str] = None
        dispatch_precheck_note: Optional[dict] = None
        precheck = self._precheck_contradiction(action)
        if precheck is not None:
            retried = True
            retry_reason = "contradiction"
            retry_prompt = prompt + (
                "\n\nNOTE: Your previous attempt contradicts an earlier "
                f"established fact (T{precheck['matches_tick']} "
                f"{precheck['matches_kind']}/{precheck['matches_npc']}): "
                f"\"{precheck['matches_text'][:160]}\". "
                "Pick a DIFFERENT angle that does not conflict with that fact. "
                "Do not negate it; build a story beat that's consistent with it."
            )
            raw2, action2 = self._llm_call_with_repair(retry_prompt, max_tokens, temperature)
            action2 = self._finalize_action(action2, focus_npc, action_kind)
            action = action2
            raw = raw2
        else:
            # Fall through to the self-repetition check only if the
            # contradiction path didn't already retry. Both checks share
            # the retry budget — one is enough to keep latency bounded.
            selfrep = self._precheck_self_repetition(action)
            if selfrep is not None and self._self_rep_retries_this_tick >= _MAX_SELF_REP_RETRIES_PER_TICK:
                # Budget exhausted — skip the retry to cap worst-case
                # per-tick latency. Record that we would have retried
                # so callers can audit how often the budget kicks in.
                dispatch_precheck_note = {
                    "skipped_self_rep_retry": True,
                    "reason": "budget_exhausted",
                    "similarity": selfrep.get("similarity"),
                    "matches_tick": selfrep.get("matches_tick"),
                }
                logger.info(
                    f"Self-rep retry budget exhausted for this tick "
                    f"(sim={selfrep.get('similarity')}); skipping"
                )
                selfrep = None

            if selfrep is not None:
                retried = True
                retry_reason = "self_repetition"
                self._self_rep_retries_this_tick += 1
                # Prescriptive retry nudge — "pick a different angle"
                # was too open-ended on 3B and frequently produced
                # noops. Listing concrete alternatives gives the model
                # something to latch onto instead of asking for
                # "invention".
                retry_prompt = prompt + (
                    "\n\nNOTE: Your previous attempt repeats a recent "
                    f"beat (T{selfrep['matches_tick']} "
                    f"{selfrep['matches_kind']}/{selfrep['matches_npc']}): "
                    f"\"{selfrep['matches_text'][:160]}\". "
                    "Write a DIFFERENT beat for this NPC. Pick one of: "
                    "(a) a conversation with another villager named in "
                    "the world state, (b) a physical observation about a "
                    "place or object in the setting, (c) a new piece of "
                    "information learned from a rumor or event, "
                    "(d) a reaction to something another NPC did. "
                    "Do not repeat the prior beat or paraphrase it."
                )
                original_action = action
                original_raw = raw
                raw2, action2 = self._llm_call_with_repair(retry_prompt, max_tokens, temperature)
                action2 = self._finalize_action(action2, focus_npc, action_kind)
                # Guard: on 3B the "pick a different angle" retry
                # sometimes degenerates to a noop, which drops the
                # content entirely. For self-rep retries (unlike
                # contradiction), the original is *content-valid* —
                # it was just slightly repetitive. Falling back to it
                # is strictly better than losing the tick to silence.
                if action2.get("action") == "noop" and original_action.get("action") != "noop":
                    logger.info(
                        "Self-repetition retry returned noop; falling "
                        "back to the original action to preserve content"
                    )
                    action = original_action
                    raw = original_raw
                else:
                    action = action2
                    raw = raw2

        dispatch_result = self._dispatch(action)

        # Ledger every successful, non-noop injection so contradictions
        # across sub-actions (within the same tick or across ticks) are
        # caught uniformly.
        if dispatch_result.get("ok") and dispatch_result.get("kind") not in (None, "noop"):
            # Bump the arc touch counter for every cast-targeted
            # dispatch. The planner uses this counter (not a
            # recent_decisions walk) to decide when to advance beats.
            touched_npc = (
                action.get("npc_id") or action.get("target") or focus_npc
            )
            if isinstance(touched_npc, str) and touched_npc not in ("all", "*"):
                self.arc_planner.record_cast_touch(touched_npc)

            ledger_text = self._ledger_text_for(action)
            if ledger_text:
                ledger_npc = (
                    action.get("npc_id")
                    or action.get("target")
                    or focus_npc
                    or "?"
                )
                warning = self.ledger.add(
                    text=ledger_text,
                    npc_id=str(ledger_npc),
                    kind=str(action.get("action", "?")),
                    tick=self.tick_count + 1,
                )
                if warning is not None:
                    dispatch_result["similarity_warning"] = warning
                # Bump bio-mention counts for whatever bio items the
                # model just quoted or paraphrased. Drives intra-bio
                # rotation on the next tick that focuses on this NPC.
                self._record_bio_mentions(focus_npc, ledger_text)

        if retried:
            if retry_reason == "contradiction":
                dispatch_result["retried_after_contradiction"] = True
            elif retry_reason == "self_repetition":
                dispatch_result["retried_after_self_repetition"] = True
        elif dispatch_precheck_note is not None:
            # Budget was exhausted — surface the skipped retry so
            # benches can count how often the budget kicks in.
            dispatch_result["skipped_self_rep_retry"] = dispatch_precheck_note

        return {
            "focus_npc": focus_npc,
            "action_kind": action_kind,
            "action": action,
            "dispatch": dispatch_result,
            "raw_response": raw,
            "retried": retried,
            "retry_reason": retry_reason,
        }

    def _finalize_action(self, action: dict, focus_npc: Optional[str],
                         action_kind: str) -> dict:
        """Apply both enforcement passes in one call. DRY for the
        worker + retry paths."""
        if focus_npc:
            action = self._enforce_focus_npc(action, focus_npc)
        action = self._enforce_action_kind(action, action_kind, focus_npc)
        return action

    def _precheck_contradiction(self, action: dict) -> Optional[dict]:
        """
        Embed the proposed action's content and check it against the
        ledger BEFORE dispatch. Returns the warning dict if NLI flags
        the candidate as a contradiction with an existing fact;
        ``None`` otherwise. Skips silently when the ledger has nothing
        to compare against or the embedder/NLI aren't loaded.
        """
        text = self._ledger_text_for(action)
        if not text:
            return None
        warning = self.ledger.check(text)
        if warning is not None and warning.get("contradiction"):
            return warning
        return None

    def _precheck_self_repetition(self, action: dict) -> Optional[dict]:
        """
        Fire when the candidate is *too similar to a recent Director
        output on the same NPC*. This catches the fixation mode where
        the model invents a scene and then paraphrases its own
        invention across later ticks (observed in the 3B biorot_v2
        bench: Bess's "dropping a tray of hot soup" scene reappeared
        at T13 and T15).

        The ledger check is restricted to same-NPC entries via
        ``restrict_to_npc`` so cross-NPC gossip propagation (where the
        top similarity match might be a different NPC talking about
        the same subject) doesn't mask a real self-repetition match
        sitting lower in the rankings.

        Two filters still apply:

        1. ``similarity >= _SELF_REPETITION_SIMILARITY`` (0.70) —
           with the NPC restriction, any same-NPC match at 0.70+ is
           almost certainly paraphrased self-repetition.
        2. ``tick_count + 1 - matches_tick <= _SELF_REPETITION_LOOKBACK_TICKS``
           — stale matches from 10+ ticks ago shouldn't block new
           content; plot threads can legitimately echo earlier beats
           after enough time passes.

        Returns the warning dict if both hold, None otherwise.
        """
        text = self._ledger_text_for(action)
        if not text:
            return None

        candidate_npc = (
            action.get("npc_id") or action.get("target") or ""
        )
        if not candidate_npc or candidate_npc in ("all", "*"):
            return None

        warning = self.ledger.check(text, restrict_to_npc=candidate_npc)
        if warning is None:
            return None
        if warning.get("similarity", 0) < _SELF_REPETITION_SIMILARITY:
            return None

        matches_tick = int(warning.get("matches_tick", 0) or 0)
        if self.tick_count + 1 - matches_tick > _SELF_REPETITION_LOOKBACK_TICKS:
            return None
        return warning

    def _architect_plan(self, n_actions: int) -> list[tuple[Optional[str], str]]:
        """
        Plan N distinct ``(focus_npc, action_kind)`` slots for a multi-
        action tick. Each slot is picked using the same focus + kind
        rotation as single-action mode, but the in-flight planning loop
        adds each chosen NPC to a temporary exclusion set so two
        workers can't compete for the same target.

        The first slot is served by burst rotation (see
        ``_consume_burst_focus``): the same NPC holds slot 0 across
        ``_BURST_ROTATION_DEPTH`` consecutive ticks so the world
        snapshot shared by every sub-action stays prefix-stable and
        llama-cpp reuses its KV cache instead of recomputing from
        scratch. Subsequent slots fall through to normal
        least-recently-touched rotation so the other workers in the
        tick still cover fresh ground.

        Returns at most ``n_actions`` slots, fewer if the world doesn't
        have enough NPCs.
        """
        plan: list[tuple[Optional[str], str]] = []
        excluded: set[str] = set()
        burst_focus = self._consume_burst_focus()
        for i in range(max(1, n_actions)):
            if i == 0 and burst_focus is not None:
                focus: Optional[str] = burst_focus
            else:
                focus = self._pick_focus_npc(extra_exclude=excluded)
            if focus is None:
                break
            excluded.add(focus)
            # Phase 3a — main-line givers bypass the per-NPC quest
            # pacing gate (unoffered cap + cooldown). Authored
            # main-line progression beats priority over organic
            # pacing; a scheduled main-line beat on a giver who
            # already has two side quests should still land.
            bypass = focus is not None and self._is_main_line_npc(focus)
            kind = self._pick_action_kind(focus, bypass_per_npc_cap=bypass)
            plan.append((focus, kind))
        return plan

    def _consume_burst_focus(self) -> Optional[str]:
        """
        Return the sticky NPC that should hold slot 0 of this tick's
        plan, advancing the burst counter by one. Three cases:

        1. **Burst still held** — ``_burst_remaining > 0`` and the
           currently bursted NPC still exists in the live cast.
           Decrement and return it.
        2. **Burst exhausted or invalid** — pick a fresh NPC via
           normal rotation (``_pick_focus_npc`` with no excludes),
           lock it in as the new burst focus, and set
           ``_burst_remaining = _BURST_ROTATION_DEPTH - 1`` so the
           freshly-picked NPC holds slot 0 for K total ticks (this
           one plus K-1 follow-ups).
        3. **No NPCs available** — return ``None``. Caller falls
           through to the regular rotation path which will also
           return None and cause the architect plan to be empty.

        The ``_burst_focus_npc`` field is persisted in ``state.json``
        so a bench run that loads prior state continues the same
        burst window instead of restarting it on every process boot.
        """
        profiles = self.engine.pie.npc_knowledge.profiles
        if not profiles:
            return None

        if (self._burst_remaining > 0
                and self._burst_focus_npc
                and self._burst_focus_npc in profiles):
            self._burst_remaining -= 1
            return self._burst_focus_npc

        # Pass zone_lock_in=True so the fresh burst focus is guaranteed
        # in-zone (when active_zones is set). The burst's K-tick hold
        # would otherwise amplify a single out-of-zone escape pick into
        # K ticks of distant focus.
        fresh = self._pick_focus_npc(zone_lock_in=True)
        if fresh is None:
            self._burst_focus_npc = None
            self._burst_remaining = 0
            return None
        self._burst_focus_npc = fresh
        self._burst_remaining = max(0, _BURST_ROTATION_DEPTH - 1)
        return fresh

    def reset_to_initial_state(self) -> dict:
        """
        Soft-reset the Director + every NPC to the world's initial
        YAML-authored state. Used by game clients to implement a
        "new game" or "restart from save" without restarting the
        server process.

        What happens:
          1. Born NPCs (Phase 2b births) are removed from the live
             profiles dict and their YAML files are deleted from
             ``npc_profiles/``.
          2. Deceased NPCs are restored to ``alive``.
          3. Every manifest NPC's mutable state (quests, events,
             dynamic knowledge lanes, player_knowledge) is wiped
             and the profile is re-loaded from its YAML file so
             the NPC returns to its authored baseline.
          4. All Director runtime state is zeroed: tick counter,
             rotation state, ledger, arcs, quest-line progress,
             identity trust, refusal timers, activity, pause,
             budget, zone set, pacing overrides, narration mode.

        State that is NOT cleared (stays from the prior boot):
          - Quest-lines config + zone config (re-loaded from YAML at
            init, stable across resets).
          - The dev-layer auto-refuse flag (read from quest_lines.yaml,
            not per-run state). Player intents ARE cleared.

        Returns a report dict summarizing what was cleaned.
        """
        profiles = self.engine.pie.npc_knowledge.profiles
        report: dict = {"born_removed": [], "deceased_restored": [],
                         "manifest_reloaded": []}

        # 1. Remove born NPCs.
        born_ids = set(profiles.keys()) - self._initial_cast
        for npc_id in list(born_ids):
            npc = profiles.pop(npc_id, None)
            if npc is not None:
                # Delete the YAML file the birth pipeline wrote.
                # Guard: only delete if the file stem matches this
                # NPC's id. Prevents accidental deletion of another
                # NPC's profile when a test-simulated birth shares a
                # profile_path from an existing NPC.
                try:
                    pp = getattr(npc, "profile_path", None)
                    if (pp is not None
                            and Path(pp).exists()
                            and Path(pp).stem == npc_id):
                        Path(pp).unlink()
                except Exception:
                    pass
                report["born_removed"].append(npc_id)

        # 2. Restore deceased + 3. Reload manifest profiles.
        for npc_id in list(self._initial_cast):
            npc = profiles.get(npc_id)
            if npc is None:
                continue
            was_dead = getattr(npc, "status", "alive") != "alive"
            # Clear mutable runtime state before reloading.
            npc.quests.clear()
            npc.events.clear()
            npc.dynamic_world_facts.clear()
            npc.dynamic_personal_knowledge.clear()
            # Re-read the YAML profile (identity, world_facts,
            # personal_knowledge, active_quests, capability_configs).
            try:
                npc._load()
            except Exception as e:
                logger.warning(f"reset: failed to reload {npc_id}: {e}")
            # Force alive even if _load read stale status from a
            # profile YAML that was manually edited.
            npc.status = "alive"
            npc.death_tick = None
            npc.death_cause = None
            npc.inheritor = None
            npc.player_knowledge = {
                "met": False,
                "recognized": False,
                "known_as": [],
                "witnessed_deeds": [],
                "heard_deeds": [],
                "first_met_tick": None,
                "last_interaction_tick": None,
            }
            if was_dead:
                report["deceased_restored"].append(npc_id)
            report["manifest_reloaded"].append(npc_id)

        # 4. Zero all Director runtime state.
        self.tick_count = 0
        self.last_tick_at = None
        self.recent_decisions.clear()
        self._npc_last_planned_tick.clear()
        self._burst_focus_npc = None
        self._burst_remaining = 0
        self._zone_escape_counter = 0
        self._kind_rotation_index = 0
        self._self_rep_retries_this_tick = 0
        self._bio_mention_counts.clear()
        self.recent_player_actions.clear()
        self._active_zones.clear()
        # Lifecycle
        self._deceased_npcs.clear()
        self._pending_death_requests.clear()
        self._pending_birth_requests.clear()
        self._birth_history.clear()
        self._autonomous_deaths_this_session = 0
        self._autonomous_births_this_session = 0
        # Phase 4a
        self._player_activity = PlayerActivity.UNKNOWN.value
        self._activity_set_at_tick = 0
        # Phase 4b
        self._last_quest_dispatched_per_npc.clear()
        self._max_unoffered_quests_override = None
        self._quest_cooldown_ticks_override = None
        # Phase 4c
        self._paused = False
        self._paused_at_tick = 0
        self._tick_budget_seconds = -1.0
        self._tick_time_log.clear()
        # Phase 3a — re-seed skeletons from config, clear refusal
        # timers + player auto-refuse intents.
        self._quest_line_state.clear()
        for line_id in self._quest_lines_config:
            self._quest_line_state[line_id] = {
                "dispatched_beats": [],
                "completed_quests": [],
                "rewards_earned": [],
                "line_status": "active",
            }
        self._refused_quest_timers.clear()
        self._player_auto_refuse_intents.clear()
        self._main_line_focus_counter = 0
        # Phase 5
        self._npc_player_identity_trust.clear()
        self._player_visible_feature = None
        self._visible_feature_to_identity.clear()
        # Narration mode
        self.narration_mode = "prose"
        self._reload_examples()

        # 5. Reset ledger + arcs. The predictive layer zeroes its edge
        # filters but KEEPS the prior matrix — the activity prior
        # generalizes across resets, the per-NPC edge filters don't
        # (plan: composition table, game-reset row). The activity
        # history JSONL also survives: it's the durable supervision
        # signal the prior was trained on.
        self.ledger.reset()
        self.arc_planner.reset()
        if self._predictive is not None:
            try:
                self._predictive.reset()
            except Exception as e:
                logger.warning(f"Predictive reset failed: {e}")
        self._last_activity_pred = None

        # 6. Re-snapshot original bios from the freshly-reloaded
        # profiles so bio rotation starts clean.
        self._snapshot_original_bios()

        # 7. Persist the clean state.
        self._save_state()

        report["tick_count"] = 0
        report["initial_cast"] = sorted(self._initial_cast)
        return {"ok": True, **report}

    def get_state(self) -> dict:
        return {
            "tick_count": self.tick_count,
            "last_tick_at": self.last_tick_at,
            "recent_decisions": self.recent_decisions,
            "recent_player_actions": self.recent_player_actions,
            "kind_rotation_index": self._kind_rotation_index,
            "player_activity": self._player_activity,
            "activity_set_at_tick": self._activity_set_at_tick,
            "lore_loaded": bool(self._lore_text),
            "example_count": len(self._examples),
            "ledger": self.ledger.stats(),
            "arc_planner": self.arc_planner.stats(),
            "predictive": self.get_predictive_state(),
        }

    def get_predictive_state(self) -> dict:
        """Predictive-lane observability rollup. Safe to call whether
        or not the layer is enabled — disabled returns a stub so
        /story/state consumers can always read ``predictive.enabled``."""
        if self._predictive is None:
            return {"enabled": False}
        try:
            stats = self._predictive.stats()
        except Exception as e:
            logger.warning(f"Predictive stats failed: {e}")
            return {"enabled": True, "error": str(e)}
        stats["boost_enabled"] = self._predictive_boost_enabled
        stats["last_activity_pred"] = self._last_activity_pred
        return stats

    def record_player_action(self, text: str,
                              target: Optional[str] = None,
                              trust_delta: Optional[int] = None,
                              quest_completed: Optional[str] = None,
                              quest_accepted: Optional[dict] = None,
                              *,
                              witness_npcs: Optional[list[str]] = None,
                              visible_feature: Optional[str] = None,
                              subject_identity: Optional[str] = None) -> dict:
        """
        Record something the player did. Surfaced in the next tick's
        world snapshot so the Director can react to player behavior.

        Args:
            text: Human-readable description of what the player did.
            target: Optional NPC id the action was directed at.
            trust_delta: Optional trust adjustment to apply to ``target``
                (positive = friendlier, negative = more hostile).
            quest_completed: Optional quest id; routes through
                ``engine.complete_quest`` and accumulates main-line
                rewards (Phase 3a).
            quest_accepted: Optional ``{id, name, given_by}`` dict.
            witness_npcs (Phase 5a): NPC ids who personally saw the
                deed. Each gets instant ``met = recognized = True``
                and the ledger index appended to their
                ``witnessed_deeds``. Non-witnesses receive it as
                ``heard_deeds`` via a simple gossip fallback.
            visible_feature (Phase 5a): String naming a feature the
                player exhibited during this action (distinctive
                cloak, bloodied hand, etc.). Stored on the Director
                and surfaced later by Phase 5b's auto-recognition.
            subject_identity (Phase 5a): Identity string the deed
                should be attributed to in the ledger + gossip
                (``"jordan"``, ``"the hooded stranger"``, ...).
                Defaults to ``"unknown_figure"`` when unspecified
                and witnesses are present; legacy callers without
                witnesses pass nothing and get a legacy entry.
        """
        if not text or not isinstance(text, str):
            return {"ok": False, "reason": "empty_text"}

        witnesses = [w for w in (witness_npcs or []) if isinstance(w, str)]
        if visible_feature is not None:
            self.set_player_visible_feature(visible_feature)
        # Phase 5b — auto-resolve subject_identity from the player's
        # current visible_feature when the caller didn't supply one
        # explicitly. Falls back to "unknown_figure" only if both
        # subject_identity AND any feature match are absent.
        effective_identity = subject_identity
        if effective_identity is None:
            feature_key = visible_feature or self._player_visible_feature
            if feature_key:
                mapped = self._visible_feature_to_identity.get(feature_key)
                if mapped:
                    effective_identity = mapped
        if effective_identity is None and witnesses:
            effective_identity = "unknown_figure"

        record = {
            "at": datetime.now(timezone.utc).isoformat(),
            "tick_at_time": self.tick_count,
            "text": text.strip()[:240],
            "target": target,
        }
        if witnesses:
            record["witness_npcs"] = list(witnesses)
        if effective_identity:
            record["subject_identity"] = effective_identity
        if visible_feature:
            record["visible_feature"] = visible_feature

        # Optional side-effect: apply the trust adjustment through the
        # engine so the next snapshot reflects it.
        if target and trust_delta:
            try:
                self.engine.adjust_trust(target, int(trust_delta),
                                          reason=f"player: {text[:60]}")
                record["trust_delta"] = int(trust_delta)
            except Exception as e:
                logger.warning(f"adjust_trust failed for {target}: {e}")

        # Optional side-effect: complete a quest. Routes through the
        # engine so trust boost + gossip propagation fire as a side
        # effect — a player ACTUALLY finishing a quest is one of the
        # strongest signals the Director can react to.
        if quest_completed:
            try:
                result = self.engine.complete_quest(str(quest_completed))
                record["quest_completed"] = str(quest_completed)
                if isinstance(result, dict) and "error" in result:
                    record["quest_completed_error"] = result["error"]
                # Phase 3a — reward-track accumulation on main-line
                # beat completion. No-op for quests with no quest_line
                # or worlds with no reward_track configured.
                reward_record = self._record_main_line_reward(str(quest_completed))
                if reward_record is not None:
                    record["main_line_reward"] = reward_record
            except Exception as e:
                logger.warning(f"complete_quest failed for {quest_completed}: {e}")
                record["quest_completed_error"] = str(e)

        # Optional side-effect: accept a quest from a specific NPC.
        if quest_accepted and isinstance(quest_accepted, dict):
            qid = quest_accepted.get("id")
            qname = quest_accepted.get("name") or qid
            qgiver = quest_accepted.get("given_by") or target or ""
            if qid and qgiver:
                try:
                    self.engine.accept_quest(str(qid), str(qname), str(qgiver))
                    record["quest_accepted"] = qid
                except Exception as e:
                    logger.warning(f"accept_quest failed for {qid}: {e}")
                    record["quest_accepted_error"] = str(e)
            else:
                record["quest_accepted_error"] = "missing_id_or_given_by"

        self.recent_player_actions.append(record)
        self.recent_player_actions = self.recent_player_actions[-8:]

        # Phase 5a — witness + gossip propagation. Fires only when the
        # caller supplied witnesses or an explicit subject_identity
        # (legacy callers without either skip the whole block).
        if witnesses or effective_identity:
            # Ledger entry for the deed, tagged with subject_identity
            # so gossip + reputation queries can group by identity.
            ledger_npc = target or (witnesses[0] if witnesses else "?")
            try:
                self.ledger.add(
                    text=text.strip()[:400],
                    npc_id=ledger_npc,
                    kind="player_action",
                    tick=self.tick_count + 1,
                    suggested_by="player_reaction",
                    subject_identity=effective_identity,
                )
            except Exception as e:
                logger.warning(f"ledger add for player action failed: {e}")

            ledger_idx = len(self.ledger.entries) - 1
            witness_set = {w for w in witnesses}
            propagated_witnesses: list[str] = []
            heard_list: list[str] = []
            for npc_id, npc in self.engine.pie.npc_knowledge.profiles.items():
                if getattr(npc, "status", "alive") != "alive":
                    continue
                pk = self._ensure_player_knowledge(npc)
                if npc_id in witness_set:
                    # Witnesses get instant recognition under the
                    # deed's subject_identity + the ledger index
                    # appended to witnessed_deeds.
                    if not pk["met"]:
                        pk["first_met_tick"] = self.tick_count + 1
                    pk["met"] = True
                    pk["recognized"] = True
                    pk["last_interaction_tick"] = self.tick_count + 1
                    if (effective_identity
                            and effective_identity not in pk["known_as"]):
                        pk["known_as"].append(effective_identity)
                    if ledger_idx >= 0 and ledger_idx not in pk["witnessed_deeds"]:
                        pk["witnessed_deeds"].append(ledger_idx)
                    if pk["known_as"]:
                        self._merge_identity_trust(npc_id, pk["known_as"])
                    propagated_witnesses.append(npc_id)
                else:
                    # Gossip fallback — the stub engine has no proper
                    # social graph, so every non-witness NPC learns
                    # the deed as hearsay. Real deployments will
                    # override with GossipPropagator. The deed is
                    # tagged with subject_identity so non-witnesses
                    # don't auto-recognize the player.
                    if ledger_idx >= 0 and ledger_idx not in pk["heard_deeds"]:
                        pk["heard_deeds"].append(ledger_idx)
                        heard_list.append(npc_id)
            record["witnessed_by"] = propagated_witnesses
            record["heard_by"] = heard_list
            record["ledger_index"] = ledger_idx

        self._save_state()
        return {"ok": True, "recorded": record}

    # ── Zone management ────────────────────────────────────────

    def set_active_zones(self, zones: Optional[list[str]]) -> dict:
        """
        Replace the Director's active zone set. Called by the game
        client whenever the player transitions between zones (usually
        POST /story/player_zone). Empty list or None = world-wide mode
        (every NPC considered, backward-compatible with pre-zone
        behaviour).

        Returns a dict the REST layer can pass straight back to the
        caller — ``{"ok": True, "active_zones": [...]}`` on success,
        ``{"ok": False, "reason": "..."}`` on validation failure.
        """
        if zones is None:
            new_zones: set[str] = set()
        elif isinstance(zones, (list, tuple, set)):
            new_zones = {str(z).strip() for z in zones if str(z).strip()}
        else:
            return {"ok": False, "reason": "zones must be a list of strings"}
        if new_zones != self._active_zones:
            logger.info(
                f"Story Director active_zones: {sorted(self._active_zones)} "
                f"-> {sorted(new_zones)}"
            )
            # Break burst rotation when the active zone set changes —
            # the sticky NPC from the prior zone shouldn't hold slot 0
            # after a zone transition.
            if self._burst_focus_npc and self._burst_focus_npc in (
                self.engine.pie.npc_knowledge.profiles
            ):
                npc = self.engine.pie.npc_knowledge.profiles[self._burst_focus_npc]
                current_zone = getattr(npc, "current_zone", "global")
                if (current_zone != "global"
                        and new_zones
                        and current_zone not in new_zones):
                    self._burst_focus_npc = None
                    self._burst_remaining = 0
        self._active_zones = new_zones
        self._save_state()
        return {"ok": True, "active_zones": sorted(self._active_zones)}

    def set_npc_current_zone(self, npc_id: str, zone: str) -> dict:
        """
        Update a mobile NPC's current zone. Rejects if the NPC is
        stationary (``mobile: false`` in profile YAML) or unknown.
        Game clients call this when moving a traveling merchant,
        wandering assassin, or any NPC whose profile declares
        mobility.
        """
        if not npc_id or not isinstance(npc_id, str):
            return {"ok": False, "reason": "missing npc_id"}
        if not zone or not isinstance(zone, str):
            return {"ok": False, "reason": "missing zone"}
        npc = self.engine.pie.npc_knowledge.profiles.get(npc_id)
        if npc is None:
            return {"ok": False, "reason": f"unknown npc '{npc_id}'"}
        if not getattr(npc, "mobile", False):
            return {
                "ok": False,
                "reason": f"npc '{npc_id}' is not mobile (profile has mobile=false)",
            }
        old_zone = getattr(npc, "current_zone", "global")
        npc.current_zone = zone.strip()
        logger.info(
            f"NPC '{npc_id}' moved: {old_zone} -> {npc.current_zone}"
        )
        # Break burst on the moved NPC if the destination leaves the
        # active zone set.
        if (self._burst_focus_npc == npc_id
                and self._active_zones
                and npc.current_zone not in self._active_zones
                and npc.current_zone != "global"):
            self._burst_focus_npc = None
            self._burst_remaining = 0
        self._save_state()
        return {
            "ok": True,
            "npc_id": npc_id,
            "previous_zone": old_zone,
            "current_zone": npc.current_zone,
        }

    def set_player_activity(self, activity: str) -> dict:
        """
        Update the player's current activity. Called by the game
        client on every meaningful state transition (entering a town,
        opening a menu, starting a fight, etc.) so the Director can
        self-pause or adjust cadence.

        Validates against ``PlayerActivity``. Unknown strings are
        rejected rather than silently coerced — a typo in the game
        integration should surface loudly in the REST response.
        """
        if not isinstance(activity, str):
            return {"ok": False, "reason": "activity must be a string"}
        activity = activity.strip()
        if activity not in _PLAYER_ACTIVITY_VALUES:
            return {
                "ok": False,
                "reason": (
                    f"unknown activity '{activity}'; must be one of "
                    f"{sorted(_PLAYER_ACTIVITY_VALUES)}"
                ),
            }
        if activity != self._player_activity:
            logger.info(
                f"Story Director activity: {self._player_activity} -> {activity} "
                f"(at tick {self.tick_count})"
            )
        self._player_activity = activity
        self._activity_set_at_tick = self.tick_count

        # Predictive lane — append the observation to the append-only
        # activity-history JSONL (the ActivityPrior's supervision
        # signal; written on EVERY post per the plan's schema section)
        # and harvest a (ledger-latent, activity) training pair. Both
        # are best-effort: a disk error must never fail the REST call.
        if self._predictive is not None:
            try:
                self._activity_history_file.parent.mkdir(
                    parents=True, exist_ok=True)
                with open(self._activity_history_file, "a",
                          encoding="utf-8") as f:
                    f.write(json.dumps({
                        "tick": self.tick_count,
                        "activity": activity,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }) + "\n")
            except Exception as e:
                logger.warning(f"activity_history append failed: {e}")
            try:
                self._predictive.note_activity(
                    self.tick_count, activity, self.ledger)
            except Exception as e:
                logger.warning(f"Predictive note_activity failed: {e}")

        self._save_state()
        return {
            "ok": True,
            "activity": self._player_activity,
            "activity_set_at_tick": self._activity_set_at_tick,
        }

    def get_player_activity(self) -> dict:
        """
        Return the current player activity and the tick at which it
        was last set. Used by the REST debug endpoint and by bench
        harnesses that need to verify activity-aware behavior.
        """
        return {
            "activity": self._player_activity,
            "activity_set_at_tick": self._activity_set_at_tick,
        }

    def set_quest_pacing(self,
                          max_unoffered: Optional[int] = None,
                          cooldown_ticks: Optional[int] = None) -> dict:
        """
        Override the Phase 4b per-NPC quest pacing tunables at runtime.
        Either argument left as ``None`` leaves that field untouched.
        Pass a negative integer to *clear* an override and fall back to
        the module-level default.

        Both values must be non-negative (0 is a valid choice — it
        disables new quest dispatches entirely, useful for cutscenes
        or tutorial phases). Persisted in ``state.json`` so a restart
        honors the pacing the game last configured.
        """
        if max_unoffered is not None:
            if not isinstance(max_unoffered, int):
                return {"ok": False, "reason": "max_unoffered must be an integer"}
            if max_unoffered < 0:
                self._max_unoffered_quests_override = None
            else:
                self._max_unoffered_quests_override = max_unoffered
        if cooldown_ticks is not None:
            if not isinstance(cooldown_ticks, int):
                return {"ok": False, "reason": "cooldown_ticks must be an integer"}
            if cooldown_ticks < 0:
                self._quest_cooldown_ticks_override = None
            else:
                self._quest_cooldown_ticks_override = cooldown_ticks
        self._save_state()
        return {
            "ok": True,
            "max_unoffered": self._effective_max_unoffered_quests(),
            "cooldown_ticks": self._effective_quest_cooldown_ticks(),
            "max_unoffered_override": self._max_unoffered_quests_override,
            "cooldown_ticks_override": self._quest_cooldown_ticks_override,
        }

    def pause_ticks(self) -> dict:
        """
        Phase 4c — hold every future tick() call until ``resume_ticks``
        is called. Does not bump tick_count, so cooldowns (Phase 4b,
        arc cooldowns) don't tick down against real time during a
        pause. Idempotent — pausing an already-paused Director is a
        no-op but still returns ok.
        """
        if not self._paused:
            self._paused = True
            self._paused_at_tick = self.tick_count
            logger.info(f"Story Director paused at tick {self.tick_count}")
            self._save_state()
        return {
            "ok": True,
            "paused": self._paused,
            "paused_at_tick": self._paused_at_tick,
        }

    def resume_ticks(self) -> dict:
        """Clear the explicit-pause flag. Idempotent."""
        if self._paused:
            self._paused = False
            logger.info(f"Story Director resumed at tick {self.tick_count}")
            self._save_state()
        return {
            "ok": True,
            "paused": self._paused,
            "paused_at_tick": self._paused_at_tick,
        }

    def get_pause_state(self) -> dict:
        """
        Return the full Phase 4c pause state: explicit flag, budget
        config, trailing-window usage snapshot, and the next-tick
        hint the client would receive right now.
        """
        self._prune_tick_time_log()
        window_sum = sum(dur for _ts, dur in self._tick_time_log)
        return {
            "paused": self._paused,
            "paused_at_tick": self._paused_at_tick,
            "tick_budget_seconds": self._tick_budget_seconds,
            "tick_budget_window_seconds": _TICK_BUDGET_WINDOW_SECONDS,
            "window_llm_seconds_used": round(window_sum, 3),
            "budget_exceeded": self._budget_exceeded(),
            "next_tick_recommended_in_seconds": self._compute_next_tick_hint(),
        }

    def set_tick_budget(self, max_seconds_per_minute: float) -> dict:
        """
        Set the rolling-window LLM-time budget. Pass a negative value
        (including -1) to clear the cap and run unconstrained — the
        pre-Phase-4c default. Non-numeric inputs are rejected.
        """
        if isinstance(max_seconds_per_minute, bool):
            return {"ok": False, "reason": "budget must be a number"}
        if not isinstance(max_seconds_per_minute, (int, float)):
            return {"ok": False, "reason": "budget must be a number"}
        value = float(max_seconds_per_minute)
        self._tick_budget_seconds = -1.0 if value < 0 else value
        self._save_state()
        return {
            "ok": True,
            "tick_budget_seconds": self._tick_budget_seconds,
            "tick_budget_window_seconds": _TICK_BUDGET_WINDOW_SECONDS,
        }

    def get_quest_pacing(self) -> dict:
        """
        Return the current effective quest pacing values plus the raw
        overrides (``None`` when the module default is in use).
        """
        return {
            "max_unoffered": self._effective_max_unoffered_quests(),
            "cooldown_ticks": self._effective_quest_cooldown_ticks(),
            "max_unoffered_override": self._max_unoffered_quests_override,
            "cooldown_ticks_override": self._quest_cooldown_ticks_override,
            "last_quest_dispatched_per_npc": dict(self._last_quest_dispatched_per_npc),
        }

    def get_zone_state(self) -> dict:
        """
        Return the current zone state for debug / audit. Maps every
        NPC to its current zone so the game client can verify
        synchronization.
        """
        npc_zones: dict[str, str] = {}
        for npc_id, npc in self.engine.pie.npc_knowledge.profiles.items():
            npc_zones[npc_id] = getattr(npc, "current_zone", "global")
        return {
            "active_zones": sorted(self._active_zones),
            "npc_zones": npc_zones,
        }

    # ── Lifecycle management (Phase 2a+) ────────────────────────

    def queue_death_request(self, npc_id: str, cause: str = "",
                             transfers_quests_to: Optional[str] = None) -> dict:
        """
        Game-authoritative death reporting. Appends a death request
        to the lifecycle queue; the next ``tick()`` drains the queue
        via ``_lifecycle_tick`` and actually dispatches the deaths.

        Queueing rather than dispatching immediately means deaths
        and story beats interleave predictably: every death is
        observable in the bench trace as a lifecycle action between
        ticks rather than as a mutation mid-tick.
        """
        if not npc_id or not isinstance(npc_id, str):
            return {"ok": False, "reason": "missing npc_id"}
        npc = self.engine.pie.npc_knowledge.profiles.get(npc_id)
        if npc is None:
            return {"ok": False, "reason": f"unknown npc '{npc_id}'"}
        if getattr(npc, "status", "alive") != "alive":
            return {
                "ok": False,
                "reason": f"npc '{npc_id}' is not alive (status={getattr(npc, 'status', '?')})",
            }
        request = {
            "npc_id": npc_id,
            "cause": cause or "unspecified",
            "transfers_quests_to": transfers_quests_to,
            "queued_at_tick": self.tick_count,
        }
        self._pending_death_requests.append(request)
        self._save_state()
        return {"ok": True, "queued": request}

    def get_graveyard(self) -> dict:
        """
        Return the full death history for audit / bench visibility.
        Contains every NPC the Director has ever marked deceased in
        this session, keyed by npc_id.
        """
        return {
            "deceased": self._deceased_npcs,
            "pending": list(self._pending_death_requests),
        }

    def _lifecycle_tick(self) -> list[dict]:
        """
        Run at the start of every tick() call, before the architect
        plan. Drains pending death requests and dispatches each one
        in FIFO order, then checks for pending births (explicit
        requests + population gap fills).

        Returns the list of lifecycle actions that ran (death
        records, birth records, etc.) for bench trace visibility.
        At most one death and one birth fire per tick so the core
        story-beat cadence isn't overwhelmed.
        """
        actions: list[dict] = []
        # Drain at most one death per tick. Extra requests stay
        # queued for subsequent ticks.
        if self._pending_death_requests:
            request = self._pending_death_requests.pop(0)
            result = self._dispatch_npc_death(
                npc_id=request["npc_id"],
                cause=request.get("cause", "unspecified"),
                transfers_quests_to=request.get("transfers_quests_to"),
            )
            actions.append({"kind": "npc_death", "result": result})

        # Autonomous death proposal — fires AFTER explicit deaths
        # drain, so a game-client death and a Director-proposed
        # death can't stack in the same tick.
        if (self._autonomous_lifecycle
                and not self._pending_death_requests
                and not any(a.get("kind") == "npc_death" for a in actions)):
            death_proposal = self._propose_autonomous_death()
            if death_proposal:
                result = self._dispatch_npc_death(
                    npc_id=death_proposal["npc_id"],
                    cause=death_proposal.get("cause", "autonomous"),
                    transfers_quests_to=death_proposal.get("transfers_quests_to"),
                )
                actions.append({"kind": "npc_death_auto", "result": result})

        # Drain at most one birth per tick (explicit request first,
        # then population gap fill if nothing explicit is queued).
        if self._pending_birth_requests:
            request = self._pending_birth_requests.pop(0)
            result = self._dispatch_npc_birth(request)
            actions.append({"kind": "npc_birth", "result": result})
        elif self._zone_config:
            # Automatic population gap fill — check if any zone is
            # below its min_population and queue a birth request.
            gap = self._find_population_gap()
            if gap:
                if (not self._autonomous_lifecycle
                        or self._autonomous_births_this_session < _MAX_AUTONOMOUS_BIRTHS_PER_SESSION):
                    result = self._dispatch_npc_birth(gap)
                    if self._autonomous_lifecycle:
                        self._autonomous_births_this_session += 1
                    actions.append({"kind": "npc_birth_auto", "result": result})

        # Phase 3a — unlock quests whose prerequisites are now
        # satisfied. Runs every lifecycle tick so a quest completed
        # via REST shows up as offerable on the next Director beat.
        unlocked = self._unlock_quests_if_prereqs_met()
        if unlocked:
            actions.append({"kind": "quests_unlocked", "unlocked": unlocked})

        # Phase 3a — expire decay-mode refusal timers. When a refusal
        # was recorded with refusal_mode='decay' and the unlock tick
        # has arrived, flip the quest back to 'available' and emit a
        # subtle re-open ledger entry.
        expired = self._expire_refusal_timers()
        if expired:
            actions.append({"kind": "refusals_expired", "expired": expired})

        return actions

    def _expire_refusal_timers(self) -> list[dict]:
        """Walk ``_refused_quest_timers`` and reopen every quest whose
        unlock tick has arrived. Returns one record per reopened
        quest for lifecycle-tick reporting."""
        if not self._refused_quest_timers:
            return []
        expired: list[dict] = []
        still_pending: dict[tuple[str, str], int] = {}
        for (npc_id, quest_id), unlock_tick in self._refused_quest_timers.items():
            if self.tick_count + 1 < unlock_tick:
                still_pending[(npc_id, quest_id)] = unlock_tick
                continue
            npc = self.engine.pie.npc_knowledge.profiles.get(npc_id)
            if npc is None:
                continue
            for q in getattr(npc, "quests", []):
                if q.id == quest_id and q.status == "refused":
                    q.status = "available"
                    giver_name = npc.identity.get("name", npc_id)
                    self._inject_tagged_event(
                        f"{giver_name}'s offer is open again: {q.name}",
                        npc_id=None,
                    )
                    expired.append({
                        "quest_id": quest_id, "npc_id": npc_id,
                        "reopened_at_tick": self.tick_count + 1,
                    })
                    break
        self._refused_quest_timers = still_pending
        return expired

    def _find_population_gap(self) -> Optional[dict]:
        """
        Scan zone configs for any zone below its ``min_population``.
        Returns a birth request dict for the most depleted zone, or
        None if every zone is at or above its floor. Only considers
        alive NPCs in each zone — deceased NPCs don't count toward
        the population floor.
        """
        alive = self._alive_npcs()
        worst_gap = 0
        worst_zone = None
        for zone_name, cfg in self._zone_config.items():
            if not isinstance(cfg, dict):
                continue
            min_pop = cfg.get("min_population", 0)
            if min_pop <= 0:
                continue
            alive_in_zone = sum(
                1 for npc in alive.values()
                if getattr(npc, "current_zone", "global") == zone_name
            )
            gap = min_pop - alive_in_zone
            if gap > worst_gap:
                worst_gap = gap
                worst_zone = zone_name
        if worst_zone is None:
            return None
        role_pool = self._zone_config[worst_zone].get("role_pool", [])
        import random
        role = random.choice(role_pool) if role_pool else "wanderer"
        return {
            "zone": worst_zone,
            "role": role,
            "reason": f"population_below_minimum (gap={worst_gap})",
        }

    def queue_birth_request(self, zone: str, role: Optional[str] = None,
                             reason: str = "game_requested") -> dict:
        """
        Game-authoritative birth request. The next lifecycle tick
        dispatches a template-based NPC generation into the specified
        zone. Returns immediately with the queued request.
        """
        if not zone or not isinstance(zone, str):
            return {"ok": False, "reason": "missing zone"}
        request = {
            "zone": zone,
            "role": role or "wanderer",
            "reason": reason,
            "queued_at_tick": self.tick_count,
        }
        self._pending_birth_requests.append(request)
        self._save_state()
        return {"ok": True, "queued": request}

    def _dispatch_npc_birth(self, request: dict) -> dict:
        """
        Generate a new NPC profile from a template, write it to the
        world's ``npc_profiles/`` directory, and register it at
        runtime via ``engine.add_profile()``.

        Phase 2b ships with **template-based generation** — Python
        scaffolds the YAML structure, picks a name from a small
        procedural pool, and fills in minimal narrative fields. A
        future enhancement wires in LLM-based generation for richer
        character flavor, but the template path is sufficient for
        population management and stress testing.
        """
        import time as _time
        import random

        zone = request.get("zone", "global")
        role = request.get("role", "wanderer")
        reason = request.get("reason", "")

        # Generate a unique id + name
        tick_stamp = self.tick_count + 1
        # Name pool — small but distinct enough for 20-30 births
        _FIRST_NAMES = [
            "Aldric", "Bryn", "Cael", "Dara", "Elara", "Finn", "Greta",
            "Halvar", "Iona", "Jareth", "Kira", "Lysander", "Maren",
            "Nils", "Orin", "Petra", "Quinn", "Rowan", "Sable", "Thane",
            "Ula", "Voss", "Wren", "Xander", "Yara", "Zane",
        ]
        _SURNAMES = [
            "Ashford", "Blackthorn", "Crestfall", "Dunmoor", "Elderwood",
            "Frosthold", "Grimshaw", "Holloway", "Ironside", "Juniper",
            "Kettleworth", "Larkspur", "Mossfield", "Nightwhisper",
            "Oakenshield", "Pendrake", "Ravenscar", "Stoneacre",
            "Thornbury", "Underhill", "Valewind", "Wolfsbane",
        ]
        # Avoid name collisions with existing NPCs
        existing_names = {
            npc.identity.get("name", "").lower()
            for npc in self.engine.pie.npc_knowledge.profiles.values()
        }
        for _ in range(50):
            first = random.choice(_FIRST_NAMES)
            surname = random.choice(_SURNAMES)
            full_name = f"{first} {surname}"
            if full_name.lower() not in existing_names:
                break
        else:
            full_name = f"Stranger T{tick_stamp}"

        npc_id = f"gen_t{tick_stamp}_{full_name.lower().replace(' ', '_')}"

        # Zone lore hook for personality seeding
        zone_cfg = self._zone_config.get(zone, {})
        lore_hook = zone_cfg.get("lore_hook", "a settlement")

        # Build the profile YAML
        profile_yaml = {
            "identity": {
                "name": full_name,
                "role": role.replace("_", " ").title(),
                "location": f"Somewhere in the {zone.replace('_', ' ')}",
                "personality": f"A {role.replace('_', ' ')} from the {zone.replace('_', ' ')}.",
                "speech_style": "Speaks plainly.",
            },
            "zone": zone,
            "mobile": False,
            "status": "alive",
            "generated": True,
            "generated_at_tick": tick_stamp,
            "world_facts": [
                f"Lives in the {zone.replace('_', ' ')} area",
            ],
            "personal_knowledge": [
                f"Arrived recently looking for work as a {role.replace('_', ' ')}",
            ],
            "active_quests": [],
            "recent_events": [],
            "capabilities": {
                "scratchpad": {"enabled": True, "max_entries": 6},
                "trust": {
                    "enabled": True,
                    "initial_level": 20,
                    "thresholds": {"wary": 0, "neutral": 25, "friendly": 50, "trusted": 75},
                },
                "emotional_state": {
                    "enabled": True,
                    "baseline_mood": "neutral",
                    "volatility": 0.4,
                    "decay_rate": 0.2,
                },
                "goals": {
                    "enabled": True,
                    "active_goals": [{
                        "id": f"settle_in_{zone}",
                        "description": f"Find a place to settle in the {zone.replace('_', ' ')}",
                        "priority": 6,
                        "keywords": [zone.replace("_", " "), "work", "settle"],
                    }],
                },
                "gossip": {"enabled": True, "max_rumors": 3, "interests": ["all"]},
            },
        }

        # Write to the world's npc_profiles directory
        world_dir_str = getattr(
            getattr(self.engine, "config", None), "world_dir", None,
        )
        if not world_dir_str:
            return {"ok": False, "reason": "no world_dir configured"}
        profiles_dir = Path(world_dir_str) / "npc_profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        profile_path = profiles_dir / f"{npc_id}.yaml"
        try:
            profile_path.write_text(
                yaml.dump(profile_yaml, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )
        except Exception as e:
            return {"ok": False, "reason": f"failed to write profile: {e}"}

        # Register the new NPC at runtime
        add_result = self.engine.add_profile(
            str(profile_path),
            social_connections=[],  # template births start with no connections
        )
        if not add_result.get("ok"):
            return {"ok": False, "reason": f"add_profile failed: {add_result.get('reason')}"}

        # Emit a FactLedger entry so existing NPCs can gossip about
        # the newcomer.
        ledger_text = (
            f"A new {role.replace('_', ' ')} named {full_name} "
            f"has arrived in the {zone.replace('_', ' ')}"
        )
        try:
            self.ledger.add(
                text=ledger_text, npc_id=npc_id,
                kind="birth", tick=tick_stamp,
            )
        except Exception as e:
            logger.warning(f"ledger add for birth of {npc_id} failed: {e}")

        record = {
            "npc_id": npc_id,
            "name": full_name,
            "role": role,
            "zone": zone,
            "birth_tick": tick_stamp,
            "reason": reason,
            "profile_path": str(profile_path),
        }
        self._birth_history.append(record)
        logger.info(
            f"StoryDirector: NPC '{npc_id}' born at T{tick_stamp} "
            f"zone={zone} role={role} reason={reason}"
        )
        return {"ok": True, "record": record}

    def set_autonomous_lifecycle(self, enabled: bool) -> dict:
        """
        Toggle autonomous lifecycle mode. When enabled, the Director
        can propose NPC deaths (arc-driven) and births
        (population-driven) without game client input. Off by
        default. Returns the new state for confirmation.
        """
        prev = self._autonomous_lifecycle
        self._autonomous_lifecycle = bool(enabled)
        if prev != self._autonomous_lifecycle:
            logger.info(
                f"Story Director autonomous lifecycle: {prev} -> "
                f"{self._autonomous_lifecycle}"
            )
        return {
            "ok": True,
            "autonomous": self._autonomous_lifecycle,
            "deaths_this_session": self._autonomous_deaths_this_session,
            "births_this_session": self._autonomous_births_this_session,
        }

    def _propose_autonomous_death(self) -> Optional[dict]:
        """
        When autonomous lifecycle is on, propose killing an NPC that
        would narratively resolve an active arc at or past the
        ``confront`` beat. Returns a death request dict if a suitable
        candidate is found, None otherwise.

        Bounded by ``_MAX_AUTONOMOUS_DEATHS_PER_SESSION`` so the
        Director can't depopulate the world in a single run. Jordan's
        design rule: the Director can kill ANYONE — no gating on
        zone, cast importance, or player proximity. The stress bench
        is the arbiter of whether this stays stable.
        """
        if not self._autonomous_lifecycle:
            return None
        if self._autonomous_deaths_this_session >= _MAX_AUTONOMOUS_DEATHS_PER_SESSION:
            return None
        alive = self._alive_npcs()
        # Phase 3a — main-line protected givers. The Director must not
        # autonomously kill an NPC whose loss would break an authored
        # main-line. Explicit /story/npc_death from the game client
        # still kills these; this guard only applies to Director-
        # proposed deaths.
        protected = self._protected_givers()
        for arc in self.arc_planner.active_arcs():
            if arc.current_beat < 2:
                continue  # only confront or later
            # Find a cast member who's alive, not main-line-protected,
            # and could die narratively
            for npc_id in arc.focus_npcs:
                if npc_id in protected:
                    logger.debug(
                        f"Autonomous death skipped: {npc_id} is a "
                        f"protected main-line giver"
                    )
                    continue
                if npc_id in alive:
                    self._autonomous_deaths_this_session += 1
                    logger.info(
                        f"Autonomous death proposed: {npc_id} "
                        f"(arc {arc.id} at beat {arc.current_beat_label})"
                    )
                    return {
                        "npc_id": npc_id,
                        "cause": f"narrative resolution of arc '{arc.theme[:60]}'",
                        "transfers_quests_to": None,
                    }
        return None

    def get_population_state(self) -> dict:
        """
        Per-zone alive count + target + min for debug / game UI.
        """
        alive = self._alive_npcs()
        zones: dict[str, dict] = {}
        for zone_name, cfg in self._zone_config.items():
            if not isinstance(cfg, dict):
                continue
            alive_in_zone = sum(
                1 for npc in alive.values()
                if getattr(npc, "current_zone", "global") == zone_name
            )
            zones[zone_name] = {
                "alive": alive_in_zone,
                "target": cfg.get("target_population", 0),
                "min": cfg.get("min_population", 0),
                "gap": max(0, cfg.get("min_population", 0) - alive_in_zone),
            }
        return {
            "total_alive": len(alive),
            "total_deceased": len(self._deceased_npcs),
            "total_born": len(self._birth_history),
            "zones": zones,
        }

    def _dispatch_npc_death(self, npc_id: str, cause: str,
                             transfers_quests_to: Optional[str]) -> dict:
        """
        Actually mark an NPC as deceased and propagate the death
        through arcs, quests, FactLedger, and burst rotation.

        Invariant: never touches an NPC that's already deceased
        (queue layer validates) and never mutates the profile's
        immutable fields.
        """
        npc = self.engine.pie.npc_knowledge.profiles.get(npc_id)
        if npc is None:
            return {"ok": False, "reason": f"unknown npc '{npc_id}'"}
        if getattr(npc, "status", "alive") != "alive":
            return {"ok": False, "reason": f"npc '{npc_id}' already dead"}

        death_tick = self.tick_count + 1
        npc.status = "deceased"
        npc.death_tick = death_tick
        npc.death_cause = cause
        if transfers_quests_to:
            npc.inheritor = transfers_quests_to

        # Arc cleanup: transition any arcs whose cast includes this NPC
        arc_result = self.arc_planner.on_cast_death(
            npc_id=npc_id, current_tick=death_tick
        )

        # Quest cleanup: abort or transfer every open quest.
        quest_records: list[dict] = []
        for q in list(getattr(npc, "quests", [])):
            if q.status not in ("available", "active"):
                continue
            if transfers_quests_to:
                target = self.engine.pie.npc_knowledge.profiles.get(
                    transfers_quests_to
                )
                if target is not None:
                    target.quests.append(q)
                    quest_records.append({
                        "quest_id": q.id,
                        "transition": "transferred",
                        "to": transfers_quests_to,
                    })
                    continue
            # No inheritor or inheritor not found → mark aborted
            q.status = "aborted"
            quest_records.append({
                "quest_id": q.id,
                "transition": "aborted",
                "reason": "giver_deceased",
            })

        # FactLedger entry so gossip can carry the news across ticks.
        if cause:
            ledger_text = (
                f"{npc.identity.get('name', npc_id)} is dead — {cause}"
            )
        else:
            ledger_text = (
                f"{npc.identity.get('name', npc_id)} has died"
            )
        try:
            self.ledger.add(
                text=ledger_text,
                npc_id=npc_id,
                kind="death",
                tick=death_tick,
            )
        except Exception as e:
            logger.warning(f"ledger add for death of {npc_id} failed: {e}")

        # Break burst rotation if the deceased NPC was holding slot 0.
        if self._burst_focus_npc == npc_id:
            self._burst_focus_npc = None
            self._burst_remaining = 0

        # Record in _deceased_npcs for persistence + snapshot.
        record = {
            "npc_id": npc_id,
            "name": npc.identity.get("name", npc_id),
            "role": npc.identity.get("role", ""),
            "zone": getattr(npc, "current_zone", "global"),
            "death_tick": death_tick,
            "death_cause": cause,
            "inheritor": transfers_quests_to,
            "arcs_affected": arc_result.get("arcs_affected", []),
            "quests_cleaned": quest_records,
        }
        self._deceased_npcs[npc_id] = record

        logger.info(
            f"StoryDirector: NPC '{npc_id}' deceased at T{death_tick} "
            f"cause='{cause}' arcs_affected={len(arc_result.get('arcs_affected', []))} "
            f"quests_cleaned={len(quest_records)}"
        )
        return {"ok": True, "record": record}

    def _alive_npcs(self) -> dict:
        """
        Return a dict of npc_id -> NPCKnowledge for every currently-
        alive NPC. Used by focus selection and the snapshot builder
        to filter out deceased NPCs from the pool.
        """
        return {
            npc_id: npc
            for npc_id, npc in self.engine.pie.npc_knowledge.profiles.items()
            if getattr(npc, "status", "alive") == "alive"
        }

    # ── World snapshot ──────────────────────────────────────────

    # Big-world tunables. The snapshot grows linearly with cast size on
    # the unbounded path, which dominates the prompt at 100+ NPCs. Above
    # the bound threshold, the snapshot is capped to a small "active
    # scene" (planned focus + arcs + recently-touched + recent player
    # targets, deduplicated, capped at SNAPSHOT_NPC_CAP). Below the
    # threshold the unbounded snapshot is identical to the pre-bound
    # behaviour so all existing tests and small-world benches see no
    # shape change.
    _SNAPSHOT_BOUND_THRESHOLD = 30
    _SNAPSHOT_NPC_CAP = 16

    def _select_snapshot_npcs(self,
                              all_npc_ids: list[str],
                              planned_focus_ids: Optional[list[str]] = None,
                              ) -> list[str]:
        """
        Pick the subset of NPCs to surface in a bounded world snapshot.

        Selection order (highest priority first):
          1. Planned focus NPCs for this tick (architect's choice — must
             always reach the snapshot or the LLM can't reason about
             them).
          2. **Active-zone NPCs** (when ``_active_zones`` is non-empty) —
             ensures the player's locale dominates the snapshot. Global
             NPCs are included too since they're always in every zone.
          3. NPCs in any active narrative arc's cast (continuity for
             multi-tick threads, even if the cast drifted out of zone).
          4. Last 8 distinct NPCs from ``recent_decisions`` (recent
             activity continuity).
          5. Last 4 distinct NPCs targeted by ``recent_player_actions``
             (player reactivity is the highest-stakes signal we have).

        The result is deduplicated, capped at ``_SNAPSHOT_NPC_CAP``, and
        intersected with the actual profile list (so a stale id from
        recent_decisions can't crash the snapshot builder).
        """
        seen = set()
        ordered: list[str] = []
        all_set = set(all_npc_ids)

        def _add(npc_id: str) -> None:
            if npc_id and npc_id not in seen and npc_id in all_set:
                seen.add(npc_id)
                ordered.append(npc_id)

        # 1. Planned focus first
        for npc_id in planned_focus_ids or []:
            _add(npc_id)

        # 1b. Phase 3a — main-line cast priority. Just below planned
        # focus so a dispatched beat on another main-line giver never
        # gets pushed off-camera; still below the architect's own
        # picks so one-off quests still receive their focus frame.
        for npc_id in self._main_line_cast():
            _add(npc_id)

        # 2. Active-zone NPCs (zone priority tier). World-wide mode
        # (empty active_zones) skips this — identical to the pre-zone
        # snapshot behaviour.
        if self._active_zones:
            for npc_id in all_npc_ids:
                if self._npc_in_active_zone(npc_id):
                    _add(npc_id)
                    if len(ordered) >= self._SNAPSHOT_NPC_CAP:
                        break

        # 3. Active arc casts
        try:
            for arc in self.arc_planner.active_arcs():
                for npc_id in (arc.focus_npcs or []):
                    _add(npc_id)
        except Exception:
            pass

        # 3. Recent decisions tail
        recent_decision_npcs: list[str] = []
        for d in reversed(self.recent_decisions):
            actions_in_decision: list[dict] = []
            if isinstance(d.get("action"), dict):
                actions_in_decision.append(d["action"])
            for sub in d.get("sub_actions", []) or []:
                if isinstance(sub, dict) and isinstance(sub.get("action"), dict):
                    actions_in_decision.append(sub["action"])
            for act in actions_in_decision:
                for key in ("npc_id", "target"):
                    val = act.get(key)
                    if isinstance(val, str):
                        recent_decision_npcs.append(val)
        for npc_id in recent_decision_npcs[:8]:
            _add(npc_id)

        # 4. Recent player action targets
        recent_player_targets = [
            pa.get("target") for pa in self.recent_player_actions
            if isinstance(pa.get("target"), str)
        ]
        for npc_id in reversed(recent_player_targets[-4:]):
            _add(npc_id)

        return ordered[:self._SNAPSHOT_NPC_CAP]

    def _world_snapshot(self, planned_focus_ids: Optional[list[str]] = None) -> str:
        """
        Compact world-state description the overseer will reason over.

        Events tagged with source="director" are FILTERED OUT — the Director
        should never see its own outputs as "world state" or it will echo
        them back and spiral into repetition. The Director's own past
        actions appear in a separate ALREADY DONE block below.

        For worlds with cast size > ``_SNAPSHOT_BOUND_THRESHOLD``, the
        per-NPC enumeration is capped to a small "active scene" instead
        of listing every profile in the world. ``planned_focus_ids`` is
        the per-tick architect plan; passing it ensures the LLM sees
        whichever NPCs the Python rotation just picked. Below the
        threshold the snapshot is identical to the pre-bound behaviour.
        """
        pie = self.engine.pie
        lines: list[str] = []

        world_name = self.engine.config.world_name or "Ashenvale"
        lines.append(f"World: {world_name}")

        # Only ALIVE NPCs reach the live roster. Deceased NPCs appear
        # in a separate "RECENTLY DEPARTED" section below when any
        # active arc cast references them, so the Director can write
        # aftermath beats without treating the dead as still walking.
        all_npc_ids = [
            npc_id for npc_id, npc in pie.npc_knowledge.profiles.items()
            if getattr(npc, "status", "alive") == "alive"
        ]
        cast_total = len(all_npc_ids)
        if cast_total > self._SNAPSHOT_BOUND_THRESHOLD:
            selected = self._select_snapshot_npcs(all_npc_ids, planned_focus_ids)
            lines.append(
                f"World cast: {cast_total} NPCs (showing {len(selected)} active "
                f"in scene; the rest exist but are off-camera this tick)"
            )
            iter_ids = selected
        else:
            iter_ids = all_npc_ids

        # NPCs with role + current mood/trust/top goal if available. The
        # top goal is the single biggest piece of motivational fuel for
        # the Director — with it, every NPC in the roster tells the
        # model WHAT THEY WANT, not just what they are.
        profiles = pie.npc_knowledge.profiles
        for npc_id in iter_ids:
            npc = profiles.get(npc_id)
            if npc is None:
                continue
            role = npc.identity.get("role", "")
            mood, trust = self._peek_npc_state(npc_id)
            bits = [f"{npc_id} ({role})"]
            if mood:
                bits.append(f"mood={mood}")
            if trust is not None:
                bits.append(f"trust={trust}")
            quests_here = [q for q in npc.quests if q.status in ("available", "active")]
            if quests_here:
                bits.append(f"quests={len(quests_here)}")
            goals = self._peek_npc_goals(npc_id)
            if goals:
                top_desc = str(goals[0].get("description", "")).strip()
                if top_desc:
                    bits.append(f"wants: {top_desc[:70]}")
            lines.append("  - " + ", ".join(bits))

        # Player block — surface everything the Director needs to react
        # to player behavior in one contiguous section. Includes quest
        # state, per-NPC trust, and the last 5 recorded player actions.
        player_lines = self._build_player_block()
        if player_lines:
            lines.append("PLAYER:")
            for pl in player_lines:
                lines.append("  " + pl)

        # Recent NON-DIRECTOR events. Filtering by source is critical:
        # feeding Director outputs back in as "world state" causes a
        # repetition spiral on small models. Walk only the selected
        # ``iter_ids`` so on bounded snapshots we don't pull events from
        # off-camera NPCs that the LLM has no context for.
        seen = set()
        recent_events: list[str] = []
        for npc_id in iter_ids:
            npc = profiles.get(npc_id)
            if npc is None:
                continue
            for e in npc.events[-4:]:
                if getattr(e, "source", "") == "director":
                    continue
                if e.description not in seen:
                    seen.add(e.description)
                    recent_events.append(e.description)
        if recent_events:
            lines.append("Recent organic events: " + " | ".join(recent_events[-5:]))

        # RECENTLY DEPARTED — surface any NPCs who died in the last
        # 10 ticks so the Director can write aftermath beats and
        # continuing-character narration that references the dead
        # without pretending they're still around. Only listed if
        # the deceased NPC is in an active arc's cast OR died very
        # recently; avoids dumping the entire graveyard every tick.
        departed_lines: list[str] = []
        recent_tick_floor = self.tick_count - 10
        arc_cast_ids: set[str] = set()
        for arc in self.arc_planner.active_arcs():
            arc_cast_ids.update(arc.focus_npcs or [])
        for npc_id, record in self._deceased_npcs.items():
            dt = record.get("death_tick", 0)
            in_arc = npc_id in arc_cast_ids
            if in_arc or dt >= recent_tick_floor:
                name = record.get("name", npc_id)
                role = record.get("role", "")
                cause = record.get("death_cause", "")
                line = f"{name}"
                if role:
                    line += f" ({role})"
                if cause:
                    line += f" — {cause}"
                line += f" [T{dt}]"
                departed_lines.append(line)
        if departed_lines:
            lines.append("RECENTLY DEPARTED:")
            for line in departed_lines[:5]:
                lines.append("  - " + line)

        # Director's own past actions — explicit DO NOT REPEAT list.
        already_done = self._format_already_done()
        if already_done:
            lines.append("ALREADY DONE (do not repeat any of these):")
            for line in already_done:
                lines.append("  - " + line)

        return "\n".join(lines)

    def _pick_examples(self, focus_npc: Optional[str],
                        action_kind: Optional[str]) -> list[dict]:
        """
        Pick a small subset of examples to show THIS worker, instead of
        dumping the whole library every tick. Two rules:

        1. **Exclude examples about the focus NPC.** The 3B bio-injection
           bench showed that when a Kael-focused tick sees the
           ``missing_hammers`` example directly, the model rewrites it
           verbatim instead of pulling from Kael's bio goals. Removing
           the focus-NPC example breaks that copy loop and lets the bio
           compete for salience.

        2. **Prefer one example matching the target action_kind.** The
           forced-focus block already says the worker MUST emit a given
           kind; showing one example of that kind stabilizes the schema
           for that output. Fill the remaining slots with different
           kinds for variety so the model sees all shapes.

        Returns at most ``_max_examples_for_mode()`` picks. Falls back
        to the full library if filtering leaves nothing (never emit an
        empty EXAMPLES block — schema parse reliability drops when the
        model loses its shape reference entirely).
        """
        if not self._examples:
            return []

        # Terse mode uses 2 picks instead of 3 — the terse library
        # examples are ~15 words each so 2 is plenty to cover both
        # the target action kind and one alternate shape. Saves
        # ~100-150 tokens per prompt with no measurable quality loss.
        max_picks = 2 if self.narration_mode == "terse" else 3

        # Rule 1: exclude examples whose primary_npc matches focus_npc
        eligible = [
            ex for ex in self._examples
            if focus_npc is None or ex.get("primary_npc") != focus_npc
        ]
        if not eligible:
            # All examples were about the focus NPC (shouldn't happen with
            # 5+ examples and 7 NPCs, but guard anyway)
            eligible = list(self._examples)

        # Rule 2: prioritize one example matching the target action_kind
        picks: list[dict] = []
        if action_kind:
            for ex in eligible:
                if ex.get("action", {}).get("action") == action_kind:
                    picks.append(ex)
                    break

        # Fill remaining slots with examples of OTHER kinds for diversity.
        # Prefer kinds we haven't shown yet this tick.
        shown_kinds = {
            p.get("action", {}).get("action") for p in picks
        }
        for ex in eligible:
            if len(picks) >= max_picks:
                break
            if ex in picks:
                continue
            ex_kind = ex.get("action", {}).get("action")
            # Skip if we already have this kind AND we haven't filled up
            if ex_kind in shown_kinds and len(picks) < max_picks:
                continue
            picks.append(ex)
            shown_kinds.add(ex_kind)

        # If we still don't have the target count, top up with anything left
        for ex in eligible:
            if len(picks) >= max_picks:
                break
            if ex not in picks:
                picks.append(ex)

        return picks[:max_picks]

    # ── Phase 3a — main-line helpers ────────────────────────────

    def _active_quest_lines(self) -> dict[str, dict]:
        """Return {line_id: config} for every line whose runtime
        state is 'active' (i.e. not completed or abandoned). Lines
        with no runtime record yet are treated as active."""
        out: dict[str, dict] = {}
        for line_id, cfg in self._quest_lines_config.items():
            if not isinstance(cfg, dict):
                continue
            st = self._quest_line_state.get(line_id)
            if st is None or st.get("line_status", "active") == "active":
                out[line_id] = cfg
        return out

    def _main_line_cast(self) -> set[str]:
        """NPC ids that appear as a ``giver`` in any beat of any
        active main-type line. Used by focus weighting, snapshot
        priority, and the prompt preference block."""
        cast: set[str] = set()
        for cfg in self._active_quest_lines().values():
            if cfg.get("type") != "main":
                continue
            for beat in cfg.get("beats", []) or []:
                if isinstance(beat, dict) and isinstance(beat.get("giver"), str):
                    cast.add(beat["giver"])
        return cast

    def _protected_givers(self) -> set[str]:
        """NPC ids the game has flagged as untouchable for
        autonomous death (per-line ``protected_givers``). The union
        across every active line, main or side. Explicit
        /story/npc_death from the game client still kills these."""
        protected: set[str] = set()
        for cfg in self._active_quest_lines().values():
            for nid in cfg.get("protected_givers", []) or []:
                if isinstance(nid, str):
                    protected.add(nid)
        return protected

    def _is_main_line_npc(self, npc_id: str) -> bool:
        """True iff ``npc_id`` is a giver on any active main-type
        line. Cheap enough to call per focus pick."""
        return npc_id in self._main_line_cast()

    def _build_intent_guidance_block(self, focus_npc: str) -> Optional[str]:
        """Phase 3a — ``GIVER CONTEXT`` + ``INTENT GUIDANCE`` prompt
        block. GIVER CONTEXT lines come from whatever bio material
        the stub/profile exposes (personality + top goal). INTENT
        GUIDANCE always renders so the LLM emits the tags regardless
        of how bare the stub is."""
        npc = self.engine.pie.npc_knowledge.get(focus_npc)
        if npc is None:
            return None
        identity = getattr(npc, "identity", None) or {}
        personality = str(identity.get("personality", "") or "").strip()
        goals = self._peek_npc_goals(focus_npc) or []
        top_goal_desc = ""
        if goals:
            top_goal_desc = str(goals[0].get("description", "") or "").strip()

        lines: list[str] = []
        if personality or top_goal_desc:
            lines.append("=== GIVER CONTEXT ===")
            if personality:
                lines.append(f"{focus_npc}'s personality: {personality[:180]}")
            if top_goal_desc:
                lines.append(f"{focus_npc}'s top goal: {top_goal_desc[:160]}")
            lines.append("")
        lines.extend([
            "INTENT GUIDANCE (include in the quest JSON):",
            "- Good-aligned (priests, honest craftsfolk) → intent: good or neutral",
            "- Gray characters (smugglers, fences, opportunists) → intent: gray",
            "- Harmful (assassins, thieves, zealots) → intent: dark or cruel",
            "Also include moral_weight (0.0 = harmless chore, 1.0 = cruel/morally heavy).",
        ])
        return "\n".join(lines)

    def _find_quest_by_id(self, quest_id: str):
        """Return the first Quest dataclass with the given id across
        every NPC's quest list, or None. Used by refusal and reward
        plumbing — both paths get a ``quest_id`` from the player and
        need the full dataclass to read Phase 3a fields."""
        for npc in self.engine.pie.npc_knowledge.profiles.values():
            for q in getattr(npc, "quests", []):
                if q.id == quest_id:
                    return q
        return None

    def process_refusal(self, quest_id: str, npc_id: str,
                         reason: Optional[str] = None) -> dict:
        """
        Phase 3a — apply the mechanics of a player-refused quest:
        flip status to 'refused', apply a moral-weight-scaled trust
        hit on the giver, emit a FactLedger entry, surface a
        "previously refused" context line on the giver's personal
        knowledge (future dialogue reads it), and schedule a
        re-eligibility timer if the quest is in decay mode.

        Returns a descriptive dict the REST layer passes back.
        """
        if not isinstance(quest_id, str) or not quest_id:
            return {"ok": False, "reason": "missing quest_id"}
        if not isinstance(npc_id, str) or not npc_id:
            return {"ok": False, "reason": "missing npc_id"}
        npc = self.engine.pie.npc_knowledge.get(npc_id)
        if npc is None:
            return {"ok": False, "reason": f"unknown npc '{npc_id}'"}
        quest = None
        for q in getattr(npc, "quests", []):
            if q.id == quest_id:
                quest = q
                break
        if quest is None:
            return {"ok": False, "reason": f"unknown quest '{quest_id}' on '{npc_id}'"}

        quest.status = "refused"

        # Trust delta: explicit override beats the moral-weight
        # formula. moral_weight=0 default → trust_delta=0 (no penalty).
        trust_delta = quest.refusal_trust_delta
        if trust_delta == 0 and quest.moral_weight:
            trust_delta = int(quest.moral_weight * -15)
        if trust_delta:
            try:
                self.engine.adjust_trust(
                    npc_id, int(trust_delta),
                    reason=f"player refused '{quest.name}'",
                )
            except Exception as e:
                logger.warning(f"adjust_trust failed on refusal: {e}")

        # FactLedger entry — the refusal is a world-fact the Director
        # can draw on later (gossip propagation, future beats).
        giver_name = npc.identity.get("name", npc_id)
        try:
            self.ledger.add(
                text=f"The player refused {giver_name}'s quest '{quest.name}'.",
                npc_id=npc_id,
                kind="refusal",
                tick=self.tick_count + 1,
                suggested_by="quest_refusal",
            )
        except Exception as e:
            logger.warning(f"ledger add for refusal failed: {e}")

        # Surface a dialogue-context line on the giver so future NPC
        # dialogue reads it via NPCKnowledge.build_context. Routed
        # through engine.add_knowledge so the dynamic-lane
        # reserve-min (set in densanon-core) keeps the profile
        # static lore intact.
        try:
            self.engine.add_knowledge(
                npc_id,
                f"The player has previously refused my quest '{quest.name}'.",
                fact_type="personal",
            )
        except Exception as e:
            logger.warning(f"add_knowledge for refusal context failed: {e}")

        # Decay mode schedules a re-eligibility timer. Permanent mode
        # just stays refused until the game client explicitly reopens.
        if quest.refusal_mode == "decay" and quest.refusal_decay_ticks > 0:
            unlock_tick = self.tick_count + 1 + quest.refusal_decay_ticks
            self._refused_quest_timers[(npc_id, quest_id)] = unlock_tick
        self._save_state()
        return {
            "ok": True,
            "quest_id": quest_id,
            "npc_id": npc_id,
            "trust_delta": int(trust_delta),
            "refusal_mode": quest.refusal_mode,
            "unlock_tick": self._refused_quest_timers.get((npc_id, quest_id)),
            "reason": reason,
        }

    def set_player_auto_refuse(self, intents: list[str]) -> dict:
        """Phase 3a — player sets which intents should auto-refuse.
        Ignored if the dev flag ``director.quest_auto_refuse.enabled``
        is false AND ``player_configurable`` is false. Returns the
        persisted set plus flags for client UI."""
        if not isinstance(intents, (list, tuple, set)):
            return {"ok": False, "reason": "intents must be a list of strings"}
        if (not self._quest_auto_refuse_enabled
                and not self._quest_auto_refuse_player_configurable):
            return {
                "ok": False,
                "reason": "auto-refuse is disabled in this world (director.quest_auto_refuse)",
            }
        cleaned = {
            str(v).strip() for v in intents
            if isinstance(v, str) and str(v).strip()
        }
        self._player_auto_refuse_intents = cleaned
        self._save_state()
        return {
            "ok": True,
            "intents": sorted(cleaned),
            "dev_enabled": self._quest_auto_refuse_enabled,
            "player_configurable": self._quest_auto_refuse_player_configurable,
        }

    def get_player_auto_refuse(self) -> dict:
        """Read-only view of the current auto-refuse config + state."""
        return {
            "dev_enabled": self._quest_auto_refuse_enabled,
            "player_configurable": self._quest_auto_refuse_player_configurable,
            "intents": sorted(self._player_auto_refuse_intents),
        }

    # ── Phase 5a — identity split ────────────────────────────────

    @staticmethod
    def _slugify_identity(text: str) -> str:
        """Normalize an identity string to a lowercase slug. The
        ``known_as`` list should be case-insensitive — "Jordan"
        and "jordan" are the same identity — but we also want to
        preserve readability for UI layers, so we slug rather than
        lowercase-in-place."""
        return str(text).strip().lower().replace(" ", "_")

    def _ensure_player_knowledge(self, npc) -> dict:
        """Return the NPC's player_knowledge dict, creating the
        default skeleton if the stub predates Phase 5a."""
        pk = getattr(npc, "player_knowledge", None)
        if not isinstance(pk, dict):
            pk = {
                "met": False,
                "recognized": False,
                "known_as": [],
                "witnessed_deeds": [],
                "heard_deeds": [],
                "first_met_tick": None,
                "last_interaction_tick": None,
            }
            npc.player_knowledge = pk
        return pk

    def _merge_identity_trust(self, npc_id: str, identities: list[str]) -> int:
        """Normalize the per-identity trust record for ``npc_id`` so
        every identity in ``identities`` carries the same max value
        across the set. Returns the max. Used by introduce / vouch
        to keep trust consistent after identity merges."""
        idmap = self._npc_player_identity_trust.setdefault(npc_id, {})
        for ident in identities:
            idmap.setdefault(ident, 0)
        max_trust = max((idmap[i] for i in identities), default=0)
        for ident in identities:
            idmap[ident] = max_trust
        return max_trust

    def introduce_player(self, to_npc: str, name: str,
                          titles: Optional[list[str]] = None) -> dict:
        """Phase 5a — player introduces themselves to an NPC. Flips
        met + recognized, merges ``name`` + ``titles`` into the
        NPC's ``known_as`` list (slugged, deduped), normalizes
        per-identity trust to the max across the expanded set, and
        stamps ``first_met_tick`` if this is the NPC's first
        meeting."""
        if not isinstance(to_npc, str) or not to_npc:
            return {"ok": False, "reason": "missing to_npc"}
        if not isinstance(name, str) or not name.strip():
            return {"ok": False, "reason": "missing name"}
        npc = self.engine.pie.npc_knowledge.profiles.get(to_npc)
        if npc is None:
            return {"ok": False, "reason": f"unknown npc '{to_npc}'"}
        pk = self._ensure_player_knowledge(npc)

        idents = [self._slugify_identity(name)]
        for t in titles or []:
            if isinstance(t, str) and t.strip():
                idents.append(self._slugify_identity(t))
        # Phase 5b — feature-based auto-recognition. If the player
        # has a visible feature registered to an identity, that
        # identity joins the NPC's known_as alongside the explicit
        # name. Lets a named introduction also cement the
        # reputation-identity ("Jordan, the Dragonslayer").
        if self._player_visible_feature:
            mapped = self._visible_feature_to_identity.get(
                self._player_visible_feature,
            )
            if mapped:
                idents.append(mapped)
        # Merge into known_as (preserve existing order + add new)
        existing = set(pk["known_as"])
        for ident in idents:
            if ident not in existing:
                pk["known_as"].append(ident)
                existing.add(ident)

        if not pk["met"]:
            pk["first_met_tick"] = self.tick_count + 1
        pk["met"] = True
        pk["recognized"] = True
        pk["last_interaction_tick"] = self.tick_count + 1

        max_trust = self._merge_identity_trust(to_npc, pk["known_as"])
        self._save_state()
        return {
            "ok": True,
            "npc_id": to_npc,
            "known_as": list(pk["known_as"]),
            "max_trust": max_trust,
            "first_met_tick": pk["first_met_tick"],
        }

    def vouch_player_to(self, voucher_npc: str, to_npc: str) -> dict:
        """Phase 5a — voucher_npc introduces the player to to_npc.
        to_npc inherits voucher's known_as identities (they know the
        player under the same names the voucher does). Trust
        normalizes to max across the merged set, consistent with
        direct introduction."""
        if voucher_npc == to_npc:
            return {"ok": False, "reason": "voucher and to_npc must differ"}
        voucher = self.engine.pie.npc_knowledge.profiles.get(voucher_npc)
        target = self.engine.pie.npc_knowledge.profiles.get(to_npc)
        if voucher is None:
            return {"ok": False, "reason": f"unknown voucher '{voucher_npc}'"}
        if target is None:
            return {"ok": False, "reason": f"unknown to_npc '{to_npc}'"}
        voucher_pk = self._ensure_player_knowledge(voucher)
        if not voucher_pk["known_as"]:
            return {"ok": False, "reason": "voucher has never met the player"}
        target_pk = self._ensure_player_knowledge(target)

        existing = set(target_pk["known_as"])
        for ident in voucher_pk["known_as"]:
            if ident not in existing:
                target_pk["known_as"].append(ident)
                existing.add(ident)

        if not target_pk["met"]:
            target_pk["first_met_tick"] = self.tick_count + 1
        target_pk["met"] = True
        target_pk["recognized"] = True
        target_pk["last_interaction_tick"] = self.tick_count + 1

        max_trust = self._merge_identity_trust(to_npc, target_pk["known_as"])
        self._save_state()
        return {
            "ok": True,
            "voucher_npc": voucher_npc,
            "to_npc": to_npc,
            "known_as": list(target_pk["known_as"]),
            "max_trust": max_trust,
        }

    def set_player_visible_feature(self, feature: Optional[str]) -> dict:
        """Record a player-visible feature (e.g. a distinctive cloak,
        a bloodied hand). Phase 5b will auto-recognize NPCs against
        a feature→identity registry; in 5a we just store the
        current value. Pass None or empty string to clear."""
        if feature is None or (isinstance(feature, str) and not feature.strip()):
            self._player_visible_feature = None
        elif isinstance(feature, str):
            self._player_visible_feature = feature.strip()
        else:
            return {"ok": False, "reason": "feature must be a string or null"}
        self._save_state()
        return {
            "ok": True,
            "player_visible_feature": self._player_visible_feature,
        }

    def register_visible_feature(self, feature: str, identity: str) -> dict:
        """Phase 5b — map a player-visible feature
        (``"dragonslayer_cloak"``) to an identity (``"the_dragonslayer"``).
        Used by auto-recognition: the first time an NPC meets the
        player while they're wearing the feature, that identity is
        auto-added to the NPC's ``known_as``."""
        if not isinstance(feature, str) or not feature.strip():
            return {"ok": False, "reason": "missing feature"}
        if not isinstance(identity, str) or not identity.strip():
            return {"ok": False, "reason": "missing identity"}
        feat = feature.strip()
        ident = self._slugify_identity(identity)
        self._visible_feature_to_identity[feat] = ident
        self._save_state()
        return {
            "ok": True, "feature": feat, "identity": ident,
            "registry_size": len(self._visible_feature_to_identity),
        }

    def get_player_reputation(self) -> dict:
        """
        Phase 5c — aggregate view of the player's standing across
        every identity the world knows them under. Per-identity
        roll-up of who-knows-them + what-deeds-they've-done, plus
        summary counts for the client HUD.

        Deed text comes straight from the FactLedger; reputation
        queries are read-only (no mutations on the identity-state
        dict)."""
        # Gather ledger entries per subject_identity (legacy entries
        # with no tag get bucketed under "player" for compat).
        deeds_by_identity: dict[str, list[dict]] = {}
        intent_summary: dict[str, int] = {
            "good": 0, "neutral": 0, "gray": 0, "dark": 0, "cruel": 0,
        }
        for idx, entry in enumerate(self.ledger.entries):
            if entry.get("kind") not in ("player_action", "refusal"):
                continue
            ident = entry.get("subject_identity") or "player"
            deeds_by_identity.setdefault(ident, []).append({
                "ledger_index": idx,
                "text": entry.get("text", ""),
                "tick": entry.get("tick"),
                "kind": entry.get("kind"),
            })

        # Walk NPCs for known_by per identity + recognition totals.
        known_by: dict[str, list[str]] = {}
        recognized_npcs: set[str] = set()
        heard_without_recognition: set[str] = set()
        for npc_id, npc in self.engine.pie.npc_knowledge.profiles.items():
            pk = getattr(npc, "player_knowledge", None) or {}
            if pk.get("recognized"):
                recognized_npcs.add(npc_id)
            for ident in pk.get("known_as", []) or []:
                known_by.setdefault(ident, []).append(npc_id)
            if (not pk.get("recognized")) and (
                pk.get("witnessed_deeds") or pk.get("heard_deeds")
            ):
                heard_without_recognition.add(npc_id)

        # Intent summary from quest dispatches with intent tags.
        for npc in self.engine.pie.npc_knowledge.profiles.values():
            for q in getattr(npc, "quests", []):
                if q.intent in intent_summary:
                    intent_summary[q.intent] += 1

        known_identities: dict[str, dict] = {}
        for ident in set(list(known_by.keys()) + list(deeds_by_identity.keys())):
            known_identities[ident] = {
                "known_by": sorted(known_by.get(ident, [])),
                "deeds": [d["text"] for d in deeds_by_identity.get(ident, [])],
            }
        return {
            "known_identities": known_identities,
            "total_npcs_who_recognize_you": len(recognized_npcs),
            "total_npcs_aware_of_deeds_without_recognition": len(
                heard_without_recognition
            ),
            "summary_by_intent": intent_summary,
            "player_visible_feature": self._player_visible_feature,
        }

    def build_reputation_hint_for_npc(self, npc_id: str) -> Optional[str]:
        """Phase 5c — produce a ``RUMOURS: ...`` dialogue-prompt
        block for the given NPC. Fires only when the NPC has heard
        or witnessed deeds but has NOT been introduced — the
        ``recognized`` flag gates this so introductions don't emit
        phantom "have you heard of X?" rumours. Returns None when
        there's nothing meaningful to surface."""
        npc = self.engine.pie.npc_knowledge.profiles.get(npc_id)
        if npc is None:
            return None
        pk = getattr(npc, "player_knowledge", None) or {}
        if pk.get("recognized"):
            return None
        heard = pk.get("heard_deeds", []) or []
        witnessed = pk.get("witnessed_deeds", []) or []
        all_indices = list(dict.fromkeys(witnessed + heard))  # stable-unique
        if not all_indices:
            return None
        # Group deeds by identity for the prompt summary.
        by_identity: dict[str, list[str]] = {}
        for idx in all_indices:
            if idx < 0 or idx >= len(self.ledger.entries):
                continue
            entry = self.ledger.entries[idx]
            ident = entry.get("subject_identity") or "player"
            by_identity.setdefault(ident, []).append(entry.get("text", "")[:140])
        if not by_identity:
            return None
        lines = ["=== RUMOURS ==="]
        for ident, deeds in by_identity.items():
            lines.append(
                f"You've heard tales of {ident.replace('_', ' ')} — "
                + "; ".join(deeds[:3])
                + "."
            )
        lines.append(
            "Don't assume the player is them, but you may bring up "
            "the rumour if it fits the moment."
        )
        return "\n".join(lines)

    def get_player_identity_state(self) -> dict:
        """Debug / client-sync view of per-NPC identity + trust state."""
        return {
            "player_visible_feature": self._player_visible_feature,
            "visible_feature_to_identity": dict(self._visible_feature_to_identity),
            "npc_player_identity_trust": {
                npc_id: dict(idmap)
                for npc_id, idmap in self._npc_player_identity_trust.items()
            },
            "npcs": {
                npc_id: dict(getattr(npc, "player_knowledge", {}))
                for npc_id, npc in self.engine.pie.npc_knowledge.profiles.items()
            },
        }

    def _record_main_line_reward(self, quest_id: str) -> Optional[dict]:
        """On quest completion, append the matching entry from the
        line's ``reward_track`` to ``_quest_line_state[line].rewards_earned``.
        Returns the record dict (or None if the quest isn't on a
        main-line or the reward track is missing this beat)."""
        q = self._find_quest_by_id(quest_id)
        if q is None or not q.quest_line:
            return None
        line_cfg = self._quest_lines_config.get(q.quest_line)
        if not isinstance(line_cfg, dict):
            return None
        reward_track = line_cfg.get("reward_track") or []
        if not isinstance(reward_track, list):
            return None
        beat_idx = int(q.quest_line_beat or 0)
        if not (0 <= beat_idx < len(reward_track)):
            return None
        reward = reward_track[beat_idx]
        st = self._quest_line_state.setdefault(q.quest_line, {
            "dispatched_beats": [],
            "completed_quests": [],
            "rewards_earned": [],
            "line_status": "active",
        })
        st["completed_quests"].append(quest_id)
        st["rewards_earned"].append(reward)
        # Mark the line completed if every beat is done.
        total_beats = len(line_cfg.get("beats") or [])
        if total_beats > 0 and len(st["completed_quests"]) >= total_beats:
            st["line_status"] = "completed"
        return {
            "quest_line": q.quest_line,
            "beat_index": beat_idx,
            "reward": reward,
        }

    def _effective_max_unoffered_quests(self) -> int:
        """Runtime override wins; otherwise the module-level default."""
        if self._max_unoffered_quests_override is not None:
            return self._max_unoffered_quests_override
        return _MAX_UNOFFERED_QUESTS_PER_NPC

    def _effective_quest_cooldown_ticks(self) -> int:
        """Runtime override wins; otherwise the module-level default."""
        if self._quest_cooldown_ticks_override is not None:
            return self._quest_cooldown_ticks_override
        return _NPC_QUEST_COOLDOWN_TICKS

    # ── Phase 4c — GPU coordination helpers ─────────────────────

    def _prune_tick_time_log(self) -> None:
        """Drop log entries whose timestamp is older than the rolling
        window (or beyond the absolute cap). Called before every
        budget evaluation so the window is always fresh."""
        if not self._tick_time_log:
            return
        cutoff = time.time() - _TICK_BUDGET_WINDOW_SECONDS
        kept = [(ts, dur) for ts, dur in self._tick_time_log if ts >= cutoff]
        if len(kept) > _TICK_TIME_LOG_CAP:
            kept = kept[-_TICK_TIME_LOG_CAP:]
        self._tick_time_log = kept

    def _record_tick_duration(self, duration_seconds: float) -> None:
        """Append a (now, duration) sample to the rolling log. Called
        at the end of every non-paused tick so the next call's budget
        check sees the latest LLM cost."""
        if duration_seconds <= 0:
            return
        self._tick_time_log.append((time.time(), float(duration_seconds)))
        self._prune_tick_time_log()

    def _budget_exceeded(self) -> bool:
        """True iff the trailing-window LLM time sum is at or above
        the configured budget. Returns False when the budget is
        unconstrained (-1) or unset."""
        if self._tick_budget_seconds < 0:
            return False
        self._prune_tick_time_log()
        window_sum = sum(dur for _ts, dur in self._tick_time_log)
        return window_sum >= self._tick_budget_seconds

    def _compute_next_tick_hint(self) -> int:
        """Seconds the game client is advised to wait before calling
        tick() again. Derived from player activity plus an arc-
        climax accelerator. See ``_NEXT_TICK_HINT_BY_ACTIVITY`` for
        the per-activity defaults; ``_NEXT_TICK_HINT_CONFRONT`` wins
        whenever any active arc has reached the confront beat and
        the activity hint would otherwise be above it — climactic
        arcs shouldn't be held behind a 15-minute wandering wait."""
        base = _NEXT_TICK_HINT_BY_ACTIVITY.get(
            self._player_activity, _NEXT_TICK_HINT_DEFAULT
        )
        # Arc-climax accelerator — only shortens the hint, never
        # lengthens it. Paused activities (combat=10, menu=30) keep
        # their fast poll so combat ending still picks up quickly.
        try:
            for arc in self.arc_planner.active_arcs():
                if arc.current_beat >= _ARC_CONFRONT_BEAT_INDEX:
                    return min(base, _NEXT_TICK_HINT_CONFRONT)
        except Exception:
            # Defensive — a broken arc planner must not crash a tick
            # response. Fall through to the base hint.
            pass
        return base

    def _pick_action_kind(self, focus_npc: Optional[str],
                           bypass_per_npc_cap: bool = False) -> str:
        """
        Python decides the action KIND (event / quest / fact) — round-robin
        so a single kind can't dominate a session. Skips 'quest' when the
        focus NPC already has ``_MAX_QUESTS_PER_NPC`` open quests; the LLM
        would otherwise keep piling work onto the same NPC.

        ``bypass_per_npc_cap`` is the Phase 3a main-line escape hatch:
        authored main-line beats bypass the Phase 4b unaccepted-cap +
        cooldown gate (but NOT the hard ``_MAX_QUESTS_PER_NPC`` open-quest
        limit, which is a sanity check even for authored content).
        Side-rotation callers leave it False.

        Advances ``self._kind_rotation_index`` even when a kind is skipped
        so the rotation stays predictable across ticks.
        """
        allowed = set(_ACTION_KIND_ROTATION)
        if focus_npc:
            npc = self.engine.pie.npc_knowledge.get(focus_npc)
            if npc is not None:
                open_quests = sum(
                    1 for q in npc.quests
                    if q.status in ("available", "active")
                )
                if open_quests >= _MAX_QUESTS_PER_NPC:
                    allowed.discard("quest")

                # Phase 4b — per-NPC quest accumulation gate. Two
                # sub-checks, both bypassable for main-line beats:
                #   1. Unaccepted-cap: too many available quests already
                #      waiting on this NPC's board.
                #   2. Cooldown window: a quest dispatch already landed
                #      on this NPC inside the recent N-tick window.
                if not bypass_per_npc_cap:
                    max_unoffered = self._effective_max_unoffered_quests()
                    if max_unoffered >= 0 and "quest" in allowed:
                        unoffered = sum(
                            1 for q in npc.quests if q.status == "available"
                        )
                        if unoffered >= max_unoffered:
                            logger.debug(
                                f"Phase 4b cap: dropping 'quest' for "
                                f"'{focus_npc}' (unoffered={unoffered} "
                                f">= {max_unoffered})"
                            )
                            allowed.discard("quest")

                    cooldown = self._effective_quest_cooldown_ticks()
                    last_dispatch = self._last_quest_dispatched_per_npc.get(focus_npc)
                    if (cooldown > 0
                            and last_dispatch is not None
                            and "quest" in allowed
                            and self.tick_count - last_dispatch < cooldown):
                        logger.debug(
                            f"Phase 4b cooldown: dropping 'quest' for "
                            f"'{focus_npc}' (last_tick={last_dispatch}, "
                            f"now={self.tick_count}, window={cooldown})"
                        )
                        allowed.discard("quest")

        # Zone hard filter for quests: out-of-zone NPCs can seed
        # events and facts (distant rumors propagate via gossip) but
        # cannot offer quests — a quest from an unreachable NPC wastes
        # player time and breaks the locality contract. Dead-end quests
        # from deceased givers (Phase 2a) are allowed because they're
        # narratively realistic; unreachable ones are not.
        if (focus_npc
                and self._active_zones
                and not self._npc_in_active_zone(focus_npc)):
            if "quest" in allowed:
                logger.debug(
                    f"Zone hard filter: dropping 'quest' for out-of-zone "
                    f"focus NPC '{focus_npc}' (active_zones={sorted(self._active_zones)})"
                )
                allowed.discard("quest")

        # Phase 4a — activity hard filter for quests. A quest offered
        # during dialogue, a dungeon run, or while wandering can't be
        # meaningfully accepted (giver absent or context wrong). Drop
        # the kind so rotation picks event or fact instead.
        if self._player_activity in _NO_QUEST_ACTIVITIES and "quest" in allowed:
            logger.debug(
                f"Activity hard filter: dropping 'quest' "
                f"(activity={self._player_activity})"
            )
            allowed.discard("quest")

        start = self._kind_rotation_index % len(_ACTION_KIND_ROTATION)
        for offset in range(len(_ACTION_KIND_ROTATION)):
            idx = (start + offset) % len(_ACTION_KIND_ROTATION)
            kind = _ACTION_KIND_ROTATION[idx]
            if kind in allowed:
                self._kind_rotation_index = (idx + 1) % len(_ACTION_KIND_ROTATION)
                return kind

        # Nothing allowed (shouldn't happen) — advance and default to event
        self._kind_rotation_index = (start + 1) % len(_ACTION_KIND_ROTATION)
        return "event"

    def _npc_in_active_zone(self, npc_id: str) -> bool:
        """
        True if the given NPC is currently in at least one of the
        Director's active zones. "global" zone NPCs (unset or
        explicitly marked global in profile YAML) are always in
        scope — they represent world-facing characters the player
        can interact with regardless of location.

        Returns False for unknown NPCs so a stale id can't sneak
        through a zone filter.
        """
        if not self._active_zones:
            return True  # world-wide mode, every NPC counts
        npc = self.engine.pie.npc_knowledge.profiles.get(npc_id)
        if npc is None:
            return False
        zone = getattr(npc, "current_zone", "global")
        if zone == "global":
            return True
        return zone in self._active_zones

    def _pick_focus_npc(self, extra_exclude: Optional[set[str]] = None,
                         zone_lock_in: bool = False) -> Optional[str]:
        """
        Python decides WHICH NPC this tick focuses on. The LLM decides WHAT
        happens to them. Three layers:

        1. **Player reactivity**: if the player did something targeting a
           specific NPC *since* the last tick, prioritize that NPC — the
           Director should respond to the player's moves immediately.
           This bypasses zone filtering because player-targeted actions
           are always relevant regardless of zone membership.
        2. **Zone locality (optional)**: when ``_active_zones`` is
           non-empty, partition the available NPCs into in-zone and
           out-of-zone. Pick from in-zone for (N-1) of N calls and from
           out-of-zone for 1 in N (controlled by ``_OUT_OF_ZONE_RATE``).
           The out-of-zone escape hatch seeds distant-rumor content that
           propagates via gossip. Empty ``_active_zones`` skips this
           layer entirely and falls through to layer 3 — backward-compat
           with all existing benches.
        3. **Round-robin rotation**: pick the least-recently-touched NPC
           from whichever pool layer 2 selected (or the full available
           list when ``_active_zones`` is empty). Keeps the story from
           fixating when the player is passive.

        ``extra_exclude`` is used by the architect's in-flight planner to
        prevent two workers in the same multi-action tick from competing
        for the same NPC. NPCs in this set are skipped at every layer.

        ``zone_lock_in`` (passed by ``_consume_burst_focus``) forces the
        pick to the in-zone pool even when the escape counter would
        otherwise fire. The burst rotation's K-tick hold would otherwise
        amplify a single escape-hatch pick into K ticks of out-of-zone
        focus — way more distant texture than the 1-in-N ratio specifies.
        Slots 1+ in multi-action mode still see the normal escape.

        The split exists because Qwen/Llama 3B — even with strongly-worded
        rules in the prompt — still fixate on a single target or abuse
        ``"all"``. We take the choice out of the model's hands.
        """
        # Filter out deceased NPCs at the source — they're permanently
        # out of the rotation pool. Deceased NPCs still appear in the
        # FactLedger and gossip system, but the Director doesn't
        # generate new story beats for them.
        alive = self._alive_npcs()
        profiles = list(alive.keys())
        if not profiles:
            return None

        excluded = set(extra_exclude or ())
        available = [nid for nid in profiles if nid not in excluded]
        if not available:
            return None

        # Layer 1: react to pending player action (bypasses zone filter —
        # if the player is actively engaging someone, they're relevant)
        pending_target = self._pending_player_target(available)
        if pending_target:
            return pending_target

        # Layer 2: zone locality partition. Only applies when
        # active_zones is set; otherwise the pool is "everything
        # available" and layer 3 sees the same list as before.
        if self._active_zones:
            in_zone = [
                nid for nid in available if self._npc_in_active_zone(nid)
            ]
            out_of_zone = [
                nid for nid in available if not self._npc_in_active_zone(nid)
            ]
            # One in _OUT_OF_ZONE_RATE picks goes to the out-of-zone
            # pool for distant-rumor content. Bump a counter and route
            # based on its modulo so the escape hatch fires predictably.
            # Burst rotation lock-in (zone_lock_in=True) bypasses the
            # escape: burst rotation's K-tick hold would otherwise
            # amplify a single out-of-zone pick into K ticks of distant
            # focus — already tested empirically on PB zoned (20 ticks,
            # 1 escape-hatch burst = 4 ticks of lighthouse focus, 33%
            # of the session on a 3-NPC in-zone pool).
            self._zone_escape_counter += 1
            prefer_out_of_zone = (
                not zone_lock_in
                and _OUT_OF_ZONE_RATE > 0
                and out_of_zone
                and self._zone_escape_counter % _OUT_OF_ZONE_RATE == 0
            )
            if prefer_out_of_zone:
                pool = out_of_zone
            elif in_zone:
                pool = in_zone
            elif out_of_zone:
                # No in-zone candidates left (all excluded or all out
                # of zone); fall back to out-of-zone rather than
                # returning None. The architect's plan still gets
                # filled; the narrative just drifts slightly away
                # from the player's zone this tick.
                pool = out_of_zone
            else:
                return None
        else:
            pool = available

        # Layer 2b: Phase 3a main-line weighting. When any active
        # main-line cast has members still present in the pool, apply
        # a 2:1 preference (tunable via _MAIN_LINE_WEIGHT_FACTOR):
        # FACTOR of every FACTOR+1 picks come from main-line, 1 goes
        # to the non-main-line pool so the rest of the cast still
        # breathes. Empty main-line cast = no-op. Applied AFTER zone
        # filtering so main-line givers must also satisfy the zone
        # contract (zone hard filter in _pick_action_kind still
        # prevents out-of-zone quest offers from main-line givers).
        main_line_ids = self._main_line_cast()
        if main_line_ids and _MAIN_LINE_WEIGHT_FACTOR > 0:
            ml_pool = [nid for nid in pool if nid in main_line_ids]
            non_ml_pool = [nid for nid in pool if nid not in main_line_ids]
            if ml_pool and non_ml_pool:
                self._main_line_focus_counter += 1
                divisor = _MAIN_LINE_WEIGHT_FACTOR + 1
                # Non-main pick fires once every 'divisor' calls;
                # main-line pick fires the other FACTOR out of divisor.
                prefer_non_main = (
                    self._main_line_focus_counter % divisor == 0
                )
                pool = non_ml_pool if prefer_non_main else ml_pool
            elif ml_pool:
                pool = ml_pool
            # else: no main-line in pool, fall through to Layer 3 as-is

        # Layer 3: least-recently-touched rotation within the chosen
        # pool. Read from the architect's PLANNED focus dict, not
        # recent_decisions. recent_decisions is capped at 5 ticks for
        # unrelated reasons, so on big-cast worlds it can't serve as a
        # round-robin trail — NPCs touched 6+ ticks ago would fall off
        # and be picked again. _npc_last_planned_tick records every
        # architect pick across the entire session, unbounded in entry
        # count, so rotation can walk a 500-NPC world in true
        # round-robin order.
        last_touched: dict[str, int] = {npc_id: -1 for npc_id in pool}
        for npc_id, tick_num in self._npc_last_planned_tick.items():
            if npc_id in last_touched:
                last_touched[npc_id] = max(last_touched[npc_id], tick_num)
        # Belt-and-braces: also fold in recent_decisions touches so
        # legacy callers and tests that populate decisions directly (no
        # planned-focus entries) still drive rotation. Newer planned
        # focus entries always win because they're written every tick.
        for d in self.recent_decisions:
            tick_num = d.get("tick", 0)
            actions_in_decision: list[dict] = []
            if isinstance(d.get("action"), dict):
                actions_in_decision.append(d["action"])
            for sub in d.get("sub_actions", []) or []:
                if isinstance(sub, dict) and isinstance(sub.get("action"), dict):
                    actions_in_decision.append(sub["action"])
            for act in actions_in_decision:
                for key in ("npc_id", "target"):
                    val = act.get(key)
                    if isinstance(val, str) and val in last_touched:
                        last_touched[val] = max(last_touched[val], tick_num)

        ordered = sorted(
            pool,
            key=lambda nid: (last_touched[nid], pool.index(nid)),
        )
        return ordered[0]

    def _pending_player_target(self, profiles: list[str]) -> Optional[str]:
        """
        Return the NPC id of the most recent player action whose timestamp
        is newer than ``self.last_tick_at`` (or any recorded action if the
        Director has never ticked). Only counts actions with a valid target
        that's actually a known profile.
        """
        if not self.recent_player_actions:
            return None
        cutoff = self.last_tick_at or ""
        candidates = [
            pa for pa in self.recent_player_actions
            if pa.get("target") in profiles
            and (not cutoff or str(pa.get("at", "")) > cutoff)
        ]
        if not candidates:
            return None
        # Most recent wins
        return candidates[-1].get("target")

    def _format_already_done(self) -> list[str]:
        """Render the Director's last N decisions as human-readable lines for
        an explicit 'do not repeat' block."""
        out: list[str] = []
        for d in self.recent_decisions[-5:]:
            act = d.get("action", {})
            kind = act.get("action", "?")
            if kind == "quest":
                quest = act.get("quest") or {}
                name = quest.get("name") or quest.get("id") or "?"
                out.append(f"{kind} / {act.get('npc_id', '?')} / \"{name}\"")
            elif kind == "event":
                target = act.get("target") or "all"
                text = (act.get("event") or act.get("description") or "")[:80]
                out.append(f"{kind} / {target} / \"{text}\"")
            elif kind == "fact":
                text = (act.get("fact") or "")[:80]
                out.append(f"{kind} / {act.get('npc_id', '?')} / \"{text}\"")
            else:
                out.append(kind)
        return out

    def _build_player_block(self) -> list[str]:
        """
        Render the player's current state as a list of snapshot lines.
        Empty list if there's nothing interesting to say — then the
        caller should skip the PLAYER header entirely.
        """
        pie = self.engine.pie
        lines: list[str] = []

        pq = getattr(pie, "player_quests", None)
        if pq is not None:
            active_names = [q.get("name", q.get("id", "?"))
                            for q in getattr(pq, "active_quests", [])]
            done_names = [q.get("name", q.get("id", "?"))
                          for q in getattr(pq, "completed_quests", [])]
            if active_names:
                lines.append(f"active quests: {', '.join(active_names)}")
            if done_names:
                lines.append(f"completed: {', '.join(done_names[-5:])}")

        # Per-NPC trust (only NPCs with a capability manager — the rest
        # just have defaults and would pad the snapshot uselessly)
        trust_bits: list[str] = []
        for npc_id in pie.npc_knowledge.profiles:
            _, trust = self._peek_npc_state(npc_id)
            if trust is not None:
                trust_bits.append(f"{npc_id}={trust}")
        if trust_bits:
            lines.append("trust with NPCs: " + ", ".join(trust_bits))

        if self.recent_player_actions:
            recent = []
            for pa in self.recent_player_actions[-5:]:
                txt = pa.get("text", "")
                tgt = pa.get("target")
                prefix = f"[{tgt}] " if tgt else ""
                recent.append(prefix + txt)
            lines.append("recent player actions:")
            for r in recent:
                lines.append("  - " + r)

        return lines

    def _peek_npc_state(self, npc_id: str) -> tuple[Optional[str], Optional[int]]:
        """Non-destructive read of mood + trust. Does not lazy-create a manager."""
        mgr = self.engine.pie.capability_managers.get(npc_id)
        if mgr is None:
            return None, None
        mood = None
        trust = None
        emo = mgr.capabilities.get("emotional_state")
        if emo is not None:
            mood = getattr(emo, "mood", None)
        trust_cap = mgr.capabilities.get("trust")
        if trust_cap is not None:
            trust = getattr(trust_cap, "level", None)
        return mood, trust

    @staticmethod
    def _bio_item_key(text: str) -> str:
        """Stable dict key for a bio item — lowercased, whitespace-normalized,
        length-bounded so tiny phrasing drift doesn't create duplicates."""
        import re
        return re.sub(r"\s+", " ", text.strip().lower())[:200]

    @staticmethod
    def _bio_content_words(text: str) -> list[str]:
        """Extract content words (>3 chars, not stopwords) from a bio
        item for the mention-overlap heuristic."""
        import re
        return [
            w for w in (m.lower() for m in re.findall(r"[A-Za-z']+", text))
            if len(w) > 3 and w not in _BIO_STOPWORDS
        ]

    def _is_bio_mentioned(self, bio_item: str, output_lower: str) -> bool:
        """
        Heuristic: treat a bio item as "mentioned" when at least 2 of
        its content words appear in the output AND those hits cover
        at least 40% of the bio item's content words. The 2-hit floor
        catches short-but-distinctive bio items (e.g. *"Serves the
        best stew"* → "best stew" alone is a clone signal), while the
        40% ratio suppresses false positives from incidental one-word
        overlap on longer items.
        """
        if not bio_item or not output_lower:
            return False
        words = self._bio_content_words(bio_item)
        if len(words) < 2:
            return False
        hits = sum(1 for w in words if w in output_lower)
        # Two absolute hits + 40% ratio. The ratio matters more on
        # long items (7-word bio needs 3 hits to qualify); the absolute
        # floor matters on short items (3-word bio needs 2 hits).
        return hits >= 2 and hits >= max(2, int(round(len(words) * 0.4)))

    def _collect_bio_items_live(self, npc_id: str) -> list[tuple[str, str]]:
        """
        Read the NPC's current bio items straight from the live
        NPCKnowledge state. Used ONCE at director init to snapshot the
        original character data. Not called from the hot path — use
        ``_original_bios[npc_id]`` everywhere else.
        """
        npc = self.engine.pie.npc_knowledge.get(npc_id)
        if npc is None:
            return []
        out: list[tuple[str, str]] = []
        for g in self._peek_npc_goals(npc_id):
            desc = str(g.get("description", "")).strip()
            if desc:
                out.append(("goal", desc))
        for pk in (getattr(npc, "personal_knowledge", None) or []):
            t = str(pk).strip()
            if t:
                out.append(("pk", t))
        for wf in (getattr(npc, "world_facts", None) or []):
            t = str(wf).strip()
            if t:
                out.append(("wf", t))
        return out

    def _snapshot_original_bios(self) -> None:
        """
        Cache every NPC's current bio items as the 'original' set. Used
        by _build_focus_npc_bio and _record_bio_mentions so later
        Director-generated facts (which get appended to NPCKnowledge by
        the dispatch layer) don't contaminate bio tracking.
        """
        try:
            profiles = self.engine.pie.npc_knowledge.profiles
        except AttributeError:
            profiles = {}
        for npc_id in profiles:
            self._original_bios[npc_id] = self._collect_bio_items_live(npc_id)

    def _record_bio_mentions(self, focus_npc: Optional[str],
                              output_text: Optional[str]) -> None:
        """
        After a worker dispatches content, scan the output text for
        matches against the focus NPC's ORIGINAL bio items (YAML only,
        not Director-appended facts) and bump mention counts. Drives
        intra-bio rotation: the next tick that focuses on this NPC
        will see less of whatever the model just quoted.
        """
        if not focus_npc or not output_text:
            return
        items = self._original_bios.get(focus_npc, [])
        if not items:
            return
        output_lower = output_text.lower()
        counts = self._bio_mention_counts.setdefault(focus_npc, {})
        for _kind, text in items:
            if self._is_bio_mentioned(text, output_lower):
                key = self._bio_item_key(text)
                counts[key] = counts.get(key, 0) + 1

    def _peek_npc_goals(self, npc_id: str) -> list[dict]:
        """
        Non-destructive read of an NPC's goals list from the capability
        manager. Returns priority-sorted dicts (as stored by GoalsCapability)
        or an empty list if the NPC has no goals capability configured.
        """
        mgr = self.engine.pie.capability_managers.get(npc_id)
        if mgr is None:
            return []
        goals_cap = mgr.capabilities.get("goals")
        if goals_cap is None:
            return []
        raw = getattr(goals_cap, "goals", None)
        if not isinstance(raw, list):
            return []
        return list(raw)

    def _build_focus_npc_bio(self, focus_npc: str) -> Optional[str]:
        """
        Compact full-bio block for the focus NPC the current worker is
        writing for. Gives the Director motivations, secrets, and
        personality to work with instead of just (role, mood, trust).

        Rotation rules:

        - Items are sourced from ``_original_bios`` (YAML-only), NOT
          from the live NPCKnowledge. This prevents Director-generated
          facts (which the dispatch layer appends to
          ``NPCKnowledge.personal_knowledge``) from polluting bio
          tracking.
        - Items with mention count >= ``_BIO_COOLDOWN_THRESHOLD`` are
          excluded from the section so the model literally cannot see
          them until other items catch up. If exclusion leaves a
          section empty, we fall back to showing the least-mentioned
          items so the bio never goes blank for a section that exists.
        - Remaining items are sorted by mention count ascending (with
          priority / list-order as tiebreaker) and truncated to the
          top-N caps below.

        Top-N caps (tight on purpose — forces rotation to actually
        hide items on small bios):

        - Top 2 goals (was 3)
        - Top 3 personal_knowledge items (was 4)
        - Top 2 world_facts (was 3)

        Returns None when the NPC has no original bio data — callers
        should skip the block entirely in that case.
        """
        npc = self.engine.pie.npc_knowledge.get(focus_npc)
        if npc is None:
            return None

        originals = self._original_bios.get(focus_npc, [])
        if not originals:
            return None

        counts = self._bio_mention_counts.get(focus_npc, {})

        def mention_count(text: str) -> int:
            return counts.get(self._bio_item_key(text), 0)

        def apply_cooldown(
            indexed: list[tuple[int, str]],
        ) -> list[tuple[int, str]]:
            """Drop items at or above the cooldown threshold. Fall
            back to the least-mentioned items if exclusion would leave
            the section empty."""
            fresh = [
                pair for pair in indexed
                if mention_count(pair[1]) < _BIO_COOLDOWN_THRESHOLD
            ]
            if fresh:
                return fresh
            # Everything's been over-mentioned — show the freshest
            # anyway so the block isn't empty
            return sorted(indexed, key=lambda p: mention_count(p[1]))

        lines: list[str] = []

        identity = getattr(npc, "identity", None) or {}
        personality = identity.get("personality")
        if personality:
            lines.append(f"Personality: {str(personality)[:180]}")

        # Goals from the original snapshot. Need the priority alongside
        # the description, so we pull goals fresh via _peek_npc_goals
        # (goals are static — they don't mutate like pk does — so
        # using the live goals cap is safe).
        goals = self._peek_npc_goals(focus_npc)
        if goals:
            indexed = [
                (i, str(g.get("description", "")).strip())
                for i, g in enumerate(goals)
                if str(g.get("description", "")).strip()
            ]
            indexed = apply_cooldown(indexed)
            # Priority lookup: build a {description: goal} map
            goal_by_desc = {str(g.get("description", "")).strip(): g for g in goals}
            indexed.sort(key=lambda pair: (
                mention_count(pair[1]),
                -int(goal_by_desc.get(pair[1], {}).get("priority", 0) or 0),
                pair[0],
            ))
            goal_lines: list[str] = []
            for _idx, desc in indexed[:2]:
                g = goal_by_desc.get(desc, {})
                prio = g.get("priority", "?")
                goal_lines.append(f"  [p{prio}] {desc[:140]}")
            if goal_lines:
                lines.append("Driving goals:")
                lines.extend(goal_lines)

        # Personal knowledge — pulled from ORIGINALS, not live state
        pk_items = [text for kind, text in originals if kind == "pk"]
        if pk_items:
            pk_indexed = [(i, t) for i, t in enumerate(pk_items)]
            pk_indexed = apply_cooldown(pk_indexed)
            pk_indexed.sort(key=lambda pair: (mention_count(pair[1]), pair[0]))
            lines.append("Private knowledge (build AROUND these, do not state literally):")
            for _idx, fact in pk_indexed[:3]:
                lines.append(f"  - {fact[:160]}")

        # World facts — same rotation pattern, from ORIGINALS
        wf_items = [text for kind, text in originals if kind == "wf"]
        if wf_items:
            wf_indexed = [(i, t) for i, t in enumerate(wf_items)]
            wf_indexed = apply_cooldown(wf_indexed)
            wf_indexed.sort(key=lambda pair: (mention_count(pair[1]), pair[0]))
            lines.append("Their view of the world:")
            for _idx, fact in wf_indexed[:2]:
                lines.append(f"  - {fact[:160]}")

        if not lines:
            return None

        # Paraphrase instruction up top. Verbatim phrasing clone was a
        # measurable failure mode on 3B (Bess's "merchant guild
        # planning to raise taxes" appeared near-verbatim across 5/8
        # ticks). This line tells the model to use these as raw
        # material, not as a script.
        header = (
            f"=== FOCUS NPC BIO: {focus_npc} ===\n"
            "(Use these as raw material — PARAPHRASE in your own "
            "words, do not quote verbatim.)\n"
        )
        return header + "\n".join(lines)

    # ── Prompt assembly ─────────────────────────────────────────

    def _build_prompt(self, snapshot: str, focus_npc: Optional[str],
                      action_kind: Optional[str] = None) -> str:
        parts: list[str] = []
        parts.append(
            "You are the Story Director for a fantasy village game. "
            "You watch the world from above and decide what happens next. "
            "Each tick you choose ONE action that moves the story forward. "
            "If the CURRENT WORLD STATE lists an ALREADY DONE block, you must "
            "pick something DIFFERENT — do not repeat actions you have already "
            "taken, and do not reuse the same text or targets."
        )
        if self.narration_mode == "terse":
            # Game-ready mode: downstream NPC dialogue will cite these
            # outputs, so keep them short, factual, and free of novel
            # narration. The content field of each action should be a
            # single third-person statement under 25 words — no
            # internal monologue, no quoted dialogue, no action
            # choreography, no adverbs. The model loves to narrate;
            # we're actively fighting that tendency here. Placed
            # right after the system directive so it's read before
            # the schema and focus blocks.
            parts.append(
                "=== OUTPUT STYLE ===\n"
                "Write the content field (event/fact/quest description) "
                "as a SINGLE third-person statement under 25 words. "
                "NO internal monologue. NO quoted dialogue in the "
                "content. NO flowery prose, action choreography, or "
                "adverbs like 'suddenly', 'quickly', 'nervously'. Just "
                "the factual beat. Good examples:\n"
                '  "Mara moved her inventory to the cellar last night."\n'
                '  "Noah found Elena\'s old letter in the rose garden."\n'
                '  "Roderick spotted strange lights near the north gate."\n'
                "Bad example (too long, too prosaic):\n"
                '  "Mara, hiding a furtive look in her eyes, suddenly '
                "dropped a tray of hot soup, spilling it on her leg, "
                'muttering a curse under her breath..."'
            )
        if self._lore_text:
            parts.append("=== SETTING ===\n" + self._lore_text)

        parts.append(
            "=== ACTION SCHEMA ===\n"
            "Respond with a single JSON object and nothing else. "
            "Allowed actions:\n"
            '  {"action": "quest", "reason": "...", "npc_id": "...", '
            '"quest": {"id": "...", "name": "...", "description": "...", '
            '"reward": "...", "objectives": ["..."]}}\n'
            '  {"action": "event", "reason": "...", '
            '"target": "all" | "<npc_id>", "event": "..."}\n'
            '  {"action": "fact", "reason": "...", "npc_id": "...", '
            '"fact": "...", "fact_type": "world" | "personal"}\n'
            "If nothing should happen yet, reply "
            '{"action": "noop", "reason": "..."}.'
        )

        picked_examples = self._pick_examples(focus_npc, action_kind)
        if picked_examples:
            ex_blocks = []
            for ex in picked_examples:
                ws = str(ex.get("world_state", "")).strip()
                action = ex.get("action", {})
                ex_blocks.append(
                    "WORLD STATE:\n" + ws + "\nACTION:\n" + json.dumps(action, ensure_ascii=False)
                )
            parts.append("=== EXAMPLES ===\n\n".join([""] + ex_blocks).strip())

        parts.append("=== CURRENT WORLD STATE ===\n" + snapshot)

        # Active narrative arc — soft guidance about the theme and current
        # beat. Python still forces focus + kind below; the arc block just
        # tells the LLM *what kind of beat* to write. Placed before FOCUS
        # NPC so the forced-focus block still holds the recency-bias slot.
        #
        # With multi-arc support, we inject the ONE arc whose cast
        # contains the current focus NPC (via ``arc_for_focus``). If the
        # worker's focus NPC isn't in any active cast, no arc block is
        # shown — the worker just writes to the forced focus without a
        # narrative theme. This keeps prompt size bounded as more arcs
        # run in parallel.
        active_arc = self.arc_planner.arc_for_focus(focus_npc)
        if active_arc is not None and active_arc.current_beat_goal:
            arc_lines = [
                "=== ACTIVE NARRATIVE ARC ===",
                f"Theme: {active_arc.theme}",
                f"Cast: {', '.join(active_arc.focus_npcs)}",
                f"Current beat ({active_arc.current_beat + 1}/{len(active_arc.beat_goals)}): "
                f"{active_arc.current_beat_goal}",
                "This tick's focus NPC is in the cast. Advance the beat.",
            ]
            parts.append("\n".join(arc_lines))

        # Focus NPC bio — the FULL motivational picture for whoever
        # this worker is about to write for. Includes personality,
        # priority-sorted goals, and personal_knowledge so the model
        # can write beats grounded in what the NPC actually wants and
        # knows — not just their role label.
        if focus_npc:
            bio_block = self._build_focus_npc_bio(focus_npc)
            if bio_block:
                parts.append(bio_block)

        # Forced focus NPC + action kind — Python made both choices.
        # Placed immediately before ACTION: so recency bias favors them
        # over the schema defaults and few-shot examples.
        if focus_npc:
            focus_lines = [
                "=== FOCUS NPC FOR THIS TICK ===",
                f"You MUST make this tick be about {focus_npc}.",
                f"The story beat should involve {focus_npc} directly.",
            ]
            if action_kind == "quest":
                focus_lines.append(
                    f'Your action field MUST be "quest" and npc_id MUST be "{focus_npc}". '
                    f'Give {focus_npc} a new quest appropriate to their role.'
                )
                # Phase 3a — inject giver context + intent guidance so
                # the LLM tags the quest's moral shape. Author-tagged
                # YAML quests bypass this path entirely (they land via
                # the profile loader, not _dispatch_quest's LLM input).
                giver_block = self._build_intent_guidance_block(focus_npc)
                if giver_block:
                    focus_lines.append("")
                    focus_lines.append(giver_block)
            elif action_kind == "fact":
                focus_lines.append(
                    f'Your action field MUST be "fact" and npc_id MUST be "{focus_npc}". '
                    f'Add a piece of knowledge {focus_npc} has learned.'
                )
            else:
                focus_lines.append(
                    f'Your action field MUST be "event" and target MUST be "{focus_npc}" '
                    f'(do NOT use "all" or any other NPC id). '
                    f'Describe something that happens to {focus_npc}.'
                )
            parts.append("\n".join(focus_lines))

        parts.append("ACTION:")
        return "\n\n".join(parts)

    # ── LLM call ────────────────────────────────────────────────

    def _llm_call(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Raw base-model call — bypasses the NPC dialogue pipeline."""
        pie = self.engine.pie
        base_model = getattr(pie, "base_model", None)
        if base_model is not None and hasattr(base_model, "generate"):
            try:
                return base_model.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["\n\n\n", "WORLD STATE:", "=== "],
                )
            except Exception as e:
                logger.error(f"Story Director base_model.generate failed: {e}")
        # Defensive fallback: PIE process() applies NPC context, not ideal but
        # lets us fail loud rather than silent.
        logger.warning("Story Director falling back to pie.process() — NPC context may leak in")
        return pie.process(prompt)

    def _llm_call_with_repair(self, prompt: str, max_tokens: int,
                              temperature: float) -> tuple[str, dict]:
        """
        LLM call with a single JSON-repair retry. If the first response
        can't be parsed into a valid action, append a repair preamble and
        try once more. Returns the final (raw, action) pair.
        """
        raw = self._llm_call(prompt, max_tokens, temperature)
        action = self._parse_action(raw)

        parse_failed = (
            action.get("action") == "noop"
            and isinstance(action.get("reason"), str)
            and (
                action["reason"].startswith("parse_error")
                or action["reason"] in ("no_json_found", "missing_action_field")
            )
        )
        if not parse_failed:
            return raw, action

        logger.info("Story Director JSON parse failed — retrying with short repair nudge")
        # Keep the repair nudge ONE short line: the 0.5B echoes longer
        # preambles as prose, which makes things worse. A terminal "ACTION:"
        # anchor is the strongest signal for the model to resume output.
        repair_prompt = prompt + "\n(respond with JSON only)\nACTION:"
        raw2 = self._llm_call(repair_prompt, max_tokens, temperature)
        action2 = self._parse_action(raw2)
        return raw2, action2

    # ── Response parsing ────────────────────────────────────────

    def _parse_action(self, raw: str) -> dict:
        """Tolerant JSON extraction. Returns a noop on parse failure."""
        if not raw:
            return {"action": "noop", "reason": "empty_response"}

        # Strip common code fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # Drop a leading "json" tag if present
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()

        candidate = _extract_first_json_object(cleaned)
        if candidate is None:
            return {"action": "noop", "reason": "no_json_found", "raw": raw[:200]}

        try:
            data = json.loads(candidate)
        except Exception as e:
            return {"action": "noop", "reason": f"parse_error: {e}", "raw": candidate[:200]}

        if not isinstance(data, dict) or "action" not in data:
            # Small models sometimes omit the label entirely but still produce
            # the shape of an action — try to infer it from the fields.
            data = data if isinstance(data, dict) else {}
            data["action"] = data.get("action", "")
        return self._coerce_action(data)

    def _coerce_action(self, data: dict) -> dict:
        """
        Small models often emit a field shape that disagrees with its
        ``action`` label — e.g., ``"action": "event"`` alongside a
        ``fact`` field and ``npc_id``. When the label doesn't match its
        required fields, infer the correct label from the fields present.

        This is the most common 0.5B failure mode: the model copies the
        schema structure but mislabels what it copied. Coercing is cheaper
        and more reliable than asking the model to try again.
        """
        action = data.get("action")

        # If the current label already matches its required fields, keep it.
        if action == "quest" and isinstance(data.get("quest"), dict) and data.get("npc_id"):
            return data
        if action == "event" and (data.get("event") or data.get("description")):
            return data
        if action == "fact" and data.get("fact") and data.get("npc_id"):
            return data
        if action == "noop":
            # Strip hallucinated schema fields tacked onto noops. The 0.5B
            # likes to emit ``{"action": "noop", "quest": {...}, "npc_id": ...}``
            # with dummy sub-fields; keep only the action + reason.
            return {"action": "noop", "reason": data.get("reason", "")}

        # Otherwise try to infer from fields, in order of specificity.
        if isinstance(data.get("quest"), dict) and data.get("npc_id"):
            logger.info(f"Story Director coerced action '{action}' -> 'quest'")
            data["action"] = "quest"
        elif data.get("fact") and data.get("npc_id"):
            logger.info(f"Story Director coerced action '{action}' -> 'fact'")
            data["action"] = "fact"
        elif data.get("event") or data.get("description"):
            logger.info(f"Story Director coerced action '{action}' -> 'event'")
            data["action"] = "event"
        else:
            data["action"] = "noop"
            data.setdefault("reason", "coerce_failed_no_matching_fields")

        return data

    def _ledger_text_for(self, action: dict) -> Optional[str]:
        """
        Extract the canonical 'fact text' for an action so the FactLedger
        can embed it. Each kind has a different shape — pick the most
        substantive description.
        """
        if not isinstance(action, dict):
            return None
        kind = action.get("action")
        if kind == "fact":
            return action.get("fact")
        if kind == "event":
            return action.get("event") or action.get("description")
        if kind == "quest":
            quest = action.get("quest") or {}
            # description is the meaty part; fall back to name
            return quest.get("description") or quest.get("name")
        return None

    def _enforce_action_kind(self, action: dict, target_kind: str,
                              focus_npc: Optional[str]) -> dict:
        """
        Force the action's kind to match what Python decided. Salvages
        the model's content when it picked the wrong kind — e.g., if we
        asked for a quest but the model emitted an event, synthesize a
        minimal quest wrapping the event's text. Noops pass through
        unchanged (a legit "nothing to do" response).
        """
        if not isinstance(action, dict):
            return action
        current = action.get("action")
        if current == target_kind or current == "noop":
            return action

        reason = action.get("reason", f"kind override from {current} to {target_kind}")
        npc_id = focus_npc or action.get("npc_id") or action.get("target")

        if target_kind == "event":
            text = (
                action.get("event")
                or action.get("fact")
                or (action.get("quest") or {}).get("description")
                or (action.get("quest") or {}).get("name")
                or "Something happens"
            )
            return {
                "action": "event",
                "reason": reason,
                "target": npc_id or "all",
                "event": str(text)[:240],
            }

        if target_kind == "quest":
            quest_data = action.get("quest")
            if not isinstance(quest_data, dict):
                text = (
                    action.get("event")
                    or action.get("fact")
                    or "A task needs doing"
                )
                quest_data = {
                    "id": f"gen_{int(time.time())}",
                    "name": str(text)[:60],
                    "description": str(text)[:240],
                    "reward": "",
                    "objectives": [str(text)[:120]],
                }
            return {
                "action": "quest",
                "reason": reason,
                "npc_id": npc_id,
                "quest": quest_data,
            }

        if target_kind == "fact":
            text = (
                action.get("fact")
                or action.get("event")
                or (action.get("quest") or {}).get("description")
                or "A fact was learned"
            )
            return {
                "action": "fact",
                "reason": reason,
                "npc_id": npc_id,
                "fact": str(text)[:240],
                "fact_type": action.get("fact_type", "world"),
            }

        return action

    def _enforce_focus_npc(self, action: dict, focus_npc: str) -> dict:
        """
        Force the action to target the Python-chosen focus NPC. If the
        model picked someone else (or 'all'), override it. Noops are left
        alone — they're a legitimate "nothing to do" output.
        """
        if not isinstance(action, dict):
            return action
        kind = action.get("action")
        if kind == "noop":
            return action
        if kind == "event":
            if action.get("target") != focus_npc:
                action["target"] = focus_npc
        elif kind in ("quest", "fact"):
            if action.get("npc_id") != focus_npc:
                action["npc_id"] = focus_npc
        return action

    # ── Action dispatch ─────────────────────────────────────────

    def _inject_tagged_event(self, description: str, npc_id: Optional[str]) -> None:
        """
        Inject an event tagged with source='director' so it won't show up
        in the Director's own world snapshot on subsequent ticks. Bypasses
        engine.inject_event (which hardcodes source='world') and writes
        directly to NPCKnowledge.
        """
        profiles = self.engine.pie.npc_knowledge.profiles
        if npc_id:
            npc = profiles.get(npc_id)
            if npc is not None:
                npc.inject_event(description, source="director")
        else:
            for npc in profiles.values():
                npc.inject_event(description, source="director")

    def _dispatch(self, action: dict) -> dict:
        kind = action.get("action")
        try:
            if kind == "quest":
                result = self._dispatch_quest(action)
            elif kind == "event":
                result = self._dispatch_event(action)
            elif kind == "fact":
                result = self._dispatch_fact(action)
            elif kind == "noop":
                result = {"ok": True, "kind": "noop"}
            else:
                result = {"ok": False, "reason": f"unknown_action_kind: {kind}"}
        except Exception as e:
            logger.exception("Story Director dispatch failed")
            result = {"ok": False, "reason": f"dispatch_error: {e}"}

        # Passive Narrative Judge observer — no-op unless env-enabled.
        # Skipped for noop and dispatch_error since there's no fact to score.
        if kind in ("quest", "event", "fact"):
            try:
                self._judge_observe(action, result)
            except Exception as e:
                logger.warning(f"_judge_observe failed (non-fatal): {e}")

        return result

    def _dispatch_quest(self, action: dict) -> dict:
        npc_id = action.get("npc_id")
        quest_data = action.get("quest") or {}
        if not npc_id or not isinstance(quest_data, dict):
            return {"ok": False, "reason": "quest_action_missing_fields"}

        npc = self.engine.pie.npc_knowledge.get(npc_id)
        if npc is None:
            return {"ok": False, "reason": f"unknown_npc: {npc_id}"}

        quest_id = str(quest_data.get("id") or f"gen_{int(time.time())}")
        # Deduplicate — refuse to re-add a quest with the same id
        if any(q.id == quest_id for q in npc.quests):
            return {"ok": False, "reason": f"quest_already_exists: {quest_id}"}

        objectives = quest_data.get("objectives") or []
        if isinstance(objectives, str):
            objectives = [objectives]

        # Phase 3a — pull the optional main-line / intent / refusal
        # fields off the LLM's quest dict. Author-tagged YAML paths
        # (NPC profile active_quests) already use the Phase 3a loader
        # and don't route through here. Missing keys fall through to
        # the Quest dataclass defaults.
        prereqs = list(quest_data.get("prerequisite_quests", []) or [])
        quest = Quest(
            id=quest_id,
            name=str(quest_data.get("name") or quest_id),
            description=str(quest_data.get("description") or ""),
            status="available",
            reward=str(quest_data.get("reward") or ""),
            objectives=[str(o) for o in objectives],
            quest_line=quest_data.get("quest_line"),
            quest_line_beat=int(quest_data.get("quest_line_beat", 0) or 0),
            prerequisite_quests=prereqs,
            intent=quest_data.get("intent"),
            moral_weight=float(quest_data.get("moral_weight", 0.0) or 0.0),
            refusal_trust_delta=int(quest_data.get("refusal_trust_delta", 0) or 0),
            refusal_mode=str(quest_data.get("refusal_mode", "permanent") or "permanent"),
            refusal_decay_ticks=int(quest_data.get("refusal_decay_ticks", 0) or 0),
        )

        # Phase 3a — sequential gating. If prerequisites are declared
        # and any of them are not yet completed anywhere in the cast,
        # the quest starts in 'locked' status. The game client never
        # shows locked quests as offers; a later lifecycle scan (or
        # explicit completion of the last prereq) flips them back to
        # 'available' via _unlock_quests_if_prereqs_met.
        if prereqs and not self._prereqs_satisfied(prereqs):
            quest.status = "locked"

        npc.add_quest(quest)

        # Phase 3a — auto-refuse. If the world's dev flag is on AND
        # the player has added this quest's intent to their filter,
        # short-circuit the offer by running the refusal pipeline
        # immediately instead of announcing it. Only fires for
        # status=='available' quests (locked beats wait for prereqs).
        auto_refused = False
        if (quest.status == "available"
                and quest.intent
                and self._quest_auto_refuse_enabled
                and quest.intent in self._player_auto_refuse_intents):
            self.process_refusal(
                quest_id=quest_id, npc_id=npc_id,
                reason="auto_refused_by_player_filter",
            )
            auto_refused = True

        # Announce the new quest only when it's actually offerable
        # AND not auto-refused. Locked main-line beats stay silent
        # until their prereq completes; auto-refused quests already
        # emitted a refusal ledger entry.
        if quest.status == "available" and not auto_refused:
            self._inject_tagged_event(
                f"{npc.identity.get('name', npc_id)} has new work to offer: {quest.name}",
                npc_id=None,
            )

        # Phase 3a — record dispatched beat on the line state so the
        # integration bench can verify beat ordering.
        if quest.quest_line and quest.quest_line in self._quest_line_state:
            line_st = self._quest_line_state[quest.quest_line]
            line_st["dispatched_beats"].append({
                "quest_id": quest_id,
                "beat_index": quest.quest_line_beat,
                "giver": npc_id,
                "at_tick": self.tick_count + 1,
                "status": quest.status,
            })

        # Phase 4b — stamp the dispatch tick so _pick_action_kind's
        # cooldown gate sees it on the next rotation. tick_count has
        # not yet been incremented for this tick at dispatch time
        # (_run_single_action runs before tick_count += 1), so we use
        # the next tick number to keep the cooldown accounting
        # consistent with _npc_last_planned_tick.
        self._last_quest_dispatched_per_npc[npc_id] = self.tick_count + 1
        return {
            "ok": True, "kind": "quest", "npc_id": npc_id,
            "quest_id": quest_id, "status": quest.status,
            "auto_refused": auto_refused,
        }

    def _prereqs_satisfied(self, prereq_ids: list[str]) -> bool:
        """True iff every prereq quest id has status='completed' on
        some NPC (the player-quest tracker is authoritative, but for
        Director-generated content we fold in the per-NPC quest
        status as a fallback for tests/benches that don't wire up the
        full engine). Unknown ids count as unsatisfied."""
        if not prereq_ids:
            return True
        completed: set[str] = set()
        player_quests = getattr(self.engine.pie, "player_quests", None)
        if player_quests is not None:
            for entry in getattr(player_quests, "completed_quests", []) or []:
                qid = entry.get("id") if isinstance(entry, dict) else None
                if isinstance(qid, str):
                    completed.add(qid)
        for npc in self.engine.pie.npc_knowledge.profiles.values():
            for q in getattr(npc, "quests", []):
                if q.status == "completed":
                    completed.add(q.id)
        return all(pid in completed for pid in prereq_ids)

    def _unlock_quests_if_prereqs_met(self) -> list[dict]:
        """Scan every NPC's quest list; any ``locked`` quest whose
        prereqs are now completed flips to ``available`` and gets a
        re-surface event. Returns a list of unlock records for the
        caller (lifecycle tick) to include in its tick report."""
        unlocked: list[dict] = []
        for npc_id, npc in self.engine.pie.npc_knowledge.profiles.items():
            for q in getattr(npc, "quests", []):
                if q.status != "locked":
                    continue
                if self._prereqs_satisfied(q.prerequisite_quests):
                    q.status = "available"
                    self._inject_tagged_event(
                        f"{npc.identity.get('name', npc_id)} has new work to offer: {q.name}",
                        npc_id=None,
                    )
                    unlocked.append({
                        "quest_id": q.id, "npc_id": npc_id,
                        "quest_line": q.quest_line,
                    })
        return unlocked

    def _dispatch_event(self, action: dict) -> dict:
        event_text = action.get("event") or action.get("description")
        if not event_text:
            return {"ok": False, "reason": "event_action_missing_text"}
        target = action.get("target")
        npc_id: Optional[str]
        if target in (None, "all", "", "*"):
            npc_id = None
        else:
            npc_id = str(target)
            if npc_id not in self.engine.pie.npc_knowledge.profiles:
                return {"ok": False, "reason": f"unknown_target: {npc_id}"}
        self._inject_tagged_event(str(event_text), npc_id)
        return {"ok": True, "kind": "event", "target": npc_id or "all"}

    def _dispatch_fact(self, action: dict) -> dict:
        npc_id = action.get("npc_id")
        fact = action.get("fact")
        if not npc_id or not fact:
            return {"ok": False, "reason": "fact_action_missing_fields"}
        fact_type = action.get("fact_type", "world")
        if fact_type not in ("world", "personal"):
            fact_type = "world"
        result = self.engine.add_knowledge(str(npc_id), str(fact), fact_type)
        if "error" in result:
            return {"ok": False, "reason": result["error"]}
        return {"ok": True, "kind": "fact", "npc_id": npc_id, "fact_type": fact_type}
