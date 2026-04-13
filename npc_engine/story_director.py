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
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import yaml

from npc_engine.bridge import NPC_ENGINE_ROOT
from npc_engine.knowledge import Quest

if TYPE_CHECKING:
    from npc_engine.engine import NPCEngine

logger = logging.getLogger("NPCEngine.story_director")


DATA_DIR = NPC_ENGINE_ROOT / "data" / "story_director"
LORE_FILE = DATA_DIR / "ashenvale_lore.md"
EXAMPLES_FILE = DATA_DIR / "examples.yaml"
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
        model isn't available.
        """
        model = self.model
        if model is None or not premise or not hypothesis:
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
        return {
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

    def check(self, text: str) -> Optional[dict]:
        """
        Compute a similarity + NLI warning for ``text`` against the
        existing ledger, WITHOUT storing it. Used by the Director's
        pre-dispatch retry path: if a candidate action would conflict
        with prior content, retry before mutating the world.

        Returns a warning dict (with ``nli`` block and ``contradiction``
        flag if applicable) or ``None`` if no match exceeds the threshold.
        """
        embedding = self._encode(text)
        if embedding is None:
            return None
        warning = self._check_similarity(embedding)
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

    def add(self, text: str, npc_id: str, kind: str, tick: int) -> Optional[dict]:
        """
        Add a new entry to the ledger and return a similarity warning if
        any prior entry exceeds the threshold. Returns None when no
        warning fires (or when embeddings are unavailable).
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

        self.entries.append({
            "text": text[:400],
            "embedding": embedding.tolist(),
            "npc_id": npc_id,
            "kind": kind,
            "tick": tick,
        })
        # Bound memory — keep last 200 entries (largest a typical session reaches)
        if len(self.entries) > 200:
            self.entries = self.entries[-200:]
        self._save()
        return warning

    def _encode(self, text: str):
        """Lazy-encode helper shared by ``check`` and ``add``."""
        if not text or not isinstance(text, str):
            return None
        embedder = self.embedder
        if embedder is None or self.np is None:
            return None
        try:
            return embedder.encode(text, normalize_embeddings=True)
        except Exception as e:
            logger.error(f"FactLedger encode failed: {e}")
            return None

    def reset(self) -> None:
        self.entries = []
        if self.storage_path.exists():
            try:
                self.storage_path.unlink()
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

    def _check_similarity(self, new_embedding) -> Optional[dict]:
        if not self.entries:
            return None
        np = self.np
        if np is None:
            return None
        try:
            existing = np.array([e["embedding"] for e in self.entries])
            sims = existing @ new_embedding  # cosine, vectors normalized
            max_idx = int(sims.argmax())
            max_sim = float(sims[max_idx])
        except Exception as e:
            logger.error(f"FactLedger similarity check failed: {e}")
            return None
        if max_sim < self.threshold:
            return None
        match = self.entries[max_idx]
        return {
            "similarity": round(max_sim, 3),
            "matches_text": match["text"][:240],
            "matches_npc": match["npc_id"],
            "matches_kind": match["kind"],
            "matches_tick": match["tick"],
        }

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self.entries = data.get("entries", [])[-200:]
        except Exception as e:
            logger.warning(f"FactLedger load failed: {e}")

    def _save(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(
                json.dumps({"entries": self.entries}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
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
    """

    id: str
    theme: str
    focus_npcs: list[str]
    beat_goals: list[str]
    current_beat: int = 0
    status: str = "active"  # active | resolved | abandoned
    started_at_tick: int = 0
    last_advanced_at_tick: int = 0

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

    Deterministic proposal (v1): greedy-cluster the recent FactLedger
    entries by cosine similarity. The densest cluster becomes an arc —
    theme = the center entry's text, focus_npcs = the unique NPC ids
    across the cluster. Fixed 4-beat skeleton, touch-counter advancement.

    No LLM call is made during proposal or advancement — the planner is
    pure Python. A future v2 could add an LLM-theming pass for richer
    theme strings, but the clusters themselves are already grounded in
    Director-written content so themes are never "made up from nothing."
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.arcs: list[NarrativeArc] = []
        self.active_arc_id: Optional[str] = None
        self._last_proposal_attempt_tick: int = 0
        self._load()

    # ── Public API ──────────────────────────────────────────────

    def active(self) -> Optional[NarrativeArc]:
        """Return the currently-active arc, or None."""
        if not self.active_arc_id:
            return None
        for arc in self.arcs:
            if arc.id == self.active_arc_id and arc.status == "active":
                return arc
        # Stale id — clear it
        self.active_arc_id = None
        return None

    def maybe_propose(self, ledger: "FactLedger", available_npcs: list[str],
                       current_tick: int) -> Optional[NarrativeArc]:
        """
        Attempt to propose a new arc if there's no active one and the
        cooldown has elapsed. Returns the new arc (and sets it active) or
        ``None`` if nothing was proposed.
        """
        if self.active() is not None:
            return None
        if current_tick - self._last_proposal_attempt_tick < _ARC_PROPOSAL_COOLDOWN_TICKS:
            return None
        self._last_proposal_attempt_tick = current_tick

        arc = self._propose_from_ledger(ledger, available_npcs, current_tick)
        if arc is not None:
            self.arcs.append(arc)
            self.active_arc_id = arc.id
            logger.info(
                f"ArcPlanner proposed arc {arc.id}: '{arc.theme[:60]}' "
                f"(focus={arc.focus_npcs})"
            )
        self.save()
        return arc

    def advance_if_beat_met(self, recent_decisions: list[dict],
                             current_tick: int) -> bool:
        """
        Count focus-NPC touches in ``recent_decisions`` since the current
        beat started. If the threshold is met, bump the beat. Resolve the
        arc when the final beat completes. Returns True when a beat
        advanced (including resolution).
        """
        arc = self.active()
        if arc is None:
            return False

        touches = 0
        for d in recent_decisions:
            tick_num = d.get("tick", 0)
            if tick_num <= arc.last_advanced_at_tick:
                continue
            actions_in_decision: list[dict] = []
            if isinstance(d.get("action"), dict):
                actions_in_decision.append(d["action"])
            for sub in d.get("sub_actions", []) or []:
                if isinstance(sub, dict) and isinstance(sub.get("action"), dict):
                    actions_in_decision.append(sub["action"])
            for act in actions_in_decision:
                for key in ("npc_id", "target"):
                    val = act.get(key)
                    if isinstance(val, str) and val in arc.focus_npcs:
                        touches += 1
                        break  # don't double-count one action

        if touches < _ARC_BEAT_ADVANCE_THRESHOLD:
            return False

        arc.current_beat += 1
        arc.last_advanced_at_tick = current_tick
        logger.info(
            f"ArcPlanner advanced {arc.id} to beat {arc.current_beat} "
            f"({arc.current_beat_label})"
        )
        if arc.is_complete:
            arc.status = "resolved"
            self.active_arc_id = None
            logger.info(f"ArcPlanner resolved {arc.id}")
        self.save()
        return True

    def stats(self) -> dict:
        active = self.active()
        return {
            "arc_count": len(self.arcs),
            "active_arc_id": self.active_arc_id,
            "active_theme": active.theme if active else None,
            "active_beat": active.current_beat_label if active else None,
            "last_proposal_attempt_tick": self._last_proposal_attempt_tick,
        }

    def reset(self) -> None:
        self.arcs = []
        self.active_arc_id = None
        self._last_proposal_attempt_tick = 0
        if self.storage_path.exists():
            try:
                self.storage_path.unlink()
            except Exception:
                pass

    # ── Internals ───────────────────────────────────────────────

    def _propose_from_ledger(self, ledger: "FactLedger",
                              available_npcs: list[str],
                              current_tick: int) -> Optional[NarrativeArc]:
        """
        Greedy clustering over the most recent ledger entries. Pick the
        entry with the most high-similarity neighbors as the cluster
        center, then collect its neighbors. Theme = center entry's text,
        focus NPCs = unique NPC ids in the cluster filtered against
        ``available_npcs``.
        """
        if len(ledger.entries) < _ARC_PROPOSAL_MIN_LEDGER_ENTRIES:
            return None
        np = ledger.np
        if np is None:
            return None

        recent = ledger.entries[-_ARC_CLUSTER_LOOKBACK:]
        try:
            embeddings = np.array([e["embedding"] for e in recent])
            # Pairwise cosine — entries are already L2-normalized on encode
            sims = embeddings @ embeddings.T
            # Zero the diagonal so self-similarity doesn't dominate
            for i in range(len(sims)):
                sims[i][i] = 0.0
            neighbor_counts = (sims >= _ARC_CLUSTER_SIMILARITY).sum(axis=1)
            best_idx = int(neighbor_counts.argmax())
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
        # filtered to the currently-available roster. Dropping "all" and
        # anything not in the world keeps the arc grounded.
        focus_npcs: list[str] = []
        seen: set[str] = set()
        avail = set(available_npcs)
        for e in cluster_entries:
            nid = e.get("npc_id") or ""
            if nid and nid in avail and nid not in seen:
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
            self.active_arc_id = data.get("active_arc_id")
            self._last_proposal_attempt_tick = int(data.get("last_proposal_attempt_tick", 0))
        except Exception as e:
            logger.warning(f"ArcPlanner load failed: {e}")
            self.arcs = []
            self.active_arc_id = None

    def save(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(
                json.dumps({
                    "arcs": [asdict(a) for a in self.arcs],
                    "active_arc_id": self.active_arc_id,
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
        self._kind_rotation_index: int = 0      # round-robin over _ACTION_KIND_ROTATION
        self.recent_player_actions: list[dict] = []  # last 8 player observations
        self._lore_text: str = ""
        self._examples: list[dict] = []
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
        self.ledger = FactLedger(LEDGER_FILE)
        self.arc_planner = ArcPlanner(ARCS_FILE)
        self._load_assets()
        self._load_state()
        self._snapshot_original_bios()

    # ── Lifecycle ───────────────────────────────────────────────

    def _load_assets(self) -> None:
        if LORE_FILE.exists():
            self._lore_text = LORE_FILE.read_text(encoding="utf-8").strip()
        else:
            logger.warning(f"Story Director lore file not found at {LORE_FILE}")

        if EXAMPLES_FILE.exists():
            try:
                data = yaml.safe_load(EXAMPLES_FILE.read_text(encoding="utf-8")) or {}
                self._examples = data.get("examples", []) or []
            except Exception as e:
                logger.error(f"Story Director failed to load examples: {e}")
                self._examples = []
        else:
            logger.warning(f"Story Director examples file not found at {EXAMPLES_FILE}")

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            self.tick_count = state.get("tick_count", 0)
            self.last_tick_at = state.get("last_tick_at")
            self.recent_decisions = state.get("recent_decisions", [])[-5:]
            self._kind_rotation_index = state.get("kind_rotation_index", 0)
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
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "tick_count": self.tick_count,
            "last_tick_at": self.last_tick_at,
            "recent_decisions": self.recent_decisions[-5:],
            "kind_rotation_index": self._kind_rotation_index,
            "recent_player_actions": self.recent_player_actions[-8:],
            "bio_mention_counts": self._bio_mention_counts,
        }
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # ── Public API ──────────────────────────────────────────────

    def tick(self, max_tokens: int = 400, temperature: float = 0.7,
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

        snapshot = self._world_snapshot()
        plan = self._architect_plan(actions_per_tick)

        # Try to propose a narrative arc BEFORE workers run so their
        # prompts can reference it. No-op if there's already an active
        # arc or the ledger is too thin — proposal is cooldown-gated.
        available_npcs = list(self.engine.pie.npc_knowledge.profiles.keys())
        self.arc_planner.maybe_propose(
            self.ledger, available_npcs, current_tick=self.tick_count + 1,
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
                }
            return {"tick": self.tick_count, "sub_actions": []}

        sub_results: list[dict] = []
        for focus_npc, action_kind in plan:
            sub_results.append(self._run_single_action(
                snapshot=snapshot,
                focus_npc=focus_npc,
                action_kind=action_kind,
                max_tokens=max_tokens,
                temperature=temperature,
            ))

        self.tick_count += 1
        self.last_tick_at = datetime.now(timezone.utc).isoformat()

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
        self.arc_planner.advance_if_beat_met(
            self.recent_decisions, current_tick=self.tick_count,
        )

        self._save_state()

        if actions_per_tick == 1:
            r = sub_results[0]
            return {
                "tick": self.tick_count,
                "action": r["action"],
                "dispatch": r["dispatch"],
                "raw_response": r["raw_response"],
            }
        return {
            "tick": self.tick_count,
            "sub_actions": sub_results,
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
        precheck = self._precheck_contradiction(action)
        if precheck is not None:
            retried = True
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

        dispatch_result = self._dispatch(action)

        # Ledger every successful, non-noop injection so contradictions
        # across sub-actions (within the same tick or across ticks) are
        # caught uniformly.
        if dispatch_result.get("ok") and dispatch_result.get("kind") not in (None, "noop"):
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
            dispatch_result["retried_after_contradiction"] = True

        return {
            "focus_npc": focus_npc,
            "action_kind": action_kind,
            "action": action,
            "dispatch": dispatch_result,
            "raw_response": raw,
            "retried": retried,
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

    def _architect_plan(self, n_actions: int) -> list[tuple[Optional[str], str]]:
        """
        Plan N distinct ``(focus_npc, action_kind)`` slots for a multi-
        action tick. Each slot is picked using the same focus + kind
        rotation as single-action mode, but the in-flight planning loop
        adds each chosen NPC to a temporary exclusion set so two
        workers can't compete for the same target.

        Returns at most ``n_actions`` slots, fewer if the world doesn't
        have enough NPCs.
        """
        plan: list[tuple[Optional[str], str]] = []
        excluded: set[str] = set()
        for _ in range(max(1, n_actions)):
            focus = self._pick_focus_npc(extra_exclude=excluded)
            if focus is None:
                break
            excluded.add(focus)
            kind = self._pick_action_kind(focus)
            plan.append((focus, kind))
        return plan

    def get_state(self) -> dict:
        return {
            "tick_count": self.tick_count,
            "last_tick_at": self.last_tick_at,
            "recent_decisions": self.recent_decisions,
            "recent_player_actions": self.recent_player_actions,
            "kind_rotation_index": self._kind_rotation_index,
            "lore_loaded": bool(self._lore_text),
            "example_count": len(self._examples),
            "ledger": self.ledger.stats(),
            "arc_planner": self.arc_planner.stats(),
        }

    def record_player_action(self, text: str,
                              target: Optional[str] = None,
                              trust_delta: Optional[int] = None,
                              quest_completed: Optional[str] = None,
                              quest_accepted: Optional[dict] = None) -> dict:
        """
        Record something the player did. Surfaced in the next tick's
        world snapshot so the Director can react to player behavior.

        Args:
            text: Human-readable description of what the player did.
            target: Optional NPC id the action was directed at.
            trust_delta: Optional trust adjustment to apply to ``target``
                (positive = friendlier, negative = more hostile). Lets
                a game client encode "player gave a gift" in one call.
            quest_completed: Optional quest id. When set, the Director
                calls ``engine.complete_quest`` so the engine's
                ``player_quests`` tracker reflects the completion. The
                trust ripple + gossip propagation happen automatically
                through the engine's existing pipeline.
            quest_accepted: Optional ``{id, name, given_by}`` dict. When
                set, the Director calls ``engine.accept_quest`` so the
                engine's tracker shows the quest as active.
        """
        if not text or not isinstance(text, str):
            return {"ok": False, "reason": "empty_text"}

        record = {
            "at": datetime.now(timezone.utc).isoformat(),
            "tick_at_time": self.tick_count,
            "text": text.strip()[:240],
            "target": target,
        }

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
        self._save_state()
        return {"ok": True, "recorded": record}

    # ── World snapshot ──────────────────────────────────────────

    def _world_snapshot(self) -> str:
        """
        Compact world-state description the overseer will reason over.

        Events tagged with source="director" are FILTERED OUT — the Director
        should never see its own outputs as "world state" or it will echo
        them back and spiral into repetition. The Director's own past
        actions appear in a separate ALREADY DONE block below.
        """
        pie = self.engine.pie
        lines: list[str] = []

        world_name = self.engine.config.world_name or "Ashenvale"
        lines.append(f"World: {world_name}")

        # NPCs with role + current mood/trust/top goal if available. The
        # top goal is the single biggest piece of motivational fuel for
        # the Director — with it, every NPC in the roster tells the
        # model WHAT THEY WANT, not just what they are.
        for npc_id, npc in pie.npc_knowledge.profiles.items():
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
        # repetition spiral on small models.
        seen = set()
        recent_events: list[str] = []
        for npc in pie.npc_knowledge.profiles.values():
            for e in npc.events[-4:]:
                if getattr(e, "source", "") == "director":
                    continue
                if e.description not in seen:
                    seen.add(e.description)
                    recent_events.append(e.description)
        if recent_events:
            lines.append("Recent organic events: " + " | ".join(recent_events[-5:]))

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

        Returns at most 3 picks. Falls back to the full library if
        filtering leaves nothing (never emit an empty EXAMPLES block —
        schema parse reliability drops when the model loses its shape
        reference entirely).
        """
        if not self._examples:
            return []

        # Rule 1: exclude examples whose primary_npc matches focus_npc
        eligible = [
            ex for ex in self._examples
            if focus_npc is None or ex.get("primary_npc") != focus_npc
        ]
        if not eligible:
            # All examples were about the focus NPC (shouldn't happen with
            # 5 examples and 7 NPCs, but guard anyway)
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
            if len(picks) >= 3:
                break
            if ex in picks:
                continue
            ex_kind = ex.get("action", {}).get("action")
            # Skip if we already have this kind AND we haven't filled up
            if ex_kind in shown_kinds and len(picks) < 3:
                continue
            picks.append(ex)
            shown_kinds.add(ex_kind)

        # If we still don't have 3, top up with anything left
        for ex in eligible:
            if len(picks) >= 3:
                break
            if ex not in picks:
                picks.append(ex)

        return picks[:3]

    def _pick_action_kind(self, focus_npc: Optional[str]) -> str:
        """
        Python decides the action KIND (event / quest / fact) — round-robin
        so a single kind can't dominate a session. Skips 'quest' when the
        focus NPC already has ``_MAX_QUESTS_PER_NPC`` open quests; the LLM
        would otherwise keep piling work onto the same NPC.

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

    def _pick_focus_npc(self, extra_exclude: Optional[set[str]] = None) -> Optional[str]:
        """
        Python decides WHICH NPC this tick focuses on. The LLM decides WHAT
        happens to them. Two layers:

        1. **Player reactivity**: if the player did something targeting a
           specific NPC *since* the last tick, prioritize that NPC — the
           Director should respond to the player's moves immediately.
        2. **Round-robin rotation**: otherwise, pick the least-recently-
           touched NPC. This keeps the story from fixating when the
           player is passive.

        ``extra_exclude`` is used by the architect's in-flight planner to
        prevent two workers in the same multi-action tick from competing
        for the same NPC. NPCs in this set are skipped at both layers.

        The split exists because Qwen/Llama 3B — even with strongly-worded
        rules in the prompt — still fixate on a single target or abuse
        ``"all"``. We take the choice out of the model's hands.
        """
        profiles = list(self.engine.pie.npc_knowledge.profiles.keys())
        if not profiles:
            return None

        excluded = set(extra_exclude or ())
        available = [nid for nid in profiles if nid not in excluded]
        if not available:
            return None

        # Layer 1: react to pending player action (but only if the
        # player's target isn't already taken by another worker)
        pending_target = self._pending_player_target(available)
        if pending_target:
            return pending_target

        # Layer 2: least-recently-touched rotation
        last_touched: dict[str, int] = {npc_id: -1 for npc_id in available}
        for d in self.recent_decisions:
            tick_num = d.get("tick", 0)
            # Aggregate touches across both the canonical "action" field
            # and any sub-actions stored on multi-action ticks.
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
            available,
            key=lambda nid: (last_touched[nid], available.index(nid)),
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
        active_arc = self.arc_planner.active()
        if active_arc is not None and active_arc.current_beat_goal:
            arc_lines = [
                "=== ACTIVE NARRATIVE ARC ===",
                f"Theme: {active_arc.theme}",
                f"Cast: {', '.join(active_arc.focus_npcs)}",
                f"Current beat ({active_arc.current_beat + 1}/{len(active_arc.beat_goals)}): "
                f"{active_arc.current_beat_goal}",
                "If this tick's focus NPC fits the cast, advance the beat. "
                "If not, write something the cast can react to next tick.",
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
                return self._dispatch_quest(action)
            if kind == "event":
                return self._dispatch_event(action)
            if kind == "fact":
                return self._dispatch_fact(action)
            if kind == "noop":
                return {"ok": True, "kind": "noop"}
            return {"ok": False, "reason": f"unknown_action_kind: {kind}"}
        except Exception as e:
            logger.exception("Story Director dispatch failed")
            return {"ok": False, "reason": f"dispatch_error: {e}"}

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

        quest = Quest(
            id=quest_id,
            name=str(quest_data.get("name") or quest_id),
            description=str(quest_data.get("description") or ""),
            status="available",
            reward=str(quest_data.get("reward") or ""),
            objectives=[str(o) for o in objectives],
        )
        npc.add_quest(quest)
        # Announce the new quest as a tagged Director event so it propagates
        # to other NPCs but does not re-enter the Director's own snapshot.
        self._inject_tagged_event(
            f"{npc.identity.get('name', npc_id)} has new work to offer: {quest.name}",
            npc_id=None,
        )
        return {"ok": True, "kind": "quest", "npc_id": npc_id, "quest_id": quest_id}

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
