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

    def add(self, text: str, npc_id: str, kind: str, tick: int) -> Optional[dict]:
        """
        Add a new entry to the ledger and return a similarity warning if
        any prior entry exceeds the threshold. Returns None when no
        warning fires (or when embeddings are unavailable).
        """
        if not text or not isinstance(text, str):
            return None
        embedder = self.embedder
        np = self.np
        if embedder is None or np is None:
            return None

        try:
            embedding = embedder.encode(text, normalize_embeddings=True)
        except Exception as e:
            logger.error(f"FactLedger encode failed: {e}")
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
        self.ledger = FactLedger(LEDGER_FILE)
        self._load_assets()
        self._load_state()

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
        }
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # ── Public API ──────────────────────────────────────────────

    def tick(self, max_tokens: int = 400, temperature: float = 0.7) -> dict:
        """
        Advance the story by one decision. Returns a dict describing what
        happened. Always returns — never raises — so a game loop can call
        this on a timer without guarding every field.
        """
        snapshot = self._world_snapshot()
        focus_npc = self._pick_focus_npc()
        action_kind = self._pick_action_kind(focus_npc)
        prompt = self._build_prompt(snapshot, focus_npc, action_kind)

        raw, action = self._llm_call_with_repair(prompt, max_tokens, temperature)

        # Python owns both the focus NPC and the action kind — the LLM
        # only decides the creative content. Enforce both after parse in
        # case the model deviated from the prompt directive.
        if focus_npc:
            action = self._enforce_focus_npc(action, focus_npc)
        action = self._enforce_action_kind(action, action_kind, focus_npc)

        dispatch_result = self._dispatch(action)

        # Record successful injections in the FactLedger and surface any
        # similarity warning. We do this AFTER dispatch so we only ledger
        # actions that actually mutated the world (no point auditing noops
        # or rejected dispatches).
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

        self.tick_count += 1
        self.last_tick_at = datetime.now(timezone.utc).isoformat()
        decision_record = {
            "tick": self.tick_count,
            "at": self.last_tick_at,
            "action": action,
            "dispatch": dispatch_result,
            "snapshot_preview": snapshot[:200],
        }
        self.recent_decisions.append(decision_record)
        self.recent_decisions = self.recent_decisions[-5:]
        self._save_state()

        return {
            "tick": self.tick_count,
            "action": action,
            "dispatch": dispatch_result,
            "raw_response": raw,
        }

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
        }

    def record_player_action(self, text: str,
                              target: Optional[str] = None,
                              trust_delta: Optional[int] = None) -> dict:
        """
        Record something the player did. Surfaced in the next tick's
        world snapshot so the Director can react to player behavior.

        Args:
            text: Human-readable description of what the player did.
            target: Optional NPC id the action was directed at.
            trust_delta: Optional trust adjustment to apply to ``target``
                (positive = friendlier, negative = more hostile). Lets
                a game client encode "player gave a gift" in one call.
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

        # NPCs with role + current mood/trust if available
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

    def _pick_focus_npc(self) -> Optional[str]:
        """
        Python decides WHICH NPC this tick focuses on. The LLM decides WHAT
        happens to them. Two layers:

        1. **Player reactivity**: if the player did something targeting a
           specific NPC *since* the last tick, prioritize that NPC — the
           Director should respond to the player's moves immediately.
        2. **Round-robin rotation**: otherwise, pick the least-recently-
           touched NPC. This keeps the story from fixating when the
           player is passive.

        The split exists because Qwen/Llama 3B — even with strongly-worded
        rules in the prompt — still fixate on a single target or abuse
        ``"all"``. We take the choice out of the model's hands.
        """
        profiles = list(self.engine.pie.npc_knowledge.profiles.keys())
        if not profiles:
            return None

        # Layer 1: react to pending player action
        pending_target = self._pending_player_target(profiles)
        if pending_target:
            return pending_target

        # Layer 2: least-recently-touched rotation
        last_touched: dict[str, int] = {npc_id: -1 for npc_id in profiles}
        for d in self.recent_decisions:
            tick_num = d.get("tick", 0)
            act = d.get("action", {})
            if not isinstance(act, dict):
                continue
            for key in ("npc_id", "target"):
                val = act.get(key)
                if isinstance(val, str) and val in last_touched:
                    last_touched[val] = max(last_touched[val], tick_num)

        ordered = sorted(
            profiles,
            key=lambda nid: (last_touched[nid], profiles.index(nid)),
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

        if self._examples:
            ex_blocks = []
            for ex in self._examples:
                ws = str(ex.get("world_state", "")).strip()
                action = ex.get("action", {})
                ex_blocks.append(
                    "WORLD STATE:\n" + ws + "\nACTION:\n" + json.dumps(action, ensure_ascii=False)
                )
            parts.append("=== EXAMPLES ===\n\n".join([""] + ex_blocks).strip())

        parts.append("=== CURRENT WORLD STATE ===\n" + snapshot)

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
