"""
Narrative Judge — coherence scorer for (fact, quest_objective) pairs.

Architecture adapts GE-Sim 2.0's World Judge (arXiv 2605.27491) to text:
  Frozen backbone : cross-encoder/nli-deberta-v3-small  +  all-MiniLM-L6-v2
  Trainable head  : sklearn LogisticRegression on an 8-dim feature vector
  Loss (training) : cross-entropy, 3-class (advance / block / neutral)

Trained on 216 hand-labeled (fact, quest_objective, label) tuples spanning
the Ashenvale and Port Blackwater worlds (see train_narrative_judge.py at
repo root). Held-out accuracy was 89.7% single-domain, 70.8% zero-shot
cross-domain, 80.0% few-shot with +57 labels in the new world.

Falls back to no-op (returns None) if any of:
  - sentence-transformers / sklearn not installed
  - data/narrative_judge.pkl artifact missing or wrong schema_version
  - NLI or embedder model fails to load

The fallback path mirrors ContradictionChecker — the Director still works,
just without the coherence-scoring lane.
"""
from __future__ import annotations

import logging
import math
import pickle
import re
from pathlib import Path
from typing import Any, Optional

from npc_engine.bridge import NPC_ENGINE_ROOT

logger = logging.getLogger("NPCEngine.narrative_judge")

# Schema version must match train_narrative_judge.py
EXPECTED_SCHEMA_VERSION = 1

DEFAULT_PICKLE_PATH = NPC_ENGINE_ROOT / "data" / "narrative_judge.pkl"
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

NLI_LABELS = ("contradiction", "entailment", "neutral")
CLASS_LABELS = ("advance", "block", "neutral")

_WORD_RE = re.compile(r"[a-zA-Z']+")
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "to", "in", "and", "or", "but",
    "is", "was", "are", "were", "be", "been", "being", "has",
    "have", "had", "do", "does", "did", "for", "on", "at", "by",
    "with", "from", "up", "about", "into", "over", "after",
    "that", "this", "these", "those", "it", "its", "as", "if",
    "than", "so", "no", "not", "very", "just", "all", "any",
    "each", "every", "some", "who", "whom", "what", "which",
    "where", "when", "why", "how", "player",
})


def _softmax(xs):
    mx = max(xs)
    exps = [math.exp(x - mx) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def _token_overlap(hypothesis: str, premise: str) -> float:
    htoks = {w.lower() for w in _WORD_RE.findall(hypothesis)
             if w.lower() not in _STOPWORDS}
    if not htoks:
        return 0.0
    ptoks = {w.lower() for w in _WORD_RE.findall(premise)
             if w.lower() not in _STOPWORDS}
    return len(htoks & ptoks) / len(htoks)


def _cosine(a, b) -> float:
    import numpy as np
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class NarrativeJudge:
    """Lazy-loaded coherence scorer. Falls back to no-op silently.

    Usage:
        judge = NarrativeJudge()
        result = judge.score("Mara confessed to the theft.",
                             "The player identified the thief.")
        # result = {"label": "advance", "scores": {"advance": 0.87, ...}}
        #          or None if the judge is unavailable.
    """

    def __init__(
        self,
        pickle_path: Path = DEFAULT_PICKLE_PATH,
        nli_model_name: str = DEFAULT_NLI_MODEL,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
    ):
        self.pickle_path = Path(pickle_path)
        self.nli_model_name = nli_model_name
        self.embed_model_name = embed_model_name
        self._artifact: Optional[dict] = None
        self._nli = None       # None = not attempted; False = unavailable
        self._embedder = None
        self._embed_cache: dict[str, Any] = {}
        # One-slot (fact, quest) -> result cache (mirrors ContradictionChecker)
        self._cached_key: Optional[tuple[str, str]] = None
        self._cached_result: Optional[dict] = None

    # ─── lazy-load helpers ─────────────────────────────────────────

    def _ensure_artifact(self) -> Optional[dict]:
        if self._artifact is None:
            try:
                with open(self.pickle_path, "rb") as f:
                    art = pickle.load(f)
                if art.get("schema_version") != EXPECTED_SCHEMA_VERSION:
                    logger.warning(
                        f"NarrativeJudge artifact schema_version "
                        f"{art.get('schema_version')} != expected "
                        f"{EXPECTED_SCHEMA_VERSION}; refusing to load.")
                    self._artifact = False
                else:
                    self._artifact = art
                    logger.info(
                        f"NarrativeJudge loaded artifact: "
                        f"{art.get('n_train')} train tuples, "
                        f"acc={art.get('train_acc'):.3f}")
            except FileNotFoundError:
                logger.info(
                    f"NarrativeJudge pickle not found at {self.pickle_path}; "
                    f"judge will no-op. Run train_narrative_judge.py to create.")
                self._artifact = False
            except Exception as e:
                logger.warning(f"NarrativeJudge artifact load failed: {e}")
                self._artifact = False
        return self._artifact if self._artifact is not False else None

    def _ensure_nli(self):
        if self._nli is None:
            try:
                from sentence_transformers import CrossEncoder
                self._nli = CrossEncoder(self.nli_model_name)
                logger.info(f"NarrativeJudge loaded NLI: {self.nli_model_name}")
            except Exception as e:
                logger.warning(f"NarrativeJudge NLI unavailable: {e}")
                self._nli = False
        return self._nli if self._nli is not False else None

    def _ensure_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embed_model_name)
                logger.info(f"NarrativeJudge loaded embedder: {self.embed_model_name}")
            except Exception as e:
                logger.warning(f"NarrativeJudge embedder unavailable: {e}")
                self._embedder = False
        return self._embedder if self._embedder is not False else None

    @property
    def available(self) -> bool:
        return (self._ensure_artifact() is not None
                and self._ensure_nli() is not None
                and self._ensure_embedder() is not None)

    # ─── feature extraction ────────────────────────────────────────

    def _embed(self, text: str):
        import numpy as np
        if text not in self._embed_cache:
            emb = self._embedder.encode(text)  # type: ignore[attr-defined]
            self._embed_cache[text] = np.asarray(emb, dtype=np.float32)
        return self._embed_cache[text]

    def _featurize(self, fact: str, quest_objective: str):
        import numpy as np
        raw = self._nli.predict([(fact, quest_objective)])[0]  # type: ignore[attr-defined]
        probs = _softmax([float(s) for s in raw])
        nli_scores = {NLI_LABELS[i]: probs[i] for i in range(len(NLI_LABELS))}
        cos = _cosine(self._embed(fact), self._embed(quest_objective))
        overlap = _token_overlap(quest_objective, fact)
        length_norm = min(len(fact) / 200.0, 1.5)
        return np.array([
            nli_scores["contradiction"],
            nli_scores["entailment"],
            nli_scores["neutral"],
            cos,
            overlap,
            length_norm,
            nli_scores["entailment"] * cos,
            nli_scores["contradiction"] * cos,
        ], dtype=np.float32)

    # ─── public API ────────────────────────────────────────────────

    def score(self, fact: str, quest_objective: str) -> Optional[dict]:
        """Score a (fact, quest_objective) pair. Returns:

            {
              "label": "advance" | "block" | "neutral",
              "confidence": float,
              "scores": {"advance": float, "block": float, "neutral": float},
            }

        or None if the judge is unavailable. Empty inputs also return None.
        """
        if not fact or not quest_objective:
            return None
        if not self.available:
            return None

        key = (fact, quest_objective)
        if key == self._cached_key and self._cached_result is not None:
            return self._cached_result

        try:
            feat = self._featurize(fact, quest_objective)
            scaler = self._artifact["scaler"]  # type: ignore[index]
            model = self._artifact["model"]    # type: ignore[index]
            X = scaler.transform(feat.reshape(1, -1))
            proba = model.predict_proba(X)[0]
            # Class order in trained model matches CLASS_LABELS order
            scores = {CLASS_LABELS[i]: float(proba[i])
                      for i in range(len(CLASS_LABELS))}
            label = max(scores.keys(), key=lambda k: scores[k])
            result = {
                "label": label,
                "confidence": round(scores[label], 3),
                "scores": {k: round(v, 3) for k, v in scores.items()},
            }
        except Exception as e:
            logger.error(f"NarrativeJudge.score failed: {e}")
            return None

        self._cached_key = key
        self._cached_result = result
        return result

    def score_against_quests(
        self, fact: str, quest_specs: dict[str, str],
    ) -> Optional[dict]:
        """Score one fact against multiple quest objectives.

        Returns a dict mapping quest_id -> score(fact, quest_obj) result,
        or None if the judge is unavailable. Useful for the SUGAR
        candidate-pool path where one candidate gets scored against
        every active quest spec.
        """
        if not fact or not quest_specs:
            return None
        if not self.available:
            return None
        return {qid: self.score(fact, obj) for qid, obj in quest_specs.items()}

    def score_against_quests_ranked(
        self, fact: str, quest_specs: dict[str, str],
    ) -> Optional[dict]:
        """Score one fact against multiple quests and return ranked
        decision metrics for the "which single quest does this advance"
        use case.

        In tightly-interconnected lore worlds the absolute advance
        probability is often high (>0.8) for multiple quests at once —
        a positive signal for plot-coherence ranking but a weak signal
        for single-quest attribution. This method returns *relative*
        metrics so a caller can gate on confidence:

          best_quest        — quest_id with highest P(advance)
          best_advance      — P(advance) for best_quest
          runner_up_quest   — quest_id with 2nd-highest P(advance)
          runner_up_advance — P(advance) for runner_up_quest
          margin            — best_advance - runner_up_advance
          softmax_peak      — softmax over advance scores, peak value
                              (1.0 = perfectly concentrated on one quest;
                              1/N = perfectly flat across N quests)
          ranked            — full list of (quest_id, advance_prob)
                              sorted descending
          per_quest         — full {qid: score result} dict (same as
                              score_against_quests output)

        Returns None if the judge is unavailable or no scoreable quests.
        """
        per_quest = self.score_against_quests(fact, quest_specs)
        if not per_quest:
            return None

        # Collect advance probabilities, dropping any None results
        adv_pairs = []
        for qid, res in per_quest.items():
            if res is None:
                continue
            adv = res.get("scores", {}).get("advance", 0.0)
            adv_pairs.append((qid, float(adv)))
        if not adv_pairs:
            return None

        # Rank descending
        adv_pairs.sort(key=lambda kv: -kv[1])
        ranked = adv_pairs
        best_qid, best_adv = ranked[0]
        runner_qid, runner_adv = (ranked[1] if len(ranked) > 1
                                   else (None, 0.0))
        margin = best_adv - runner_adv

        # Softmax over advance scores. Lower-temperature softmax would
        # sharpen but we want the raw concentration as a calibration
        # signal, so use T=1.
        advances = [adv for _, adv in ranked]
        probs = _softmax(advances)
        softmax_peak = probs[0]

        return {
            "best_quest": best_qid,
            "best_advance": round(best_adv, 4),
            "runner_up_quest": runner_qid,
            "runner_up_advance": round(runner_adv, 4),
            "margin": round(margin, 4),
            "softmax_peak": round(softmax_peak, 4),
            "ranked": [(qid, round(adv, 4)) for qid, adv in ranked],
            "per_quest": per_quest,
        }
