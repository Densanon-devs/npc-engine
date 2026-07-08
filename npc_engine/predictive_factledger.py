"""
Predictive FactLedger — the passive forward-predictor lane (v1).

Turns the FactLedger from a passive *record* into a passive
*record + cheap forward-predictor*, per
``data/story_director/PHASE_PREDICTIVE_FACTLEDGER_PLAN.md``. Two
devices, both pure NumPy, no model calls, no GPU:

- ``EdgeFilter`` — Perpetua*-style Bayesian filters on cyclic
  FactLedger relationships. One Beta(1, 1) posterior per
  (edge key, tick-phase bucket); update is a one-line conjugate
  count bump, predict is the posterior mean. Cold filters return
  exactly 0.5, so the arc-proposal boost derived from them is
  exactly zero — cold edges literally do nothing.
- ``ActivityPrior`` — Being-H0.7-style prior branch: a single
  linear projection from a recent-ledger summary latent to logits
  over the ``PlayerActivity`` labels. Trained by closed-form ridge
  regression on (latent, observed-activity) pairs harvested from
  ``/story/activity`` posts. No weight matrix yet = uniform
  distribution + ``cold=True``.

``PredictiveLayer`` owns both, serializes them to an ``.npz``
sidecar (same pattern as ``fact_ledger.embeddings.npy``), and
replays history from the activity-history JSONL on warm start.
The hot path (``predict_next``) is < 1 ms — pure linalg, no I/O.

This module deliberately does NOT import ``story_director`` —
the activity label set is injected at construction so the
dependency points one way (StoryDirector -> PredictiveLayer).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("NPCEngine.predictive_factledger")

# Number of tick-phase buckets for cyclic edge filters. The game's
# only clock is the tick counter, so "time of day" is tick % cycle.
# 8 buckets ≈ a coarse day-cycle at typical town cadence; small
# enough that each bucket accumulates observations quickly.
_EDGE_BUCKET_CYCLE = 8

# Minimum (latent, label) pairs before the ActivityPrior will fit.
# Below this the prior stays cold (uniform) — fitting a 384-dim
# projection on a handful of points would just memorize them.
_PRIOR_MIN_FIT_SAMPLES = 8

# Ridge regularization for the closed-form fit. The latent is an
# L2-normalized embedding mean, so features are O(0.1); λ=1.0 keeps
# the solve well-conditioned even with near-duplicate latents.
_PRIOR_RIDGE_LAMBDA = 1.0

# How many of the most recent ledger entries feed the summary latent.
_PRIOR_RECENT_WINDOW = 12

# Predicted-vs-reported activity disagreement is only logged when the
# prior's top-label confidence clears this. Uniform over 9 labels is
# ~0.11, so 0.55 means "the prior is genuinely committed".
_DRIFT_CONFIDENCE_THRESHOLD = 0.55

# Cap on the multiplicative arc-proposal boost derived from edge
# priors — the prior may nudge cluster selection by at most 10% of
# the base score, never dominate it (plan: "cap < 0.10 of total").
EDGE_PRIOR_BOOST_CAP = 0.10

# Activity-history records older than this are pruned at
# warm-from-history time (plan risk register: cap unbounded growth).
_ACTIVITY_HISTORY_MAX_AGE_DAYS = 90

# In-memory supervision-pair cap. Each pair carries a ~384-dim float64
# latent; without a cap a multi-day server session would grow the list
# (and the per-post ridge refit cost) without bound. The newest pairs
# win — recency matters more than volume for a per-engagement prior.
_MAX_ACTIVITY_PAIRS = 512

# Sidecar format version, stored in the .npz for forward compat.
_SIDECAR_VERSION = 1


def _np():
    """Lazy numpy import, mirroring FactLedger's pattern. Returns the
    module or None when unavailable (layer degrades to cold no-op)."""
    try:
        import numpy  # noqa: WPS433
        return numpy
    except Exception:
        return None


class EdgeFilter:
    """One Perpetua*-style Bayesian filter per cyclic edge kind.

    State: per (key, bucket) success/fail counts under a Beta(1, 1)
    prior over the edge's truth probability. ``update`` is the
    conjugate posterior bump; ``predict`` returns the posterior mean
    ``(s + 1) / (s + f + 2)`` — exactly 0.5 when unobserved.

    ``key`` is any hashable tuple identifying the edge instance
    (v1 registers ``("npc_beat", npc_id)`` edges: "does this NPC
    receive a Director beat at this tick phase?"). Unknown edge
    kinds are ignored by the layer, never crashed on — new kinds
    need an explicit ``PredictiveLayer.register_edge_kind`` call.
    """

    def __init__(self, edge_kind: str, bucket_count: int = _EDGE_BUCKET_CYCLE):
        self.edge_kind = edge_kind
        self.bucket_count = max(1, int(bucket_count))
        # {(key_tuple, bucket): [success, fail]}
        self.counts: dict[tuple, list[int]] = {}

    def update(self, key: tuple, bucket: int, observed: bool) -> None:
        bucket = int(bucket) % self.bucket_count
        cell = self.counts.setdefault((tuple(key), bucket), [0, 0])
        cell[0 if observed else 1] += 1

    def predict(self, key: tuple, bucket: int) -> float:
        bucket = int(bucket) % self.bucket_count
        s, f = self.counts.get((tuple(key), bucket), (0, 0))
        return (s + 1) / (s + f + 2)

    def known_keys(self) -> set[tuple]:
        return {key for (key, _bucket) in self.counts}

    def reset(self) -> None:
        self.counts.clear()

    # ── (De)serialization — JSON-string payload for the npz ────────

    def to_json(self) -> str:
        flat = [
            [list(key), bucket, s, f]
            for (key, bucket), (s, f) in self.counts.items()
        ]
        return json.dumps({
            "edge_kind": self.edge_kind,
            "bucket_count": self.bucket_count,
            "counts": flat,
        })

    @classmethod
    def from_json(cls, payload: str) -> "EdgeFilter":
        data = json.loads(payload)
        filt = cls(str(data["edge_kind"]), int(data["bucket_count"]))
        for key_list, bucket, s, f in data.get("counts", []):
            filt.counts[(tuple(key_list), int(bucket))] = [int(s), int(f)]
        return filt


@dataclass
class ActivityPrediction:
    """One prior-branch rollup, surfaced on /story/state.

    ``cold`` means the prior had no fitted weight matrix (or no
    usable latent) and the distribution is uniform — consumers must
    skip the drift check on cold predictions.
    """

    distribution: dict[str, float]
    top_activity: str
    confidence: float
    cold: bool
    tick: int
    drift: bool = False
    reported_activity: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "distribution": {k: round(v, 4) for k, v in self.distribution.items()},
            "top_activity": self.top_activity,
            "confidence": round(self.confidence, 4),
            "cold": self.cold,
            "tick": self.tick,
            "drift": self.drift,
            "reported_activity": self.reported_activity,
        }


class ActivityPrior:
    """Being-H0.7-style prior projection: recent-ledger summary latent
    -> logits over the PlayerActivity labels via ONE linear matrix.

    ``fit`` solves ridge regression against one-hot labels in closed
    form (pure NumPy, milliseconds at any realistic history size);
    ``predict`` is a matmul + softmax. No weight matrix = uniform.
    """

    def __init__(self, labels: list[str]):
        self.labels: list[str] = list(labels)
        self.weight_matrix = None  # np.ndarray (latent_dim + 1, n_labels) or None

    @property
    def is_fitted(self) -> bool:
        return self.weight_matrix is not None

    def uniform(self) -> dict[str, float]:
        p = 1.0 / max(1, len(self.labels))
        return {label: p for label in self.labels}

    def fit(self, history: list[tuple], min_samples: int = _PRIOR_MIN_FIT_SAMPLES) -> bool:
        """Fit W from (latent_vector, activity_label) pairs. Returns
        True when a matrix was produced; False leaves any existing
        matrix untouched (never downgrade a warm prior to cold)."""
        np = _np()
        if np is None:
            return False
        usable = [
            (vec, label) for vec, label in history
            if vec is not None and label in self.labels
        ]
        if len(usable) < max(2, int(min_samples)):
            return False
        distinct = {label for _vec, label in usable}
        if len(distinct) < 2:
            # One-class history — the "prior" would be a constant;
            # uniform + cold is more honest than false certainty.
            return False
        try:
            X = np.asarray([vec for vec, _ in usable], dtype=np.float64)
            # Bias column so the prior can encode base rates.
            X = np.hstack([X, np.ones((X.shape[0], 1))])
            Y = np.zeros((X.shape[0], len(self.labels)), dtype=np.float64)
            index = {label: i for i, label in enumerate(self.labels)}
            for row, (_vec, label) in enumerate(usable):
                Y[row, index[label]] = 1.0
            lam = _PRIOR_RIDGE_LAMBDA
            A = X.T @ X + lam * np.eye(X.shape[1])
            self.weight_matrix = np.linalg.solve(A, X.T @ Y)
            return True
        except Exception as e:
            logger.warning(f"ActivityPrior fit failed: {e}")
            return False

    def predict(self, latent) -> dict[str, float]:
        np = _np()
        if np is None or self.weight_matrix is None or latent is None:
            return self.uniform()
        try:
            x = np.asarray(latent, dtype=np.float64).ravel()
            x = np.concatenate([x, [1.0]])
            if x.shape[0] != self.weight_matrix.shape[0]:
                # Latent dim changed (e.g. different embedder) —
                # stale matrix, fall back to uniform.
                return self.uniform()
            # Ridge against one-hot targets makes the raw scores an
            # approximation of class probabilities already, so the
            # link is clip-to-positive + normalize. (A softmax would
            # squash a well-separated ~one-hot score vector to a
            # ~0.25 peak over 9 labels and the drift-confidence
            # threshold could never fire.)
            scores = np.clip(x @ self.weight_matrix, 0.0, None)
            total = float(scores.sum())
            if total <= 1e-9:
                return self.uniform()
            probs = scores / total
            return {label: float(p) for label, p in zip(self.labels, probs)}
        except Exception as e:
            logger.warning(f"ActivityPrior predict failed: {e}")
            return self.uniform()


class PredictiveLayer:
    """Owns the edge filters and the activity prior. One instance per
    StoryDirector. Hot path is < 1 ms — pure linalg, no I/O.

    v1 registers a single edge kind, ``npc_beat`` — "does NPC X
    receive a Director beat at tick phase B?" — which feeds the
    per-NPC arc-proposal boost. Additional cyclic kinds
    (npc_in_zone, faction_presence) plug in via
    ``register_edge_kind`` once their observation streams exist.
    """

    def __init__(self, storage_path: Path, labels: list[str],
                 history_path: Optional[Path] = None):
        self.storage_path = Path(storage_path)
        self.history_path = Path(history_path) if history_path else None
        self.prior = ActivityPrior(labels)
        self.edge_filters: dict[str, EdgeFilter] = {}
        self.register_edge_kind("npc_beat")
        # In-memory (latent, label) supervision pairs harvested from
        # activity posts this session; warm_from_history re-derives
        # the full set from the JSONL + ledger.
        self._activity_pairs: list[tuple] = []
        self._last_prediction: Optional[ActivityPrediction] = None
        self._drift_count: int = 0
        self._observation_count: int = 0
        self._loaded_from_sidecar: bool = False
        self._load()

    # ── Edge kinds ─────────────────────────────────────────────────

    def register_edge_kind(self, edge_kind: str,
                           bucket_count: int = _EDGE_BUCKET_CYCLE) -> EdgeFilter:
        filt = self.edge_filters.get(edge_kind)
        if filt is None:
            filt = EdgeFilter(edge_kind, bucket_count)
            self.edge_filters[edge_kind] = filt
        return filt

    # ── Latents ────────────────────────────────────────────────────

    @staticmethod
    def summary_latent(entries: list[dict], window: int = _PRIOR_RECENT_WINDOW):
        """Mean of the last ``window`` ledger-entry embeddings. Returns
        None when no embeddings are available (→ cold prediction)."""
        np = _np()
        if np is None:
            return None
        vecs = [
            e.get("embedding") for e in entries[-window:]
            if e.get("embedding") is not None
        ]
        if not vecs:
            return None
        try:
            return np.asarray(vecs, dtype=np.float64).mean(axis=0)
        except Exception:
            return None

    # ── Hot path ───────────────────────────────────────────────────

    def predict_next(self, ledger, tick: int,
                     current_activity: str) -> tuple[ActivityPrediction, dict[str, float]]:
        """Run the prior branch. Returns the activity prediction plus
        per-NPC edge priors for the NEXT tick's phase bucket.

        Sets ``drift=True`` (and bumps the drift counter) when the
        fitted prior confidently disagrees with the game-reported
        activity — v1 observes only; no behavior change hangs off it.
        """
        latent = self.summary_latent(getattr(ledger, "entries", []))
        cold = not self.prior.is_fitted or latent is None
        if cold:
            distribution = self.prior.uniform()
        else:
            distribution = self.prior.predict(latent)
        top_activity = max(distribution, key=distribution.get) if distribution else ""
        confidence = distribution.get(top_activity, 0.0)

        prediction = ActivityPrediction(
            distribution=distribution,
            top_activity=top_activity,
            confidence=confidence,
            cold=cold,
            tick=int(tick),
            reported_activity=current_activity,
        )
        if (not cold
                and confidence >= _DRIFT_CONFIDENCE_THRESHOLD
                and top_activity != current_activity):
            prediction.drift = True
            self._drift_count += 1
            logger.debug(
                f"predicted_drift: prior says '{top_activity}' "
                f"(p={confidence:.2f}) but client reports "
                f"'{current_activity}' at tick {tick}"
            )

        next_bucket = (int(tick) + 1) % _EDGE_BUCKET_CYCLE
        edge_priors: dict[str, float] = {}
        beat_filter = self.edge_filters.get("npc_beat")
        if beat_filter is not None:
            for key in beat_filter.known_keys():
                npc_id = key[0] if key else None
                if isinstance(npc_id, str):
                    edge_priors[npc_id] = beat_filter.predict(key, next_bucket)

        self._last_prediction = prediction
        return prediction, edge_priors

    # ── Observation ────────────────────────────────────────────────

    def record_observation(self, tick: int, activity: str,
                           ledger_delta: list[dict]) -> None:
        """Feed one completed tick into the edge filters. NPCs that
        received a beat this tick get a success at this tick's phase
        bucket; every other already-known NPC gets a fail — that
        contrast is what makes the posterior cyclic rather than a
        plain popularity count."""
        try:
            bucket = int(tick) % _EDGE_BUCKET_CYCLE
            beat_filter = self.register_edge_kind("npc_beat")
            touched: set[str] = set()
            for entry in ledger_delta or []:
                npc_id = entry.get("npc_id")
                if isinstance(npc_id, str) and npc_id not in ("all", "*", "?"):
                    touched.add(npc_id)
            for npc_id in touched:
                beat_filter.update((npc_id,), bucket, True)
            for key in beat_filter.known_keys():
                npc_id = key[0] if key else None
                if isinstance(npc_id, str) and npc_id not in touched:
                    beat_filter.update(key, bucket, False)
            self._observation_count += 1
        except Exception as e:
            logger.warning(f"PredictiveLayer record_observation failed: {e}")

    def note_activity(self, tick: int, activity: str, ledger) -> None:
        """Harvest one supervision pair from a /story/activity post:
        (summary latent of the ledger as it stood when the activity
        was reported, reported activity). Refit is attempted on every
        harvest — the closed-form solve is milliseconds and only
        succeeds once the sample/diversity floor is met."""
        try:
            latent = self.summary_latent(getattr(ledger, "entries", []))
            if latent is not None:
                self._activity_pairs.append((latent, str(activity)))
                if len(self._activity_pairs) > _MAX_ACTIVITY_PAIRS:
                    self._activity_pairs = self._activity_pairs[-_MAX_ACTIVITY_PAIRS:]
                self.prior.fit(self._activity_pairs)
        except Exception as e:
            logger.warning(f"PredictiveLayer note_activity failed: {e}")

    # ── Warm start ─────────────────────────────────────────────────

    def warm_from_history(self, ledger) -> dict:
        """Replay the activity-history JSONL + the existing ledger to
        train both devices. The only 'training' step in v1 — CPU,
        seconds at any realistic ledger size.

        Returns a small report dict for logging/observability.
        """
        report = {"history_records": 0, "pairs": 0, "fitted": False,
                  "edge_updates": 0, "pruned": 0}
        # 1. Edge filters from the full ledger stream: group entries
        # by tick; present NPCs get a success at that tick's bucket,
        # known-but-absent NPCs get a fail.
        entries = list(getattr(ledger, "entries", []))
        by_tick: dict[int, list[dict]] = {}
        for e in entries:
            t = e.get("tick")
            if isinstance(t, (int, float)):
                by_tick.setdefault(int(t), []).append(e)
        for t in sorted(by_tick):
            self.record_observation(t, "", by_tick[t])
            report["edge_updates"] += 1

        # 2. Activity prior from the JSONL. Latent per record is
        # re-derived from the ledger as it stood at that record's
        # tick (entries with tick <= record tick).
        if self.history_path is None or not self.history_path.exists():
            return report
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=_ACTIVITY_HISTORY_MAX_AGE_DAYS)
        kept_lines: list[str] = []
        records: list[dict] = []
        try:
            raw_lines = self.history_path.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            logger.warning(f"PredictiveLayer history read failed: {e}")
            return report
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ts = rec.get("ts")
            try:
                stale = (ts is not None
                         and datetime.fromisoformat(str(ts)) < cutoff)
            except Exception:
                stale = False
            if stale:
                report["pruned"] += 1
                continue
            kept_lines.append(line)
            records.append(rec)
        if report["pruned"]:
            try:
                self.history_path.write_text(
                    "\n".join(kept_lines) + ("\n" if kept_lines else ""),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.warning(f"PredictiveLayer history prune failed: {e}")
        report["history_records"] = len(records)

        pairs: list[tuple] = []
        for rec in records:
            activity = rec.get("activity")
            tick = rec.get("tick")
            if not isinstance(activity, str) or not isinstance(tick, (int, float)):
                continue
            visible = [e for e in entries
                       if isinstance(e.get("tick"), (int, float))
                       and e["tick"] <= int(tick)]
            latent = self.summary_latent(visible)
            if latent is not None:
                pairs.append((latent, activity))
        report["pairs"] = len(pairs)
        if pairs:
            self._activity_pairs = (pairs + self._activity_pairs)[-_MAX_ACTIVITY_PAIRS:]
            report["fitted"] = self.prior.fit(self._activity_pairs)
        return report

    # ── Lifecycle ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Game reset: zero the edge filters but KEEP the prior matrix
        — the prior generalizes across resets, the edge filters don't
        (plan: composition table, game-reset row).

        The activity-history JSONL also survives reset while the tick
        counter restarts at 0. A later sidecar-less warm replay would
        therefore pair pre-reset activity labels with post-reset ledger
        latents for overlapping tick numbers — mildly mislabeled
        supervision. Accepted for v1: the sidecar (saved right here)
        normally short-circuits the replay, and the prior it carries
        was fitted on correctly-paired data."""
        for filt in self.edge_filters.values():
            filt.reset()
        self._last_prediction = None
        self._drift_count = 0
        self._observation_count = 0
        self.save()

    def stats(self) -> dict:
        """Compact rollup for /story/state."""
        return {
            "enabled": True,
            "prior_fitted": self.prior.is_fitted,
            "activity_pairs": len(self._activity_pairs),
            "edge_filters": {
                kind: len(filt.counts)
                for kind, filt in self.edge_filters.items()
            },
            "tracked_npcs": len(
                self.edge_filters["npc_beat"].known_keys()
            ) if "npc_beat" in self.edge_filters else 0,
            "observation_count": self._observation_count,
            "drift_count": self._drift_count,
            "loaded_from_sidecar": self._loaded_from_sidecar,
            "last_prediction": (
                self._last_prediction.to_dict()
                if self._last_prediction is not None else None
            ),
        }

    # ── Sidecar persistence ────────────────────────────────────────

    def save(self) -> None:
        """Write the .npz sidecar. Re-derivable from the JSONL +
        ledger if deleted — purely a startup-cost optimization."""
        np = _np()
        if np is None:
            return
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict = {
                "version": np.asarray(_SIDECAR_VERSION),
                "labels": np.asarray(self.prior.labels, dtype=object),
                "edge_filters": np.asarray(
                    [filt.to_json() for filt in self.edge_filters.values()],
                    dtype=object,
                ),
            }
            if self.prior.weight_matrix is not None:
                payload["prior_weight_matrix"] = self.prior.weight_matrix
            np.savez(self.storage_path, **payload)
        except Exception as e:
            logger.warning(f"PredictiveLayer save failed: {e}")

    def _load(self) -> None:
        np = _np()
        if np is None or not self.storage_path.exists():
            return
        try:
            with np.load(self.storage_path, allow_pickle=True) as data:
                labels = [str(x) for x in data["labels"].tolist()]
                if labels == self.prior.labels and "prior_weight_matrix" in data:
                    self.prior.weight_matrix = data["prior_weight_matrix"]
                elif "prior_weight_matrix" in data:
                    logger.warning(
                        "PredictiveLayer sidecar label set changed; "
                        "discarding stale prior matrix"
                    )
                for payload in data["edge_filters"].tolist():
                    try:
                        filt = EdgeFilter.from_json(str(payload))
                        self.edge_filters[filt.edge_kind] = filt
                    except Exception:
                        continue
            self._loaded_from_sidecar = True
        except Exception as e:
            logger.warning(f"PredictiveLayer sidecar load failed: {e}")
