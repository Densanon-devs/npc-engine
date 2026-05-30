"""
Narrative Judge — trained-head feasibility test.

Architecture (mirrors GE-Sim 2.0's World Judge):
  Frozen backbone   = MiniLM sentence-embedding model
                    + cross-encoder/nli-deberta-v3-small NLI scores
  Trainable head    = sklearn LogisticRegression OR MLPClassifier
  Loss              = cross-entropy (3-class: advance / block / neutral)
  Supervision       = 144 hand-labeled (fact, quest_objective, label) tuples
                      from narrative_judge_dataset.py

Per-pair feature vector (8 dims):
  0  NLI(contradiction)
  1  NLI(entailment)
  2  NLI(neutral)
  3  cosine_sim(fact_emb, quest_emb)
  4  token_overlap_fraction(quest -> fact)
  5  fact_length_normalized
  6  NLI(entailment) * cosine_sim     (interaction)
  7  NLI(contradiction) * cosine_sim  (interaction)

Two evaluation passes:
  1. 80/20 stratified holdout — accuracy, confusion matrix, per-class F1
  2. 50-tick replay (from narrative_judge_prototype.py) re-scored with
     the trained head; separability gap per quest

Verdict thresholds (same as Part B):
  gap > 0.20   -> Narrative Judge is viable
  0.05-0.20    -> marginal
  < 0.05       -> doesn't transfer
"""
from __future__ import annotations

import math
import re
import sys
import time
from typing import Tuple

import numpy as np

from narrative_judge_dataset import DATASET, QUESTS

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LABELS = ("contradiction", "entailment", "neutral")
CLASS_LABELS = ("advance", "block", "neutral")

# ─────────────────────────────────────────────────────────────────
# Backbones
# ─────────────────────────────────────────────────────────────────

def load_nli():
    from sentence_transformers import CrossEncoder
    print(f"Loading NLI: {NLI_MODEL_NAME}...")
    m = CrossEncoder(NLI_MODEL_NAME)
    return m


def load_embedder():
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedder: {EMBED_MODEL_NAME}...")
    m = SentenceTransformer(EMBED_MODEL_NAME)
    return m


def softmax(x):
    mx = max(x)
    exps = [math.exp(s - mx) for s in x]
    total = sum(exps)
    return [e / total for e in exps]


def nli_scores(nli, premise: str, hypothesis: str) -> dict:
    raw = nli.predict([(premise, hypothesis)])[0]
    probs = softmax([float(s) for s in raw])
    return {LABELS[i]: probs[i] for i in range(len(LABELS))}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


_WORD_RE = re.compile(r"[a-zA-Z']+")
_STOPWORDS = {"the", "a", "an", "of", "to", "in", "and", "or", "but",
              "is", "was", "are", "were", "be", "been", "being", "has",
              "have", "had", "do", "does", "did", "for", "on", "at", "by",
              "with", "from", "up", "about", "into", "over", "after",
              "that", "this", "these", "those", "it", "its", "as", "if",
              "than", "so", "no", "not", "very", "just", "all", "any",
              "each", "every", "some", "who", "whom", "what", "which",
              "where", "when", "why", "how", "player"}


def token_overlap(hypothesis: str, premise: str) -> float:
    """Fraction of (non-stopword) hypothesis tokens appearing in premise."""
    htoks = {w.lower() for w in _WORD_RE.findall(hypothesis) if w.lower() not in _STOPWORDS}
    if not htoks:
        return 0.0
    ptoks = {w.lower() for w in _WORD_RE.findall(premise) if w.lower() not in _STOPWORDS}
    return len(htoks & ptoks) / len(htoks)


# ─────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────

def featurize(nli, embedder, fact: str, quest_objective: str,
              embed_cache: dict | None = None) -> np.ndarray:
    """Compute 8-dim feature vector for a (fact, quest_objective) pair."""
    if embed_cache is None:
        embed_cache = {}

    def cached_embed(text: str) -> np.ndarray:
        if text not in embed_cache:
            embed_cache[text] = np.asarray(embedder.encode(text), dtype=np.float32)
        return embed_cache[text]

    nli_s = nli_scores(nli, fact, quest_objective)
    cos = cosine(cached_embed(fact), cached_embed(quest_objective))
    overlap = token_overlap(quest_objective, fact)
    length_norm = min(len(fact) / 200.0, 1.5)
    return np.array([
        nli_s["contradiction"],
        nli_s["entailment"],
        nli_s["neutral"],
        cos,
        overlap,
        length_norm,
        nli_s["entailment"] * cos,
        nli_s["contradiction"] * cos,
    ], dtype=np.float32)


def build_feature_matrix(nli, embedder) -> tuple[np.ndarray, np.ndarray, list]:
    """Featurize entire DATASET. Returns (X, y, raw_rows)."""
    print(f"Featurizing {len(DATASET)} tuples (this may take a few minutes — NLI on CPU)...")
    embed_cache: dict[str, np.ndarray] = {}
    X = []
    y = []
    raw = []
    t0 = time.time()
    for i, (fact, qid, label) in enumerate(DATASET):
        quest_obj = QUESTS[qid]
        feat = featurize(nli, embedder, fact, quest_obj, embed_cache)
        X.append(feat)
        y.append(CLASS_LABELS.index(label))
        raw.append((fact, qid, label, feat))
        if (i + 1) % 30 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(DATASET) - i - 1) / rate
            print(f"  {i+1}/{len(DATASET)} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
    elapsed = time.time() - t0
    print(f"Featurization done in {elapsed:.0f}s. {len(embed_cache)} distinct strings embedded.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), raw


# ─────────────────────────────────────────────────────────────────
# Eval helpers
# ─────────────────────────────────────────────────────────────────

def confusion_matrix(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_confusion(cm, title=""):
    if title:
        print(title)
    print(f"  {'':>10} " + " ".join(f"{c:>10}" for c in CLASS_LABELS) + "  (predicted)")
    for i, c in enumerate(CLASS_LABELS):
        print(f"  {c:>10} " + " ".join(f"{cm[i, j]:>10}" for j in range(3)))
    print()


def macro_f1(cm):
    """3-class macro F1 from confusion matrix."""
    f1s = []
    for i in range(3):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return f1s, sum(f1s) / len(f1s)


# ─────────────────────────────────────────────────────────────────
# Part B — 50-tick replay re-scoring (re-use entries from prototype)
# ─────────────────────────────────────────────────────────────────

def part_b_rescore(nli, embedder, model, scaler=None):
    """Rerun the 50-tick replay using the trained model.

    For each replay entry, compute features against each of the 4 quests.
    The predicted advance-probability (class 0) is the new "best-match"
    score. Report separability gap per quest (same metric as Part B in
    narrative_judge_prototype.py).
    """
    from narrative_judge_prototype import REPLAY_ENTRIES, ACTIVE_QUESTS
    EXPECT_PREFIX = {
        "missing_hammers": "hammers",
        "counterfeit_steel": "steel",
        "silverwood_ruins": "silverwood",
        "wolf_bounty": "wolf",
    }

    print(f"Re-scoring 50-tick replay with trained head...")
    embed_cache: dict[str, np.ndarray] = {}
    per_tick = []
    for tick, (text, tag) in enumerate(REPLAY_ENTRIES, 1):
        quest_advance_scores = {}
        for q in ACTIVE_QUESTS:
            feat = featurize(nli, embedder, text, q["objective"], embed_cache)
            X = feat.reshape(1, -1)
            if scaler is not None:
                X = scaler.transform(X)
            proba = model.predict_proba(X)[0]
            # advance is class index 0 (per CLASS_LABELS order)
            quest_advance_scores[q["id"]] = float(proba[0])
        best_qid = max(quest_advance_scores.keys(),
                       key=lambda k: quest_advance_scores[k])
        per_tick.append({
            "tick": tick,
            "tag": tag,
            "text": text,
            "best_quest": best_qid,
            "best_advance_prob": quest_advance_scores[best_qid],
            "scores": quest_advance_scores,
        })

    print()
    print("Per-quest separability (trained-head Part B):")
    print()
    for q in ACTIVE_QUESTS:
        qid = q["id"]
        expect_prefix = EXPECT_PREFIX[qid]
        scores = [(t["tick"], t["tag"], t["scores"][qid]) for t in per_tick]
        scores.sort(key=lambda x: -x[2])
        print(f"Quest [{qid}]:  {q['objective']}")
        print("  Top 5 advance-prob ticks:")
        for tick, tag, p in scores[:5]:
            on = "ON " if expect_prefix in tag else "off"
            print(f"    T{tick:>2} p_advance={p:.2f} [{on}] {tag:>22}")
        on_topic = [p for _, tag, p in scores
                    if expect_prefix in tag and "contradiction" not in tag]
        off_topic = [p for _, tag, p in scores if expect_prefix not in tag]
        if on_topic and off_topic:
            on_min, on_max = min(on_topic), max(on_topic)
            off_max = max(off_topic)
            gap = on_min - off_max
            print(f"  Separability: on-topic [{on_min:.2f}, {on_max:.2f}]  "
                  f"vs off-topic max {off_max:.2f}  -> gap={gap:+.2f}")
        print()
    return per_tick


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    # 1. Load backbones
    nli = load_nli()
    embedder = load_embedder()
    print()

    # 2. Featurize the labeled dataset
    X, y, raw = build_feature_matrix(nli, embedder)
    print()

    # 3. Stratified 80/20 split
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42,
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    print()

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. Train baselines
    # 4a. Logistic regression
    lr = LogisticRegression(max_iter=2000, multi_class="multinomial",
                            class_weight="balanced", random_state=42)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    acc_lr = (y_pred_lr == y_test).mean()
    f1s_lr, mf1_lr = macro_f1(cm_lr)

    print("=" * 72)
    print("Logistic Regression (8-dim feature, class_weight=balanced)")
    print("=" * 72)
    print(f"Held-out accuracy: {acc_lr:.3f}  ({(y_pred_lr == y_test).sum()}/{len(y_test)})")
    print(f"Macro-F1: {mf1_lr:.3f}  per-class F1: " +
          " ".join(f"{c}={f:.2f}" for c, f in zip(CLASS_LABELS, f1s_lr)))
    print_confusion(cm_lr)
    # Feature importance (coefficients)
    print("Per-class coefficients (positive = pushes prediction toward class):")
    feat_names = ["nli_con", "nli_ent", "nli_neu", "cos_sim", "tok_over",
                  "len_norm", "ent*cos", "con*cos"]
    print(f"  {'feature':>10} " + " ".join(f"{c:>10}" for c in CLASS_LABELS))
    for i, fn in enumerate(feat_names):
        print(f"  {fn:>10} " + " ".join(f"{lr.coef_[c, i]:>+10.3f}" for c in range(3)))
    print()

    # 4b. MLP
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000,
                        activation="relu", random_state=42)
    mlp.fit(X_train_s, y_train)
    y_pred_mlp = mlp.predict(X_test_s)
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
    acc_mlp = (y_pred_mlp == y_test).mean()
    f1s_mlp, mf1_mlp = macro_f1(cm_mlp)

    print("=" * 72)
    print("MLP (hidden=(32,16), relu)")
    print("=" * 72)
    print(f"Held-out accuracy: {acc_mlp:.3f}  ({(y_pred_mlp == y_test).sum()}/{len(y_test)})")
    print(f"Macro-F1: {mf1_mlp:.3f}  per-class F1: " +
          " ".join(f"{c}={f:.2f}" for c, f in zip(CLASS_LABELS, f1s_mlp)))
    print_confusion(cm_mlp)

    # 4c. Raw NLI baseline (for comparison — Part A from prototype)
    # Map NLI label-of-max-prob -> direction: entailment->advance,
    # contradiction->block, neutral->neutral.
    print("=" * 72)
    print("Raw-NLI baseline (no training — same as Part A in prototype)")
    print("=" * 72)
    label_to_class = {"entailment": 0, "contradiction": 1, "neutral": 2}
    y_pred_nli = []
    for i in range(len(X)):
        nli_con, nli_ent, nli_neu = X[i, 0], X[i, 1], X[i, 2]
        top = max(enumerate([nli_con, nli_ent, nli_neu]), key=lambda t: t[1])
        nli_lab_name = ["contradiction", "entailment", "neutral"][top[0]]
        y_pred_nli.append(label_to_class[nli_lab_name])
    y_pred_nli = np.array(y_pred_nli)
    # Evaluate on the test-set indices only for fair comparison
    _, test_idx = train_test_split(np.arange(len(X)), test_size=0.20,
                                    stratify=y, random_state=42)
    acc_nli = (y_pred_nli[test_idx] == y_test).mean()
    cm_nli = confusion_matrix(y_test, y_pred_nli[test_idx])
    f1s_nli, mf1_nli = macro_f1(cm_nli)
    print(f"Held-out accuracy: {acc_nli:.3f}  ({(y_pred_nli[test_idx] == y_test).sum()}/{len(y_test)})")
    print(f"Macro-F1: {mf1_nli:.3f}  per-class F1: " +
          " ".join(f"{c}={f:.2f}" for c, f in zip(CLASS_LABELS, f1s_nli)))
    print_confusion(cm_nli)

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Raw NLI:             acc={acc_nli:.3f}  macro-F1={mf1_nli:.3f}")
    print(f"  Logistic Regression: acc={acc_lr:.3f}  macro-F1={mf1_lr:.3f}")
    print(f"  MLP (32,16):         acc={acc_mlp:.3f}  macro-F1={mf1_mlp:.3f}")
    print(f"  Lift over raw NLI:   LR=+{acc_lr-acc_nli:.3f}  MLP=+{acc_mlp-acc_nli:.3f}")
    print()

    # 5. Re-score Part B replay with the better of (LR, MLP)
    best_name, best_model = ("MLP", mlp) if acc_mlp > acc_lr else ("LR", lr)
    print(f"Using {best_name} for Part B replay re-score...\n")
    part_b_rescore(nli, embedder, best_model, scaler=scaler)

    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  - acc >= 0.70 AND Part B gap >= 0.20 -> Narrative Judge viable")
    print(f"  - 0.50-0.70 / gap 0.05-0.20         -> marginal, threshold-tunable")
    print(f"  - < 0.50 / gap <= 0.05              -> doesn't transfer; reject")
    print()


if __name__ == "__main__":
    main()
