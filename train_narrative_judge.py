"""
Train the production Narrative Judge LR model on combined
Ashenvale + Port Blackwater labels (216 tuples).

Pickles (scaler, model) to data/narrative_judge.pkl.
This is the artifact npc_engine/narrative_judge.py loads at runtime.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from narrative_judge_dataset import DATASET, QUESTS
from narrative_judge_dataset_pb import DATASET_PB, QUESTS_PB
from narrative_judge_trainer import (
    load_nli, load_embedder, featurize, CLASS_LABELS,
)

ALL_QUESTS = {**QUESTS, **QUESTS_PB}

# Schema version — bump if feature extractor or class labels change.
SCHEMA_VERSION = 1
FEATURE_NAMES = ["nli_con", "nli_ent", "nli_neu", "cos_sim", "tok_over",
                 "len_norm", "ent*cos", "con*cos"]


def main():
    nli = load_nli()
    embedder = load_embedder()
    print()

    # Combine datasets
    combined = DATASET + DATASET_PB
    print(f"Training on {len(combined)} tuples "
          f"({len(DATASET)} Ashenvale + {len(DATASET_PB)} Port Blackwater).")

    # Featurize
    embed_cache: dict[str, np.ndarray] = {}
    X = []
    y = []
    t0 = time.time()
    for i, (fact, qid, label) in enumerate(combined):
        feat = featurize(nli, embedder, fact, ALL_QUESTS[qid], embed_cache)
        X.append(feat)
        y.append(CLASS_LABELS.index(label))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(combined)} featurized ({time.time()-t0:.0f}s)")
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"Featurization done in {time.time()-t0:.0f}s.")

    # Train on full dataset (no holdout — we already validated 89.7% / 70-80%
    # cross-domain in narrative_judge_cross_domain.py).
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)

    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                            random_state=42)
    lr.fit(X_s, y)
    train_acc = lr.score(X_s, y)
    print(f"Train accuracy on combined dataset: {train_acc:.3f}")

    # Pickle (scaler, model) plus metadata
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "class_labels": list(CLASS_LABELS),
        "scaler": scaler,
        "model": lr,
        "n_train": len(combined),
        "train_acc": float(train_acc),
        "trained_on": "Ashenvale (144) + Port Blackwater (72)",
    }

    out_path = Path(__file__).parent / "data" / "narrative_judge.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nWrote {out_path}")
    print(f"  Size: {out_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
