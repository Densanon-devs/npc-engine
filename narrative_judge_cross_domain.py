"""
Narrative Judge — cross-domain validation.

Tests whether the trained head generalizes from Ashenvale (144 labeled
tuples) to Port Blackwater (72 labeled tuples, pirate port, different
NPCs and quest shapes).

Three configurations:
  1. Zero-shot A -> PB:  train on Ashenvale only, test on Port Blackwater
  2. Zero-shot PB -> A:  train on Port Blackwater only, test on Ashenvale
                         (symmetric check — is one domain just harder?)
  3. Few-shot PB:        train on Ashenvale + 80% of PB, test on 20% PB
                         (the "small labeled set in new domain" scenario)

Reuses the feature extractor from narrative_judge_trainer.py.

Verdict:
  Zero-shot acc >= 0.70 -> Narrative Judge generalizes across worlds
  Few-shot acc >= 0.80  -> small new-world labels close any gap
  Both lower            -> needs per-world training data
"""
from __future__ import annotations

import time
import numpy as np

from narrative_judge_dataset import DATASET, QUESTS
from narrative_judge_dataset_pb import DATASET_PB, QUESTS_PB
from narrative_judge_trainer import (
    load_nli, load_embedder, featurize, confusion_matrix,
    print_confusion, macro_f1, CLASS_LABELS,
)

ALL_QUESTS = {**QUESTS, **QUESTS_PB}


def featurize_dataset(nli, embedder, dataset, all_quests):
    """Featurize a dataset using cached embeddings.

    dataset: list of (fact, quest_id, label)
    all_quests: dict mapping quest_id -> objective text
    """
    embed_cache: dict[str, np.ndarray] = {}
    X = []
    y = []
    t0 = time.time()
    for i, (fact, qid, label) in enumerate(dataset):
        quest_obj = all_quests[qid]
        feat = featurize(nli, embedder, fact, quest_obj, embed_cache)
        X.append(feat)
        y.append(CLASS_LABELS.index(label))
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(dataset)} featurized "
                  f"({time.time()-t0:.0f}s elapsed)")
    print(f"  Featurization done in {time.time()-t0:.0f}s.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_and_eval(X_train, y_train, X_test, y_test, label: str):
    """Train LR + MLP, return (lr_metrics, mlp_metrics)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                            random_state=42)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    acc_lr = (y_pred_lr == y_test).mean()
    f1s_lr, mf1_lr = macro_f1(cm_lr)

    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000,
                        activation="relu", random_state=42)
    mlp.fit(X_train_s, y_train)
    y_pred_mlp = mlp.predict(X_test_s)
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
    acc_mlp = (y_pred_mlp == y_test).mean()
    f1s_mlp, mf1_mlp = macro_f1(cm_mlp)

    print(f"\n{'-' * 60}")
    print(f"{label}")
    print(f"{'-' * 60}")
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"  Logistic Regression: acc={acc_lr:.3f}  macro-F1={mf1_lr:.3f}  "
          f"per-class F1: " + " ".join(f"{c}={f:.2f}" for c, f in zip(CLASS_LABELS, f1s_lr)))
    print(f"  MLP (32,16):         acc={acc_mlp:.3f}  macro-F1={mf1_mlp:.3f}  "
          f"per-class F1: " + " ".join(f"{c}={f:.2f}" for c, f in zip(CLASS_LABELS, f1s_mlp)))
    print()
    print_confusion(cm_lr, "  Logistic Regression confusion:")
    print_confusion(cm_mlp, "  MLP confusion:")

    return {
        "lr": {"acc": acc_lr, "mf1": mf1_lr, "cm": cm_lr},
        "mlp": {"acc": acc_mlp, "mf1": mf1_mlp, "cm": cm_mlp},
    }


def main():
    nli = load_nli()
    embedder = load_embedder()
    print()

    # Featurize both datasets once (cached embeddings amortize across configs)
    print("Featurizing Ashenvale (144 tuples)...")
    X_ash, y_ash = featurize_dataset(nli, embedder, DATASET, ALL_QUESTS)
    print()
    print("Featurizing Port Blackwater (72 tuples)...")
    X_pb, y_pb = featurize_dataset(nli, embedder, DATASET_PB, ALL_QUESTS)
    print()

    # Config 1: zero-shot Ashenvale -> Port Blackwater
    print("=" * 72)
    print("CONFIG 1: Zero-shot transfer  Ashenvale -> Port Blackwater")
    print("=" * 72)
    r1 = train_and_eval(X_ash, y_ash, X_pb, y_pb,
                        "Train: Ashenvale (144).  Test: PB (72).")

    # Config 2: zero-shot Port Blackwater -> Ashenvale
    print("=" * 72)
    print("CONFIG 2: Zero-shot transfer  Port Blackwater -> Ashenvale")
    print("=" * 72)
    r2 = train_and_eval(X_pb, y_pb, X_ash, y_ash,
                        "Train: PB (72).  Test: Ashenvale (144).")

    # Config 3: combined Ashenvale + 80% PB -> test 20% PB held-out
    from sklearn.model_selection import train_test_split
    X_pb_train, X_pb_test, y_pb_train, y_pb_test = train_test_split(
        X_pb, y_pb, test_size=0.20, stratify=y_pb, random_state=42,
    )
    X_combined = np.vstack([X_ash, X_pb_train])
    y_combined = np.concatenate([y_ash, y_pb_train])

    print("=" * 72)
    print("CONFIG 3: Few-shot  Ashenvale + 80% PB  ->  20% PB held-out")
    print("=" * 72)
    r3 = train_and_eval(X_combined, y_combined, X_pb_test, y_pb_test,
                        f"Train: Ashenvale (144) + PB-train ({len(X_pb_train)}).  "
                        f"Test: PB-test ({len(X_pb_test)}).")

    # Final summary
    print("=" * 72)
    print("CROSS-DOMAIN SUMMARY")
    print("=" * 72)
    print(f"  Config 1 (zero-shot A->PB):       LR acc={r1['lr']['acc']:.3f}  MLP acc={r1['mlp']['acc']:.3f}")
    print(f"  Config 2 (zero-shot PB->A):       LR acc={r2['lr']['acc']:.3f}  MLP acc={r2['mlp']['acc']:.3f}")
    print(f"  Config 3 (few-shot A+PB->PBtest): LR acc={r3['lr']['acc']:.3f}  MLP acc={r3['mlp']['acc']:.3f}")
    print()
    print("Verdict thresholds:")
    print("  Zero-shot acc >= 0.70 -> Narrative Judge generalizes across worlds")
    print("  Few-shot acc >= 0.80  -> small new-world labels close any gap")
    print()


if __name__ == "__main__":
    main()
