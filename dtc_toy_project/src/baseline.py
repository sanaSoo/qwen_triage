"""
Tabular logistic-regression baseline for Task 1.

Run this BEFORE fine-tuning. It's fast, gives you a number to beat, and forces you to
exercise the scoring metric (metrics.existence_score) before adding LLM complexity on
top of it.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import DATA_FILE, LSI_KEYS
from .data_common import load_cases
from .data_task1 import load_run1_examples, existence_labels, stratified_split
from .metrics import existence_score


def featurize_tabular(predict):
    cr = predict.get("casualty_report") or {}
    trauma_count = sum(1 for k, v in cr.items() if k.startswith("trauma_") and v is True)
    return [
        cr.get("age_years") or 0,
        cr.get("hr") or 0,
        cr.get("rr") or 0,
        trauma_count,
        cr.get("alertness_motor") or 0,
        1 if "severe" in (cr.get("description") or "").lower() else 0,
    ]


def main():
    cases = load_cases(DATA_FILE)
    examples = load_run1_examples(cases)
    print(f"Loaded {len(examples)} Task 1 Run 1 examples from {len(cases)} cases.")

    train_examples, val_examples = stratified_split(examples)
    print(f"Train: {len(train_examples)} cases, Val: {len(val_examples)} cases "
          f"(stratified split -- each LSI with enough positives gets some in both sides)")

    X_train = np.array([featurize_tabular(ex["predict"]) for ex in train_examples])
    X_val = np.array([featurize_tabular(ex["predict"]) for ex in val_examples])
    train_labels = [existence_labels(ex["response"]) for ex in train_examples]
    val_labels = [existence_labels(ex["response"]) for ex in val_examples]

    y_true_val = {k: np.array([val_labels[i][k] for i in range(len(val_examples))])
                  for k in LSI_KEYS}
    y_score_val = {k: np.zeros(len(val_examples)) for k in LSI_KEYS}

    for k in LSI_KEYS:
        y_train_k = np.array([train_labels[i][k] for i in range(len(train_examples))])
        if y_train_k.sum() == 0 or y_train_k.sum() == len(y_train_k):
            print(f"  Skipping {k}: no positive/negative variation in training split.")
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train_k)
        y_score_val[k] = clf.predict_proba(X_val)[:, 1]

    per_lsi, avg = existence_score(y_true_val, y_score_val)
    print("\nBaseline Existence Score (per LSI):")
    for k, v in per_lsi.items():
        print(f"  {k}: {v if v is not None else 'undefined (no pos/neg variation)'}")
    print(f"\nBaseline averaged Existence Score: {avg}")


if __name__ == "__main__":
    main()