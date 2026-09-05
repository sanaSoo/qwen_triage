"""
Tabular logistic-regression baseline for Task 2 (Continuous Alert).

Same purpose as baseline.py (Task 1): a fast, non-LLM reference point using
hand-picked features from the same running-state data the fine-tuned model sees
(casualty_report, prior_interventions, recent_trends -- see data_task2.py), scored
with the identical time-stratified Continuous Alert Score used in train_task2.py, so
the comparison is apples-to-apples.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import DATA_FILE, LSI_KEYS, SEGMENT_STRIDE
from .data_common import load_cases
from .data_task2 import load_task2_examples, task2_labels, split_by_case
from .metrics import continuous_alert_score


def featurize_task2_tabular(example):
    cr = example.get("casualty_report") or {}
    trauma_count = sum(1 for k, v in cr.items() if k.startswith("trauma_") and v is True)
    trends = example.get("recent_trends") or []

    if trends:
        latest = trends[-1]
        first = trends[0]
        latest_hr = latest.get("hr_bpm", 0) or 0
        latest_sbp = latest.get("sbp_mmhg", 0) or 0
        latest_spo2 = latest.get("spo2_pct", 0) or 0
        delta_hr = (latest.get("hr_bpm", 0) or 0) - (first.get("hr_bpm", 0) or 0)
        delta_sbp = (latest.get("sbp_mmhg", 0) or 0) - (first.get("sbp_mmhg", 0) or 0)
    else:
        latest_hr = latest_sbp = latest_spo2 = delta_hr = delta_sbp = 0.0

    return [
        cr.get("age_years") or 0,
        trauma_count,
        len(example.get("prior_interventions") or []),
        latest_hr,
        latest_sbp,
        latest_spo2,
        delta_hr,
        delta_sbp,
    ]


def main():
    cases = load_cases(DATA_FILE)
    examples = load_task2_examples(cases, segment_stride=SEGMENT_STRIDE)
    print(f"Loaded {len(examples)} Task 2 segment examples from {len(cases)} cases "
          f"(segment_stride={SEGMENT_STRIDE}).")

    train_examples, val_examples = split_by_case(examples)
    print(f"Train segments: {len(train_examples)}, Val segments: {len(val_examples)}")

    X_train = np.array([featurize_task2_tabular(ex) for ex in train_examples])
    X_val = np.array([featurize_task2_tabular(ex) for ex in val_examples])
    train_labels = [task2_labels(ex["response"]) for ex in train_examples]
    val_labels = [task2_labels(ex["response"]) for ex in val_examples]

    y_score_val = {k: np.zeros(len(val_examples)) for k in LSI_KEYS}

    for i, k in enumerate(LSI_KEYS):
        y_train_k = np.array([lab[i] for lab in train_labels])
        if y_train_k.sum() == 0 or y_train_k.sum() == len(y_train_k):
            print(f"  Skipping {k}: no positive/negative variation in training split.")
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train_k)
        y_score_val[k] = clf.predict_proba(X_val)[:, 1]

    all_scores = []
    for i, ex in enumerate(val_examples):
        for j, k in enumerate(LSI_KEYS):
            all_scores.append({
                "elapsed_sec": ex["elapsed_sec"],
                "lsi_key": k,
                "y_true": val_labels[i][j],
                "y_score": y_score_val[k][i],
            })

    per_lsi, avg = continuous_alert_score(all_scores)
    print("\nBaseline Task 2 Continuous Alert Score (per LSI):")
    for k, v in per_lsi.items():
        print(f"  {k}: {v}")
    print(f"\nBaseline averaged Continuous Alert Score: {avg}")
    print("\nCompare this against src/train_task2.py's fine-tuned result.")


if __name__ == "__main__":
    main()
