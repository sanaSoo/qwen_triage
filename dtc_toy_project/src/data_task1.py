"""
Task 1 (First Look, Run 1) data pipeline.

Run 1's predict message contains only casualty_report + ehr.prior_interventions,
no physiological signals (ICD Section 3.2.1.1, Run schedule in Rules Section 9.6.2.1).
This makes it a pure tabular + short-text -> structured numeric vector problem,
the simplest possible scope for a first fine-tuning pass.
"""

import random

from .config import LSI_KEYS, VAL_FRACTION, RANDOM_SEED


def load_run1_examples(cases):
    """cases: list of case dicts, as returned by data_common.load_cases()."""
    examples = []
    for case in cases:
        for run in case["task1_runs"]:
            if run["predict"]["segment_id"].endswith("run1"):
                examples.append({
                    "case_id": case["case_id"],
                    "predict": run["predict"],
                    "response": run["response"],
                })
    return examples


def serialize_predict(predict):
    """Turn the structured casualty_report + ehr into a compact, deterministic prompt."""
    cr = predict.get("casualty_report") or {}
    ehr = predict.get("ehr") or {}
    trauma_sites = [
        k.replace("trauma_", "") for k, v in cr.items()
        if k.startswith("trauma_") and v is True
    ]
    lines = [
        f"Sex: {cr.get('sex')}, Age: {cr.get('age_years')}, "
        f"Weight: {cr.get('weight_kg'):.1f}kg" if cr.get("weight_kg") else "",
        f"HR: {cr.get('hr'):.0f}, RR: {cr.get('rr'):.0f}" if cr.get("hr") else "",
        f"Trauma sites: {', '.join(trauma_sites) if trauma_sites else 'none reported'}",
        f"GCS - ocular:{cr.get('alertness_ocular')} verbal:{cr.get('alertness_verbal')} "
        f"motor:{cr.get('alertness_motor')}",
        f"Description: {cr.get('description', '')}",
        f"Prior interventions: {ehr.get('prior_interventions', [])}",
        f"Time to hospital admission: {predict.get('hosp_adm_time', 0):.0f} sec",
        f"Prediction horizon: {predict.get('num_bins')} bins of 15 minutes each, "
        f"starting from stop_time={predict.get('stop_time')}",
    ]
    return "\n".join(l for l in lines if l)


def existence_labels(response):
    """
    Reduce the variable-length lsi_predictions vectors to a fixed 6-value existence
    label (does LSI k occur at all within the horizon). This is exactly the reduction
    the Rules Doc's own Existence Score uses internally (Section 9.6.2.4.1, max over
    time-bins) -- so this is implementing the real first-stage metric, not a shortcut
    around it.
    """
    preds = response["lsi_predictions"]
    labels = {k: float(max(preds[k]) > 0) for k in LSI_KEYS}
    labels["any_lsi"] = float(preds["any_lsi"] > 0.5)
    return labels


def stratified_split(examples, val_fraction=VAL_FRACTION, seed=RANDOM_SEED):
    """
    Stratified train/val split for Task 1's multi-label existence targets.

    A naive sequential or fully-random split can, by bad luck (or because the source
    data was generated/ordered in per-LSI batches), place ALL positive examples for a
    rare LSI into just one side of the split -- leaving the other side with zero
    positive/negative variation and an undefined AP score (metrics.existence_score
    returns None for that LSI), even when the LSI has plenty of positive examples
    overall in the full dataset.

    This guarantees each LSI with at least 2 positive examples gets at least 1
    positive example on both sides of the split, then fills the rest of the
    validation set randomly from whatever's left.
    """
    rng = random.Random(seed)
    n = len(examples)
    labels = [existence_labels(ex["response"]) for ex in examples]

    target_val_size = max(1, round(n * val_fraction))
    val_indices = set()

    for k in LSI_KEYS:
        pos_idx = [i for i in range(n) if labels[i][k] > 0]
        if len(pos_idx) < 2:
            continue  # not enough positives to safely put any in val without starving train
        rng.shuffle(pos_idx)
        n_val_pos = max(1, round(len(pos_idx) * val_fraction))
        n_val_pos = min(n_val_pos, len(pos_idx) - 1)  # never fully drain train's positives
        val_indices.update(pos_idx[:n_val_pos])

    remaining = [i for i in range(n) if i not in val_indices]
    rng.shuffle(remaining)
    needed = max(0, target_val_size - len(val_indices))
    val_indices.update(remaining[:needed])

    val_indices = sorted(val_indices)
    val_set = set(val_indices)
    train_indices = [i for i in range(n) if i not in val_set]

    train_examples = [examples[i] for i in train_indices]
    val_examples = [examples[i] for i in val_indices]
    return train_examples, val_examples