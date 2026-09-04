"""
Task 1 (First Look) data pipeline. Covers all three runs (ICD Section 3.2.1.1,
Rules Section 9.6.2.1):

  Run 1: casualty_report + prior interventions only, no vitals.
  Run 2: same as Run 1, PLUS initial_lsis (interventions in first 5 min) and
         5 minutes of pre-hospital vitals trends.
  Run 3: casualty_report is EMPTY -- replaced by ehr.pta_vitals ("basic EHR": a
         different vitals/GCS snapshot), plus the same initial_lsis and vitals
         trends as Run 2.

Raw waveform data (vs.signal) is present in Run 2/3 but deliberately skipped --
no signal encoder in this project, consistent with the rest of the pipeline.
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


def load_all_run_examples(cases):
    """
    Loads Run 1, Run 2, and Run 3 examples for every case. Each case contributes up
    to 3 examples -- one per run -- reflecting the different input completeness DARPA
    evaluates separately and then averages (Rules Section 9.6.2.4: First Look Score =
    mean across runs).
    """
    examples = []
    for case in cases:
        for run in case["task1_runs"]:
            seg_id = run["predict"]["segment_id"]
            for run_num in (1, 2, 3):
                if seg_id.endswith(f"-run{run_num}"):
                    examples.append({
                        "case_id": case["case_id"],
                        "run": run_num,
                        "predict": run["predict"],
                        "response": run["response"],
                    })
                    break
    return examples


def summarize_trends(trends):
    """
    Summarize a vs.trends dict (first ~5 min of pre-hospital vitals, Run 2/3) into a
    short text description: latest value and change from the start of the window, per
    vital sign. Mirrors the same crude-but-adequate summarization approach used for
    Task 2's streaming vitals (no signal encoder -- see data_task2.serialize_task2_state).
    """
    if not trends:
        return None
    keys = [k for k in trends if k != "t_sec" and trends.get(k)]
    if not keys:
        return None
    parts = []
    for k in keys:
        vals = trends[k]
        if not vals:
            continue
        first, last = vals[0], vals[-1]
        parts.append(f"{k}: {last:.1f} (change from start: {last - first:+.1f})")
    return "; ".join(parts) if parts else None


def serialize_predict(predict, run=None):
    """
    Turn the structured casualty_report/ehr/vs fields into a compact, deterministic
    prompt. Handles all three runs uniformly by only including a section when the
    relevant field is actually present -- Run 1 naturally produces the same output as
    before (backward compatible), Run 2/3 add whichever extra sections their data has.
    """
    cr = predict.get("casualty_report") or {}
    ehr = predict.get("ehr") or {}
    vs = predict.get("vs") or {}
    trauma_sites = [
        k.replace("trauma_", "") for k, v in cr.items()
        if k.startswith("trauma_") and v is True
    ]

    lines = []
    if run is not None:
        lines.append(f"Run: {run}")

    if cr:
        lines += [
            f"Sex: {cr.get('sex')}, Age: {cr.get('age_years')}, "
            f"Weight: {cr.get('weight_kg'):.1f}kg" if cr.get("weight_kg") else "",
            f"HR: {cr.get('hr'):.0f}, RR: {cr.get('rr'):.0f}" if cr.get("hr") else "",
            f"Trauma sites: {', '.join(trauma_sites) if trauma_sites else 'none reported'}",
            f"GCS - ocular:{cr.get('alertness_ocular')} verbal:{cr.get('alertness_verbal')} "
            f"motor:{cr.get('alertness_motor')}",
            f"Description: {cr.get('description', '')}",
        ]
    else:
        lines.append("Casualty Report: not available this run")

    pta = ehr.get("pta_vitals")
    if pta:
        lines.append(
            f"Basic EHR vitals -- HR:{pta.get('PTA_HR', 0):.0f} SBP:{pta.get('PTA_SBP', 0):.0f} "
            f"DBP:{pta.get('PTA_DBP', 0):.0f} RR:{pta.get('PTA_RR', 0):.0f} "
            f"GCS total:{pta.get('PTA_GCS_TOTAL')}"
        )

    lines.append(f"Prior interventions: {ehr.get('prior_interventions', [])}")
    if "initial_lsis" in ehr:
        lines.append(f"Initial interventions (first 5 min): {ehr.get('initial_lsis', [])}")

    trend_summary = summarize_trends(vs.get("trends") or {})
    if trend_summary:
        lines.append(f"Initial vitals trend (first 5 min): {trend_summary}")

    lines.append(f"Time to hospital admission: {predict.get('hosp_adm_time', 0):.0f} sec")
    lines.append(f"Prediction horizon: {predict.get('num_bins')} bins of 15 minutes each, "
                  f"starting from stop_time={predict.get('stop_time')}")

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
            continue
        rng.shuffle(pos_idx)
        n_val_pos = max(1, round(len(pos_idx) * val_fraction))
        n_val_pos = min(n_val_pos, len(pos_idx) - 1)
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


def stratified_split_by_case(cases, val_fraction=VAL_FRACTION, seed=RANDOM_SEED):
    """
    Stratified split at the CASE level (not per-run-example level) -- Run 1/2/3
    examples from the SAME case must stay together on the same side of the split, or
    the model would see e.g. a case's Run 1 in training and Run 2 (a near-identical
    restatement of the same underlying patient, just with more fields revealed) in
    validation -- leaking information and inflating the validation score.

    Uses Run 1's response as representative of the case's ground truth labels, since
    all runs of the same case target the same underlying LSI outcome window (only
    input completeness differs across runs, per the ICD's run schedule). This
    assumption isn't independently re-verified here -- worth a spot-check if a
    per-run result looks inconsistent with the others.
    """
    rng = random.Random(seed)
    case_labels = {}
    for case in cases:
        case_id = case["case_id"]
        chosen = None
        for run in case["task1_runs"]:
            if run["predict"]["segment_id"].endswith("-run1"):
                chosen = run["response"]
                break
        if chosen is None and case["task1_runs"]:
            chosen = case["task1_runs"][0]["response"]
        if chosen is not None:
            case_labels[case_id] = existence_labels(chosen)

    case_ids = list(case_labels.keys())
    n = len(case_ids)
    target_val_size = max(1, round(n * val_fraction))
    val_case_ids = set()

    for k in LSI_KEYS:
        pos_ids = [cid for cid in case_ids if case_labels[cid][k] > 0]
        if len(pos_ids) < 2:
            continue
        rng.shuffle(pos_ids)
        n_val_pos = max(1, round(len(pos_ids) * val_fraction))
        n_val_pos = min(n_val_pos, len(pos_ids) - 1)
        val_case_ids.update(pos_ids[:n_val_pos])

    remaining = [cid for cid in case_ids if cid not in val_case_ids]
    rng.shuffle(remaining)
    needed = max(0, target_val_size - len(val_case_ids))
    val_case_ids.update(remaining[:needed])

    train_cases = [c for c in cases if c["case_id"] not in val_case_ids]
    val_cases = [c for c in cases if c["case_id"] in val_case_ids]
    return train_cases, val_cases
