"""
Task 2 (Continuous Alert) data pipeline.

task2_segments in the source case files are already pre-built predict/response pairs
(ICD Section 3.3), one per 30-second segment. The model must accumulate context across
segments -- casualty_report/ehr are only sent once early on and not repeated -- so this
module maintains a running text-serializable state per case, rather than treating each
segment as an independent example.

IMPORTANT: segments within a case are highly autocorrelated. Always split train/val by
case_id, never by segment -- see split_by_case() below.
"""

import random

from .config import LSI_KEYS, VITALS_WINDOW_SEGMENTS, VAL_FRACTION, RANDOM_SEED


def load_task2_examples(cases, vitals_window_segments=VITALS_WINDOW_SEGMENTS):
    """cases: list of case dicts, as returned by data_common.load_cases()."""
    examples = []
    for case in cases:
        segs = case["task2_segments"]
        history_trends = []
        casualty_desc = None
        prior_interventions = []

        for seg in segs:
            p = seg["predict"]
            cr = p.get("casualty_report") or {}
            ehr = p.get("ehr") or {}
            vs = p.get("vs") or {}

            if cr:
                casualty_desc = cr
            if "prior_interventions" in ehr:
                prior_interventions = ehr["prior_interventions"]

            trends = vs.get("trends") or {}
            if trends.get("hr_bpm"):
                history_trends.append({
                    k: trends[k][-1] for k in
                    ["hr_bpm", "sbp_mmhg", "dbp_mmhg", "rr_bpm", "spo2_pct", "temp_c"]
                    if trends.get(k)
                })
            if vitals_window_segments:
                history_trends = history_trends[-vitals_window_segments:]

            examples.append({
                "case_id": case["case_id"],
                "segment_id": p["segment_id"],
                "elapsed_sec": p.get("start_time", 0.0),
                "casualty_report": casualty_desc,
                "prior_interventions": prior_interventions,
                "recent_trends": list(history_trends),
                "response": seg["response"],
            })
    return examples


def serialize_task2_state(example):
    """Crude but adequate: hand-computed trend deltas rather than a real signal encoder."""
    cr = example["casualty_report"] or {}
    trends = example["recent_trends"]
    trauma_sites = [
        k.replace("trauma_", "") for k, v in cr.items()
        if k.startswith("trauma_") and v is True
    ]
    lines = [
        f"Sex: {cr.get('sex')}, Age: {cr.get('age_years')}",
        f"Trauma sites: {', '.join(trauma_sites) if trauma_sites else 'none reported'}",
        f"Description: {cr.get('description', '')}",
        f"Prior interventions: {example['prior_interventions']}",
    ]
    if trends:
        latest = trends[-1]
        lines.append(
            f"Latest vitals — HR:{latest.get('hr_bpm', 0):.0f} "
            f"SBP:{latest.get('sbp_mmhg', 0):.0f} SpO2:{latest.get('spo2_pct', 0):.0f}"
        )
        if len(trends) > 1:
            first = trends[0]
            lines.append(
                f"Trend over window — HR change: "
                f"{latest.get('hr_bpm', 0) - first.get('hr_bpm', 0):+.0f}, "
                f"SBP change: "
                f"{latest.get('sbp_mmhg', 0) - first.get('sbp_mmhg', 0):+.0f}"
            )
    else:
        lines.append("No vitals data yet (pre-admission segment)")
    return "\n".join(lines)


def task2_labels(response):
    preds = response["lsi_predictions"]
    return [float(preds[k]) for k in LSI_KEYS]


def split_by_case(examples, val_fraction=VAL_FRACTION, seed=RANDOM_SEED):
    """Split by case_id, NOT by segment -- segments within a case are correlated and
    segment-level splitting will leak information, producing falsely optimistic
    validation scores."""
    case_ids = list({e["case_id"] for e in examples})
    random.Random(seed).shuffle(case_ids)
    n_val = max(1, int(len(case_ids) * val_fraction))
    val_cases = set(case_ids[:n_val])
    train = [e for e in examples if e["case_id"] not in val_cases]
    val = [e for e in examples if e["case_id"] in val_cases]
    return train, val
