"""
Scoring metrics matching the DTC Rules Document.

existence_score()      -> Task 1, Rules Section 9.6.2.4.1
continuous_alert_score() -> Task 2, Rules Section 9.6.3.3
task3_eval()            -> Task 3, informal (no official small-N metric in the Rules doc;
                           this is a reasonable stand-in, not an official formula)

Note: with ~100 cases, AP is frequently undefined (all-positive or all-negative subsets).
These functions return None for undefined cases rather than silently substituting 0 or 1
-- report the gap, don't paper over it.
"""

from sklearn.metrics import average_precision_score
import numpy as np

from .config import LSI_KEYS, TASK2_TIME_STRATA_SEC


def existence_score(y_true, y_score, prevalence=None):
    """
    y_true, y_score: dict[lsi_key] -> np.array of shape (N_cases,)
    prevalence: optional dict[lsi_key] -> float, prior prevalence pi_k
    """
    scores = {}
    for k in LSI_KEYS:
        yt, ys = np.asarray(y_true[k]), np.asarray(y_score[k])
        if yt.sum() == 0 or yt.sum() == len(yt):
            scores[k] = None
            continue
        ap = average_precision_score(yt, ys)
        pi = prevalence.get(k, yt.mean()) if prevalence else yt.mean()
        norm_ap = (ap - pi) / (1 - pi) if pi < 1 else 0.0
        scores[k] = max(0.0, norm_ap)
    valid = [v for v in scores.values() if v is not None]
    return scores, (sum(valid) / len(valid) if valid else None)


def continuous_alert_score(examples_with_scores):
    """
    examples_with_scores: list of dicts with keys 'elapsed_sec', 'lsi_key', 'y_true', 'y_score'
    """
    results = {}
    for k in LSI_KEYS:
        strat_scores = []
        for lo, hi in TASK2_TIME_STRATA_SEC:
            subset = [
                e for e in examples_with_scores
                if e["lsi_key"] == k and lo <= e["elapsed_sec"] < hi
            ]
            if len(subset) < 2:
                continue
            yt = np.array([e["y_true"] for e in subset])
            ys = np.array([e["y_score"] for e in subset])
            if yt.sum() == 0 or yt.sum() == len(yt):
                continue
            ap = average_precision_score(yt, ys)
            pi = yt.mean()
            norm_ap = max(0.0, (ap - pi) / (1 - pi)) if pi < 1 else 0.0
            strat_scores.append(norm_ap)
        if strat_scores:
            results[k] = sum(strat_scores) / len(strat_scores)
    valid = list(results.values())
    return results, (sum(valid) / len(valid) if valid else None)


def task3_eval(scenario, allocations):
    """Informal Task 3 metric: evacuation accuracy + resource-set Jaccard overlap."""
    gt = scenario["ground_truth"]
    n = len(gt)
    if n == 0:
        return {"evac_accuracy": None, "mean_resource_jaccard": None}

    evac_correct = 0
    resource_jaccard = []
    for p in scenario["patients"]:
        pid = p["patient_id"]
        pred_evac = pid in allocations["evacuated_patients"]
        true_evac = gt[pid]["evacuated"]
        evac_correct += int(pred_evac == true_evac)

        pred_res = next(
            (a["resources"] for a in allocations["resource_assignments"]
             if a["patient_id"] == pid), []
        )
        true_res = next((a["resources"] for a in gt[pid]["assignments"]), [])
        pred_set, true_set = set(pred_res), set(true_res)
        union = pred_set | true_set
        resource_jaccard.append(len(pred_set & true_set) / len(union) if union else 1.0)

    return {
        "evac_accuracy": evac_correct / n,
        "mean_resource_jaccard": (
            sum(resource_jaccard) / len(resource_jaccard) if resource_jaccard else None
        ),
    }

def task3_trivial_baselines(scenario):
    """
    Reference points for interpreting task3_eval's numbers -- neither evac_accuracy nor
    mean_resource_jaccard has a built-in "how good is this really" floor the way
    normalized AP does (Task 1/2), so a score like 0.34 is meaningless without knowing
    what a trivial, uninformed allocator would score on the same scenario.

    evac_majority_baseline: accuracy of always predicting whichever outcome (evacuated
        vs not) is more common in this scenario's ground truth. A real allocator scoring
        BELOW this has learned less than just guessing the majority class every time.

    resource_empty_baseline: mean Jaccard of always predicting "no resources assigned"
        for every patient. Jaccard counts an empty/empty match as 1.0, so this baseline
        equals the fraction of patients who genuinely received nothing in ground truth --
        a real allocator scoring below this is doing worse than assigning nothing at all.
    """
    gt = scenario["ground_truth"]
    n = len(gt)
    if n == 0:
        return {"evac_majority_baseline": None, "resource_empty_baseline": None}

    evac_flags = [v["evacuated"] for v in gt.values()]
    evac_rate = sum(evac_flags) / n
    evac_majority_baseline = max(evac_rate, 1 - evac_rate)

    empty_correct = sum(1 for v in gt.values() if len(v["assignments"]) == 0
                         or all(len(a.get("resources", [])) == 0 for a in v["assignments"]))
    resource_empty_baseline = empty_correct / n

    return {
        "evac_majority_baseline": evac_majority_baseline,
        "resource_empty_baseline": resource_empty_baseline,
        "evac_rate": evac_rate,  # useful context: what fraction actually got evacuated
    }
