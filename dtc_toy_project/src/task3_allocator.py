"""
Task 3 (Resource Allocation) -- deterministic, no LLM.

This is a constraint-satisfaction / ranking problem, not a learning problem: the ICD's
hard constraints (Section 3.4.2 -- exact resource caps, hospital_bed required for any
other resource, no duplicate resource per patient, blood_high/blood_low mutually
exclusive) are satisfied by construction here, which an LLM emitting free-text JSON
cannot guarantee.

REVISED based on diagnose_task3_resources.py findings against real ground truth:

1. hospital_bed's stated cap is NOT a literal ceiling -- actual assignments exceeded it
   by 2.8x-4.8x in every resource-scarce scenario checked. It DOES loosely scale with
   how many patients receive care, just not 1:1. HOSPITAL_BED_CAP_MULTIPLIER below is a
   rough calibration from only 3 usable scenarios (observed ratios: 2.82x, 4.76x,
   3.47x) -- a first-pass estimate, likely to need recalibration with more data.

2. Evacuation could NOT be explained by hospital_bed exhaustion, the "evacuation"
   resource count, or this severity_score heuristic -- average severity_score for
   evacuated patients was EQUAL TO OR LOWER than non-evacuated patients in every
   scenario checked. Whatever drives real evacuation isn't captured by these six
   features, and forcing an "evacuate the most severe" rule would be actively wrong
   here, not just unhelpful. Since evacuation is also rare (0-10% per scenario),
   predicting "nobody evacuated" matches or beats the majority-class baseline in every
   scenario tested -- so that's what this allocator does now, rather than pretending to
   have a real evacuation policy it doesn't have evidence for.
"""

from .config import DATA_FILE
from .data_common import load_cases
from .data_task3 import load_task3_scenarios
from .metrics import task3_eval, task3_trivial_baselines

HOSPITAL_BED_CAP_MULTIPLIER = 3.5  # rough calibration, see module docstring above


def severity_score(patient_message):
    """Simple, transparent heuristic -- not learned. Swap in a Task 1/2 model's
    existence-probability output here for continuity across tasks if desired, but note
    the ICD only provides ehr.prior_interventions + casualty_report + vs for Task 3
    (Section 3.4.1.1), a narrower feature set than Task 1 Run 3."""
    cr = patient_message.get("casualty_report") or {}
    score = 0.0
    if cr.get("severe_hemorrhage"):
        score += 3.0
    trauma_count = sum(1 for k, v in cr.items() if k.startswith("trauma_") and v is True)
    score += trauma_count * 0.5
    if (cr.get("alertness_motor") or 6) < 6:
        score += 1.0
    hr = cr.get("hr") or 80
    if hr > 120 or hr < 50:
        score += 1.0
    return score


def allocate_resources(scenario_id, patients, resources):
    scored = [(p, severity_score(p)) for p in patients]
    scored.sort(key=lambda x: -x[1])  # highest severity first

    n = len(scored)
    bed_cap = resources.get("hospital_bed", 0)
    n_get_care = min(n, round(bed_cap * HOSPITAL_BED_CAP_MULTIPLIER))

    remaining = dict(resources)
    assignments = []

    for i, (p, sev) in enumerate(scored):
        pid = p["patient_id"]

        if i >= n_get_care:
            continue

        res_list = ["hospital_bed"]

        cr = p.get("casualty_report") or {}
        if cr.get("severe_hemorrhage") and remaining.get("blood", 0) > 0:
            res_list.append("blood_high" if sev > 3 else "blood_low")
            remaining["blood"] -= 1
        if cr.get("trauma_head") and remaining.get("ventilator", 0) > 0:
            res_list.append("ventilator")
            remaining["ventilator"] -= 1
        if sev > 4 and remaining.get("surgery", 0) > 0:
            res_list.append("surgery")
            remaining["surgery"] -= 1

        assignments.append({"patient_id": pid, "resources": res_list})

    return {
        "case_id": scenario_id,
        "evacuated_patients": [],
        "resource_assignments": assignments,
    }


def main():
    cases = load_cases(DATA_FILE)
    scenarios = load_task3_scenarios(cases)
    print(f"Reassembled {len(scenarios)} scenario(s) from {len(cases)} cases.")
    print("Note: scenarios missing some participating patients' case files will look")
    print("incomplete -- only trust results for scenarios where every patient is present.\n")

    for sid, scenario in scenarios.items():
        print(f"{sid} raw resources dict: {scenario['resources']}, "
              f"patient count: {len(scenario['patients'])}")
        allocation = allocate_resources(sid, scenario["patients"], scenario["resources"])
        result = task3_eval(scenario, allocation)
        baseline = task3_trivial_baselines(scenario)
        print(f"\n{sid} — patients: {len(scenario['patients'])}, "
              f"evac_rate={baseline['evac_rate']:.3f}")
        print(f"  evac_accuracy:       {result['evac_accuracy']:.4f}   "
              f"(majority-class baseline: {baseline['evac_majority_baseline']:.4f})")
        print(f"  mean_resource_jaccard: {result['mean_resource_jaccard']:.4f}   "
              f"(predict-nothing baseline: {baseline['resource_empty_baseline']:.4f})")


if __name__ == "__main__":
    main()
