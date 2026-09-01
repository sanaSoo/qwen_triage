"""
Task 3 (Resource Allocation) data loading.

Each case's task3_records only holds THAT case's own patient-slice of a multi-patient
scenario (one record per scenario_id the patient appears in). To reconstruct a full
scenario matching the ICD's Message 12a/12b shape, records from every case sharing the
same scenario_id must be gathered together.

With a small case sample, expect many scenarios to only be partially reconstructable
(missing some participating patients) -- only evaluate scenarios where every patient's
record is present, or the comparison against ground truth isn't meaningful.
"""

from collections import defaultdict


def load_task3_scenarios(cases):
    """cases: list of case dicts, as returned by data_common.load_cases()."""
    scenarios = defaultdict(lambda: {"resources": None, "patients": [], "ground_truth": {}})
    for case in cases:
        for rec in case.get("task3_records", []):
            sid = rec["scenario_id"]
            scenarios[sid]["resources"] = rec["resources"]
            scenarios[sid]["patients"].append(rec["patient_message"])
            scenarios[sid]["ground_truth"][rec["patient_message"]["patient_id"]] = {
                "assignments": rec["resource_assignments_for_patient"],
                "evacuated": rec["evacuated"],
            }
    return scenarios


def filter_complete_scenarios(scenarios, expected_patient_counts=None):
    """
    Only keep scenarios where reconstruction looks complete. If expected_patient_counts
    (dict[scenario_id] -> int) is known from elsewhere, use it; otherwise this is a
    best-effort filter and should be treated as advisory, not a guarantee of
    completeness.
    """
    if not expected_patient_counts:
        return scenarios
    return {
        sid: s for sid, s in scenarios.items()
        if len(s["patients"]) >= expected_patient_counts.get(sid, 0)
    }
