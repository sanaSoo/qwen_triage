"""
Optional realism pass: simulate the ICD's sequential message loop (Section 3.1, steps
2-10) in plain Python, without RabbitMQ. Useful to internalize the shape of the real
interface -- this is the piece that would, in a real submission, become your
DTC_BaseModel.predict() and cleanup() callbacks (ICD Section 5.1.2).
"""

from .config import DATA_FILE
from .data_common import load_cases
from .data_task2 import serialize_task2_state


def simulate_task2_run(case, model_predict_fn):
    history_state = {"casualty_report": None, "prior_interventions": [], "recent_trends": []}
    log = []
    for seg in case["task2_segments"]:
        p = seg["predict"]

        if p.get("casualty_report"):
            history_state["casualty_report"] = p["casualty_report"]
        if "prior_interventions" in (p.get("ehr") or {}):
            history_state["prior_interventions"] = p["ehr"]["prior_interventions"]
        trends = (p.get("vs") or {}).get("trends") or {}
        if trends.get("hr_bpm"):
            history_state["recent_trends"].append(
                {k: trends[k][-1] for k in trends if trends[k]}
            )

        prompt = serialize_task2_state(history_state)
        response = model_predict_fn(prompt)  # plug in your trained classifier here
        log.append({"segment_id": p["segment_id"], "response": response})

        if p["end_of_case"]:
            break  # mirrors the Evaluator sending a Cleanup message (ICD Section 3.2.4)
    return log


def dummy_model_predict_fn(prompt):
    """Stand-in so this script runs without a trained model loaded."""
    return {"airway_invasive": 0.0, "blood_products": 0.0, "chest_decompression": 0.0,
            "surgical": 0.0, "vaso_cardioactive_medications": 0.0}


def main():
    cases = load_cases(DATA_FILE)
    if not cases:
        print("No cases found in data/all_cases.json.")
        return
    case = cases[0]
    log = simulate_task2_run(case, dummy_model_predict_fn)
    print(f"Simulated {len(log)} segments for {case['case_id']}.")
    print("First 3 log entries:", log[:3])


if __name__ == "__main__":
    main()
