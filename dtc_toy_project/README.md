# DTC Toy Project — LoRA Fine-Tuning Practice

A learning exercise: fine-tune Qwen2.5-1.5B-Instruct with LoRA to predict life-saving intervention
(LSI) needs from trauma casualty data, using the DARPA Triage Challenge (DTC) Data Competition's
message formats and scoring metrics as a realistic, well-specified problem to practice on.

**This is not a competition submission.** No RabbitMQ, no Docker, no CI/CD harness — everything
here simulates the ICD's predict/response message shapes in plain Python so the project stays
focused on the fine-tuning itself.

## Scope

- **Task 1 (First Look), Run 1** — fine-tuned (LoRA + classification head)
- **Task 2 (Continuous Alert)** — fine-tuned (same architecture, sequential state)
- **Task 3 (Resource Allocation)** — rule-based only, no LLM (it's a constraint-satisfaction
  problem, not a learning problem)

## Directory structure

```
dtc_toy_project/
├── README.md                  <- you are here
├── requirements.txt
├── data/                      <- put your case-*.json files here (not committed)
├── outputs/                   <- trained adapters, metrics, logs land here
└── src/
    ├── __init__.py
    ├── config.py               <- shared constants (LSI keys, model name, paths)
    ├── data_task1.py           <- Task 1 loading, serialization, labels
    ├── data_task2.py           <- Task 2 loading, running-state serialization, labels
    ├── data_task3.py           <- Task 3 scenario reassembly across case files
    ├── model.py                <- LSIClassifier (base model + LoRA + classification head)
    ├── metrics.py              <- existence_score, continuous_alert_score, task3_eval
    ├── train_task1.py          <- fine-tuning entry point for Task 1
    ├── train_task2.py          <- fine-tuning entry point for Task 2
    ├── task3_allocator.py      <- rule-based allocator (no LLM)
    ├── simulate_task2_loop.py  <- optional: plain-Python message-loop simulation
    └── baseline.py             <- logistic regression baseline (run before fine-tuning)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Put your combined case data file at `data/all_cases.json`. It should be a single JSON file
containing all cases, in one of these shapes:

- a JSON array: `[ {"case_id": "...", "task1_runs": [...], ...}, {...}, ... ]`
- a wrapper dict: `{"cases": [ {...}, {...} ]}`
- a dict keyed by case id: `{"case-119": {...}, "case-120": {...}}`

`src/data_common.py` auto-detects which of these shapes you have. Each individual case dict
should have the shape described in the DTC ICD/Rules docs: `task1_runs`, `task2_segments`,
`task3_records`, `ground_truth`.

## Run order

```bash
# 1. Sanity-check baseline before touching the LLM
python -m src.baseline

# 2. Fine-tune Task 1 (LoRA)
python -m src.train_task1

# 3. Fine-tune Task 2 (LoRA)
python -m src.train_task2

# 4. Task 3 — no training step, just run the allocator + eval
python -m src.task3_allocator

# 5. (optional) simulate the sequential message loop for Task 2
python -m src.simulate_task2_loop
```

Each training script prints per-LSI and averaged scores using the metrics in `metrics.py`, which
implement the DTC Rules Document's actual Existence Score (Section 9.6.2.4.1) and Continuous Alert
Score (Section 9.6.3.3) formulas — so the numbers you get are directly comparable to how DARPA
would score the real thing, just on your own small held-out split.

## What to expect

- With ~100 cases, expect noisy/undefined AP scores for rare LSI types — this is a real,
  documented consequence of small-N, not a bug (see `metrics.py` docstrings).
- The fine-tuned model may not beat the `baseline.py` logistic regression — that's a legitimate
  and common small-data finding, not a failure.
- The point of this project is the fine-tuning *process* (LoRA setup, training loop, evaluation
  discipline), not beating any particular number.
