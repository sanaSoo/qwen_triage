"""
Loads all cases from a single combined JSON file (data/all_cases.json), rather than
one file per case. Handles the common shapes this might come in:

  1. A JSON array:            [ {case_id: ..., task1_runs: ...}, {...}, ... ]
  2. A wrapper dict, list:    { "cases": [ {...}, {...} ], ...other metadata... }
  3. A wrapper dict, dict:    { "cases": {"case-119": {...}, "case-120": {...}}, ...metadata... }
  4. A plain dict keyed by id (no wrapper): { "case-119": {...}, "case-120": {...} }

Also guards against non-case values slipping through (e.g. metadata fields sitting
alongside a "cases" key at the top level) by filtering to dict entries that actually
look like a case (have a "case_id" field).
"""

import json

from .config import DATA_FILE


def _looks_like_case(obj):
    return isinstance(obj, dict) and "case_id" in obj


def load_cases(path=DATA_FILE):
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        candidates = raw
    elif isinstance(raw, dict):
        if "cases" in raw:
            inner = raw["cases"]
            if isinstance(inner, list):
                candidates = inner
            elif isinstance(inner, dict):
                candidates = list(inner.values())
            else:
                raise ValueError(
                    f"Found a 'cases' key in {path} but its value is a "
                    f"{type(inner)}, expected a list or dict of case objects."
                )
        else:
            # no "cases" wrapper -- assume the whole dict is keyed by case_id
            candidates = list(raw.values())
    else:
        raise ValueError(
            f"Unrecognized top-level JSON structure in {path}: expected a list or dict."
        )

    cases = [c for c in candidates if _looks_like_case(c)]
    dropped = len(candidates) - len(cases)
    if dropped:
        print(f"Note: dropped {dropped} non-case entr{'y' if dropped == 1 else 'ies'} "
              f"while loading {path} (likely metadata fields, not actual cases).")

    print(f"Loaded {len(cases)} cases from {path}")
    return cases