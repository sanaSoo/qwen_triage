"""Shared constants used across the project."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "all_cases.json"   # single combined file containing every case
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# --- Model ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN = 512

# --- LSI labels (DTC Rules Doc Section 9.5.2, Table 1 in the ICD) ---
LSI_KEYS = [
    "airway_invasive",
    "blood_products",
    "chest_decompression",
    "surgical",
    "vaso_cardioactive_medications",
]

# Task 1 response includes any_lsi; Task 2 does not (ICD Sections 3.2.2 vs 3.3.2)
TASK1_OUTPUT_KEYS = LSI_KEYS + ["any_lsi"]
TASK2_OUTPUT_KEYS = LSI_KEYS

# --- Training ---
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
VAL_FRACTION = 0.1 # adjust if we increase number of samples (10 case threshold per LSI)
RANDOM_SEED = 0

# --- LoRA ---
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# --- Task 2 ---
VITALS_WINDOW_SEGMENTS = 10  # how many recent vitals segments to keep in running state
SEGMENT_STRIDE = 5  # only 1 in N all-negative segments becomes a training example
TASK2_TIME_STRATA_SEC = [(0, 1800), (1800, 3600), (3600, 7200), (7200, 14400)]
