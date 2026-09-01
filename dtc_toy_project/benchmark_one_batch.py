"""
Times a single forward+backward pass on real data to estimate total training time.
Run this from the project root: python3 benchmark_one_batch.py

This does NOT touch your currently-running training job -- if it's still running,
this will compete for the same GPU/CPU resources and make both slower temporarily,
but it will still give you a usable number. For a cleaner read, Ctrl+C the running
job first, run this, then decide whether to restart.
"""

import time
import torch
import torch.nn as nn

from src.config import DATA_FILE, TASK1_OUTPUT_KEYS, MAX_SEQ_LEN, BATCH_SIZE, LEARNING_RATE
from src.data_common import load_cases
from src.data_task1 import load_run1_examples, serialize_predict, existence_labels, stratified_split
from src.model import build_model_and_tokenizer

print("Loading data...")
cases = load_cases(DATA_FILE)
examples = load_run1_examples(cases)
train_examples, val_examples = stratified_split(examples)
n_batches_per_epoch = len(train_examples) // BATCH_SIZE
print(f"Train examples: {len(train_examples)}, batches/epoch: {n_batches_per_epoch}")

print("\nLoading model (this part is separately slow on first run if not cached)...")
t0 = time.time()
classifier, tokenizer, device = build_model_and_tokenizer(n_outputs=len(TASK1_OUTPUT_KEYS))
print(f"Model load took {time.time() - t0:.1f}s")

optimizer = torch.optim.AdamW(classifier.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

# Build one batch manually
batch_examples = train_examples[:BATCH_SIZE]
texts = [serialize_predict(ex["predict"]) for ex in batch_examples]
enc = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_SEQ_LEN,
                return_tensors="pt")
input_ids = enc["input_ids"].to(device)
attn_mask = enc["attention_mask"].to(device)
labels = [existence_labels(ex["response"]) for ex in batch_examples]
y = torch.tensor([[lab[k] for k in TASK1_OUTPUT_KEYS] for lab in labels],
                  dtype=torch.float).to(device)

print("\nRunning 3 warmup+timed batches...")
classifier.train()
times = []
for i in range(3):
    t0 = time.time()
    optimizer.zero_grad()
    preds = classifier(input_ids, attn_mask)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()
    elapsed = time.time() - t0
    times.append(elapsed)
    print(f"  batch {i+1}: {elapsed:.2f}s (loss={loss.item():.4f})")

avg = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]  # skip first (warmup)
print(f"\nAverage per-batch time (excluding first/warmup): {avg:.2f}s")

NUM_EPOCHS = 10
total_batches = n_batches_per_epoch * NUM_EPOCHS
est_seconds = avg * total_batches
print(f"\nEstimated total training time for {NUM_EPOCHS} epochs "
      f"({total_batches} batches): {est_seconds/60:.1f} minutes ({est_seconds/3600:.2f} hours)")