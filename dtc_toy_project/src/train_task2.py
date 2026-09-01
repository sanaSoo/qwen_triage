"""Fine-tune the LSIClassifier (Qwen2.5-1.5B + LoRA) on Task 2, Continuous Alert."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import (
    DATA_FILE, OUTPUTS_DIR, LSI_KEYS, MAX_SEQ_LEN, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
)
from .data_common import load_cases
from .data_task2 import load_task2_examples, serialize_task2_state, task2_labels, split_by_case
from .model import build_model_and_tokenizer
from .metrics import continuous_alert_score


class Task2Dataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=MAX_SEQ_LEN):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = serialize_task2_state(ex)
        enc = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        y = torch.tensor(task2_labels(ex["response"]), dtype=torch.float)
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), y


def main():
    cases = load_cases(DATA_FILE)
    examples = load_task2_examples(cases)
    print(f"Loaded {len(examples)} Task 2 segment examples from {len(cases)} cases.")

    # CRITICAL: split by case, not by segment (see data_task2.split_by_case docstring)
    train_examples, val_examples = split_by_case(examples)
    print(f"Train segments: {len(train_examples)}, Val segments: {len(val_examples)}")

    classifier, tokenizer, device = build_model_and_tokenizer(n_outputs=len(LSI_KEYS))

    train_ds = Task2Dataset(train_examples, tokenizer)
    val_ds = Task2Dataset(val_examples, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    for epoch in range(NUM_EPOCHS):
        classifier.train()
        total_loss = 0.0
        for input_ids, attn_mask, y in train_loader:
            input_ids, attn_mask, y = input_ids.to(device), attn_mask.to(device), y.to(device)
            optimizer.zero_grad()
            preds = classifier(input_ids, attn_mask)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} — train loss: {total_loss / len(train_loader):.4f}")

    # --- Evaluate using the time-stratified Continuous Alert Score ---
    classifier.eval()
    all_scores = []
    with torch.no_grad():
        idx = 0
        for input_ids, attn_mask, y in val_loader:
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            preds = classifier(input_ids, attn_mask).cpu().numpy()
            y = y.numpy()
            batch_examples = val_examples[idx: idx + len(y)]
            idx += len(y)
            for row_pred, row_true, ex in zip(preds, y, batch_examples):
                for i, k in enumerate(LSI_KEYS):
                    all_scores.append({
                        "elapsed_sec": ex["elapsed_sec"],
                        "lsi_key": k,
                        "y_true": row_true[i],
                        "y_score": row_pred[i],
                    })

    per_lsi, avg = continuous_alert_score(all_scores)
    print("\nFine-tuned Task 2 Continuous Alert Score (per LSI):")
    for k, v in per_lsi.items():
        print(f"  {k}: {v}")
    print(f"\nFine-tuned averaged Continuous Alert Score: {avg}")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    classifier.base.save_pretrained(OUTPUTS_DIR / "task2_lora_adapter")
    print(f"\nSaved LoRA adapter to {OUTPUTS_DIR / 'task2_lora_adapter'}")


if __name__ == "__main__":
    main()
