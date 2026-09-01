"""Fine-tune the LSIClassifier (Qwen2.5-1.5B + LoRA) on Task 1, Run 1."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import (
    DATA_FILE, OUTPUTS_DIR, LSI_KEYS, TASK1_OUTPUT_KEYS,
    MAX_SEQ_LEN, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
)
from .data_common import load_cases
from .data_task1 import (
    load_run1_examples, serialize_predict, existence_labels, stratified_split,
)
from .model import build_model_and_tokenizer
from .metrics import existence_score


class Task1Dataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=MAX_SEQ_LEN):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = serialize_predict(ex["predict"])
        enc = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        labels = existence_labels(ex["response"])
        y = torch.tensor([labels[k] for k in TASK1_OUTPUT_KEYS], dtype=torch.float)
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), y


def main():
    cases = load_cases(DATA_FILE)
    examples = load_run1_examples(cases)
    print(f"Loaded {len(examples)} Task 1 Run 1 examples.")

    train_examples, val_examples = stratified_split(examples)
    print(f"Train: {len(train_examples)} cases, Val: {len(val_examples)} cases "
          f"(stratified split -- each LSI with enough positives gets some in both sides)")

    classifier, tokenizer, device = build_model_and_tokenizer(n_outputs=len(TASK1_OUTPUT_KEYS))

    train_ds = Task1Dataset(train_examples, tokenizer)
    val_ds = Task1Dataset(val_examples, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    print(f"Starting training: {len(train_loader)} batches/epoch, {NUM_EPOCHS} epochs")
    OUTPUTS_DIR.mkdir(exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        classifier.train()
        total_loss = 0.0
        for batch_idx, (input_ids, attn_mask, y) in enumerate(train_loader):
            input_ids, attn_mask, y = input_ids.to(device), attn_mask.to(device), y.to(device)
            optimizer.zero_grad()
            preds = classifier(input_ids, attn_mask)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  epoch {epoch+1} batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={loss.item():.4f}", flush=True)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} — train loss: {total_loss / len(train_loader):.4f}",
              flush=True)

        # Safety checkpoint after every epoch -- if the run gets interrupted overnight
        # (crash, power loss, OS reclaiming memory), you still have a usable adapter
        # from whichever epoch last completed, instead of losing the whole run.
        checkpoint_dir = OUTPUTS_DIR / f"task1_lora_adapter_epoch{epoch+1}"
        classifier.base.save_pretrained(checkpoint_dir)
        print(f"  Saved checkpoint: {checkpoint_dir}", flush=True)

    # --- Evaluate on held-out split using the real Existence Score ---
    classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attn_mask, y in val_loader:
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            preds = classifier(input_ids, attn_mask)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    y_true = {k: all_labels[:, i] for i, k in enumerate(LSI_KEYS)}
    y_score = {k: all_preds[:, i] for i, k in enumerate(LSI_KEYS)}
    per_lsi, avg = existence_score(y_true, y_score)

    print("\nFine-tuned Task 1 Existence Score (per LSI):")
    for k, v in per_lsi.items():
        print(f"  {k}: {v if v is not None else 'undefined (no pos/neg variation)'}")
    print(f"\nFine-tuned averaged Existence Score: {avg}")
    print("Compare this against src/baseline.py's output.")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    classifier.base.save_pretrained(OUTPUTS_DIR / "task1_lora_adapter_final")
    print(f"\nSaved final LoRA adapter to {OUTPUTS_DIR / 'task1_lora_adapter_final'}")


if __name__ == "__main__":
    main()