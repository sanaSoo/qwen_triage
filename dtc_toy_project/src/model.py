"""
Shared model architecture for Task 1 and Task 2.

Deliberately NOT free-text JSON generation -- small models are unreliable at emitting
exactly-shaped, well-formed numeric vectors. Instead: base model -> mean-pooled hidden
state -> linear classification head -> sigmoid. This guarantees valid output shape by
construction and trains cleanly with BCELoss.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
)


def get_device():
    """Prefer CUDA (NVIDIA), then MPS (Apple Silicon), then fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LSIClassifier(nn.Module):
    def __init__(self, base_model, hidden_size, n_outputs):
        super().__init__()
        self.base = base_model
        self.head = nn.Linear(hidden_size, n_outputs)

    def forward(self, input_ids, attention_mask):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = out.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        logits = self.head(pooled)
        return torch.sigmoid(logits)


def build_model_and_tokenizer(n_outputs):
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    base_model = get_peft_model(base_model, lora_config)

    classifier = LSIClassifier(base_model, hidden_size=base_model.config.hidden_size,
                                n_outputs=n_outputs)
    classifier = classifier.to(device)
    return classifier, tokenizer, device
