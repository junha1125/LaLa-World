from typing import List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class FrozenTextEncoder(nn.Module):
    """Frozen Qwen3-0.6B last hidden states, padded to a fixed `max_length` (the DiT RoPE table is sized at build time)."""

    def __init__(self, model_path: str, max_length: int = 96, dtype=torch.bfloat16):
        super().__init__()
        self.model_path = model_path
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_path, dtype=dtype).eval()
        self.model.requires_grad_(False)
        self.feature_dim = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, texts: List[str]):
        device = next(self.model.parameters()).device
        tok = self.tokenizer(texts, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        ids = tok["input_ids"].to(device)
        mask = tok["attention_mask"].to(device)
        attn_in = mask.clone()
        attn_in[:, 0] = 1  # "" has no real token; keep one attended key so the attention row is finite
        out = self.model(input_ids=ids, attention_mask=attn_in, use_cache=False)
        return {"tokens": out.last_hidden_state, "attention_mask": mask}
