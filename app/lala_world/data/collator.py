import torch


class TrajCollator:
    """Multi-frame batches: frames [B,C,3,H,W] (zero beyond ctx_len), ctx_len [B], input_ids / text_mask [B,C,L],
    target [B,3,H,W]. Pair items ({state, action: str, target}) become d = 1 samples."""

    def __init__(self, tokenizer, max_text_tokens=64, text_dropout_p=0.0):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        self.text_dropout_p = text_dropout_p

    def __call__(self, batch):
        frames = [b["frames"] if "frames" in b else [b["state"]] for b in batch]
        actions = [list(b["action"]) if "frames" in b else [b["action"]] for b in batch]
        if self.text_dropout_p > 0:
            drop = torch.rand(len(batch)) < self.text_dropout_p
            actions = [[""] * len(a) if drop[i] else a for i, a in enumerate(actions)]
        ctx_len = torch.tensor([len(f) for f in frames], dtype=torch.long)
        B, C = len(batch), int(ctx_len.max())
        tok = self.tokenizer(
            [t for a in actions for t in a],
            padding="longest",
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        L = tok["input_ids"].shape[1]
        input_ids = torch.full((B, C, L), self.tokenizer.pad_token_id, dtype=tok["input_ids"].dtype)
        text_mask = torch.zeros(B, C, L, dtype=tok["attention_mask"].dtype)
        pos = 0
        for i, a in enumerate(actions):
            input_ids[i, : len(a)] = tok["input_ids"][pos : pos + len(a)]
            text_mask[i, : len(a)] = tok["attention_mask"][pos : pos + len(a)]
            pos += len(a)
        frames_px = torch.zeros(B, C, *frames[0][0].shape)
        for i, f in enumerate(frames):
            frames_px[i, : len(f)] = torch.stack(f, 0)
        return {
            "frames": frames_px,
            "ctx_len": ctx_len,
            "target": torch.stack([b["target"] for b in batch], 0),
            "input_ids": input_ids,
            "text_mask": text_mask,
            "action": actions,
            "meta": [b["meta"] for b in batch],
        }


class PolicyCollator:
    """Batches PolicyJsonlDataset items (all with the same number of views)."""

    def __init__(self, tokenizer, max_text_tokens=64):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens

    def __call__(self, batch):
        n_views = {len(b["frames"]) for b in batch}
        if len(n_views) != 1:
            raise ValueError(f"mixed view counts in one batch: {sorted(n_views)}")
        tasks = [b["task"] for b in batch]
        tok = self.tokenizer(
            tasks, padding="longest", truncation=True, max_length=self.max_text_tokens, return_tensors="pt"
        )
        return {
            "frames": torch.stack([torch.stack(b["frames"], 0) for b in batch], 0),
            "future_frames": torch.stack([torch.stack(b["future_frames"], 0) for b in batch], 0),
            "proprio": torch.stack([b["proprio"] for b in batch], 0).float(),
            "future_proprio": torch.stack([b["future_proprio"] for b in batch], 0).float(),
            "actions": torch.stack([b["actions"] for b in batch], 0).float(),
            "value": torch.tensor([b["value"] for b in batch], dtype=torch.float32),
            "mode": torch.tensor([b["mode"] for b in batch], dtype=torch.long),
            "input_ids": tok["input_ids"],
            "text_mask": tok["attention_mask"],
            "task": tasks,
            "meta": [b["meta"] for b in batch],
        }
