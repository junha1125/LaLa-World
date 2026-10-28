"""One JSONL line = one episode (dense frames + per-transition action texts); one item = one (context, target) window."""

import json
import zlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .pair_jsonl import DEFAULT_IMAGE_SIZE, DEFAULT_MEAN, DEFAULT_STD, VAL_MOD, default_transform, is_val_id

WINDOW = 5  # frames per action chunk


class TrajJsonlDataset(Dataset):
    def __init__(self, jsonl_path, split="train", image_size=DEFAULT_IMAGE_SIZE, mean=DEFAULT_MEAN, std=DEFAULT_STD,
                 transform=None, val_mod=VAL_MOD, domain_prefix=True, action_field="actions_semantic", clip_len=8,
                 window_stride=1, window=WINDOW, max_samples=None, subsample_seed=0):
        if split not in ("train", "val", "all"):
            raise ValueError(f"split must be train|val|all, got {split!r}")
        if clip_len < 2 or window_stride < 1 or window < 1:
            raise ValueError(f"bad clip_len/window_stride/window: {clip_len}/{window_stride}/{window}")
        records = []
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if split != "all" and is_val_id(r["id"], val_mod) != (split == "val"):
                    continue
                texts = r[action_field]
                frames = r["frames"]
                if len(texts) != len(frames) - window:
                    raise ValueError(f"{r['id']}: {len(frames)} frames but {len(texts)} chunk texts")
                records.append({"dataset": r.get("dataset", "unknown"), "id": r["id"], "frames": frames, "texts": texts})
        if not records:
            raise ValueError(f"no records for split={split!r} in {jsonl_path}")
        self.jsonl_path = str(jsonl_path)
        self.split = split
        self.records = records
        self.domain_prefix = bool(domain_prefix)
        self.action_field = action_field
        self.clip_len = int(clip_len)
        self.window_stride = int(window_stride)
        self.window = int(window)
        self.transform = transform if transform is not None else default_transform(image_size, mean, std)
        rec_idx, starts = [], []
        for k, r in enumerate(records):
            n = len(r["texts"])
            rec_idx.append(np.full((len(range(0, n, window_stride)),), k, dtype=np.int32))
            starts.append(np.arange(0, n, window_stride, dtype=np.int32))
        self._rec_idx = np.concatenate(rec_idx)
        self._start = np.concatenate(starts)
        if max_samples is not None and 0 < max_samples < len(self._start):
            keep = np.sort(np.random.default_rng(subsample_seed).choice(len(self._start), int(max_samples), replace=False))
            self._rec_idx, self._start = self._rec_idx[keep], self._start[keep]
            used = np.unique(self._rec_idx)
            remap = np.full(len(records), -1, dtype=np.int32)
            remap[used] = np.arange(len(used), dtype=np.int32)
            self._rec_idx = remap[self._rec_idx]
            self.records = [records[k] for k in used.tolist()]

    def __len__(self):
        return len(self._start)

    def _load(self, path, rec_id):
        try:
            return self.transform(Image.open(path).convert("RGB"))
        except (OSError, FileNotFoundError) as e:
            raise RuntimeError(f"failed to load image for id={rec_id}: {path}") from e

    def _depth(self, rec_id, u):
        d_max = min(self.clip_len - 1, u // self.window + 1)
        if d_max == 1:
            return 1
        if self.split == "val":
            return 1 + zlib.crc32(f"{rec_id}/{u:06d}".encode("utf-8")) % d_max
        return int(torch.randint(1, d_max + 1, ()).item())

    def __getitem__(self, i):
        r = self.records[self._rec_idx[i]]
        u = int(self._start[i])
        d = self._depth(r["id"], u)
        starts = range(u - self.window * (d - 1), u + 1, self.window)
        prefix = f"[{r['dataset']}] " if self.domain_prefix else ""
        return {
            "frames": [self._load(r["frames"][s], r["id"]) for s in starts],
            "action": [prefix + r["texts"][s] for s in starts],
            "target": self._load(r["frames"][u + self.window], r["id"]),
            "meta": {"dataset": r["dataset"], "id": f"{r['id']}/{u:06d}"},
        }
