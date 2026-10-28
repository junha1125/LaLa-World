"""Cosmos-Policy-style policy samples from episode JSONL (LIBERO / RoboCasa releases).

One line = one episode; images stay in the release HDF5 files and are read lazily:
  {"dataset", "source": "demo"|"rollout", "id", "task", "h5", "group", "keys": {view: h5 key}, "num_steps",
   "success", "actions": [[A] x T], "proprio": [[P] x T]}
One sample = one timestep t: current / future (t + chunk_size) frames per view, the action chunk a[t:t+K]
(last action repeated past the end), proprio, the Monte-Carlo value and the Cosmos conditioning mode
(0 policy, 1 world model, 2 value; `sample_mode: rollout` sources draw 1 / 2, policy sources are 0).
Actions / proprio are min-max scaled to [-1, 1] with `stats_path`.
"""


import io
import json
import math
import random
import zlib

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from app.lala_world.data.pair_jsonl import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MEAN,
    DEFAULT_STD,
    VAL_MOD,
    default_transform,
    is_val_id,
)

MODE_POLICY, MODE_WM, MODE_VALUE = 0, 1, 2
MODE_NAMES = ("policy", "wm", "value")


def load_stats(stats_path):
    stats = json.load(open(stats_path))
    out = {}
    for key in ("actions", "proprio"):
        lo = np.asarray(stats[f"{key}_min"], dtype=np.float32)
        hi = np.asarray(stats[f"{key}_max"], dtype=np.float32)
        if np.any(hi <= lo):
            raise ValueError(f"{stats_path}: {key} has max <= min in some dim: min {lo} max {hi}")
        out[key] = (lo, hi)
    return out


def scale_to_unit(x, lo, hi):
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def unscale_from_unit(y, lo, hi):
    return (y + 1.0) / 2.0 * (hi - lo) + lo


def monte_carlo_value(num_steps, t, success, gamma):
    """Cosmos compute_monte_carlo_returns: sparse terminal reward, discounted, rescaled to [-1, 1]."""
    return 2.0 * (gamma ** (num_steps - 1 - t)) * float(success) - 1.0


def cosmos_aug_params(stronger, rng):
    """One parameter set per SAMPLE (cosmos apply_image_aug), reused for every frame of the sample."""
    side = math.sqrt(0.9)  # RandomResizedCrop scale=(0.9, 0.9), ratio 1 -> square crop of 90% area
    return {
        "crop": (rng.random(), rng.random(), side),  # relative top-left, relative side
        "angle": rng.uniform(-5.0, 5.0) if stronger else 0.0,
        "brightness": rng.uniform(0.7, 1.3) if stronger else rng.uniform(0.8, 1.2),
        "contrast": rng.uniform(0.6, 1.4) if stronger else rng.uniform(0.8, 1.2),
        "saturation": rng.uniform(0.5, 1.5) if stronger else rng.uniform(0.8, 1.2),
        "hue": rng.uniform(-0.05, 0.05),
    }


def apply_cosmos_aug(img, p):
    """img: PIL RGB (H, W). Crop -> resize back to (H, W) -> rotate -> color jitter."""
    W, H = img.size
    fy, fx, side = p["crop"]
    h, w = int(round(H * side)), int(round(W * side))
    top, left = int(round(fy * (H - h))), int(round(fx * (W - w)))
    img = TF.resized_crop(img, top, left, h, w, [H, W], antialias=True)
    if p["angle"]:
        img = TF.rotate(img, p["angle"], expand=False)
    img = TF.adjust_brightness(img, p["brightness"])
    img = TF.adjust_contrast(img, p["contrast"])
    img = TF.adjust_saturation(img, p["saturation"])
    img = TF.adjust_hue(img, p["hue"])
    return img


class PolicyJsonlDataset(Dataset):
    """split: 'train' | 'val' | 'all'. val = stable ~2% of EPISODES (see is_val_id)."""

    def __init__(
        self,
        jsonl_path,
        split="train",
        image_size=DEFAULT_IMAGE_SIZE,
        mean=DEFAULT_MEAN,
        std=DEFAULT_STD,
        transform=None,
        val_mod=VAL_MOD,
        stats_path=None,
        chunk_size=16,
        views=("primary",),
        sample_mode="policy",
        gamma=0.99,
        balance_success=False,
        max_episodes_per_task=None,
        max_samples=None,
        subsample_seed=0,
        aug=False,
        aug_stronger=True,
        task_field="task",
        pv_pred=True,
    ):
        if split not in ("train", "val", "all"):
            raise ValueError(f"split must be train|val|all, got {split!r}")
        if sample_mode not in ("policy", "rollout"):
            raise ValueError(f"sample_mode must be policy|rollout, got {sample_mode!r}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if stats_path is None:
            raise ValueError("policy_jsonl needs stats_path (actions/proprio min-max json)")
        self.jsonl_path = str(jsonl_path)
        self.split = split
        self.chunk_size = int(chunk_size)
        self.views = tuple(views)
        self.sample_mode = sample_mode
        self.pv_pred = bool(pv_pred)
        self.gamma = float(gamma)
        self.aug = bool(aug) and split == "train"
        self.aug_stronger = bool(aug_stronger)
        self.task_field = task_field
        self.stats = load_stats(stats_path)
        self.transform = transform if transform is not None else default_transform(image_size, mean, std)

        records = []
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if split != "all" and is_val_id(r["id"], val_mod) != (split == "val"):
                    continue
                for v in self.views:
                    if v not in r["keys"]:
                        raise ValueError(
                            f"{r['id']}: views {list(self.views)} but record has keys {sorted(r['keys'])}"
                        )
                actions = np.asarray(r["actions"], dtype=np.float32)
                proprio = np.asarray(r["proprio"], dtype=np.float32)
                T = int(r["num_steps"])
                if actions.shape[0] != T or proprio.shape[0] != T:
                    raise ValueError(f"{r['id']}: num_steps {T} but actions {actions.shape} proprio {proprio.shape}")
                lo, hi = self.stats["actions"]
                if actions.shape[1] != lo.shape[0]:
                    raise ValueError(f"{r['id']}: action dim {actions.shape[1]} != stats dim {lo.shape[0]}")
                records.append(
                    {
                        "dataset": r.get("dataset", "unknown"),
                        "id": r["id"],
                        "task": r[task_field],
                        "h5": r["h5"],
                        "group": r.get("group", ""),
                        "keys": {v: r["keys"][v] for v in self.views},
                        "num_steps": T,
                        "success": bool(r.get("success", True)),
                        "actions": scale_to_unit(actions, *self.stats["actions"]),
                        "proprio": scale_to_unit(proprio, *self.stats["proprio"]),
                    }
                )
        if not records:
            raise ValueError(f"no records for split={split!r} in {jsonl_path}")
        if max_episodes_per_task and max_episodes_per_task > 0 and split == "train":
            records = self._cap_per_task(records, int(max_episodes_per_task), subsample_seed)
        self.records = records

        rec_idx, starts = [], []
        for k, r in enumerate(records):
            rec_idx.append(np.full((r["num_steps"],), k, dtype=np.int32))
            starts.append(np.arange(r["num_steps"], dtype=np.int32))
        self._rec_idx = np.concatenate(rec_idx)
        self._start = np.concatenate(starts)
        if balance_success:
            self._balance_success()
        if max_samples is not None and 0 < max_samples < len(self._start):
            keep = np.sort(
                np.random.default_rng(subsample_seed).choice(len(self._start), int(max_samples), replace=False)
            )
            self._rec_idx, self._start = self._rec_idx[keep], self._start[keep]
        self._h5 = {}  # lazily opened per worker process (never before the DataLoader fork)

    @staticmethod
    def _cap_per_task(records, n, seed):
        by_task = {}
        for r in records:
            by_task.setdefault(r["task"], []).append(r)
        rng = np.random.default_rng(seed)
        keep = []
        for task in sorted(by_task):
            eps = sorted(by_task[task], key=lambda r: r["id"])
            if len(eps) > n:
                eps = [eps[i] for i in sorted(rng.choice(len(eps), n, replace=False).tolist())]
            keep.extend(eps)
        return keep

    def _balance_success(self):
        succ = np.array([self.records[k]["success"] for k in self._rec_idx])
        n_s, n_f = int(succ.sum()), int((~succ).sum())
        if n_s == 0 or n_f == 0:
            return
        reps = int(round(n_s / n_f))  # repeat the rarer class's timesteps to ~match the other
        if reps > 1:
            fail = ~succ
            self._rec_idx = np.concatenate([self._rec_idx] + [self._rec_idx[fail]] * (reps - 1))
            self._start = np.concatenate([self._start] + [self._start[fail]] * (reps - 1))
        elif n_f > n_s:
            reps = int(round(n_f / n_s))
            if reps > 1:
                self._rec_idx = np.concatenate([self._rec_idx] + [self._rec_idx[succ]] * (reps - 1))
                self._start = np.concatenate([self._start] + [self._start[succ]] * (reps - 1))

    def __len__(self):
        return len(self._start)

    def _file(self, path):
        f = self._h5.get(path)
        if f is None:
            import h5py

            f = h5py.File(path, "r")
            self._h5[path] = f
        return f

    def _frame(self, rec, view, t):
        f = self._file(rec["h5"])
        key = rec["keys"][view] if not rec["group"] else f"{rec['group']}/{rec['keys'][view]}"
        buf = f[key][t]
        if getattr(buf, "ndim", 0) == 3:  # raw (H, W, 3) uint8 frame (e.g. the RoboCasa demo files)
            return Image.fromarray(np.asarray(buf)).convert("RGB")
        buf = buf.tobytes() if hasattr(buf, "tobytes") else bytes(buf)
        try:
            return Image.open(io.BytesIO(buf)).convert("RGB")
        except OSError as e:
            raise RuntimeError(f"bad JPEG for id={rec['id']} view={view} t={t}") from e

    def _mode(self, rec, t):
        if self.sample_mode == "policy":
            return MODE_POLICY
        if not self.pv_pred:  # no value slot in the model -> rollouts are world-model samples only
            return MODE_WM
        if self.split == "val":  # stable across evaluations
            return MODE_WM if zlib.crc32(f"{rec['id']}/{t:06d}".encode("utf-8")) % 2 == 0 else MODE_VALUE
        return MODE_WM if random.random() < 0.5 else MODE_VALUE

    def __getitem__(self, i):
        rec = self.records[self._rec_idx[i]]
        t = int(self._start[i])
        T, K = rec["num_steps"], self.chunk_size
        fut = min(t + K, T - 1)
        imgs = {("cur", v): self._frame(rec, v, t) for v in self.views}
        imgs.update({("fut", v): self._frame(rec, v, fut) for v in self.views})
        if self.aug:
            p = cosmos_aug_params(self.aug_stronger, random)
            imgs = {k: apply_cosmos_aug(im, p) for k, im in imgs.items()}
        chunk = rec["actions"][t : t + K]
        if chunk.shape[0] < K:
            chunk = np.concatenate([chunk, np.repeat(rec["actions"][-1:], K - chunk.shape[0], axis=0)], axis=0)
        return {
            "frames": [self.transform(imgs[("cur", v)]) for v in self.views],  # V x [3,H,W]
            "future_frames": [self.transform(imgs[("fut", v)]) for v in self.views],
            "proprio": torch.from_numpy(rec["proprio"][t].copy()),
            "future_proprio": torch.from_numpy(rec["proprio"][fut].copy()),
            "actions": torch.from_numpy(np.ascontiguousarray(chunk)),  # [K, A]
            "value": float(monte_carlo_value(T, t, rec["success"], self.gamma)),
            "mode": int(self._mode(rec, t)),
            "task": rec["task"],
            "meta": {"dataset": rec["dataset"], "id": f"{rec['id']}/{t:06d}", "success": rec["success"]},
        }
