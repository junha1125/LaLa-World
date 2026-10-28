"""One JSONL line = one (state image, action text, target image) pair."""

import json
import random
import zlib

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_IMAGE_SIZE = 224
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)
VAL_MOD = 50  # crc32(id) % 50 == 0 -> val (~2%)


def is_val_id(sample_id: str, val_mod: int = VAL_MOD) -> bool:
    return zlib.crc32(sample_id.encode("utf-8")) % val_mod == 0


def default_transform(image_size=DEFAULT_IMAGE_SIZE, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    return T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize(mean, std)])


class PairJsonlDataset(Dataset):
    def __init__(self, jsonl_path, split="train", image_size=DEFAULT_IMAGE_SIZE, mean=DEFAULT_MEAN, std=DEFAULT_STD,
                 transform=None, val_mod=VAL_MOD, domain_prefix=True, action_field="action", max_samples=None,
                 subsample_seed=0):
        if split not in ("train", "val", "all"):
            raise ValueError(f"split must be train|val|all, got {split!r}")
        records = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        if split != "all":
            records = [r for r in records if is_val_id(r["id"], val_mod) == (split == "val")]
        if not records:
            raise ValueError(f"no records for split={split!r} in {jsonl_path}")
        if max_samples is not None and 0 < max_samples < len(records):
            keep = sorted(random.Random(subsample_seed).sample(range(len(records)), int(max_samples)))
            records = [records[i] for i in keep]
        self.jsonl_path = str(jsonl_path)
        self.split = split
        self.records = records
        self.domain_prefix = bool(domain_prefix)
        self.action_field = action_field
        self.transform = transform if transform is not None else default_transform(image_size, mean, std)

    def __len__(self):
        return len(self.records)

    def _load(self, path, rec_id):
        try:
            return self.transform(Image.open(path).convert("RGB"))
        except (OSError, FileNotFoundError) as e:
            raise RuntimeError(f"failed to load image for id={rec_id}: {path}") from e

    def __getitem__(self, i):
        r = self.records[i]
        dataset = r.get("dataset", "unknown")
        raw = r[self.action_field]
        return {
            "state": self._load(r["state"], r["id"]),
            "action": f"[{dataset}] {raw}" if self.domain_prefix else raw,
            "target": self._load(r["target"], r["id"]),
            "meta": {"dataset": dataset, "id": r["id"]},
        }
