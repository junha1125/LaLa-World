"""Image stream over the LaLa-World data mix: every item carries 2 frames that are flattened into the image batch."""

import os

import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from app.lala_world.data import build_dataset
from src.utils.yaml_utils import expand_env_vars

IDENTITY_MEAN = (0.0, 0.0, 0.0)
IDENTITY_STD = (1.0, 1.0, 1.0)


def load_sources(path):
    os.environ.setdefault("LALA_DATA_ROOT", "data")
    with open(path) as f:
        return expand_env_vars(yaml.safe_load(f))["sources"]


def source_name(source):
    name = source["path"].rsplit("/", 1)[-1].replace(".jsonl", "")
    if source.get("kind") == "policy_jsonl":
        return f"{name}_{'_'.join(source.get('views', ['primary']))}"
    return name


def build_sources(sources, split, image_size):
    for s in sources:
        if s.get("kind") == "policy_jsonl" and len(s.get("views", ["primary"])) != 1:
            raise ValueError(f"policy_jsonl source {source_name(s)} needs exactly one view, got {s['views']}")
    return [build_dataset(s, split, image_size, IDENTITY_MEAN, IDENTITY_STD) for s in sources]


def build_train_dataset(sources, image_size):
    datasets = build_sources(sources, "train", image_size)
    return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]


def build_val_subsets(sources, image_size, per_source, seed=0):
    out = []
    for s, ds in zip(sources, build_sources(sources, "val", image_size)):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[: min(per_source, len(ds))].sort().values.tolist()
        out.append((source_name(s), Subset(ds, idx)))
    return out


def item_images(b):
    if "future_frames" in b:
        return list(b["frames"]) + list(b["future_frames"])
    if "frames" in b:
        return list(b["frames"]) + [b["target"]]
    return [b["state"], b["target"]]


def collate_images(batch):
    return torch.stack([im for b in batch for im in item_images(b)], 0)


def make_train_loader(dataset, batch_items, num_workers, rank, world_size, seed):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_items,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_images,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return loader, sampler
