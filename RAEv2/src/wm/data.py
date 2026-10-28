"""(state pixels, text, target pixels) batches on the LaLa-World data contract; every traj source is a single transition."""

import copy
import os

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from wm import LALA_ROOT
from wm.lala_data import build_pair_dataset

IDENTITY_MEAN = (0.0, 0.0, 0.0)  # pixels in [0, 1]; RAE.encode applies the encoder's own normalisation
IDENTITY_STD = (1.0, 1.0, 1.0)


def resolve_sources(sources, root=None):
    """Relative data paths are joined with data.root; a relative root is relative to the LaLa-World repository.
    LALA_DATA_ROOT (or LALA_DATA, the planning scripts' name) overrides data.root."""
    root = os.environ.get("LALA_DATA_ROOT") or os.environ.get("LALA_DATA") or root
    if root and not os.path.isabs(root):
        root = os.path.join(LALA_ROOT, root)
    out = []
    for s in sources:
        s = copy.deepcopy(dict(s))
        if "path" in s and not os.path.isabs(s["path"]):
            if not root:
                raise ValueError(f"relative data path {s['path']!r} needs data.root or LALA_DATA_ROOT")
            s["path"] = os.path.join(root, s["path"])
        if s.get("kind") == "traj_jsonl":
            if s.get("clip_len", 2) != 2:
                raise ValueError(f"traj source {s['path']}: clip_len must be 2 (single transition)")
            s["clip_len"] = 2
        out.append(s)
    return out


def build_sources(sources, split, image_size):
    return [build_pair_dataset(s, split, image_size, IDENTITY_MEAN, IDENTITY_STD) for s in sources]


def build_train_dataset(sources, image_size):
    datasets = build_sources(sources, "train", image_size)
    return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]


def source_name(source):
    base = source.get("path", "").rsplit("/", 1)[-1].replace(".jsonl", "")
    field = source.get("action_field")
    if field and source.get("kind") == "traj_jsonl":
        base += "." + field.replace("actions_", "").replace("_no_proprio", "")
    return base


def build_val_subsets(sources, image_size, per_source, seed=0):
    """Fixed seeded val subset per source, kept separate so in-batch retrieval stays within one domain."""
    out = []
    for s, ds in zip(sources, build_sources(sources, "val", image_size)):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[: min(per_source, len(ds))].sort().values.tolist()
        out.append((source_name(s), Subset(ds, idx)))
    return out


def collate_pairs(batch):
    states, texts, targets, metas = [], [], [], []
    for b in batch:
        if "frames" in b:
            if len(b["frames"]) != 1:
                raise ValueError(f"expected a single context frame, got {len(b['frames'])} ({b['meta']})")
            states.append(b["frames"][0])
            texts.append(b["action"][0])
        else:
            states.append(b["state"])
            texts.append(b["action"])
        targets.append(b["target"])
        metas.append(b["meta"])
    return {"state": torch.stack(states, 0), "target": torch.stack(targets, 0), "text": texts, "meta": metas}


def make_train_loader(dataset, batch_size, num_workers, rank, world_size, seed):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True,
                        drop_last=True, collate_fn=collate_pairs, persistent_workers=num_workers > 0,
                        prefetch_factor=4 if num_workers > 0 else None)
    return loader, sampler


def make_val_loader(subset, batch_size, num_workers, rank, world_size):
    sampler = DistributedSampler(subset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    return DataLoader(subset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True,
                      drop_last=False, collate_fn=collate_pairs)
