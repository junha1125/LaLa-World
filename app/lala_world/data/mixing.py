"""data.sources entries -> datasets, concatenated. Batch composition follows source sizes; a source's
share is set with its `max_samples` (seeded subset, -1 = all)."""

from torch.utils.data import ConcatDataset

from app.lala_world.data.pair_jsonl import PairJsonlDataset
from app.lala_world.data.policy_jsonl import PolicyJsonlDataset
from app.lala_world.data.traj_jsonl import TrajJsonlDataset

KINDS = {"jsonl": PairJsonlDataset, "traj_jsonl": TrajJsonlDataset, "policy_jsonl": PolicyJsonlDataset}


def build_dataset(source, split, image_size, mean, std):
    source = dict(source)
    kind = source.pop("kind")
    if kind not in KINDS:
        raise ValueError(f"unknown source kind {kind!r}; known: {sorted(KINDS)}")
    if source.get("max_samples", None) is not None and source["max_samples"] <= 0:
        source["max_samples"] = None
    return KINDS[kind](source.pop("path"), split=split, image_size=image_size, mean=mean, std=std, **source)


def build_concat_dataset(sources, split, image_size, mean, std):
    if not sources:
        raise ValueError("data.sources is empty")
    datasets = [build_dataset(s, split, image_size, mean, std) for s in sources]
    return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
