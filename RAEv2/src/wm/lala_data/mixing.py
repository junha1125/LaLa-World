from .pair_jsonl import PairJsonlDataset
from .traj_jsonl import TrajJsonlDataset


def build_pair_dataset(source, split, image_size, mean, std):
    source = dict(source)
    source.pop("weight", None)
    kind = source.pop("kind")
    if kind == "jsonl":
        return PairJsonlDataset(source.pop("path"), split=split, image_size=image_size, mean=mean, std=std, **source)
    if kind == "traj_jsonl":
        return TrajJsonlDataset(source.pop("path"), split=split, image_size=image_size, mean=mean, std=std, **source)
    raise ValueError(f"unknown source kind {kind!r} (jsonl | traj_jsonl)")
