import json

import pytest
import torch
from PIL import Image

from app.lala_world.data import PairJsonlDataset, TrajCollator, TrajJsonlDataset, build_dataset, is_val_id
from app.lala_world.losses import build_loss
from tests.conftest import QWEN35, needs_qwen

F = 16  # frames per synthetic episode -> F - 5 transitions


def _frame_id(img):
    return round(((float(img[0].mean()) * 0.5) + 0.5) * 255 / 8)


def write_traj_jsonl(tmp_path, n_ep=40, dataset="syn"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    frames = []
    for k in range(F):
        p = tmp_path / f"frame_{k:02d}.png"
        Image.new("RGB", (32, 32), (k * 8, 0, 0)).save(p)
        frames.append(str(p))
    path = tmp_path / "traj.jsonl"
    with open(path, "w") as f:
        for i in range(n_ep):
            rec = {
                "dataset": dataset,
                "id": f"{dataset}_traj/ep{i:04d}",
                "frames": frames,
                "actions_semantic_no_proprio": [f"chunk {i}/{u}" for u in range(F - 5)],
                "actions_lang6_no_proprio": [f"lang {i}/{u}" for u in range(F - 5)],
            }
            f.write(json.dumps(rec) + "\n")
    return path


def write_pair_jsonl(tmp_path, n=200):
    tmp_path.mkdir(parents=True, exist_ok=True)
    img = tmp_path / "img.png"
    Image.new("RGB", (48, 64), (255, 0, 0)).save(img)
    path = tmp_path / "pairs.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "dataset": "edit",
                        "id": f"edit/{i:06d}",
                        "state": str(img),
                        "action": f"edit number {i}",
                        "target": str(img),
                    }
                )
                + "\n"
            )
    return path


def test_pair_dataset_split_and_items(tmp_path):
    path = write_pair_jsonl(tmp_path, n=500)
    train, val, everything = (PairJsonlDataset(path, split=s) for s in ("train", "val", "all"))
    train_ids, val_ids = {r["id"] for r in train.records}, {r["id"] for r in val.records}
    assert train_ids.isdisjoint(val_ids) and len(train_ids) + len(val_ids) == len(everything.records) == 500
    assert 1 <= len(val_ids) <= 30 and all(is_val_id(s) for s in val_ids)
    item = everything[0]
    assert item["state"].shape == (3, 224, 224) and item["action"] == "edit number 0"
    assert PairJsonlDataset(path, split="all", domain_prefix=True)[0]["action"] == "[edit] edit number 0"
    assert len(PairJsonlDataset(path, split="all", max_samples=20)) == 20


def test_traj_dataset_enumeration_depth_and_split(tmp_path):
    path = write_traj_jsonl(tmp_path, n_ep=200)
    assert len(TrajJsonlDataset(path, split="all")) == 200 * (F - 5)
    assert len(TrajJsonlDataset(path, split="all", window_stride=5)) == 200 * 3
    train, val = TrajJsonlDataset(path, split="train"), TrajJsonlDataset(path, split="val")
    assert {r["id"] for r in train.records}.isdisjoint({r["id"] for r in val.records}) and len(val) > 0
    ds = TrajJsonlDataset(path, split="all", clip_len=8)
    item0 = ds[0]
    assert len(item0["frames"]) == 1 and item0["action"] == ["chunk 0/0"] and _frame_id(item0["target"]) == 5
    torch.manual_seed(0)
    seen = set()
    for _ in range(40):  # u = 10 -> d in {1, 2, 3}
        item = ds[10]
        d = len(item["frames"])
        seen.add(d)
        starts = list(range(10 - 5 * (d - 1), 11, 5))
        assert item["action"] == [f"chunk 0/{s}" for s in starts]
        assert [_frame_id(fr) for fr in item["frames"]] == starts and _frame_id(item["target"]) == 15
    assert seen == {1, 2, 3}
    torch.manual_seed(0)
    a = val[len(val) - 1]
    torch.manual_seed(1)
    assert val[len(val) - 1]["action"] == a["action"]  # val depth is deterministic


def test_traj_window_1_and_field(tmp_path):
    frames = []
    for k in range(F):
        p = tmp_path / f"w1_{k:02d}.png"
        Image.new("RGB", (32, 32), (k * 8, 0, 0)).save(p)
        frames.append(str(p))
    path = tmp_path / "w1.jsonl"
    path.write_text(
        json.dumps(
            {
                "dataset": "rope",
                "id": "rope_traj/ep0000",
                "frames": frames,
                "actions_semantic_no_proprio": [f"push {u}" for u in range(F - 1)],
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="frames but"):
        TrajJsonlDataset(path, split="all")
    ds = TrajJsonlDataset(path, split="all", window=1, clip_len=4)
    assert len(ds) == F - 1 and ds[0]["action"] == ["push 0"] and _frame_id(ds[0]["target"]) == 1
    path2 = write_traj_jsonl(tmp_path / "b", n_ep=1)
    assert TrajJsonlDataset(path2, split="all", action_field="actions_lang6_no_proprio")[0]["action"] == ["lang 0/0"]
    assert TrajJsonlDataset(path2, split="all", domain_prefix=True)[0]["action"] == ["[syn] chunk 0/0"]


def test_max_samples_and_source_config(tmp_path):
    path = write_traj_jsonl(tmp_path, n_ep=10)
    full = TrajJsonlDataset(path, split="all")
    sub = TrajJsonlDataset(path, split="all", max_samples=17, subsample_seed=0)
    same = TrajJsonlDataset(path, split="all", max_samples=17, subsample_seed=0)
    assert len(sub) == len(same) == 17 and len(sub.records) <= 10
    assert [int(x) for x in sub._start] == [int(x) for x in same._start]
    assert len(TrajJsonlDataset(path, split="all", max_samples=-1)) == len(full)
    src = {"kind": "traj_jsonl", "path": str(path), "max_samples": 12}
    assert len(build_dataset(src, "all", 224, (0.5,) * 3, (0.5,) * 3)) == 12
    assert len(
        build_dataset({"kind": "traj_jsonl", "path": str(path), "max_samples": -1}, "all", 224, (0.5,) * 3, (0.5,) * 3)
    ) == len(full)
    with pytest.raises(ValueError, match="unknown source kind"):
        build_dataset({"kind": "imagenet_aug", "path": "x"}, "all", 224, (0.5,) * 3, (0.5,) * 3)


@needs_qwen
def test_traj_collator_mixed_batch_and_dropout(tmp_path):
    from transformers import AutoTokenizer

    from app.lala_world.models.hub import resolve_model_id

    tok = AutoTokenizer.from_pretrained(resolve_model_id(QWEN35))
    ds = TrajJsonlDataset(write_traj_jsonl(tmp_path, n_ep=1), split="all", clip_len=8)
    torch.manual_seed(3)
    traj = next(it for it in (ds[10] for _ in range(60)) if len(it["frames"]) == 3)
    pair = {
        "state": ds[0]["frames"][0],
        "action": "plain pair text",
        "target": ds[0]["target"],
        "meta": {"dataset": "syn", "id": "syn/pair"},
    }
    batch = TrajCollator(tok, max_text_tokens=32)([traj, pair])
    assert batch["frames"].shape[:2] == (2, 3) and batch["ctx_len"].tolist() == [3, 1]
    assert batch["target"].shape == (2, 3, 224, 224)
    real = batch["text_mask"].sum(dim=2)
    assert (real[0] > 0).all() and real[1, 0] > 0 and (real[1, 1:] == 0).all()
    assert torch.equal(batch["frames"][1, 1:], torch.zeros_like(batch["frames"][1, 1:]))
    for j, expected in enumerate(traj["action"]):
        ids = batch["input_ids"][0, j][batch["text_mask"][0, j].bool()]
        assert tok.decode(ids, skip_special_tokens=True).strip() == expected
    dropped = TrajCollator(tok, max_text_tokens=32, text_dropout_p=1.0)([traj, pair])
    assert all(all(t == "" for t in a) for a in dropped["action"])


def test_losses():
    import torch.nn.functional as F_

    torch.manual_seed(0)
    pred, target = torch.randn(2, 6, 8), torch.randn(2, 6, 8)
    loss, logs = build_loss({"kind": "wm_regression", "l2_weight": 1.0})(pred, target)
    assert torch.allclose(loss, F_.mse_loss(pred, target)) and set(logs) == {"loss_l2"}
    loss, _ = build_loss({"kind": "wm_regression", "cos_weight": 1.0, "l1_weight": 1.0, "l2_weight": 0.0})(
        pred, target
    )
    assert torch.allclose(loss, (1 - F_.cosine_similarity(pred, target, dim=-1)).mean() + F_.l1_loss(pred, target))
    loss, _ = build_loss({"kind": "ijepa"})(pred, target)
    assert torch.allclose(loss, F_.smooth_l1_loss(pred, F_.layer_norm(target, (8,))))
    with pytest.raises(ValueError):
        build_loss({"kind": "wm_regression", "l2_weight": 0.0})
    with pytest.raises(ValueError, match="unknown loss"):
        build_loss({"kind": "mystery"})
