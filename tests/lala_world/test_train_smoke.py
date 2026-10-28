"""config -> app.lala_world.train.main -> finite loss, checkpoint, val; grad accumulation equivalence."""

import os

import torch

from tests.conftest import QWEN35, SIGLIP2, needs_qwen
from tests.lala_world.test_data import write_pair_jsonl, write_traj_jsonl

pytestmark = needs_qwen


def base_args(tmp_path, folder, sources, **opt):
    optimization = {"epochs": 1, "lr": 1e-4, "start_lr": 1e-5, "warmup_steps": 2, "clip_grad": 1.0}
    optimization.update(opt)
    return {
        "app": "lala_world",
        "folder": str(tmp_path / folder),
        "meta": {"seed": 0, "dtype": "bfloat16", "quick_debug": True, "log_freq": 1},
        "data": {
            "sources": sources,
            "val_groups": {"syngroup": ["syn"]},
            "max_text_tokens": 32,
            "batch_size": 4,
            "num_workers": 0,
        },
        "model": {
            "vision": {"kind": "siglip2", "model_id": SIGLIP2},
            "llm": {"kind": "qwen3_5", "model_id": QWEN35, "num_layers": 2},
        },
        "loss": {"kind": "wm_regression", "l2_weight": 1.0},
        "optimization": optimization,
        "logging": {"use_wandb": False},
    }


def test_train_smoke_mixed_sources(tmp_path):
    from app.lala_world.train import main

    sources = [
        {"kind": "traj_jsonl", "path": str(write_traj_jsonl(tmp_path / "a")), "clip_len": 4},
        {"kind": "jsonl", "path": str(write_pair_jsonl(tmp_path / "b"))},
    ]
    result = main(base_args(tmp_path, "run", sources))
    assert torch.isfinite(torch.tensor(result["final_train_loss"]))
    ckpt_path = tmp_path / "run" / "lala-latest.pth.tar"
    assert ckpt_path.is_symlink() and os.readlink(ckpt_path) == "lala-e1.pth.tar"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 1 and any(k.startswith("predictor.") for k in ckpt["model"])
    assert (tmp_path / "run" / "val.jsonl").exists() and (tmp_path / "run" / "train.csv").exists()
    # the checkpoint reloads through the eval path
    from app.lala_world.eval_common import load_model

    encoder, predictor, cfg, epoch = load_model(str(ckpt_path), torch.device("cuda"))
    assert epoch == 1 and predictor.n_vis == encoder.num_patches == 196


def test_rolling_checkpoints_and_samples_per_epoch(tmp_path):
    from app.lala_world.train import main

    args = base_args(
        tmp_path,
        "run_roll",
        [{"kind": "jsonl", "path": str(write_pair_jsonl(tmp_path / "p"))}],
        epochs=5,
        samples_per_epoch=8,
    )
    args["meta"].update(quick_debug=False, log_freq=100, eval_freq=100, save_every_freq=2, save_keep_last=2)
    result = main(args)
    assert result["ipe"] == 2
    run = tmp_path / "run_roll"
    assert sorted(p.name for p in run.glob("lala-e*.pth.tar")) == ["lala-e4.pth.tar", "lala-e5.pth.tar"]
    assert os.readlink(run / "lala-latest.pth.tar") == "lala-e5.pth.tar"


def test_grad_accum_matches_big_batch(tmp_path):
    from app.lala_world.train import main

    jsonl = str(write_pair_jsonl(tmp_path / "p"))
    runs = {}
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        for name, bs, accum in (("big", 8, 1), ("accum", 4, 2), ("small", 4, 1)):
            args = base_args(
                tmp_path,
                f"ga_{name}",
                [{"kind": "jsonl", "path": jsonl}],
                epochs=2,
                samples_per_epoch=32,
                grad_accum_steps=accum,
                lr=1e-3,
                start_lr=1e-4,
                clip_grad=0.0,
            )
            args["meta"].update(dtype="float32", quick_debug=False, log_freq=100, eval_freq=100)
            args["data"]["batch_size"] = bs
            result = main(args)
            runs[name] = (
                result,
                torch.load(tmp_path / f"ga_{name}" / "lala-latest.pth.tar", map_location="cpu", weights_only=False),
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    assert runs["big"][0]["ipe"] == runs["accum"][0]["ipe"] == 4

    def diff(a, b):
        return max((a["model"][k].float() - b["model"][k].float()).abs().max().item() for k in a["model"])

    equiv, control = diff(runs["big"][1], runs["accum"][1]), diff(runs["big"][1], runs["small"][1])
    # the linear-attention kernels round differently per batch size, so the accumulated run drifts by ~3e-3 at this
    # learning rate; a 1/N scaling bug would move it by at least as much as the genuinely different trajectory
    assert equiv < 1e-2, (equiv, control)
    assert control > 2 * max(equiv, 1e-6), (equiv, control)
