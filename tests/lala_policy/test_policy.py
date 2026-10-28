"""LaLa-Policy: layout, modes, padding invariance, world-model loading, flow head; policy data fixture + collator."""

import io
import json
import os

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from app.lala_world.data import PolicyCollator, PolicyJsonlDataset, is_val_id
from app.lala_world.data.policy_jsonl import MODE_POLICY, MODE_VALUE, MODE_WM, monte_carlo_value
from tests.conftest import QWEN35, SIGLIP2, needs_qwen

A_DIM, P_DIM = 7, 9
GRID, N_VIS, VIS_DIM = (2, 2), 4, 16
K, V = 4, 2


def _jpeg(color, size=32):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="JPEG")
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def make_fixture(tmp_path, T=12, n_demos=6):
    """Demo JSONL (policy samples) + rollout JSONL (wm / value samples) + stats, with both splits non-empty."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    demo_h5 = tmp_path / "demo.hdf5"
    with h5py.File(demo_h5, "w") as f:
        for d in range(n_demos):
            g = f.create_group(f"data/demo_{d}")
            vl = h5py.vlen_dtype(np.uint8)
            prim = g.create_dataset("obs/agentview_rgb_jpeg", (T,), dtype=vl)
            wrist = g.create_dataset("obs/eye_in_hand_rgb_jpeg", (T,), dtype=vl)
            for t in range(T):
                prim[t] = _jpeg((t * 8 % 255, 30, d * 40 % 255))
                wrist[t] = _jpeg((0, t * 8 % 255, 90))
    tasks = ["put the bowl on the plate", "open the drawer"]
    demo_recs = []
    while not any(is_val_id(r["id"]) for r in demo_recs) or len(demo_recs) < n_demos:
        i = len(demo_recs)
        acts = np.linspace(-1, 1, T)[:, None] * np.ones((T, A_DIM)) * 0.9
        acts[:, -1] = np.where(np.arange(T) % 2 == 0, -1.0, 1.0)
        demo_recs.append(
            {
                "dataset": "syn_libero",
                "source": "demo",
                "id": f"syn/task{i % 2}/demo_{i % n_demos}_{i}",
                "task": tasks[i % 2],
                "h5": str(demo_h5),
                "group": f"data/demo_{i % n_demos}",
                "keys": {"primary": "obs/agentview_rgb_jpeg", "wrist": "obs/eye_in_hand_rgb_jpeg"},
                "num_steps": T,
                "success": True,
                "actions": acts.tolist(),
                "proprio": (np.arange(T)[:, None] * 0.01 + np.arange(P_DIM)[None] * 0.1).tolist(),
            }
        )
        assert len(demo_recs) < 400
    roll_recs, i = [], 0
    while len(roll_recs) < 4 or not any(is_val_id(r["id"]) for r in roll_recs):
        p, Ti, succ = tmp_path / f"roll_{i}.hdf5", T + i % 3, i % 2 == 0
        with h5py.File(p, "w") as f:
            vl = h5py.vlen_dtype(np.uint8)
            prim = f.create_dataset("primary_images_jpeg", (Ti,), dtype=vl)
            wrist = f.create_dataset("wrist_images_jpeg", (Ti,), dtype=vl)
            for t in range(Ti):
                prim[t] = _jpeg((200, t * 5 % 255, 10))
                wrist[t] = _jpeg((10, 200, t * 5 % 255))
        acts = np.random.RandomState(Ti).uniform(-1, 1, (Ti, A_DIM))
        prop = np.random.RandomState(Ti + 1).uniform(0, 1, (Ti, P_DIM))
        roll_recs.append(
            {
                "dataset": "syn_libero",
                "source": "rollout",
                "id": f"syn/rollout/ep{i}",
                "task": tasks[i % 2],
                "h5": str(p),
                "group": "",
                "keys": {"primary": "primary_images_jpeg", "wrist": "wrist_images_jpeg"},
                "num_steps": Ti,
                "success": succ,
                "actions": acts.tolist(),
                "proprio": prop.tolist(),
            }
        )
        i += 1
        assert i < 400
    demo_jsonl, roll_jsonl = tmp_path / "demo.jsonl", tmp_path / "roll.jsonl"
    for path, recs in ((demo_jsonl, demo_recs), (roll_jsonl, roll_recs)):
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    stats_path = tmp_path / "stats.json"
    json.dump(
        {
            "actions_min": [-1.0] * A_DIM,
            "actions_max": [1.0] * A_DIM,
            "proprio_min": [0.0] * P_DIM,
            "proprio_max": [1.0] * P_DIM,
        },
        open(stats_path, "w"),
    )
    return str(demo_jsonl), str(roll_jsonl), str(stats_path), T


def test_policy_dataset_items_modes_and_balance(tmp_path):
    demo_jsonl, roll_jsonl, stats_path, T = make_fixture(tmp_path)
    ds = PolicyJsonlDataset(
        demo_jsonl, split="all", image_size=32, stats_path=stats_path, chunk_size=5, views=("primary", "wrist")
    )
    assert len(ds) == len(ds.records) * T
    item, last = ds[0], ds[T - 1]
    assert len(item["frames"]) == 2 and item["frames"][0].shape == (3, 32, 32) and item["actions"].shape == (5, A_DIM)
    assert item["mode"] == MODE_POLICY and item["actions"].abs().max() <= 1.0 + 1e-6
    assert torch.allclose(last["actions"][0], last["actions"][-1]) and torch.allclose(
        last["future_proprio"], last["proprio"]
    )
    assert abs(item["value"] - monte_carlo_value(T, 0, True, 0.99)) < 1e-6 and abs(last["value"] - 1.0) < 1e-6
    roll = PolicyJsonlDataset(
        roll_jsonl, split="all", image_size=32, stats_path=stats_path, chunk_size=4, sample_mode="rollout"
    )
    assert {roll[i]["mode"] for i in range(0, len(roll), 3)} == {MODE_WM, MODE_VALUE}
    nopv = PolicyJsonlDataset(
        roll_jsonl,
        split="all",
        image_size=32,
        stats_path=stats_path,
        chunk_size=4,
        sample_mode="rollout",
        pv_pred=False,
    )
    assert {nopv[i]["mode"] for i in range(0, len(nopv), 3)} == {MODE_WM}
    bal = PolicyJsonlDataset(
        roll_jsonl,
        split="all",
        image_size=32,
        stats_path=stats_path,
        chunk_size=4,
        sample_mode="rollout",
        balance_success=True,
    )
    assert len(bal) >= len(roll)
    tr = PolicyJsonlDataset(
        demo_jsonl, split="train", image_size=32, stats_path=stats_path, chunk_size=4, max_episodes_per_task=1
    )
    per_task = {}
    for r in tr.records:
        per_task[r["task"]] = per_task.get(r["task"], 0) + 1
    assert max(per_task.values()) == 1


@needs_qwen
def test_policy_collator(tmp_path):
    from transformers import AutoTokenizer

    from app.lala_world.models.hub import resolve_model_id

    demo_jsonl, _, stats_path, T = make_fixture(tmp_path)
    ds = PolicyJsonlDataset(demo_jsonl, split="train", image_size=32, stats_path=stats_path, chunk_size=4, aug=True)
    a, b = ds[0], ds[0]
    assert not torch.allclose(a["frames"][0], b["frames"][0])  # random augmentation
    batch = PolicyCollator(AutoTokenizer.from_pretrained(resolve_model_id(QWEN35)), max_text_tokens=16)(
        [ds[i] for i in range(3)]
    )
    assert (
        batch["frames"].shape == (3, 1, 3, 32, 32)
        and batch["actions"].shape == (3, 4, A_DIM)
        and batch["mode"].tolist() == [0, 0, 0]
    )


@needs_qwen
class TestPolicyModel:
    @pytest.fixture(scope="class")
    def policy(self):
        from app.lala_policy.model import LaLaPolicy
        from app.lala_world.models import LaLaWorld, build_llm_backbone

        torch.manual_seed(0)
        backbone = build_llm_backbone({"kind": "qwen3_5", "model_id": QWEN35, "num_layers": 2})
        wm = LaLaWorld(backbone, vis_dim=VIS_DIM, target_dim=VIS_DIM, grid_hw=GRID)
        return LaLaPolicy(wm, n_views=V, proprio_dim=P_DIM, action_dim=A_DIM, chunk_size=K).cuda()

    @staticmethod
    def _tok(policy, texts):
        t = policy.wm.backbone.tokenizer(texts, padding="longest", truncation=True, max_length=32, return_tensors="pt")
        return t["input_ids"].cuda(), t["attention_mask"].cuda()

    @staticmethod
    def _inputs(B, seed=0):
        g = torch.Generator().manual_seed(seed)
        x = {
            "vis": torch.randn(B, V, N_VIS, VIS_DIM, generator=g),
            "future_vis": torch.randn(B, V, N_VIS, VIS_DIM, generator=g),
            "proprio": torch.rand(B, P_DIM, generator=g) * 2 - 1,
            "future_proprio": torch.rand(B, P_DIM, generator=g) * 2 - 1,
            "actions": torch.rand(B, K, A_DIM, generator=g) * 2 - 1,
            "value": torch.rand(B, generator=g) * 2 - 1,
        }
        return {k: v.cuda() for k, v in x.items()}

    def test_position_layout(self):
        from app.lala_policy.model import policy_position_ids

        pos = policy_position_ids(GRID, V, [3, 1], 3, K)
        S = 3 + (V * N_VIS + 1) + (K + V * N_VIS + 2)
        assert pos.shape == (3, 2, S)
        p = pos[:, 1]
        assert p[:, :2].sum() == 0 and p[:, 2 + V * N_VIS].tolist() == [0, V * GRID[0], 0]
        a0 = 2 + V * N_VIS + 2
        assert p[0, a0 : a0 + K].tolist() == [2] * K and p[2, a0 : a0 + K].tolist() == list(range(K))
        assert p[:, -1].tolist() == [3, V * GRID[0] + 2, 0]
        q = policy_position_ids(GRID, V, [3, 1], 3, K, pv_pred=False)
        assert torch.equal(q, pos[:, :, : S - 2])

    def test_modes_shapes_and_grad_set(self, policy):
        from app.lala_policy.losses import PolicyLoss

        x = self._inputs(3)
        ids, mask = self._tok(policy, ["put the bowl on the plate", "open the drawer", "close it"])
        out = policy(
            x["vis"],
            ids,
            mask,
            x["proprio"],
            torch.tensor([0, 1, 2], device="cuda"),
            x["actions"],
            x["future_proprio"],
            x["future_vis"],
        )
        assert out["actions"].shape == (3, K, A_DIM) and out["future_latent"].shape == (3, V, N_VIS, VIS_DIM)
        assert out["future_proprio"].shape == (3, P_DIM) and out["value"].shape == (3,)
        with pytest.raises(ValueError, match="actions"):
            policy(x["vis"], ids, mask, x["proprio"], torch.tensor([1, 0, 0], device="cuda"))
        grads = {}
        for name, mode in (
            ("mixed", torch.tensor([0, 1, 2], device="cuda")),
            ("policy_only", torch.zeros(3, dtype=torch.long, device="cuda")),
        ):
            policy.zero_grad(set_to_none=True)
            out = policy(x["vis"], ids, mask, x["proprio"], mode, x["actions"], x["future_proprio"], x["future_vis"])
            loss, _ = PolicyLoss()(
                out,
                dict(actions=x["actions"], future_proprio=x["future_proprio"], value=x["value"], mode=mode),
                x["future_vis"],
            )
            loss.backward()
            grads[name] = {n for n, prm in policy.named_parameters() if prm.grad is not None}
        policy.zero_grad(set_to_none=True)
        assert grads["mixed"] == grads["policy_only"]  # DDP static_graph needs a mode-independent grad set

    def test_padding_invariance(self, policy):
        x = self._inputs(2, seed=1)
        ids, mask = self._tok(policy, ["open the drawer", "pick up the black bowl between the plate and the ramekin"])
        mode = torch.zeros(2, dtype=torch.long, device="cuda")
        with torch.no_grad():
            both = policy(x["vis"], ids, mask, x["proprio"], mode)
            ids1, mask1 = self._tok(policy, ["open the drawer"])
            alone = policy(x["vis"][:1], ids1, mask1, x["proprio"][:1], mode[:1])
        for k in ("actions", "future_latent", "future_proprio", "value"):
            rel = (both[k][:1] - alone[k]).abs().max() / (alone[k].abs().max() + 1e-6)
            assert rel.item() < 5e-2, (k, rel)

    def test_load_world_model(self, policy):
        from app.lala_world.models import LaLaWorld

        src = LaLaWorld(policy.wm.backbone, vis_dim=VIS_DIM, target_dim=VIS_DIM, grid_hw=GRID).cuda()
        with torch.no_grad():
            src.query_token.add_(1.0)
        policy.load_world_model(src.state_dict())
        assert torch.equal(policy.wm.query_token, src.query_token)
        bad = dict(src.state_dict())
        bad.pop("query_token")
        with pytest.raises(RuntimeError, match="missing"):
            policy.load_world_model(bad)

    def test_flow_head_nopv(self, policy):
        from app.lala_policy.losses import PolicyLoss
        from app.lala_policy.model import LaLaPolicy

        head = {"kind": "flow", "layers": 2, "heads": 4, "head_dim": 8, "out_dim": 16, "steps": 3, "dropout": 0.0}
        torch.manual_seed(0)
        pol = LaLaPolicy(
            policy.wm, n_views=V, proprio_dim=P_DIM, action_dim=A_DIM, chunk_size=K, action_head=head, pv_pred=False
        ).cuda()
        fh = pol.flow_head
        assert (
            set(fh.targets) == {"actions"} and fh.n_tokens == K and pol.action_head is None and pol.head_norm is None
        )
        x = self._inputs(3, seed=5)
        ids, mask = self._tok(pol, ["put the bowl on the plate", "open the drawer", "close it"])
        pol.train()
        out = pol(x["vis"], ids, mask, x["proprio"], torch.tensor([0, 1, 0], device="cuda"), x["actions"])
        assert set(out) == {"future_latent", "velocity", "flow_target", "flow_t", "actions"}
        loss, logs = PolicyLoss()(
            out, dict(actions=x["actions"], mode=torch.tensor([0, 1, 0], device="cuda")), x["future_vis"]
        )
        assert set(logs) == {"loss_action", "loss_latent", "n_policy", "n_wm", "n_value"} and torch.isfinite(loss)
        loss.backward()
        assert {n for n, p in pol.named_parameters() if p.grad is not None} == {
            n for n, p in pol.named_parameters() if p.requires_grad
        }
        pol.zero_grad(set_to_none=True)
        pol.eval()
        g = torch.Generator(device="cuda")
        with torch.no_grad():
            s1 = pol(
                x["vis"],
                ids,
                mask,
                x["proprio"],
                torch.zeros(3, dtype=torch.long, device="cuda"),
                flow_generator=g.manual_seed(7),
            )
            s2 = pol(
                x["vis"],
                ids,
                mask,
                x["proprio"],
                torch.zeros(3, dtype=torch.long, device="cuda"),
                flow_generator=g.manual_seed(7),
            )
            s3 = pol(
                x["vis"],
                ids,
                mask,
                x["proprio"],
                torch.zeros(3, dtype=torch.long, device="cuda"),
                flow_generator=g.manual_seed(8),
            )
        assert (
            s1["actions"].shape == (3, K, A_DIM)
            and torch.equal(s1["actions"], s2["actions"])
            and not torch.allclose(s1["actions"], s3["actions"])
        )
        with pytest.raises(ValueError, match="pv_pred"):
            pol(
                x["vis"],
                ids,
                mask,
                x["proprio"],
                torch.tensor([2, 0, 0], device="cuda"),
                x["actions"],
                x["future_proprio"],
                x["future_vis"],
            )


@needs_qwen
@pytest.mark.parametrize("variant", ["l1_pv", "flow_nopv"])
def test_policy_train_smoke_and_act(tmp_path, variant):
    from app.lala_policy.train import main
    from evals.lala_policy.policy import PolicyAgent

    demo_jsonl, roll_jsonl, stats_path, T = make_fixture(tmp_path / "data")
    args = {
        "app": "lala_policy",
        "folder": str(tmp_path / "run"),
        "meta": {"seed": 0, "dtype": "bfloat16", "quick_debug": True, "log_freq": 1, "init_from": None},
        "data": {
            "sources": [
                {"kind": "policy_jsonl", "path": demo_jsonl, "sample_mode": "policy"},
                {"kind": "policy_jsonl", "path": roll_jsonl, "sample_mode": "rollout", "balance_success": True},
            ],
            "stats_path": stats_path,
            "max_text_tokens": 32,
            "batch_size": 4,
            "num_workers": 0,
            "aug": True,
        },
        "model": {
            "vision": {"kind": "siglip2", "model_id": SIGLIP2},
            "llm": {"kind": "qwen3_5", "model_id": QWEN35, "num_layers": 2},
            "policy": {"views": ["primary", "wrist"], "chunk_size": 4, "proprio_dim": P_DIM, "action_dim": A_DIM},
        },
        "loss": {"action_w": 1.0, "latent_w": 1.0, "proprio_w": 1.0, "value_w": 1.0},
        "optimization": {"epochs": 1, "lr": 1e-4, "start_lr": 1e-5, "warmup_steps": 2, "clip_grad": 1.0},
        "logging": {"use_wandb": False},
    }
    if variant == "flow_nopv":
        args["model"]["policy"].update(
            pv_pred=False,
            action_head={"kind": "flow", "layers": 2, "heads": 4, "head_dim": 8, "out_dim": 16, "steps": 2},
        )
    pv_pred = args["model"]["policy"].get("pv_pred", True)
    result = main(args)
    assert torch.isfinite(torch.tensor(result["final_train_loss"]))
    ckpt_path = tmp_path / "run" / "lala-latest.pth.tar"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 1 and "stats" in ckpt and ckpt["config"]["model"]["policy"]["pv_pred"] is pv_pred
    val = json.loads(open(tmp_path / "run" / "val.jsonl").readline())
    assert ("value_l1" in val) is pv_pred and "action_l1" in val
    agent = PolicyAgent(str(ckpt_path), device="cuda:0")
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    actions, extras = agent.act(
        {"primary": img, "wrist": img}, np.zeros(P_DIM), "open the drawer", return_extras=True, imagine_wm=True
    )
    assert actions.shape == (4, A_DIM) and np.isfinite(actions).all() and extras["future_latent"].shape[0] == 2
    assert torch.isfinite(extras["future_latent_wm"]).all() and ("value" in extras) is pv_pred


@needs_qwen
def test_init_from_world_model_checkpoint(tmp_path):
    from app.lala_policy.train import main
    from app.lala_world.models import LaLaWorld, build_llm_backbone

    demo_jsonl, roll_jsonl, stats_path, T = make_fixture(tmp_path / "data")
    model_cfg = {
        "vision": {"kind": "siglip2", "model_id": SIGLIP2},
        "llm": {"kind": "qwen3_5", "model_id": QWEN35, "num_layers": 2},
        "policy": {
            "views": ["primary", "wrist"],
            "chunk_size": 4,
            "proprio_dim": P_DIM,
            "action_dim": A_DIM,
            "pv_pred": False,
        },
    }
    backbone = build_llm_backbone(model_cfg["llm"])
    wm = LaLaWorld(backbone, vis_dim=768, target_dim=768, grid_hw=(14, 14))
    with torch.no_grad():
        wm.query_token.fill_(0.123)
    wm_ckpt = tmp_path / "wm.pth.tar"
    torch.save(
        {
            "epoch": 7,
            "model": {f"predictor.{k}": v for k, v in wm.state_dict().items()},
            "config": {"app": "lala_world", "model": model_cfg},
        },
        wm_ckpt,
    )
    args = {
        "app": "lala_policy",
        "folder": str(tmp_path / "run"),
        "meta": {"seed": 0, "dtype": "bfloat16", "quick_debug": True, "log_freq": 1, "init_from": str(wm_ckpt)},
        "data": {
            "sources": [
                {"kind": "policy_jsonl", "path": demo_jsonl, "sample_mode": "policy"},
                {"kind": "policy_jsonl", "path": roll_jsonl, "sample_mode": "rollout"},
            ],
            "stats_path": stats_path,
            "max_text_tokens": 32,
            "batch_size": 4,
            "num_workers": 0,
        },
        "model": model_cfg,
        "loss": {},
        "optimization": {"epochs": 1, "lr": 1e-4, "start_lr": 1e-5, "warmup_steps": 2},
        "logging": {"use_wandb": False},
    }
    main(args)
    ckpt = torch.load(tmp_path / "run" / "lala-latest.pth.tar", map_location="cpu", weights_only=False)
    assert abs(float(ckpt["model"]["predictor.wm.query_token"].mean()) - 0.123) < 0.01
    bad = json.loads(json.dumps(args))
    bad["model"]["llm"]["num_layers"] = 3
    bad["folder"] = str(tmp_path / "run_bad")
    with pytest.raises(ValueError, match="num_layers"):
        main(bad)


def test_eval_scripts_one_driver_line_per_task():
    import subprocess

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for script, n_tasks in (("run_robocasa_all.sh", 24), ("run_libero_all.sh", 40)):
        env = dict(os.environ, LALA_POLICY_EVAL_DRYRUN="1", LALA_POLICY_EVAL_EXTRA="--flow-steps 1 ", PYTHONPATH=repo)
        run = subprocess.run(
            ["bash", f"evals/lala_policy/{script}", "/x/lala-e12.pth.tar", "tag", "50", "4", "0 1"],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert run.returncode == 0, run.stderr[-800:]
        lines = run.stdout.strip("\n").split("\n")
        assert len(lines) == n_tasks and all(line.endswith("--flow-steps 1") for line in lines)
