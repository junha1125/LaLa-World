"""Planning adapter mechanics (context window) and the planner / spec plumbing, without a checkpoint."""

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from evals.lala_world.sim_spec import DEFORM_R, DeformSpec, SimSpec, load_spec
from evals.lala_world.wm import PlanningWM, trained_ctxt_window
from evals.simu_env_planning.planning.objectives import ReprTargetDistMPCObjective
from evals.simu_env_planning.planning.planner import CEMPlanner
from tests.conftest import QWEN35, needs_cuda, needs_qwen

GRID, N_VIS, DIM = (2, 2), 4, 16


class _NoDyn:
    mode = "none"


class _TextSpec:
    plan_space = "text"

    @staticmethod
    def plan_to_z(u):
        return u

    @staticmethod
    def denorm_action(z):
        return z


class _Recorder(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = []

    def forward(self, vis, input_ids, text_mask, ctx_len=None):
        out = self.inner(vis, input_ids, text_mask, ctx_len=ctx_len)
        self.calls.append(
            {"vis": vis.detach().clone(), "ids": input_ids.detach().clone(), "out": out.detach().clone()}
        )
        return out


@pytest.fixture(scope="module")
def predictor():
    from app.lala_world.models import LaLaWorld, build_llm_backbone

    backbone = build_llm_backbone({"kind": "qwen3_5", "model_id": QWEN35, "num_layers": 2})
    torch.manual_seed(0)
    return LaLaWorld(backbone, vis_dim=DIM, target_dim=DIM, grid_hw=GRID).cuda().eval()


def _make_wm(predictor, w):
    wm = PlanningWM.__new__(PlanningWM)
    wm.spec, wm.dyn, wm.device = _TextSpec(), _NoDyn(), torch.device("cuda")
    wm.domain_prefix, wm.batch_chunk, wm.ctxt_window = False, 512, w
    wm.render = lambda prop, a, prefix=False: f"go {float(a[0]):+.2f}"
    wm.predictor = predictor
    wm.tokenizer = predictor.backbone.tokenizer
    if wm.tokenizer.pad_token is None:
        wm.tokenizer.pad_token = wm.tokenizer.eos_token
    wm.max_text_tokens, wm.use_amp = 32, False
    return wm


@needs_qwen
def test_single_frame_unroll_feeds_predictions_back(predictor):
    wm = _make_wm(predictor, w=1)
    z0 = torch.randn(1, N_VIS, DIM, device="cuda")
    acts = torch.randn(3, 5, 1, device="cuda")
    preds = wm.unroll((z0, None), acts)
    state, ref = z0.expand(5, -1, -1).contiguous(), []
    for h in range(3):
        out = wm._predict(state, [wm.render(None, acts[h, b].cpu().numpy()) for b in range(5)])
        ref.append(out)
        state = out
    assert torch.equal(preds, torch.stack(ref))


@needs_qwen
def test_sliding_window_contents_and_history(predictor):
    wm = _make_wm(predictor, w=2)
    rec = _Recorder(wm.predictor)
    wm.predictor = rec
    z0, B = torch.randn(1, N_VIS, DIM, device="cuda"), 3
    acts = torch.randn(3, B, 1, device="cuda")
    preds = wm.unroll((z0, None), acts)
    assert preds.shape == (3, B, N_VIS, DIM) and len(rec.calls) == 3
    c0, c1, c2 = rec.calls
    assert c0["vis"].ndim == 3 and torch.equal(c0["vis"], z0.expand(B, -1, -1))
    assert c1["vis"].shape[1] == 2 and torch.equal(c1["vis"][:, 1], c0["out"])
    assert torch.equal(c2["vis"][:, 0], c0["out"]) and torch.equal(c2["vis"][:, 1], c1["out"])
    assert wm.tokenizer.decode(c2["ids"][0, 0], skip_special_tokens=True).strip() == f"go {float(acts[1, 0, 0]):+.2f}"
    rec.calls.clear()
    fa, fb, fc = (torch.randn(1, N_VIS, DIM, device="cuda") for _ in range(3))
    wm.unroll((([fa, fb, fc], ["past a", "past b"]), None), torch.randn(1, 2, 1, device="cuda"))
    call = rec.calls[0]
    assert call["vis"].shape[1] == 2 and torch.equal(call["vis"][:, 0], fb.expand(2, -1, -1))
    assert wm.tokenizer.decode(call["ids"][0, 0], skip_special_tokens=True).strip() == "past b"


@needs_cuda
def test_cem_planner_converges_on_a_linear_toy_model():
    """unroll = cumulative sum of the actions; CEM must find the actions that reach the target latent."""
    target = torch.tensor([[1.0, -2.0]], device="cuda")

    def unroll(z_init, act_suffix):
        return z_init[None] + act_suffix.cumsum(0)  # (H, B, A)

    planner = CEMPlanner(
        unroll=unroll,
        iterations=20,
        num_samples=200,
        horizon=2,
        action_dim=2,
        num_elites=20,
        max_norms=[3.0],
        max_norm_dims=[[0, 1]],
        local_generator=torch.Generator(device="cuda").manual_seed(0),
        num_act_stepped=2,
    )
    planner.set_objective(ReprTargetDistMPCObjective(None, target_enc=target, sum_all_diffs=False))
    res = planner.plan(torch.zeros(1, 2, device="cuda"))
    assert res.actions.shape == (2, 2) and torch.allclose(res.actions.sum(0), target[0], atol=0.05)
    assert res.losses[-1] < res.losses[0]


def test_trained_ctxt_window_follows_clip_len():
    sources = [{"kind": "traj_jsonl", "clip_len": 4}, {"kind": "traj_jsonl", "clip_len": 2}, {"kind": "imagenet_aug"}]
    assert trained_ctxt_window({"data": {"sources": sources}}) == 3
    assert trained_ctxt_window({"data": {"sources": sources[1:]}}) == 1
    assert trained_ctxt_window({"data": {}}) == 1
    assert trained_ctxt_window({}) == 1


def test_specs_ship_with_the_repo():
    spec = load_spec("wall")
    assert type(spec) is SimSpec and spec.plan_space == "text" and spec.plan_dim == 2
    u = np.random.default_rng(1).standard_normal((3, 2))
    assert np.array_equal(spec.plan_to_z(u), u)
    assert spec.render_semantic(None, np.array([0.0, 0.0]), prefix=False).startswith("move the agent")
    assert spec.render_lang6(None, np.array([0.0, 0.0]), prefix=False).startswith("move the agent")
    droid = load_spec("droid")
    assert droid.render_semantic(None, np.zeros(7), prefix=False).startswith("move gripper")
    for env in ("rope", "granular"):
        d = load_spec(env)
        assert isinstance(d, DeformSpec) and d.plan_dim == 4
        rng = np.random.default_rng(0)
        u = rng.standard_normal((500, 4))
        norms, dims = d.plan_max_norms()
        for n, dd in zip(norms, dims):
            u[:, dd] = np.clip(u[:, dd], -n, n)
        net = d.denorm_action(d.plan_to_z(u))
        end_env = (net[:, :2] + net[:, 2:]) @ DEFORM_R
        assert np.all(np.abs(end_env) <= 1.5 + 1e-6)  # the CEM end point stays inside the data box
    assert os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "..", "evals", "lala_world", "proprio_dyn.json")
    )
