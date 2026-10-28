"""CPU-only checks of the RAEv2 world-model package:  PYTHONPATH=src pytest RAEv2/tests"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from stage2.transport import Transport  # noqa: E402
from wm.cond import I2ITextCondEmbedder, apply_text_dropout, i2i_rope, make_context  # noqa: E402
from wm.data import resolve_sources  # noqa: E402
from wm.merge_planning import wilson  # noqa: E402
from wm.model import _remap_truncated, build_dit, count_params  # noqa: E402


def test_tiny_dit_forward_and_truncation():
    cfg = {"input_size": 4, "in_channels": 8, "hidden_size": [32, 48], "depth": [3, 1], "num_heads": [2, 2],
           "base_model_depth": 2, "num_t_tokens": 2}
    model = build_dit(cfg, text_dim=16, n_text_tokens=5).eval()
    assert isinstance(model.ctx_embedder, I2ITextCondEmbedder)
    x = torch.randn(2, 8, 4, 4)
    ctx = make_context(torch.randn(2, 8, 4, 4), torch.randn(2, 5, 16))
    mask = torch.ones(2, 5)
    with torch.no_grad():
        full, base = model(x, torch.rand(2), context=ctx, attn_mask=mask)
    assert full.shape == x.shape and base.shape == x.shape
    sd = model.state_dict()
    sd2 = _remap_truncated(sd, n_enc_ckpt=3, n_enc=2, n_dec=1)
    small = build_dit({**cfg, "depth": [2, 1], "base_model_depth": 2}, 16, 5)
    res = small.load_state_dict({k: v for k, v in sd2.items() if not k.startswith(("ctx_embedder.", "enc_rope."))}, strict=False)
    assert not res.unexpected_keys and count_params(small) < count_params(model)
    assert torch.equal(sd2["blocks.2.adaln_modulation.1.weight"], sd["blocks.3.adaln_modulation.1.weight"])


def test_rope_shares_grid_angles_with_context():
    rope = i2i_rope(head_dim=8, num_patches=16, num_t_tokens=2, n_text_tokens=3)
    assert rope.freqs_cos.shape == (16 + 2 + 16 + 3, 8)
    assert torch.equal(rope.freqs_cos[:16], rope.freqs_cos[18:34])
    assert torch.all(rope.freqs_cos[34:] == 1)


def test_text_dropout_replaces_whole_rows():
    torch.manual_seed(0)
    ctx = make_context(torch.zeros(6, 2, 2, 2), torch.ones(6, 4, 3))
    mask = torch.ones(6, 4)
    out, m = apply_text_dropout(ctx, mask, torch.zeros(1, 4, 3), torch.zeros(1, 4), p=0.5)
    dropped = (out["text"].abs().sum(dim=(1, 2)) == 0)
    assert torch.equal(dropped, m.sum(1) == 0) and 0 < int(dropped.sum()) < 6


def test_transport_x_prediction_loss_is_zero_for_the_truth():
    tr = Transport(prediction="x", time_dist_shift=2.0, t_eps=0.05)
    t, x0, x1 = tr.sample(torch.randn(3, 2, 2, 2))
    assert t.shape == (3,) and torch.all((t > 0) & (t < 1))
    te = t.view(-1, 1, 1, 1)
    xt = (1 - te) * x1 + te * x0
    vt = (xt - x1) / te.clamp_min(tr.t_eps)
    assert tr.compute_loss(x1, vt, xt, t).abs().max() < 1e-5


def test_wilson_and_sources():
    lo, hi = wilson(18, 24)
    assert 0.54 < lo < 0.56 and 0.87 < hi < 0.89
    os.environ.pop("LALA_DATA_ROOT", None)
    os.environ.pop("LALA_DATA", None)
    out = resolve_sources([{"kind": "traj_jsonl", "path": "jsonl/traj/wall.jsonl"}, {"kind": "jsonl", "path": "/abs/x.jsonl"}], "/root/data")
    assert out[0]["path"] == "/root/data/jsonl/traj/wall.jsonl" and out[0]["clip_len"] == 2 and out[1]["path"] == "/abs/x.jsonl"
