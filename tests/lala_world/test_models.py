import pytest
import torch

from app.lala_world.models import LaLaWorld, build_llm_backbone, build_vision_encoder
from app.lala_world.models.hub import model_name, resolve_model_id
from app.lala_world.models.pos_embed import Additive3DSinCos, position_ids_3d_queries, position_ids_3d_seq
from app.lala_world.models.predictor import pack_left, pack_left_seq
from tests.conftest import QWEN35, SIGLIP2, needs_qwen, needs_siglip2

GRID, N_VIS, VIS_DIM = (2, 2), 4, 16


def test_resolve_model_id(monkeypatch, tmp_path):
    monkeypatch.setenv("LALA_MODELS", str(tmp_path / "none"))  # no local copies: fall back to the hub ids
    assert resolve_model_id("/nonexistent/path/Qwen3.5-0.8B") == "Qwen/Qwen3.5-0.8B"
    assert resolve_model_id("/nonexistent/facebookresearch_dinov2_main") == "facebookresearch/dinov2"
    assert resolve_model_id("some/unknown-id") == "some/unknown-id"
    (tmp_path / "models" / "Qwen3.5-0.8B").mkdir(parents=True)
    monkeypatch.setenv("LALA_MODELS", str(tmp_path / "models"))  # a local copy wins over the hub id
    assert resolve_model_id("Qwen/Qwen3.5-0.8B") == str(tmp_path / "models" / "Qwen3.5-0.8B")
    assert model_name("/a/b/Qwen3.5-0.8B/") == model_name("Qwen/Qwen3.5-0.8B") == "Qwen3.5-0.8B"


def test_position_ids_queries_layout():
    pos = position_ids_3d_queries(grid_hw=(2, 2), text_lens=[2], max_text_len=2, n_queries=4)
    expected = torch.tensor(
        [[0, 0, 0, 0, 1, 2, 3, 3, 3, 3], [0, 0, 1, 1, 1, 2, 0, 0, 1, 1], [0, 1, 0, 1, 1, 2, 0, 1, 0, 1]]
    )
    assert torch.equal(pos[:, 0], expected)
    a = position_ids_3d_queries((14, 14), [2], max_text_len=4, n_queries=196)
    b = position_ids_3d_queries((14, 14), [2], max_text_len=9, n_queries=196)
    assert torch.equal(a[:, 0, 2:], b[:, 0, 7:])  # real-slot ids do not depend on the pad amount


def test_position_ids_seq_time_axis():
    lens, ctx = torch.tensor([[3], [5]]), torch.ones(2, dtype=torch.long)
    seq = position_ids_3d_seq(GRID, lens, ctx, n_queries=4, total_len=N_VIS + 5 + 4)
    assert torch.equal(seq, position_ids_3d_queries(GRID, [3, 5], max_text_len=5, n_queries=4))
    pos = position_ids_3d_seq(
        GRID, torch.tensor([[2, 3]]), torch.tensor([2]), n_queries=4, total_len=2 * N_VIS + 5 + 4
    )
    assert pos[0, 0].tolist() == [0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 7, 7, 7]


def test_additive_sincos():
    pe = Additive3DSinCos(768, ref_scale=5.0)
    assert pe.dims == [192, 288, 288] and float(pe.gain) == 5.0 and pe.gain.requires_grad
    out = pe(position_ids_3d_queries((2, 2), [2], max_text_len=2, n_queries=4))
    assert out.shape == (1, 10, 768)


def test_pack_left_and_seq_agree():
    torch.manual_seed(0)
    B, N, L, Nq, D = 3, 4, 6, 2, 5
    vis, txt, q = torch.randn(B, N, D), torch.randn(B, L, D), torch.randn(1, Nq, D).expand(B, -1, -1)
    text_lens = torch.tensor([2, 6, 4])
    x_ref, m_ref = pack_left(vis, txt, text_lens, q)
    x_seq, m_seq = pack_left_seq(vis[:, None], txt[:, None], text_lens[:, None], torch.ones(B, dtype=torch.long), q)
    assert torch.equal(x_ref, x_seq) and torch.equal(m_ref, m_seq)
    assert m_ref[0].tolist() == [0] * 4 + [1] * 8  # sample 0: 2 real text tokens -> 4 pad slots


@needs_siglip2
def test_siglip2_encoder():
    enc = build_vision_encoder({"kind": "siglip2", "model_id": SIGLIP2})
    assert (enc.embed_dim, enc.num_patches, enc.grid_size, enc.image_size) == (768, 196, (14, 14), 224)
    assert enc.image_mean == (0.5, 0.5, 0.5) and all(not p.requires_grad for p in enc.parameters())
    enc.train()
    assert not enc.model.training
    with torch.no_grad():
        out = enc(torch.randn(2, 3, 224, 224))
    assert out["patches"].shape == (2, 196, 768)
    with pytest.raises(ValueError, match="frozen"):
        build_vision_encoder({"kind": "siglip2", "model_id": SIGLIP2, "frozen": False})


@needs_qwen
class TestQwenPredictor:
    @pytest.fixture(scope="class")
    def backbone(self):
        return build_llm_backbone({"kind": "qwen3_5", "model_id": QWEN35, "num_layers": 2}).cuda().eval()

    def test_backbone_props(self, backbone):
        assert backbone.hidden_size == 1024 and backbone.embed_scale == 1.0 and len(backbone.model.layers) == 2
        with pytest.raises(ValueError, match="batch size 1"):
            m = torch.ones(1, 6, dtype=torch.long, device="cuda")
            m[0, 0] = 0
            backbone(torch.randn(1, 6, 1024, device="cuda") * 0.05, m)

    def test_num_layers_prefix_matches_full_model(self, backbone):
        full = build_llm_backbone({"kind": "qwen3_5", "model_id": QWEN35}).cuda().eval()
        e = torch.randn(2, 7, 1024, device="cuda") * 0.05
        mask = torch.ones(2, 7, dtype=torch.long, device="cuda")
        with torch.no_grad():
            h = backbone(e, mask)
            ref = full.model(inputs_embeds=e, attention_mask=mask, use_cache=False, output_hidden_states=True)
        assert torch.equal(h, full.model.norm(ref.hidden_states[2]))

    def test_padding_invariance_single_and_multi_frame(self, backbone):
        torch.manual_seed(0)
        model = LaLaWorld(backbone, vis_dim=VIS_DIM, target_dim=24, grid_hw=GRID).cuda().eval()
        tok = backbone.tokenizer
        vis = torch.randn(1, N_VIS, VIS_DIM, device="cuda")
        a = tok(["move the cup left"], padding="longest", return_tensors="pt")
        b = tok(["move the cup left", " ".join(["padding"] * 30)], padding="longest", return_tensors="pt")
        with torch.no_grad():
            alone = model(vis, a["input_ids"].cuda(), a["attention_mask"].cuda())
            batched = model(vis.expand(2, -1, -1).contiguous(), b["input_ids"].cuda(), b["attention_mask"].cuda())
            c1 = model(
                vis[:, None],
                a["input_ids"].cuda()[:, None],
                a["attention_mask"].cuda()[:, None],
                ctx_len=torch.ones(1, dtype=torch.long, device="cuda"),
            )
        assert alone.shape == (1, N_VIS, 24)
        rel = (alone[0] - batched[0]).abs().max() / alone.abs().max()
        assert rel.item() < 5e-2, rel  # fla kernels run reduced precision; a structural break is O(1)
        assert torch.equal(alone, c1)  # a C=1 batch takes the single-frame path
        # multi-frame: a d=2 sample next to a d=3 partner
        texts = ["move the cup left", "now push it right", "then lift it"]
        t = tok(texts, padding="longest", return_tensors="pt")
        L = t["input_ids"].shape[1]
        ids = torch.zeros(2, 3, L, dtype=torch.long)
        mask = torch.zeros(2, 3, L, dtype=torch.long)
        ids[0, :2], mask[0, :2] = t["input_ids"][:2], t["attention_mask"][:2]
        ids[1], mask[1] = t["input_ids"], t["attention_mask"]
        v = torch.randn(2, 3, N_VIS, VIS_DIM, device="cuda")
        with torch.no_grad():
            out = model(v, ids.cuda(), mask.cuda(), ctx_len=torch.tensor([2, 3], device="cuda"))
            solo = model(v[:1, :2], ids[:1, :2].cuda(), mask[:1, :2].cuda(), ctx_len=torch.tensor([2], device="cuda"))
        assert out.shape == (2, N_VIS, 24) and torch.isfinite(out).all()
        rel = (out[0] - solo[0]).abs().max() / solo.abs().max()
        assert rel.item() < 5e-2, rel

    def test_grads_reach_every_trainable_part(self, backbone):
        model = LaLaWorld(backbone, vis_dim=VIS_DIM, target_dim=24, grid_hw=GRID).cuda().train()
        t = backbone.tokenizer(["move left", "go right"], padding="longest", return_tensors="pt")
        out = model(torch.randn(2, N_VIS, VIS_DIM, device="cuda"), t["input_ids"].cuda(), t["attention_mask"].cuda())
        out.float().sum().backward()
        assert model.query_token.grad is not None and model.additive_pe.gain.grad is not None
        assert all(p.grad is not None for p in model.connector.parameters())
        assert all(p.grad is not None for p in model.out_head.parameters())
        assert any(p.grad is not None for p in model.backbone.model.layers.parameters())
