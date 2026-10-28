"""RAEv2 world-model backend for the LaLa-World CEM planning stack (`--wm rae`).

State = normalised RAE latent of the frame (1024 x 16 x 16, handed around as (B, 256, 1024)). Each predicted frame is a
flow-matching sample from pure noise with `nfe` Euler steps (nfe = 1 is the one-shot x-prediction), fed back as the
next context. CEM cost: `latent` = L2 in the RAE latent space, `pixel` = L2 of the decoded last frame to the goal image.
"""

import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from stage1 import RAE  # noqa: E402
from stage2.transport import Transport  # noqa: E402
from wm import abs_path  # noqa: E402
from wm import cond as cond_mod  # noqa: E402
from wm.model import build_dit, time_shift  # noqa: E402
from wm.text_encoder import FrozenTextEncoder  # noqa: E402


def _model_path(path):
    local = abs_path(path)
    return local if os.path.isdir(local) else path


class RaeWMSimWM:
    numeric_hist = False  # executed chunks are handed back as text
    TEXT_CACHE_MAX = 4096

    def __init__(self, ckpt_path, spec, dyn, device, domain_prefix=True, batch_chunk=300, text_source="semantic",
                 ctxt_window=1, nfe=1, cost="latent", weights="ema", seed=0, cfg_scale=2.0, ig_scale=1.0):
        if cost not in ("latent", "pixel"):
            raise ValueError(f"cost must be latent|pixel, got {cost!r}")
        self.spec = spec
        self.dyn = dyn
        self.device = device
        self.domain_prefix = domain_prefix
        self.batch_chunk = int(batch_chunk)
        self.render = spec.renderer(text_source)
        self.nfe = int(nfe)
        self.cost = cost
        self.cfg_scale = float(cfg_scale)  # classifier-free guidance on the text (x_hat_null + w (x_hat_text - x_hat_null))
        self.ig_scale = float(ig_scale)  # RAEv2 internal guidance (base head), off at 1.0
        self.ig_interval = (0.1, 1.0)
        if ctxt_window is not None and int(ctxt_window) != 1:
            print(f"[rae] ignoring --ctxt-window {ctxt_window}: the RAE world model is single-frame")
        self.ctxt_window = 1
        torch.backends.cuda.matmul.allow_tf32 = True

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = cfg = ck["config"]
        self.step = int(ck["step"])
        self.train_seconds = float(ck.get("train_seconds", 0.0))
        self.epoch = f"s{self.step}"
        if cfg.get("dinov3_repo_dir"):
            os.environ.setdefault("DINOV3_REPO_DIR", abs_path(cfg["dinov3_repo_dir"]))
        s1 = dict(cfg["stage_1"])
        for k in ("decoder_config_path", "pretrained_decoder_path", "normalization_stat_path"):
            s1[k] = abs_path(s1[k])
        self.rae = RAE(**s1).to(device).eval()
        self.rae.requires_grad_(False)
        self.text_encoder = FrozenTextEncoder(_model_path(cfg["text_encoder"]["model_path"]), int(cfg["text_encoder"]["max_length"])).to(device)
        self.model = build_dit(cfg["model"], self.text_encoder.feature_dim, self.text_encoder.max_length).to(device)
        self.model.load_state_dict({k: v.float() for k, v in ck[weights].items()}, strict=True)
        self.model.eval().requires_grad_(False)
        self.latent_size = tuple(cfg["misc"]["latent_size"])
        self.transport = Transport(
            prediction=cfg["transport"].get("prediction", "x"),
            time_dist_type=cfg["transport"].get("time_dist_type", "logit-normal_0_1"),
            time_dist_shift=time_shift(cfg["misc"], self.latent_size),
            t_eps=float(cfg["transport"].get("t_eps", 0.05)),
        )
        t_grid = torch.linspace(1.0, 0.0, self.nfe + 1)
        shift = self.transport.time_dist_shift
        self.t_grid = (shift * t_grid / (1 + (shift - 1) * t_grid)).tolist()
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.transform = transforms.Compose([transforms.Resize((self.rae.resolution, self.rae.resolution)), transforms.ToTensor()])
        self.text_cache = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            null = self.text_encoder([""])
        self.null_tok, self.null_mask = null["tokens"].float(), null["attention_mask"]
        if self.cfg_scale > 1.0 and float(cfg.get("conditioning", {}).get("text_dropout_p", 0.0)) <= 0:
            print("[rae] WARNING: cfg_scale > 1 but the snapshot was trained without text dropout")
        self.timing = {"unroll_calls": 0, "unroll_seconds": 0.0, "dit_forwards": 0, "dit_samples": 0}
        n_par = sum(p.numel() for p in self.model.parameters())
        print(f"[rae] {os.path.basename(ckpt_path)} step {self.step} ({self.train_seconds/3600:.2f} h trained), "
              f"{weights} weights, DiT {n_par/1e6:.0f}M (enc depth {self.model.num_enc_blocks}), nfe {self.nfe}, "
              f"cfg {self.cfg_scale}, ig {self.ig_scale}, cost {self.cost}")

    @torch.no_grad()
    def _pixels(self, img_hwc_uint8):
        return self.transform(Image.fromarray(np.ascontiguousarray(img_hwc_uint8))).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def encode_frame(self, img_hwc_uint8, use_teacher=False):
        """(H,W,3) uint8 -> (1, 256, 1024). With cost='pixel' the goal (use_teacher=True) stays an image (1, 3, 256, 256)."""
        px = self._pixels(img_hwc_uint8)
        if use_teacher and self.cost == "pixel":
            return px
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = self.rae.encode(px).float()
        return z.flatten(2).transpose(1, 2).contiguous()

    def _to_map(self, z_seq):
        B, P, C = z_seq.shape
        h = int(P ** 0.5)
        return z_seq.transpose(1, 2).reshape(B, C, h, h).contiguous()

    @torch.no_grad()
    def _encode_texts(self, texts):
        if len(self.text_cache) > self.TEXT_CACHE_MAX:
            self.text_cache.clear()
        todo = [t for t in dict.fromkeys(texts) if t not in self.text_cache]
        for s in range(0, len(todo), 512):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                enc = self.text_encoder(todo[s : s + 512])
            for i, t in enumerate(todo[s : s + 512]):
                self.text_cache[t] = (enc["tokens"][i].to(torch.bfloat16), enc["attention_mask"][i])
        tok = torch.stack([self.text_cache[t][0] for t in texts]).float()
        mask = torch.stack([self.text_cache[t][1] for t in texts])
        return tok, mask

    @torch.no_grad()
    def _guided_xpred(self, x, t_val, context, mask):
        B = x.shape[0]
        use_cfg = self.cfg_scale > 1.0
        use_ig = self.ig_scale != 1.0 and self.ig_interval[0] <= t_val <= self.ig_interval[1]
        if use_cfg:
            x_in = torch.cat([x, x])
            ctx_in = {"ctx": torch.cat([context["ctx"], context["ctx"]]),
                      "text": torch.cat([context["text"], self.null_tok.expand(B, -1, -1)])}
            mask_in = torch.cat([mask, self.null_mask.expand(B, -1)])
        else:
            x_in, ctx_in, mask_in = x, context, mask
        t = torch.full((x_in.shape[0],), t_val, device=self.device)
        out = self.model(x_in, t, context=ctx_in, attn_mask=mask_in)
        full, base = (out if isinstance(out, tuple) else (out, None))
        full = full.float()
        if use_ig and base is not None:
            full = base.float() + self.ig_scale * (full - base.float())
        if use_cfg:
            x_c, x_u = full.chunk(2, dim=0)
            full = x_u + self.cfg_scale * (x_c - x_u)
        self.timing["dit_forwards"] += 1
        self.timing["dit_samples"] += int(x_in.shape[0])
        return full

    @torch.no_grad()
    def _sample_next(self, ctx_map, tok, mask):
        B = ctx_map.shape[0]
        x = torch.randn(B, *self.latent_size, device=self.device, generator=self.gen)
        context = cond_mod.make_context(ctx_map, tok)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(self.nfe):
                t_cur, t_next = self.t_grid[i], self.t_grid[i + 1]
                x_hat = self._guided_xpred(x, t_cur, context, mask)
                v = (x - x_hat) / max(t_cur, self.transport.t_eps)
                x = x - (t_cur - t_next) * v
        return x.float()

    @torch.no_grad()
    def _predict(self, z_seq, texts):
        tok, mask = self._encode_texts(texts)
        ctx = self._to_map(z_seq)
        out = []
        for s in range(0, ctx.shape[0], self.batch_chunk):
            e = s + self.batch_chunk
            out.append(self._sample_next(ctx[s:e], tok[s:e], mask[s:e]))
        return torch.cat(out).flatten(2).transpose(1, 2).contiguous()

    @torch.no_grad()
    def decode(self, z_seq):
        """(B, 256, C) -> pixels (B, 3, 256, 256) in [0, 1] with the frozen RAE decoder."""
        out = []
        for s in range(0, z_seq.shape[0], self.batch_chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(self.rae.decode(self._to_map(z_seq[s : s + self.batch_chunk])).float().clamp(0, 1))
        return torch.cat(out)

    @torch.no_grad()
    def unroll(self, z_init, act_suffix):
        """z_init = (ctx, proprio); act_suffix (H, B, A) in the CEM space -> (H, B, 256, C) predicted latents
        (cost='pixel': (H, B, 3, 256, 256) with only the last step decoded)."""
        t0 = time.time()
        ctx, proprio0 = z_init
        cur = (ctx[0][-1] if isinstance(ctx, tuple) else ctx)
        H, B, _ = act_suffix.shape
        cur = cur.expand(B, -1, -1).contiguous()
        use_prop = self.dyn.mode != "none"
        prop = np.repeat(np.asarray(proprio0, dtype=np.float64)[None], B, axis=0) if use_prop else None
        act_np = act_suffix.detach().cpu().numpy()
        preds = []
        for h in range(H):
            cand = [self.render(prop[b] if use_prop else None, self.spec.plan_to_z(act_np[h, b]), prefix=self.domain_prefix)
                    for b in range(B)]
            cur = self._predict(cur, cand)
            preds.append(cur)
            if use_prop:
                prop = self.dyn.step(prop, self.spec.denorm_action(self.spec.plan_to_z(act_np[h])))
        self.timing["unroll_calls"] += 1
        if self.cost == "pixel":
            last = self.decode(preds[-1])
            out = torch.zeros(H, *last.shape, device=self.device)
            out[-1] = last
            self.timing["unroll_seconds"] += time.time() - t0
            return out
        self.timing["unroll_seconds"] += time.time() - t0
        return torch.stack(preds)
