"""LaLa-World adapter for CEM planning: renders candidate actions to text, predicts the next latent and feeds the
prediction back as the next context frame. ctxt_window w keeps the last w (frame, text) blocks as context."""

import numpy as np
import torch
from PIL import Image

from app.lala_world.data import default_transform
from app.lala_world.eval_common import load_model


def trained_ctxt_window(cfg):
    """Context frames the checkpoint was trained with: its longest trajectory clip_len minus the target frame."""
    sources = cfg.get("data", {}).get("sources", [])
    clips = [int(s.get("clip_len", 0)) for s in sources if s.get("kind") == "traj_jsonl"]
    return max(max(clips) - 1, 1) if clips else 1


class PlanningWM:
    def __init__(
        self,
        ckpt_path,
        spec,
        dyn,
        device,
        domain_prefix=False,
        batch_chunk=512,
        text_source="semantic",
        ctxt_window=None,
    ):
        self.spec = spec
        self.dyn = dyn
        self.device = device
        self.domain_prefix = domain_prefix
        self.batch_chunk = batch_chunk
        self.render = spec.renderer(text_source)
        self.encoder, self.predictor, self.cfg, self.epoch = load_model(ckpt_path, device)
        trained = trained_ctxt_window(self.cfg)
        self.ctxt_window = trained if ctxt_window is None else int(ctxt_window)
        if self.ctxt_window < 1:
            raise ValueError(f"ctxt_window must be >= 1, got {ctxt_window}")
        if self.ctxt_window > trained:
            print(f"[lala] ctxt_window {self.ctxt_window} > {trained} context frames seen in training")
        self.tokenizer = self.predictor.backbone.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_text_tokens = self.cfg["data"].get("max_text_tokens", 64)
        self.transform = default_transform(self.encoder.image_size, self.encoder.image_mean, self.encoder.image_std)
        self.use_amp = device.type == "cuda" and self.cfg["meta"].get("dtype", "bfloat16") == "bfloat16"

    @torch.no_grad()
    def encode_frame(self, img_hwc_uint8, use_teacher=False):
        px = self.transform(Image.fromarray(img_hwc_uint8)).unsqueeze(0).to(self.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            return self.encoder(px)["patches"].float()

    def _tokenize(self, texts):
        return self.tokenizer(
            texts, padding="longest", truncation=True, max_length=self.max_text_tokens, return_tensors="pt"
        )

    @torch.no_grad()
    def _predict(self, state_feats, texts):
        tok = self._tokenize(texts)
        ids, mask = tok["input_ids"].to(self.device), tok["attention_mask"].to(self.device)
        out = []
        for s in range(0, state_feats.shape[0], self.batch_chunk):
            e = s + self.batch_chunk
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                out.append(self.predictor(state_feats[s:e], ids[s:e], mask[s:e]).float())
        return torch.cat(out)

    @torch.no_grad()
    def _predict_ctx(self, frames, texts):
        """frames: list of w (B,P,D) in time order; texts[j]: the B texts of the chunk leaving frames[j]."""
        B, w = frames[0].shape[0], len(frames)
        tok = self._tokenize([texts[j][b] for b in range(B) for j in range(w)])
        L = tok["input_ids"].shape[1]
        ids = tok["input_ids"].reshape(B, w, L).to(self.device)
        mask = tok["attention_mask"].reshape(B, w, L).to(self.device)
        vis = torch.stack(frames, dim=1)
        cl = torch.full((B,), w, dtype=torch.long, device=self.device)
        out = []
        for s in range(0, B, self.batch_chunk):
            e = s + self.batch_chunk
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                out.append(self.predictor(vis[s:e], ids[s:e], mask[s:e], ctx_len=cl[s:e]).float())
        return torch.cat(out)

    @torch.no_grad()
    def unroll(self, z_init, act_suffix):
        """z_init = (ctx, proprio_raw); ctx = patches (1,P,D) or (frames list, executed texts list).
        act_suffix (H,B,A) in the planning space -> predicted latents (H,B,P,D)."""
        ctx, proprio0 = z_init
        hist_f, hist_t = ctx if isinstance(ctx, tuple) else ([ctx], [])
        w = self.ctxt_window
        hist_f = list(hist_f)[-w:]
        hist_t = list(hist_t)[-(len(hist_f) - 1) :] if len(hist_f) > 1 else []
        H, B, _ = act_suffix.shape
        frames = [f.expand(B, -1, -1).contiguous() for f in hist_f]
        texts = [[t] * B for t in hist_t]
        use_prop = self.dyn.mode != "none"
        prop = np.repeat(np.asarray(proprio0, dtype=np.float64)[None], B, axis=0) if use_prop else None
        act_np = act_suffix.detach().cpu().numpy()
        preds = []
        for h in range(H):
            cand = [
                self.render(
                    prop[b] if use_prop else None, self.spec.plan_to_z(act_np[h, b]), prefix=self.domain_prefix
                )
                for b in range(B)
            ]
            pred = self._predict(frames[0], cand) if len(frames) == 1 else self._predict_ctx(frames, texts + [cand])
            preds.append(pred)
            frames = (frames + [pred])[-w:]
            texts = (texts + [cand])[-(w - 1) :] if w > 1 else []
            if use_prop:
                prop = self.dyn.step(prop, self.spec.denorm_action(self.spec.plan_to_z(act_np[h])))
        return torch.stack(preds)
