"""PolicyAgent: observation dict -> raw action chunk (the eval-side wrapper used by every simulator driver).

  agent = PolicyAgent(ckpt, device="cuda:0")
  actions = agent.act({"primary": HxWx3 uint8, "wrist": ...}, proprio_raw (P,), "put the bowl on the stove")
With a flow head the chunk is a `flow_steps`-step Euler sample from a generator seeded with `seed`.
`act(..., return_extras=True)` also returns the predicted future latent; `imagine_wm=True` adds a world-model pass
(mode 1) conditioned on the chunk about to be executed.
"""

import numpy as np
import torch
from PIL import Image

from app.lala_policy.eval_common import load_policy_model
from app.lala_world.data import default_transform
from app.lala_world.data.policy_jsonl import load_stats, scale_to_unit, unscale_from_unit


class PolicyAgent:
    def __init__(self, ckpt_path, device="cuda:0", stats_path=None, flow_steps=None, seed=None):
        self.device = torch.device(device)
        self.encoder, self.predictor, self.cfg, stats, self.epoch = load_policy_model(ckpt_path, self.device)
        pol = self.cfg["model"]["policy"]
        self.views = list(pol.get("views", ["primary"]))
        self.chunk_size = int(pol.get("chunk_size", 16))
        self.pv_pred = bool(self.predictor.pv_pred)
        self.action_head_kind = getattr(self.predictor, "action_head_kind", "l1")
        self.flow_steps, self.flow_generator, self.imagine_generator = None, None, None
        if self.action_head_kind == "flow":
            self.flow_steps = int(flow_steps) if flow_steps else int(self.predictor.flow_head.steps)
            self.flow_generator = torch.Generator(device=self.device)
            self.flow_generator.manual_seed(int(seed if seed is not None else 0))
            # the imagine_wm pass runs the flow sampler too (its sample is discarded); it draws from this
            # generator so flow_generator -- hence every executed action -- is the same with or without it
            self.imagine_generator = torch.Generator(device=self.device)
            self.imagine_generator.manual_seed(0)
        if stats_path is not None:
            self.stats = load_stats(stats_path)
        elif stats is not None:
            self.stats = {
                k: (np.asarray(stats[f"{k}_min"], dtype=np.float32), np.asarray(stats[f"{k}_max"], dtype=np.float32))
                for k in ("actions", "proprio")
            }
        else:
            raise ValueError("checkpoint has no normalization stats; pass stats_path")
        self.transform = default_transform(self.encoder.image_size, self.encoder.image_mean, self.encoder.image_std)
        self.tokenizer = self.predictor.wm.backbone.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_text_tokens = self.cfg["data"].get("max_text_tokens", 64)
        self.use_amp = self.device.type == "cuda" and self.cfg["meta"].get("dtype", "bfloat16") == "bfloat16"

    def _encode(self, images):
        """{view: HxWx3 uint8} -> (V,N,D) patch latents on the device (the training loader's resize / normalize)."""
        px = torch.stack([self.transform(Image.fromarray(np.asarray(images[v]))) for v in self.views], 0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            return self.encoder(px.to(self.device))["patches"]

    @torch.no_grad()
    def encode(self, images):
        """Latents of real frames, (V,N,D) fp32 on the cpu -- what `future_latent` is trained to match."""
        return self._encode(images).float().cpu()

    @torch.no_grad()
    def act(self, images, proprio, task, return_extras=False, imagine_wm=False):
        if imagine_wm and not return_extras:
            raise ValueError("imagine_wm returns its latent in the extras: pass return_extras=True")
        feats = self._encode(images)[None]  # (1,V,N,D)
        prop = scale_to_unit(np.asarray(proprio, dtype=np.float32), *self.stats["proprio"])
        prop = torch.from_numpy(prop)[None].to(self.device)
        tok = self.tokenizer([task], truncation=True, max_length=self.max_text_tokens, return_tensors="pt")
        ids, mask = tok["input_ids"].to(self.device), tok["attention_mask"].to(self.device)
        mode = torch.zeros(1, dtype=torch.long, device=self.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            out = self.predictor(
                feats, ids, mask, prop, mode, flow_steps=self.flow_steps, flow_generator=self.flow_generator
            )
        actions = unscale_from_unit(out["actions"][0].float().cpu().numpy(), *self.stats["actions"])
        if not return_extras:
            return actions
        extras = {"future_latent": out["future_latent"][0].float().cpu()}
        if self.pv_pred:  # model.policy.pv_pred: false -> no future-proprio / value prediction exists
            extras["value"] = float(out["value"][0].float())
            extras["future_proprio"] = unscale_from_unit(
                out["future_proprio"][0].float().cpu().numpy(), *self.stats["proprio"]
            )
        if imagine_wm:
            # mode 1: the action slots carry the chunk itself instead of queries, p(s' | s, a). The flow head
            # noises its whole target set together, so with pv_pred it is handed the sampled p' / V as well
            # (they only feed its discarded velocity; the p' / V slots stay queries without future_vis).
            given = {"actions": out["actions"]}
            if self.pv_pred and self.action_head_kind == "flow":
                given.update(future_proprio=out["future_proprio"], value=out["value"])
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                wm = self.predictor(
                    feats,
                    ids,
                    mask,
                    prop,
                    torch.ones_like(mode),
                    flow_steps=self.flow_steps,
                    flow_generator=self.imagine_generator,
                    **given,
                )
            extras["current_latent"] = feats[0].float().cpu()
            extras["future_latent_wm"] = wm["future_latent"][0].float().cpu()
        return actions, extras
