"""Joint policy / world-model / value loss under Cosmos's per-sample mode masks.

action: L1 (l1 head) or flow-matching velocity MSE (flow head) over policy samples; latent: L2 on the future
latent over policy + wm samples; proprio / value (pv_pred only): over policy + wm / all samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.lala_world.data.policy_jsonl import MODE_POLICY, MODE_VALUE, MODE_WM


def _masked_mean(per_sample, mask):
    mask = mask.to(per_sample.dtype)
    return (per_sample * mask).sum() / mask.sum().clamp(min=1.0)


class PolicyLoss(nn.Module):
    def __init__(self, action_w=1.0, latent_w=1.0, proprio_w=1.0, value_w=1.0, latent_loss="l2"):
        super().__init__()
        if latent_loss not in ("l2", "l1", "smooth_l1"):
            raise ValueError(f"latent_loss must be l2|l1|smooth_l1, got {latent_loss!r}")
        self.w = dict(action=float(action_w), latent=float(latent_w), proprio=float(proprio_w), value=float(value_w))
        self.latent_loss = latent_loss

    def forward(self, out, batch, target_latent):
        """out: predictor dict; batch: collated dict (actions/future_proprio/value/mode); target_latent (B,V,N,D)."""
        mode = batch["mode"]
        is_policy, is_wm = mode == MODE_POLICY, mode == MODE_WM
        flow = "velocity" in out  # flow head: velocity MSE per target kind stands in for the L1 terms
        tl = target_latent.to(out["future_latent"].dtype)
        if self.latent_loss == "l2":
            lat = F.mse_loss(out["future_latent"], tl, reduction="none")
        elif self.latent_loss == "l1":
            lat = F.l1_loss(out["future_latent"], tl, reduction="none")
        else:
            lat = F.smooth_l1_loss(out["future_latent"], tl, reduction="none")
        lat = lat.mean(dim=(1, 2, 3))

        def per_sample(name):
            if flow:
                v, u = out["velocity"][name].float(), out["flow_target"][name].float()
                err = F.mse_loss(v, u, reduction="none")
            else:
                err = F.l1_loss(out[name], batch[name].to(out[name].dtype), reduction="none")
            return err.reshape(err.shape[0], -1).mean(dim=1)

        terms = {
            "action": _masked_mean(per_sample("actions"), is_policy),
            "latent": _masked_mean(lat, is_policy | is_wm),
        }
        if "future_proprio" in out:  # pv_pred
            terms["proprio"] = _masked_mean(per_sample("future_proprio"), is_policy | is_wm)
        if "value" in out:
            terms["value"] = _masked_mean(per_sample("value"), torch.ones_like(is_policy))
        loss = sum(self.w[k] * v for k, v in terms.items())
        logs = {f"loss_{k}": v.detach() for k, v in terms.items()}
        logs["n_policy"] = is_policy.sum().float()
        logs["n_wm"] = is_wm.sum().float()
        logs["n_value"] = (mode == MODE_VALUE).sum().float()
        return loss, logs


def build_policy_loss(cfg):
    cfg = dict(cfg or {})
    cfg.pop("kind", None)
    return PolicyLoss(**cfg)
