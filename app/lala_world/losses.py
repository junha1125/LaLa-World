import torch.nn as nn
import torch.nn.functional as F


class WMRegressionLoss(nn.Module):
    """Weighted cos / L1 / L2 / smooth-L1 between predicted and target patch latents."""

    def __init__(self, cos_weight=0.0, l1_weight=0.0, l2_weight=1.0, smooth_l1_weight=0.0, normalize_reps=False):
        super().__init__()
        weights = dict(cos=cos_weight, l1=l1_weight, l2=l2_weight, smooth_l1=smooth_l1_weight)
        if all(w == 0 for w in weights.values()):
            raise ValueError("all wm_regression weights are zero")
        self.weights = weights
        self.normalize_reps = normalize_reps

    def forward(self, pred, target):
        if self.normalize_reps:
            pred = F.layer_norm(pred, (pred.size(-1),))
            target = F.layer_norm(target, (target.size(-1),))
        components = {}
        if self.weights["cos"]:
            components["cos"] = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
        if self.weights["l1"]:
            components["l1"] = F.l1_loss(pred, target)
        if self.weights["l2"]:
            components["l2"] = F.mse_loss(pred, target)
        if self.weights["smooth_l1"]:
            components["smooth_l1"] = F.smooth_l1_loss(pred, target)
        loss = sum(self.weights[k] * v for k, v in components.items())
        return loss, {f"loss_{k}": v.detach() for k, v in components.items()}


class IJepaLoss(nn.Module):
    """I-JEPA recipe: layer-normalised target, smooth-L1."""

    def forward(self, pred, target):
        target = F.layer_norm(target, (target.size(-1),))
        loss = F.smooth_l1_loss(pred, target)
        return loss, {"loss_smooth_l1": loss.detach()}


LOSS_REGISTRY = {"wm_regression": WMRegressionLoss, "ijepa": IJepaLoss}


def build_loss(cfg):
    cfg = dict(cfg)
    kind = cfg.pop("kind", "wm_regression")
    if kind not in LOSS_REGISTRY:
        raise ValueError(f"unknown loss kind {kind!r}; known: {sorted(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[kind](**cfg)
