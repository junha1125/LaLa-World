import torch
import torch.nn.functional as F


@torch.no_grad()
def telemetry_logs(pred, target, state_feats=None, loss_fn=None):
    """Collapse / health signals: pred_std -> 0 or pred_pairwise_cos -> 1 means a dead predictor;
    the loss must fall below identity_baseline_loss (copying the current frame) to be non-trivial."""
    logs = {
        "telemetry/pred_std": pred.float().std(dim=(0, 1)).mean(),
        "telemetry/target_std": target.float().std(dim=(0, 1)).mean(),
        "telemetry/pred_target_cos": F.cosine_similarity(pred.float(), target.float(), dim=-1).mean(),
    }
    if pred.shape[0] > 1:
        pooled = F.normalize(pred.float().mean(dim=1), dim=-1)
        sim = pooled @ pooled.t()
        logs["telemetry/pred_pairwise_cos"] = sim[~torch.eye(len(sim), dtype=torch.bool, device=sim.device)].mean()
    if state_feats is not None and state_feats.shape == target.shape and loss_fn is not None:
        logs["telemetry/identity_baseline_loss"] = loss_fn(state_feats.float(), target.float())[0]
    return {k: float(v) for k, v in logs.items()}
