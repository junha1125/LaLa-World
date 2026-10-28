from typing import Any, Dict

import torch
from torch.optim import Optimizer


class MuonAdamW(Optimizer):
    """Muon for the 2-D parameters, AdamW for the rest (RAEv2 recipe)."""

    def __init__(self, muon_opt: Optimizer, adamw_opt: Optimizer):
        self._muon = muon_opt
        self._adamw = adamw_opt
        self.param_groups = muon_opt.param_groups + adamw_opt.param_groups
        self.defaults: Dict[str, Any] = {}

    @property
    def state(self) -> Dict:
        merged: Dict = {}
        merged.update(self._muon.state)
        merged.update(self._adamw.state)
        return merged

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        self._muon.step(closure=closure)
        self._adamw.step(closure=closure)

    def state_dict(self) -> Dict[str, Any]:
        return {"muon": self._muon.state_dict(), "adamw": self._adamw.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._muon.load_state_dict(state_dict["muon"])
        self._adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = self._muon.param_groups + self._adamw.param_groups


def build_optimizer(parameters, cfg: dict):
    """cfg: training.optimizer of the yaml (type adamw | gmuon)."""
    params = list(parameters)
    lr = float(cfg.get("lr", 2e-4))
    betas = tuple(cfg.get("betas", (0.9, 0.95)))
    wd = float(cfg.get("weight_decay", 0.0))
    eps = float(cfg.get("eps", 1e-8))
    if cfg.get("type", "adamw") == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd, eps=eps, fused=True), f"AdamW(lr={lr}, wd={wd})"
    if cfg["type"] != "gmuon":
        raise ValueError(f"unknown optimizer {cfg['type']!r} (adamw | gmuon)")
    from gram_newton_schulz import Muon as GMuon

    muon_params = [p for p in params if p.ndim == 2]
    other_params = [p for p in params if p.ndim != 2]
    adamw_opt = torch.optim.AdamW(
        other_params if other_params else [torch.nn.Parameter(torch.empty(0))],
        lr=float(cfg.get("adamw_lr", lr)), betas=betas, weight_decay=wd, eps=eps,
    )
    gmuon_opt = GMuon(
        muon_params,
        lr=lr,
        momentum=float(cfg.get("momentum", 0.95)),
        nesterov=bool(cfg.get("nesterov", True)),
        weight_decay=wd,
        ns_coefficients_preset=cfg.get("ns_coefficients_preset", "POLAR_EXPRESS_COEFFICIENTS"),
        ns_use_kernels=bool(cfg.get("ns_use_kernels", False)),
        adjust_lr="rms_norm",
    )
    msg = f"GMuon(lr={lr}, momentum={cfg.get('momentum', 0.95)}, {len(muon_params)} 2D params, {len(other_params)} AdamW params)"
    return MuonAdamW(gmuon_opt, adamw_opt), msg
