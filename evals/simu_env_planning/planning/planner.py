# Copyright (c) Facebook, Inc. and its affiliates. (JEPA-WMs CEMPlanner, unchanged apart from trimming)
from typing import Callable, List, NamedTuple

import torch


class PlanningResult(NamedTuple):
    actions: torch.Tensor
    losses: torch.Tensor = None
    prev_elite_losses_mean: torch.Tensor = None
    prev_elite_losses_std: torch.Tensor = None
    info: dict = None
    plan_metrics: dict = None
    pred_frames_over_iterations: List = None
    predicted_best_encs_over_iterations: List = None


class CEMPlanner:
    def __init__(
        self,
        unroll: Callable,
        iterations: int = 6,
        num_samples: int = 512,
        horizon: int = 32,
        action_dim: int = 4,
        var_scale: float = 1,
        num_elites: int = 64,
        momentum_mean: float = 0.0,
        momentum_std: float = 0.0,
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = None,
        local_generator: torch.Generator = None,
        num_act_stepped: int = None,
        **kwargs,
    ):
        self.objective = None
        self.unroll = unroll
        self.iterations = iterations
        self.num_samples = num_samples
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = torch.device("cuda")
        self.var_scale = var_scale
        self.num_elites = num_elites
        self.momentum_mean = momentum_mean
        self.momentum_std = momentum_std
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.local_generator = local_generator
        self.num_act_stepped = num_act_stepped

    def set_objective(self, objective):
        self.objective = objective

    def cost_function(self, actions, z_init):
        return self.objective(self.unroll(z_init, actions), actions)

    @torch.no_grad()
    def plan(self, z_init, steps_left=None):
        plan_length = self.horizon if steps_left is None else min(self.horizon, steps_left)
        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.var_scale * torch.ones(plan_length, self.action_dim, device=self.device)
        actions = torch.empty(plan_length, self.num_samples, self.action_dim, device=self.device)
        losses, elite_means, elite_stds, predicted_best_encs_over_iterations = [], [], [], []
        for _ in range(self.iterations):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length, self.num_samples, self.action_dim, device=std.device, generator=self.local_generator
            )
            actions[:, 0, :] = mean  # keep the previous mean as a candidate
            if self.max_norms is not None:
                for h in range(plan_length):
                    for dims, maxnorm in zip(self.max_norm_dims, self.max_norms):
                        actions[h, :, dims] = torch.clip(actions[h, :, dims], min=-maxnorm, max=maxnorm)
            cost = self.cost_function(actions, z_init).unsqueeze(1)
            losses.append(cost.min().item())
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())
            mean = torch.mean(elite_actions, dim=1) * (1 - self.momentum_mean) + mean * self.momentum_mean
            std = torch.std(elite_actions, dim=1) * (1 - self.momentum_std) + std * self.momentum_std
            predicted_best_encs_over_iterations.append(self.unroll(z_init, act_suffix=mean.unsqueeze(1)))
        return PlanningResult(
            actions=mean[: self.num_act_stepped],
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
