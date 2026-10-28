# Copyright (c) Facebook, Inc. and its affiliates. (JEPA-WMs, trimmed to the tensor objectives)


class BaseMPCObjective:
    def __call__(self, encodings, actions, keepdims=False):
        raise NotImplementedError


class ReprTargetDistMPCObjective(BaseMPCObjective):
    """Squared distance between the predicted latents (T x B x ...) and the target latent (1 x ...)."""

    def __init__(self, cfg, target_enc, sum_all_diffs=False, **kwargs):
        self.cfg = cfg
        self.target_enc = target_enc
        self.sum_all_diffs = sum_all_diffs

    def __call__(self, encodings, actions, keepdims=False):
        diff = (self.target_enc - encodings).pow(2).mean(dim=tuple(range(2, encodings.ndim)))
        if not keepdims:
            return diff.sum(0) if self.sum_all_diffs else diff[-1]
        return diff.cumsum(0).flip(0) if self.sum_all_diffs else diff
