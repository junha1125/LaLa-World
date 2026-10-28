import torch as th


def _expand_t(t, x):
    return t.view(t.size(0), *([1] * (len(x.size()) - 1)))


def get_time_sampler(time_dist_type: str):
    parts = time_dist_type.split("_")
    if parts[0] != "logit-normal" or len(parts) != 3:
        raise NotImplementedError(f"Unknown time distribution: {time_dist_type} (expected 'logit-normal_MU_SIGMA')")
    mu, sigma = float(parts[1]), float(parts[2])
    return lambda bs: (th.randn(bs) * sigma + mu).sigmoid()


class Transport:
    """Flow matching with x_t = (1 - t) x_1 + t x_0 (t = 1 is pure noise), x- or velocity-prediction."""

    def __init__(self, prediction="velocity", time_dist_type="logit-normal_0_1", time_dist_shift=1.0, t_eps=0.05):
        self.prediction = prediction
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift
        self.t_eps = t_eps
        self.time_sampler = get_time_sampler(time_dist_type)

    def sample(self, x1):
        x0 = th.randn_like(x1)
        t = self.time_sampler(x1.shape[0]).to(x1)
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t, x0, x1

    def convert_model_pred(self, output, xt, t):
        if self.prediction == "velocity":
            return output
        if self.prediction == "x":
            return (xt - output) / _expand_t(t, xt).clamp_min(self.t_eps)
        raise ValueError(f"unknown prediction type {self.prediction!r}")

    def compute_loss(self, output, vt, xt, t):
        return (self.convert_model_pred(output, xt, t) - vt) ** 2
