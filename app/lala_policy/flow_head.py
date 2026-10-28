"""FlowHead: Isaac-GR00T's flow-matching action head (DiT) conditioned on the LLM hidden states.

tokens [state | noisy action x K | (noisy p' | noisy V)] -> `layers` DiT blocks (even blocks cross-attend to the
LLM states, odd blocks self-attend, AdaLN on the flow time) -> per-target decoder MLPs. GR00T convention:
t = (1 - Beta(1.5, 1)) * noise_s, x_t = (1 - t) noise + t x, velocity target u = x - noise; Euler sampling at eval.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_ORDER = ("actions", "future_proprio", "value")


def timestep_embedding(t, dim=256):
    """diffusers Timesteps(num_channels=dim, flip_sin_to_cos=True, downscale_freq_shift=1): (B,) -> (B, dim) fp32."""
    half = dim // 2
    exponent = -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    emb = t.float()[:, None] * torch.exp(exponent)[None]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return torch.cat([emb[:, half:], emb[:, :half]], dim=-1)  # flip_sin_to_cos -> [cos | sin]


def token_time_encoding(t, dim):
    """GR00T SinusoidalPositionalEncoding (embodiment_conditioned_mlp.py): (B,) -> (B, dim) fp32."""
    half = dim // 2
    exponent = -torch.arange(half, dtype=torch.float32, device=t.device) * (math.log(10000.0) / half)
    freqs = t.float()[:, None] * torch.exp(exponent)[None]
    return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


def sample_flow_time(n, alpha, beta, noise_s, device):
    """GR00T sample_time: t = (1 - Beta(alpha, beta)) * noise_s  (alpha 1.5, beta 1: more mass near t = 0 = noise)."""
    dist = torch.distributions.Beta(
        torch.tensor(float(alpha), device=device), torch.tensor(float(beta), device=device)
    )
    return (1.0 - dist.sample((n,))) * float(noise_s)


def _bcast(t, x):
    return t.to(x.dtype).view(-1, *([1] * (x.dim() - 1)))


def noisy_targets(x, t, noise):
    """x_t = (1 - t) noise + t x,  u = x - noise   (x / noise (B, ...), t (B,))."""
    tt = _bcast(t, x)
    return (1 - tt) * noise + tt * x, x - noise


def x0_from_velocity(x_t, t, v):
    """The clean estimate implied by a velocity prediction: x_hat = x_t + (1 - t) v."""
    return x_t + (1 - _bcast(t, x_t)) * v.to(x_t.dtype)


def _init_small(module, std):
    """GR00T CategorySpecificLinear: W ~ N(0, 0.02), b = 0."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
            nn.init.zeros_(m.bias)
    return module


class _TimestepEncoder(nn.Module):
    """GR00T TimestepEncoder: Timesteps(256) -> TimestepEmbedding(256 -> dim): Linear, SiLU, Linear."""

    def __init__(self, dim):
        super().__init__()
        self.linear_1 = nn.Linear(256, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, t_disc):
        e = timestep_embedding(t_disc).to(self.linear_1.weight.dtype)
        return self.linear_2(F.silu(self.linear_1(e)))


class _AdaLayerNorm(nn.Module):
    """dit.py AdaLayerNorm: LN(no affine)(x) * (1 + scale) + shift, (scale, shift) = Linear(SiLU(temb))."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

    def forward(self, x, temb):
        scale, shift = self.linear(F.silu(temb)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class _Attention(nn.Module):
    """diffusers Attention as the GR00T DiT builds it: biased q/k/v/out projections, dropout on the output
    projection only (no attention-probability dropout), cross_dim for encoder keys/values."""

    def __init__(self, dim, heads, head_dim, cross_dim=None, dropout=0.0):
        super().__init__()
        inner = heads * head_dim
        self.heads, self.head_dim = int(heads), int(head_dim)
        self.to_q = nn.Linear(dim, inner)
        self.to_k = nn.Linear(cross_dim or dim, inner)
        self.to_v = nn.Linear(cross_dim or dim, inner)
        self.to_out = nn.Linear(inner, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, ctx=None, ctx_pad_mask=None):
        B, Tq, _ = x.shape
        kv = x if ctx is None else ctx.to(x.dtype)
        q = self.to_q(x).view(B, Tq, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(kv).view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(kv).view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        mask = None if ctx_pad_mask is None else (~ctx_pad_mask)[:, None, None, :]  # True = may attend
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.drop(self.to_out(o.transpose(1, 2).reshape(B, Tq, -1)))


class _Block(nn.Module):
    """dit.py BasicTransformerBlock with norm_type ada_norm, final_dropout, gelu-approximate FF (4x)."""

    def __init__(self, dim, heads, head_dim, cross_dim, dropout):
        super().__init__()
        self.norm1 = _AdaLayerNorm(dim)
        self.attn = _Attention(dim, heads, head_dim, cross_dim, dropout)
        self.final_dropout = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, temb, ctx=None, ctx_pad_mask=None):
        x = x + self.final_dropout(self.attn(self.norm1(x, temb), ctx, ctx_pad_mask))
        return x + self.ff(self.norm3(x))


class _TokenEncoder(nn.Module):
    """GR00T MultiEmbodimentActionEncoder (one embodiment): W3(swish(W2([W1(x), sinusoidal(t)])))."""

    def __init__(self, in_dim, width, init_std):
        super().__init__()
        self.width = int(width)
        self.w1 = nn.Linear(in_dim, width)
        self.w2 = nn.Linear(2 * width, width)
        self.w3 = nn.Linear(width, width)
        _init_small(self, init_std)

    def forward(self, x, t_disc):
        a = self.w1(x)  # (B, T, w)
        tau = token_time_encoding(t_disc, self.width).to(a.dtype)[:, None].expand(-1, a.shape[1], -1)
        return self.w3(F.silu(self.w2(torch.cat([a, tau], dim=-1))))


class _MLP(nn.Module):
    """GR00T CategorySpecificMLP (one embodiment): Linear -> ReLU -> Linear."""

    def __init__(self, in_dim, hidden, out_dim, init_std):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden)
        self.layer2 = nn.Linear(hidden, out_dim)
        _init_small(self, init_std)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class FlowHead(nn.Module):
    """cond_dim = the LLM hidden size. `steps` is the eval-time Euler step count kept in the checkpoint config;
    drivers can override it per run (run_*_eval.py --flow-steps)."""

    def __init__(
        self,
        action_dim,
        proprio_dim,
        chunk_size,
        cond_dim,
        pv_pred=True,
        layers=16,
        heads=32,
        head_dim=48,
        out_dim=1024,
        dropout=0.2,
        cond="all",
        time_beta=(1.5, 1.0),
        noise_s=0.999,
        timestep_buckets=1000,
        steps=4,
        state_dropout_p=0.0,
        init_std=0.02,
    ):
        super().__init__()
        if cond not in ("all", "slots"):
            raise ValueError(f"action_head.cond must be all|slots, got {cond!r}")
        if int(steps) < 1 or int(layers) < 1 or int(heads) < 1 or int(head_dim) < 1:
            raise ValueError("action_head.steps / layers / heads / head_dim must be >= 1")
        if not 0.0 <= float(state_dropout_p) < 1.0:
            raise ValueError(f"action_head.state_dropout_p must be in [0, 1), got {state_dropout_p}")
        self.action_dim, self.proprio_dim, self.chunk_size = int(action_dim), int(proprio_dim), int(chunk_size)
        self.pv_pred = bool(pv_pred)
        self.width = int(heads) * int(head_dim)
        self.out_dim, self.cond_mode, self.steps = int(out_dim), cond, int(steps)
        self.time_beta = (float(time_beta[0]), float(time_beta[1]))
        self.noise_s, self.timestep_buckets = float(noise_s), int(timestep_buckets)
        self.state_dropout_p = float(state_dropout_p)
        # target kinds in token order: name -> (n_tokens, dim)
        self.targets = {"actions": (self.chunk_size, self.action_dim)}
        if self.pv_pred:
            self.targets["future_proprio"] = (1, self.proprio_dim)
            self.targets["value"] = (1, 1)
        n_tok = sum(n for n, _ in self.targets.values())
        self.n_tokens = n_tok
        self.vlln = nn.LayerNorm(int(cond_dim))
        self.cond_pos = nn.Parameter(torch.randn(n_tok, int(cond_dim)) * init_std) if cond == "slots" else None
        self.state_encoder = _MLP(self.proprio_dim, self.out_dim, self.width, init_std)
        self.encoders = nn.ModuleDict(
            {k: _TokenEncoder(d, self.width, init_std) for k, (_, d) in self.targets.items()}
        )
        self.pos = nn.Embedding(n_tok, self.width)
        nn.init.normal_(self.pos.weight, std=init_std)
        self.time = _TimestepEncoder(self.width)
        self.blocks = nn.ModuleList(
            [
                _Block(self.width, int(heads), int(head_dim), int(cond_dim) if i % 2 == 0 else None, float(dropout))
                for i in range(int(layers))
            ]
        )
        self.norm_out = nn.LayerNorm(self.width, eps=1e-6, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(self.width, 2 * self.width)
        self.proj_out_2 = nn.Linear(self.width, self.out_dim)
        self.decoders = nn.ModuleDict(
            {k: _MLP(self.out_dim, self.out_dim, d, init_std) for k, (_, d) in self.targets.items()}
        )

    # ------------------------------------------------------------------ shapes
    def _as_tokens(self, name, x):
        n, d = self.targets[name]
        return x.reshape(x.shape[0], n, d)

    def _from_tokens(self, name, y, B):
        n, d = self.targets[name]
        if name == "actions":
            return y.reshape(B, n, d)
        if name == "value":
            return y.reshape(B)
        return y.reshape(B, d)

    def check_targets(self, x):
        missing = [k for k in self.targets if k not in x]
        if missing:
            raise ValueError(f"flow head needs {missing} (pv_pred={self.pv_pred})")

    def discretize(self, t):
        return (t.float() * self.timestep_buckets).long()

    # ------------------------------------------------------------------ network
    def forward(self, x_t, t_disc, cond, proprio, cond_pad_mask=None):
        """x_t: dict of noisy targets (actions (B,K,A), future_proprio (B,P), value (B,)); t_disc (B,) long bucket;
        cond (B,M,cond_dim) LLM states; proprio (B,P) -> dict of velocities with the same shapes."""
        self.check_targets(x_t)
        B = proprio.shape[0]
        if x_t["actions"].shape[1] != self.chunk_size:
            raise ValueError(f"expected a chunk of {self.chunk_size} actions, got {x_t['actions'].shape[1]}")
        if self.cond_pos is not None and cond.shape[1] != self.n_tokens:
            raise ValueError(f"cond=slots expects {self.n_tokens} conditioning tokens, got {cond.shape[1]}")
        toks = torch.cat([self.encoders[k](self._as_tokens(k, x_t[k]), t_disc) for k in self.targets], dim=1)
        toks = toks + self.pos.weight[None].to(toks.dtype)
        state = self.state_encoder(proprio.to(toks.dtype))[:, None]  # (B,1,w)
        if self.training and self.state_dropout_p > 0:
            keep = (torch.rand(B, device=state.device) >= self.state_dropout_p).to(state.dtype)
            state = state * keep[:, None, None]
        h = torch.cat([state, toks], dim=1)
        temb = self.time(t_disc).to(h.dtype)
        c = self.vlln(cond.to(self.vlln.weight.dtype))
        if self.cond_pos is not None:
            c = c + self.cond_pos[None].to(c.dtype)
        for i, blk in enumerate(self.blocks):
            if i % 2 == 0:
                h = blk(h, temb, c, cond_pad_mask)
            else:
                h = blk(h, temb)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        h = self.norm_out(h) * (1 + scale[:, None]) + shift[:, None]
        y = self.proj_out_2(h)[:, 1:]  # drop the state token
        out, off = {}, 0
        for k, (n, _) in self.targets.items():
            out[k] = self._from_tokens(k, self.decoders[k](y[:, off : off + n]), B)
            off += n
        return out

    def noise_like(self, targets, generator=None):
        return {
            k: torch.randn(targets[k].shape, device=targets[k].device, dtype=torch.float32, generator=generator)
            for k in self.targets
        }

    @torch.no_grad()
    def sample(self, cond, proprio, cond_pad_mask=None, steps=None, generator=None, noise=None):
        """GR00T get_action: x = noise at t = 0, `steps` Euler steps of dt = 1/steps; fp32 state, dict of fp32."""
        steps = int(steps or self.steps)
        B, dev = cond.shape[0], cond.device
        shapes = {
            "actions": (B, self.chunk_size, self.action_dim),
            "future_proprio": (B, self.proprio_dim),
            "value": (B,),
        }
        x = {
            k: (
                noise[k].float()
                if noise is not None and k in noise
                else torch.randn(shapes[k], device=dev, dtype=torch.float32, generator=generator)
            )
            for k in self.targets
        }
        dt = 1.0 / steps
        for i in range(steps):
            t_disc = torch.full((B,), int((i / steps) * self.timestep_buckets), device=dev, dtype=torch.long)
            v = self.forward(x, t_disc, cond, proprio, cond_pad_mask)
            x = {k: x[k] + dt * v[k].float() for k in x}
        return x
