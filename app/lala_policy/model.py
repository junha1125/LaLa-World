"""LaLa-Policy: the LaLa-World predictor extended with proprio input, action queries and an action head.

Sequence (left-padded):  [pad | vis view 0..V-1 | proprio | task text | action q x K | future vis q x V*N | (p' | V)]
The tail is in prediction order for the causal LLM. `mode` per sample (Cosmos Policy conditioning split):
  0 policy  action / future slots are queries          1 wm  actions are inputs, future slots queries
  2 value   actions and future frame / proprio are inputs, value is the only query (pv_pred only)
`pv_pred: false` drops the future-proprio / value slots (actions + future frame only, the paper's setting).
Action head: `l1` (LayerNorm + Linear per slot) or `flow` (Isaac-GR00T flow-matching DiT,
app/lala_policy/flow_head.py).
The wrapped LaLaWorld weights are loaded verbatim from a pretraining checkpoint (meta.init_from).
"""

import torch
import torch.nn as nn

from app.lala_policy.flow_head import FlowHead, noisy_targets, sample_flow_time, x0_from_velocity
from app.lala_world.models.connector import MLP2x
from app.lala_world.models.pos_embed import _grid_ids
from app.lala_world.models.predictor import LaLaWorld


def policy_position_ids(grid_hw, n_views, text_lens, max_text_len, chunk_size, pv_pred=True):
    """(3, B, S) for the left-padded policy layout; pad slots keep 0 (attention-masked)."""
    gh, gw = grid_hw
    n_vis = gh * gw
    grid = _grid_ids(gh, gw)  # (3, N): t=0, h, w
    B = len(text_lens)
    n_tail = chunk_size + n_views * n_vis + (2 if pv_pred else 0)
    S = max_text_len + (n_views * n_vis + 1) + n_tail
    pos = torch.zeros(3, B, S, dtype=torch.long)
    row = n_views * gh  # first free row beyond every view's grid
    for i, L in enumerate(text_lens):
        L = int(L)
        if L > max_text_len:
            raise ValueError(f"text_len {L} > max_text_len {max_text_len}")
        off = max_text_len - L
        for v in range(n_views):
            pos[:, i, off : off + n_vis] = grid
            pos[1, i, off : off + n_vis] += v * gh
            off += n_vis
        pos[:, i, off] = torch.tensor([0, row, 0])
        off += 1
        ramp = torch.arange(1, L + 1, dtype=torch.long)
        pos[:, i, off : off + L] = ramp.expand(3, L)
        off += L
        pos[0, i, off : off + chunk_size] = L + 1
        pos[1, i, off : off + chunk_size] = row + 1
        pos[2, i, off : off + chunk_size] = torch.arange(chunk_size)
        off += chunk_size
        for v in range(n_views):
            pos[:, i, off : off + n_vis] = grid
            pos[0, i, off : off + n_vis] = L + 1
            pos[1, i, off : off + n_vis] += v * gh
            off += n_vis
        if pv_pred:
            pos[:, i, off] = torch.tensor([L + 1, row, 0])
            pos[:, i, off + 1] = torch.tensor([L + 2, row + 2, 0])
    return pos


class LaLaPolicy(nn.Module):
    def __init__(
        self,
        wm,
        n_views=1,
        proprio_dim=9,
        action_dim=7,
        chunk_size=16,
        init_std=0.02,
        action_head=None,
        pv_pred=True,
    ):
        super().__init__()
        if not isinstance(wm, LaLaWorld):
            raise TypeError("wm must be an LaLaWorld")
        if n_views < 1 or chunk_size < 1:
            raise ValueError(f"n_views {n_views} and chunk_size {chunk_size} must be >= 1")
        self.wm = wm
        self.n_views = int(n_views)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.pv_pred = bool(pv_pred)
        dim = wm.backbone.hidden_size
        self.view_embed = nn.Parameter(torch.zeros(self.n_views, dim))
        self.proprio_in = MLP2x(self.proprio_dim, dim)
        self.action_in = MLP2x(self.action_dim, dim)
        self.action_query = nn.Parameter(torch.randn(dim) * init_std)
        # p' / V slots (queries + heads) exist only with pv_pred: a parameter that never receives a gradient
        # would break DDP static_graph
        self.proprio_query = nn.Parameter(torch.randn(dim) * init_std) if self.pv_pred else None
        self.value_query = nn.Parameter(torch.randn(dim) * init_std) if self.pv_pred else None
        # l1: the heads read LayerNorm-ed hidden states (the LLM's last hidden states carry a few very large
        # channels) with a small init, so they start near zero output; out_head keeps the pretraining path.
        # flow: every target (action / p' / V) comes out of the FlowHead, so no linear head and no head_norm
        # exist (a parameter that never receives a gradient would break DDP static_graph).
        head_cfg = dict(action_head or {})
        kind = str(head_cfg.pop("kind", "l1"))
        if kind not in ("l1", "flow"):
            raise ValueError(f"model.policy.action_head.kind must be l1|flow, got {kind!r}")
        self.action_head_kind = kind
        l1 = kind == "l1"
        self.head_norm = nn.LayerNorm(dim) if l1 else None
        self.action_head = nn.Linear(dim, self.action_dim) if l1 else None
        self.proprio_head = nn.Linear(dim, self.proprio_dim) if l1 and self.pv_pred else None
        self.value_head = nn.Linear(dim, 1) if l1 and self.pv_pred else None
        for head in (self.action_head, self.proprio_head, self.value_head):
            if head is not None:
                nn.init.normal_(head.weight, std=init_std)
                nn.init.zeros_(head.bias)
        self.flow_head = (
            None
            if l1
            else FlowHead(
                self.action_dim,
                self.proprio_dim,
                self.chunk_size,
                cond_dim=dim,
                pv_pred=self.pv_pred,
                init_std=init_std,
                **head_cfg,
            )
        )

    @property
    def n_vis(self):
        return self.wm.n_vis

    @property
    def n_tail(self):
        return self.chunk_size + self.n_views * self.n_vis + (2 if self.pv_pred else 0)

    def forward(
        self,
        vis,
        input_ids,
        text_mask,
        proprio,
        mode,
        actions=None,
        future_proprio=None,
        future_vis=None,
        value=None,
        flow_t=None,
        flow_noise=None,
        flow_steps=None,
        flow_generator=None,
    ):
        """vis (B,V,N,Dv) current frame patches; input_ids/text_mask (B,L); proprio (B,P); mode (B) long.

        actions (B,K,A) / future_proprio (B,P) / future_vis (B,V,N,Dv) are the ground truths used as INPUTS
        by the wm/value modes (their slots stay queries for the other modes); pass None when every sample
        is mode 0. Returns dict: actions (B,K,A), future_latent (B,V,N,Dt) and, with pv_pred, future_proprio
        (B,P), value (B).

        flow head only: with the ground truths given (training / val: actions, and future_proprio + value (B)
        when pv_pred) every target is noised at one t (B,) = flow_t or Beta-sampled (GR00T convention, t = 0 =
        noise), with flow_noise (dict per target) or fresh Gaussian noise; the dict then also carries velocity
        {actions, future_proprio, value}, flow_target (same keys), flow_t (B), and actions / future_proprio /
        value are the detached x0 estimates in train mode or, in eval mode, a `flow_steps`-step Euler sample
        drawn with flow_generator. Without ground truths (a policy call) only the sample is produced. The l1
        head ignores value and the flow_* arguments.
        """
        B, V, N, _ = vis.shape
        if V != self.n_views or N != self.n_vis:
            raise ValueError(f"expected vis (B,{self.n_views},{self.n_vis},Dv), got {tuple(vis.shape)}")
        K, L = self.chunk_size, input_ids.shape[1]
        wm = self.wm
        scale = wm.backbone.embed_scale
        dev = vis.device
        # Branch on what the CALLER passed, never on what this batch happens to hold: with a small
        # per-GPU batch some micro-batch eventually contains no mode>0 sample, `action_in` would then
        # receive no gradient on that rank, and DDP static_graph aborts the run. The m_in / m_fut masks
        # below already zero these inputs out per sample, so the numbers are unchanged either way.
        need_in = actions is not None
        need_fut = self.pv_pred and future_proprio is not None and future_vis is not None
        if bool((mode > 0).any()) and not need_in:
            raise ValueError("wm/value-mode samples need `actions`")
        if bool((mode == 2).any()):
            if not self.pv_pred:
                raise ValueError("value-mode (mode 2) samples need pv_pred: true (there is no value slot)")
            if not need_fut:
                raise ValueError("value-mode samples need `future_proprio` and `future_vis`")

        vis_emb = (wm.connector(vis) + self.view_embed[None, :, None, :]) * scale  # (B,V,N,D)
        vis_emb = vis_emb.reshape(B, V * N, -1)
        txt_emb = wm.backbone.embed_text(input_ids).to(vis_emb.dtype)
        text_lens = text_mask.sum(dim=1)
        prop_emb = (self.proprio_in(proprio) * scale).to(vis_emb.dtype)[:, None]  # (B,1,D)

        m_in = (mode > 0)[:, None, None].to(vis_emb.dtype)  # action slots carry inputs
        m_fut = (mode == 2)[:, None, None].to(vis_emb.dtype)  # future slots carry inputs
        act_q = (self.action_query * scale).to(vis_emb.dtype).expand(B, K, -1)
        if need_in:
            act_in = (self.action_in(actions) * scale).to(vis_emb.dtype)
            act_slots = m_in * act_in + (1 - m_in) * act_q
        else:
            act_slots = act_q
        fv_q = (wm.query_token * scale).to(vis_emb.dtype).expand(B, V, N, -1)
        fv_q = (fv_q + (self.view_embed[None, :, None, :] * scale).to(vis_emb.dtype)).reshape(B, V * N, -1)
        fp_q = (self.proprio_query * scale).to(vis_emb.dtype).expand(B, 1, -1) if self.pv_pred else None
        if need_fut:
            fv_in = ((wm.connector(future_vis) + self.view_embed[None, :, None, :]) * scale).to(vis_emb.dtype)
            fv_in = fv_in.reshape(B, V * N, -1)
            fp_in = (self.proprio_in(future_proprio) * scale).to(vis_emb.dtype)[:, None]
            fv_slots = m_fut * fv_in + (1 - m_fut) * fv_q
            fp_slots = m_fut * fp_in + (1 - m_fut) * fp_q
        else:
            fv_slots, fp_slots = fv_q, fp_q

        # left-padded head [pad | vis | proprio | text] via one gather (as predictor.pack_left), fixed tail
        head_src = torch.cat([vis_emb, prop_emb, txt_emb], dim=1)  # (B, V*N+1+L, D)
        n_head_fixed = V * N + 1
        offsets = (L - text_lens).long().to(dev)
        slots = torch.arange(n_head_fixed + L, device=dev)
        idx = (slots[None, :] - offsets[:, None]).clamp(min=0)
        head = head_src.gather(1, idx[..., None].expand(-1, -1, head_src.shape[-1]))
        tail = [act_slots, fv_slots]
        if self.pv_pred:
            val_q = (self.value_query * scale).to(vis_emb.dtype).expand(B, 1, -1)
            tail += [fp_slots, val_q]
        x = torch.cat([head] + tail, dim=1)
        S = x.shape[1]
        attn_mask = (torch.arange(S, device=dev)[None, :] >= offsets[:, None]).long()
        pos3d = policy_position_ids(wm.grid_hw, V, text_lens.tolist(), L, K, self.pv_pred).to(dev)

        h = wm._run_backbone(x, attn_mask, pos3d)
        tail_h = h[:, -self.n_tail :]
        act_h = tail_h[:, :K]
        fv_h = tail_h[:, K : K + V * N]
        out = {"future_latent": wm.out_head(fv_h).reshape(B, V, N, -1)}
        pv_h = tail_h[:, K + V * N :] if self.pv_pred else None  # (B, 2, D): p' then V
        if self.flow_head is None:
            out["actions"] = self.action_head(self.head_norm(act_h))
            if self.pv_pred:
                out["future_proprio"] = self.proprio_head(self.head_norm(pv_h[:, 0]))
                out["value"] = self.value_head(self.head_norm(pv_h[:, 1])).squeeze(-1)
            return out
        fh = self.flow_head
        if fh.cond_mode == "all":
            cond, cond_pad = h, attn_mask == 0
        else:  # the target-slot states, in token order [action x K | p' | V]
            cond, cond_pad = (torch.cat([act_h, pv_h], dim=1) if self.pv_pred else act_h), None
        clean = {"actions": actions}
        if self.pv_pred:
            clean.update(future_proprio=future_proprio, value=value)
        given = [k for k, v_ in clean.items() if v_ is not None]
        if given:
            if len(given) != len(clean):
                raise ValueError(f"flow head: got {given} but needs all of {list(clean)} to noise them together")
            t = flow_t if flow_t is not None else sample_flow_time(B, *fh.time_beta, fh.noise_s, dev)
            clean = {k: v_.float() for k, v_ in clean.items()}
            noise = fh.noise_like(clean)
            if flow_noise is not None:
                noise.update({k: v_.float() for k, v_ in flow_noise.items()})
            x_t, u = {}, {}
            for k in clean:
                x_t[k], u[k] = noisy_targets(clean[k], t, noise[k])
            v = fh(x_t, fh.discretize(t), cond, proprio, cond_pad)
            out.update(velocity=v, flow_target=u, flow_t=t)
            out.update({k: x0_from_velocity(x_t[k], t, v[k]).detach() for k in clean})
        elif self.training:
            raise ValueError(
                "flow head: a training forward needs the clean targets (actions, and p' / V with pv_pred)"
            )
        if not self.training:  # val / policy call: the policy's real output is the sampled target set
            out.update(fh.sample(cond, proprio, cond_pad, steps=flow_steps, generator=flow_generator))
        return out

    def load_world_model(self, wm_state):
        """Load `predictor.*`-stripped LaLaWorld weights into the wrapped world model (strict)."""
        missing, unexpected = self.wm.load_state_dict(wm_state, strict=False)
        unexpected = [k for k in unexpected]
        if missing or unexpected:
            raise RuntimeError(f"world-model checkpoint mismatch: missing {missing} unexpected {unexpected}")
        return self


def build_policy_predictor(wm_predictor, cfg_policy):
    """cfg_policy: {views: [...], proprio_dim, action_dim, chunk_size, pv_pred, action_head: {kind: l1|flow, ...}}."""
    return LaLaPolicy(
        wm_predictor,
        n_views=len(cfg_policy.get("views", ["primary"])),
        proprio_dim=int(cfg_policy.get("proprio_dim", 9)),
        action_dim=int(cfg_policy.get("action_dim", 7)),
        chunk_size=int(cfg_policy.get("chunk_size", 16)),
        action_head=cfg_policy.get("action_head"),
        pv_pred=bool(cfg_policy.get("pv_pred", True)),
    )
