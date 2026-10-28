"""Per-environment action normalisation specs and the planning-time text renderers.

The renderers reproduce the training JSONL texts byte for byte (run_planning verifies this at start-up):
semantic = numeric sentence, lang6 / lang4 = binned qualitative phrases. Specs live in specs/.
"""

import json
import math
import os
from bisect import bisect_right

import numpy as np
import yaml

from evals.lala_world.paths import LANG_BINS_DIR, SPEC_DIR

DYN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proprio_dyn.json")

# rope / granular text space = screen-aligned 45-degree rotation of the push start + delta [sx, sy, dx, dy];
# R is its own inverse: env start = [sx, sy] @ R, env end = [sx + dx, sy + dy] @ R.
DEFORM_R = np.array([[-1.0, 1.0], [1.0, 1.0]]) / np.sqrt(2.0)
DEFORM_ENVS = ("rope", "granular")


def load_spec(env_name):
    cls = DroidSpec if env_name == "droid" else DeformSpec if env_name in DEFORM_ENVS else SimSpec
    return cls(env_name)


def _w(v, pos, neg):
    return f"{abs(v):.2f} {pos if v >= 0 else neg}"


SEMANTIC_TEMPLATES = {
    "wall": (
        lambda p: f"the agent is {_w(p[0], 'right of', 'left of')} center, {_w(p[1], 'below', 'above')} center",
        lambda a: f"move the agent {_w(a[0], 'right', 'left')} and {_w(a[1], 'down', 'up')}",
    ),
    "point_maze": (
        lambda p: f"the ball is {_w(p[0], 'right of', 'left of')} center, {_w(p[1], 'below', 'above')} center, "
        f"rolling {_w(p[2], 'right', 'left')} and {_w(p[3], 'down', 'up')}",
        lambda a: f"push the ball with force {_w(a[0], 'right', 'left')} and {_w(a[1], 'down', 'up')}",
    ),
    "pusht": (
        lambda p: f"the round pusher is {_w(p[0], 'right of', 'left of')} center, "
        f"{_w(p[1], 'below', 'above')} center, moving {_w(p[2], 'right', 'left')} and {_w(p[3], 'down', 'up')}",
        lambda a: f"move the pusher {_w(a[0], 'right', 'left')} and {_w(a[1], 'down', 'up')} to push the gray T block",
    ),
    # metaworld: robot-frame axes (+x right, +y forward/away, +z up); dgrip>0 = tighten.
    # Mirrors dataset-note build_sim_pairs.py MetaworldSpec.render verbatim.
    "metaworld": (
        lambda p: f"the robot gripper is {_w(p[0], 'right of', 'left of')} center, "
        f"{_w(p[1], 'ahead of', 'behind')} center, {_w(p[2], 'above', 'below')} center, "
        f"grip {abs(p[3]):.2f} more {'open' if p[3] >= 0 else 'closed'} than usual",
        lambda a: f"move the gripper {_w(a[0], 'right', 'left')}, {_w(a[1], 'forward', 'backward')}, "
        f"{_w(a[2], 'up', 'down')} and {'tighten' if a[3] >= 0 else 'loosen'} the grip by {abs(a[3]):.2f}",
    ),
    # rope/granular: no proprio (dummy in dino_wm) — the proprio slot is never used.
    # Action = screen-aligned start+delta [sx, sy, dx, dy]; mirrors build_sim_traj.py Rope/Granular.
    "rope": (
        None,
        lambda a: f"put the stick {_w(a[0], 'right of', 'left of')} center, {_w(a[1], 'below', 'above')} center, "
        f"then push the rope {_w(a[2], 'right', 'left')} and {_w(a[3], 'down', 'up')}",
    ),
    "granular": (
        None,
        lambda a: f"put the board {_w(a[0], 'right of', 'left of')} center, {_w(a[1], 'below', 'above')} center, "
        f"then push the pile {_w(a[2], 'right', 'left')} and {_w(a[3], 'down', 'up')}",
    ),
}

LANG_TEMPLATES = {
    "wall": "move the agent {0} and {1}",
    "point_maze": "push the ball {0} and {1}",
    "pusht": "move the pusher {0} and {1} to push the gray T block",
    "metaworld": "move the gripper {0}, {1}, {2} and {3}",
    "rope": "place the stick {0} and {1}, then push the rope {2} and {3}",
    "granular": "place the board {0} and {1}, then push the pile {2} and {3}",
    # droid: {0}-{5} = 6 binned continuous dims, {6} = discrete close|open|hold
    "droid": "move the gripper {0}, {1} and {2}, rotate {3}, {4} and {5}, then {6} the gripper",
}


class SimSpec:
    def __init__(self, env_name, spec_dir=SPEC_DIR, bins_dir=None):
        with open(os.path.join(spec_dir, f"{env_name}_v1.yaml")) as f:
            s = yaml.safe_load(f)
        self.name = s["dataset"]
        self.prefix = s["domain_prefix"]
        self.action_dims = s["action_dims"]
        self.proprio_dims = s["proprio_dims"]
        self.action_mean = np.array(s["action_mean"], dtype=np.float64)
        self.action_std = np.array(s["action_std"], dtype=np.float64)
        self.proprio_mean = np.array(s["proprio_mean"], dtype=np.float64)
        self.proprio_std = np.array(s["proprio_std"], dtype=np.float64)
        lo = (np.array(s["action_clip_lo"]) - self.action_mean) / self.action_std
        hi = (np.array(s["action_clip_hi"]) - self.action_mean) / self.action_std
        self.z_clip = float(np.max(np.abs(np.concatenate([lo, hi]))))
        with open(os.path.join(bins_dir or LANG_BINS_DIR, f"{env_name}_v1.yaml")) as f:
            self.lang_dims = yaml.safe_load(f)["dims"]

    def denorm_action(self, z):
        return np.asarray(z, dtype=np.float64) * self.action_std + self.action_mean

    # ---- planning search space: the CEM variable u -> text z. Default: u IS the text z (identity).
    plan_space = "text"

    @property
    def plan_dim(self):
        return len(self.action_dims)

    def plan_to_z(self, u):
        """CEM variable -> z of the text net space (what render/denorm_action consume)."""
        return np.asarray(u, dtype=np.float64)

    def plan_max_norms(self):
        """(max_norms, max_norm_dims) for CEMPlanner: clip every u dim to +-z_clip (spec p1/p99)."""
        return [self.z_clip], [list(range(self.plan_dim))]

    def render(self, proprio_raw, action_z, prefix=True):
        a = ", ".join(f"{k}={v:+.2f}" for k, v in zip(self.action_dims, np.asarray(action_z, dtype=np.float64)))
        if proprio_raw is None:
            text = f"action: {a}"
        else:
            pz = (np.asarray(proprio_raw, dtype=np.float64) - self.proprio_mean) / self.proprio_std
            p = ", ".join(f"{k}={v:+.2f}" for k, v in zip(self.proprio_dims, pz))
            text = f"proprio: {p}; action: {a}"
        return self.prefix + text if prefix else text

    def _with_proprio(self, proprio_raw, act_clause, prefix):
        if proprio_raw is not None and len(self.proprio_dims) == 0:
            if len(np.atleast_1d(proprio_raw)):
                raise ValueError(f"{self.name} has no proprio dims but got proprio {proprio_raw!r}")
            proprio_raw = None  # traj JSONL stores [] per transition for no-proprio envs
        if proprio_raw is None:
            text = act_clause
        else:
            pz = (np.asarray(proprio_raw, dtype=np.float64) - self.proprio_mean) / self.proprio_std
            text = f"{SEMANTIC_TEMPLATES[self.name][0](pz)}; {act_clause}"
        return self.prefix + text if prefix else text

    def render_semantic(self, proprio_raw, action_z, prefix=True):
        act_fn = SEMANTIC_TEMPLATES[self.name][1]
        a = act_fn(np.asarray(action_z, dtype=np.float64) + self.action_mean / self.action_std)
        return self._with_proprio(proprio_raw, a, prefix)

    def _render_lang(self, proprio_raw, action_z, prefix, nb):
        v = np.asarray(action_z, dtype=np.float64) + self.action_mean / self.action_std
        parts = []
        for dim, x in zip(self.lang_dims, v):
            b = dim[f"bins{nb}"]
            side = "pos" if x >= 0 else "neg"
            parts.append(b[f"{side}_phrases"][bisect_right(b[f"{side}_abs_edges"], abs(x))])
        return self._with_proprio(proprio_raw, LANG_TEMPLATES[self.name].format(*parts), prefix)

    def render_lang6(self, proprio_raw, action_z, prefix=True):
        return self._render_lang(proprio_raw, action_z, prefix, 6)

    def render_lang4(self, proprio_raw, action_z, prefix=True):
        return self._render_lang(proprio_raw, action_z, prefix, 4)

    def renderer(self, text_source):
        return {
            "numeric": self.render,
            "semantic": self.render_semantic,
            "lang6": self.render_lang6,
            "lang4": self.render_lang4,
        }[text_source]


class DeformSpec(SimSpec):
    """rope/granular. The text net space stays [sx, sy, dx, dy] (screen-aligned start + delta), but the CEM
    searches in the data's own support: u = [sx, sy, x_end, y_end] — start in the text's rotated coords,
    end in the raw pybullet plane (the collection box |end| <= 1.5 is axis-aligned there, so a per-dim
    clip enforces it). Why (deform_cause_analysis, 2026-08-21): every training push ends inside the centre
    box (corr(start, delta) = -0.96); sampling start and delta independently puts ~91% of proposals outside
    it — pushes that never touch the object and were never seen in training — and CEM optimises the
    model's unconstrained predictions for them. Sampling (start, end) keeps ~85% inside, which is the search
    space dino_wm's own planner uses. plan_to_z converts u to the text z right before rendering/execution,
    so the model input is unchanged. plan_space "start_delta" reproduces the old sampling for comparison."""

    def __init__(self, env_name, spec_dir=SPEC_DIR, bins_dir=None, plan_space="start_end"):
        super().__init__(env_name, spec_dir, bins_dir)
        with open(os.path.join(spec_dir, f"{env_name}_v1.yaml")) as f:
            ps = yaml.safe_load(f).get("plan_space")
        if ps is None:
            raise KeyError(f"{env_name}_v1.yaml has no plan_space block")
        assert ps["dims"] == ["sx", "sy", "x_end", "y_end"], ps["dims"]
        self.plan_mean = np.array(ps["mean"], dtype=np.float64)
        self.plan_std = np.array(ps["std"], dtype=np.float64)
        self.end_box = float(ps["end_box"])
        self.set_plan_space(plan_space)

    def set_plan_space(self, mode):
        if mode not in ("start_end", "start_delta"):
            raise ValueError(f"plan_space must be start_end|start_delta, got {mode!r}")
        self.plan_space = mode

    def plan_to_z(self, u):
        u = np.asarray(u, dtype=np.float64)
        if self.plan_space == "start_delta":
            return u
        a = u * self.plan_std + self.plan_mean
        start_rot, end_env = a[..., :2], a[..., 2:]
        delta_rot = (end_env - start_rot @ DEFORM_R) @ DEFORM_R  # R is its own inverse
        net = np.concatenate([start_rot, delta_rot], axis=-1)
        return (net - self.action_mean) / self.action_std

    def plan_max_norms(self):
        if self.plan_space == "start_delta":
            return super().plan_max_norms()
        end_clip = float(np.min((self.end_box - np.abs(self.plan_mean[2:])) / self.plan_std[2:]))
        return [self.z_clip, end_clip], [[0, 1], [2, 3]]


DROID_GRIP_THRESH = 0.1  # dgrip>0.1 close, <-0.1 open, else hold (build_droid_traj.py rule)


class DroidSpec(SimSpec):
    """droid (traj build only): continuous 6D z-score + gripper as an extra 7th planner dim.

    action_z layout (A=7): z[:6] = (net6 - action_mean)/action_std, z[6] = dgrip/grip_std
    (mean not subtracted — dgrip is ~81% exact-zero, spec `gripper:` block; never a plain
    Gaussian z-score dim). semantic text is physical units (spec text_normalized: false);
    lang6/lang4 bin the 6 continuous dims and append the discrete grip clause. All renders
    mirror dataset-note-multi/droid/build_droid_traj.py byte-for-byte. No numeric render.
    """

    def __init__(self, env_name, spec_dir=SPEC_DIR, bins_dir=None):
        super().__init__(env_name, spec_dir=spec_dir, bins_dir=bins_dir)
        with open(os.path.join(spec_dir, f"{env_name}_v1.yaml")) as f:
            g = yaml.safe_load(f)["gripper"]
        self.grip_std = float(g["std"])
        self.grip_z_clip = max(abs(float(g["clip_lo"])), abs(float(g["clip_hi"]))) / self.grip_std

    def norm_action(self, net):
        net = np.asarray(net, dtype=np.float64)
        return np.concatenate(
            [(net[..., :6] - self.action_mean) / self.action_std, net[..., 6:] / self.grip_std], axis=-1
        )

    def denorm_action(self, z):
        z = np.asarray(z, dtype=np.float64)
        return np.concatenate([z[..., :6] * self.action_std + self.action_mean, z[..., 6:] * self.grip_std], axis=-1)

    def _grip_word(self, dgrip):
        return "close" if dgrip > DROID_GRIP_THRESH else "open" if dgrip < -DROID_GRIP_THRESH else "hold"

    def render(self, proprio_raw, action_z, prefix=True):
        raise ValueError("droid has no numeric render (traj build ships semantic/lang only)")

    def render_semantic(self, proprio_raw, action_z, prefix=True):
        a = self.denorm_action(action_z)
        fb = f"{abs(a[0]) * 100:.1f}cm {'forward' if a[0] >= 0 else 'backward'}"
        lr = f"{abs(a[1]) * 100:.1f}cm {'left' if a[1] >= 0 else 'right'}"
        ud = f"{abs(a[2]) * 100:.1f}cm {'up' if a[2] >= 0 else 'down'}"
        rot = (
            f"roll {math.degrees(a[3]):+.0f}deg, pitch {math.degrees(a[4]):+.0f}deg, "
            f"yaw {math.degrees(a[5]):+.0f}deg"
        )
        act = f"move gripper {fb}, {lr}, {ud}; rotate {rot}; {self._grip_word(a[6])} gripper"
        return self._with_proprio(proprio_raw, act, prefix)

    def _render_lang(self, proprio_raw, action_z, prefix, nb):
        z = np.asarray(action_z, dtype=np.float64)
        v = z[:6] + self.action_mean / self.action_std
        parts = []
        for dim, x in zip(self.lang_dims, v):
            b = dim[f"bins{nb}"]
            side = "pos" if x >= 0 else "neg"
            parts.append(b[f"{side}_phrases"][bisect_right(b[f"{side}_abs_edges"], abs(x))])
        text = LANG_TEMPLATES[self.name].format(*parts, self._grip_word(z[6] * self.grip_std))
        return self._with_proprio(proprio_raw, text, prefix)


class ProprioDynamics:
    """proprio_{t+5} ~= [proprio_t, action_chunk_raw, 1] @ W, fit by fit_proprio_dynamics.py."""

    def __init__(self, env_name, mode="linear", dyn_path=DYN_PATH):
        self.mode = mode
        if mode == "linear":
            with open(dyn_path) as f:
                d = json.load(f)[env_name]
            self.W = np.array(d["W"], dtype=np.float64)
            self.r2 = d["r2"]

    def step(self, proprio_raw, action_raw):
        x = np.concatenate([proprio_raw, action_raw, np.ones((proprio_raw.shape[0], 1))], axis=1)
        return x @ self.W
