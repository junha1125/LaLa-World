"""Simulation datasets (metaworld, point_maze, pusht, wall, rope, granular) -> episode-record JSONL.

One JSONL line per episode: every frame of the episode (dense, stride 1) plus the action text for every
window start u (u -> u + window). window = 5 simulator steps (summed per-step actions) for the first four
environments and 1 push for rope/granular. Six text fields are written per transition:

  actions_semantic              proprio clause + numerical action clause (net/std)
  actions_semantic_no_proprio   numerical action clause only
  actions_lang6 / actions_lang4 proprio clause + qualitative (6- or 4-bin) action clause
  actions_lang6_no_proprio / actions_lang4_no_proprio   qualitative clause only

  python data_preparation/build_sim_traj.py --env wall --root data --stats-only   # specs/ + lang_bins/ (already shipped)
  python data_preparation/build_sim_traj.py --env wall --root data --workers 24   # frames + jsonl/traj/wall.jsonl
"""

import argparse
import functools
import io
import json
import os
import pickle
import sys
from bisect import bisect_right

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import add_root_arg, data_root  # noqa: E402

WINDOW = 5
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_DIR = os.path.join(HERE, "specs")
BINS_DIR = os.path.join(HERE, "lang_bins")
ROOT = None  # set in main(); adapters resolve paths lazily through raw_dir()

R6 = lambda a: [round(float(x), 6) for x in np.atleast_1d(a)]  # noqa: E731


def raw_dir():
    return os.path.join(data_root(ROOT), "sim_raw")


def _w(v, pos, neg):
    return f"{abs(v):.2f} {pos if v >= 0 else neg}"


def save_jpeg(arr_hwc_uint8, path):
    Image.fromarray(arr_hwc_uint8).save(path, quality=95)


INTENSITY = {
    "move": ["slightly {d}", "{d}", "far {d}"],
    "force": ["gently {d}", "{d}", "hard {d}"],
    "grip": ["{d} the grip a little", "{d} the grip", "{d} the grip a lot"],
}


def phrases(kind, direction, n_bins):
    full = [t.format(d=direction) for t in INTENSITY[kind]]
    return full if n_bins == 6 else [full[0], full[2]]


class Metaworld:
    name = "metaworld"
    window = WINDOW
    action_keys = ["dx", "dy", "dz", "dgrip"]
    proprio_keys = ["x", "y", "z", "grip"]
    units_note = {
        "action": "5-step sum of per-step EE delta commands (raw range [-1, 1], simulator units); dgrip > 0 = close",
        "proprio": "state[:4] = EE xyz + gripper openness (0 = closed), simulator units",
    }
    axis_note = "robot frame: +x = right, +y = forward (away from the robot), +z = up (not screen directions)"
    lang_dims = [
        ("dx", "move", "right", "left"),
        ("dy", "move", "forward", "backward"),
        ("dz", "move", "up", "down"),
        ("dgrip", "grip", "tighten", "loosen"),
    ]
    lang_template = "move the gripper {0}, {1}, {2} and {3}"

    @property
    def root(self):
        return os.path.join(raw_dir(), "metaworld", "data")

    def render_proprio(self, pz):
        return (
            f"the robot gripper is {_w(pz[0], 'right of', 'left of')} center, "
            f"{_w(pz[1], 'ahead of', 'behind')} center, {_w(pz[2], 'above', 'below')} center, "
            f"grip {abs(pz[3]):.2f} more {'open' if pz[3] >= 0 else 'closed'} than usual"
        )

    def render_action(self, av):
        return (
            f"move the gripper {_w(av[0], 'right', 'left')}, {_w(av[1], 'forward', 'backward')}, "
            f"{_w(av[2], 'up', 'down')} and {'tighten' if av[3] >= 0 else 'loosen'} the grip by {abs(av[3]):.2f}"
        )

    def units(self):
        return sorted(f for f in os.listdir(self.root) if f.endswith(".parquet"))

    def iter_trajs(self, unit, with_frames):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(os.path.join(self.root, unit))
        cols = ["task", "seed", "episode", "states", "actions"] + (["video"] if with_frames else [])
        for rg in range(pf.num_row_groups):
            tb = pf.read_row_group(rg, columns=cols)
            for i in range(tb.num_rows):
                states = np.asarray(tb["states"][i].as_py(), dtype=np.float64)[:-1]
                actions = np.asarray(tb["actions"][i].as_py(), dtype=np.float64)
                traj = f"{tb['task'][i].as_py()}_s{tb['seed'][i].as_py()}_e{tb['episode'][i].as_py():03d}"
                get = None
                if with_frames:
                    video = tb["video"][i].as_py()["bytes"]

                    def get(idxs, _v=video):
                        import imageio

                        frames = np.stack([fr for fr in imageio.get_reader(io.BytesIO(_v), format="mp4")])
                        return frames[idxs]

                yield traj, actions, states[:, :4], len(actions), get


class TensorEnv:
    window = WINDOW
    chunk = 64

    def units(self):
        n = self._n()
        return [list(range(i, min(i + self.chunk, n))) for i in range(0, n, self.chunk)]


class PointMaze(TensorEnv):
    name = "point_maze"
    action_keys = ["fx", "fy"]
    proprio_keys = ["x", "y", "vx", "vy"]
    units_note = {
        "action": "5-step sum of the per-step force (impulse approximation)",
        "proprio": "state = x, y, vx, vy (simulator units)",
    }
    axis_note = "image: +x = right (corr +1.00), +y = down (corr +1.00)"
    lang_dims = [("fx", "force", "right", "left"), ("fy", "force", "down", "up")]
    lang_template = "push the ball {0} and {1}"

    @property
    def root(self):
        return os.path.join(raw_dir(), "point_maze")

    def render_proprio(self, pz):
        return (
            f"the ball is {_w(pz[0], 'right of', 'left of')} center, {_w(pz[1], 'below', 'above')} center, "
            f"rolling {_w(pz[2], 'right', 'left')} and {_w(pz[3], 'down', 'up')}"
        )

    def render_action(self, av):
        return f"push the ball with force {_w(av[0], 'right', 'left')} and {_w(av[1], 'down', 'up')}"

    @functools.lru_cache(maxsize=1)
    def _data(self):
        import torch

        st = torch.load(f"{self.root}/states.pth", map_location="cpu").double().numpy()
        ac = torch.load(f"{self.root}/actions.pth", map_location="cpu").double().numpy()
        sl = torch.load(f"{self.root}/seq_lengths.pth", map_location="cpu").numpy()
        return st, ac, sl

    def _n(self):
        return len(self._data()[2])

    def iter_trajs(self, unit, with_frames):
        st, ac, sl = self._data()
        for i in unit:
            get = None
            if with_frames:

                def get(idxs, _i=i):
                    import torch

                    img = torch.load(f"{self.root}/obses/episode_{_i:03d}.pth", map_location="cpu").numpy()
                    return img[idxs].astype(np.uint8)

            yield f"ep{i:04d}", ac[i, : sl[i]], st[i, : sl[i]], int(sl[i]), get


class Wall(TensorEnv):
    name = "wall"
    action_keys = ["dx", "dy"]
    proprio_keys = ["x", "y"]
    units_note = {
        "action": "5-step sum of the per-step position delta (simulator units)",
        "proprio": "state = x, y position (simulator units)",
    }
    axis_note = "image: +x = right (corr +0.95), +y = down (corr +0.90)"
    lang_dims = [("dx", "move", "right", "left"), ("dy", "move", "down", "up")]
    lang_template = "move the agent {0} and {1}"

    @property
    def root(self):
        return os.path.join(raw_dir(), "wall_single")

    def render_proprio(self, pz):
        return f"the agent is {_w(pz[0], 'right of', 'left of')} center, {_w(pz[1], 'below', 'above')} center"

    def render_action(self, av):
        return f"move the agent {_w(av[0], 'right', 'left')} and {_w(av[1], 'down', 'up')}"

    @functools.lru_cache(maxsize=1)
    def _data(self):
        import torch

        st = torch.load(f"{self.root}/states.pth", map_location="cpu").double().numpy()
        ac = torch.load(f"{self.root}/actions.pth", map_location="cpu").double().numpy()
        return st, ac

    def _n(self):
        return len(self._data()[0])

    def iter_trajs(self, unit, with_frames):
        st, ac = self._data()
        for i in unit:
            get = None
            if with_frames:

                def get(idxs, _i=i):
                    import torch

                    img = torch.load(f"{self.root}/obses/episode_{_i:03d}.pth", map_location="cpu").numpy()
                    sel = img[idxs]
                    if sel.shape[1] == 3 and sel.shape[-1] != 3:
                        sel = sel.transpose(0, 2, 3, 1)
                    return np.clip(np.rint(sel), 0, 255).astype(np.uint8)

            yield f"ep{i:04d}", ac[i], st[i], ac.shape[1] + 1, get


class PushT(TensorEnv):
    name = "pusht"
    action_keys = ["dx", "dy"]
    proprio_keys = ["x", "y", "vx", "vy"]
    units_note = {
        "action": "5-step sum of the pusher target displacement in px (env interface uses px/100, action_scale = 100)",
        "proprio": "state[:2] = pusher x, y (px) + velocity vx, vy",
    }
    axis_note = "image: +x = right (corr +0.86), +y = down (corr +0.84)"
    lang_dims = [("dx", "move", "right", "left"), ("dy", "move", "down", "up")]
    lang_template = "move the pusher {0} and {1} to push the gray T block"
    chunk = 256

    @property
    def root(self):
        return os.path.join(raw_dir(), "pusht_noise", "train")  # val/ (21 episodes) is kept for planning goals

    def render_proprio(self, pz):
        return (
            f"the round pusher is {_w(pz[0], 'right of', 'left of')} center, {_w(pz[1], 'below', 'above')} center, "
            f"moving {_w(pz[2], 'right', 'left')} and {_w(pz[3], 'down', 'up')}"
        )

    def render_action(self, av):
        return f"move the pusher {_w(av[0], 'right', 'left')} and {_w(av[1], 'down', 'up')} to push the gray T block"

    @functools.lru_cache(maxsize=1)
    def _data(self):
        import torch

        st = torch.load(f"{self.root}/states.pth", map_location="cpu").double().numpy()
        ac = torch.load(f"{self.root}/rel_actions.pth", map_location="cpu").double().numpy()
        vel = torch.load(f"{self.root}/velocities.pth", map_location="cpu").double().numpy()
        with open(f"{self.root}/seq_lengths.pkl", "rb") as f:
            sl = pickle.load(f)
        return st, ac, vel, sl

    def _n(self):
        return len(self._data()[3])

    def iter_trajs(self, unit, with_frames):
        st, ac, vel, sl = self._data()
        for i in unit:
            T = int(sl[i])
            proprio = np.concatenate([st[i, :T, :2], vel[i, :T]], axis=1)
            get = None
            if with_frames:

                def get(idxs, _i=i):
                    import decord

                    decord.bridge.set_bridge("native")
                    vr = decord.VideoReader(f"{self.root}/obses/episode_{_i:03d}.mp4", num_threads=1)
                    return vr.get_batch(list(idxs)).asnumpy()

            yield f"ep{i:05d}", ac[i, :T], proprio, T, get


# pybullet plane [ax, ay] <-> screen-aligned [h, v] (+h = screen right, +v = screen down), a 45-degree rotation.
# Self-inverse, so the planner uses the same matrix to map text actions back to pushes.
DEFORM_R = np.array([[-1.0, 1.0], [1.0, 1.0]]) / np.sqrt(2.0)


class Deformable(TensorEnv):
    """DINO-WM deformable format: actions.pth + <ep>/obses.pth. One action = one push macro
    [x_start, y_start, x_end, y_end]; net = screen-aligned start + delta [sx, sy, dx, dy].
    frames 20 = actions 20, so the last action is unused (19 transitions). No proprio."""

    window = 1
    chunk = 16
    action_keys = ["sx", "sy", "dx", "dy"]
    proprio_keys = []

    def render_proprio(self, pz):
        return None

    @functools.lru_cache(maxsize=1)
    def _actions(self):
        import torch

        return torch.load(f"{self.root}/actions.pth", map_location="cpu").double().numpy()

    def _n(self):
        return len(self._actions())

    def iter_trajs(self, unit, with_frames):
        ac = self._actions()
        for i in unit:
            a = ac[i]
            net = np.concatenate([a[:, :2] @ DEFORM_R, (a[:, 2:] - a[:, :2]) @ DEFORM_R], axis=1)
            get = None
            if with_frames:

                def get(idxs, _i=i):
                    import torch

                    img = torch.load(f"{self.root}/{_i:06d}/obses.pth", map_location="cpu").numpy()
                    return img[idxs].astype(np.uint8)

            yield f"ep{i:04d}", net, np.zeros((a.shape[0], 0)), a.shape[0], get

    def plan_space_block(self):
        """CEM variables u = [sx, sy, x_end, y_end]: screen-aligned start (as in the text) + raw end point
        (the collection box |end| <= 1.5 is axis aligned in the raw plane). Sampling start and delta
        independently would put most end points outside the box."""
        a = self._actions().reshape(-1, 4)
        start_rot, end = a[:, :2] @ DEFORM_R, a[:, 2:]
        return {
            "dims": ["sx", "sy", "x_end", "y_end"],
            "mean": R6(np.concatenate([start_rot.mean(0), end.mean(0)])),
            "std": R6(np.concatenate([start_rot.std(0), end.std(0)])),
            "end_box": 1.5,
            "note": "CEM variables u = [sx, sy, x_end, y_end] (start: screen-aligned rotated coordinates = text, "
            "end: raw plane coordinates, collection box |end| <= 1.5). Before rendering, delta = (end - start@R)@R "
            "joins the text net [sx, sy, dx, dy] (sim_spec.DeformSpec.plan_to_z).",
        }


class Rope(Deformable):
    name = "rope"
    units_note = {
        "action": "one push macro. Raw action [x_start, y_start, x_end, y_end] (pybullet plane, y = -FleX z; start in "
        "[-4, 4]^2, end in [-1.5, 1.5]^2) rotated by R = [[-1, 1], [1, 1]]/sqrt(2) (self-inverse) into start + delta: "
        "[sx, sy] = start@R, [dx, dy] = (end - start)@R. No summation (window = 1). "
        "Execution: start = [sx, sy]@R, end = [sx + dx, sy + dy]@R",
        "proprio": "none (the arm returns to rest after each push)",
    }
    axis_note = "image (45-degree corner camera, after rotation): +sx = right corr(sx, col) = +0.99, +sy = down corr(sy, row) = +0.99"
    lang_dims = [
        ("sx", "move", "right", "left"),
        ("sy", "move", "below", "above"),
        ("dx", "move", "right", "left"),
        ("dy", "move", "down", "up"),
    ]
    lang_template = "place the stick {0} and {1}, then push the rope {2} and {3}"

    @property
    def root(self):
        return os.path.join(raw_dir(), "deformable", "rope")

    def render_action(self, av):
        return (
            f"put the stick {_w(av[0], 'right of', 'left of')} center, {_w(av[1], 'below', 'above')} center, "
            f"then push the rope {_w(av[2], 'right', 'left')} and {_w(av[3], 'down', 'up')}"
        )


class Granular(Deformable):
    name = "granular"
    units_note = Rope.units_note
    axis_note = "image (45-degree corner camera, after rotation): +sx = right corr(sx, col) = +0.98, +sy = down corr(sy, row) = +0.97"
    lang_dims = Rope.lang_dims
    lang_template = "place the board {0} and {1}, then push the pile {2} and {3}"

    @property
    def root(self):
        return os.path.join(raw_dir(), "deformable", "granular")

    def render_action(self, av):
        return (
            f"put the board {_w(av[0], 'right of', 'left of')} center, {_w(av[1], 'below', 'above')} center, "
            f"then push the pile {_w(av[2], 'right', 'left')} and {_w(av[3], 'down', 'up')}"
        )


ENVS = {c.name: c for c in [Metaworld(), PointMaze(), Wall(), PushT(), Rope(), Granular()]}


def net_windows(actions, proprio, n_frames, window):
    for u in range(0, n_frames - window):
        yield u, actions[u : u + window].sum(axis=0), proprio[u]


def run_stats(env, limit=None):
    nets, props, n_ep = [], [], 0
    units = env.units()[:limit] if limit else env.units()
    for unit in units:
        for _, actions, proprio, n_frames, _ in env.iter_trajs(unit, with_frames=False):
            n_ep += 1
            for _, net, pr in net_windows(actions, proprio, n_frames, env.window):
                nets.append(net)
                props.append(pr)
    nets, props = np.stack(nets), np.stack(props)
    n_tr = len(nets)

    spec = {
        "spec_version": 3,
        "dataset": env.name,
        "domain_prefix": f"[{env.name}] ",
        "source_jsonl": f"jsonl/traj/{env.name}.jsonl",
        "record_layout": "traj (one line per episode; n_records = number of stride-1 transitions)",
        "n_records": int(n_tr),
        "n_episodes": int(n_ep),
        "purpose": f"shared normalization space for text rendering and planning "
        f"(action chunk = {env.window}-step net action, all stride-1 window starts)",
        "text_normalized": True,
        "net_window": env.window,
        "action_dim": len(env.action_keys),
        "action_dims": env.action_keys,
        "units": env.units_note,
        "action_mean": R6(nets.mean(0)),
        "action_std": R6(nets.std(0)),
        "action_clip_lo": R6(np.percentile(nets, 1, axis=0)),
        "action_clip_hi": R6(np.percentile(nets, 99, axis=0)),
        "proprio_dim": len(env.proprio_keys),
        "proprio_dims": env.proprio_keys,
        "proprio_mean": R6(props.mean(0)),
        "proprio_std": R6(props.std(0)),
        "planning": {
            "sample_space": "z-score (action_mean/std) Gaussian samples, clipped in z space",
            "clip": "z-space clip bounds = (action_clip_lo/hi - mean)/std (p1/p99)",
            "render_value": "semantic/lang value = z + mean/std (= net/std, mean not subtracted); sign picks the direction word",
        },
        "render": {
            "fields": (
                {
                    "actions_semantic": "proprio clause (z-score) + '; ' + numerical action clause (net/std)",
                    "actions_semantic_no_proprio": "numerical action clause only",
                    "actions_lang6": "proprio clause + '; ' + lang6 action clause",
                    "actions_lang6_no_proprio": "lang6 action clause only",
                    "actions_lang4": "proprio clause + '; ' + lang4 action clause",
                    "actions_lang4_no_proprio": "lang4 action clause only",
                }
                if env.proprio_keys
                else {
                    "actions_semantic": "numerical action clause only (no proprio; identical to the no_proprio field)",
                    "actions_semantic_no_proprio": "numerical action clause only",
                    "actions_lang6": "lang6 action clause only (identical to no_proprio)",
                    "actions_lang6_no_proprio": "lang6 action clause only",
                    "actions_lang4": "lang4 action clause only (identical to no_proprio)",
                    "actions_lang4_no_proprio": "lang4 action clause only",
                }
            ),
            "numerical": "none (no key=value z-score rendering in this build)",
            "axis_grounding": env.axis_note,
            "note": "the [" + env.name + "] prefix is added by the training loader when domain_prefix is on. "
            "action_raw keeps the raw-unit net action and proprio",
        },
    }
    if isinstance(env, Deformable):
        spec["plan_space"] = env.plan_space_block()

    std = np.array(spec["action_std"])
    v = nets / std
    bins = {
        "spec_version": 1,
        "dataset": env.name,
        "source_jsonl": spec["source_jsonl"],
        "value_space": f"action_raw.net_action / action_std ({env.name}_v1.yaml); the same value as the semantic text (mean not subtracted)",
        "side_rule": "v >= 0 -> pos direction word, v < 0 -> neg (same as the semantic render)",
        "bin_rule": "bin = #(abs_edges <= |v|) (bisect_right on the round-6 edges); counts are re-tallied with the same rule",
        "edge_rule": "per direction, np.quantile of |v| (6-bin: [1/3, 2/3], 4-bin: [0.5]), round-6",
        "n_pairs": int(n_tr),
        "purpose": "training lang text and planning renders share the same bin edges and phrases",
        "builder": "data_preparation/build_sim_traj.py",
        "dims": [],
    }
    for i, (k, kind, pos_d, neg_d) in enumerate(env.lang_dims):
        x = v[:, i]
        an, ap = np.abs(x[x < 0]), np.abs(x[x >= 0])
        dim = {
            "key": k,
            "kind": kind,
            "pos_dir": pos_d,
            "neg_dir": neg_d,
            "zero_count": int((x == 0.0).sum()),
            "neg_count": int(len(an)),
            "pos_count": int(len(ap)),
        }
        for nb, qs in [(6, [1 / 3, 2 / 3]), (4, [0.5])]:
            en = [round(float(e), 6) for e in np.quantile(an, qs)]
            ep = [round(float(e), 6) for e in np.quantile(ap, qs)]
            cn = np.bincount([bisect_right(en, a) for a in an], minlength=len(en) + 1)
            cp = np.bincount([bisect_right(ep, a) for a in ap], minlength=len(ep) + 1)
            dim[f"bins{nb}"] = {
                "neg_abs_edges": en,
                "pos_abs_edges": ep,
                "neg_counts": [int(c) for c in cn],
                "pos_counts": [int(c) for c in cp],
                "neg_phrases": phrases(kind, neg_d, nb),
                "pos_phrases": phrases(kind, pos_d, nb),
            }
        bins["dims"].append(dim)

    os.makedirs(SPEC_DIR, exist_ok=True)
    os.makedirs(BINS_DIR, exist_ok=True)
    spec_path = os.path.join(SPEC_DIR, f"{env.name}_v1.yaml")
    bins_path = os.path.join(BINS_DIR, f"{env.name}_v1.yaml")
    with open(spec_path, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True, width=110)
    with open(bins_path, "w") as f:
        yaml.safe_dump(bins, f, sort_keys=False, allow_unicode=True, width=110)
    print(f"== {env.name} episodes={n_ep} transitions={n_tr} -> {spec_path} + {bins_path}")
    for i, k in enumerate(env.action_keys):
        print(
            f"  net {k:6s} mean={spec['action_mean'][i]:+.4f} std={spec['action_std'][i]:.4f}"
            f" clip=[{spec['action_clip_lo'][i]:+.4f}, {spec['action_clip_hi'][i]:+.4f}]"
        )


@functools.lru_cache(maxsize=8)
def load_spec(env_name):
    with open(os.path.join(SPEC_DIR, f"{env_name}_v1.yaml")) as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=8)
def load_bins(env_name):
    with open(os.path.join(BINS_DIR, f"{env_name}_v1.yaml")) as f:
        return yaml.safe_load(f)


def build_unit(env_name, unit, skip_frames=False, root=None):
    global ROOT
    ROOT = root
    env = ENVS[env_name]
    spec, bins = load_spec(env_name), load_bins(env_name)
    as_ = np.array(spec["action_std"])
    pm, ps = np.array(spec["proprio_mean"]), np.array(spec["proprio_std"])
    frames_root = os.path.join(data_root(root), env_name, "frames")
    records = []
    counts = {nb: [{"neg": [0] * (nb // 2), "pos": [0] * (nb // 2)} for _ in env.lang_dims] for nb in (6, 4)}
    for traj, actions, proprio, n_frames, get_frames in env.iter_trajs(unit, with_frames=not skip_frames):
        tdir = os.path.join(frames_root, traj)
        paths = [os.path.join(tdir, f"{t:06d}.jpg") for t in range(n_frames)]
        if skip_frames:
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                raise RuntimeError(f"--skip-frames but missing {missing[0]} (+{len(missing) - 1} more)")
        else:
            imgs = get_frames(list(range(n_frames)))
            os.makedirs(tdir, exist_ok=True)
            for t in range(n_frames):
                save_jpeg(imgs[t], paths[t])
        sem, sem_np, l6, l6_np, l4, l4_np = [], [], [], [], [], []
        raw_net, raw_pr = [], []
        for _, net, pr in net_windows(actions, proprio, n_frames, env.window):
            pz = (pr - pm) / ps
            av = net / as_
            head = env.render_proprio(pz)
            sem_a = env.render_action(av)
            clause = {}
            for nb in (6, 4):
                parts = []
                for j, (dim, x) in enumerate(zip(bins["dims"], av)):
                    b = dim[f"bins{nb}"]
                    side = "pos" if x >= 0 else "neg"
                    idx = bisect_right(b[f"{side}_abs_edges"], abs(x))
                    counts[nb][j][side][idx] += 1
                    parts.append(b[f"{side}_phrases"][idx])
                clause[nb] = env.lang_template.format(*parts)
            sem.append(f"{head}; {sem_a}" if head else sem_a)
            sem_np.append(sem_a)
            l6.append(f"{head}; {clause[6]}" if head else clause[6])
            l6_np.append(clause[6])
            l4.append(f"{head}; {clause[4]}" if head else clause[4])
            l4_np.append(clause[4])
            raw_net.append(R6(net))
            raw_pr.append(R6(pr))
        records.append(
            {
                "dataset": env_name,
                "id": f"{env_name}_traj/{traj}",
                "frames": paths,
                "actions_semantic": sem,
                "actions_semantic_no_proprio": sem_np,
                "actions_lang6": l6,
                "actions_lang6_no_proprio": l6_np,
                "actions_lang4": l4,
                "actions_lang4_no_proprio": l4_np,
                "action_raw": {"text_source": "traj_v1", "net_action": raw_net, "proprio": raw_pr},
            }
        )
    return records, counts


def run_build(env, workers, limit=None, skip_frames=False, out_suffix=""):
    from concurrent.futures import ProcessPoolExecutor

    spec, bins = load_spec(env.name), load_bins(env.name)
    units = env.units()[:limit] if limit else env.units()
    out_dir = os.path.join(data_root(ROOT), "jsonl", "traj")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{env.name}{out_suffix}.jsonl")
    n_tr = n_ep = 0
    total = {nb: [{"neg": [0] * (nb // 2), "pos": [0] * (nb // 2)} for _ in env.lang_dims] for nb in (6, 4)}
    with ProcessPoolExecutor(max_workers=workers) as ex, open(out_path + ".tmp", "w") as f:
        for records, counts in ex.map(
            build_unit, [env.name] * len(units), units, [skip_frames] * len(units), [ROOT] * len(units)
        ):
            for r in records:
                f.write(json.dumps(r) + "\n")
                n_tr += len(r["actions_semantic"])
            n_ep += len(records)
            for nb in (6, 4):
                for j in range(len(env.lang_dims)):
                    for side in ("neg", "pos"):
                        total[nb][j][side] = [a + b for a, b in zip(total[nb][j][side], counts[nb][j][side])]
            print(f"\r{env.name}: {n_ep} episodes / {n_tr}/{spec['n_records']} transitions", end="", flush=True)
    print()
    if limit is None:
        if n_tr != spec["n_records"] or n_ep != spec["n_episodes"]:
            raise RuntimeError(f"{env.name}: built {n_ep} ep/{n_tr} tr != spec {spec['n_episodes']}/{spec['n_records']}")
        for nb in (6, 4):
            for j, dim in enumerate(bins["dims"]):
                for side in ("neg", "pos"):
                    want = dim[f"bins{nb}"][f"{side}_counts"]
                    got = total[nb][j][side]
                    if got != want:
                        raise RuntimeError(f"{env.name}/{dim['key']} bins{nb} {side} counts {got} != yaml {want}")
    os.replace(out_path + ".tmp", out_path)
    print(f"== {env.name}: {n_ep} episodes, {n_tr} transitions -> {out_path}")


def main():
    global ROOT, SPEC_DIR, BINS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=sorted(ENVS))
    add_root_arg(ap)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None, help="work units (smoke); skips the total/bin-count checks")
    ap.add_argument("--skip-frames", action="store_true", help="frames already exist: verify only")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--spec-dir", default=SPEC_DIR)
    ap.add_argument("--bins-dir", default=BINS_DIR)
    args = ap.parse_args()
    ROOT, SPEC_DIR, BINS_DIR = args.root, args.spec_dir, args.bins_dir
    env = ENVS[args.env]
    if args.stats_only:
        run_stats(env, limit=args.limit)
    else:
        run_build(env, args.workers, limit=args.limit, skip_frames=args.skip_frames, out_suffix=args.out_suffix)


if __name__ == "__main__":
    main()
