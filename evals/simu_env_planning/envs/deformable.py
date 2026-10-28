"""Rope / granular deformable planning envs (port of DINO-WM's FlexEnvWrapper).

1 action = 1 push macro [x_start, y_start, x_end, y_end] in pybullet plane coordinates (the text space is the
screen-aligned [sx, sy, dx, dy]; run_planning converts before stepping). Needs the compiled PyFleX runtime at
$PYFLEX_HOME (a directory holding PyFleX/bindings/build and nvgl/10_nvidia.json) plus pybullet and bs4.
"""

import math
import os
import sys

PYFLEX_HOME = os.environ.get("PYFLEX_HOME", os.path.expanduser("~/pyflex"))
os.environ.setdefault("PYFLEXROOT", f"{PYFLEX_HOME}/PyFleX")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", f"{PYFLEX_HOME}/nvgl/10_nvidia.json")
os.environ.setdefault("EGL_GPU", "0")
_build = f"{os.environ['PYFLEXROOT']}/bindings/build"
if os.path.isdir(_build) and _build not in sys.path:
    sys.path.append(_build)
_flags = sys.getdlopenflags()
sys.setdlopenflags(_flags | os.RTLD_DEEPBIND)
try:
    import pyflex  # noqa: F401  (flex_env below reuses this module from sys.modules)
finally:
    sys.setdlopenflags(_flags)

import gym
import numpy as np
import torch
import yaml

from evals.simu_env_planning.envs.deformable_env.sim_env.flex_env import FlexEnv
from evals.simu_env_planning.envs.wrappers.pixels import PixelWrapper
from evals.simu_env_planning.envs.wrappers.tensor import TensorWrapper
from evals.simu_env_planning.envs.wrappers.time_limit import TimeLimit

CONF_DIR = os.path.join(os.path.dirname(__file__), "deformable_env", "conf")

# Success = final CD below the CD of a rigid |delta| = 0.75 world-unit shift (~21 px at 224px,
# ~10% of the table) of that object's own goal shape. CD scales with shape, not just with
# displacement, so the same tolerance is a different CD on the rope and on the granular pile;
# measured over the fixed 24/12 goal states, 16 shift directions averaged.
SUCCESS_CD = {"rope": 0.900, "granular": 0.414}


def chamfer_distance(x, y):
    # dino_wm metric: symmetric mean-min L2 over the first 3 dims
    x = np.asarray(x, dtype=np.float64)[:, :3]
    y = np.asarray(y, dtype=np.float64)[:, :3]
    d = np.linalg.norm(x[:, None] - y[None, :], axis=2)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean())


class DeformEnv(FlexEnv):
    """gym-style step/reset over FlexEnv's push primitive."""

    def __init__(self, object_name):
        with open(os.path.join(CONF_DIR, f"{object_name}.yaml")) as f:
            config = yaml.safe_load(f)
        super().__init__(config=config)

    def _obs(self):
        img = self.get_one_view_img()  # (H, W, 5): color + depth, color channel-flipped in the dataset
        visual = np.ascontiguousarray(img[..., :3][..., ::-1]).astype(np.uint8)
        return {"visual": visual, "proprio": np.zeros(1, dtype=np.float32)}

    def _info(self):
        return {"state": self.get_positions().reshape(-1, 4).copy(), "proprio": np.zeros(1, dtype=np.float32)}

    def gym_reset(self):
        self.imgs_list, self.particle_pos_list, self.eef_states_list = [], [], []
        super().reset(save_data=False)
        return self._obs(), self._info()

    def gym_step(self, action):
        ret = super().step(np.asarray(action, dtype=np.float64), save_data=False, data=([], [], []))
        info = self._info()
        if ret is None:  # FlexEnv NaN guard tripped mid-push
            info["sim_exploded"] = True
        return self._obs(), 0.0, False, info


class DeformableEnvWrapper(gym.Wrapper):
    def __init__(self, env, cfg=None, success_cd=None):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.action_dim = 4
        self.success_cd = SUCCESS_CD[env.obj] if success_cd is None else float(success_cd)
        # dino_wm evaluates every instance in a fresh env, so its "reset state" is the untouched object.
        # FlexEnv.reset() does NOT rebuild the scene (robot reset + 200 settle steps only): with one env
        # per run the live reset state drifts to wherever the previous episode left the object, and goals
        # derived from it become history-dependent (found 2026-08-21, deform_cause_analysis). Capture the
        # untouched particle state once, right after scene creation, and derive every goal from it.
        _, info = self.env.gym_reset()
        self._canonical = np.asarray(info["state"], dtype=np.float32).copy()
        self.last_goal_params = None

    def eval_state(self, goal_state, cur_state):
        cd = chamfer_distance(cur_state, goal_state)
        # upstream dino_wm reports CD only (success_threshold = 0, so success is always False);
        # CD stays the headline metric and a thresholded success is added on top (see SUCCESS_CD).
        return {"success": bool(cd < self.success_cd), "state_dist": cd}

    def update_env(self, env_info):
        pass

    def canonical_state(self):
        """Particle state of the untouched object (== dataset frame 0; fresh-scene CD 0.000)."""
        return self._canonical.copy()

    def sample_random_init_goal_states(self, seed):
        """dino_wm FlexEnvWrapper port: goal = scaled/rotated/shifted copy of the (canonical) reset state.
        Only np.random (seeded here) is consumed, so the goal is a pure function of `seed`."""
        self.env.seed(seed)
        state = torch.tensor(self._canonical)

        def transfer_state(state, scale, theta, delta):
            assert state.shape[1] == 4
            state = state.clone()
            state[:, 0] *= scale
            state[:, 2] *= scale
            theta = math.radians(theta)
            rotation_matrix_y = torch.tensor(
                [
                    [math.cos(theta), 0, math.sin(theta)],
                    [0, 1, 0],
                    [-math.sin(theta), 0, math.cos(theta)],
                ]
            )
            state[:, :3] = torch.matmul(state[:, :3], rotation_matrix_y.T.to(state.dtype))
            state[:, 0] += delta * np.random.choice([-1, 1])
            state[:, 2] += delta * np.random.choice([-1, 1])
            return state

        if self.env.obj == "granular":
            goal_scale, goal_theta, goal_delta = np.random.uniform(0.6, 0.9), 0, np.random.uniform(-1, 1)
        elif self.env.obj == "rope":
            goal_scale, goal_theta, goal_delta = 1, np.random.uniform(0, 90), np.random.uniform(1, -1)
        else:
            raise ValueError(f"unknown deformable object {self.env.obj!r}")
        goal_state = transfer_state(state, goal_scale, goal_theta, goal_delta)
        self.last_goal_params = {"scale": float(goal_scale), "theta_deg": float(goal_theta), "delta": float(goal_delta)}
        return state.numpy(), goal_state.numpy()

    def prepare(self, seed, init_state, env_info=None):
        self.env.seed(seed)
        self.env.gym_reset()
        init_state = np.asarray(init_state, dtype=np.float32)
        self.env.set_states(init_state)
        pyflex.set_velocities(np.zeros(init_state.shape[0] * 3, dtype=np.float32))  # the loaded state starts at rest
        return self.env._obs(), self.env._info()

    def reset(self, seed=0, **kwargs):
        self.env.seed(seed)
        return self.env.gym_reset()

    def step(self, action):
        obs, reward, done, info = self.env.gym_step(action)
        return obs, reward, None, done, info

    def render(self, *args, **kwargs):
        kwargs.pop("mode", None)
        w, h = kwargs.pop("width", None), kwargs.pop("height", None)
        img = self.env._obs()["visual"]
        assert w in (None, img.shape[1]) and h in (None, img.shape[0]), f"fixed render size {img.shape[:2]}"
        return img


def make_env(cfg):
    """Same wrapper chain as init.make_env builds for wall/maze, over the PyFleX env."""
    gym.logger.set_level(40)
    if not cfg.task_specification.task.startswith("deform-"):
        raise ValueError("Unknown task:", cfg.task_specification.task)
    obj = cfg.task_specification.task.split("-", 1)[1]
    env = DeformEnv(obj)
    env = DeformableEnvWrapper(env, cfg, success_cd=cfg.task_specification.get("success_cd"))
    env = TimeLimit(env, max_episode_steps=cfg.task_specification.max_episode_steps)
    env.max_episode_steps = env._max_episode_steps
    env = TensorWrapper(env)
    env = PixelWrapper(cfg, env)
    return env
