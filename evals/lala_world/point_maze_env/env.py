"""Modern-mujoco port of the PointMaze planning env.

The bundled maze env (evals/simu_env_planning/envs/pointmaze_env/maze_model.py +
pointmaze_gym_wrap.py) requires d4rl + mujoco-py + MuJoCo 2.1 binaries + gym 0.23,
none of which install on this machine (Python 3.12). This module reimplements the
exact surface the planning eval uses on the maintained `mujoco` pip package:

- identical generated MJCF (same point_maze() builder, U_MAZE spec),
- identical step semantics (action clip [-1,1], per-step qvel clip +-5, ctrl held
  over frame_skip=5 substeps of dt=0.01),
- mujoco_py's rendering convention: free camera at azimuth 90 / elevation -90,
  and the raw bottom-up GL buffer (modern Renderer output flipped vertically).

The d4rl OfflineEnv mixin is dropped: its kwargs (ref_min_score, dataset_url, ...)
were stored but never used by the eval path.

Parity against sim_raw/point_maze (state replay + pixel diff + camera calibration):
python -m evals.lala_world.point_maze_env.validate
"""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import gym
import mujoco
import numpy as np

from evals.simu_env_planning.envs.pointmaze_env.dynamic_mjc import MJCModel
from evals.simu_env_planning.envs.wrappers.pixels import PixelWrapper
from evals.simu_env_planning.envs.wrappers.tensor import TensorWrapper
from evals.simu_env_planning.envs.wrappers.time_limit import TimeLimit

WALL = 10
EMPTY = 11
GOAL = 12

U_MAZE = "#####\\" + "#GOO#\\" + "###O#\\" + "#OOO#\\" + "#####"

OFF_TARGET = np.array([10, 10])

# maze_model.STATE_RANGES (6x2: pos, vel, target) — used by MazeEnv.reset when no init state is set
ENV_STATE_RANGES = np.array(
    [
        [0.39318362, 3.2198412],
        [0.62660956, 3.2187355],
        [-5.2262554, 5.2262554],
        [-5.2262554, 5.2262554],
        [0.90001136, 3.0999563],
        [0.9000267, 3.0999668],
    ]
)

# Free-camera pose matching the mujoco_py offscreen context that rendered the
# dataset (azimuth 90 / elevation -90 set in prepare_for_render; lookat/distance
# were mujoco_py context defaults). Calibrated against sim_raw/point_maze frames
# by validate_point_maze_port.py --calibrate.
CAM_LOOKAT = (3.0, 3.0, 0.0)
CAM_DISTANCE = 9.6904
CAM_AZIMUTH = 90
CAM_ELEVATION = -90


def parse_maze(maze_str):
    lines = maze_str.strip().split("\\")
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == "#":
                maze_arr[w][h] = WALL
            elif tile == "G":
                maze_arr[w][h] = GOAL
            elif tile == " " or tile == "O" or tile == "0":
                maze_arr[w][h] = EMPTY
            else:
                raise ValueError("Unknown tile type: %s" % tile)
    return maze_arr


def point_maze(maze_str):
    # verbatim copy of maze_model.point_maze so the MJCF stays byte-identical
    maze_arr = parse_maze(maze_str)

    mjcmodel = MJCModel("point_maze")
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(damping=1, limited="false")
    default.geom(friction=".5 .1 .1", density="1000", margin="0.002", condim="1", contype="2", conaffinity="1")

    asset = mjcmodel.root.asset()
    asset.texture(
        type="2d", name="groundplane", builtin="checker", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", width=100, height=100
    )
    asset.texture(
        name="skybox",
        type="skybox",
        builtin="gradient",
        rgb1=".4 .6 .8",
        rgb2="0 0 0",
        width="800",
        height="800",
        mark="random",
        markrgb="1 1 1",
    )
    asset.material(name="groundplane", texture="groundplane", texrepeat="20 20")
    asset.material(name="wall", rgba=".7 .5 .3 1")
    asset.material(name="target", rgba=".6 .3 .3 1")

    visual = mjcmodel.root.visual()
    visual.headlight(ambient=".4 .4 .4", diffuse=".8 .8 .8", specular="0.1 0.1 0.1")
    visual.map(znear=0.01)
    visual.quality(shadowsize=2048)

    worldbody = mjcmodel.root.worldbody()
    worldbody.geom(
        name="ground",
        size="40 40 0.25",
        pos="0 0 -0.1",
        type="plane",
        contype=1,
        conaffinity=0,
        material="groundplane",
    )

    particle = worldbody.body(name="particle", pos=[1.2, 1.2, 0])
    particle.geom(name="particle_geom", type="sphere", size=0.1, rgba="0.0 0.0 1.0 0.0", contype=1)
    particle.site(name="particle_site", pos=[0.0, 0.0, 0], size=0.2, rgba="0.3 0.6 0.3 1")
    particle.joint(name="ball_x", type="slide", pos=[0, 0, 0], axis=[1, 0, 0])
    particle.joint(name="ball_y", type="slide", pos=[0, 0, 0], axis=[0, 1, 0])

    worldbody.site(name="target_site", pos=[0.0, 0.0, 0], size=0.2, material="target")

    width, height = maze_arr.shape
    for w in range(width):
        for h in range(height):
            if maze_arr[w, h] == WALL:
                worldbody.geom(
                    conaffinity=1,
                    type="box",
                    name="wall_%d_%d" % (w, h),
                    material="wall",
                    pos=[w + 1.0, h + 1.0, 0],
                    size=[0.5, 0.5, 0.2],
                )

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)

    return mjcmodel


class MazeEnv(gym.Env):
    def __init__(
        self,
        maze_spec=U_MAZE,
        reward_type="dense",
        reset_target=True,
        return_value="state",  # 'obs' or 'state'
        with_target=False,
        **kwargs,  # swallows the d4rl OfflineEnv kwargs (ref_min_score, dataset_url, ...)
    ):
        self.with_target = with_target
        self.reset_target = reset_target
        self.str_maze_spec = maze_spec
        self.maze_arr = parse_maze(maze_spec)
        self.reward_type = reward_type
        self.reset_locations = list(zip(*np.where(self.maze_arr == EMPTY)))
        self.reset_locations.sort()

        self._target = np.array([0.0, 0.0])
        self.return_value = return_value

        model = point_maze(maze_spec)
        with model.asfile() as f:
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 5
        mujoco.mj_forward(self.model, self.data)
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self._target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        if self.return_value == "obs":
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self._renderer = None
        self._renderer_size = None
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._cam.fixedcamid = -1
        self._cam.lookat[:] = CAM_LOOKAT
        self._cam.distance = CAM_DISTANCE
        self._cam.azimuth = CAM_AZIMUTH
        self._cam.elevation = CAM_ELEVATION

        self.goal_locations = list(zip(*np.where(self.maze_arr == GOAL)))
        if len(self.goal_locations) == 1:
            self.set_target(self.goal_locations[0])
        elif len(self.goal_locations) > 1:
            raise ValueError("More than 1 goal specified!")
        else:
            self.set_target(np.array(self.reset_locations[0]).astype(np.float32))
        self.empty_and_goal_locations = self.reset_locations + self.goal_locations

        self.seed()
        self.reset_to_state = None

    def set_state(self, qpos, qvel):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def do_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()
        self.do_simulation(action, self.frame_skip)
        self.set_marker()
        state = self._get_obs()["proprio"]
        state = np.concatenate([state, self._target]) if self.with_target else state
        if self.reward_type == "sparse":
            reward = 1.0 if np.linalg.norm(state[0:2] - self._target) <= 0.5 else 0.0
        elif self.reward_type == "dense":
            reward = np.exp(-np.linalg.norm(state[0:2] - self._target))
        else:
            raise ValueError("Unknown reward type %s" % self.reward_type)
        done = False
        if self.return_value == "obs":
            visual = self._render_frame()
            ob = {
                "visual": visual,
                "proprio": state,
            }
        else:
            ob = {
                "visual": state,
                "proprio": state,
            }

        info = {}
        info["state"] = state
        info["target"] = self._target
        info["obs"] = ob if self.return_value == "obs" else None
        info["pos_agent"] = state[:2]
        return ob, reward, done, info

    def _get_obs(self):
        obs = {
            "visual": np.concatenate([self.data.qpos, self.data.qvel]).ravel().astype(np.float32),
            "proprio": np.concatenate([self.data.qpos, self.data.qvel]).ravel().astype(np.float32),
        }
        return obs

    def get_target(self):
        return self._target

    def set_target(self, target_location=None):
        if target_location is None:
            idx = self.np_random.choice(len(self.empty_and_goal_locations))
            reset_location = np.array(self.empty_and_goal_locations[idx]).astype(np.float32)
            target_location = reset_location + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        self._target = target_location

    def set_marker(self):
        if not self.with_target:
            self.set_target(OFF_TARGET)
        self.data.site_xpos[self._target_site_id] = np.array([self._target[0] + 1, self._target[1] + 1, 0.0])

    def clip_velocity(self):
        qvel = np.clip(self.data.qvel, -5.0, 5.0)
        self.set_state(self.data.qpos, qvel)

    def set_init_state(self, init_state):
        self.reset_to_state = init_state
        target = init_state[-2:] if init_state is not None else None
        self.set_target(target)
        self.set_marker()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.set_init_state(self.reset_to_state)
        state = self.reset_to_state
        if state is None:
            rs = self.random_state
            state = np.array(
                [
                    rs.uniform(low=ENV_STATE_RANGES[0][0], high=ENV_STATE_RANGES[0][1]),
                    rs.uniform(low=ENV_STATE_RANGES[1][0], high=ENV_STATE_RANGES[1][1]),
                    rs.uniform(low=ENV_STATE_RANGES[2][0], high=ENV_STATE_RANGES[2][1]),
                    rs.uniform(low=ENV_STATE_RANGES[3][0], high=ENV_STATE_RANGES[3][1]),
                    rs.uniform(low=ENV_STATE_RANGES[4][0], high=ENV_STATE_RANGES[4][1]),
                    rs.uniform(low=ENV_STATE_RANGES[5][0], high=ENV_STATE_RANGES[5][1]),
                ]
            )
        qpos, qvel = state[:2], state[2:4]
        self.set_state(qpos, qvel)
        self.set_marker()
        obs = self._get_obs()
        if self.return_value == "obs":
            visual = self._render_frame()
            obs["visual"] = visual
        state = state[:4] if self.with_target else state
        return obs, state

    def _render_frame(self, width=224, height=224):
        if self._renderer is None or self._renderer_size != (width, height):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderer_size = (width, height)
        self._renderer.update_scene(self.data, camera=self._cam)
        img = self._renderer.render()
        # mujoco_py returned the raw bottom-up GL buffer; the dataset follows it
        return img[::-1].copy()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        self.random_state = np.random.RandomState(seed)

    def prepare_for_render(self):
        self.return_value = "obs"
        init_state = np.array([1.0856, 1.9746, 0.0098, 0.0217])
        self.set_state(init_state[:2], init_state[2:])
        img = self._render_frame()
        self._cam.azimuth = CAM_AZIMUTH
        self._cam.elevation = CAM_ELEVATION
        img1 = self._render_frame()
        return img, img1


# pointmaze_gym_wrap.STATE_RANGES (4x2) — used for random init/goal sampling
STATE_RANGES = np.array(
    [
        [0.39318362, 3.2198412],
        [0.62660956, 3.2187355],
        [-5.2262554, 5.2262554],
        [-5.2262554, 5.2262554],
    ]
)


class PointMazeWrapper(gym.Wrapper):
    def __init__(self, env, cfg=None):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.action_dim = self.env.action_space.shape[0]

    def sample_random_init_goal_states(self, seed):
        rs = np.random.RandomState(seed)

        def generate_state():
            valid = False
            while not valid:
                x = rs.uniform(0.5, 3.1)
                y = rs.uniform(0.5, 3.1)
                valid = ((0.5 <= x <= 1.1 or 2.5 <= x <= 3.1) and (0.5 <= y <= 3.1)) or (
                    (1.1 < x < 2.5) and (2.5 <= y <= 3.1)
                )
            state = np.array(
                [
                    x,
                    y,
                    rs.uniform(low=STATE_RANGES[2][0], high=STATE_RANGES[2][1]),
                    rs.uniform(low=STATE_RANGES[3][0], high=STATE_RANGES[3][1]),
                ]
            )
            return state

        init_state = generate_state()
        goal_state = generate_state()
        return init_state, goal_state

    def update_env(self, env_info):
        pass

    def eval_state(self, goal_state, cur_state):
        success = np.linalg.norm(goal_state[:2] - cur_state[:2]) < 0.5
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            "success": success,
            "state_dist": state_dist,
        }

    def prepare(self, seed, init_state, env_info=None):
        self.env.prepare_for_render()
        self.env.seed(seed)
        self.env.set_init_state(init_state)
        obs, info = self.reset()
        return obs, info

    def reset(self, **kwargs):
        obs, state = self.env.reset()
        info = {"state": state, "proprio": obs["proprio"]}
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["proprio"] = info["state"]
        return obs, reward, None, done, info

    def render(self, *args, **kwargs):
        # gym 0.23 turned render(mode=...) into a positional arg the original wrapper
        # dropped via *args; gym 0.26 forwards it as a kwarg, so drop it here
        kwargs.pop("mode", None)
        obs = self.env._render_frame(**kwargs)
        return obs


def make_env(cfg, env_cls=None):
    """Same wrapper chain as evals.simu_env_planning.envs.init.make_env builds for
    maze-* tasks, with the modern-mujoco MazeEnv swapped in."""
    gym.logger.set_level(40)
    if cfg.task_specification.goal_source == "random_state":
        cfg.task_specification.max_episode_steps = cfg.frameskip * cfg.task_specification.goal_H
    if not cfg.task_specification.task.startswith("maze-"):
        raise ValueError("Unknown task:", cfg.task_specification.task)
    # kwargs kept from pointmaze_gym_wrap.make_env; the d4rl-specific ones are ignored by the port
    kwargs = {
        "reward_type": "sparse",
        "reset_target": False,
        "ref_min_score": 23.85,
        "ref_max_score": 161.86,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5",
    }
    env = MazeEnv(**kwargs)
    env = PointMazeWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=cfg.task_specification.max_episode_steps)
    env.max_episode_steps = env._max_episode_steps
    env = TensorWrapper(env)
    if cfg.task_specification.get("obs", "state") in ["rgb", "rgb_state"]:
        env = PixelWrapper(cfg, env)
    try:
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except AttributeError:
        cfg.obs_shape = {cfg.task_specification.get("obs", "state"): env.observation_space.shape}
    if cfg.task_specification.get("obs", "state") == "rgb_state":
        cfg.obs_shape = {"state": [4], "rgb": cfg.obs_shape["rgb_state"]}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.meta.seed_steps = max(1000, 5 * cfg.episode_length)
    return env
