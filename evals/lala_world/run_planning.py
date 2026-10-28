"""MPC (CEM) planning with LaLa-World on the simulation benchmarks: wall, point_maze, pusht, metaworld, rope, granular.

  python -m evals.lala_world.run_planning --ckpt <ckpt> --env wall --episodes 96
Candidate action chunks are rendered to the training text format, rolled out in latent space and scored by the
L2 distance to the encoded goal. Results: <out>/<env>/<tag>/result.json (+ init|goal|final strips and GIFs).
"""

import argparse
import functools
import json
import os
import pickle
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from app.lala_world.data import is_val_id
from evals.lala_world.backends import add_backend_args, build_backend, run_tag
from evals.lala_world.latent_decode import load_decoder, save_pred_decode_strip
from evals.lala_world.latent_pca import save_pred_pca_strip
from evals.lala_world.paths import DEFAULT_OUT, SIM_RAW, TARGETS_DIR, TRAJ_JSONL
from evals.lala_world.sim_spec import DEFORM_R, ProprioDynamics, load_spec
from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.objectives import ReprTargetDistMPCObjective
from evals.simu_env_planning.planning.planner import CEMPlanner

ENV_CONF = {
    # wall executes in image units (the rendered frames follow pos + 2 * action), hence exec_scale 1.0
    "wall": {
        "task": "wall-base",
        "goal_source": "random_state",
        "exec_scale": 1.0,
        "env": {"with_target": True, "with_velocity": True, "freeze_rand_vec": False},
    },
    "point_maze": {
        "task": "maze-base",
        "goal_source": "random_state",
        "exec_scale": 1.0,
        "env": {"with_target": True, "with_velocity": True, "freeze_rand_vec": False},
    },
    "pusht": {
        "task": "pusht-base",
        "goal_source": "dset",
        "exec_scale": 0.01,
        "env": {"with_target": True, "with_velocity": True, "freeze_rand_vec": False},
    },
    # goal = final frame of a scripted-expert rollout of the same task variant (--mw-task)
    "metaworld": {
        "task": "mw-reach",
        "goal_source": "expert",
        "exec_scale": 1.0,
        "env": {"with_target": False, "with_velocity": False, "freeze_rand_vec": False},
    },
    # rope / granular (PyFleX): 1 chunk = 1 push; init = val-episode particle state, goal = transformed reset state
    "rope": {
        "task": "deform-rope",
        "goal_source": "random_state",
        "exec_scale": 1.0,
        "frameskip": 1,
        "success_cd": 0.900,
        "env": {"with_target": False, "with_velocity": False, "freeze_rand_vec": False},
    },
    "granular": {
        "task": "deform-granular",
        "goal_source": "random_state",
        "exec_scale": 1.0,
        "frameskip": 1,
        "success_cd": 0.414,
        "env": {"with_target": False, "with_velocity": False, "freeze_rand_vec": False},
    },
}
FRAMESKIP = 5
DEFORM_ENVS = ("rope", "granular")
MW_EXPERT_MAX_STEPS = 200
PUSHT_VAL = f"{SIM_RAW}/pusht_noise/val"
DEFORM_RAW = f"{SIM_RAW}/deformable"
SHIPPED_TARGETS = {"rope": "rope_seed1_ep24.npz", "granular": "granular_seed1_ep12.npz"}


def verify_render(spec, no_proprio, text_source, n=200):
    """The planner's text must match the training JSONL byte for byte."""
    field = {"semantic": "actions_semantic", "lang6": "actions_lang6", "lang4": "actions_lang4"}[text_source]
    if no_proprio:
        field += "_no_proprio"
    render = spec.renderer(text_source)
    bad = done = 0
    with open(f"{TRAJ_JSONL}/{spec.name}.jsonl") as f:
        for line in f:
            if done >= n:
                break
            r = json.loads(line)
            raw = r["action_raw"]
            for u, ref in enumerate(r[field]):
                if done >= n:
                    break
                z = (np.array(raw["net_action"][u]) - spec.action_mean) / spec.action_std
                got = render(None if no_proprio else raw["proprio"][u], z, prefix=False)
                if got != ref:
                    bad += 1
                    if bad <= 3:
                        print(f"RENDER MISMATCH\n jsonl: {ref}\n  ours: {got}")
                done += 1
    if bad > n // 100:
        raise AssertionError(f"render parity failed: {bad}/{done} mismatches")
    print(f"render parity ok: {done - bad}/{done} byte-identical")


def build_env(env_name, goal_H, num_samples, success_cd=None):
    ec = ENV_CONF[env_name]
    cfg = OmegaConf.create(
        {
            "task_specification": {
                "task": ec["task"],
                "obs": "rgb_state",
                "obs_concat_channels": False,
                "goal_source": ec["goal_source"],
                "succ_def": "simu",
                "done_at_succ": False,
                "goal_H": goal_H,
                "num_frames": 1,
                "num_proprios": 1,
                "img_size": 224,
                "multitask": False,
                "env": ec["env"],
                "success_cd": success_cd if success_cd is not None else ec.get("success_cd"),
            },
            "frameskip": ec.get("frameskip", FRAMESKIP),
            "planner": {"horizon": goal_H, "num_samples": num_samples},
            "meta": {"seed": 0},
        }
    )
    if env_name == "metaworld":
        cfg.task_specification.max_episode_steps = MW_EXPERT_MAX_STEPS
    if env_name == "point_maze":
        from evals.lala_world.point_maze_env import make_env as maze_make_env

        return maze_make_env(cfg), cfg
    if env_name in DEFORM_ENVS:
        from evals.simu_env_planning.envs.deformable import make_env as deform_make_env

        cfg.task_specification.max_episode_steps = cfg.frameskip * goal_H
        return deform_make_env(cfg), cfg
    env = make_env(cfg)
    if env_name == "metaworld":
        env.unwrapped.seeded_rand_vec = True
    return env, cfg


def obs_to_img(obs):
    return obs[-1].permute(1, 2, 0).numpy().astype(np.uint8)


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def banner_frame(arr, text):
    im = Image.fromarray(arr)
    out = Image.new("RGB", (im.width, im.height + 22), "black")
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)
    for size in (14, 12, 10, 8):
        font = _font(size)
        if d.textlength(text, font=font) <= im.width - 6:
            break
    d.text((3, im.height + 3), text, fill="white", font=font)
    return out


def save_this_ep(n, ep, ep_offset):
    """n < 0 = every episode, else the shard's own first n episodes."""
    return n < 0 or ep - ep_offset < n


def save_pred_strip(wm, result, cur_img, goal_img, path, label):
    """[current | imagined +1.. | goal] through the backend's own decoder (the planner's final mean-plan unroll)."""
    enc = result.predicted_best_encs_over_iterations[-1]
    if enc.dim() != 4:
        return
    pred = (wm.decode(enc[:, 0]).permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    side = pred.shape[1]
    cur, goal = (np.asarray(Image.fromarray(im).resize((side, side), Image.BILINEAR)) for im in (cur_img, goal_img))
    banner_frame(np.concatenate([cur, *pred, goal], axis=1), label).save(path)


def proprio_of(info):
    p = info["proprio"]
    p = p[-1] if p.ndim > 1 else p
    return np.asarray(p, dtype=np.float64)


def load_pusht_val():
    st = torch.load(f"{PUSHT_VAL}/states.pth", map_location="cpu").double().numpy()
    ac = torch.load(f"{PUSHT_VAL}/rel_actions.pth", map_location="cpu").double().numpy()
    vel = torch.load(f"{PUSHT_VAL}/velocities.pth", map_location="cpu").double().numpy()
    with open(f"{PUSHT_VAL}/seq_lengths.pkl", "rb") as f:
        sl = [int(x) for x in pickle.load(f)]
    return st, ac, vel, sl


@functools.lru_cache(maxsize=1)
def load_deform_val(env_name):
    st = torch.load(f"{DEFORM_RAW}/{env_name}/states.pth", map_location="cpu")
    val = [i for i in range(st.shape[0]) if is_val_id(f"{env_name}_traj/ep{i:04d}")]
    return st[val].clone().numpy()


def episode_seed(seed, ep):
    return (seed * seed + ep * seed) % (2**32 - 2)


def default_targets(args):
    """rope / granular: the shipped fixed problem set when it covers the run, else a per-run file."""
    shipped = os.path.join(TARGETS_DIR, SHIPPED_TARGETS[args.env])
    if args.seed == 1 and os.path.exists(shipped):
        n = len(np.load(shipped, allow_pickle=False)["init"])
        if args.ep_offset + args.episodes <= n:
            return shipped
    return os.path.join(args.out, args.env, f"targets_seed{args.seed}_ep{args.ep_offset + args.episodes}.npz")


def load_or_make_targets(args, env):
    """Fixed (init, goal) particle states per episode, shared by every checkpoint evaluated on the same problems."""
    path = args.targets
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta["env"] != args.env or meta["seed"] != args.seed:
            raise ValueError(
                f"{path} holds env={meta['env']} seed={meta['seed']}; this run asks {args.env}/{args.seed}"
            )
        n_need = args.ep_offset + args.episodes
        if len(z["init"]) < n_need:
            raise ValueError(f"{path} holds {len(z['init'])} episodes; this run asks {n_need}")
        params = meta.get("params") or [None] * len(z["init"])
        print(f"targets: loaded {len(z['init'])} episodes from {path}")
        return {i: {"init": z["init"][i], "goal": z["goal"][i], "params": params[i]} for i in range(n_need)}
    if args.ep_offset:
        raise ValueError("--ep-offset needs an existing --targets file (generate it once with --targets-only)")
    val = load_deform_val(args.env)
    inits, goals, params = [], [], []
    for ep in range(args.episodes):
        ep_seed = episode_seed(args.seed, ep)
        rng = np.random.default_rng(ep_seed)
        _, goal_s = env.sample_random_init_goal_states(ep_seed)
        init_s = val[int(rng.integers(len(val))), int(rng.integers(val.shape[1] - 1))]
        inits.append(np.asarray(init_s, dtype=np.float32))
        goals.append(np.asarray(goal_s, dtype=np.float32))
        params.append(getattr(env, "last_goal_params", None))
    meta = {
        "env": args.env,
        "seed": args.seed,
        "episodes": args.episodes,
        "params": params,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, init=np.stack(inits), goal=np.stack(goals), meta=json.dumps(meta))
    print(f"targets: generated {args.episodes} episodes -> {path}")
    return {i: {"init": inits[i], "goal": goals[i], "params": params[i]} for i in range(args.episodes)}


def set_episode(env_name, env, ep_seed, goal_H, rng, target=None):
    """-> (init_obs, init_info, goal_img, goal_state, extra)."""
    if ENV_CONF[env_name]["goal_source"] == "expert":  # metaworld
        from evals.simu_env_planning.envs.metaworld import task_name_to_policy

        def mw_reset():
            env.unwrapped.np_random = np.random.default_rng(ep_seed)
            return env.reset()

        policy = task_name_to_policy(ENV_CONF[env_name]["task"])()
        obs, info = mw_reset()
        done, expert_success, steps = False, False, 0
        while not done and steps < MW_EXPERT_MAX_STEPS:
            a = policy.get_action(np.asarray(info["state"], dtype=np.float64))
            obs, _, done, _, info = env.step(torch.tensor(a, dtype=torch.float32))
            done, steps = bool(done), steps + 1
            if float(info.get("success", 0.0)) >= 1:
                expert_success = True
                break
        goal_img = obs_to_img(obs)
        state_g = np.asarray(info["state"], dtype=np.float64)
        init_obs, iinfo = mw_reset()
        return init_obs, iinfo, goal_img, state_g, {"expert_success": expert_success, "expert_steps": steps}
    if ENV_CONF[env_name]["goal_source"] == "random_state":
        extra = {}
        if target is not None:
            init_s, goal_s = target["init"], target["goal"]
            extra["goal_params"] = target.get("params")
        else:
            init_s, goal_s = env.sample_random_init_goal_states(ep_seed)
            if env_name in DEFORM_ENVS:
                val = load_deform_val(env_name)
                init_s = val[int(rng.integers(len(val))), int(rng.integers(val.shape[1] - 1))]
                extra["goal_params"] = getattr(env, "last_goal_params", None)
        goal_obs, ginfo = env.prepare(ep_seed, goal_s)
        init_obs, iinfo = env.prepare(ep_seed, init_s)
        if env_name in DEFORM_ENVS:
            err = float(np.abs(np.asarray(iinfo["state"], dtype=np.float64)[:, :3] - np.asarray(init_s)[:, :3]).mean())
            if err > 0.05:
                raise RuntimeError(f"init state not loaded into the simulator (mean |dpos| {err:.3f})")
            extra["init_load_err"] = round(err, 4)
        return init_obs, iinfo, obs_to_img(goal_obs), np.asarray(ginfo["state"], dtype=np.float64), extra
    # pusht: init = a val-trajectory state, goal = the state after replaying its recorded actions
    st, ac, vel, sl = load_pusht_val()
    need = FRAMESKIP * goal_H + 1
    while True:
        i = int(rng.integers(len(sl)))
        if sl[i] >= need:
            break
    off = int(rng.integers(sl[i] - need + 1))
    init7 = np.concatenate([st[i, off], vel[i, off]])
    exec_actions = torch.tensor(ac[i, off : off + need - 1] * ENV_CONF[env_name]["exec_scale"], dtype=torch.float32)
    env.prepare(ep_seed, init7)
    obses, rewards, dones, infos = env.step_multiple(exec_actions)
    goal_img = obs_to_img(obses[-1])
    state_g = np.asarray(infos[-1]["state"], dtype=np.float64)
    init_obs, iinfo = env.prepare(ep_seed, init7)
    return init_obs, iinfo, goal_img, state_g, {}


def run_episode(args, env, wm, planner, ep, ep_seed, out_dir, target=None):
    rng = np.random.default_rng(ep_seed)
    env.reset(seed=ep_seed)
    init_obs, info, goal_img, state_g, extra = set_episode(args.env, env, ep_seed, args.goal_H, rng, target)
    zg = wm.encode_frame(goal_img, use_teacher=True)
    planner.set_objective(ReprTargetDistMPCObjective(None, target_enc=zg, sum_all_diffs=False))
    init_res = env.eval_state(state_g, np.asarray(info["state"], dtype=np.float64))
    obs, cur_info, done, flag_success = init_obs, info, False, False
    frames, frame_texts = [obs_to_img(init_obs)], [""]
    plan_losses, texts, executed = [], [], 0
    ep_pca = None
    pix_dec = getattr(wm, "pixel_decoder", None)
    pushes_moved = pushes_inbox = 0
    use_ctx = wm.ctxt_window > 1
    hist_f = [wm.encode_frame(obs_to_img(init_obs))] if use_ctx else []
    hist_t = []
    t0 = time.time()
    while executed < args.goal_H and not done:
        pz = proprio_of(cur_info)
        if use_ctx:
            ctx = (hist_f[-wm.ctxt_window :], hist_t[-(wm.ctxt_window - 1) :] if len(hist_f) > 1 else [])
        else:
            ctx = wm.encode_frame(obs_to_img(obs))
        result = planner.plan((ctx, pz), steps_left=args.goal_H - executed)
        plan_losses.append(result.losses[-1].item())
        r = len(plan_losses) - 1
        enc = result.predicted_best_encs_over_iterations[-1]
        cur_lat = hist_f[-1] if use_ctx else ctx
        if hasattr(wm, "decode") and save_this_ep(args.save_pred_episodes, ep, args.ep_offset):
            save_pred_strip(
                wm,
                result,
                obs_to_img(obs),
                goal_img,
                os.path.join(out_dir, f"ep{ep:03d}_replan{r}_pred.png"),
                f"replan {r}: current | imagined +1..+{len(enc)} | goal",
            )
        elif save_this_ep(args.save_pred_pca, ep, args.ep_offset):
            ep_pca = save_pred_pca_strip(
                ep_pca,
                cur_lat,
                enc,
                zg,
                obs_to_img(obs),
                goal_img,
                os.path.join(out_dir, f"ep{ep:03d}_replan{r}_pred_pca.png"),
                banner=banner_frame,
                label=f"replan {r}: current px|pca | imagined +1..+{len(enc)} (pca) | goal pca|px",
            )
        if (
            pix_dec is not None
            and not hasattr(wm, "decode")
            and save_this_ep(args.save_pred_decode, ep, args.ep_offset)
        ):
            save_pred_decode_strip(
                pix_dec,
                cur_lat,
                enc,
                zg,
                obs_to_img(obs),
                goal_img,
                os.path.join(out_dir, f"ep{ep:03d}_replan{r}_pred_dec.png"),
                banner=banner_frame,
                label=f"replan {r}: current px|dec | imagined +1..+{len(enc)} (dec) | goal dec|px",
            )
        to_z = wm.spec.plan_to_z
        for a_u in result.actions.cpu().numpy():
            a_z = to_z(a_u)
            text = wm.render(None, a_z, prefix=False)
            texts.append(text)
            if use_ctx:
                prop_now = proprio_of(cur_info) if wm.dyn.mode != "none" else None
                hist_t.append(wm.render(prop_now, a_z, prefix=wm.domain_prefix))
            net = wm.spec.denorm_action(a_z) * ENV_CONF[args.env]["exec_scale"]
            if args.env in DEFORM_ENVS:  # [sx,sy,dx,dy] -> one env push [x_s, y_s, x_e, y_e]
                end_env = (net[:2] + net[2:]) @ DEFORM_R
                acts = torch.tensor(np.concatenate([net[:2] @ DEFORM_R, end_env])[None], dtype=torch.float32)
                prev_state = np.asarray(cur_info["state"], dtype=np.float64)
            else:  # the chunk sum is split evenly over the frameskip sub-steps
                fs = ENV_CONF[args.env].get("frameskip", FRAMESKIP)
                acts = torch.tensor(np.repeat((net / fs)[None], fs, 0), dtype=torch.float32)
            obses, rewards, dones, infos = env.step_multiple(acts)
            flag_success = flag_success or any(float(i.get("success", 0.0) or 0.0) >= 1 for i in infos)
            frames.extend(obs_to_img(o) for o in obses)
            frame_texts.extend([text] * len(obses))
            obs, cur_info = obses[-1], infos[-1]
            if args.env in DEFORM_ENVS:
                new_state = np.asarray(cur_info["state"], dtype=np.float64)
                pushes_moved += int(np.abs(new_state[:, :3] - prev_state[:, :3]).mean() > 0.01)
                pushes_inbox += int(np.all(np.abs(end_env) <= 1.5))
            if use_ctx:
                hist_f.append(wm.encode_frame(obs_to_img(obs)))
            executed += 1
            done = bool(dones[-1])
            if done:
                break
    res = env.eval_state(state_g, np.asarray(cur_info["state"], dtype=np.float64))
    if save_this_ep(args.save_episodes, ep, args.ep_offset):
        strip = np.concatenate([obs_to_img(init_obs), goal_img, obs_to_img(obs)], axis=1)
        Image.fromarray(strip).save(os.path.join(out_dir, f"ep{ep:03d}_init_goal_final.png"))
    if save_this_ep(args.video_episodes, ep, args.ep_offset):
        ims = [banner_frame(np.concatenate([f, goal_img], axis=1), t) for f, t in zip(frames, frame_texts)]
        ims[0].save(
            os.path.join(out_dir, f"ep{ep:03d}.gif"), save_all=True, append_images=ims[1:], duration=100, loop=0
        )
    success = flag_success if args.env == "metaworld" else bool(res["success"])  # metaworld: success at any step
    row = {
        "ep": ep,
        "seed": ep_seed,
        "success": success,
        "end_dist": float(res["state_dist"]),
        "init_dist": float(init_res["state_dist"]),
        "plan_loss": plan_losses[-1],
        "plan_time_s": round(time.time() - t0, 1),
        "texts": texts,
    }
    if args.env == "metaworld":
        row["success_dist"] = bool(res["success"])
    if args.env in DEFORM_ENVS:
        row.update({"pushes_executed": executed, "pushes_moved": pushes_moved, "pushes_inbox": pushes_inbox})
    row.update(extra)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--env", required=True, choices=list(ENV_CONF))
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--goal-H", type=int, default=6, help="episode length in action chunks")
    p.add_argument("--horizon", type=int, default=6, help="planning horizon H")
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--num-elites", type=int, default=10)
    p.add_argument("--num-act-stepped", type=int, default=6, help="chunks executed per plan (= horizon: open-loop)")
    p.add_argument("--var-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--proprio-mode",
        default="none",
        choices=["none", "linear"],
        help="linear: proprio clause in the text, propagated by proprio_dyn.json",
    )
    p.add_argument("--text-source", default="semantic", choices=["semantic", "lang6", "lang4"])
    p.add_argument(
        "--domain-prefix",
        default="off",
        choices=["on", "off"],
        help="on: '[env] ' prefix, for checkpoints trained with it",
    )
    p.add_argument(
        "--ctxt-window",
        type=int,
        default=None,
        help="context frames the predictor sees during imagination and across executed chunks "
        "(default: the checkpoint's training context, clip_len - 1)",
    )
    p.add_argument(
        "--save-episodes", type=int, default=3, help="init|goal|final strips for the first N episodes (-1 = all)"
    )
    p.add_argument("--video-episodes", type=int, default=-1)
    p.add_argument(
        "--save-pred-episodes",
        type=int,
        default=-1,
        help="rae: decoded imagined frames for the first N episodes (-1 = all)",
    )
    p.add_argument(
        "--save-pred-pca",
        type=int,
        default=0,
        help="imagined latents as PCA strips for the first N episodes (-1 = all)",
    )
    p.add_argument(
        "--save-pred-decode", type=int, default=0, help="imagined latents decoded to pixels with --decoder-ckpt"
    )
    p.add_argument(
        "--decoder-ckpt", default=None, help="decoder/ checkpoint trained for this checkpoint's vision encoder"
    )
    p.add_argument("--mw-task", default="reach", help="metaworld task (reach, reach-wall, ...)")
    p.add_argument("--tag", default=None)
    p.add_argument("--verify-render", default="on", choices=["on", "off"])
    p.add_argument(
        "--plan-space",
        default="start_end",
        choices=["start_end", "start_delta"],
        help="rope / granular CEM search space",
    )
    p.add_argument(
        "--deform-success-cd",
        type=float,
        default=None,
        help="rope / granular success threshold on the final Chamfer distance",
    )
    p.add_argument(
        "--targets",
        default=None,
        help="rope / granular fixed (init, goal) problem set (.npz); default: the shipped seed-1 set",
    )
    p.add_argument("--targets-only", action="store_true")
    p.add_argument("--ep-offset", type=int, default=0, help="first episode index (sharding over GPUs)")
    p.add_argument("--out", default=DEFAULT_OUT)
    add_backend_args(p)
    args = p.parse_args()

    if args.env == "metaworld":
        ENV_CONF["metaworld"]["task"] = f"mw-{args.mw_task}"
    if args.env not in DEFORM_ENVS and (
        args.plan_space != "start_end" or args.targets or args.targets_only or args.deform_success_cd is not None
    ):
        p.error("--plan-space / --targets / --deform-success-cd apply to rope and granular only")
    device = torch.device("cuda:0")
    spec = load_spec(args.env)
    if args.env in DEFORM_ENVS:
        spec.set_plan_space(args.plan_space)
        if args.targets is None:
            args.targets = default_targets(args)
    if not spec.proprio_dims and args.proprio_mode != "none":
        args.proprio_mode = "none"
    if args.verify_render == "on":
        verify_render(spec, args.proprio_mode == "none", args.text_source)
    dyn = ProprioDynamics(args.env, mode=args.proprio_mode)

    env, cfg = build_env(args.env, args.goal_H, args.num_samples, args.deform_success_cd)
    targets = load_or_make_targets(args, env) if args.targets else None
    if args.targets_only:
        return
    wm = build_backend(args, spec, dyn, device)
    wm.pixel_decoder = load_decoder(args.decoder_ckpt, device) if args.decoder_ckpt else None
    tag = run_tag(args, wm, f"-{args.mw_task}" if args.env == "metaworld" else "")
    out_dir = os.path.join(args.out, args.env, tag)
    os.makedirs(out_dir, exist_ok=True)

    max_norms, max_norm_dims = spec.plan_max_norms()
    planner = CEMPlanner(
        unroll=wm.unroll,
        iterations=args.iterations,
        num_samples=args.num_samples,
        horizon=args.horizon,
        action_dim=spec.plan_dim,
        var_scale=args.var_scale,
        num_elites=args.num_elites,
        max_norms=max_norms,
        max_norm_dims=max_norm_dims,
        local_generator=torch.Generator(device="cuda").manual_seed(args.seed),
        num_act_stepped=args.num_act_stepped,
    )
    rows = []
    for ep in range(args.ep_offset, args.ep_offset + args.episodes):
        row = run_episode(
            args, env, wm, planner, ep, episode_seed(args.seed, ep), out_dir, targets[ep] if targets else None
        )
        rows.append(row)
        info = f" moved={row['pushes_moved']}/{row['pushes_executed']}" if args.env in DEFORM_ENVS else ""
        print(
            f"ep {ep:03d} succ={int(row['success'])} end_dist={row['end_dist']:.2f} init_dist={row['init_dist']:.2f} "
            f"plan_loss={row['plan_loss']:.4f}{info} {row['plan_time_s']}s",
            flush=True,
        )
    summary = {
        "ckpt": os.path.abspath(args.ckpt),
        "env": args.env,
        "episodes": args.episodes,
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "end_dist_mean": float(np.mean([r["end_dist"] for r in rows])),
        "init_dist_mean": float(np.mean([r["init_dist"] for r in rows])),
        "args": dict(vars(args)),
        "plan_space": getattr(spec, "plan_space", "text"),
        "success_cd": cfg.task_specification.get("success_cd"),
        "targets": args.targets,
        "plan_time_s_mean": float(np.mean([r["plan_time_s"] for r in rows])),
        "wm_timing": getattr(wm, "timing", None),
        "wm_train_seconds": getattr(wm, "train_seconds", None),
        "rows": rows,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(
        f"\n== {tag} on {args.env}: success {summary['success_rate']:.2%} "
        f"({sum(r['success'] for r in rows)}/{len(rows)}), "
        f"end_dist {summary['end_dist_mean']:.2f} (init {summary['init_dist_mean']:.2f}) -> {out_dir}/result.json"
    )


if __name__ == "__main__":
    main()
