"""Zero-shot RoboCasa MPC planning (JEPA-WMs Rc-R / Rc-Pl protocol) with LaLa-World.

Nothing from RoboCasa is in the training data: candidates are sampled in the DROID action space, rendered as
DROID sentences, and executed through the RoboCasa OSC_POSE controller (one 0.25 s chunk = 5 identical 20 Hz
steps). Problems come from the 14 teleoperated PnPCounterTop demos released with JEPA-WMs, cut into reach /
pick / place segments; success is the subtask condition at the final step. Runs inside the RoboCasa venv:
  python -m evals.lala_world.run_planning_robocasa --ckpt <ckpt> --subtask reach --episodes 32
Always compare against the floors (--policy hold | zero | random).
"""

import argparse
import json
import os
import time

# mujoco picks its GL backend at first context creation; robocasa's env module sets these too, but
# it is imported lazily inside build_env, so pin them before anything can grab a context.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import h5py
import numpy as np

# `evals/simu_env_planning/envs/robocasa.py` roots the object assets it rewrites demo XMLs to at
# $JEPAWM_HOME/robocasa/... and reads the variable at import time. Point it at the installed
# package so the two agree wherever robocasa lives.
import robocasa  # noqa: E402
import torch
from omegaconf import OmegaConf
from PIL import Image

from evals.lala_world.robocasa_task import sanitize_ep_meta  # noqa: E402  (also registers PnPCounterTop)

os.environ.setdefault("JEPAWM_HOME", os.path.dirname(os.path.dirname(robocasa.__path__[0])))
from evals.lala_world.latent_decode import load_decoder, save_pred_decode_strip  # noqa: E402
from evals.lala_world.latent_pca import save_pred_pca_strip  # noqa: E402
from evals.lala_world.paths import DEFAULT_OUT, EVAL_ONLY  # noqa: E402
from evals.lala_world.run_planning import banner_frame, episode_seed, obs_to_img, save_this_ep  # noqa: E402
from evals.lala_world.sim_spec import ProprioDynamics, load_spec  # noqa: E402
from evals.lala_world.wm import PlanningWM  # noqa: E402
from evals.simu_env_planning.planning.objectives import ReprTargetDistMPCObjective  # noqa: E402
from evals.simu_env_planning.planning.planner import CEMPlanner  # noqa: E402

RCASA_H5 = f"{EVAL_ONLY}/robocasa/combine_all_im256.hdf5"
# reach / pick / place segment ids, and which segments each subtask spans (robocasa_dset.get_frames)
SUBTASK_SEGMENTS = {
    "reach": (0,),
    "pick": (1,),
    "place": (2,),
    "reach-pick": (0, 1),
    "pick-place": (1, 2),
    "reach-pick-place": (0, 1, 2),
}
CAMERA = "robot0_leftview"
PLAN_DIM = 7  # DroidSpec: 6 z-scored continuous dims + the gripper-closedness delta (its own clip group)
ACTION_REPEAT = 5  # 4 fps droid chunk -> 20 Hz robocasa steps (paper: "5 times action repeat")
MAX_ENV_STEPS = 60


def rcasa_wrapper(env):
    """Walk down the wrapper stack (Pixel -> Tensor -> TimeLimit -> RoboCasaWrapper).

    Matched on the class, not on `hasattr`: `gym.Wrapper.__getattr__` forwards reads to the
    wrapped env, so every layer answers to `subtask_success`, while `__setattr__` does *not*
    forward -- assigning `goal_obj_pos` on an outer layer would silently miss the wrapper the
    success check reads it from.
    """
    while type(env).__name__ != "RoboCasaWrapper":
        env = env.env
    return env


def load_problems(subtask, img_size):
    """One problem per demo that contains every segment the subtask needs."""
    segs = SUBTASK_SEGMENTS[subtask]
    probs = []
    with h5py.File(RCASA_H5, "r") as f:
        for key in sorted(f["data"], key=lambda k: int(k.split("_")[1])):
            d = f[f"data/{key}"]
            seg = d["meta_data_info/current_task_segment"][:]
            idx = np.where(np.isin(seg, segs))[0]
            if len(idx) == 0 or set(segs) - set(seg[idx].tolist()):
                continue  # robocasa_dset raises here; we just drop the demo from the pool
            i0, i1 = int(idx[0]), int(idx[-1])
            probs.append(
                {
                    "demo": key,
                    "i0": i0,
                    "i1": i1,
                    "n_frames": len(idx),
                    "init_state": d["states"][i0].copy(),
                    "goal_state": d["states"][i1].copy(),
                    "init_img_dset": _resize(d[f"obs/{CAMERA}_image"][i0], img_size),
                    "goal_img_dset": _resize(d[f"obs/{CAMERA}_image"][i1], img_size),
                    # place goal = where the demo finally left the object (plan_evaluator.set_episode)
                    "goal_obj_pos": d["meta_data_info/current_obj_pos"][-1].copy(),
                    "env_info": {
                        "model_xml": d.attrs["model_file"],
                        "ep_meta": sanitize_ep_meta(json.loads(d.attrs["ep_meta"])),
                    },
                }
            )
    if not probs:
        raise RuntimeError(f"no demo in {RCASA_H5} covers subtask {subtask!r}")
    return probs


def _resize(img, size):
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def build_env(subtask, img_size, gripper, cam_configs):
    """The jepa-wms rcasa_custom env, built with the teleop demos' Robotiq gripper.

    `create_env` in robocasa's env_utils has no gripper knob and PandaOmron defaults to the Panda
    gripper, but every demo XML carries a Robotiq85 hand. `reset_from_xml_string` swaps the model
    but not the python robot description, so `_setup_references` would then look up Panda finger
    joints that the loaded model does not have. Constructing with Robotiq85 keeps the two in sync.
    """
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    from evals.lala_world.robocasa_task import register_teleop_cameras
    from evals.simu_env_planning.envs.init import PixelWrapper, TensorWrapper
    from evals.simu_env_planning.envs.robocasa import RoboCasaWrapper
    from evals.simu_env_planning.envs.wrappers.time_limit import TimeLimit

    register_teleop_cameras(cam_configs)
    cfg = OmegaConf.create(
        {
            "task_specification": {
                "task": "robocasa-PnPCounterTop",
                "obs": "rgb",
                "obs_concat_channels": False,
                "goal_source": "dset",
                "succ_def": "simu",
                "done_at_succ": False,
                "max_episode_steps": MAX_ENV_STEPS,
                "num_frames": 1,
                "num_proprios": 1,
                "img_size": img_size,
                "multitask": False,
                "env": {
                    "manip_only": True,
                    "custom_task": False,
                    "camera_name": CAMERA,
                    "rescale_act_droid_to_rcasa": True,
                    "subtask": subtask,
                    "reach_threshold": 0.2,
                    "place_threshold": 0.15,
                },
            },
            "meta": {"seed": 0},
        }
    )
    base = robosuite.make(
        env_name="PnPCounterTop",
        robots="PandaOmron",
        gripper_types=gripper,
        controller_configs=load_composite_controller_config(controller=None, robot="PandaOmron"),
        camera_names=[CAMERA],
        camera_widths=img_size,
        camera_heights=img_size,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        # the offscreen renderer stays on for `RoboCasaWrapper.render`, but robosuite need not
        # build a camera *observation* every step -- nothing reads it, and it doubles render cost
        use_camera_obs=False,
        camera_depths=False,
        seed=cfg.meta.seed,
        translucent_robot=False,
        # the demos' objects are 12/13 AI-generated; robocasa only samples the objaverse
        # registry by default and `_create_obj` looks the recorded mjcf up through it
        obj_registries=("objaverse", "aigen"),
    )
    env = TimeLimit(RoboCasaWrapper(base, cfg, "PnPCounterTop", camera_name=CAMERA), MAX_ENV_STEPS)
    env.max_episode_steps = env._max_episode_steps
    return PixelWrapper(cfg, TensorWrapper(env)), cfg


def plan_max_norms(spec, xyz_max, rot_max, grip_max):
    """Physical per-chunk bounds -> z-space clip groups.

    The rcasa config clips the planner in raw droid units (`max_norms: [0.05, 0.01, 1.]` over
    dims [[0,1,2],[3,4,5],[6]]); we sample in z, so each group is divided by the widest std in it
    (offline droid eval's convention) and no dim can exceed the physical bound. `None` falls back
    to the spec's own p1/p99 support, which for rotation is ~0.19 rad -- 19x the rcasa clip.
    """
    xyz = spec.z_clip if xyz_max is None else float(xyz_max) / float(np.max(spec.action_std[:3]))
    rot = spec.z_clip if rot_max is None else float(rot_max) / float(np.max(spec.action_std[3:6]))
    grip = spec.grip_z_clip if grip_max is None else float(grip_max) / spec.grip_std
    return [xyz, rot, grip], [[0, 1, 2], [3, 4, 5], [6]]


def sim_state(env):
    return np.asarray(rcasa_wrapper(env).env.sim.get_state().flatten(), dtype=np.float64)


def fresh_info(env):
    """`RoboCasaWrapper.prepare()` builds its info dict from `reset()` and only *afterwards* loads
    the demo state, so the info it returns still describes the freshly sampled scene. Re-derive the
    subtask distances from the simulator as it stands now."""
    w = rcasa_wrapper(env)
    inner = w.env  # the robosuite env itself
    obs = inner._get_observations(force_update=True)
    return w.get_obs_proprio_succ_from_info(dict(obs))[1]


def info_metrics(info):
    return {
        "hand_obj_dist": float(info.get("hand_obj_dist", -1.0)),
        "obj_goal_dist": float(info.get("obj_goal_dist", -1.0)),
        "obj_lift": float(info.get("obj_lift", -1.0)),
    }


def run_episode(args, env, wm, planner, prob, ep, ep_seed, out_dir):
    spec = wm.spec
    rng = np.random.default_rng(ep_seed)
    wrap = rcasa_wrapper(env)
    wrap.goal_obj_pos = prob["goal_obj_pos"] if "place" in args.subtask else None

    # rendering the goal costs a full scene rebuild; the zero/random floors never look at it
    need_sim_goal = args.goal_source == "sim" and args.policy == "plan"
    goal_img = prob["goal_img_dset"]
    if need_sim_goal:
        goal_obs, _ = env.prepare(ep_seed, prob["goal_state"], env_info=prob["env_info"])
        goal_img = obs_to_img(goal_obs)
    init_obs, _stale = env.prepare(ep_seed, prob["init_state"], env_info=prob["env_info"])
    info = fresh_info(env)
    loaded = sim_state(env)  # prepare() swallows a failed set_state, so check rather than trust it
    init_load_err = (
        float(np.abs(loaded - prob["init_state"]).max()) if loaded.shape == prob["init_state"].shape else float("inf")
    )
    dset_gap = float(np.abs(obs_to_img(init_obs).astype(np.int16) - prob["init_img_dset"].astype(np.int16)).mean())

    if args.policy == "plan":
        zg = wm.encode_frame(goal_img)
        planner.set_objective(ReprTargetDistMPCObjective(None, target_enc=zg, sum_all_diffs=False))

    n_chunks = MAX_ENV_STEPS // ACTION_REPEAT
    obs, cur_info, done = init_obs, info, False
    frames, frame_texts = [obs_to_img(init_obs)], [""]
    plan_losses, texts, succ_any, executed = [], [], False, 0
    ep_pca = None  # --save-pred-pca: one PCA basis per episode, fitted at the first replan
    pix_dec = getattr(wm, "pixel_decoder", None)  # --decoder-ckpt
    # closest approach over the episode: final-step success (the paper's rule) says nothing about
    # how near the arm actually got, and on reach the two come apart badly
    closest = {"hand_obj_dist": info["hand_obj_dist"], "obj_goal_dist": info.get("obj_goal_dist", -1.0)}
    use_ctx = wm.ctxt_window > 1
    hist_f = [wm.encode_frame(obs_to_img(init_obs))] if use_ctx else []
    hist_t = []
    t0 = time.time()
    while executed < n_chunks and not done:
        if args.policy == "plan":
            if use_ctx:
                ctx = (hist_f[-wm.ctxt_window :], hist_t[-(wm.ctxt_window - 1) :])
            else:
                ctx = wm.encode_frame(obs_to_img(obs))
            result = planner.plan((ctx, None), steps_left=n_chunks - executed)
            chunk = result.actions.cpu().numpy()
            plan_losses.append(result.losses[-1].item())
            r = len(plan_losses) - 1
            enc = result.predicted_best_encs_over_iterations[-1]
            cur_lat = hist_f[-1] if use_ctx else ctx
            if save_this_ep(args.save_pred_pca, ep, args.ep_offset):
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
            if pix_dec is not None and save_this_ep(args.save_pred_decode, ep, args.ep_offset):
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
        elif args.policy == "zero":  # blank plan: z = 0, i.e. the dataset's mean droid action
            chunk = np.zeros((args.num_act_stepped, PLAN_DIM))
        elif args.policy == "hold":  # physically motionless: net = 0 exactly
            chunk = np.tile(spec.norm_action(np.zeros(PLAN_DIM)), (args.num_act_stepped, 1))
        else:  # random: same distribution/clipping the planner starts from
            mn, dims = planner.max_norms, planner.max_norm_dims
            chunk = args.var_scale * rng.standard_normal((args.num_act_stepped, PLAN_DIM))
            for m, ds in zip(mn, dims):
                chunk[:, ds] = np.clip(chunk[:, ds], -m, m)
        for a_z in chunk:
            net = spec.denorm_action(a_z)  # droid units; RoboCasaWrapper rescales to the controller
            text = wm.render(None, a_z, prefix=False)
            texts.append(text)
            label = text
            if use_ctx:
                hist_t.append(wm.render(None, a_z, prefix=wm.domain_prefix))
            acts = torch.tensor(np.repeat(net[None], ACTION_REPEAT, 0), dtype=torch.float32)
            obses, _, dones, infos = env.step_multiple(acts)
            succ_any = succ_any or any(float(i.get("success", 0.0) or 0.0) >= 1 for i in infos)
            for i in infos:
                for k in closest:
                    v = float(i.get(k, -1.0))
                    if v >= 0:
                        closest[k] = min(closest[k], v)
            frames.extend(obs_to_img(o) for o in obses)
            frame_texts.extend([label] * len(obses))
            obs, cur_info = obses[-1], infos[-1]
            if use_ctx:
                hist_f.append(wm.encode_frame(obs_to_img(obs)))
            executed += 1
            done = bool(dones[-1])
            if done or executed >= n_chunks:
                break

    if save_this_ep(args.save_episodes, ep, args.ep_offset):
        strip = np.concatenate([obs_to_img(init_obs), goal_img, obs_to_img(obs)], axis=1)
        Image.fromarray(strip).save(os.path.join(out_dir, f"ep{ep:03d}_{prob['demo']}_init_goal_final.png"))
    if save_this_ep(args.video_episodes, ep, args.ep_offset):
        ims = [banner_frame(np.concatenate([f, goal_img], axis=1), t) for f, t in zip(frames, frame_texts)]
        ims[0].save(
            os.path.join(out_dir, f"ep{ep:03d}.gif"), save_all=True, append_images=ims[1:], duration=100, loop=0
        )

    row = {
        "ep": ep,
        "seed": ep_seed,
        "demo": prob["demo"],
        "success": bool(float(cur_info.get("success", 0.0) or 0.0) >= 1),  # final step (paper rule)
        "success_any": bool(succ_any),
        "chunks": executed,
        "init_load_err": round(init_load_err, 6),
        "sim_dset_img_gap": round(dset_gap, 2),
        "plan_loss": plan_losses[-1] if plan_losses else None,
        "plan_time_s": round(time.time() - t0, 1),
        "texts": texts,
    }
    row.update(info_metrics(cur_info))
    row.update({f"min_{k}": v for k, v in closest.items()})
    row.update({f"init_{k}": v for k, v in info_metrics(info).items()})
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--subtask", default="reach", choices=list(SUBTASK_SEGMENTS))
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument(
        "--ep-offset",
        type=int,
        default=0,
        help="first episode index (episodes ep-offset .. ep-offset+episodes-1 with the usual "
        "seed formula), so one subtask can be sharded over GPUs and merged by episode "
        "index like run_planning.py. The CEM generator is seeded once per process, so a "
        "different split changes the per-episode CEM noise -- keep the split when "
        "re-running a tag",
    )
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--num-elites", type=int, default=10)
    p.add_argument("--num-act-stepped", type=int, default=1, help="chunks executed per plan (rcasa: 1 -> 12 replans)")
    p.add_argument("--var-scale", type=float, default=1.0, help="CEM init std in z-space (droid eval convention)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--text-source", default="semantic", choices=["semantic", "lang6", "lang4"])
    p.add_argument("--domain-prefix", default="off", choices=["on", "off"])
    p.add_argument("--ctxt-window", type=int, default=None, help="default: the checkpoint's training context")
    p.add_argument("--xyz-max-norm", type=float, default=0.05, help="m per chunk, dims 0-2 (rcasa config: 0.05)")
    p.add_argument("--rot-max-norm", type=float, default=0.01, help="rad per chunk, dims 3-5 (rcasa config: 0.01)")
    p.add_argument("--grip-max-norm", type=float, default=None, help="closedness delta, dim 6 (None = spec support)")
    p.add_argument(
        "--policy",
        default="plan",
        choices=["plan", "zero", "hold", "random"],
        help="floors: hold = no motion at all, zero = the blank plan CEM starts from "
        "(z=0 is the mean droid action, not stillness), random = clipped Gaussian",
    )
    p.add_argument(
        "--episode-order",
        default="random",
        choices=["random", "cycle"],
        help="random = the original's sampling with replacement; cycle = one pass over the demo pool",
    )
    p.add_argument(
        "--goal-source",
        default="sim",
        choices=["sim", "dset"],
        help="sim: render the goal state through the same camera as the agent's frames",
    )
    p.add_argument("--gripper", default="Robotiq85Gripper")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="pick the GPU here rather than with CUDA_VISIBLE_DEVICES: robosuite asserts "
        "MUJOCO_EGL_DEVICE_ID is listed in CUDA_VISIBLE_DEVICES but then indexes EGL "
        "devices globally, so restricting visibility to one GPU makes the two "
        "requirements unsatisfiable for any GPU but 0",
    )
    p.add_argument(
        "--save-episodes",
        type=int,
        default=3,
        help="episodes whose init/goal strip is saved, counted from --ep-offset so "
        "each shard saves its own first N (-1 = all)",
    )
    p.add_argument("--video-episodes", type=int, default=3)
    p.add_argument(
        "--save-pred-pca",
        type=int,
        default=0,
        help="each replan's imagined latents as a PCA strip ep..._replanN_pred_pca.png (-1 = all, 0 = off)",
    )
    p.add_argument(
        "--save-pred-decode",
        type=int,
        default=0,
        help="same as --save-pred-pca but decoded to pixels with --decoder-ckpt, saved as "
        "ep..._replanN_pred_dec.png (-1 = all, 0 = off)",
    )
    p.add_argument(
        "--decoder-ckpt", default=None, help="decoder/ checkpoint trained for this checkpoint's vision encoder"
    )
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    torch.cuda.set_device(args.gpu)  # before any allocation; CEMPlanner builds on bare "cuda"
    device = torch.device(f"cuda:{args.gpu}")
    spec = load_spec("droid")  # droid vocabulary and action space: the zero-shot premise
    probs = load_problems(args.subtask, args.img_size)
    print(
        f"{args.subtask}: {len(probs)} of 14 demos have the needed segments -> "
        + ", ".join(f"{p['demo']}({p['n_frames']}f)" for p in probs)
    )

    wm = PlanningWM(
        args.ckpt,
        spec,
        ProprioDynamics("droid", mode="none"),
        device,
        domain_prefix=args.domain_prefix == "on",
        text_source=args.text_source,
        ctxt_window=args.ctxt_window,
    )
    wm.pixel_decoder = load_decoder(args.decoder_ckpt, device) if args.decoder_ckpt else None
    env, cfg = build_env(args.subtask, args.img_size, args.gripper, probs[0]["env_info"]["ep_meta"]["cam_configs"])

    run_name = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    tag = args.tag or f"{run_name}-e{wm.epoch}-{args.subtask}" + ("" if args.policy == "plan" else f"-{args.policy}")
    if args.ep_offset:
        tag += f"-ep{args.ep_offset}"
    out_dir = os.path.join(args.out, "robocasa", tag)
    os.makedirs(out_dir, exist_ok=True)

    max_norms, max_norm_dims = plan_max_norms(spec, args.xyz_max_norm, args.rot_max_norm, args.grip_max_norm)
    print(
        f"CEM z clips: xyz<={max_norms[0]:.2f} rot<={max_norms[1]:.3f} grip<={max_norms[2]:.2f} "
        f"(phys: {args.xyz_max_norm} m, {args.rot_max_norm} rad, {args.grip_max_norm})"
    )
    planner = CEMPlanner(
        unroll=wm.unroll,
        iterations=args.iterations,
        num_samples=args.num_samples,
        horizon=args.horizon,
        action_dim=PLAN_DIM,
        var_scale=args.var_scale,
        num_elites=args.num_elites,
        max_norms=max_norms,
        max_norm_dims=max_norm_dims,
        local_generator=torch.Generator(device=device).manual_seed(args.seed),
        num_act_stepped=args.num_act_stepped,
    )

    def summarize(rows):
        def mean(k):
            return float(np.mean([r[k] for r in rows]))

        return {
            "ckpt": os.path.abspath(args.ckpt),
            "env": "robocasa",
            "subtask": args.subtask,
            "episodes": args.episodes,
            "episodes_done": len(rows),
            "success_rate": mean("success"),
            "success_any_rate": mean("success_any"),
            "hand_obj_dist_mean": mean("hand_obj_dist"),
            "min_hand_obj_dist_mean": mean("min_hand_obj_dist"),
            "min_obj_goal_dist_mean": mean("min_obj_goal_dist"),
            "obj_goal_dist_mean": mean("obj_goal_dist"),
            "init_hand_obj_dist_mean": mean("init_hand_obj_dist"),
            "init_obj_goal_dist_mean": mean("init_obj_goal_dist"),
            "init_load_err_max": max(r["init_load_err"] for r in rows),
            "sim_dset_img_gap_mean": mean("sim_dset_img_gap"),
            "n_problems": len(probs),
            "max_norms_z": max_norms,
            "args": dict(vars(args)),
            "rows": rows,
        }

    def dump(rows):
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(summarize(rows), f, indent=1)

    rows = []
    for ep in range(args.ep_offset, args.ep_offset + args.episodes):
        ep_seed = episode_seed(args.seed, ep)
        if args.episode_order == "cycle":
            prob = probs[ep % len(probs)]
        else:
            prob = probs[int(np.random.default_rng(ep_seed).integers(len(probs)))]
        row = run_episode(args, env, wm, planner, prob, ep, ep_seed, out_dir)
        rows.append(row)
        loss = f"{row['plan_loss']:.4f}" if row["plan_loss"] is not None else "n/a"
        print(
            f"ep {ep:03d} {row['demo']:8s} succ={int(row['success'])} (any={int(row['success_any'])}) "
            f"hand_obj={row['hand_obj_dist']:.3f}<-{row['init_hand_obj_dist']:.3f} "
            f"obj_goal={row['obj_goal_dist']:.3f}<-{row['init_obj_goal_dist']:.3f} "
            f"plan_loss={loss} load_err={row['init_load_err']:.1e} imggap={row['sim_dset_img_gap']:.1f} "
            f"{row['plan_time_s']}s"
        )
        dump(rows)  # partial result.json after every episode: a killed run still reports

    summary = summarize(rows)
    print(
        f"\n== {tag} robocasa/{args.subtask} ({args.policy}): success {summary['success_rate']:.1%} "
        f"({sum(r['success'] for r in rows)}/{len(rows)}), any-step {summary['success_any_rate']:.1%}, "
        f"hand_obj {summary['hand_obj_dist_mean']:.3f} (init {summary['init_hand_obj_dist_mean']:.3f}), "
        f"obj_goal {summary['obj_goal_dist_mean']:.3f} (init {summary['init_obj_goal_dist_mean']:.3f}) "
        f"-> {out_dir}/result.json"
    )


if __name__ == "__main__":
    main()
