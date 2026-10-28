"""RoboCasa evaluation of a LaLa-Policy checkpoint (Cosmos Policy protocol).

Per task: 50 episodes over the 5 held-out test scenes (10 each, object split B), a fresh env per episode seeded
with seed * episode * 256, 10 no-op steps, then the policy is queried every --open-loop steps (chunk 32, execute 16);
success = env._check_success(); step budgets 300-1000 per task. The 7-D arm action is padded with the no-op mobile
base action. Run inside the RoboCasa venv:
  MUJOCO_GL=egl python -m evals.lala_policy.run_robocasa_eval --ckpt <ckpt> --task PnPCounterToCab --gpu 0 --out <dir>
Writes <out>/robocasa/<tag>/<task>.json after every episode.
"""

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from evals.lala_policy import rollout_video  # noqa: E402
from evals.lala_policy.imagine_viz import ImagineRecorder, add_arguments, build_decoder, save_this_ep  # noqa: E402

TASK_MAX_STEPS = {
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}
TASKS = list(TASK_MAX_STEPS)
TEST_SCENES = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))  # (layout, style); 2 styles never seen in training
NUM_STEPS_WAIT = 10
MOBILE_BASE_NOOP = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
# cosmos_policy/experiments/robot/robocasa/robocasa_controller_configs.pkl, inlined
CONTROLLER = {
    "type": "OSC_POSE",
    "input_max": 1,
    "input_min": -1,
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "kp": 150,
    "damping_ratio": 1,
    "impedance_mode": "fixed",
    "kp_limits": [0, 300],
    "damping_ratio_limits": [0, 10],
    "position_limits": None,
    "orientation_limits": None,
    "uncouple_pos_ori": True,
    "control_delta": True,
    "interpolation": None,
    "ramp_ratio": 0.2,
}
CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]


def make_env(task, seed, scene, img_res):
    import robocasa  # noqa: F401  (registers the kitchen envs with robosuite)
    import robosuite

    return robosuite.make(
        env_name=task,
        robots="PandaMobile",
        controller_configs=dict(CONTROLLER),
        camera_names=CAMERAS,
        camera_widths=img_res,
        camera_heights=img_res,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split="B",
        generative_textures=None,
        randomize_cameras=False,
        layout_and_style_ids=(scene,),
        translucent_robot=False,
    )


def policy_inputs(obs, flip):
    f = (lambda x: np.ascontiguousarray(np.flipud(x))) if flip else np.ascontiguousarray
    images = {
        "primary": f(obs["robot0_agentview_left_image"]),
        "secondary": f(obs["robot0_agentview_right_image"]),
        "wrist": f(obs["robot0_eye_in_hand_image"]),
    }
    proprio = np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]))
    return images, proprio


def save_gif(frames, path, every=4, size=128):
    ims = [Image.fromarray(f).resize((size, size)) for f in frames[::every]]
    if ims:
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=100, loop=0)


def run_task(policy, task, args, out_dir, decoder=None):
    path = os.path.join(out_dir, f"{task}.json")
    episodes = args.episodes if args.episodes is not None else list(range(args.trials))
    if os.path.exists(path) and not args.overwrite:
        prev = json.load(open(path))
        if prev.get("n", 0) >= len(episodes):
            print(f"[skip] {task}: {prev['n']} episodes already done -> {path}", flush=True)
            return prev
    max_steps = args.max_steps or TASK_MAX_STEPS[task]
    result = {
        "task": task,
        "ckpt": args.ckpt,
        "epoch": policy.epoch,
        "seed": args.seed,
        "open_loop": args.open_loop,
        "chunk_size": policy.chunk_size,
        "action_head": policy.action_head_kind,
        "flow_steps": policy.flow_steps,
        "flip": not args.no_flip,
        "max_steps": max_steps,
        "img_res": args.env_res,
        "episodes": [],
        "n": 0,
        "n_success": 0,
        "success_rate": 0.0,
        "elapsed_s": 0.0,
    }
    t_task = time.time()
    for i, ep in enumerate(episodes):
        t_ep = time.time()
        scene = TEST_SCENES[(ep // 10) % len(TEST_SCENES)]
        env = make_env(task, args.seed * ep * 256, scene, args.env_res)
        obs = env.reset()  # every reset re-samples the scene: reset exactly once per episode (cosmos)
        task_text = env.get_ep_meta()["lang"]
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))
        rec = ImagineRecorder(policy, decoder) if save_this_ep(args.save_imagine, i) else None
        save_video = ep < args.save_videos or rec is not None
        video = None
        if save_video and args.video_res:  # every step at --video-res, next to the 128 px GIF
            running = os.path.join(out_dir, f"{task}_ep{ep:02d}_running.mp4")
            video = rollout_video.RolloutVideo(running, args.video_res, args.video_cams, args.env_res, gif_every=4)
        queue, frames, success, steps = [], [], False, 0
        for t in range(max_steps):
            if rec is not None and rec.wants(t):
                rec.frame(t, policy_inputs(obs, not args.no_flip)[0])
            if not queue:
                images, proprio = policy_inputs(obs, not args.no_flip)
                if rec is None:
                    chunk = policy.act(images, proprio, task_text)
                else:  # the same chunk, plus the imagined t+K frame kept for the comparison
                    chunk = rec.act(t, images, proprio, task_text)
                queue = [a for a in chunk[: args.open_loop]]
            action = queue.pop(0)
            if action.shape[-1] == 7 and env.action_dim == 12:
                action = np.concatenate([action, MOBILE_BASE_NOOP])
            if save_video:
                frames.append(np.flipud(obs["robot0_agentview_left_image"]))
            if video is not None:
                video.frame(env.sim, obs)
            obs, _, _, _ = env.step(action)
            steps = t + 1
            if env._check_success():
                success = True
                break
        if video is not None:
            video.frame(env.sim, obs)  # the state the episode ended in
        env.close()
        outcome = "ok" if success else "fail"
        if video is not None:
            stem = os.path.join(out_dir, f"{task}_ep{ep:02d}_{outcome}")
            video.close(stem + ".mp4", stem + ".gif")  # the GIF from the same renders, not the 128 px one
        if rec is not None:
            rec.frame(steps, policy_inputs(obs, not args.no_flip)[0])
            rec.finish(
                steps,
                os.path.join(out_dir, "imagine", task, f"ep{ep:02d}_{outcome}"),
                f"{task} ep{ep} {outcome} | executes {args.open_loop} of {policy.chunk_size}",
            )
        result["episodes"].append(
            {
                "ep": ep,
                "scene": list(scene),
                "task_text": task_text,
                "success": bool(success),
                "steps": steps,
                "sec": round(time.time() - t_ep, 1),
            }
        )
        if rec is not None:  # the step from which the renders were broken (imagine_viz.render_fault), else null
            result["episodes"][-1]["render_fault_at"] = rec.fault_at
        result["n"] = len(result["episodes"])
        result["n_success"] = sum(e["success"] for e in result["episodes"])
        result["success_rate"] = result["n_success"] / result["n"]
        result["elapsed_s"] = round(time.time() - t_task, 1)
        if save_video and frames and video is None:
            save_gif(frames, os.path.join(out_dir, f"{task}_ep{ep:02d}_{outcome}.gif"))
        with open(path, "w") as f:
            json.dump(result, f, indent=1)
        print(
            f"{task} ep {ep} scene {scene}: {'SUCCESS' if success else 'fail'} in {steps} steps "
            f"({time.time() - t_ep:.0f}s) "
            f"| running {result['n_success']}/{result['n']} | '{task_text}'",
            flush=True,
        )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", nargs="+", default=None, help=f"task names (default: all 24): {TASKS}")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument(
        "--seed",
        type=int,
        default=195,
        help="l1 head: label only (env + policy deterministic); flow head: sampling-noise seed",
    )
    ap.add_argument(
        "--flow-steps",
        type=int,
        default=None,
        help="flow head only: Euler steps (default = the ckpt's action_head.steps)",
    )
    ap.add_argument("--open-loop", type=int, default=16, help="actions executed per query (cosmos RoboCasa: 16 of 32)")
    ap.add_argument("--env-res", type=int, default=224)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--save-videos", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("LALA_POLICY_EVAL_OUT", "logs/policy_eval"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="override the protocol step limit (smoke tests only)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="run these protocol episode indices instead of range(trials): scene = TEST_SCENES[(ep // 10) %% 5], "
        "env seed = seed * ep * 256 (not the protocol -- e.g. 0 10 20 30 40 = one episode per test scene)",
    )
    add_arguments(ap)
    rollout_video.add_arguments(ap, CAMERAS)
    args = ap.parse_args()
    for t in args.task or []:
        if t not in TASK_MAX_STEPS:
            raise SystemExit(f"unknown task {t!r}; choose from {TASKS}")

    from evals.lala_policy.policy import PolicyAgent

    policy = PolicyAgent(
        args.ckpt,
        device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",
        flow_steps=args.flow_steps,
        seed=args.seed,
    )
    if args.open_loop > policy.chunk_size:
        raise ValueError(f"--open-loop {args.open_loop} > chunk_size {policy.chunk_size}")
    tag = args.tag or f"{os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))}-e{policy.epoch}"
    out_dir = os.path.join(args.out, "robocasa", tag)
    os.makedirs(out_dir, exist_ok=True)
    decoder = build_decoder(args, policy)
    results = [run_task(policy, t, args, out_dir, decoder) for t in (args.task or TASKS)]
    n = sum(r["n"] for r in results)
    ns = sum(r["n_success"] for r in results)
    print(f"robocasa [{tag}] {len(results)} tasks: {ns}/{n} = {100 * ns / max(1, n):.1f}% -> {out_dir}")


if __name__ == "__main__":
    main()
