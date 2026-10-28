"""LIBERO evaluation of a LaLa-Policy checkpoint (Cosmos Policy protocol).

Per task: the 50 benchmark init states, 10 no-op steps, then the policy is queried every --open-loop steps and the
chunk executed open-loop; success = env done; step budget 220 / 280 / 300 / 520 (spatial / object / goal / 10).
Images are flipped vertically before the policy (LIBERO renders upside-down, as in the training data).
Run inside the LIBERO venv:
  MUJOCO_GL=egl python -m evals.lala_policy.run_libero_eval --ckpt <ckpt> --suite libero_spatial --tasks 0 1 2 \
      --gpu 0 --out <dir>
Writes <out>/libero/<tag>/<suite>/task<k>.json after every episode (finished tasks are skipped on rerun).
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

TASK_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
NUM_STEPS_WAIT = 10
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
CAMERAS = ["agentview", "robot0_eye_in_hand"]


def ensure_libero_config():
    """LIBERO asks interactive questions when ~/.libero/config.yaml is missing; write the default one instead."""
    import yaml

    path = os.path.expanduser("~/.libero/config.yaml")
    if os.path.exists(path):
        return
    import libero

    root = os.path.join(list(libero.__path__)[0], "libero")
    cfg = {
        "benchmark_root": root,
        "bddl_files": os.path.join(root, "bddl_files"),
        "init_states": os.path.join(root, "init_files"),
        "datasets": os.path.join(os.path.dirname(root), "datasets"),
        "assets": os.path.join(root, "assets"),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


def load_init_states(suite, task_id):
    """benchmark.get_task_init_states with weights_only=False (the .pruned_init files pickle numpy arrays)."""
    from libero.libero import get_libero_path

    task = suite.get_task(task_id)
    path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    return torch.load(path, weights_only=False)


def make_env(suite, task_id, resolution):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)  # cosmos: the seed affects object placement even with fixed init states
    return env, task.language


def policy_inputs(obs, flip):
    prim, wrist = obs["agentview_image"], obs["robot0_eye_in_hand_image"]
    if flip:
        prim, wrist = np.flipud(prim), np.flipud(wrist)
    proprio = np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]))
    return {"primary": np.ascontiguousarray(prim), "wrist": np.ascontiguousarray(wrist)}, proprio


def save_gif(frames, path, every=3, size=128):
    ims = [Image.fromarray(f).resize((size, size)) for f in frames[::every]]
    if ims:
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=100, loop=0)


def run_task(policy, suite, suite_name, task_id, args, out_dir, decoder=None):
    path = os.path.join(out_dir, f"task{task_id:02d}.json")
    if os.path.exists(path) and not args.overwrite:
        prev = json.load(open(path))
        if prev.get("n", 0) >= args.trials:
            print(f"[skip] {suite_name} task {task_id}: {prev['n']} episodes already done -> {path}", flush=True)
            return prev
    env, task_text = make_env(suite, task_id, args.env_res)
    init_states = load_init_states(suite, task_id)
    max_steps = args.max_steps or TASK_MAX_STEPS[suite_name]
    result = {
        "suite": suite_name,
        "task_id": task_id,
        "task": task_text,
        "ckpt": args.ckpt,
        "epoch": policy.epoch,
        "seed": args.seed,
        "open_loop": args.open_loop,
        "chunk_size": policy.chunk_size,
        "action_head": policy.action_head_kind,
        "flow_steps": policy.flow_steps,
        "flip": not args.no_flip,
        "max_steps": max_steps,
        "episodes": [],
        "n": 0,
        "n_success": 0,
        "success_rate": 0.0,
        "elapsed_s": 0.0,
    }
    t_task = time.time()
    for ep in range(args.trials):
        t_ep = time.time()
        env.reset()
        obs = env.set_init_state(init_states[ep % len(init_states)])
        rec = ImagineRecorder(policy, decoder) if save_this_ep(args.save_imagine, ep) else None
        save_video = ep < args.save_videos or rec is not None
        video = None
        if save_video and args.video_res:  # every step at --video-res, next to the 128 px GIF
            running = os.path.join(out_dir, f"task{task_id:02d}_ep{ep:02d}_running.mp4")
            video = rollout_video.RolloutVideo(running, args.video_res, args.video_cams, args.env_res, gif_every=3)
        queue, frames, success, t = [], [], False, 0
        while t < max_steps + NUM_STEPS_WAIT:
            if t < NUM_STEPS_WAIT:
                obs, _, done, _ = env.step(DUMMY_ACTION)
                t += 1
                continue
            if rec is not None and rec.wants(t - NUM_STEPS_WAIT):
                rec.frame(t - NUM_STEPS_WAIT, policy_inputs(obs, not args.no_flip)[0])
            if not queue:
                images, proprio = policy_inputs(obs, not args.no_flip)
                if rec is None:
                    chunk = policy.act(images, proprio, task_text)
                else:  # the same chunk, plus the imagined t+K frame kept for the comparison
                    chunk = rec.act(t - NUM_STEPS_WAIT, images, proprio, task_text)
                queue = [a for a in chunk[: args.open_loop]]
            action = queue.pop(0)
            if args.binarize_gripper:
                action = action.copy()
                action[-1] = 1.0 if action[-1] > 0 else -1.0
            if save_video:
                frames.append(np.flipud(obs["agentview_image"]))
            if video is not None:
                video.frame(env.sim, obs)
            obs, _, done, _ = env.step(action.tolist())
            t += 1
            if done:
                success = True
                break
        outcome = "ok" if success else "fail"
        if video is not None:
            video.frame(env.sim, obs)  # the state the episode ended in
            stem = os.path.join(out_dir, f"task{task_id:02d}_ep{ep:02d}_{outcome}")
            video.close(stem + ".mp4", stem + ".gif")  # the GIF from the same renders, not the 128 px one
        if rec is not None:
            rec.frame(t - NUM_STEPS_WAIT, policy_inputs(obs, not args.no_flip)[0])
            rec.finish(
                t - NUM_STEPS_WAIT,
                os.path.join(out_dir, "imagine", f"task{task_id:02d}", f"ep{ep:02d}_{outcome}"),
                f"{suite_name} task {task_id} ep{ep} {outcome} | executes {args.open_loop} of {policy.chunk_size}"
                f" | {task_text}",
            )
        result["episodes"].append(
            {"ep": ep, "success": bool(success), "steps": t - NUM_STEPS_WAIT, "sec": round(time.time() - t_ep, 1)}
        )
        if rec is not None:  # the step from which the renders were broken (imagine_viz.render_fault), else null
            result["episodes"][-1]["render_fault_at"] = rec.fault_at
        result["n"] = len(result["episodes"])
        result["n_success"] = sum(e["success"] for e in result["episodes"])
        result["success_rate"] = result["n_success"] / result["n"]
        result["elapsed_s"] = round(time.time() - t_task, 1)
        if save_video and frames and video is None:
            save_gif(frames, os.path.join(out_dir, f"task{task_id:02d}_ep{ep:02d}_{outcome}.gif"))
        with open(path, "w") as f:
            json.dump(result, f, indent=1)
        print(
            f"{suite_name} task {task_id} ep {ep}: {'SUCCESS' if success else 'fail'} in {t - NUM_STEPS_WAIT} steps "
            f"({time.time() - t_ep:.0f}s) | running {result['n_success']}/{result['n']}",
            flush=True,
        )
    env.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", required=True, choices=sorted(TASK_MAX_STEPS))
    ap.add_argument("--tasks", type=int, nargs="*", default=None, help="task ids (default: all 10)")
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
    ap.add_argument(
        "--open-loop", type=int, default=16, help="actions executed per policy query (cosmos LIBERO: 16 = chunk)"
    )
    ap.add_argument("--env-res", type=int, default=256)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument(
        "--binarize-gripper", action="store_true", help="snap the gripper command to +-1 (cosmos does not)"
    )
    ap.add_argument("--save-videos", type=int, default=2, help="GIFs for the first N episodes of each task")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("LALA_POLICY_EVAL_OUT", "logs/policy_eval"))
    ap.add_argument("--tag", default=None, help="default <run folder>-e<epoch>")
    ap.add_argument("--max-steps", type=int, default=None, help="override the protocol step limit (smoke tests only)")
    ap.add_argument("--overwrite", action="store_true")
    add_arguments(ap)
    rollout_video.add_arguments(ap, CAMERAS)
    args = ap.parse_args()

    ensure_libero_config()
    from libero.libero import benchmark

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
    out_dir = os.path.join(args.out, "libero", tag, args.suite)
    os.makedirs(out_dir, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    tasks = args.tasks if args.tasks else list(range(suite.n_tasks))
    decoder = build_decoder(args, policy)
    results = [run_task(policy, suite, args.suite, k, args, out_dir, decoder) for k in tasks]
    n = sum(r["n"] for r in results)
    ns = sum(r["n_success"] for r in results)
    print(f"{args.suite} [{tag}] tasks {tasks}: {ns}/{n} = {100 * ns / max(1, n):.1f}% -> {out_dir}")


if __name__ == "__main__":
    main()
