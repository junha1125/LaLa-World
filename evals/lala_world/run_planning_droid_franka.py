"""Offline open-loop CEM planning on the DROID (franka) evaluation episodes released with JEPA-WMs.

Frames are on a 4 fps grid; episode = frame u (init) -> frame u + H (goal), one H-step plan, no execution.
Metric: |sum planned - sum expert| of the xyz displacement; Action Score = max(0, 800 * (0.1 - error_xyz)).
  python -m evals.lala_world.run_planning_droid_franka --ckpt <ckpt> --horizon 3 --episodes 64
"""

import argparse
import glob
import json
import os
import time

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from evals.lala_world.backends import add_backend_args, build_backend, run_tag
from evals.lala_world.latent_decode import load_decoder, save_pred_decode_strip
from evals.lala_world.latent_pca import save_pred_pca_strip
from evals.lala_world.paths import DEFAULT_OUT, EVAL_ONLY
from evals.lala_world.sim_spec import ProprioDynamics, load_spec
from evals.simu_env_planning.planning.objectives import ReprTargetDistMPCObjective
from evals.simu_env_planning.planning.planner import CEMPlanner

FRANKA_ROOT = f"{EVAL_ONLY}/franka_custom/data"
FRANKA_STEP = 8  # 30 fps recordings -> 4 fps grid


def rel_euler(r_from, r_to):
    return Rotation.from_matrix(r_to.as_matrix() @ r_from.as_matrix().T).as_euler("xyz")


def build_records():
    recs = []
    for path in sorted(glob.glob(f"{FRANKA_ROOT}/**/episode.h5", recursive=True)):
        rid = "/".join(path.split("/")[-4:-1])
        try:
            with h5py.File(path, "r") as f:
                o = f["episode_data/observation"]
                T = o["cartesian_position"].shape[0]
                gi = np.arange(0, T, FRANKA_STEP)
                if len(gi) < 2:
                    continue
                frames = o["exterior_image_2_left"][:][gi]
                cart = o["cartesian_position"][:][gi]
                clo = o["gripper_position"][:][gi]
        except OSError as e:
            print(f"skip {rid}: unreadable h5 ({str(e)[:80]})")
            continue
        poses = np.concatenate([cart, clo[:, None]], axis=1)
        rots = Rotation.from_euler("xyz", poses[:, 3:6])
        K = len(gi) - 1
        net = np.zeros((K, 7))
        net[:, :3] = poses[1:, :3] - poses[:-1, :3]
        for k in range(K):
            net[k, 3:6] = rel_euler(rots[k], rots[k + 1])
        net[:, 6] = poses[1:, 6] - poses[:-1, 6]
        recs.append((rid, frames, net))
    return recs


def run_probe(wm, spec, recs, n):
    """1-step prediction error on the grid transitions vs copying the current frame (identity) and empty text."""
    trans = [
        (frames[k - 1], frames[k], frames[k + 1], net[k - 1], net[k])
        for _, frames, net in recs
        for k in range(1, len(net))
    ]
    idx = np.linspace(0, len(trans) - 1, min(n, len(trans))).round().astype(int)
    trans = [trans[i] for i in idx]
    z_prev = torch.cat([wm.encode_frame(t[0]) for t in trans])
    z_st = torch.cat([wm.encode_frame(t[1]) for t in trans])
    z_tg = torch.cat([wm.encode_frame(t[2]) for t in trans])
    tx_prev = [wm.render(None, spec.norm_action(t[3]), prefix=wm.domain_prefix) for t in trans]
    tx = [wm.render(None, spec.norm_action(t[4]), prefix=wm.domain_prefix) for t in trans]
    res = {}
    B = z_st.shape[0]
    multi = hasattr(wm, "_predict_ctx")  # single-frame backends (--wm rae) have no context path
    for s in range(0, B, 256):
        e = min(s + 256, B)
        pairs = [("pred_c1", wm._predict(z_st[s:e], tx[s:e])), ("empty_c1", wm._predict(z_st[s:e], [""] * (e - s)))]
        if multi:
            pairs += [
                ("pred_c2", wm._predict_ctx([z_prev[s:e], z_st[s:e]], [tx_prev[s:e], tx[s:e]])),
                ("empty_c2", wm._predict_ctx([z_prev[s:e], z_st[s:e]], [[""] * (e - s)] * 2)),
            ]
        for key, val in pairs:
            res.setdefault(key, []).append(((val - z_tg[s:e]) ** 2).mean(dim=(1, 2)))
    out = {k: round(float(torch.cat(v).mean()), 4) for k, v in res.items()}
    out["identity"] = round(float(((z_st - z_tg) ** 2).mean()), 4)
    out["n"] = B
    print(
        f"probe n={B} ctxt1: pred={out['pred_c1']} identity={out['identity']} empty={out['empty_c1']}"
        + (f" | ctxt2: pred={out['pred_c2']} empty={out['empty_c2']}" if multi else "")
    )
    return out


def run_episode(args, wm, planner, recs, ep, ep_seed, out_dir):
    rng = np.random.default_rng(ep_seed)
    for _ in range(200):
        rid, frames, net = recs[int(rng.integers(len(recs)))]
        if len(net) >= args.horizon:
            break
    else:
        raise RuntimeError("no episode with a complete planning window found")
    u = int(rng.integers(len(net) - args.horizon + 1))
    init_img, goal_img = frames[u], frames[u + args.horizon]
    zg = wm.encode_frame(goal_img, use_teacher=True)
    planner.set_objective(ReprTargetDistMPCObjective(None, target_enc=zg, sum_all_diffs=False))
    t0 = time.time()
    ctx = wm.encode_frame(init_img)
    result = planner.plan((ctx, None), steps_left=args.horizon)
    enc = result.predicted_best_encs_over_iterations[-1]
    if (
        hasattr(wm, "decode")
        and enc.dim() == 4
        and (args.save_pred_episodes < 0 or ep - args.ep_offset < args.save_pred_episodes)
    ):
        pred = (wm.decode(enc[:, 0]).permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        side = pred.shape[1]
        ends = [np.asarray(Image.fromarray(i).resize((side, side), Image.BILINEAR)) for i in (init_img, goal_img)]
        Image.fromarray(np.concatenate([ends[0], *pred, ends[1]], axis=1)).save(
            os.path.join(out_dir, f"ep{ep:03d}_pred.png")
        )
    elif args.save_pred_pca < 0 or ep - args.ep_offset < args.save_pred_pca:
        save_pred_pca_strip(None, ctx, enc, zg, init_img, goal_img, os.path.join(out_dir, f"ep{ep:03d}_pred_pca.png"))
    dec = getattr(wm, "pixel_decoder", None)
    if dec is not None and (args.save_pred_decode < 0 or ep - args.ep_offset < args.save_pred_decode):
        save_pred_decode_strip(
            dec, ctx, enc, zg, init_img, goal_img, os.path.join(out_dir, f"ep{ep:03d}_pred_dec.png")
        )
    planned_z = result.actions.cpu().numpy()
    planned_net = wm.spec.denorm_action(planned_z)
    expert_net = net[u : u + args.horizon]
    diff = np.abs(planned_net.sum(0) - expert_net.sum(0))
    floor = np.abs(expert_net.sum(0))
    pv, ev = planned_net.sum(0)[:3], expert_net.sum(0)[:3]
    row = {
        "ep": ep,
        "seed": ep_seed,
        "id": rid,
        "offset": u,
        "end_dist_xyz": float(diff[:3].sum()),
        "end_dist_ori": float(diff[3:6].sum()),
        "end_dist_clo": float(diff[6]),
        "floor_xyz": float(floor[:3].sum()),
        "floor_ori": float(floor[3:6].sum()),
        "floor_clo": float(floor[6]),
        "cos_xyz": float(pv @ ev / (np.linalg.norm(pv) * np.linalg.norm(ev) + 1e-12)),
        "plan_loss": result.losses[-1].item(),
        "plan_time_s": round(time.time() - t0, 1),
        "planned_text": [wm.render(None, z, prefix=False) for z in planned_z],
        "expert_text": [wm.render(None, wm.spec.norm_action(a), prefix=False) for a in expert_net],
    }
    if args.save_episodes < 0 or ep - args.ep_offset < args.save_episodes:
        Image.fromarray(np.concatenate([init_img, goal_img], axis=1)).save(
            os.path.join(out_dir, f"ep{ep:03d}_init_goal.png")
        )
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--num-elites", type=int, default=10)
    p.add_argument("--var-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--text-source", default="semantic", choices=["semantic", "lang6", "lang4"])
    p.add_argument("--domain-prefix", default="off", choices=["on", "off"])
    p.add_argument("--ctxt-window", type=int, default=None, help="default: the checkpoint's training context")
    p.add_argument(
        "--rot-max-norm",
        type=float,
        default=None,
        help="per-chunk rotation bound in radians for dims 3-5 (None = spec clip)",
    )
    p.add_argument("--probe", type=int, default=256, help="1-step probe transitions before planning (0 = off)")
    p.add_argument("--save-episodes", type=int, default=3)
    p.add_argument("--save-pred-episodes", type=int, default=-1, help="rae: decoded imagined frames (-1 = all)")
    p.add_argument("--save-pred-pca", type=int, default=0)
    p.add_argument("--save-pred-decode", type=int, default=0)
    p.add_argument("--decoder-ckpt", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--ep-offset", type=int, default=0)
    p.add_argument("--out", default=DEFAULT_OUT)
    add_backend_args(p)
    args = p.parse_args()

    device = torch.device("cuda:0")
    spec = load_spec("droid")
    wm = build_backend(args, spec, ProprioDynamics("droid", mode="none"), device)
    wm.pixel_decoder = load_decoder(args.decoder_ckpt, device) if args.decoder_ckpt else None
    recs = build_records()
    print(
        f"droid: {len(recs)} episodes, "
        f"{sum(max(0, len(net) - args.horizon + 1) for _, _, net in recs)} H={args.horizon} windows"
    )

    tag = run_tag(args, wm, f"-{args.text_source}-w{wm.ctxt_window}")
    out_dir = os.path.join(args.out, "droid", tag)
    os.makedirs(out_dir, exist_ok=True)
    probe = run_probe(wm, spec, recs, args.probe) if args.probe else {}

    if args.rot_max_norm is not None:
        rot_z = float(args.rot_max_norm) / float(np.max(spec.action_std[3:6]))
        max_norms, max_norm_dims = [spec.z_clip, rot_z, spec.grip_z_clip], [[0, 1, 2], [3, 4, 5], [6]]
    else:
        max_norms, max_norm_dims = [spec.z_clip, spec.grip_z_clip], [[0, 1, 2, 3, 4, 5], [6]]
    planner = CEMPlanner(
        unroll=wm.unroll,
        iterations=args.iterations,
        num_samples=args.num_samples,
        horizon=args.horizon,
        action_dim=7,
        var_scale=args.var_scale,
        num_elites=args.num_elites,
        max_norms=max_norms,
        max_norm_dims=max_norm_dims,
        local_generator=torch.Generator(device="cuda").manual_seed(args.seed),
        num_act_stepped=args.horizon,
    )
    rows = []
    for ep in range(args.ep_offset, args.ep_offset + args.episodes):
        ep_seed = (args.seed * args.seed + ep * args.seed) % (2**32 - 2)
        row = run_episode(args, wm, planner, recs, ep, ep_seed, out_dir)
        rows.append(row)
        print(
            f"ep {ep:03d} xyz={row['end_dist_xyz']:.4f}/{row['floor_xyz']:.4f} "
            f"ori={row['end_dist_ori']:.4f}/{row['floor_ori']:.4f} "
            f"clo={row['end_dist_clo']:.3f}/{row['floor_clo']:.3f} cos={row['cos_xyz']:+.2f} {row['plan_time_s']}s",
            flush=True,
        )

    def mean(k):
        return float(np.mean([r[k] for r in rows]))

    summary = {
        "ckpt": os.path.abspath(args.ckpt),
        "env": "droid",
        "episodes": args.episodes,
        "end_dist_xyz_mean": mean("end_dist_xyz"),
        "end_dist_ori_mean": mean("end_dist_ori"),
        "end_dist_clo_mean": mean("end_dist_clo"),
        "floor_xyz_mean": mean("floor_xyz"),
        "floor_ori_mean": mean("floor_ori"),
        "floor_clo_mean": mean("floor_clo"),
        "cos_xyz_mean": mean("cos_xyz"),
        "action_score": max(0.0, 800.0 * (0.1 - mean("end_dist_xyz"))),
        "plan_time_s_mean": mean("plan_time_s"),
        "wm_timing": getattr(wm, "timing", None),
        "wm_train_seconds": getattr(wm, "train_seconds", None),
        "probe": probe,
        "args": dict(vars(args)),
        "rows": rows,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(
        f"\n== {tag} on droid (H={args.horizon}, {args.episodes} eps): error_xyz {summary['end_dist_xyz_mean']:.4f} "
        f"(do-nothing floor {summary['floor_xyz_mean']:.4f}), Action Score {summary['action_score']:.1f} "
        f"-> {out_dir}/result.json"
    )


if __name__ == "__main__":
    main()
