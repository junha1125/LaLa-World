"""DROID (lerobot/droid_1.0.1) -> episode-record JSONL for LaLa-World training.

Frames are taken on a 4 fps grid (every 4th frame of the 15 fps video) from the exterior_1_left camera.
One action = the end-effector delta between consecutive grid frames:
[dx, dy, dz (m), droll, dpitch, dyaw (rad), dgrip]; the first six are continuous (spec/bins), dgrip is
rendered as close/open/hold. There is no proprio text.

  python data_preparation/droid/build_droid_traj.py --root data --stats-only   # specs/droid_v1.yaml + lang_bins/droid_v1.yaml
  python data_preparation/droid/build_droid_traj.py --root data --workers 48   # frames + jsonl/traj/droid.jsonl
"""
import argparse
import functools
import glob
import json
import math
import os
import subprocess
import sys
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import add_root_arg, data_root  # noqa: E402

WINDOW = 1
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_PATH = os.path.join(HERE, "..", "specs", "droid_v1.yaml")
BINS_PATH = os.path.join(HERE, "..", "lang_bins", "droid_v1.yaml")
VIDEO_KEY = "observation.images.exterior_1_left"
GRIP_THRESH = 0.1

R6 = lambda a: [round(float(x), 6) for x in np.atleast_1d(a)]  # noqa: E731

ACTION_KEYS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
LANG_DIMS = [
    ("dx", "move", "forward", "backward"),
    ("dy", "move", "left", "right"),
    ("dz", "move", "up", "down"),
    ("droll", "rot", "roll right", "roll left"),
    ("dpitch", "rot", "pitch down", "pitch up"),
    ("dyaw", "rot", "turn left", "turn right"),
]
LANG_TEMPLATE = "move the gripper {0}, {1} and {2}, rotate {3}, {4} and {5}, then {6} the gripper"
INTENSITY = {"move": ["slightly {d}", "{d}", "far {d}"], "rot": ["slightly {d}", "{d}", "sharply {d}"]}

ROOT = None  # set in main()


def paths():
    root = data_root(ROOT)
    return root, os.path.join(root, "sim_raw", "droid_lerobot"), os.path.join(root, "droid", "frames")


def phrases(kind, direction, n_bins):
    full = [t.format(d=direction) for t in INTENSITY[kind]]
    return full if n_bins == 6 else [full[0], full[2]]


def grip_word(dgrip):
    return "close" if dgrip > GRIP_THRESH else "open" if dgrip < -GRIP_THRESH else "hold"


def render_semantic(a):
    fb = f"{abs(a[0]) * 100:.1f}cm {'forward' if a[0] >= 0 else 'backward'}"
    lr = f"{abs(a[1]) * 100:.1f}cm {'left' if a[1] >= 0 else 'right'}"
    ud = f"{abs(a[2]) * 100:.1f}cm {'up' if a[2] >= 0 else 'down'}"
    rot = (f"roll {math.degrees(a[3]):+.0f}deg, pitch {math.degrees(a[4]):+.0f}deg, "
           f"yaw {math.degrees(a[5]):+.0f}deg")
    return f"move gripper {fb}, {lr}, {ud}; rotate {rot}; {grip_word(a[6])} gripper"


@functools.lru_cache(maxsize=1)
def grid_step():
    _, raw, _ = paths()
    info = json.load(open(os.path.join(raw, "meta", "info.json")))
    h, w, _ = info["features"][VIDEO_KEY]["shape"]
    return max(1, round(info["fps"] / 4.0)), h, w


@functools.lru_cache(maxsize=1)
def episode_index():
    _, raw, _ = paths()
    rows = []
    for f in sorted(glob.glob(os.path.join(raw, "meta", "episodes", "chunk-*", "file-*.parquet"))):
        rows.append(pd.read_parquet(f, columns=[
            "episode_index", "length", "data/chunk_index", "data/file_index",
            f"videos/{VIDEO_KEY}/chunk_index", f"videos/{VIDEO_KEY}/file_index",
            f"videos/{VIDEO_KEY}/from_timestamp"]))
    E = pd.concat(rows, ignore_index=True)
    out = {}
    for _, r in E.sort_values("episode_index").iterrows():
        key = (int(r["data/chunk_index"]), int(r["data/file_index"]))
        out.setdefault(key, []).append((
            int(r["episode_index"]), int(r[f"videos/{VIDEO_KEY}/chunk_index"]),
            int(r[f"videos/{VIDEO_KEY}/file_index"]), float(r[f"videos/{VIDEO_KEY}/from_timestamp"]),
            int(r["length"])))
    return out


def units():
    return sorted(episode_index().keys())


def iter_unit_actions(unit, limit_episodes=None):
    step, _, _ = grid_step()
    _, raw, _ = paths()
    dch, dfi = unit
    dpath = os.path.join(raw, "data", f"chunk-{dch:03d}", f"file-{dfi:03d}.parquet")
    df = pd.read_parquet(dpath, columns=[
        "episode_index", "observation.state.cartesian_position", "observation.state.gripper_position"])
    groups = dict(tuple(df.groupby("episode_index")))
    n = 0
    for ep, vch, vfi, vts, length in episode_index()[unit]:
        if limit_episodes is not None and n >= limit_episodes:
            return
        sub = groups.get(ep)
        if sub is None:
            raise RuntimeError(f"ep{ep} in meta but not in {dpath}")
        if len(sub) != length:
            raise RuntimeError(f"ep{ep}: data rows {len(sub)} != meta length {length}")
        S = np.stack([np.asarray(x, dtype=np.float64) for x in sub["observation.state.cartesian_position"]])
        G = np.asarray(list(sub["observation.state.gripper_position"]), dtype=np.float64).reshape(-1)
        grid = np.arange(0, len(S), step)
        if len(grid) < 2:
            continue
        P, Gg = S[grid], G[grid]
        Rg = Rotation.from_euler("xyz", P[:, 3:6])
        dang = (Rg[1:] * Rg[:-1].inv()).as_euler("xyz")
        acts = np.concatenate([P[1:, :3] - P[:-1, :3], dang, (Gg[1:] - Gg[:-1])[:, None]], axis=1)
        n += 1
        yield ep, vch, vfi, vts, len(grid), np.round(acts, 6)


def extract_grid_frames(vts, F, mp4):
    step, h, w = grid_step()
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-ss", f"{vts:.6f}", "-i", mp4,
           "-vf", f"select='not(mod(n\\,{step}))'", "-vsync", "0", "-frames:v", str(F),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or len(p.stdout) != F * h * w * 3:
        raise RuntimeError(f"ffmpeg {mp4} vts={vts}: rc={p.returncode} "
                           f"bytes={len(p.stdout)}!={F * h * w * 3} {p.stderr[-300:]!r}")
    return np.frombuffer(p.stdout, dtype=np.uint8).reshape(F, h, w, 3)


def stats_unit(unit):
    nets, n_ep, grip_cnt = [], 0, {"close": 0, "open": 0, "hold": 0}
    for _, _, _, _, _, acts in iter_unit_actions(unit):
        n_ep += 1
        nets.append(acts)
        for dg in acts[:, 6]:
            grip_cnt[grip_word(dg)] += 1
    return (np.concatenate(nets) if nets else np.zeros((0, 7))), n_ep, grip_cnt


def run_stats(workers, limit=None):
    us = units()[:limit] if limit else units()
    nets, n_ep, grip_cnt = [], 0, {"close": 0, "open": 0, "hold": 0}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (a, ne, gc) in enumerate(ex.map(stats_unit, us)):
            nets.append(a)
            n_ep += ne
            for k in grip_cnt:
                grip_cnt[k] += gc[k]
            print(f"\rstats {i + 1}/{len(us)} units, eps={n_ep}", end="", flush=True)
    print()
    nets = np.concatenate(nets)
    n_tr = len(nets)
    c6, dg = nets[:, :6], nets[:, 6]
    step, _, _ = grid_step()

    spec = {
        "spec_version": 3,
        "dataset": "droid",
        "domain_prefix": "[droid] ",
        "source_jsonl": "jsonl/traj/droid.jsonl",
        "record_layout": "traj (one line per episode; n_records = number of 4 fps grid transitions)",
        "n_records": int(n_tr),
        "n_episodes": int(n_ep),
        "purpose": f"shared normalization space for text rendering and planning (action = one grid-step EE delta, window={WINDOW})",
        "text_normalized": False,
        "net_window": WINDOW,
        "frame_grid": {
            "native_fps": 15, "step": step, "dt_sec": round(step / 15.0, 6),
            "note": f"frames/<ep>/<t:06d>.jpg: t = grid index, native index = step*t. Camera {VIDEO_KEY} "
                    "(fixed external camera, placement differs per scene; direction words are in the robot base frame)",
        },
        "action_dim": 6,
        "action_dims": ACTION_KEYS,
        "units": {
            "action": "one grid-step EE delta: dxyz in m (difference), deuler in rad (relative rotation euler(R_g@R_c^T)), "
                      "raw SI units. net_action[6] = dgrip (gripper opening delta, + = closing)",
            "proprio": "none (absolute EE pose stays in the source parquet)",
        },
        "action_mean": R6(c6.mean(0)),
        "action_std": R6(c6.std(0)),
        "action_clip_lo": R6(np.percentile(c6, 1, axis=0)),
        "action_clip_hi": R6(np.percentile(c6, 99, axis=0)),
        "proprio_dim": 0,
        "proprio_dims": [],
        "proprio_mean": [],
        "proprio_std": [],
        "gripper": {
            "kind": "discrete3",
            "raw_field": "net_action[6] (dgrip)",
            "rule": {"close": f"dgrip > {GRIP_THRESH}", "open": f"dgrip < -{GRIP_THRESH}", "hold": "otherwise"},
            "train_freq": {k: round(v / n_tr, 4) for k, v in grip_cnt.items()},
            "zero_frac": round(float((dg == 0.0).mean()), 4),
            "mean": round(float(dg.mean()), 6),
            "std": round(float(dg.std()), 6),
            "clip_lo": round(float(np.percentile(dg, 1)), 6),
            "clip_hi": round(float(np.percentile(dg, 99)), 6),
            "note": "dominated by exact zeros, so it is not a Gaussian continuous dimension; planning treats it as a 3-way discrete choice",
        },
        "planning": {
            "sample_space": "continuous 6-D z-score (action_mean/std) Gaussian samples, clipped in z space; gripper separate (see gripper)",
            "clip": "z-space clip bounds = (action_clip_lo/hi - mean)/std (p1/p99)",
            "render_value": "semantic text uses the physical-unit net (= z*std+mean); lang text bins |net/std|",
        },
        "render": {
            "fields": {
                "actions_semantic": "physical-unit action clause only (no proprio; identical to the no_proprio field)",
                "actions_semantic_no_proprio": "physical-unit action clause only",
                "actions_lang6": "lang6 action clause only (identical to no_proprio)",
                "actions_lang6_no_proprio": "lang6 action clause only",
                "actions_lang4": "lang4 action clause only (identical to no_proprio)",
                "actions_lang4_no_proprio": "lang4 action clause only",
            },
            "semantic_template": "move gripper {|dx|cm} forward|backward, {|dy|cm} left|right, {|dz|cm} up|down; "
                                 "rotate roll {deg:+d}deg, pitch {deg:+d}deg, yaw {deg:+d}deg; close|open|hold gripper",
            "numerical": "none",
            "axis_grounding": "robot base frame: +x = forward, +y = left, +z = up",
            "note": "the [droid] prefix is added by the training loader when domain_prefix is on. "
                    "action_raw.net_action keeps the round-6 raw 7-D action; rendering and binning use the same rounded values",
        },
    }

    std6 = np.array(spec["action_std"])
    v = c6 / std6
    bins = {
        "spec_version": 1,
        "dataset": "droid",
        "source_jsonl": spec["source_jsonl"],
        "value_space": "action_raw.net_action[:6] / action_std (droid_v1.yaml); unitless, mean not subtracted",
        "side_rule": "v >= 0 -> pos direction word, v < 0 -> neg",
        "bin_rule": "bin = #(abs_edges <= |v|) (bisect_right on the round-6 edges); counts are re-tallied with the same rule",
        "edge_rule": "per direction, np.quantile of |v| (6-bin: [1/3, 2/3], 4-bin: [0.5]), round-6",
        "gripper": "no bins; the discrete 3-way close|open|hold word fills '..., then {word} the gripper'",
        "n_pairs": int(n_tr),
        "purpose": "training lang text and planning renders share the same bin edges and phrases",
        "builder": "data_preparation/droid/build_droid_traj.py",
        "dims": [],
    }
    for i, (k, kind, pos_d, neg_d) in enumerate(LANG_DIMS):
        x = v[:, i]
        an, ap = np.abs(x[x < 0]), np.abs(x[x >= 0])
        dim = {"key": k, "kind": kind, "pos_dir": pos_d, "neg_dir": neg_d,
               "zero_count": int((x == 0.0).sum()), "neg_count": int(len(an)), "pos_count": int(len(ap))}
        for nb, qs in [(6, [1 / 3, 2 / 3]), (4, [0.5])]:
            en = [round(float(e), 6) for e in np.quantile(an, qs)]
            ep = [round(float(e), 6) for e in np.quantile(ap, qs)]
            cn = np.bincount([bisect_right(en, a) for a in an], minlength=len(en) + 1)
            cp = np.bincount([bisect_right(ep, a) for a in ap], minlength=len(ep) + 1)
            dim[f"bins{nb}"] = {"neg_abs_edges": en, "pos_abs_edges": ep,
                               "neg_counts": [int(c) for c in cn], "pos_counts": [int(c) for c in cp],
                               "neg_phrases": phrases(kind, neg_d, nb), "pos_phrases": phrases(kind, pos_d, nb)}
        bins["dims"].append(dim)

    os.makedirs(os.path.dirname(SPEC_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(BINS_PATH), exist_ok=True)
    with open(SPEC_PATH, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True, width=110)
    with open(BINS_PATH, "w") as f:
        yaml.safe_dump(bins, f, sort_keys=False, allow_unicode=True, width=110)
    print(f"== droid episodes={n_ep} transitions={n_tr} grip_freq={spec['gripper']['train_freq']} "
          f"zero_frac={spec['gripper']['zero_frac']}\n   -> {SPEC_PATH} + {BINS_PATH}")


@functools.lru_cache(maxsize=1)
def load_spec_bins():
    with open(SPEC_PATH) as f:
        spec = yaml.safe_load(f)
    with open(BINS_PATH) as f:
        bins = yaml.safe_load(f)
    return spec, bins


def build_unit(unit, limit_episodes=None):
    spec, bins = load_spec_bins()
    _, raw, frames_root = paths()
    std6 = np.array(spec["action_std"])
    records = []
    counts = {nb: [{"neg": [0] * (nb // 2), "pos": [0] * (nb // 2)} for _ in LANG_DIMS] for nb in (6, 4)}
    for ep, vch, vfi, vts, F, acts in iter_unit_actions(unit, limit_episodes):
        tdir = os.path.join(frames_root, f"ep{ep:06d}")
        fpaths = [os.path.join(tdir, f"{t:06d}.jpg") for t in range(F)]
        if not all(os.path.exists(p) for p in fpaths):
            mp4 = os.path.join(raw, "videos", VIDEO_KEY, f"chunk-{vch:03d}", f"file-{vfi:03d}.mp4")
            imgs = extract_grid_frames(vts, F, mp4)
            os.makedirs(tdir, exist_ok=True)
            for t in range(F):
                Image.fromarray(imgs[t]).save(fpaths[t], quality=95)
        sem, l6, l4 = [], [], []
        for a in acts:
            sem.append(render_semantic(a))
            av = a[:6] / std6
            clause = {}
            for nb in (6, 4):
                parts = []
                for j, (dim, x) in enumerate(zip(bins["dims"], av)):
                    b = dim[f"bins{nb}"]
                    side = "pos" if x >= 0 else "neg"
                    idx = bisect_right(b[f"{side}_abs_edges"], abs(x))
                    counts[nb][j][side][idx] += 1
                    parts.append(b[f"{side}_phrases"][idx])
                clause[nb] = LANG_TEMPLATE.format(*parts, grip_word(a[6]))
            l6.append(clause[6])
            l4.append(clause[4])
        records.append({
            "dataset": "droid",
            "id": f"droid_traj/ep{ep:06d}",
            "frames": fpaths,
            "actions_semantic": sem,
            "actions_semantic_no_proprio": sem,
            "actions_lang6": l6,
            "actions_lang6_no_proprio": l6,
            "actions_lang4": l4,
            "actions_lang4_no_proprio": l4,
            "action_raw": {"text_source": "traj_v1", "net_action": [R6(a) for a in acts],
                           "proprio": [[] for _ in acts]},
        })
    return records, counts


def run_build(workers, limit=None, out_suffix="", limit_episodes=None):
    spec, bins = load_spec_bins()
    root, _, _ = paths()
    us = units()[:limit] if limit else units()
    os.makedirs(os.path.join(root, "jsonl", "traj"), exist_ok=True)
    out_path = os.path.join(root, "jsonl", "traj", f"droid{out_suffix}.jsonl")
    n_tr = n_ep = 0
    total = {nb: [{"neg": [0] * (nb // 2), "pos": [0] * (nb // 2)} for _ in LANG_DIMS] for nb in (6, 4)}
    with ProcessPoolExecutor(max_workers=workers) as ex, open(out_path + ".tmp", "w") as f:
        for records, counts in ex.map(build_unit, us, [limit_episodes] * len(us)):
            for r in records:
                f.write(json.dumps(r) + "\n")
                n_tr += len(r["actions_semantic"])
            n_ep += len(records)
            for nb in (6, 4):
                for j in range(len(LANG_DIMS)):
                    for side in ("neg", "pos"):
                        total[nb][j][side] = [a + b for a, b in zip(total[nb][j][side], counts[nb][j][side])]
            print(f"\rdroid: {n_ep} episodes / {n_tr}/{spec['n_records']} transitions", end="", flush=True)
    print()
    if limit is None and limit_episodes is None:
        if n_tr != spec["n_records"] or n_ep != spec["n_episodes"]:
            raise RuntimeError(f"droid: built {n_ep} ep/{n_tr} tr != spec {spec['n_episodes']}/{spec['n_records']}")
        for nb in (6, 4):
            for j, dim in enumerate(bins["dims"]):
                for side in ("neg", "pos"):
                    want = dim[f"bins{nb}"][f"{side}_counts"]
                    got = total[nb][j][side]
                    if got != want:
                        raise RuntimeError(f"droid/{dim['key']} bins{nb} {side} counts {got} != yaml {want}")
    os.replace(out_path + ".tmp", out_path)
    print(f"== droid: {n_ep} episodes, {n_tr} transitions -> {out_path}")


def main():
    global ROOT, SPEC_PATH, BINS_PATH
    ap = argparse.ArgumentParser()
    add_root_arg(ap)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None, help="parquet units (smoke)")
    ap.add_argument("--limit-episodes", type=int, default=None, help="episodes per unit (smoke)")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--spec-path", default=SPEC_PATH)
    ap.add_argument("--bins-path", default=BINS_PATH)
    args = ap.parse_args()
    ROOT = args.root
    SPEC_PATH, BINS_PATH = args.spec_path, args.bins_path
    if args.stats_only:
        run_stats(args.workers, limit=args.limit)
    else:
        run_build(args.workers, limit=args.limit, out_suffix=args.out_suffix, limit_episodes=args.limit_episodes)


if __name__ == "__main__":
    main()
