"""RoboCasa MimicGen (mg_im) HDF5 -> Cosmos-schema HDF5 (per-frame JPEG) + episode JSONL + stats.

Only the demos listed in the official in-file mask (mask/300_demos by default, the GR00T-N1 300-demo
protocol) are repacked; the original gzip-chunked images are too slow for per-sample random access.

Outputs under --out: hdf5/<Task>__<mask>.hdf5, jsonl/robocasa_mg_<N>.jsonl, stats_<N>.json

  python data_preparation/policy/build_robocasa_mg_jsonl.py --root data --mask 300_demos --workers 24
"""
import argparse
import glob
import io
import json
import multiprocessing as mp
import os
import sys
import time

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import add_root_arg, data_root  # noqa: E402

ACTION_DIM = 7
VIEWS = {
    "obs/robot0_agentview_left_rgb_jpeg": "obs/robot0_agentview_left_image",
    "obs/robot0_agentview_right_rgb_jpeg": "obs/robot0_agentview_right_image",
    "obs/robot0_eye_in_hand_rgb_jpeg": "obs/robot0_eye_in_hand_image",
}
DEMO_KEYS = {
    "primary": "obs/robot0_agentview_left_rgb_jpeg",
    "secondary": "obs/robot0_agentview_right_rgb_jpeg",
    "wrist": "obs/robot0_eye_in_hand_rgb_jpeg",
}


def demo_num(name):
    return int(name.split("_")[1])


def task_of(path):  # .../single_stage/kitchen_<group>/<Task>/mg/<date>/x.hdf5
    return path.split(os.sep)[-4]


def encode_jpeg(frame, quality):
    bio = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame)).save(bio, "JPEG", quality=quality)
    return np.frombuffer(bio.getvalue(), dtype=np.uint8)


def repack_one(args):
    src, dst, mask, quality = args
    task = task_of(src)
    if os.path.exists(dst):
        with h5py.File(dst, "r") as f:
            if f.attrs.get("complete", False):
                return task, "skip", len(f["data"]), None
    tmp = dst + ".tmp"
    t0 = time.time()
    vlen = h5py.vlen_dtype(np.uint8)
    n_steps = 0
    with h5py.File(src, "r") as f, h5py.File(tmp, "w") as g:
        demos = sorted((x.decode() if isinstance(x, bytes) else str(x) for x in f["mask"][mask][:]), key=demo_num)
        g.attrs["source_h5"] = os.path.abspath(src)
        g.attrs["mask"] = mask
        g.attrs["jpeg_quality"] = quality
        g.attrs["env_args"] = f["data"].attrs["env_args"]
        for demo in demos:
            s = f["data"][demo]
            o = s["obs"]
            actions = s["actions"][:].astype(np.float32)
            T = actions.shape[0]
            proprio = np.concatenate(
                [o["robot0_gripper_qpos"][:], o["robot0_eef_pos"][:], o["robot0_eef_quat"][:]], axis=1
            ).astype(np.float32)
            if proprio.shape != (T, 9):
                raise ValueError(f"{src}:{demo} proprio {proprio.shape}")
            d = g.create_group(f"data/{demo}")
            d.attrs["task_description"] = json.loads(s.attrs["ep_meta"])["lang"]
            d.attrs["ep_meta"] = s.attrs["ep_meta"]
            d.attrs["source_group"] = f"data/{demo}"
            d.create_dataset("actions", data=actions)
            d.create_dataset("robot_states", data=proprio)
            for out_key, in_key in VIEWS.items():
                frames = o[in_key.split("/")[1]][:]
                if frames.shape[0] != T:
                    raise ValueError(f"{src}:{demo} {in_key} has {frames.shape[0]} frames but {T} actions")
                ds = d.create_dataset(out_key, (T,), dtype=vlen)
                for t in range(T):
                    ds[t] = encode_jpeg(frames[t], quality)
            n_steps += T
        g.attrs["complete"] = True
    with h5py.File(tmp, "r") as g:
        assert len(g["data"]) == len(demos), (dst, len(g["data"]), len(demos))
        for demo in (demos[0], demos[-1]):
            T = g[f"data/{demo}/actions"].shape[0]
            for key in DEMO_KEYS.values():
                ds = g[f"data/{demo}/{key}"]
                assert ds.shape[0] == T
                for t in (0, T // 2, T - 1):
                    im = Image.open(io.BytesIO(ds[t].tobytes()))
                    im.load()
                    assert im.size == (128, 128), im.size
    os.replace(tmp, dst)
    return task, "done", len(demos), f"{n_steps} steps, {os.path.getsize(dst) / 1e9:.1f} GB, {time.time() - t0:.0f} s"


def index_demos(files, mask, check_jpeg_n):
    records, n_steps, per_task, langs = [], 0, {}, {}
    for path in files:
        task = os.path.basename(path).split("__")[0]
        with h5py.File(path, "r") as f:
            assert f.attrs["mask"] == mask, (path, f.attrs["mask"])
            for demo in sorted(f["data"].keys(), key=demo_num):
                g = f[f"data/{demo}"]
                actions = g["actions"][:, :ACTION_DIM].astype(np.float32)
                proprio = g["robot_states"][:].astype(np.float32)
                T = actions.shape[0]
                for key in DEMO_KEYS.values():
                    if g[key].shape[0] != T:
                        raise ValueError(f"{path}:{demo} {key} has {g[key].shape[0]} frames but {T} actions")
                for i in range(min(check_jpeg_n, T)):
                    im = Image.open(io.BytesIO(g[DEMO_KEYS["primary"]][i * (T - 1) // max(1, check_jpeg_n - 1)].tobytes()))
                    im.load()
                text = str(g.attrs["task_description"])
                records.append(
                    {
                        "dataset": "robocasa_mg",
                        "suite": task,
                        "source": "demo",
                        "id": f"robocasa_mg/{task}/{mask}/{demo}",
                        "task": text,
                        "h5": os.path.abspath(path),
                        "group": f"data/{demo}",
                        "keys": DEMO_KEYS,
                        "num_steps": int(T),
                        "success": True,
                        "actions": actions.round(6).tolist(),
                        "proprio": proprio.round(6).tolist(),
                    }
                )
                n_steps += T
                per_task[task] = per_task.get(task, 0) + 1
                langs.setdefault(task, set()).add(text)
    return records, n_steps, per_task, {k: len(v) for k, v in langs.items()}


def main():
    ap = argparse.ArgumentParser()
    add_root_arg(ap)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mask", default="300_demos")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--check-jpeg", type=int, default=3)
    ap.add_argument("--skip-repack", action="store_true")
    args = ap.parse_args()
    root = data_root(args.root)
    raw = args.raw or os.path.join(root, "robocasa_mg_raw")
    out = args.out or os.path.join(root, "robocasa_mg")
    cosmos_stats = os.path.join(root, "cosmos_robocasa", "stats.json")
    n_tag = args.mask.split("_")[0]

    srcs = sorted(glob.glob(os.path.join(raw, "v0.1/single_stage/*/*/mg/*/demo_gentex_im128_randcams.hdf5")))
    if args.tasks:
        srcs = [p for p in srcs if task_of(p) in set(args.tasks)]
    if not srcs:
        raise FileNotFoundError(f"no mg_im files under {raw}")
    os.makedirs(os.path.join(out, "hdf5"), exist_ok=True)
    jobs = [(p, os.path.join(out, "hdf5", f"{task_of(p)}__{args.mask}.hdf5"), args.mask, args.jpeg_quality) for p in srcs]
    if not args.skip_repack:
        with mp.get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
            for task, status, n, info in pool.imap_unordered(repack_one, jobs):
                print(f"[{time.strftime('%F %T')}] {status} {task}: {n} demos {info or ''}", flush=True)
    dsts = [j[1] for j in jobs]
    records, n_steps, per_task, n_langs = index_demos(dsts, args.mask, args.check_jpeg)
    jsonl_path = os.path.join(out, "jsonl", f"robocasa_mg_{n_tag}.jsonl")
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    acts = np.concatenate([np.asarray(r["actions"], dtype=np.float32) for r in records])
    props = np.concatenate([np.asarray(r["proprio"], dtype=np.float32) for r in records])
    stats = {}
    for name, x in (("actions", acts), ("proprio", props)):
        stats[f"{name}_min"] = x.min(0).tolist()
        stats[f"{name}_max"] = x.max(0).tolist()
        stats[f"{name}_mean"] = x.mean(0).tolist()
        stats[f"{name}_std"] = x.std(0).tolist()
        stats[f"{name}_median"] = np.median(x, 0).tolist()
    stats["source"] = f"RoboCasa v0.1 mg_im mask/{args.mask}, actions sliced to 7 dims, recomputed from {jsonl_path}"
    stats["summary"] = {
        "mask": args.mask, "demos": len(records), "demo_steps": n_steps, "tasks": len(per_task),
        "demos_per_task": per_task, "unique_task_strings_per_task": n_langs, "jpeg_quality": args.jpeg_quality,
    }
    if os.path.exists(cosmos_stats):
        c = json.load(open(cosmos_stats))
        stats["within_cosmos_stats_range"] = {
            k: bool(np.all(np.asarray(stats[f"{k}_min"]) >= np.asarray(c[f"{k}_min"])[: len(stats[f"{k}_min"])] - 1e-6)
                    and np.all(np.asarray(stats[f"{k}_max"]) <= np.asarray(c[f"{k}_max"])[: len(stats[f"{k}_max"])] + 1e-6))
            for k in ("actions", "proprio")
        }
    stats_path = os.path.join(out, f"stats_{n_tag}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: v for k, v in stats["summary"].items() if not isinstance(v, dict)}, indent=1))
    print("within cosmos stats range:", stats.get("within_cosmos_stats_range"))
    print(f"-> {jsonl_path} ({os.path.getsize(jsonl_path) / 1e6:.0f} MB), {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
