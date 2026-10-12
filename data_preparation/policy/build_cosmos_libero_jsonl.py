"""Index the Cosmos Policy LIBERO release as episode-record JSONL (images stay in the HDF5 files).

  {"dataset": "libero_spatial", "suite": ..., "source": "demo" | "rollout", "id": ..., "task": ...,
   "h5": "/abs/path.hdf5", "group": "data/demo_3" | "", "keys": {"primary": ..., "wrist": ...},
   "num_steps": T, "success": bool, "actions": [[7] x T], "proprio": [[9] x T]}

Outputs under --out: jsonl/libero_<suite>.jsonl, jsonl/libero_<suite>_rollouts.jsonl, stats.json

  python data_preparation/policy/build_cosmos_libero_jsonl.py --root data
"""

import argparse
import glob
import io
import json
import os
import re
import sys

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import add_root_arg, data_root  # noqa: E402

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DEMO_KEYS = {"primary": "obs/agentview_rgb_jpeg", "wrist": "obs/eye_in_hand_rgb_jpeg"}
ROLLOUT_KEYS = {"primary": "primary_images_jpeg", "wrist": "wrist_images_jpeg"}
ROLLOUT_RE = re.compile(r"episode_data--suite=(?P<suite>[a-z0-9_]+)--.*--task=(?P<task>\d+)--ep=(?P<ep>\d+)--success=(?P<succ>True|False)--")


def task_from_filename(path):
    words = os.path.basename(path)[: -len("_demo.hdf5")].split("_")
    command = ""
    for w in words:
        if "SCENE" in w:
            command = ""
            continue
        command += w + " "
    return command.strip()


def check_jpeg(dataset, idx):
    buf = dataset[idx]
    im = Image.open(io.BytesIO(buf.tobytes() if hasattr(buf, "tobytes") else bytes(buf)))
    im.load()
    return im.size


def index_demos(raw, suite, check_jpeg_n):
    files = sorted(glob.glob(os.path.join(raw, "success_only", f"{suite}_regen", "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"no success_only files for {suite} under {raw}")
    records, n_steps = [], 0
    for path in files:
        task = task_from_filename(path)
        stem = os.path.basename(path)[: -len(".hdf5")]
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            for demo in demos:
                g = f[f"data/{demo}"]
                actions = g["actions"][:].astype(np.float32)
                proprio = g["robot_states"][:].astype(np.float32)
                T = actions.shape[0]
                for view, key in DEMO_KEYS.items():
                    if g[key].shape[0] != T:
                        raise ValueError(f"{path}:{demo} {key} has {g[key].shape[0]} frames but {T} actions")
                if proprio.shape != (T, 9) or actions.shape != (T, 7):
                    raise ValueError(f"{path}:{demo} shapes actions {actions.shape} proprio {proprio.shape}")
                for i in range(min(check_jpeg_n, T)):
                    check_jpeg(g[DEMO_KEYS["primary"]], i * (T - 1) // max(1, check_jpeg_n - 1))
                records.append(
                    {
                        "dataset": suite,
                        "suite": suite,
                        "source": "demo",
                        "id": f"{suite}/{stem}/{demo}",
                        "task": task,
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
    return records, n_steps


def index_rollouts(raw, suite, check_jpeg_n):
    files = sorted(glob.glob(os.path.join(raw, "all_episodes", f"episode_data--suite={suite}--*.hdf5")))
    if not files:
        raise FileNotFoundError(f"no all_episodes files for {suite} under {raw}")
    records, n_steps, n_succ = [], 0, 0
    for path in files:
        m = ROLLOUT_RE.search(os.path.basename(path))
        if m is None:
            raise ValueError(f"unparseable rollout file name {path}")
        with h5py.File(path, "r") as f:
            actions = f["actions"][:].astype(np.float32)
            proprio = f["proprio"][:].astype(np.float32)
            T = actions.shape[0]
            for key in ROLLOUT_KEYS.values():
                if f[key].shape[0] != T:
                    raise ValueError(f"{path} {key} has {f[key].shape[0]} frames but {T} actions")
            success = bool(f.attrs["success"])
            if success != (m["succ"] == "True"):
                raise ValueError(f"{path}: attrs.success {success} != file name")
            for i in range(min(check_jpeg_n, T)):
                check_jpeg(f[ROLLOUT_KEYS["primary"]], i * (T - 1) // max(1, check_jpeg_n - 1))
            task = str(f.attrs["task_description"])
        records.append(
            {
                "dataset": suite,
                "suite": suite,
                "source": "rollout",
                "id": f"{suite}/rollout/task{int(m['task']):02d}/ep{int(m['ep']):04d}",
                "task": task,
                "h5": os.path.abspath(path),
                "group": "",
                "keys": ROLLOUT_KEYS,
                "num_steps": int(T),
                "success": success,
                "actions": actions.round(6).tolist(),
                "proprio": proprio.round(6).tolist(),
            }
        )
        n_steps += T
        n_succ += int(success)
    return records, n_steps, n_succ


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    add_root_arg(ap)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--suites", nargs="+", default=list(SUITES))
    ap.add_argument("--check-jpeg", type=int, default=3)
    ap.add_argument("--skip-rollouts", action="store_true")
    args = ap.parse_args()
    root = data_root(args.root)
    raw = args.raw or os.path.join(root, "cosmos_libero_raw")
    out = args.out or os.path.join(root, "cosmos_libero")

    summary = {}
    all_demo_actions, all_demo_proprio = [], []
    for suite in args.suites:
        demos, n_demo_steps = index_demos(raw, suite, args.check_jpeg)
        write_jsonl(demos, os.path.join(out, "jsonl", f"{suite}.jsonl"))
        tasks = sorted({r["task"] for r in demos})
        summary[suite] = {"demos": len(demos), "demo_steps": n_demo_steps, "tasks": len(tasks)}
        for r in demos:
            all_demo_actions.append(np.asarray(r["actions"], dtype=np.float32))
            all_demo_proprio.append(np.asarray(r["proprio"], dtype=np.float32))
        print(f"{suite}: {len(demos)} demos, {n_demo_steps} steps, {len(tasks)} tasks", flush=True)
        if not args.skip_rollouts:
            rollouts, n_roll_steps, n_succ = index_rollouts(raw, suite, args.check_jpeg)
            write_jsonl(rollouts, os.path.join(out, "jsonl", f"{suite}_rollouts.jsonl"))
            missing = {r["task"] for r in rollouts} - set(tasks)
            if missing:
                raise ValueError(f"{suite}: rollout task texts not in demo file names: {sorted(missing)}")
            summary[suite].update({"rollouts": len(rollouts), "rollout_steps": n_roll_steps, "rollout_success": n_succ})
            print(f"{suite}: {len(rollouts)} rollouts ({n_succ} success), {n_roll_steps} steps", flush=True)

    stats_src = os.path.join(raw, "success_only", "dataset_statistics.json")
    stats = json.load(open(stats_src)) if os.path.exists(stats_src) else {}
    acts = np.concatenate(all_demo_actions)
    props = np.concatenate(all_demo_proprio)
    recomputed = {
        "actions_min": acts.min(0).tolist(),
        "actions_max": acts.max(0).tolist(),
        "proprio_min": props.min(0).tolist(),
        "proprio_max": props.max(0).tolist(),
    }
    if stats and len(args.suites) == len(SUITES):
        for k, v in recomputed.items():
            if not np.allclose(v, stats[k], atol=1e-4):
                raise ValueError(f"{k}: recomputed {v} != cosmos dataset_statistics.json {stats[k]}")
        print("min/max match cosmos dataset_statistics.json")
    else:
        stats.update(recomputed)
    stats["source"] = "nvidia/LIBERO-Cosmos-Policy success_only/dataset_statistics.json"
    stats["summary"] = summary
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
