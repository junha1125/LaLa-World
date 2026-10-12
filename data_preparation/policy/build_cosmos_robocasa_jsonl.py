"""Index the Cosmos Policy RoboCasa release as episode-record JSONL (same schema as the LIBERO builder;
three views, the first 7 of the 12 action dims = arm 6-DoF + gripper).

Outputs under --out: jsonl/robocasa.jsonl, jsonl/robocasa_rollouts.jsonl, stats.json

  python data_preparation/policy/build_cosmos_robocasa_jsonl.py --root data
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

ACTION_DIM = 7
DEMO_KEYS = {
    "primary": "obs/robot0_agentview_left_rgb_jpeg",
    "secondary": "obs/robot0_agentview_right_rgb_jpeg",
    "wrist": "obs/robot0_eye_in_hand_rgb_jpeg",
}
DEMO_KEYS_RAW = {k: v[: -len("_jpeg")] for k, v in DEMO_KEYS.items()}
ROLLOUT_KEYS = {"primary": "primary_images_jpeg", "secondary": "secondary_images_jpeg", "wrist": "wrist_images_jpeg"}
ROLLOUT_RE = re.compile(r"episode_data--task=(?P<task>.+?)--(?P<stamp>[\d_-]+)--ep=(?P<ep>\d+)--success=(?P<succ>True|False)--")


def check_jpeg(dataset, idx):
    buf = dataset[idx]
    im = Image.open(io.BytesIO(buf.tobytes() if hasattr(buf, "tobytes") else bytes(buf)))
    im.load()
    return im.size


def task_name_from_path(path, raw_root):
    rel = os.path.relpath(path, raw_root).split(os.sep)
    return rel[2] if rel[0] == "success_only" else rel[1]


def index_demos(raw, check_jpeg_n):
    files = sorted(glob.glob(os.path.join(raw, "success_only", "kitchen_*", "*", "*", "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"no success_only files under {raw}")
    records, n_steps, per_task = [], 0, {}
    for path in files:
        task_name = task_name_from_path(path, raw)
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            for demo in demos:
                g = f[f"data/{demo}"]
                keys = DEMO_KEYS if DEMO_KEYS["primary"] in g else DEMO_KEYS_RAW
                if keys["primary"] not in g:
                    raise KeyError(f"{path}:{demo} has neither {DEMO_KEYS['primary']} nor {DEMO_KEYS_RAW['primary']}")
                actions = g["actions"][:, :ACTION_DIM].astype(np.float32)
                proprio = g["robot_states"][:].astype(np.float32)
                T = actions.shape[0]
                for view, key in keys.items():
                    if g[key].shape[0] != T:
                        raise ValueError(f"{path}:{demo} {key} has {g[key].shape[0]} frames but {T} actions")
                if proprio.shape != (T, 9):
                    raise ValueError(f"{path}:{demo} proprio shape {proprio.shape}")
                if keys is DEMO_KEYS:
                    for i in range(min(check_jpeg_n, T)):
                        check_jpeg(g[keys["primary"]], i * (T - 1) // max(1, check_jpeg_n - 1))
                task = str(g.attrs["task_description"]) if "task_description" in g.attrs else None
                if not task and "ep_meta" in g.attrs:
                    task = json.loads(g.attrs["ep_meta"]).get("lang")
                if not task:
                    raise KeyError(f"{path}:{demo}: no task_description / ep_meta lang (attrs: {list(g.attrs)})")
                records.append(
                    {
                        "dataset": "robocasa",
                        "suite": task_name,
                        "source": "demo",
                        "id": f"robocasa/{task_name}/{os.path.basename(path)[:-5]}/{demo}",
                        "task": task,
                        "h5": os.path.abspath(path),
                        "group": f"data/{demo}",
                        "keys": keys,
                        "num_steps": int(T),
                        "success": True,
                        "actions": actions.round(6).tolist(),
                        "proprio": proprio.round(6).tolist(),
                    }
                )
                n_steps += T
                per_task[task_name] = per_task.get(task_name, 0) + 1
    return records, n_steps, per_task


def index_rollouts(raw, check_jpeg_n):
    files = sorted(glob.glob(os.path.join(raw, "all_episodes", "*", "*", "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"no all_episodes files under {raw}")
    records, n_steps, n_succ, per_task = [], 0, 0, {}
    for path in files:
        task_name = task_name_from_path(path, raw)
        m = ROLLOUT_RE.search(os.path.basename(path))
        if m is None:
            raise ValueError(f"unparseable rollout file name {path}")
        with h5py.File(path, "r") as f:
            actions = f["actions"][:, :ACTION_DIM].astype(np.float32)
            proprio = f["proprio"][:].astype(np.float32)
            T = actions.shape[0]
            for key in ROLLOUT_KEYS.values():
                if f[key].shape[0] != T:
                    raise ValueError(f"{path} {key} has {f[key].shape[0]} frames but {T} actions")
            success = str(f.attrs["success"]).lower() == "true"
            if success != (m["succ"] == "True"):
                raise ValueError(f"{path}: attrs.success {f.attrs['success']!r} != file name")
            for i in range(min(check_jpeg_n, T)):
                check_jpeg(f[ROLLOUT_KEYS["primary"]], i * (T - 1) // max(1, check_jpeg_n - 1))
            task = str(f.attrs["task_description"])
        records.append(
            {
                "dataset": "robocasa",
                "suite": task_name,
                "source": "rollout",
                "id": f"robocasa/{task_name}/rollout/{m['stamp']}/ep{int(m['ep']):04d}",
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
        per_task[task_name] = per_task.get(task_name, 0) + 1
    return records, n_steps, n_succ, per_task


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
    ap.add_argument("--check-jpeg", type=int, default=3)
    ap.add_argument("--skip-rollouts", action="store_true")
    args = ap.parse_args()
    root = data_root(args.root)
    raw = args.raw or os.path.join(root, "cosmos_robocasa_raw")
    out = args.out or os.path.join(root, "cosmos_robocasa")

    demos, n_demo_steps, per_task = index_demos(raw, args.check_jpeg)
    write_jsonl(demos, os.path.join(out, "jsonl", "robocasa.jsonl"))
    print(f"demos: {len(demos)} episodes, {n_demo_steps} steps, {len(per_task)} tasks: {per_task}", flush=True)
    summary = {"demos": len(demos), "demo_steps": n_demo_steps, "tasks": len(per_task), "demos_per_task": per_task}
    if not args.skip_rollouts:
        rollouts, n_roll_steps, n_succ, roll_per_task = index_rollouts(raw, args.check_jpeg)
        write_jsonl(rollouts, os.path.join(out, "jsonl", "robocasa_rollouts.jsonl"))
        print(f"rollouts: {len(rollouts)} episodes ({n_succ} success), {n_roll_steps} steps", flush=True)
        summary.update({"rollouts": len(rollouts), "rollout_steps": n_roll_steps, "rollout_success": n_succ, "rollouts_per_task": roll_per_task})

    stats_src = os.path.join(raw, "success_only", "dataset_statistics.json")
    stats = json.load(open(stats_src)) if os.path.exists(stats_src) else {}
    acts = np.concatenate([np.asarray(r["actions"], dtype=np.float32) for r in demos])
    props = np.concatenate([np.asarray(r["proprio"], dtype=np.float32) for r in demos])
    recomputed = {
        "actions_min": acts.min(0).tolist(),
        "actions_max": acts.max(0).tolist(),
        "proprio_min": props.min(0).tolist(),
        "proprio_max": props.max(0).tolist(),
    }
    if stats:
        for k, v in recomputed.items():
            ref = np.asarray(stats[k], dtype=np.float32)[: len(v)]
            if not np.allclose(v, ref, atol=1e-3):
                print(f"WARNING {k}: recomputed {np.round(v, 4).tolist()} vs cosmos {ref.round(4).tolist()}")
        for k in ("actions_min", "actions_max"):
            stats[k] = list(stats[k])[:ACTION_DIM]
    else:
        stats.update(recomputed)
    stats["recomputed"] = recomputed
    stats["source"] = "nvidia/RoboCasa-Cosmos-Policy success_only/dataset_statistics.json (actions sliced to 7 dims)"
    stats["summary"] = summary
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
