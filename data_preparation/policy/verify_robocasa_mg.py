"""Open each downloaded mg_im HDF5 and report demos/steps/image shapes per task (-> summary.json).
    python data_preparation/policy/verify_robocasa_mg.py --root data
"""
import argparse
import csv
import json
import os
import sys

import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import add_root_arg, data_root  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
add_root_arg(ap)
ap.add_argument("--raw", default=None)
a = ap.parse_args()
raw = a.raw or os.path.join(data_root(a.root), "robocasa_mg_raw")
rows = list(csv.DictReader(open(os.path.join(HERE, "robocasa_mg_urls.tsv")), delimiter="\t"))
summary, total = {}, 0
print("| task | demos | steps | image obs (shape) | states | GB |\n|---|---|---|---|---|---|")
for r in rows:
    p = os.path.join(raw, r["relpath"])
    if not os.path.exists(p):
        print(f"| {r['task']} | (missing) | | | | |")
        continue
    with h5py.File(p, "r") as f:
        d = f["data"]
        demos = list(d.keys())
        g = d[demos[0]]
        imgs = {k: list(g["obs"][k].shape[1:]) for k in g["obs"] if k.endswith("_image")}
        steps = int(d.attrs.get("total", sum(d[k]["actions"].shape[0] for k in demos)))
        info = dict(demos=len(demos), steps=steps, image_obs=imgs, states="states" in g,
                    action_dim=int(g["actions"].shape[1]), size_gb=round(os.path.getsize(p) / 1e9, 1))
    summary[r["task"]] = info
    total += info["demos"]
    print(f"| {r['task']} | {info['demos']} | {info['steps']} | {', '.join(f'{k} {v}' for k, v in imgs.items())} | {info['states']} | {info['size_gb']} |")
summary["_total_demos"] = total
json.dump(summary, open(os.path.join(raw, "summary.json"), "w"), indent=1)
print(f"\ntotal demos {total} over {len(summary) - 1} tasks -> {raw}/summary.json")
