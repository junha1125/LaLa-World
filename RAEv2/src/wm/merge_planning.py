"""Merge the per-GPU shards of a planning evaluation (<out>/<env>/<tag>[-ep<offset>]/result.json) into
<out>/<env>/<tag>/merged.json and a table <out>/summary.md.

    PYTHONPATH=src python src/wm/merge_planning.py --out <planning dir>
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag-prefix", default="")
    args = ap.parse_args()

    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.out, "*", "*", "result.json"))):
        env, tag = path.split(os.sep)[-3], path.split(os.sep)[-2]
        base = re.sub(r"-ep\d+$", "", tag)
        if args.tag_prefix and not base.startswith(args.tag_prefix):
            continue
        with open(path) as f:
            groups[(env, base)].append(json.load(f))

    lines = ["| env | tag | episodes | success | 95% CI | end_dist (init) | plan s/ep | DiT fwd/ep |",
             "|---|---|---|---|---|---|---|---|"]
    for (env, base), results in sorted(groups.items()):
        rows, timing = {}, defaultdict(float)
        for r in results:
            for row in r["rows"]:
                rows[row["ep"]] = row
            for k, v in (r.get("wm_timing") or {}).items():
                timing[k] += v
        rows = [rows[k] for k in sorted(rows)]
        n = len(rows)
        merged = {"env": env, "tag": base, "episodes": n, "shards": len(results), "ckpt": results[0].get("ckpt"),
                  "args": results[0].get("args"), "wm_train_seconds": results[0].get("wm_train_seconds"),
                  "plan_time_s_mean": sum(r["plan_time_s"] for r in rows) / max(n, 1),
                  "dit_forwards_per_episode": timing.get("dit_forwards", 0) / max(n, 1), "rows": rows}
        if env == "franka":
            xyz = sum(r["end_dist_xyz"] for r in rows) / max(n, 1)
            floor = sum(r["floor_xyz"] for r in rows) / max(n, 1)
            merged.update({"end_dist_xyz_mean": xyz, "floor_xyz_mean": floor,
                           "action_score": max(0.0, 800 * (0.1 - xyz)), "floor_action_score": max(0.0, 800 * (0.1 - floor)),
                           "probe": next((r.get("probe") for r in results if r.get("probe")), None)})
            succ_txt, ci_txt = f"score {merged['action_score']:.1f}", f"floor {merged['floor_action_score']:.1f}"
            dist_txt = f"xyz {xyz:.4f} (floor {floor:.4f})"
        else:
            k = sum(int(bool(r["success"])) for r in rows)
            lo, hi = wilson(k, n)
            merged.update({"success_rate": k / max(n, 1), "success_ci95": [lo, hi],
                           "end_dist_mean": sum(r["end_dist"] for r in rows) / max(n, 1),
                           "init_dist_mean": sum(r["init_dist"] for r in rows) / max(n, 1)})
            succ_txt, ci_txt = f"{100*k/max(n,1):.1f}% ({k}/{n})", f"[{100*lo:.0f}, {100*hi:.0f}]"
            dist_txt = f"{merged['end_dist_mean']:.2f} ({merged['init_dist_mean']:.2f})"
        out_dir = os.path.join(args.out, env, base)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "merged.json"), "w") as f:
            json.dump(merged, f, indent=1)
        lines.append(f"| {env} | {base} | {n} | {succ_txt} | {ci_txt} | {dist_txt} | {merged['plan_time_s_mean']:.0f} | "
                     f"{merged['dit_forwards_per_episode']:.0f} |")
    table = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
