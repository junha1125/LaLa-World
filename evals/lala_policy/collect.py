"""Aggregate run_libero_eval / run_robocasa_eval JSONs into one table (per suite / task, Wilson 95% CI).

  python -m evals.lala_policy.collect --out logs/policy_eval [--tag <tag> ...]
Writes <out>/summary.md and <out>/summary.json; tags default to every tag found under <out>/{libero,robocasa}.
"""

import argparse
import glob
import json
import math
import os

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def collect_libero(root, tag):
    out = {}
    for suite in SUITES:
        files = sorted(glob.glob(os.path.join(root, "libero", tag, suite, "task*.json")))
        if not files:
            continue
        rows = [json.load(open(f)) for f in files]
        n, k = sum(r["n"] for r in rows), sum(r["n_success"] for r in rows)
        out[suite] = {
            "n": n,
            "k": k,
            "rate": k / max(1, n),
            "ci": wilson(k, n),
            "tasks": len(rows),
            "per_task": {r["task"]: r["success_rate"] for r in rows},
        }
    if out:
        rates = [v["rate"] for v in out.values()]
        out["average"] = {"rate": sum(rates) / len(rates), "suites": len(rates)}
    return out


def collect_robocasa(root, tag):
    files = sorted(glob.glob(os.path.join(root, "robocasa", tag, "*.json")))
    files = [f for f in files if not f.endswith("summary.json")]
    if not files:
        return {}
    rows = [json.load(open(f)) for f in files]
    per_task = {r["task"]: {"n": r["n"], "k": r["n_success"], "rate": r["success_rate"]} for r in rows}
    n, k = sum(r["n"] for r in rows), sum(r["n_success"] for r in rows)
    rates = [r["success_rate"] for r in rows]
    return {
        "per_task": per_task,
        "n": n,
        "k": k,
        "rate_pooled": k / max(1, n),
        "ci": wilson(k, n),
        "average": sum(rates) / len(rates),
        "tasks": len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("LALA_POLICY_EVAL_OUT", "logs/policy_eval"))
    ap.add_argument("--tag", nargs="*", default=None)
    args = ap.parse_args()
    tags = args.tag or sorted(
        {
            os.path.basename(p)
            for d in ("libero", "robocasa")
            for p in glob.glob(os.path.join(args.out, d, "*"))
            if os.path.isdir(p)
        }
    )
    summary, lines = {}, []
    lines.append(
        "| tag | " + " | ".join(s.replace("libero_", "") for s in SUITES) + " | LIBERO avg | RoboCasa avg (24 tasks) |"
    )
    lines.append("|---|" + "---|" * (len(SUITES) + 2))
    for tag in tags:
        lib, rc = collect_libero(args.out, tag), collect_robocasa(args.out, tag)
        summary[tag] = {"libero": lib, "robocasa": rc}
        cells = []
        for s in SUITES:
            v = lib.get(s)
            cells.append(
                f"{100 * v['rate']:.1f} [{100 * v['ci'][0]:.0f}-{100 * v['ci'][1]:.0f}] n={v['n']}" if v else "—"
            )
        cells.append(f"{100 * lib['average']['rate']:.1f} ({lib['average']['suites']} suites)" if lib else "—")
        cells.append(f"{100 * rc['average']:.1f} ({rc['tasks']} tasks, n={rc['n']})" if rc else "—")
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    md = (
        "\n".join(lines)
        + "\n\nsuccess % [Wilson 95% CI]; LIBERO avg = mean over suites, RoboCasa avg = mean over tasks.\n"
    )
    print(md)
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, "summary.md"), "w").write(md)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
