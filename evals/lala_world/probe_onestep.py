"""1-step prediction probe on the training JSONL: L2(prediction, target) against the identity floor (copying the
current frame) and the empty-text floor. A checkpoint whose prediction does not beat identity cannot plan.

  python -m evals.lala_world.probe_onestep --ckpt <ckpt> --env wall pusht --val-only --ctxt-windows 1 2 3
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from app.lala_world.data import is_val_id
from evals.lala_world.paths import TRAJ_JSONL
from evals.lala_world.sim_spec import ProprioDynamics, load_spec
from evals.lala_world.wm import PlanningWM


def _transitions(rec):
    raw = rec["action_raw"]
    w = len(rec["frames"]) - len(rec["actions_semantic"])
    return [
        {
            "dataset": rec["dataset"],
            "id": f"{rec['id']}/{u:06d}",
            "u": u,
            "w": w,
            "state": rec["frames"][u],
            "target": rec["frames"][u + w],
            "semantic": rec["actions_semantic"][u],
            "lang6": rec["actions_lang6"][u],
            "lang4": rec["actions_lang4"][u],
            "proprio": raw["proprio"][u],
        }
        for u in range(len(rec["frames"]) - w)
    ]


def load_records(path, n, val_only, ctx_depth=1):
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if val_only:
        recs = [r for r in recs if is_val_id(r["id"])]
    trans = [t for r in recs for t in _transitions(r)]
    if ctx_depth > 1:
        by_id = {t["id"]: t for t in trans}
        trans = [t for t in trans if t["u"] >= (ctx_depth - 1) * t["w"]]
        for t in trans:
            ep = t["id"].rsplit("/", 1)[0]
            t["ctx"] = [by_id[f"{ep}/{t['u'] - j * t['w']:06d}"] for j in range(ctx_depth - 1, 0, -1)]
    if len(trans) > n:
        idx = np.linspace(0, len(trans) - 1, n).round().astype(int)
        trans = [trans[i] for i in idx]
    return trans


def render_body(rec, spec, text_source, proprio_text):
    body = rec[text_source]
    if proprio_text == "off" and spec.proprio_dims:
        body = body[body.index("; ") + 2 :]
    return body


def encode(wm, cache, img_path):
    if img_path not in cache:
        cache[img_path] = wm.encode_frame(np.array(Image.open(img_path).convert("RGB")))
    return cache[img_path]


def run_cell(wm, spec, recs, text_source, proprio_text, cache, batch=256, ctxt=1):
    chains = [(r["ctx"][len(r["ctx"]) - ctxt + 1 :] if ctxt > 1 else []) + [r] for r in recs]

    def body(t):
        b = render_body(t, spec, text_source, proprio_text)
        return spec.prefix + b if wm.domain_prefix else b

    zt = torch.cat([encode(wm, cache, r["target"]) for r in recs])
    z_blocks = [torch.cat([encode(wm, cache, c[j]["state"]) for c in chains]) for j in range(ctxt)]
    txt_blocks = [[body(c[j]) for c in chains] for j in range(ctxt)]
    pred_mse, empty_mse = [], []
    for s in range(0, zt.shape[0], batch):
        e = min(s + batch, zt.shape[0])
        fb = [zb[s:e] for zb in z_blocks]
        if ctxt == 1:
            pred = wm._predict(fb[0], txt_blocks[0][s:e])
            empty = wm._predict(fb[0], [""] * (e - s))
        else:
            pred = wm._predict_ctx(fb, [tb[s:e] for tb in txt_blocks])
            empty = wm._predict_ctx(fb, [[""] * (e - s)] * ctxt)
        pred_mse.append(((pred - zt[s:e]) ** 2).mean(dim=(1, 2)))
        empty_mse.append(((empty - zt[s:e]) ** 2).mean(dim=(1, 2)))
    ident = ((z_blocks[-1] - zt) ** 2).mean(dim=(1, 2))
    return {
        "pred": round(float(torch.cat(pred_mse).mean()), 4),
        "identity": round(float(ident.mean()), 4),
        "empty": round(float(torch.cat(empty_mse).mean()), 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--env", required=True, nargs="+")
    p.add_argument("--n", type=int, default=256, help="transitions per cell")
    p.add_argument("--domain-prefix", default="off", choices=["on", "off"])
    p.add_argument("--jsonl-dir", default=TRAJ_JSONL)
    p.add_argument("--text-source", default="semantic", choices=["semantic", "lang6", "lang4"])
    p.add_argument(
        "--proprio-text",
        default="off",
        choices=["on", "off"],
        help="off: drop the proprio clause (no-proprio checkpoints)",
    )
    p.add_argument("--val-only", action="store_true", help="restrict to the val split (crc32(id) %% 50 == 0)")
    p.add_argument(
        "--ctxt-windows",
        nargs="*",
        type=int,
        default=None,
        help="context window sizes to probe, all scoring the same targets (default: the training context)",
    )
    p.add_argument("--out", default=None, help="append one JSON line per cell")
    args = p.parse_args()
    wm = PlanningWM(
        args.ckpt,
        load_spec(args.env[0]),
        ProprioDynamics(args.env[0], mode="none"),
        torch.device("cuda:0"),
        domain_prefix=args.domain_prefix == "on",
    )
    windows = args.ctxt_windows or [wm.ctxt_window]
    ctx_depth = max(windows)
    run = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    for env in args.env:
        spec = load_spec(env)
        recs = load_records(os.path.join(args.jsonl_dir, f"{spec.name}.jsonl"), args.n, args.val_only, ctx_depth)
        cache = {}
        for ctxt in windows:
            out = run_cell(wm, spec, recs, args.text_source, args.proprio_text, cache, ctxt=ctxt)
            print(
                f"{env} n={len(recs)}{' val-only' if args.val_only else ''} text={args.text_source} ctxt={ctxt}: "
                f"L2(pred,target)={out['pred']}  L2(identity,target)={out['identity']}  "
                f"L2(empty,target)={out['empty']}"
            )
            if args.out:
                with open(args.out, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "ckpt": os.path.abspath(args.ckpt),
                                "run": run,
                                "epoch": wm.epoch,
                                "env": env,
                                "text_source": args.text_source,
                                "val_only": args.val_only,
                                "n": len(recs),
                                "ctxt": ctxt,
                                **out,
                            }
                        )
                        + "\n"
                    )


if __name__ == "__main__":
    main()
