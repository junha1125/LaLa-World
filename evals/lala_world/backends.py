"""World-model backends for the planning runners: LaLa-World (default) and the RAEv2 generative baseline (--wm rae,
RAEv2/src/wm/planning.py, found through $RAE_ROOT or <repo>/RAEv2)."""

import argparse
import os
import sys

from evals.lala_world.wm import PlanningWM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def add_backend_args(p):
    p.add_argument(
        "--wm", default="lala", choices=["lala", "rae"], help="world model: LaLa-World checkpoint | RAEv2 snapshot"
    )
    p.add_argument("--rae-nfe", type=int, default=1, help="rae: Euler steps per predicted frame")
    p.add_argument("--rae-cfg", type=float, default=2.0, help="rae: classifier-free guidance scale on the text")
    p.add_argument("--rae-cost", default="latent", choices=["latent", "pixel"], help="rae: CEM cost space")
    p.add_argument("--rae-ig", type=float, default=1.0, help="rae: internal guidance scale (1.0 = off)")
    p.add_argument("--rae-weights", default="ema", choices=["ema", "model"])
    p.add_argument("--data", default="traj", choices=["traj"], help=argparse.SUPPRESS)


def build_backend(args, spec, dyn, device):
    kw = dict(domain_prefix=args.domain_prefix == "on", text_source=args.text_source, ctxt_window=args.ctxt_window)
    if args.wm == "lala":
        return PlanningWM(args.ckpt, spec, dyn, device, **kw)
    src = os.path.join(os.environ.get("RAE_ROOT", os.path.join(REPO_ROOT, "RAEv2")), "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from wm.planning import RaeWMSimWM

    return RaeWMSimWM(
        args.ckpt,
        spec,
        dyn,
        device,
        nfe=args.rae_nfe,
        cost=args.rae_cost,
        weights=args.rae_weights,
        cfg_scale=args.rae_cfg,
        ig_scale=args.rae_ig,
        **kw,
    )


def run_tag(args, wm, suffix=""):
    ckpt = os.path.abspath(args.ckpt)
    run_name = os.path.basename(os.path.dirname(ckpt))
    if args.wm == "rae":  # snapshots live in <run>/checkpoints/step-XXXX.pt
        run_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
    tag = args.tag or f"{run_name}-e{wm.epoch}{suffix}"
    if args.wm == "rae" and not args.tag:
        tag += (
            f"-nfe{args.rae_nfe}-cfg{args.rae_cfg:g}"
            + (f"-ig{args.rae_ig:g}" if args.rae_ig != 1.0 else "")
            + f"-{args.rae_cost}"
        )
    if args.ep_offset:
        tag += f"-ep{args.ep_offset}"
    return tag
