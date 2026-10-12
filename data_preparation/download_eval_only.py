"""Evaluation-only data from facebook/jepa-wms (not used for training): the RoboCasa demo HDF5 for zero-shot
planning and the Franka trajectories for the DROID action score. Lands in <root>/eval_only/.

  python data_preparation/download_eval_only.py --root data
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import add_root_arg, data_root  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_root_arg(ap)
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()
    from huggingface_hub import snapshot_download

    dest = os.path.join(data_root(args.root), "eval_only")
    snapshot_download("facebook/jepa-wms", repo_type="dataset", local_dir=dest, max_workers=args.max_workers,
                      allow_patterns=["robocasa/combine_all_im256.hdf5", "franka_custom/data/**"])
    print(dest)


if __name__ == "__main__":
    main()
