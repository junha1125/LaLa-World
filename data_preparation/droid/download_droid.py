"""Download the LeRobot DROID release (lerobot/droid_1.0.1), one external camera only (~165 GB).

  python data_preparation/droid/download_droid.py --root data
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import add_root_arg, data_root  # noqa: E402

VIEW = "observation.images.exterior_1_left"


def main():
    ap = argparse.ArgumentParser()
    add_root_arg(ap)
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()
    from huggingface_hub import snapshot_download

    dest = os.path.join(data_root(args.root), "sim_raw", "droid_lerobot")
    snapshot_download("lerobot/droid_1.0.1", repo_type="dataset", local_dir=dest, max_workers=args.max_workers,
                      allow_patterns=["meta/**", "data/**", f"videos/{VIEW}/**"])
    print(dest)


if __name__ == "__main__":
    main()
