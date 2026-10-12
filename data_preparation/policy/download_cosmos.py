"""Download the Cosmos Policy LIBERO / RoboCasa releases (HDF5 demos + replay rollouts; the *.pkl text
embeddings are skipped).

  python data_preparation/policy/download_cosmos.py --root data --what libero|robocasa|all
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import add_root_arg, data_root  # noqa: E402

REPOS = {"libero": ("nvidia/LIBERO-Cosmos-Policy", "cosmos_libero_raw"),
         "robocasa": ("nvidia/RoboCasa-Cosmos-Policy", "cosmos_robocasa_raw")}


def main():
    ap = argparse.ArgumentParser()
    add_root_arg(ap)
    ap.add_argument("--what", choices=["libero", "robocasa", "all"], default="all")
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()
    from huggingface_hub import snapshot_download

    for key, (repo, sub) in REPOS.items():
        if args.what in (key, "all"):
            dest = os.path.join(data_root(args.root), sub)
            print(f"{repo} -> {dest}", flush=True)
            snapshot_download(repo, repo_type="dataset", local_dir=dest, ignore_patterns=["*.pkl"], max_workers=args.max_workers)


if __name__ == "__main__":
    main()
