"""Download OmniEdit-Filtered-1.2M parquet shards (TIGER-Lab/OmniEdit-Filtered-1.2M, 571 shards, 2.98 TB).
The paper used every 4th shard (143 shards, 25%, 301,226 pairs); that is the default.

  python data_preparation/edit/download_omniedit.py --root data            # shards 0, 4, 8, ... (~745 GB)
  python data_preparation/edit/download_omniedit.py --root data --all      # all 571 shards
  python data_preparation/edit/download_omniedit.py --root data --status
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import add_root_arg, data_root  # noqa: E402

REPO_ID = "TIGER-Lab/OmniEdit-Filtered-1.2M"
TOTAL_SHARDS = 571


def shard_rel(i):
    return f"data/train-{i:05d}-of-{TOTAL_SHARDS:05d}.parquet"


def have_shards(local_dir):
    have = set()
    for p in glob.glob(os.path.join(local_dir, "data", "train-*.parquet")):
        if os.path.getsize(p) > 0:
            have.add(int(os.path.basename(p).split("-")[1]))
    return have


def main():
    ap = argparse.ArgumentParser()
    add_root_arg(ap)
    ap.add_argument("--stride", type=int, default=4, help="take shards i with i %% stride == 0 (4 = the paper's subset)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-workers", type=int, default=16)
    args = ap.parse_args()
    local_dir = os.path.join(data_root(args.root), "omniedit_raw")
    want = set(range(TOTAL_SHARDS)) if args.all else {i for i in range(TOTAL_SHARDS) if i % args.stride == 0}
    have = have_shards(local_dir)
    print(f"[omniedit] {local_dir}: have {len(have)} shards, want {len(want)}, missing {len(want - have)}")
    if args.status or not (want - have):
        return
    from huggingface_hub import snapshot_download

    snapshot_download(REPO_ID, repo_type="dataset", local_dir=local_dir, max_workers=args.max_workers,
                      allow_patterns=[shard_rel(i) for i in sorted(want - have)])
    print(f"[omniedit] have {len(have_shards(local_dir))} shards")


if __name__ == "__main__":
    main()
