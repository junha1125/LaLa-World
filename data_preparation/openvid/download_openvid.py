"""Download the released OpenVid frame sequences (junha1125/openvid-frame-sequences-1M: sequences.jsonl +
64 frame tars, 54 GB) into <root>/openvid_raw/ and extract the tars. The change captions
(outputs/captions.jsonl) are downloaded from the same repo when present; otherwise run caption_sequences.py.

  python data_preparation/openvid/download_openvid.py --root data
"""
import argparse
import os
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import add_root_arg, data_root  # noqa: E402

REPO_ID = "junha1125/openvid-frame-sequences-1M"


def main():
    ap = argparse.ArgumentParser()
    add_root_arg(ap)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--delete-tars", action="store_true")
    ap.add_argument("--frames-only", action="store_true", help="skip the captions download")
    args = ap.parse_args()
    dest = os.path.join(data_root(args.root), "openvid_raw")
    from huggingface_hub import snapshot_download

    pats = ["sequences.jsonl", "frames/*.tar", "meta/manifest.csv"] + ([] if args.frames_only else ["outputs/*"])
    snapshot_download(REPO_ID, repo_type="dataset", local_dir=dest, allow_patterns=pats, max_workers=args.max_workers)
    tars = sorted(f for f in os.listdir(os.path.join(dest, "frames")) if f.endswith(".tar"))
    for name in tars:
        tar_path = os.path.join(dest, "frames", name)
        stamp = tar_path + ".extracted"
        if os.path.exists(stamp):
            continue
        print(f"extract {name}", flush=True)
        with tarfile.open(tar_path) as t:
            t.extractall(dest, filter="data")
        open(stamp, "w").close()
        if args.delete_tars:
            os.remove(tar_path)
    caps = os.path.join(dest, "outputs", "captions.jsonl")
    print(f"[openvid] frames extracted under {dest}/frames; captions: {'found' if os.path.exists(caps) else 'MISSING -> run caption_sequences.py'}")


if __name__ == "__main__":
    main()
