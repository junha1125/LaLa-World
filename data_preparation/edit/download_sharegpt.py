"""Download the image-to-image split of ShareGPT-4o-Image (FreedomIntelligence/ShareGPT-4o-Image,
text_and_image_to_image.json + 10 tars, 171 GB) and extract the images to <root>/sharegpt/image/.

  python data_preparation/edit/download_sharegpt.py --root data [--delete-tars]
"""
import argparse
import os
import shutil
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import add_root_arg, data_root  # noqa: E402

REPO_ID = "FreedomIntelligence/ShareGPT-4o-Image"


def main():
    ap = argparse.ArgumentParser()
    add_root_arg(ap)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--delete-tars", action="store_true")
    ap.add_argument("--parts", type=int, nargs="*", default=None, help="tar part indices (default: all 10)")
    args = ap.parse_args()
    root = data_root(args.root)
    raw = os.path.join(root, "sharegpt_raw")
    out = os.path.join(root, "sharegpt")
    from huggingface_hub import snapshot_download

    snapshot_download(REPO_ID, repo_type="dataset", local_dir=raw, max_workers=args.max_workers,
                      allow_patterns=["text_and_image_to_image.json"] + [f"text_and_image_to_image_part_{i}.tar" for i in (args.parts if args.parts is not None else range(10))])
    os.makedirs(out, exist_ok=True)
    shutil.copy2(os.path.join(raw, "text_and_image_to_image.json"), os.path.join(out, "text_and_image_to_image.json"))
    for name in sorted(os.listdir(raw)):
        if not name.endswith(".tar"):
            continue
        tar_path = os.path.join(raw, name)
        stamp = tar_path + ".extracted"
        if os.path.exists(stamp):
            continue
        print(f"extract {name} -> {out}", flush=True)
        with tarfile.open(tar_path) as t:
            t.extractall(out, filter="data")
        open(stamp, "w").close()
        if args.delete_tars:
            os.remove(tar_path)
    n = len(os.listdir(os.path.join(out, "image"))) if os.path.isdir(os.path.join(out, "image")) else 0
    print(f"[sharegpt] {n} images under {out}/image")


if __name__ == "__main__":
    main()
