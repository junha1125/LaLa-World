"""Download the simulation datasets into <root>/sim_raw/: Push-T, PointMaze, Wall and Meta-World from
facebook/jepa-wms (Hugging Face), Rope and Granular from the DINO-WM OSF release.

  python data_preparation/download_sim.py --root data [--env wall point_maze pusht metaworld rope granular]
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import add_root_arg, data_root  # noqa: E402

HF_REPO = "facebook/jepa-wms"
HF = {  # env -> (allow pattern, directory created under sim_raw/)
    "pusht": ("pusht/pusht_noise.zip", "pusht_noise"),
    "point_maze": ("point_maze/point_maze.zip", "point_maze"),
    "wall": ("wall/wall_single.zip", "wall_single"),
    "metaworld": ("metaworld/data/*", "metaworld/data"),
}
OSF = "https://osf.io/download/{id}/?view_only=a56a296ce3b24cceaf408383a175ce28"
OSF_DEFORMABLE = [("w9qry", "deformable.zip"), ("vfj7m", "deformable.z01"), ("sdzyx", "deformable.z02"), ("xs426", "deformable.z03")]
ENVS = ("wall", "point_maze", "pusht", "metaworld", "rope", "granular")


def main():
    ap = argparse.ArgumentParser()
    add_root_arg(ap)
    ap.add_argument("--env", nargs="+", default=list(ENVS), choices=ENVS)
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()
    raw = os.path.join(data_root(args.root), "sim_raw")
    os.makedirs(raw, exist_ok=True)
    want = set(args.env)

    need = [e for e in want if e in HF and not os.path.isdir(os.path.join(raw, HF[e][1]))]
    if need:
        from huggingface_hub import snapshot_download

        hf = os.path.join(raw, "_hf")
        snapshot_download(HF_REPO, repo_type="dataset", local_dir=hf, allow_patterns=[HF[e][0] for e in need], max_workers=8)
        for e in need:
            pat, out_dir = HF[e]
            if pat.endswith(".zip"):
                print(f"unzip {pat} -> {raw}/{out_dir}", flush=True)
                with zipfile.ZipFile(os.path.join(hf, pat)) as z:
                    z.extractall(raw)
            else:
                os.makedirs(os.path.join(raw, "metaworld"), exist_ok=True)
                shutil.move(os.path.join(hf, "metaworld", "data"), os.path.join(raw, out_dir))
        if not args.keep_archives:
            shutil.rmtree(hf, ignore_errors=True)

    if want & {"rope", "granular"} and not os.path.isdir(os.path.join(raw, "deformable", "rope")):
        dl = os.path.join(raw, "_osf")
        os.makedirs(dl, exist_ok=True)
        for fid, name in OSF_DEFORMABLE:
            url, dest = OSF.format(id=fid), os.path.join(dl, name)
            size = int(requests.head(url, allow_redirects=True, timeout=60).headers.get("Content-Length", -1))
            if os.path.exists(dest) and os.path.getsize(dest) == size:
                print(f"have {name}", flush=True)
                continue
            print(f"download {name} ({size / 1e9:.1f} GB)", flush=True)
            subprocess.run(["curl", "-L", "-C", "-", "--retry", "10", "--retry-delay", "15", "-o", dest, url], check=True)
        print("merge split zip + unzip (needs zip/unzip)", flush=True)
        subprocess.run(["zip", "-q", "-s-", "deformable.zip", "-O", "deformable_full.zip"], cwd=dl, check=True)
        subprocess.run(["unzip", "-q", "deformable_full.zip", "-d", raw], cwd=dl, check=True)
        if not args.keep_archives:
            shutil.rmtree(dl, ignore_errors=True)

    for e in sorted(want):
        d = HF[e][1] if e in HF else f"deformable/{e}"
        print(f"{e:11s} {'ok' if os.path.isdir(os.path.join(raw, d)) else 'MISSING'}  {raw}/{d}")


if __name__ == "__main__":
    main()
