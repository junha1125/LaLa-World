"""Print the motion-score percentile ladder of the candidate pool.

The pool is everything select_clips.py would keep *except* the motion cut, so the
number printed here is exactly the value to pass as --motion-lo.
"""
import argparse, csv, json, os, re, bisect

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _root import openvid_root
ROOT = openvid_root()
META = os.path.join(ROOT, "meta")
csv.field_size_limit(10**9)
TALKING = re.compile(r"\b(talk|talks|talking|interview|interviews|interviewed|"
                     r"podcast|podcasts)\b", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("--n-frames", type=int, default=10)
ap.add_argument("--step", type=float, default=0.5)
ap.add_argument("--margin", type=float, default=0.25)
ap.add_argument("--motion-hi", type=float, default=60.0)
ap.add_argument("--min-aesthetic", type=float, default=5.0)
ap.add_argument("--keep-talking", action="store_true")
ap.add_argument("--at", type=float, default=None, help="report the percentile of this score")
A = ap.parse_args()
MIN_SEC = (A.n_frames - 1) * A.step + A.margin

idx = {json.loads(l)["clip"] for l in open(os.path.join(META, "part_index.jsonl"))}
mot = []
with open(os.path.join(META, "OpenVid-1M.csv"), newline="") as f:
    for r in csv.DictReader(f):
        v = r["video"]
        if v.startswith(("celebv_", "pixabay_")) or v not in idx:
            continue
        try:
            m, sec, aes = float(r["motion score"]), float(r["seconds"]), float(r["aesthetic score"])
        except ValueError:
            continue
        if sec < MIN_SEC or m > A.motion_hi or aes < A.min_aesthetic:
            continue
        if not A.keep_talking and TALKING.search(r["caption"]):
            continue
        mot.append(m)

mot.sort()
n = len(mot)
print(f"candidate pool (no motion cut): {n:,} clips")
for top in (10, 20, 30, 40, 50, 70, 100):
    print(f"  top {top:>3}%  ->  motion >= {mot[int(n * (1 - top/100))]:.4f}")
if A.at is not None:
    print(f"  motion {A.at} sits at top {100 - bisect.bisect_left(mot, A.at)*100/n:.1f}%")
