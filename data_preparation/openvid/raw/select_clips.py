import argparse, csv, hashlib, json, os, re

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _root import openvid_root
ROOT = openvid_root()
META = os.path.join(ROOT, "meta")
csv.field_size_limit(10**9)

# Only the "a person is talking at the camera" cases: those clips can carry a high
# motion score (lip/hand movement) while nothing in the scene actually changes.
TALKING = re.compile(r"\b(talk|talks|talking|interview|interviews|interviewed|"
                     r"podcast|podcasts)\b", re.I)

p = argparse.ArgumentParser()
p.add_argument("--n-frames",    type=int,   default=10)
p.add_argument("--step",        type=float, default=0.5)
p.add_argument("--max-windows", type=int,   default=2,
               help="hard cap on sequences taken from one clip")
p.add_argument("--margin",      type=float, default=0.25,
               help="seconds of slack required past the last frame of a window")
p.add_argument("--motion-lo",   type=float, default=5.27)
p.add_argument("--motion-hi",   type=float, default=60.0)
p.add_argument("--min-aesthetic", type=float, default=5.0)
p.add_argument("--keep-talking", action="store_true")
p.add_argument("--order", choices=["efficiency", "shuffle", "quality"], default="efficiency")
p.add_argument("--out", default=os.path.join(META, "manifest.csv"))
args = p.parse_args()

SPAN = (args.n_frames - 1) * args.step      # first frame -> last frame
SLOT = args.n_frames * args.step            # start-to-start distance between windows

def n_windows(sec):
    """How many non-overlapping windows fit, capped at --max-windows."""
    w = 0
    while w < args.max_windows and sec >= w * SLOT + SPAN + args.margin:
        w += 1
    return w

print("loading part index ...", flush=True)
idx = {}
with open(os.path.join(META, "part_index.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        idx[d["clip"]] = (d["part"], d["member"], d["bytes"])
print(f"  {len(idx):,} clips indexed", flush=True)

print("loading OpenVidHD.csv ...", flush=True)
hd = set()
with open(os.path.join(META, "OpenVidHD.csv"), newline="") as f:
    for r in csv.DictReader(f):
        hd.add(r["video"])
print(f"  {len(hd):,} HD clips", flush=True)

print("scanning OpenVid-1M.csv ...", flush=True)
rows = []
stats = dict(total=0, prefix=0, short=0, motion=0, aesth=0, talking=0, noidx=0, bad=0)
with open(os.path.join(META, "OpenVid-1M.csv"), newline="") as f:
    for r in csv.DictReader(f):
        stats["total"] += 1
        try:
            v = r["video"]
            if v.startswith(("celebv_", "pixabay_")):     # 512x512 face crops / 2.67s stock
                stats["prefix"] += 1; continue
            sec, mot = float(r["seconds"]), float(r["motion score"])
            nw = n_windows(sec)
            if nw == 0:
                stats["short"] += 1; continue
            if not (args.motion_lo <= mot <= args.motion_hi):
                stats["motion"] += 1; continue
            if float(r["aesthetic score"]) < args.min_aesthetic:
                stats["aesth"] += 1; continue
            if not args.keep_talking and TALKING.search(r["caption"]):
                stats["talking"] += 1; continue
            if v not in idx:
                stats["noidx"] += 1; continue
            part, member, nbytes = idx[v]
            rows.append(dict(video=v, part=part, member=member, bytes=nbytes,
                             windows=nw, seconds=sec, fps=float(r["fps"]), motion=mot,
                             aesthetic=float(r["aesthetic score"]),
                             camera=r["camera motion"], hd=int(v in hd),
                             caption=r["caption"]))
        except Exception:
            stats["bad"] += 1

print(json.dumps(stats, indent=2), flush=True)
print(f"kept: {len(rows):,} clips", flush=True)

# Sort order == download priority: any prefix of the manifest is a usable dataset.
if args.order == "efficiency":      # most sequences per downloaded GB
    rows.sort(key=lambda d: d["bytes"] / d["windows"])
elif args.order == "shuffle":       # unbiased sample at any cut point
    rows.sort(key=lambda d: hashlib.md5(("openvid" + d["video"]).encode()).hexdigest())
else:
    rows.sort(key=lambda d: -(d["aesthetic"] + 2*d["hd"] - abs(d["motion"] - 12)/20))

cum = 0
fields = ["rank","video","part","member","bytes","cum_gb","windows",
          "seconds","fps","motion","aesthetic","camera","hd","caption"]
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for i, d in enumerate(rows):
        cum += d["bytes"]
        w.writerow({"rank": i, "cum_gb": round(cum/1e9, 4), **d})

tot = sum(d["windows"] for d in rows)
print(f"\nmanifest -> {args.out}", flush=True)
print(f"  {args.n_frames} frames x {args.step}s (span {SPAN}s), <= {args.max_windows} windows/clip, "
      f"motion in [{args.motion_lo}, {args.motion_hi}]", flush=True)
print(f"  order={args.order}  {len(rows):,} clips / {tot:,} sequences / "
      f"{tot*(args.n_frames-1):,} pairs / {cum/1e9:,.1f} GB", flush=True)
for b in (200, 400, 600, 800, 1000):
    c = s = 0; acc = 0
    for d in rows:
        acc += d["bytes"]
        if acc/1e9 > b: break
        c += 1; s += d["windows"]
    print(f"  budget {b:>5} GB -> {c:>7,} clips / {s:>7,} seq / {s*(args.n_frames-1):>9,} pairs", flush=True)
