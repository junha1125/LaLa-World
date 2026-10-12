import argparse, collections, json, os, random, statistics

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _root import openvid_root
ROOT = openvid_root()
ap = argparse.ArgumentParser()
ap.add_argument("jsonl", nargs="?", default=os.path.join(ROOT, "sequences.jsonl"))
ap.add_argument("--full", action="store_true", help="stat every frame path (else sample 2000 seqs)")
ap.add_argument("--max-windows", type=int, default=2)
ap.add_argument("--motion-lo", type=float, default=5.27)
A = ap.parse_args()

seqs = nframes = 0
bad, ids = [], set()
per_video = collections.Counter()
nf_seen, step_seen = collections.Counter(), collections.Counter()
motions, sample = [], []

for line in open(A.jsonl):
    d = json.loads(line)
    seqs += 1
    ids.add(d["id"])
    per_video[d["video"]] += 1
    nf_seen[d["n_frames"]] += 1
    step_seen[d["step_sec"]] += 1
    motions.append(float(d["meta"]["motion"]))
    fr = d["frames"]
    nframes += len(fr)
    if len(fr) != d["n_frames"] or len(d["timestamps"]) != len(fr):
        bad.append((d["id"], "len mismatch"))
    ts = d["timestamps"]
    if any(round(ts[i+1] - ts[i], 3) != d["step_sec"] for i in range(len(ts)-1)):
        bad.append((d["id"], "irregular timestamps"))
    if A.full:
        for p in fr:
            if not os.path.exists(os.path.join(ROOT, p)):
                bad.append((d["id"], f"missing {p}")); break
    else:
        sample.append((d["id"], fr))

if not A.full and sample:
    for sid, fr in random.Random(0).sample(sample, min(2000, len(sample))):
        for p in fr:
            if not os.path.exists(os.path.join(ROOT, p)):
                bad.append((sid, f"missing {p}")); break

over = [v for v, c in per_video.items() if c > A.max_windows]
low = [m for m in motions if m < A.motion_lo]

print(f"sequences        : {seqs:,}")
print(f"unique ids       : {len(ids):,}   (dupes: {seqs - len(ids)})")
print(f"source clips     : {len(per_video):,}   (mean {seqs/max(1,len(per_video)):.2f} windows/clip)")
print(f"windows/clip     : {dict(sorted(collections.Counter(per_video.values()).items()))}"
      f"   over cap({A.max_windows}): {len(over)}")
print(f"n_frames         : {dict(nf_seen)}")
print(f"step_sec         : {dict(step_seen)}")
print(f"frames listed    : {nframes:,}")
print(f"adjacent pairs   : {nframes - seqs:,}   <- VLM caption calls")
if motions:
    ms = sorted(motions)
    print(f"motion           : min {ms[0]:.2f}  med {statistics.median(ms):.2f}  "
          f"max {ms[-1]:.2f}   below {A.motion_lo}: {len(low)}")
print(f"path check       : {'FULL' if A.full else f'sampled {min(2000,len(sample))} seqs'} "
      f"-> {len(bad)} problems")
for b in bad[:10]:
    print("   ", b)

fs = []
for dp, _, fn in os.walk(os.path.join(ROOT, "frames")):
    fs += [os.path.getsize(os.path.join(dp, f)) for f in fn if f.endswith(".jpg")]
if fs:
    print(f"jpgs on disk     : {len(fs):,}  ({sum(fs)/1e9:.1f} GB, mean {statistics.mean(fs)/1024:.0f} KB)")
    print(f"orphan jpgs      : {len(fs) - nframes:,}")
vd = os.path.join(ROOT, "videos")
vids = [os.path.getsize(os.path.join(vd, f)) for f in os.listdir(vd) if f.endswith(".mp4")]
print(f"videos on disk   : {len(vids):,}  ({sum(vids)/1e9:.1f} GB)")
