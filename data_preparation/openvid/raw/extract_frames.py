import argparse, csv, hashlib, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _root import openvid_root
ROOT = openvid_root()
csv.field_size_limit(10**9)

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=os.path.join(ROOT, "meta", "manifest.csv"))
ap.add_argument("--videos",   default=os.path.join(ROOT, "videos"))
ap.add_argument("--out",      default=os.path.join(ROOT, "sequences.jsonl"))
ap.add_argument("--n-frames", type=int,   default=10)
ap.add_argument("--step",     type=float, default=0.5)
ap.add_argument("--width",    type=int,   default=768)
ap.add_argument("--quality",  type=int,   default=3)
ap.add_argument("--workers",  type=int,   default=max(1, os.cpu_count() - 8))
ap.add_argument("--limit",    type=int,   default=None)
A = ap.parse_args()

N, STEP, W, Q = A.n_frames, A.step, A.width, A.quality
WINDOW_SEC = N * STEP          # start-to-start distance between windows (5.0 s)

def shard(seq_id):
    return hashlib.md5(seq_id.encode()).hexdigest()[:2]   # 256 buckets

def rel_dir(seq_id):
    return os.path.join("frames", shard(seq_id), seq_id)

def extract_window(video_path, out_abs, start):
    os.makedirs(out_abs, exist_ok=True)
    have = sorted(f for f in os.listdir(out_abs) if f.endswith(".jpg"))
    if len(have) >= N:
        return "skip"
    for f in have:                                   # partial -> redo cleanly
        os.remove(os.path.join(out_abs, f))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(start), "-i", video_path,
         # eof_action=pass: without it the fps filter drops the final frame on clips
         # whose duration only just covers the window (the N*STEP boundary cases).
         "-vf", f"fps={1/STEP}:eof_action=pass,scale={W}:-2:flags=lanczos",
         "-frames:v", str(N), "-q:v", str(Q),
         "-start_number", "0",                      # ffmpeg defaults to 1 -> would break f00.jpg paths
         os.path.join(out_abs, "f%02d.jpg"), "-y"],
        check=True, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    n = len([f for f in os.listdir(out_abs) if f.endswith(".jpg")])
    return "ok" if n == N else f"short:{n}"

def job(row):
    vp = os.path.join(A.videos, row["video"])
    if not os.path.exists(vp):
        return []
    stem, out = row["video"][:-4], []
    for w in range(int(row["windows"])):
        start  = w * WINDOW_SEC
        seq_id = f"{stem}__w{w}"
        rd     = rel_dir(seq_id)
        od     = os.path.join(ROOT, rd)
        try:
            st = extract_window(vp, od, start)
        except subprocess.CalledProcessError:
            continue
        except Exception:
            continue
        if st.startswith("short"):
            try:
                for f in os.listdir(od): os.remove(os.path.join(od, f))
                os.rmdir(od)
            except Exception: pass
            continue
        out.append({
            "id": seq_id,
            "video": row["video"],
            "frames":     [f"{rd}/f{i:02d}.jpg" for i in range(N)],
            "timestamps": [round(start + i * STEP, 2) for i in range(N)],
            "step_sec": STEP,
            "n_frames": N,
            "video_caption": row["caption"],
            "meta": {k: row[k] for k in ("seconds", "fps", "motion",
                                         "aesthetic", "camera", "hd")},
        })
    return out

rows = list(csv.DictReader(open(A.manifest, newline="")))
rows = [r for r in rows if os.path.exists(os.path.join(A.videos, r["video"]))]
if A.limit:
    rows = rows[:A.limit]
print(f"clips on disk: {len(rows):,}  workers: {A.workers}  -> {A.out}", flush=True)

t0, n, done = time.time(), 0, 0
with open(A.out, "w") as f, ProcessPoolExecutor(A.workers) as ex:
    futs = [ex.submit(job, r) for r in rows]
    for fut in as_completed(futs):
        done += 1
        try:
            recs = fut.result()
        except Exception as e:
            print(f"  [ERR] {e}", flush=True); continue
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
        if done % 500 == 0:
            el = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] clips {done:,}/{len(rows):,}  "
                  f"seqs {n:,}  {done/el:.1f} clip/s  "
                  f"ETA {(len(rows)-done)/(done/el)/60:.1f} min", flush=True)
        f.flush()
print(f"DONE  {n:,} sequences -> {n * (N - 1):,} adjacent frame pairs", flush=True)
