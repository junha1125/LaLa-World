import argparse, collections, csv, json, os, sys, threading, time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hfzip import open_part, make_session, read_member

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _root import openvid_root
ROOT = openvid_root()
csv.field_size_limit(10**9)

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=os.path.join(ROOT, "meta", "manifest.csv"))
ap.add_argument("--budget-gb", type=float, default=None)
ap.add_argument("--out", default=os.path.join(ROOT, "videos"))
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--chunk", type=int, default=150, help="clips per worker task (allows >1 thread per zip part)")
ap.add_argument("--limit", type=int, default=None, help="benchmark: only first N clips")
ap.add_argument("--retries", type=int, default=4)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)

jobs = []
with open(args.manifest, newline="") as f:
    for r in csv.DictReader(f):
        if args.budget_gb and float(r["cum_gb"]) > args.budget_gb:
            break
        jobs.append({"video": r["video"], "member": r["member"],
                     "part": int(r["part"]), "bytes": int(r["bytes"])})
if args.limit:
    jobs = jobs[:args.limit]

total_jobs = len(jobs)
total_bytes = sum(j["bytes"] for j in jobs)
jobs = [j for j in jobs if not os.path.exists(os.path.join(args.out, j["video"]))]
print(f"manifest jobs: {total_jobs:,} ({total_bytes/1e9:.1f} GB) | todo: {len(jobs):,} "
      f"({sum(j['bytes'] for j in jobs)/1e9:.1f} GB) | already on disk: {total_jobs-len(jobs):,}", flush=True)

by_part = collections.defaultdict(list)
for j in jobs:
    by_part[j["part"]].append(j)

# split each part's jobs into chunks -> several threads may work the same part
tasks = []
for p, js in sorted(by_part.items()):
    for k in range(0, len(js), args.chunk):
        tasks.append((p, js[k:k+args.chunk]))
tasks.sort(key=lambda t: -len(t[1]))
print(f"parts: {len(by_part)} | tasks: {len(tasks)} | workers: {args.workers}", flush=True)

LOCK = threading.Lock()
THROTTLE_UNTIL = [0.0]

def note_throttle(e, secs=20):
    """Global brake: when HF starts 429-ing, every thread pauses."""
    if "429" in str(e) or "Too Many Requests" in str(e):
        with LOCK:
            THROTTLE_UNTIL[0] = max(THROTTLE_UNTIL[0], time.time() + secs)

def wait_throttle():
    while True:
        with LOCK:
            t = THROTTLE_UNTIL[0]
        d = t - time.time()
        if d <= 0:
            return
        time.sleep(min(d, 5))
STAT = dict(got=0, failed=0, nbytes=0, t0=time.time())
STOP = threading.Event()

def bump(got=0, failed=0, nbytes=0):
    with LOCK:
        STAT["got"] += got; STAT["failed"] += failed; STAT["nbytes"] += nbytes

def reporter():
    last = 0
    while not STOP.wait(30):
        with LOCK:
            g, fl, nb, t0 = STAT["got"], STAT["failed"], STAT["nbytes"], STAT["t0"]
        el = time.time() - t0
        inst = (nb - last) / 30 / 1e6
        last = nb
        eta = (len(jobs) - g) / (g / el) / 3600 if g else float("nan")
        print(f"[{time.strftime('%H:%M:%S')}] {g:,}/{len(jobs):,} clips  {nb/1e9:.1f} GB  "
              f"avg {nb/el/1e6:.0f} MB/s  now {inst:.0f} MB/s  fail {fl}  ETA {eta:.1f} h", flush=True)

def fetch_task(part, todo):
    s = make_session()
    z = rf = None
    for attempt in range(args.retries):
        try:
            z, rf = open_part(part, s); break
        except Exception as e:
            if attempt == args.retries - 1:
                bump(failed=len(todo))
                return part, 0, len(todo), f"open_part failed: {e}"
            time.sleep(3 * (attempt + 1))
    got = failed = 0
    for j in todo:
        wait_throttle()
        dst = os.path.join(args.out, j["video"])
        if os.path.exists(dst):
            continue
        tmp = dst + f".part{threading.get_ident()}"
        for attempt in range(args.retries):
            try:
                data = read_member(z, rf, j["member"])
                with open(tmp, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, dst)
                got += 1
                bump(got=1, nbytes=len(data))
                break
            except Exception as e:
                note_throttle(e)
                try: os.path.exists(tmp) and os.remove(tmp)
                except Exception: pass
                if attempt == args.retries - 1:
                    failed += 1; bump(failed=1)
                    print(f"  [FAIL] {j['video']} part{part}: {e}", flush=True)
                else:
                    wait_throttle()
                    time.sleep(3 * (attempt + 1))
                    try:
                        z, rf = open_part(part, s)   # re-open on error
                    except Exception:
                        pass
    return part, got, failed, "ok"

th = threading.Thread(target=reporter, daemon=True); th.start()
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futs = [ex.submit(fetch_task, p, js) for p, js in tasks]
    done = 0
    for fut in as_completed(futs):
        done += 1
        try:
            fut.result()
        except Exception as e:
            print(f"  [TASK ERR] {e}", flush=True)
STOP.set()
el = time.time() - STAT["t0"]
print(f"DONE  got={STAT['got']:,}  failed={STAT['failed']:,}  "
      f"{STAT['nbytes']/1e9:.1f} GB in {el/60:.1f} min ({STAT['nbytes']/el/1e6:.0f} MB/s)", flush=True)
