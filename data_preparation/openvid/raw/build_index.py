import json, os, sys, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hfzip import open_part
OUT = os.path.join(ROOT, "meta", "part_index.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _root import openvid_root
ROOT = openvid_root()

def index_part(i):
    s = requests.Session()
    for attempt in range(4):
        try:
            z, rf = open_part(i, s)
            return i, [(n, z.getinfo(n).compress_size)
                       for n in z.namelist() if n.endswith(".mp4")], rf.bytes_fetched
        except Exception as e:
            if attempt == 3:
                return i, None, 0
            time.sleep(3 * (attempt + 1))

# resume: skip parts already fully written
done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        for line in f:
            try: done.add(json.loads(line)["part"])
            except Exception: pass
todo = [i for i in range(186) if i not in done]
print(f"already indexed parts: {len(done)}, todo: {len(todo)}", flush=True)

fails = []
with open(OUT, "a") as out, ThreadPoolExecutor(max_workers=10) as ex:
    futs = [ex.submit(index_part, i) for i in todo]
    for fut in as_completed(futs):
        i, members, nb = fut.result()
        if members is None:
            print(f"[FAIL] part{i}", flush=True); fails.append(i); continue
        for m, csz in members:
            out.write(json.dumps({"clip": m.split("/")[-1], "part": i,
                                  "member": m, "bytes": csz}) + "\n")
        out.flush()
        print(f"part{i}: {len(members)} clips ({nb/1e6:.1f} MB read)", flush=True)
print("FAILED_PARTS:", fails, flush=True)
print("INDEX_DONE", flush=True)
