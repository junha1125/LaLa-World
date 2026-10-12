"""Re-validate jsonl/traj/droid.jsonl independently of the builder (schema, ids, totals vs spec,
frames on disk for a sample, byte-identical re-rendering from action_raw).

  python data_preparation/droid/validate_droid_traj.py --root data [--tokenizer hf-models/Qwen3.5-0.8B]
"""
import argparse
import heapq
import json
import os
import sys
from bisect import bisect_right
from zlib import crc32

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_droid_traj as B  # noqa: E402
from common import add_root_arg, data_root  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_root_arg(ap)
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--frames-sample", type=int, default=300)
    ap.add_argument("--parity-n", type=int, default=400)
    ap.add_argument("--tokenizer", default=None, help="optional tokenizer path/id to report the longest text")
    args = ap.parse_args()
    root = data_root(args.root)
    jsonl = args.jsonl or os.path.join(root, "jsonl", "traj", "droid.jsonl")

    spec, bins = B.load_spec_bins()
    std6 = np.array(spec["action_std"])
    rng = np.random.RandomState(0)
    n_ep = n_tr = n_val = frames_total = 0
    ids = set()
    longest = {k: [] for k in ("sem", "l6", "l4")}
    parity_done = 0
    frame_check_eps = []
    with open(jsonl) as f:
        for line in f:
            r = json.loads(line)
            n_ep += 1
            ids.add(r["id"])
            F, T = len(r["frames"]), len(r["actions_semantic"])
            raw = r["action_raw"]
            assert T == F - 1, (r["id"], F, T)
            for k in ("actions_semantic_no_proprio", "actions_lang6", "actions_lang6_no_proprio",
                      "actions_lang4", "actions_lang4_no_proprio"):
                assert len(r[k]) == T, (r["id"], k)
            assert r["actions_semantic"] == r["actions_semantic_no_proprio"], r["id"]
            assert len(raw["net_action"]) == T and len(raw["proprio"]) == T, r["id"]
            assert all(len(a) == 7 for a in raw["net_action"]), r["id"]
            n_tr += T
            frames_total += F
            n_val += crc32(r["id"].encode("utf-8")) % 50 == 0
            for key, arr in (("sem", r["actions_semantic"]), ("l6", r["actions_lang6"]), ("l4", r["actions_lang4"])):
                h = longest[key]
                for t in arr:
                    if len(h) < 200:
                        heapq.heappush(h, (len(t), t))
                    elif len(t) > h[0][0]:
                        heapq.heapreplace(h, (len(t), t))
            if parity_done < args.parity_n:
                for u in rng.choice(T, size=min(3, T), replace=False):
                    a = np.array(raw["net_action"][u])
                    assert B.render_semantic(a) == r["actions_semantic"][u], (r["id"], u, "sem")
                    av = a[:6] / std6
                    for nb, field in ((6, "actions_lang6"), (4, "actions_lang4")):
                        parts = []
                        for dim, x in zip(bins["dims"], av):
                            b = dim[f"bins{nb}"]
                            side = "pos" if x >= 0 else "neg"
                            parts.append(b[f"{side}_phrases"][bisect_right(b[f"{side}_abs_edges"], abs(x))])
                        assert B.LANG_TEMPLATE.format(*parts, B.grip_word(a[6])) == r[field][u], (r["id"], u, field)
                    parity_done += 1
            if len(frame_check_eps) < args.frames_sample and rng.rand() < 0.01:
                frame_check_eps.append((r["frames"], r["id"]))

    assert len(ids) == n_ep, f"duplicate ids: {n_ep - len(ids)}"
    assert n_ep == spec["n_episodes"], (n_ep, spec["n_episodes"])
    assert n_tr == spec["n_records"], (n_tr, spec["n_records"])
    print(f"schema ok: {n_ep} episodes, {n_tr} transitions, {frames_total} frame refs, ids unique")
    print(f"val episodes (crc32%50==0): {n_val}")
    print(f"render parity ok: {parity_done} transitions x 3 fields re-rendered byte-identical")

    miss = extra = 0
    for fpaths, rid in frame_check_eps:
        for p in fpaths:
            if not os.path.exists(p):
                miss += 1
                print(f"MISSING {p}")
        on_disk = len(os.listdir(os.path.dirname(fpaths[0])))
        if on_disk != len(fpaths):
            extra += 1
            print(f"{rid}: dir has {on_disk} files, jsonl refs {len(fpaths)}")
    assert miss == 0 and extra == 0
    print(f"frames ok: {len(frame_check_eps)} sampled episodes, all refs exist, dir counts match")

    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        for key, h in longest.items():
            best = max(((len(tok("[droid] " + t)["input_ids"]), t) for _, t in h), default=(0, ""))
            print(f"longest {key}: {best[0]} tokens (prefix incl.) | {best[1]}")


if __name__ == "__main__":
    main()
