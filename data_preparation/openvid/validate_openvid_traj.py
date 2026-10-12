"""Re-validate jsonl/traj/openvid.jsonl against sequences.jsonl and captions.jsonl (schema, ids, split guard,
frames on disk for a sample).

  python data_preparation/openvid/validate_openvid_traj.py --root data [--tokenizer hf-models/Qwen3.5-0.8B]
"""
import argparse
import heapq
import json
import os
import random
import sys
from collections import defaultdict
from zlib import crc32

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import add_root_arg, data_root  # noqa: E402

LANG_FIELDS = ("actions_lang6", "actions_lang6_no_proprio", "actions_lang4", "actions_lang4_no_proprio")


def main():
    ap = argparse.ArgumentParser()
    add_root_arg(ap)
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--openvid-root", default=None)
    ap.add_argument("--frames-sample", type=int, default=300)
    ap.add_argument("--tok-top", type=int, default=200)
    ap.add_argument("--val-mod", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()
    data = data_root(args.root)
    root = args.openvid_root or os.path.join(data, "openvid_raw")
    jsonl = args.jsonl or os.path.join(data, "jsonl", "traj", "openvid.jsonl")

    src_frames, src_video = {}, {}
    with open(os.path.join(root, "sequences.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            src_frames[r["id"]] = [os.path.join(root, p) for p in r["frames"]]
            src_video[r["id"]] = r["video"]
    src_caps = defaultdict(dict)
    with open(os.path.join(root, "outputs", "captions.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            src_caps[r["sequence_id"]][r["pair_index"]] = " ".join(r["caption"].split())
    print(f"source: sequences {len(src_frames)} | captions {sum(len(v) for v in src_caps.values())}")

    rng = random.Random(args.seed)
    ids, clip_split = set(), defaultdict(set)
    n_rec = n_tr = n_frames = n_val = n_tr_val = 0
    longest, frame_check = [], []
    with open(jsonl) as f:
        for line in f:
            r = json.loads(line)
            n_rec += 1
            rid = r["id"]
            assert rid not in ids, f"duplicate id {rid}"
            ids.add(rid)
            assert r["dataset"] == "openvid" and rid.startswith("openvid_traj/"), rid
            sid = rid[len("openvid_traj/"):]
            F, T = len(r["frames"]), len(r["actions_semantic"])
            assert T == F - 1, (rid, F, T)
            assert r["actions_semantic"] == r["actions_semantic_no_proprio"], rid
            for k in LANG_FIELDS:
                assert r[k] == [], (rid, k)
            raw = r["action_raw"]
            assert raw["net_action"] == [] and raw["proprio"] == [], rid
            assert len(raw["timestamps"]) == F and raw["video"] == src_video[sid], rid
            assert r["frames"] == src_frames[sid], f"{rid}: frame paths differ from sequences.jsonl"
            want = src_caps[sid]
            assert len(want) == T, (rid, len(want), T)
            for u, t in enumerate(r["actions_semantic"]):
                assert t == want[u], f"{rid}/{u}: caption mismatch"
                assert t and "\n" not in t and len(t.split()) <= 70, f"{rid}/{u}: caption format"
            v = crc32(rid.encode("utf-8")) % args.val_mod == 0
            clip_split[raw["video"]].add(v)
            n_val += int(v)
            n_tr_val += T * int(v)
            n_tr += T
            n_frames += F
            for t in r["actions_semantic"]:
                if len(longest) < args.tok_top:
                    heapq.heappush(longest, (len(t), t))
                elif len(t) > longest[0][0]:
                    heapq.heapreplace(longest, (len(t), t))
            if len(frame_check) < args.frames_sample and rng.random() < 0.01:
                frame_check.append((rid, r["frames"]))

    print(f"schema ok: {n_rec} sequences, {n_tr} transitions, {n_frames} frame refs, ids unique, captions match source")
    straddle = [c for c, s in clip_split.items() if len(s) > 1]
    assert not straddle, f"{len(straddle)} clips straddle train/val (e.g. {straddle[:3]})"
    print(f"split guard ok: {len(clip_split)} clips in a single split | val sequences {n_val} | train/val transitions {n_tr - n_tr_val}/{n_tr_val}")
    miss = extra = 0
    for rid, paths in frame_check:
        for p in paths:
            if not os.path.exists(p):
                miss += 1
                print(f"MISSING {p}")
        on_disk = len(os.listdir(os.path.dirname(paths[0])))
        if on_disk != len(paths):
            extra += 1
            print(f"{rid}: dir has {on_disk} files, jsonl refs {len(paths)}")
    assert miss == 0 and extra == 0
    print(f"frames ok: {len(frame_check)} sampled sequences")
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        best = max(((len(tok("[openvid] " + t, add_special_tokens=False)["input_ids"]), t) for _, t in longest))
        print(f"longest semantic: {best[0]} tokens (prefix incl.) | {best[1]}")


if __name__ == "__main__":
    main()
