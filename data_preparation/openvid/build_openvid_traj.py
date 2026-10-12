"""OpenVid ten-frame sequences + adjacent-pair change captions -> episode-record JSONL (window = 1).

Reads <root>/openvid_raw/sequences.jsonl and <root>/openvid_raw/outputs/captions.jsonl, keeps sequences whose
nine captions all exist, and drops a second window of a clip when it would land in the other train/val split
(the loader splits by crc32(id) % 50 == 0). lang6/lang4 fields are empty: there is no numerical action.

  python data_preparation/openvid/build_openvid_traj.py --root data --check-frames all
"""

import argparse
import json
import os
import random
import sys
from zlib import crc32

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import add_root_arg, data_root  # noqa: E402

WINDOW = 1
DATASET_TAG = "openvid"
TEXT_SOURCE = "openvid_qwen3vl_literal_v3"
CUT_PREFIXES = ("a scene cut changes the view from", "hard cut")
NO_CHANGE_PREFIX = "no discernible change"
MAX_WORDS = 70


def load_sequences(root, limit=0):
    seqs = []
    with open(os.path.join(root, "sequences.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            frames = r["frames"]
            if len(frames) != r["n_frames"] or len(r["timestamps"]) != r["n_frames"]:
                raise ValueError(f"{r['id']}: n_frames/frames/timestamps mismatch")
            if r["n_frames"] < WINDOW + 1:
                raise ValueError(f"{r['id']}: {r['n_frames']} frames, no transition")
            seqs.append((r["id"], r))
            if limit and len(seqs) >= limit:
                break
    return seqs


def load_captions(root, seq_frames):
    caps = {sid: [None] * (len(fr) - WINDOW) for sid, fr in seq_frames.items()}
    n_line = n_used = 0
    with open(os.path.join(root, "outputs", "captions.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n_line += 1
            slot = caps.get(r["sequence_id"])
            if slot is None:
                continue
            u = r["pair_index"]
            if not 0 <= u < len(slot):
                raise ValueError(f"{r['pair_id']}: pair_index {u} out of range")
            if r["before_frame_index"] != u or r["after_frame_index"] != u + WINDOW:
                raise ValueError(f"{r['pair_id']}: frame indices do not match pair_index")
            fr = seq_frames[r["sequence_id"]]
            if r["before_frame"] != fr[u] or r["after_frame"] != fr[u + WINDOW]:
                raise ValueError(f"{r['pair_id']}: frame paths differ from sequences.jsonl")
            text = " ".join(r["caption"].split())
            if not text:
                raise ValueError(f"{r['pair_id']}: empty caption")
            if len(text.split()) > MAX_WORDS:
                raise ValueError(f"{r['pair_id']}: {len(text.split())} words (> {MAX_WORDS})")
            if slot[u] is not None and slot[u] != text:
                raise ValueError(f"{r['pair_id']}: two different captions for one pair")
            slot[u] = text
            n_used += 1
    return caps, n_line, n_used


def main():
    ap = argparse.ArgumentParser()
    add_root_arg(ap)
    ap.add_argument("--openvid-root", default=None, help="default <root>/openvid_raw")
    ap.add_argument("--out", default=None, help="default <root>/jsonl/traj/openvid.jsonl")
    ap.add_argument("--val-mod", type=int, default=50)
    ap.add_argument("--no-split-guard", action="store_true")
    ap.add_argument("--drop-cut-seqs", action="store_true")
    ap.add_argument("--check-frames", choices=("none", "sample", "all"), default="sample")
    ap.add_argument("--frames-sample", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data = data_root(args.root)
    root = (args.openvid_root or os.path.join(data, "openvid_raw")).rstrip("/")
    out = args.out or os.path.join(data, "jsonl", "traj", "openvid.jsonl")

    seqs = load_sequences(root, args.limit)
    seq_frames = {sid: r["frames"] for sid, r in seqs}
    print(f"[openvid] sequences.jsonl: {len(seqs)} sequences")
    caps, n_cap_line, n_cap_used = load_captions(root, seq_frames)
    print(f"[openvid] captions.jsonl: {n_cap_line} lines ({n_cap_used} attached)")

    cand, n_incomplete, n_cut_dropped = [], 0, 0
    for sid, r in seqs:
        texts = caps[sid]
        if any(t is None for t in texts):
            n_incomplete += 1
            continue
        cuts = [u for u, t in enumerate(texts) if t.lower().startswith(CUT_PREFIXES)]
        nochg = [u for u, t in enumerate(texts) if t.lower().startswith(NO_CHANGE_PREFIX)]
        if cuts and args.drop_cut_seqs:
            n_cut_dropped += 1
            continue
        cand.append((sid, r, texts, cuts, nochg))
    print(f"[openvid] complete: {len(cand)} | incomplete dropped {n_incomplete} | scene-cut sequences dropped {n_cut_dropped}")

    def is_val(sid):
        return crc32(f"{DATASET_TAG}_traj/{sid}".encode("utf-8")) % args.val_mod == 0

    n_guard = 0
    if not args.no_split_guard:
        base, kept = {}, []
        for item in cand:
            sid, r = item[0], item[1]
            clip, v = r["video"], is_val(sid)
            if clip not in base:
                base[clip] = v
                kept.append(item)
            elif base[clip] == v:
                kept.append(item)
            else:
                n_guard += 1
        cand = kept
    print(f"[openvid] split guard dropped: {n_guard}")

    if args.check_frames != "none":
        rng = random.Random(args.seed)
        pick = cand if args.check_frames == "all" else rng.sample(cand, min(args.frames_sample, len(cand)))
        miss = 0
        for sid, r, *_ in pick:
            for p in r["frames"]:
                if not os.path.exists(os.path.join(root, p)):
                    miss += 1
                    print(f"MISSING {p}")
        if miss:
            raise SystemExit(f"[openvid] {miss} frames missing")
        print(f"[openvid] frames ok: {len(pick)} sequences checked (--check-frames {args.check_frames})")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    n_val = n_frames = n_tr = n_tr_val = cut_pairs = no_change_pairs = 0
    wc, clips = [], set()
    with open(out, "w") as fout:
        for sid, r, texts, cuts, nochg in cand:
            frames = [os.path.join(root, p) for p in r["frames"]]
            m = r.get("meta", {})
            rec = {
                "dataset": DATASET_TAG,
                "id": f"{DATASET_TAG}_traj/{sid}",
                "frames": frames,
                "actions_semantic": texts,
                "actions_semantic_no_proprio": texts,
                "actions_lang6": [],
                "actions_lang6_no_proprio": [],
                "actions_lang4": [],
                "actions_lang4_no_proprio": [],
                "action_raw": {
                    "text_source": TEXT_SOURCE,
                    "net_action": [],
                    "proprio": [],
                    "video": r["video"],
                    "step_sec": r["step_sec"],
                    "timestamps": r["timestamps"],
                    "video_caption": r.get("video_caption", ""),
                    "scene_cut_idx": cuts,
                    "no_change_idx": nochg,
                    "meta": {
                        "seconds": float(m["seconds"]),
                        "fps": float(m["fps"]),
                        "motion": float(m["motion"]),
                        "aesthetic": float(m["aesthetic"]),
                        "camera": m.get("camera", ""),
                        "hd": int(m.get("hd", 0)),
                    },
                },
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_val += is_val(sid)
            n_tr_val += len(texts) * is_val(sid)
            n_frames += len(frames)
            n_tr += len(texts)
            cut_pairs += len(cuts)
            no_change_pairs += len(nochg)
            clips.add(r["video"])
            wc.extend(len(t.split()) for t in texts)
    wc.sort()
    q = lambda p: wc[int(p * (len(wc) - 1))]  # noqa: E731
    print(f"[openvid] DONE {len(cand)} sequences ({len(clips)} clips) -> {out} ({os.path.getsize(out) / 1e9:.2f} GB)")
    print(f"[openvid] transitions {n_tr} | frame refs {n_frames} | val sequences {n_val} | train/val transitions {n_tr - n_tr_val}/{n_tr_val}")
    print(f"[openvid] caption words p50 {q(.5)} / p90 {q(.9)} / p99 {q(.99)} / max {wc[-1]} | "
          f"scene-cut captions {cut_pairs} | no-change captions {no_change_pairs}")


if __name__ == "__main__":
    main()
