"""OmniEdit parquet shards -> (state image, edit instruction, target image) pair JSONL + extracted frames.

Each parquet row is one independent edit; the longest paraphrase in edited_prompt_list is the action.
Images are extracted once (long side <= 512, JPEG q90) to <root>/omniedit/frames/<shard>/<row>_{src,tgt}.jpg;
re-running skips shards whose frames exist, so a partial download can be extended later.

  python data_preparation/edit/build_omniedit_pair.py --root data --workers 32
"""
import argparse
import glob
import io
import json
import multiprocessing as mp
import os
import sys

import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import add_root_arg, data_root  # noqa: E402

DATASET_TAG = "omniedit"
TEXT_SOURCE = "edit_instruction"
COLS = ["omni_edit_id", "task", "src_img", "edited_img", "edited_prompt_list",
        "width", "height", "sc_score_1", "sc_score_2", "pq_score", "o_score"]


def pick_prompt(prompts, mode):
    prompts = [p for p in (prompts or []) if p and p.strip()]
    if not prompts:
        return None, -1
    idx = 0 if mode == "first" else max(range(len(prompts)), key=lambda k: len(prompts[k]))
    return prompts[idx].strip(), idx


def resize_encode(raw_bytes, long_side, quality):
    im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    w, h = im.size
    if max(w, h) > long_side:
        im.thumbnail((long_side, long_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), im.size


def process_shard(task):
    (shard_path, frames_dir, out_parts_dir, long_side, quality, prompt_mode,
     min_o_score, skip_existing, limit_rows) = task
    stem = os.path.splitext(os.path.basename(shard_path))[0]
    fdir = os.path.join(frames_dir, stem)
    os.makedirs(fdir, exist_ok=True)
    part_path = os.path.join(out_parts_dir, stem + ".jsonl")
    n_write = n_skip_score = n_empty_prompt = n_extract = n_reuse = n_err = 0
    pf = pq.ParquetFile(shard_path)
    row = 0
    with open(part_path, "w") as fout:
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=COLS)
            cols = {c: t.column(c).to_pylist() for c in COLS}
            for k in range(t.num_rows):
                r = row
                row += 1
                if limit_rows and r >= limit_rows:
                    break
                o = cols["o_score"][k]
                if min_o_score and (o is None or o < min_o_score):
                    n_skip_score += 1
                    continue
                action, pidx = pick_prompt(cols["edited_prompt_list"][k], prompt_mode)
                if action is None:
                    n_empty_prompt += 1
                    continue
                s_path = os.path.join(fdir, f"{r:05d}_src.jpg")
                t_path = os.path.join(fdir, f"{r:05d}_tgt.jpg")
                try:
                    if skip_existing and os.path.exists(s_path) and os.path.exists(t_path):
                        n_reuse += 1
                    else:
                        sb, _ = resize_encode(cols["src_img"][k]["bytes"], long_side, quality)
                        tb, _ = resize_encode(cols["edited_img"][k]["bytes"], long_side, quality)
                        with open(s_path, "wb") as f:
                            f.write(sb)
                        with open(t_path, "wb") as f:
                            f.write(tb)
                        n_extract += 1
                except Exception:
                    n_err += 1
                    continue
                rec = dict(
                    dataset=DATASET_TAG,
                    id=f"{DATASET_TAG}/{stem}/{r:05d}",
                    state=s_path, action=action, target=t_path,
                    action_raw=dict(
                        text_source=TEXT_SOURCE,
                        omni_edit_id=cols["omni_edit_id"][k],
                        task=cols["task"][k],
                        prompt_idx=pidx,
                        prompt_list=cols["edited_prompt_list"][k],
                        o_score=o, sc_score_1=cols["sc_score_1"][k],
                        sc_score_2=cols["sc_score_2"][k], pq_score=cols["pq_score"][k],
                        orig_wh=[cols["width"][k], cols["height"][k]],
                    ),
                )
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_write += 1
    return dict(stem=stem, part=part_path, n_write=n_write, n_extract=n_extract,
                n_reuse=n_reuse, n_skip_score=n_skip_score, n_empty_prompt=n_empty_prompt, n_err=n_err)


def main():
    ap = argparse.ArgumentParser()
    add_root_arg(ap)
    ap.add_argument("--parquet_glob", default=None)
    ap.add_argument("--frames_dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--long_side", type=int, default=512)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--prompt_mode", choices=["longest", "first"], default="longest")
    ap.add_argument("--min_o_score", type=float, default=0.0)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false", default=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max_shards", type=int, default=0, help="0 = all (smoke: 1)")
    ap.add_argument("--limit_rows", type=int, default=0, help="0 = all (smoke: rows per shard)")
    args = ap.parse_args()
    root = data_root(args.root)
    parquet_glob = args.parquet_glob or os.path.join(root, "omniedit_raw", "data", "train-*.parquet")
    frames_dir = args.frames_dir or os.path.join(root, "omniedit", "frames")
    out = args.out or os.path.join(root, "jsonl", "pair", "omniedit.jsonl")

    shards = sorted(glob.glob(parquet_glob))
    if args.max_shards:
        shards = shards[: args.max_shards]
    if not shards:
        sys.exit(f"no parquet matched {parquet_glob}")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    parts_dir = out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    print(f"[omniedit] {len(shards)} shards | workers={args.workers} | long_side={args.long_side} q{args.quality} "
          f"| prompt={args.prompt_mode} | skip_existing={args.skip_existing}")
    tasks = [(sp, frames_dir, parts_dir, args.long_side, args.quality, args.prompt_mode,
              args.min_o_score, args.skip_existing, args.limit_rows) for sp in shards]
    results = []
    with mp.Pool(min(args.workers, len(shards))) as pool:
        for i, res in enumerate(pool.imap_unordered(process_shard, tasks)):
            results.append(res)
            print(f"[omniedit] shard {i + 1}/{len(shards)} {res['stem']}: write={res['n_write']} extract={res['n_extract']} "
                  f"reuse={res['n_reuse']} err={res['n_err']}")
    results.sort(key=lambda d: d["stem"])
    tot = dict(n_write=0, n_extract=0, n_reuse=0, n_skip_score=0, n_empty_prompt=0, n_err=0)
    with open(out, "w") as fo:
        for res in results:
            for kk in tot:
                tot[kk] += res[kk]
            with open(res["part"]) as fi:
                for line in fi:
                    fo.write(line)
    print(f"[omniedit] DONE records={tot['n_write']} -> {out} ({os.path.getsize(out) / 1e6:.1f} MB) | "
          f"frames extracted={tot['n_extract']} reused={tot['n_reuse']} decode_err={tot['n_err']}")


if __name__ == "__main__":
    main()
