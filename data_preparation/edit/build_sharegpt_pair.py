"""ShareGPT-4o-Image I2I records -> (state image, instruction, target image) pair JSONL.
Images are referenced in place under <root>/sharegpt/image/; nothing is copied.

  python data_preparation/edit/build_sharegpt_pair.py --root data
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import add_root_arg, data_root  # noqa: E402

TEXT_SOURCE = "edit_instruction"


def main():
    ap = argparse.ArgumentParser()
    add_root_arg(ap)
    ap.add_argument("--sharegpt_root", default=None, help="holds image/ and text_and_image_to_image.json")
    ap.add_argument("--json", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset_tag", default="sharegpt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify_open", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = data_root(args.root)
    sroot = (args.sharegpt_root or os.path.join(root, "sharegpt")).rstrip("/")
    out = args.out or os.path.join(root, "jsonl", "pair", "sharegpt.jsonl")
    jpath = args.json or os.path.join(sroot, "text_and_image_to_image.json")
    data = json.load(open(jpath))
    print(f"[sharegpt] loaded {len(data)} I2I records from {jpath}")
    if args.limit:
        data = data[: args.limit]
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def absp(rel):
        return rel if os.path.isabs(rel) else os.path.join(sroot, rel)

    n_write = n_bad_struct = n_empty_prompt = n_missing = 0
    written_targets = []
    with open(out, "w") as fout:
        for i, r in enumerate(data):
            ii, oo = r.get("input_image"), r.get("output_image")
            prompt = (r.get("input_prompt") or "").strip()
            if not (isinstance(ii, list) and len(ii) >= 1 and oo):
                n_bad_struct += 1
                continue
            if not prompt:
                n_empty_prompt += 1
                continue
            state, target = absp(ii[0]), absp(oo)
            if not (os.path.exists(state) and os.path.exists(target)):
                n_missing += 1
                continue
            rec = dict(
                dataset=args.dataset_tag,
                id=f"{args.dataset_tag}/{i:06d}",
                state=state, action=prompt, target=target,
                action_raw=dict(text_source=TEXT_SOURCE, input_resolution=r.get("input_image_resolution"),
                                output_resolution=r.get("output_image_resolution"), src_rel=ii[0], tgt_rel=oo),
            )
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written_targets.append(target)
            n_write += 1
    open_fail = 0
    if args.verify_open and written_targets:
        from PIL import Image

        for p in random.Random(args.seed).sample(written_targets, min(args.verify_open, len(written_targets))):
            try:
                Image.open(p).convert("RGB").load()
            except Exception:
                open_fail += 1
    print(f"[sharegpt] DONE written={n_write} -> {out} | skipped bad_struct={n_bad_struct} empty_prompt={n_empty_prompt} "
          f"missing_file={n_missing} | decode spot-check fail={open_fail}")


if __name__ == "__main__":
    main()
