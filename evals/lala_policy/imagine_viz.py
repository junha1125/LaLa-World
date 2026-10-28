"""Imagined vs simulated future frames of a policy rollout (run_{robocasa,libero}_eval --save-imagine).

A policy query at step t returns the chunk a[t : t+K] and, from the same forward, the latent of the frame at
t+K (the training target: the frame at min(t + chunk_size, T - 1), every view). ImagineRecorder keeps two
imaginations per query and compares both, after the episode, with the frame the simulator reached at t+K:

  policy  the future latent of the policy pass itself (mode 0): the action slots are queries, so this is the
          model's picture of where its own plan leads; with a flow head it does not see the sampled chunk
  wm      the world-model pass (mode 1) of the same state with the chunk just produced as the action input,
          p(s' | s, a) for exactly the actions that get executed (PolicyAgent.act(imagine_wm=True))

With --open-loop = chunk_size (LIBERO's protocol: 16 of 16) the frame at t+K is the result of executing exactly
the imagined chunk. With a shorter open loop (Cosmos RoboCasa: 16 of 32) the rest of the window is re-planned
from a later frame, so the executed actions are not the imagined ones and the match is approximate. A query
whose t+K lies beyond the episode end is compared with the last frame (`clamped`, as the training target is)
and stays out of the aggregates; so does every query from the step on at which the simulator's render breaks
(render_fault; `render_fault_at` in metrics.json and in the task JSON's episode entry). Recording leaves the
rollout as it is: the policy pass and its sampling stream are the ones of a plain act().

Per episode, <task json folder>/imagine/<task>/ep<NN>_<ok|fail>/ :
  q<t>.png      one row per view: [frame t | frame t dec | imagined policy | imagined wm | frame t+K dec | frame t+K]
                (`dec` = the real frame's own latent through the pixel decoder, i.e. the decoder's ceiling)
  sheet.png     the first view's row of every query, top to bottom
  compare.gif   one frame per query: sim frame t+K over imagined wm over imagined policy, every view
  metrics.json  per query and view: err_sq_{policy,wm} = ||pred - z_{t+K}||^2 and id_sq = ||z_t - z_{t+K}||^2
                (their ratio is val.jsonl's latent_rel_l2; < 1 beats copying the current frame), and the PSNR
                against the real t+K pixels of the decoded imaginations, the decoded current frame (identity)
                and the decoded real t+K latent (ceiling) -- all four through the same decoder
Without --decoder-ckpt only metrics.json (its latent part) is written.

  python -m evals.lala_policy.imagine_viz --out <root> [--tag <tag> ...]   # -> <root>/imagine_summary.{md,json}
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

KINDS = ("policy", "wm")
COLUMNS = ("frame t", "frame t (dec)", "imagined t+K: policy", "imagined t+K: wm", "sim t+K (dec)", "sim t+K")
PSNRS = ("policy", "wm", "identity", "ceiling")


def add_arguments(ap):
    ap.add_argument(
        "--save-imagine",
        type=int,
        default=0,
        help="first N episodes of each task (-1 = all): compare every query's imagined t+K frame with the "
        "frame the simulator reaches (evals/lala_policy/imagine_viz.py)",
    )
    ap.add_argument(
        "--decoder-ckpt",
        default=None,
        help="decoder/ checkpoint of the policy's vision encoder: --save-imagine then also saves decoded strips "
        "(without it, latent metrics only)",
    )


def build_decoder(args, policy):
    if not args.save_imagine or not args.decoder_ckpt:
        return None
    from decoder.model import load_decoder

    return load_decoder(args.decoder_ckpt, policy.device)


def save_this_ep(n, i):
    """n < 0 = every episode, else the first n of the episodes this process runs."""
    return n < 0 or i < n


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def _text_bar(width, text, height=22):
    bar = Image.new("RGB", (width, height), "black")
    d = ImageDraw.Draw(bar)
    for size in (14, 12, 10, 8):
        font = _font(size)
        if d.textlength(text, font=font) <= width - 6:
            break
    d.text((3, 3), text, fill="white", font=font)
    return np.asarray(bar)


def _header(side, columns, gutter=0):
    bar = Image.new("RGB", (gutter + side * len(columns), 22), "black")
    d = ImageDraw.Draw(bar)
    for i, name in enumerate(columns):
        d.text((gutter + i * side + 3, 4), name, fill="white", font=_font(12))
    return np.asarray(bar)


def _tag(panel, text):
    im = Image.fromarray(panel)
    d = ImageDraw.Draw(im)
    font = _font(11)
    d.rectangle((0, 0, d.textlength(text, font=font) + 6, 15), fill="black")
    d.text((3, 1), text, fill="white", font=font)
    return np.asarray(im)


def psnr(a, b):
    """uint8 images of one shape -> dB."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10.0 * np.log10(255.0**2 / max(mse, 1e-12)))


def render_fault(images):
    """{view: HxWx3 uint8} -> True for a broken offscreen render. The software renderer of this image breaks now
    and then, from some step of an episode on: a camera returns another camera's image, then stripe noise, and the
    policy keeps acting on it. Mean |d/dx| of a clean RoboCasa / LIBERO frame is 3-12 and of the noise 40-150; two
    clean views differ by a mean |difference| above 20."""
    ims = [np.asarray(im).astype(np.float32) for im in images.values()]
    if max(np.abs(np.diff(a, axis=1)).mean() for a in ims) > 20.0:
        return True
    return any(a.shape == b.shape and np.abs(a - b).mean() < 2.0 for i, a in enumerate(ims) for b in ims[i + 1 :])


class ImagineRecorder:
    """One episode. Steps are the driver's policy-step index: act() replaces policy.act() at every query,
    frame(t, images) is owed whenever wants(t) and once more for the last observation, then finish()."""

    def __init__(self, policy, decoder=None):
        self.policy, self.decoder, self.K = policy, decoder, int(policy.chunk_size)
        self.queries, self.frames, self.wanted = [], {}, set()
        self.fault_at = None  # set by finish(): the first recorded step with a broken render (render_fault)

    def act(self, t, images, proprio, task):
        chunk, extras = self.policy.act(images, proprio, task, return_extras=True, imagine_wm=True)
        self.queries.append(
            {
                "t": int(t),
                "images": images,
                "z_cur": extras["current_latent"],
                "policy": extras["future_latent"],
                "wm": extras["future_latent_wm"],
            }
        )
        self.wanted.add(int(t) + self.K)
        return chunk

    def wants(self, t):
        return t in self.wanted

    def frame(self, t, images):
        self.frames[int(t)] = images

    @torch.no_grad()
    def _decode(self, latents):
        """(B,N,D) -> (B,H,W,3) uint8."""
        dec = self.decoder
        device = next(dec.parameters()).device
        if latents.shape[1] != dec.n_patches:
            raise ValueError(
                f"decoder expects {dec.n_patches} patches, latents have {latents.shape[1]} "
                "-- wrong decoder for this checkpoint's encoder"
            )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            rec = dec(latents.to(device).float()).float().clamp(0, 1)
        return (rec.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)

    def finish(self, steps, out_dir, label):
        """steps = policy steps executed; frame(steps, <last observation>) must have been given."""
        os.makedirs(out_dir, exist_ok=True)
        views = list(self.policy.views)
        V = len(views)
        seen = {**self.frames, **{q["t"]: q["images"] for q in self.queries}}
        self.fault_at = min((t for t, images in seen.items() if render_fault(images)), default=None)
        records, sheet_rows, gif_frames = [], [], []
        for q in self.queries:
            t, tf = q["t"], min(q["t"] + self.K, steps)
            fut = self.frames[tf]
            z_fut = self.policy.encode(fut)  # (V,N,D)
            rec = {"t": t, "t_future": tf, "clamped": q["t"] + self.K > steps, "views": {}}
            # a render stays broken once it breaks: every comparison reaching that step is void
            rec["render_fault"] = self.fault_at is not None and tf >= self.fault_at
            for i, v in enumerate(views):
                rec["views"][v] = {"id_sq": float(((q["z_cur"][i] - z_fut[i]) ** 2).sum())}
                for k in KINDS:
                    rec["views"][v][f"err_sq_{k}"] = float(((q[k][i] - z_fut[i]) ** 2).sum())
            id_sq = max(sum(r["id_sq"] for r in rec["views"].values()), 1e-12)
            rel = {k: sum(r[f"err_sq_{k}"] for r in rec["views"].values()) / id_sq for k in KINDS}
            note = ("BROKEN RENDER | " if rec["render_fault"] else "") + (
                f"rel-L2 policy {rel['policy']:.2f} wm {rel['wm']:.2f}"
            )
            if self.decoder is not None:
                px = self._decode(torch.cat([q["z_cur"], q["policy"], q["wm"], z_fut]))  # (4V,H,W,3)
                side = px.shape[1]
                cur_dec, im_pol, im_wm, fut_dec = (px[j * V : (j + 1) * V] for j in range(4))
                rows = []
                for i, v in enumerate(views):
                    cur_px, fut_px = (
                        np.asarray(Image.fromarray(np.asarray(im[v])).resize((side, side), Image.BILINEAR))
                        for im in (q["images"], fut)
                    )
                    rec["views"][v].update(
                        psnr_policy=psnr(im_pol[i], fut_px),
                        psnr_wm=psnr(im_wm[i], fut_px),
                        psnr_identity=psnr(cur_dec[i], fut_px),
                        psnr_ceiling=psnr(fut_dec[i], fut_px),
                    )
                    rows.append([_tag(cur_px, v), cur_dec[i], im_pol[i], im_wm[i], fut_dec[i], fut_px])
                mean = {p: np.mean([r[f"psnr_{p}"] for r in rec["views"].values()]) for p in PSNRS}
                note += " | PSNR vs sim t+K: " + " ".join(f"{p} {mean[p]:.1f}" for p in PSNRS)
                when = f"t={t} -> t+K={tf}" + (" (episode end, clamped)" if rec["clamped"] else "")
                grid = np.concatenate([np.concatenate(r, axis=1) for r in rows], axis=0)
                bar = _text_bar(grid.shape[1], f"{label} | {when} | {note}")
                Image.fromarray(np.concatenate([_header(side, COLUMNS), grid, bar], axis=0)).save(
                    os.path.join(out_dir, f"q{t:04d}.png")
                )
                gutter = np.zeros((side, 64, 3), dtype=np.uint8)
                sheet_rows.append(np.concatenate([_tag(gutter, f"t={t}"), *rows[0]], axis=1))
                # sim t+K / imagined wm / imagined policy, one row each, the views side by side
                stack = [np.concatenate(list(p), axis=1) for p in ([r[5] for r in rows], im_wm, im_pol)]
                stack.append(_text_bar(stack[0].shape[1], f"{when} | sim / wm / policy"))
                gif_frames.append(np.concatenate(stack, axis=0))
            records.append(rec)
        if sheet_rows:
            side = sheet_rows[0].shape[0]
            sheet = np.concatenate([_header(side, COLUMNS, gutter=64), *sheet_rows], axis=0)
            Image.fromarray(sheet).save(os.path.join(out_dir, "sheet.png"))
            ims = [Image.fromarray(f) for f in gif_frames]
            ims[0].save(
                os.path.join(out_dir, "compare.gif"), save_all=True, append_images=ims[1:], duration=800, loop=0
            )
        meta = {"label": label, "steps": steps, "chunk_size": self.K, "render_fault_at": self.fault_at}
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({**meta, "queries": records}, f, indent=1)
        return records


def aggregate(episodes):
    """[metrics.json dict] -> pooled numbers over the queries that are neither clamped nor reach a broken render
    (rel_l2 = ratio of sums, as val.jsonl)."""
    sums = {f"err_sq_{k}": 0.0 for k in KINDS}
    sums["id_sq"] = 0.0
    psnrs = {p: [] for p in PSNRS}
    n = n_clamped = n_fault = 0
    for ep in episodes:
        for q in ep["queries"]:
            if q.get("render_fault"):
                n_fault += 1
                continue
            if q["clamped"]:
                n_clamped += 1
                continue
            n += 1
            for r in q["views"].values():
                for key in sums:
                    sums[key] += r[key]
                for p in PSNRS:
                    if f"psnr_{p}" in r:
                        psnrs[p].append(r[f"psnr_{p}"])
    out = {"episodes": len(episodes), "queries": n, "clamped": n_clamped}
    out["broken_render_episodes"] = sum(ep.get("render_fault_at") is not None for ep in episodes)
    out["broken_render_queries"] = n_fault
    for k in KINDS:
        out[f"rel_l2_{k}"] = sums[f"err_sq_{k}"] / sums["id_sq"] if sums["id_sq"] > 0 else None
    for p in PSNRS:
        out[f"psnr_{p}"] = float(np.mean(psnrs[p])) if psnrs[p] else None
    return out


def collect(imagine_dir):
    """<...>/imagine -> {"overall": {all / success / failure episodes}, "per_task": {task: numbers}}."""
    per_task, by_outcome = {}, {"ok": [], "fail": []}
    for task_dir in sorted(glob.glob(os.path.join(imagine_dir, "*"))):
        # metrics.json is the last file an episode writes: a folder without it is still being written
        done = sorted(glob.glob(os.path.join(task_dir, "ep*", "metrics.json")))
        eps = {os.path.dirname(f): json.load(open(f)) for f in done}
        if eps:
            per_task[os.path.basename(task_dir)] = aggregate(list(eps.values()))
        for d, m in eps.items():
            by_outcome["ok" if d.endswith("_ok") else "fail"].append(m)
    overall = {
        "all": aggregate(by_outcome["ok"] + by_outcome["fail"]),
        "success episodes": aggregate(by_outcome["ok"]),
        "failure episodes": aggregate(by_outcome["fail"]),
    }
    return {"overall": overall, "per_task": per_task}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the eval root holding {robocasa/<tag>,libero/<tag>/<suite>}/imagine")
    ap.add_argument("--tag", nargs="*", default=None)
    args = ap.parse_args()
    found = glob.glob(os.path.join(args.out, "robocasa", "*", "imagine"))
    found += glob.glob(os.path.join(args.out, "libero", "*", "*", "imagine"))
    cols = ["episodes", "queries", "clamped", "broken_render_episodes", "broken_render_queries"]
    cols += ["rel_l2_policy", "rel_l2_wm"] + [f"psnr_{p}" for p in PSNRS]

    def cells(a):
        return " | ".join(
            "—" if a[c] is None else f"{a[c]:.3f}" if isinstance(a[c], float) else str(a[c]) for c in cols
        )

    summary, lines = {}, []
    for imagine_dir in sorted(found):
        name = os.path.relpath(os.path.dirname(imagine_dir), args.out)  # robocasa/<tag> | libero/<tag>/<suite>
        if args.tag and name.split(os.sep)[1] not in args.tag:
            continue
        summary[name] = collect(imagine_dir)
        lines += [f"## {name}", "", "| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        lines += [f"| **{k}** | {cells(a)} |" for k, a in summary[name]["overall"].items()]
        lines += [f"| {task} | {cells(a)} |" for task, a in summary[name]["per_task"].items()] + [""]
    lines.append(
        "queries = policy queries that are neither clamped (t+K beyond the episode end) nor reach a broken render "
        "(render_fault); rel_l2 = sum ||pred - z_{t+K}||^2 / sum ||z_t - z_{t+K}||^2 over those queries and their "
        "views (< 1 beats copying the current frame); psnr_* = mean dB against the simulator's t+K pixels, all "
        "through the decoder: policy / wm = decoded imagination, identity = decoded current frame, ceiling = decoded "
        "real t+K latent."
    )
    md = "\n".join(lines) + "\n"
    print(md)
    open(os.path.join(args.out, "imagine_summary.md"), "w").write(md)
    json.dump(summary, open(os.path.join(args.out, "imagine_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
