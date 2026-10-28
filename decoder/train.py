"""Train a pixel decoder for one frozen LaLa-World vision encoder (torchrun DDP).

    python -m torch.distributed.run --standalone --nproc_per_node 8 -m decoder.train --config <config.yaml>

Loss = L1 + lpips_weight * LPIPS(vgg) on images reconstructed from feature-noise-augmented latents
(sigma ~ U(0, noise_tau) x measured feature RMS, the RAEv2 stage-1 recipe without the GAN). AdamW, cosine schedule,
EMA weights, bf16. Relaunching with the same config resumes from <run_dir>/checkpoints/latest.pt.
"""

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml
from PIL import Image, ImageDraw
from torch.nn.parallel import DistributedDataParallel as DDP

from decoder.data import build_train_dataset, build_val_subsets, collate_images, load_sources, make_train_loader
from decoder.encoders import build_frozen_encoder, normalize_pixels
from decoder.model import ViTPixelDecoder


def lr_at(step, total, o):
    if step < o["warmup_steps"]:
        return 1e-6 + (o["lr"] - 1e-6) * step / max(o["warmup_steps"], 1)
    t = (step - o["warmup_steps"]) / max(total - o["warmup_steps"], 1)
    return o["final_lr"] + 0.5 * (o["lr"] - o["final_lr"]) * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1 - decay)


@torch.no_grad()
def encode(enc, x01):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        return enc(normalize_pixels(x01, enc))["patches"].float()


def psnr(rec, tgt):
    mse = ((rec.clamp(0, 1) - tgt) ** 2).mean(dim=(1, 2, 3))
    return -10.0 * torch.log10(mse + 1e-10)


@torch.no_grad()
def run_val(enc, ema, lpips_model, val_subsets, cfg, device, step, images_seen, run_dir):
    """Per-source PSNR/LPIPS on the EMA decoder (clean features) + an [orig; recon] grid PNG."""
    ema.eval()
    n_grid = cfg["log"]["grid_per_source"]
    metrics, bands = {}, []
    for name, subset in val_subsets:
        loader = torch.utils.data.DataLoader(subset, batch_size=16, collate_fn=collate_images, num_workers=4)
        ps, lp, grid_pairs = [], [], None
        for x in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                rec = ema(encode(enc, x)).float()
                lp.append(lpips_model(rec * 2 - 1, x * 2 - 1).flatten())
            ps.append(psnr(rec, x))
            if grid_pairs is None:
                grid_pairs = (x[:n_grid].cpu(), rec[:n_grid].clamp(0, 1).cpu())
        metrics[name] = {
            "psnr": round(float(torch.cat(ps).mean()), 2),
            "lpips": round(float(torch.cat(lp).mean()), 4),
            "n": sum(len(p) for p in ps),
        }
        orig, rec = grid_pairs
        rows = [
            np.concatenate(list((im.permute(1, 2, 0).numpy() * 255).astype(np.uint8) for im in t), axis=1)
            for t in (orig, rec)
        ]
        band = Image.new("RGB", (rows[0].shape[1], 18), "black")
        ImageDraw.Draw(band).text(
            (4, 3), f"{name}  psnr {metrics[name]['psnr']}  lpips {metrics[name]['lpips']}", fill="white"
        )
        bands += [np.asarray(band), *rows]
    Image.fromarray(np.concatenate(bands, axis=0)).save(os.path.join(run_dir, "samples", f"step_{step:07d}.png"))
    row = {
        "step": step,
        "images_seen": images_seen,
        "per_source": metrics,
        "psnr_mean": round(float(np.mean([m["psnr"] for m in metrics.values()])), 2),
        "lpips_mean": round(float(np.mean([m["lpips"] for m in metrics.values()])), 4),
    }
    with open(os.path.join(run_dir, "val.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(
        f"[val] step {step} psnr {row['psnr_mean']} lpips {row['lpips_mean']} "
        + " ".join(f"{k}:{v['psnr']}" for k, v in metrics.items()),
        flush=True,
    )


def save_ckpt(path, decoder, ema, opt, meta):
    tmp = path + ".tmp"
    payload = {"decoder": decoder.state_dict(), "ema": ema.state_dict(), **meta}
    if opt is not None:
        payload["opt"] = opt.state_dict()
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    torch.manual_seed(cfg["seed"] * 1000 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_cudnn_sdp(False)  # the cuDNN SDPA bf16 backward can produce NaN

    run_dir = cfg["run_dir"]
    if rank == 0:
        os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        with open(os.path.join(run_dir, "config_used.yaml"), "w") as f:
            yaml.safe_dump(cfg, f)

    enc = build_frozen_encoder(cfg["vision"], device)
    gh, _ = enc.grid_size
    patch_px = enc.image_size // gh
    dcfg = cfg["decoder"]
    heads = dcfg.get("num_heads") or dcfg["dim"] // 64
    decoder = ViTPixelDecoder(
        enc.embed_dim,
        dcfg["dim"],
        dcfg["depth"],
        heads,
        gh,
        patch_px,
        dcfg.get("mlp_ratio", 4.0),
        dcfg.get("qk_norm", False),
    ).to(device)
    ddp = DDP(decoder, device_ids=[device.index])
    ema = copy.deepcopy(decoder).requires_grad_(False)
    n_params = sum(p.numel() for p in decoder.parameters())

    import lpips as lpips_lib

    lpips_model = lpips_lib.LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)

    o = cfg["optim"]
    # decay only matrix weights; norms, biases and the pos table stay decay-free (standard ViT split)
    decay = [p for n, p in decoder.named_parameters() if p.ndim >= 2 and n != "pos"]
    no_decay = [p for n, p in decoder.named_parameters() if p.ndim < 2 or n == "pos"]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": o["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=o["lr"],
        betas=tuple(o["betas"]),
    )

    src_path = cfg["data"]["sources"]
    if not os.path.isabs(src_path):
        src_path = os.path.join(os.path.dirname(os.path.abspath(args.config)), src_path)
    sources = load_sources(src_path)
    train_ds = build_train_dataset(sources, enc.image_size)
    loader, sampler = make_train_loader(
        train_ds, cfg["data"]["batch_items"], cfg["data"]["num_workers"], rank, world, cfg["seed"]
    )
    imgs_per_step = world * cfg["data"]["batch_items"] * 2
    total_steps = math.ceil(o["total_images"] / imgs_per_step)
    val_subsets = build_val_subsets(sources, enc.image_size, cfg["log"]["val_per_source"]) if rank == 0 else None

    step, images_seen, epoch, feat_rms = 0, 0, 0, None
    latest = os.path.join(run_dir, "checkpoints", "latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu")
        decoder.load_state_dict(ck["decoder"])
        ema.load_state_dict(ck["ema"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        elif rank == 0:  # a step_*.pt snapshot copied to latest.pt: weights only, fresh optimizer
            print("checkpoint has no optimizer state — starting with a fresh optimizer", flush=True)
        step, images_seen, epoch, feat_rms = ck["step"], ck["images_seen"], ck["epoch"], ck["feat_rms"]
        if rank == 0:
            print(f"resumed {latest} at step {step} ({images_seen / 1e6:.1f}M images)", flush=True)

    if rank == 0:
        n_items = len(train_ds)
        print(
            f"decoder {n_params / 1e6:.1f}M params (dim {dcfg['dim']} depth {dcfg['depth']} heads {heads}, "
            f"grid {gh} patch {patch_px}px) | pool {n_items} items = {2 * n_items} images | "
            f"{imgs_per_step} images/step, {total_steps} steps for {o['total_images'] / 1e6:.0f}M images",
            flush=True,
        )
        if not os.path.exists(os.path.join(run_dir, "train.csv")):
            with open(os.path.join(run_dir, "train.csv"), "w") as f:
                f.write("step,images_seen,loss,l1,lpips,lr,imgs_per_s\n")

    tau, ema_decay, w_lpips = dcfg["noise_tau"], o["ema_decay"], o["lpips_weight"]
    log_every, eval_every, save_every = cfg["log"]["log_every"], cfg["log"]["eval_every"], cfg["log"]["save_every"]
    gn_ema, gval = None, 0.0  # running grad-norm scale for the spike gate (identical on every rank)
    t_last, s_last = time.time(), step
    while step < total_steps:
        sampler.set_epoch(epoch)
        for x in loader:
            if step >= total_steps:
                break
            x = x.to(device, non_blocking=True)
            z = encode(enc, x)
            if feat_rms is None:  # measured once, shared across ranks, frozen into every checkpoint
                local = z.std()
                dist.all_reduce(local)
                feat_rms = float(local.item() / world)
                if rank == 0:
                    print(f"feature RMS = {feat_rms:.4f} (noise sigma ~ U(0, {tau}) * rms)", flush=True)
            sigma = tau * torch.rand(z.shape[0], 1, 1, device=device) * feat_rms
            zn = z + sigma * torch.randn_like(z)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, o)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                rec = ddp(zn).float()
                lp = lpips_model(rec * 2 - 1, x * 2 - 1).mean()
            l1 = (rec - x).abs().mean()
            loss = l1 + w_lpips * lp
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # skip non-finite and spike steps (grad norm > 8x its running scale); grads are DDP-synced,
            # so every rank agrees
            gn = torch.nn.utils.clip_grad_norm_(decoder.parameters(), o["clip_grad"])
            ok = bool(torch.isfinite(gn))
            gval = float(gn) if ok else float("inf")
            if ok and (gn_ema is None or gval < 8 * gn_ema):
                opt.step()
                ema_update(ema, decoder, ema_decay)
                gn_ema = gval if gn_ema is None else 0.98 * gn_ema + 0.02 * gval
            elif rank == 0:
                print(f"step {step}: grad norm {gval:.2f} (scale {gn_ema or 0:.3f}), step skipped", flush=True)
            step += 1
            images_seen += imgs_per_step

            if rank == 0 and step % log_every == 0:
                rate = (step - s_last) * imgs_per_step / max(time.time() - t_last, 1e-6)
                t_last, s_last = time.time(), step
                eta_h = (total_steps - step) * imgs_per_step / rate / 3600
                fl, f1, fp = (t.detach().item() for t in (loss, l1, lp))
                print(
                    f"step {step}/{total_steps} ({images_seen / 1e6:.2f}M imgs) loss {fl:.4f} "
                    f"(l1 {f1:.4f} lpips {fp:.4f}) gn {gval:.2f}/{gn_ema or 0:.3f} "
                    f"lr {opt.param_groups[0]['lr']:.2e} {rate:.0f} img/s eta {eta_h:.1f}h",
                    flush=True,
                )
                with open(os.path.join(run_dir, "train.csv"), "a") as f:
                    f.write(
                        f"{step},{images_seen},{fl:.5f},{f1:.5f},{fp:.5f},"
                        f"{opt.param_groups[0]['lr']:.3e},{rate:.0f},{gval:.3f}\n"
                    )
            if step % eval_every == 0 or step == total_steps:
                if rank == 0:
                    run_val(enc, ema, lpips_model, val_subsets, cfg, device, step, images_seen, run_dir)
                dist.barrier()
            if step % save_every == 0 or step == total_steps:
                if rank == 0:
                    meta = {
                        "step": step,
                        "images_seen": images_seen,
                        "epoch": epoch,
                        "feat_rms": feat_rms,
                        "cfg": cfg,
                        "vision": cfg["vision"],
                        "grid": gh,
                        "patch_px": patch_px,
                    }
                    save_ckpt(latest, decoder, ema, opt, meta)
                    if step % cfg["log"]["keep_every"] == 0 or step == total_steps:
                        save_ckpt(
                            os.path.join(run_dir, "checkpoints", f"step_{step:07d}.pt"), decoder, ema, None, meta
                        )
                dist.barrier()
        epoch += 1
    if rank == 0:
        print(f"done: {step} steps, {images_seen / 1e6:.1f}M images", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
