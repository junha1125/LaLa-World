"""LaLa-World training (config `app: lala_world`).

Frozen vision encoder (its patch latents are both input and target), trainable connector + Qwen3.5 backbone +
query token + out_head. Run from the repository root:
    python -m app.main --fname configs/lala_world/base.yaml --devices cuda:0 ... cuda:7
"""

import json
import os
import re
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from app.lala_world.data import TrajCollator, build_concat_dataset
from app.lala_world.losses import build_loss
from app.lala_world.models import LaLaWorld, build_llm_backbone, build_vision_encoder
from app.lala_world.telemetry import telemetry_logs
from src.utils.logging import CSVLogger, get_logger
from src.utils.schedulers import WarmupCosineSchedule

logger = get_logger(__name__)

CKPT_PREFIX = "lala"
_SNAPSHOT_RE = re.compile(rf"{CKPT_PREFIX}-e(\d+)\.pth\.tar")


def save_rolling_checkpoint(folder, ckpt, keep_last):
    """Write <prefix>-e<epoch>.pth.tar, point <prefix>-latest.pth.tar at it and prune old snapshots
    (keep_last <= 0 keeps all)."""
    path = os.path.join(folder, f"{CKPT_PREFIX}-e{ckpt['epoch']}.pth.tar")
    torch.save(ckpt, path)
    latest = os.path.join(folder, f"{CKPT_PREFIX}-latest.pth.tar")
    tmp = latest + ".tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(os.path.basename(path), tmp)
    os.replace(tmp, latest)
    if keep_last and keep_last > 0:
        snapshots = sorted((int(m.group(1)), m.group(0)) for m in map(_SNAPSHOT_RE.fullmatch, os.listdir(folder)) if m)
        for _, name in snapshots[:-keep_last]:
            os.remove(os.path.join(folder, name))
    return path


class ModelWithLoss(nn.Module):
    def __init__(self, predictor, loss_fn):
        super().__init__()
        self.predictor = predictor
        self.loss_fn = loss_fn

    def forward(self, vis_patches, input_ids, text_mask, target, ctx_len=None):
        pred = self.predictor(vis_patches, input_ids, text_mask, ctx_len=ctx_len)
        loss, logs = self.loss_fn(pred, target.to(pred.dtype))
        return loss, logs, pred


@torch.no_grad()
def encode_ctx(frames, ctx_len, target_px, encoder):
    """frames (B,C,3,H,W) zero beyond ctx_len -> ctx_feats (B,C,N,D) (zero beyond ctx_len), target_feats (B,N,D)."""
    B, C = frames.shape[:2]
    valid = (torch.arange(C, device=frames.device)[None] < ctx_len[:, None]).reshape(-1)
    ctx_px = frames.reshape(B * C, *frames.shape[2:])[valid]
    feats = encoder(torch.cat([ctx_px, target_px], dim=0))["patches"]
    ctx_flat, target_feats = feats[: ctx_px.shape[0]], feats[ctx_px.shape[0] :]
    ctx_feats = torch.zeros(B * C, *ctx_flat.shape[1:], dtype=ctx_flat.dtype, device=ctx_flat.device)
    ctx_feats[valid] = ctx_flat
    return ctx_feats.reshape(B, C, *ctx_flat.shape[1:]), target_feats


def last_ctx(ctx_feats, ctx_len):
    return ctx_feats[torch.arange(ctx_feats.shape[0], device=ctx_feats.device), ctx_len - 1]


def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def init_wandb(cfgs_logging, folder, config, rank):
    if not (cfgs_logging.get("use_wandb", False) and rank == 0):
        return None
    import wandb

    run_id_file = os.path.join(folder, "wandb_run_id.txt")
    if os.path.exists(run_id_file):
        with open(run_id_file) as f:
            run_id = f.read().strip()
        wandb.init(project=cfgs_logging["project"], id=run_id, resume="allow", dir=folder, config=config)
    else:
        wandb.init(project=cfgs_logging["project"], dir=folder, config=config)
        with open(run_id_file, "w") as f:
            f.write(wandb.run.id)
    wandb.run.name = cfgs_logging.get("run_name") or os.path.basename(folder)
    return wandb


def all_reduce_min(flag, world_size):
    if world_size > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag)


def main(args, resume_preempt=False):
    cfgs_meta, cfgs_data, cfgs_model = args["meta"], args["data"], args["model"]
    cfgs_opt, cfgs_logging = args["optimization"], args.get("logging", {})
    folder = args["folder"]
    os.makedirs(folder, exist_ok=True)
    quick_debug = cfgs_meta.get("quick_debug", False)
    seed = cfgs_meta.get("seed", 234)
    torch.manual_seed(seed)
    if dist.is_available() and dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and cfgs_meta.get("dtype", "bfloat16") == "bfloat16"

    # ------------------------------------------------------------------ model
    encoder = build_vision_encoder(cfgs_model["vision"]).to(device)
    backbone = build_llm_backbone(cfgs_model["llm"])
    predictor = LaLaWorld(backbone, vis_dim=encoder.embed_dim, target_dim=encoder.embed_dim, grid_hw=encoder.grid_size)
    model = ModelWithLoss(predictor, build_loss(args.get("loss", {}))).to(device)
    ddp_model = DistributedDataParallel(model, static_graph=True) if world_size > 1 else model
    inner = ddp_model.module if world_size > 1 else model
    logger.info(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")

    # ------------------------------------------------------------------ data
    max_text_tokens = cfgs_data.get("max_text_tokens", 64)
    collator = TrajCollator(backbone.tokenizer, max_text_tokens, cfgs_data.get("text_dropout_p", 0.0))
    val_collator = TrajCollator(backbone.tokenizer, max_text_tokens)
    if cfgs_data.get("mixing", "concat") != "concat":
        raise ValueError("data.mixing must be concat")
    batch_size = cfgs_data.get("batch_size", 32)
    grad_accum = max(1, int(cfgs_opt.get("grad_accum_steps") or 1))
    samples_per_epoch = cfgs_opt.get("samples_per_epoch")
    requested_ipe = max(1, samples_per_epoch // (batch_size * world_size * grad_accum)) if samples_per_epoch else None

    def make_loader(split, drop_last):
        num_workers = cfgs_data.get("num_workers", 8)
        dataset = build_concat_dataset(
            cfgs_data["sources"], split, encoder.image_size, encoder.image_mean, encoder.image_std
        )
        sampler = (
            DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
            if world_size > 1
            else None
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            generator=torch.Generator().manual_seed(seed),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=cfgs_data.get("pin_mem", True) and device.type == "cuda",
            drop_last=drop_last,
            collate_fn=collator if split == "train" else val_collator,
            persistent_workers=num_workers > 0,
        )

    train_loader = make_loader("train", drop_last=True)
    val_loader = make_loader("val", drop_last=False)
    ipe = len(train_loader) // grad_accum
    if ipe == 0:
        raise ValueError(f"data provides {len(train_loader)} batches/epoch, fewer than grad_accum_steps={grad_accum}")
    if requested_ipe and requested_ipe < ipe:
        ipe = requested_ipe
    num_epochs = cfgs_opt.get("epochs", 20)
    if quick_debug:
        ipe, num_epochs = min(ipe, 4), 1
    total_steps = ipe * num_epochs
    micro_ipe = ipe * grad_accum
    logger.info(
        f"steps/epoch {ipe} (world {world_size}, bs/gpu {batch_size}, accum {grad_accum} "
        f"-> global batch {batch_size * world_size * grad_accum})"
    )

    # ------------------------------------------------------------------ optim
    optimizer = torch.optim.AdamW(
        param_groups(model, cfgs_opt.get("weight_decay", 0.05)),
        lr=cfgs_opt.get("lr", 2e-4),
        betas=tuple(cfgs_opt.get("betas", (0.9, 0.98))),
    )
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=cfgs_opt.get("warmup_steps", 1000),
        start_lr=cfgs_opt.get("start_lr", 1e-5),
        ref_lr=cfgs_opt.get("lr", 2e-4),
        final_lr=cfgs_opt.get("final_lr", 1e-6),
        T_max=total_steps,
    )
    clip_grad = cfgs_opt.get("clip_grad", 1.0)
    clip_params = [p for p in model.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ resume
    start_epoch, train_seconds = 0, 0.0
    read_path = cfgs_meta.get("read_checkpoint") or os.path.join(folder, f"{CKPT_PREFIX}-latest.pth.tar")
    if (cfgs_meta.get("load_checkpoint", False) or resume_preempt) and os.path.exists(read_path):
        ckpt = torch.load(read_path, map_location="cpu", weights_only=False)
        inner.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler._step = ckpt["sched_step"]
        start_epoch = ckpt["epoch"]
        train_seconds = float(ckpt.get("train_seconds", 0.0))
        logger.info(f"resumed from {read_path} at epoch {start_epoch}")

    wandb = init_wandb(cfgs_logging, folder, args, rank)
    csv_logger = None
    if rank == 0:
        csv_logger = CSVLogger(
            os.path.join(folder, "train.csv"),
            ("%d", "epoch"),
            ("%d", "itr"),
            ("%.5f", "loss"),
            ("%.5f", "rel_l2"),
            ("%.2e", "lr"),
            ("%.1f", "imgs_per_sec"),
            ("%.1f", "train_min"),
        )

    # ------------------------------------------------------------------ loop
    global_step = start_epoch * ipe
    nan_streak = grad_streak = 0
    final_loss = float("nan")
    rel_acc = torch.zeros(2, dtype=torch.float64, device=device)
    max_train_minutes = cfgs_meta.get("max_train_minutes") or 0
    log_freq = cfgs_meta.get("log_freq", 50)
    stop = False
    for epoch in range(start_epoch, num_epochs):
        ddp_model.train()
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        last_log_time, last_log_step = time.time(), global_step
        for itr, batch in enumerate(train_loader):
            if itr >= micro_ipe:
                break
            t_step = time.time()
            micro, opt_itr = itr % grad_accum, itr // grad_accum
            frames = batch["frames"].to(device, non_blocking=True)
            ctx_len = batch["ctx_len"].to(device, non_blocking=True)
            target_px = batch["target"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            text_mask = batch["text_mask"].to(device, non_blocking=True)
            if micro == 0:
                lr = scheduler.step()
                window_ok, window_loss = True, 0.0
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                ctx_feats, target_feats = encode_ctx(frames, ctx_len, target_px, encoder)
                loss, loss_logs, pred = ddp_model(ctx_feats, input_ids, text_mask, target_feats, ctx_len=ctx_len)
            # every rank must agree on skipping a step, otherwise DDP desynchronises
            if not all_reduce_min(torch.isfinite(loss.detach()).int(), world_size):
                nan_streak += 1
                logger.warning(f"non-finite loss at step {global_step} (streak {nan_streak})")
                if nan_streak >= 3:
                    raise FloatingPointError("3 consecutive non-finite losses, aborting")
                window_ok = False
            else:
                nan_streak = 0
                window_loss += loss.detach().item()
                with torch.no_grad():
                    rel_acc[0] += ((pred.detach().float() - target_feats.float()) ** 2).sum()
                    rel_acc[1] += ((last_ctx(ctx_feats, ctx_len).float() - target_feats.float()) ** 2).sum()
            (loss / grad_accum if grad_accum > 1 else loss).backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_seconds += time.time() - t_step
            if micro < grad_accum - 1:
                continue
            t_step = time.time()
            if window_ok:
                grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, clip_grad or float("inf"))
                if not all_reduce_min(torch.isfinite(grad_norm).int(), world_size):
                    grad_streak += 1
                    logger.warning(f"non-finite grads at step {global_step} (streak {grad_streak}); step dropped")
                    if grad_streak >= 3:
                        raise FloatingPointError("3 consecutive non-finite grad steps, aborting")
                    window_ok = False
                else:
                    grad_streak = 0
                    optimizer.step()
                    final_loss = window_loss / grad_accum
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_seconds += time.time() - t_step
            if max_train_minutes and opt_itr % log_freq == 0:
                stop_flag = torch.tensor(int(train_seconds / 60 >= max_train_minutes), device=device)
                if world_size > 1:
                    dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
                stop = bool(stop_flag.item())
            if window_ok and opt_itr % log_freq == 0:
                n_imgs = frames.shape[0] * grad_accum * world_size * max(1, global_step - last_log_step)
                imgs_per_sec = n_imgs / max(1e-6, time.time() - last_log_time)
                last_log_time, last_log_step = time.time(), global_step
                if world_size > 1:
                    dist.all_reduce(rel_acc)
                rel_l2 = (rel_acc[0] / rel_acc[1].clamp_min(1e-12)).item()
                rel_acc.zero_()
                logs = {
                    "train/loss": final_loss,
                    "train/rel_l2": rel_l2,
                    "train/lr": lr,
                    "train/epoch": epoch,
                    "train/grad_norm": float(grad_norm),
                    "train/train_min": train_seconds / 60,
                }
                logs.update({f"train/{k}": float(v) for k, v in loss_logs.items()})
                logs.update(telemetry_logs(pred, target_feats, last_ctx(ctx_feats, ctx_len), inner.loss_fn))
                if rank == 0:
                    logger.info(
                        f"[e{epoch} i{opt_itr}/{ipe}] loss {final_loss:.4f} rel_l2 {rel_l2:.4f} lr {lr:.2e} "
                        f"trained {train_seconds / 60:.1f} min"
                    )
                    csv_logger.log(epoch, opt_itr, final_loss, rel_l2, lr, imgs_per_sec, train_seconds / 60)
                    if wandb:
                        wandb.log(logs, step=global_step)
            if stop:
                logger.info(f"max_train_minutes {max_train_minutes} reached; stopping")
                break

        # -------------------------------------------------------------- val
        if (epoch + 1) % cfgs_meta.get("eval_freq", 1) == 0 or epoch == num_epochs - 1 or stop:
            val_logs = evaluate(
                ddp_model,
                inner,
                encoder,
                val_loader,
                collator,
                device,
                use_amp,
                quick_debug,
                cfgs_data.get("val_groups"),
            )
            if rank == 0:
                logger.info(f"[e{epoch}] " + " ".join(f"{k} {v:.4f}" for k, v in val_logs.items()))
                with open(os.path.join(folder, "val.jsonl"), "a") as f:
                    f.write(
                        json.dumps(
                            {"epoch": epoch + 1, "step": global_step, "train_min": train_seconds / 60, **val_logs}
                        )
                        + "\n"
                    )
                if wandb:
                    wandb.log({f"val/{k}": v for k, v in val_logs.items()}, step=global_step)

        # -------------------------------------------------------------- save
        if rank == 0 and ((epoch + 1) % cfgs_meta.get("save_every_freq", 1) == 0 or epoch == num_epochs - 1 or stop):
            ckpt = {
                "epoch": epoch + 1,
                "model": inner.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sched_step": scheduler._step,
                "config": args,
                "train_seconds": train_seconds,
            }
            logger.info(
                f"saved checkpoint to {save_rolling_checkpoint(folder, ckpt, cfgs_meta.get('save_keep_last', -1))}"
            )
        if stop:
            break
    if wandb:
        wandb.finish()
    return {"final_train_loss": final_loss, "epochs_run": num_epochs - start_epoch, "ipe": ipe}


@torch.no_grad()
def evaluate(
    ddp_model, inner, encoder, val_loader, collator, device, use_amp, quick_debug, val_groups=None, max_batches=50
):
    """Val loss, loss with every action text blanked (text_dropout_delta = empty - real > 0 means the text is used)
    and rel_l2 (prediction error over the error of copying the last context frame), overall and per data.val_groups."""
    ddp_model.eval()
    tokenizer = collator.tokenizer
    group_of = {name: g for g, names in (val_groups or {}).items() for name in names}
    keys = ["total"] + sorted((val_groups or {}).keys()) + ["other"]
    acc = {k: [0.0, 0.0, 0.0, 0.0, 0.0] for k in keys}  # loss, loss_empty, n, sq_pred, sq_identity
    ds_acc = {}
    if quick_debug:
        max_batches = 2

    def add(key, loss_real, loss_empty, n, sq_pred, sq_id):
        a = acc[key]
        a[0] += float(loss_real) * n
        a[1] += float(loss_empty) * n
        a[2] += n
        a[3] += float(sq_pred)
        a[4] += float(sq_id)

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        frames = batch["frames"].to(device, non_blocking=True)
        ctx_len = batch["ctx_len"].to(device, non_blocking=True)
        target_px = batch["target"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        text_mask = batch["text_mask"].to(device, non_blocking=True)
        B, C = frames.shape[:2]
        empty = tokenizer([""], return_tensors="pt")
        empty_ids = empty["input_ids"].to(device)[None].expand(B, C, -1)
        slot = torch.arange(C, device=device)[None, :] < ctx_len[:, None]
        empty_mask = empty["attention_mask"].to(device)[None].expand(B, C, -1) * slot[..., None].long()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            ctx_feats, target_feats = encode_ctx(frames, ctx_len, target_px, encoder)
            loss, _, pred = ddp_model(ctx_feats, input_ids, text_mask, target_feats, ctx_len=ctx_len)
            loss_empty, _, pred_empty = ddp_model(ctx_feats, empty_ids, empty_mask, target_feats, ctx_len=ctx_len)
            sq_pred = ((pred.float() - target_feats.float()) ** 2).sum(dim=(1, 2))
            sq_id = ((last_ctx(ctx_feats, ctx_len).float() - target_feats.float()) ** 2).sum(dim=(1, 2))
            add("total", loss, loss_empty, B, sq_pred.sum(), sq_id.sum())
            by_group = {}
            for j, meta in enumerate(batch["meta"]):
                by_group.setdefault(group_of.get(meta["dataset"], "other"), []).append(j)
                d = ds_acc.setdefault(meta["dataset"], [0.0, 0.0, 0])
                d[0] += float(sq_pred[j])
                d[1] += float(sq_id[j])
                d[2] += 1
            for g, idx in by_group.items():
                target_g = target_feats[idx].to(pred.dtype)
                loss_g, _ = inner.loss_fn(pred[idx], target_g)
                loss_ge, _ = inner.loss_fn(pred_empty[idx], target_g)
                add(g, loss_g, loss_ge, len(idx), sq_pred[idx].sum(), sq_id[idx].sum())

    stats = torch.tensor([acc[k] for k in keys], dtype=torch.float64, device=device)
    ds_names = sorted(group_of)
    ds_stats = torch.tensor([ds_acc.get(d, [0.0, 0.0, 0]) for d in ds_names], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats)
        dist.all_reduce(ds_stats)
    logs = {}
    for k, (loss_sum, empty_sum, n, sq_pred_sum, sq_id_sum) in zip(keys, stats.tolist()):
        if n > 0:
            logs[f"{k}_loss"] = loss_sum / n
            logs[f"{k}_loss_empty_text"] = empty_sum / n
            logs[f"{k}_text_dropout_delta"] = (empty_sum - loss_sum) / n
            logs[f"{k}_rel_l2"] = sq_pred_sum / max(sq_id_sum, 1e-12)
    ds_rel = {d: sp / max(si, 1e-12) for d, (sp, si, n) in zip(ds_names, ds_stats.tolist()) if n > 0}
    for d, v in ds_rel.items():
        logs[f"ds/{d}_rel_l2"] = v
    if ds_rel:
        logs["dommean_rel_l2"] = sum(ds_rel.values()) / len(ds_rel)
    ddp_model.train()
    return logs
