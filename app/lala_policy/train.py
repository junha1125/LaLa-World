"""LaLa-Policy fine-tuning (config `app: lala_policy`).

Frozen vision encoder; trainable = the LaLa-World predictor (seeded from meta.init_from) + the policy modules
(proprio / action MLPs, queries, view embeddings, action head). Run from the repository root:
    python -m app.main --fname configs/lala_policy/base.yaml --devices cuda:0 ... cuda:7
"""

import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from app.lala_policy.losses import build_policy_loss
from app.lala_policy.model import build_policy_predictor
from app.lala_world.data import PolicyCollator, build_concat_dataset
from app.lala_world.data.policy_jsonl import MODE_POLICY, MODE_WM
from app.lala_world.models import LaLaWorld, build_llm_backbone, build_vision_encoder
from app.lala_world.models.hub import model_name
from app.lala_world.train import CKPT_PREFIX, init_wandb, param_groups, save_rolling_checkpoint
from src.utils.logging import CSVLogger, get_logger
from src.utils.schedulers import WarmupCosineSchedule

logger = get_logger(__name__)

INIT_COMPAT_KEYS = (
    ("vision", "kind"),
    ("vision", "model_id"),
    ("llm", "kind"),
    ("llm", "model_id"),
    ("llm", "num_layers"),
)


class PolicyWithLoss(nn.Module):
    """Container so DDP sees every trainable parameter (predictor AND loss module)."""

    def __init__(self, predictor, loss_fn):
        super().__init__()
        self.predictor = predictor
        self.loss_fn = loss_fn

    def forward(self, vis, batch, target_latent, future_vis):
        out = self.predictor(
            vis,
            batch["input_ids"],
            batch["text_mask"],
            batch["proprio"],
            batch["mode"],
            actions=batch["actions"],
            future_proprio=batch["future_proprio"],
            future_vis=future_vis,
            value=batch["value"],
        )
        loss, logs = self.loss_fn(out, batch, target_latent)
        return loss, logs, out


def build_predictor(cfg_model, encoder):
    backbone = build_llm_backbone(cfg_model["llm"])
    wm = LaLaWorld(backbone, vis_dim=encoder.embed_dim, target_dim=encoder.embed_dim, grid_hw=encoder.grid_size)
    return build_policy_predictor(wm, cfg_model["policy"])


def load_init_from(predictor, path, cfg_model):
    """Seed the wrapped world model from a LaLa-World checkpoint (strict on the WM keys)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    src = ckpt["config"]["model"]
    for section, key in INIT_COMPAT_KEYS:
        a, b = src.get(section, {}).get(key), cfg_model.get(section, {}).get(key)
        if key == "model_id":
            a, b = model_name(a), model_name(b)
        if a != b:
            raise ValueError(f"init_from {path}: model.{section}.{key} = {a!r} but this config has {b!r}")
    state = {k[len("predictor.") :]: v for k, v in ckpt["model"].items() if k.startswith("predictor.")}
    predictor.load_world_model(state)
    n = sum(v.numel() for v in state.values())
    logger.info(f"world model initialised from {path} (epoch {ckpt.get('epoch')}, {n / 1e6:.1f}M params)")
    return int(ckpt.get("epoch", -1))


@torch.no_grad()
def encode_views(encoder, frames, use_amp):
    """(B,V,3,H,W) -> (B,V,N,D) with the frozen encoder."""
    B, V = frames.shape[:2]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        feats = encoder(frames.reshape(B * V, *frames.shape[2:]))["patches"]
    return feats.reshape(B, V, *feats.shape[1:]).detach()


def _push_policy_defaults(cfgs_data, cfg_policy):
    """model.policy.{views, chunk_size, pv_pred} and data.stats_path are the single source of truth for the sources."""
    pv_pred = bool(cfg_policy.setdefault("pv_pred", True))  # written back so the checkpoint config carries it
    for s in cfgs_data["sources"]:
        if s.get("kind") != "policy_jsonl":
            raise ValueError(f"lala_policy accepts kind policy_jsonl sources only, got {s.get('kind')!r}")
        s.setdefault("views", list(cfg_policy.get("views", ["primary"])))
        s.setdefault("chunk_size", int(cfg_policy.get("chunk_size", 16)))
        s.setdefault("pv_pred", pv_pred)
        if "stats_path" not in s and cfgs_data.get("stats_path"):
            s["stats_path"] = cfgs_data["stats_path"]
        if list(s["views"]) != list(cfg_policy.get("views", ["primary"])):
            raise ValueError(
                f"source {s.get('path')} views {s['views']} != model.policy.views {cfg_policy.get('views')}"
            )
        if int(s["chunk_size"]) != int(cfg_policy.get("chunk_size", 16)):
            raise ValueError(f"source {s.get('path')} chunk_size {s['chunk_size']} != model.policy.chunk_size")
        if bool(s["pv_pred"]) != pv_pred:
            raise ValueError(f"source {s.get('path')} pv_pred {s['pv_pred']} != model.policy.pv_pred {pv_pred}")
        s.setdefault("aug", bool(cfgs_data.get("aug", True)))
        s.setdefault("aug_stronger", bool(cfgs_data.get("aug_stronger", True)))


def main(args, resume_preempt=False):
    cfgs_meta = args["meta"]
    cfgs_data = args["data"]
    cfgs_model = args["model"]
    cfgs_opt = args["optimization"]
    cfgs_logging = args.get("logging", {})
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

    # ---------------------------------------------------------------- models
    if not cfgs_model["vision"].get("frozen", True):
        raise ValueError("lala_policy keeps the vision encoder frozen (model.vision.frozen must be true)")
    encoder = build_vision_encoder(cfgs_model["vision"]).to(device)
    encoder.eval()
    cfg_policy = cfgs_model["policy"]
    predictor = build_predictor(cfgs_model, encoder).to(device)
    init_from = cfgs_meta.get("init_from")
    if init_from:
        load_init_from(predictor, init_from, cfgs_model)
    else:
        logger.info("no init_from: world model NOT pretrained (LLM as released, connector/queries/heads random)")
    loss_fn = build_policy_loss(args.get("loss", {})).to(device)
    model = PolicyWithLoss(predictor, loss_fn).to(device)
    ddp_model = DistributedDataParallel(model, static_graph=True) if world_size > 1 else model
    inner = ddp_model.module if world_size > 1 else model
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_new = sum(p.numel() for n, p in predictor.named_parameters() if not n.startswith("wm.") and p.requires_grad)
    n_head = sum(p.numel() for p in predictor.flow_head.parameters()) if predictor.flow_head is not None else 0
    logger.info(
        f"trainable params: {n_train / 1e6:.1f}M (policy-specific {n_new / 1e6:.2f}M, of which "
        f"{predictor.action_head_kind} head {n_head / 1e6:.2f}M) | encoder frozen | pv_pred {predictor.pv_pred}"
    )

    # ---------------------------------------------------------------- data
    _push_policy_defaults(cfgs_data, cfg_policy)
    backbone = predictor.wm.backbone
    max_text_tokens = cfgs_data.get("max_text_tokens", 64)
    collator = PolicyCollator(backbone.tokenizer, max_text_tokens=max_text_tokens)
    batch_size = cfgs_data.get("batch_size", 32)
    grad_accum = max(1, int(cfgs_opt.get("grad_accum_steps") or 1))
    samples_per_epoch = cfgs_opt.get("samples_per_epoch")
    requested_ipe = max(1, samples_per_epoch // (batch_size * world_size * grad_accum)) if samples_per_epoch else None
    mean, std, image_size = encoder.image_mean, encoder.image_std, encoder.image_size

    def make_loader(split, drop_last):
        num_workers = cfgs_data.get("num_workers", 8)
        pin = cfgs_data.get("pin_mem", True) and device.type == "cuda"
        dataset = build_concat_dataset(cfgs_data["sources"], split, image_size, mean, std)
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
            pin_memory=pin,
            drop_last=drop_last,
            collate_fn=collator,
            persistent_workers=num_workers > 0,
        )

    train_loader = make_loader("train", drop_last=True)
    val_loader = make_loader("val", drop_last=False)
    ipe = len(train_loader) // grad_accum
    if ipe == 0:
        raise ValueError(f"data provides {len(train_loader)} batches/epoch, fewer than grad_accum_steps={grad_accum}")
    if requested_ipe and requested_ipe < ipe:
        ipe = requested_ipe
    elif requested_ipe and requested_ipe > ipe:
        logger.info(f"samples_per_epoch asks {requested_ipe} steps/epoch but the data only provides {ipe}")
    num_epochs = cfgs_opt.get("epochs", 10)
    if quick_debug:
        ipe, num_epochs = min(ipe, 4), 1
    total_steps = ipe * num_epochs
    micro_ipe = ipe * grad_accum
    logger.info(
        f"iterations per epoch: {ipe} (world={world_size}, bs/gpu={batch_size}, grad_accum={grad_accum} "
        f"-> global batch {batch_size * world_size * grad_accum}); train samples {len(train_loader.dataset)}"
    )
    stats_paths = sorted({s["stats_path"] for s in cfgs_data["sources"]})
    stats = json.load(open(stats_paths[0]))  # kept in the checkpoint so eval needs no data files
    if len(stats_paths) > 1:
        logger.info(f"note: {len(stats_paths)} stats files across sources; checkpoint stores {stats_paths[0]}")

    # ---------------------------------------------------------------- optim
    optimizer = torch.optim.AdamW(
        param_groups(model, cfgs_opt.get("weight_decay", 0.05)),
        lr=cfgs_opt.get("lr", 1e-4),
        betas=tuple(cfgs_opt.get("betas", (0.9, 0.98))),
    )
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=cfgs_opt.get("warmup_steps", 1000),
        start_lr=cfgs_opt.get("start_lr", 1e-6),
        ref_lr=cfgs_opt.get("lr", 1e-4),
        final_lr=cfgs_opt.get("final_lr", 1e-6),
        T_max=total_steps,
    )
    clip_grad = cfgs_opt.get("clip_grad", 1.0)
    clip_params = [p for p in model.parameters() if p.requires_grad]

    # ---------------------------------------------------------------- resume
    start_epoch, train_seconds = 0, 0.0
    latest_path = os.path.join(folder, f"{CKPT_PREFIX}-latest.pth.tar")
    read_path = cfgs_meta.get("read_checkpoint") or latest_path
    if (cfgs_meta.get("load_checkpoint", False) or resume_preempt) and os.path.exists(read_path):
        ckpt = torch.load(read_path, map_location="cpu", weights_only=False)
        inner.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler._step = ckpt["sched_step"]
        start_epoch = ckpt["epoch"]
        train_seconds = float(ckpt.get("train_seconds", 0.0))
        logger.info(f"resumed from {read_path} at epoch {start_epoch}")

    wandb = init_wandb(cfgs_logging, folder, args, rank)
    csv_logger = (
        CSVLogger(
            os.path.join(folder, "train.csv"),
            ("%d", "epoch"),
            ("%d", "itr"),
            ("%.5f", "loss"),
            ("%.5f", "action_l1"),
            ("%.5f", "latent"),
            ("%.5f", "value_l1"),
            ("%.2e", "lr"),
            ("%.1f", "imgs_per_sec"),
            ("%.1f", "train_min"),
        )
        if rank == 0
        else None
    )

    # ---------------------------------------------------------------- loop
    global_step = start_epoch * ipe
    nan_streak = grad_streak = 0
    final_loss = float("nan")
    max_train_minutes = cfgs_meta.get("max_train_minutes") or 0
    stop = False
    for epoch in range(start_epoch, num_epochs):
        ddp_model.train()
        encoder.eval()
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        last_log_time, last_log_step = time.time(), global_step
        for itr, batch in enumerate(train_loader):
            if itr >= micro_ipe:
                break
            t_step = time.time()
            micro, opt_itr = itr % grad_accum, itr // grad_accum
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if micro == 0:
                lr = scheduler.step()
                window_ok, window_loss, window_logs = True, 0.0, {}
            vis = encode_views(encoder, batch["frames"], use_amp)
            future_vis = encode_views(encoder, batch["future_frames"], use_amp)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, loss_logs, out = ddp_model(vis, batch, future_vis, future_vis)

            finite = torch.isfinite(loss.detach()).int()
            if world_size > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite:
                nan_streak += 1
                logger.warning(f"non-finite loss on some rank at step {global_step} (streak {nan_streak})")
                if nan_streak >= 3:
                    raise FloatingPointError("3 consecutive non-finite losses, aborting")
                window_ok = False
            else:
                nan_streak = 0
                window_loss += loss.detach().item()
                for k, v in loss_logs.items():
                    window_logs[k] = window_logs.get(k, 0.0) + float(v)
            (loss / grad_accum if grad_accum > 1 else loss).backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_seconds += time.time() - t_step
            if micro < grad_accum - 1:
                continue
            t_step = time.time()
            if window_ok:
                grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, clip_grad or float("inf"))
                grads_finite = torch.isfinite(grad_norm).int()
                if world_size > 1:
                    dist.all_reduce(grads_finite, op=dist.ReduceOp.MIN)
                if not grads_finite:
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
            if max_train_minutes and opt_itr % cfgs_meta.get("log_freq", 50) == 0:
                stop_flag = torch.tensor(int(train_seconds / 60 >= max_train_minutes), device=device)
                if world_size > 1:
                    dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
                stop = bool(stop_flag.item())
            if window_ok and opt_itr % cfgs_meta.get("log_freq", 50) == 0:
                n_imgs = batch["frames"].shape[0] * grad_accum * world_size * max(1, global_step - last_log_step)
                imgs_per_sec = n_imgs / max(1e-6, time.time() - last_log_time)
                last_log_time, last_log_step = time.time(), global_step
                logs = {f"train/{k}": v / grad_accum for k, v in window_logs.items()}
                logs.update(
                    {
                        "train/loss": final_loss,
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/grad_norm": float(grad_norm),
                        "train/train_min": train_seconds / 60,
                        "train/pred_action_std": float(out["actions"].detach().float().std()),
                    }
                )
                if rank == 0:
                    logger.info(
                        f"[e{epoch} i{opt_itr}/{ipe}] loss {final_loss:.4f} "
                        f"action {logs.get('train/loss_action', 0):.4f} "
                        f"latent {logs.get('train/loss_latent', 0):.4f} "
                        f"proprio {logs.get('train/loss_proprio', 0):.4f} value {logs.get('train/loss_value', 0):.4f} "
                        f"lr {lr:.2e} trained {train_seconds / 60:.1f} min"
                    )
                    if csv_logger:
                        csv_logger.log(
                            epoch,
                            opt_itr,
                            final_loss,
                            logs.get("train/loss_action", 0.0),
                            logs.get("train/loss_latent", 0.0),
                            logs.get("train/loss_value", 0.0),
                            lr,
                            imgs_per_sec,
                            train_seconds / 60,
                        )
                    if wandb:
                        wandb.log(logs, step=global_step)
            if stop:
                logger.info(f"max_train_minutes {max_train_minutes} reached at epoch {epoch}; stopping")
                break

        # ------------------------------------------------------------ val
        if (epoch + 1) % cfgs_meta.get("eval_freq", 1) == 0 or epoch == num_epochs - 1 or stop:
            val_logs = evaluate(ddp_model, encoder, val_loader, device, use_amp, quick_debug)
            if rank == 0:
                logger.info(f"[e{epoch}] " + " ".join(f"{k} {v:.4f}" for k, v in val_logs.items()))
                with open(os.path.join(folder, "val.jsonl"), "a") as f:
                    f.write(json.dumps({"epoch": epoch + 1, "step": global_step, **val_logs}) + "\n")
                if wandb:
                    wandb.log({f"val/{k}": v for k, v in val_logs.items()}, step=global_step)

        # ------------------------------------------------------------ save
        if rank == 0 and ((epoch + 1) % cfgs_meta.get("save_every_freq", 1) == 0 or epoch == num_epochs - 1 or stop):
            ckpt = {
                "epoch": epoch + 1,
                "model": inner.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sched_step": scheduler._step,
                "config": args,
                "train_seconds": train_seconds,
                "stats": stats,
            }
            saved_path = save_rolling_checkpoint(folder, ckpt, keep_last=cfgs_meta.get("save_keep_last", -1))
            logger.info(f"saved checkpoint to {saved_path}")
        if stop:
            break

    if wandb:
        wandb.finish()
    return {"final_train_loss": final_loss, "epochs_run": num_epochs - start_epoch, "ipe": ipe}


@torch.no_grad()
def evaluate(ddp_model, encoder, val_loader, device, use_amp, quick_debug, max_batches=50):
    """Masked-by-mode val metrics, sums all-reduced across ranks:
    action_l1 (policy samples), latent_l2 + latent_rel_l2 (= L2 / L2 of copying the current frame; < 1
    beats identity), proprio_l1 (policy + wm), value_l1 (all), plus per-dataset action_l1."""
    ddp_model.eval()
    encoder.eval()
    acc = {}  # name -> [sum, count]

    def add(name, s, n):
        a = acc.setdefault(name, [0.0, 0.0])
        a[0] += float(s)
        a[1] += float(n)

    if quick_debug:
        max_batches = 2
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        vis = encode_views(encoder, batch["frames"], use_amp)
        future_vis = encode_views(encoder, batch["future_frames"], use_amp)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, _, out = ddp_model(vis, batch, future_vis, future_vis)
        mode = batch["mode"]
        pol, wm = mode == MODE_POLICY, mode == MODE_WM
        act_l1 = (out["actions"].float() - batch["actions"]).abs().mean(dim=(1, 2))
        lat_sq = ((out["future_latent"].float() - future_vis.float()) ** 2).sum(dim=(1, 2, 3))
        id_sq = ((vis.float() - future_vis.float()) ** 2).sum(dim=(1, 2, 3))
        lat_l2 = ((out["future_latent"].float() - future_vis.float()) ** 2).mean(dim=(1, 2, 3))
        add("action_l1", act_l1[pol].sum(), pol.sum())
        add("latent_l2", lat_l2[pol | wm].sum(), (pol | wm).sum())
        add("latent_sq", lat_sq[pol | wm].sum(), 1)
        add("identity_sq", id_sq[pol | wm].sum(), 1)
        if "future_proprio" in out:  # pv_pred
            prop_l1 = (out["future_proprio"].float() - batch["future_proprio"]).abs().mean(dim=1)
            add("proprio_l1", prop_l1[pol | wm].sum(), (pol | wm).sum())
        if "value" in out:
            val_l1 = (out["value"].float() - batch["value"]).abs()
            add("value_l1", val_l1.sum(), val_l1.numel())
        for j, meta in enumerate(batch["meta"]):
            if pol[j]:
                add(f"ds/{meta['dataset']}_action_l1", act_l1[j], 1)
    names = sorted(acc)
    stats = torch.tensor([acc[k] for k in names], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        names_all = [None] * dist.get_world_size()
        dist.all_gather_object(names_all, names)
        union = sorted(set().union(*names_all))
        if union != names:  # rank-dependent per-dataset keys: align before the all_reduce
            stats = torch.tensor([acc.get(k, [0.0, 0.0]) for k in union], dtype=torch.float64, device=device)
            names = union
        dist.all_reduce(stats)
    logs = {}
    for k, (s, n) in zip(names, stats.tolist()):
        if k in ("latent_sq", "identity_sq"):
            continue
        if n > 0:
            logs[k] = s / n
    if "latent_sq" in names and "identity_sq" in names:
        lat = stats[names.index("latent_sq"), 0].item()
        idn = stats[names.index("identity_sq"), 0].item()
        logs["latent_rel_l2"] = lat / max(idn, 1e-12)
    ddp_model.train()
    return logs
