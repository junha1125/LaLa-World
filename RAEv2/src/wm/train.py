"""Train the RAEv2 world model: frozen RAE (DINOv3-L) + DiT on (current frame, text) -> next-frame latent.

    PYTHONPATH=src torchrun --nproc_per_node=8 src/wm/train.py --config configs/rae_xl.yaml --compile

Snapshots: <run_dir>/checkpoints/step-XXXXXXX.pt (bf16 model + ema + config + train_seconds), latest.pt (+ optimizer).
Logs: train.csv (loss, rel_l2 = ||x_hat - e||^2 / ||s_t - e||^2 on the training batch), val.jsonl (per-source val).
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from copy import deepcopy

# per-rank inductor cache; the tag pins the SDPA backend chosen below (cached graphs bake it in)
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), f"torchinductor_rae_wm_nocudnn_rank{os.environ.get('LOCAL_RANK', '0')}"),
)
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import yaml  # noqa: E402

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from stage1 import RAE  # noqa: E402
from stage2.transport import Transport  # noqa: E402
from wm import abs_path  # noqa: E402
from wm import cond as cond_mod  # noqa: E402
from wm.data import build_train_dataset, build_val_subsets, make_train_loader, make_val_loader, resolve_sources  # noqa: E402
from wm.model import build_dit, count_params, load_stage2_init, predict_x_t1, time_shift  # noqa: E402
from wm.optim import build_optimizer  # noqa: E402
from wm.text_encoder import FrozenTextEncoder  # noqa: E402
from wm.utils import cleanup_distributed, create_logger, setup_distributed, update_ema  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# the cuDNN SDPA bf16 backward produced NaN grads mid-run (QK-norm gains grown); flash / mem-efficient kernels are fine
torch.backends.cuda.enable_cudnn_sdp(False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", default=None, help="overrides run_dir of the config")
    p.add_argument("--resume", default=None, help="latest.pt to resume (default: <run_dir>/checkpoints/latest.pt if present)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--resume-snapshot", default=None, help="weights-only step-XXXXXXX.pt to continue from with a fresh optimizer")
    p.add_argument("--compile", action="store_true", help="torch.compile the DiT forward")
    p.add_argument("--smoke", type=int, default=0, help="run N steps with a tiny val set and exit (no checkpoints)")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def model_path(path):
    """HF hub ids are passed through; local relative paths are resolved against the RAEv2 directory."""
    local = abs_path(path)
    return local if os.path.isdir(local) else path


def lr_lambda_factory(warmup_steps, total_steps, final_ratio):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if step >= total_steps:
            return final_ratio
        return 1.0 - (1.0 - final_ratio) * (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return f


def fm_losses(transport, model, x1, model_kwargs, base_model_coeff=1.0):
    """x-prediction flow-matching loss, plus the internal-guidance base head (REPA-style second output)."""
    t, x0, x1 = transport.sample(x1)
    te = t.view(-1, 1, 1, 1)
    xt = (1 - te) * x1 + te * x0
    vt = (xt - x1) / te.clamp_min(transport.t_eps)
    out = model(xt, t, **model_kwargs)
    base = None
    if isinstance(out, tuple):
        out, base = out
    terms = {"loss": transport.compute_loss(out, vt, xt, t), "t": t, "x0": x0}
    if base is not None:
        loss_base = transport.compute_loss(base, vt, xt, t)
        terms["loss"] = terms["loss"] + base_model_coeff * loss_base
        terms["loss_base"] = loss_base
    return terms


def all_reduce_grads(params, world_size, bucket_bytes=512 * 2**20):
    """Average .grad across ranks in flat buckets (DDP's reducer + torch.compile was unstable on the NGC image)."""
    bucket, size = [], 0

    def flush():
        if not bucket:
            return
        flat = torch._utils._flatten_dense_tensors(bucket)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat /= world_size
        for g, f in zip(bucket, torch._utils._unflatten_dense_tensors(flat, bucket)):
            g.copy_(f)

    for p in params:
        if p.grad is None:
            continue
        bucket.append(p.grad)
        size += p.grad.numel() * p.grad.element_size()
        if size >= bucket_bytes:
            flush()
            bucket, size = [], 0
    flush()


def all_reduce_mean(t, world_size):
    if world_size > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= world_size
    return t


@torch.no_grad()
def encode_batch(rae, text_encoder, batch, device):
    state = batch["state"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z_src = rae.encode(state).float()
        z_tgt = rae.encode(target).float()
        enc = text_encoder(batch["text"])
    return z_src, z_tgt, enc["tokens"].float(), enc["attention_mask"]


@torch.no_grad()
def run_val(models, rae, text_encoder, transport, val_loaders, device, world_size, latent_size, seed):
    """Per source: x_l2 (one-shot prediction error), id_l2 (copy the current frame), rel_l2 = x_l2 / id_l2,
    in-batch retrieval R@1 of the prediction (r1) and of the identity baseline (r1_id), fm_loss."""
    results = {}
    for mname, model in models.items():
        per_dom = defaultdict(lambda: torch.zeros(6, device=device))
        for dname, loader in val_loaders:
            g = torch.Generator(device=device).manual_seed(seed)
            for batch in loader:
                z_src, z_tgt, txt, mask = encode_batch(rae, text_encoder, batch, device)
                context = cond_mod.make_context(z_src, txt)
                B = z_src.shape[0]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x_hat = predict_x_t1(model, context, mask, latent_size, generator=g).float()
                    terms = fm_losses(transport, model, z_tgt, dict(context=context, attn_mask=mask), 1.0)
                x_l2 = ((x_hat - z_tgt) ** 2).mean(dim=(1, 2, 3))
                id_l2 = ((z_src - z_tgt) ** 2).mean(dim=(1, 2, 3))
                d_pred = torch.cdist(x_hat.flatten(1), z_tgt.flatten(1))
                d_id = torch.cdist(z_src.flatten(1), z_tgt.flatten(1))
                r1 = (d_pred.argmin(1) == torch.arange(B, device=device)).float()
                r1_id = (d_id.argmin(1) == torch.arange(B, device=device)).float()
                fm = terms["loss"].float().mean(dim=(1, 2, 3))
                s = per_dom[dname]
                s[0] += x_l2.sum(); s[1] += id_l2.sum(); s[2] += r1.sum(); s[3] += r1_id.sum(); s[4] += fm.sum(); s[5] += B
        tot = torch.zeros(6, device=device)
        for dname in sorted(per_dom):
            s = per_dom[dname].clone()
            if world_size > 1:
                dist.all_reduce(s, op=dist.ReduceOp.SUM)
            tot += s
            n = max(s[5].item(), 1)
            results[f"val_{mname}/{dname}/x_l2"] = (s[0] / n).item()
            results[f"val_{mname}/{dname}/id_l2"] = (s[1] / n).item()
            results[f"val_{mname}/{dname}/rel_l2"] = (s[0] / max(s[1].item(), 1e-12)).item()
            results[f"val_{mname}/{dname}/r1"] = (s[2] / n).item()
            results[f"val_{mname}/{dname}/r1_id"] = (s[3] / n).item()
            results[f"val_{mname}/{dname}/fm_loss"] = (s[4] / n).item()
        n = max(tot[5].item(), 1)
        results[f"val_{mname}/all/x_l2"] = (tot[0] / n).item()
        results[f"val_{mname}/all/id_l2"] = (tot[1] / n).item()
        results[f"val_{mname}/all/rel_l2"] = (tot[0] / max(tot[1].item(), 1e-12)).item()
        results[f"val_{mname}/all/r1"] = (tot[2] / n).item()
        results[f"val_{mname}/all/r1_id"] = (tot[3] / n).item()
        results[f"val_{mname}/all/fm_loss"] = (tot[4] / n).item()
        doms = sorted(per_dom)
        results[f"val_{mname}/dommean/rel_l2"] = sum(results[f"val_{mname}/{d}/rel_l2"] for d in doms) / len(doms)
        results[f"val_{mname}/dommean/r1"] = sum(results[f"val_{mname}/{d}/r1"] for d in doms) / len(doms)
    return results


def save_snapshot(path, model, ema_model, step, train_seconds, cfg, optimizer=None, scheduler=None, epoch=0):
    bf16 = optimizer is None
    state = {
        "step": step, "epoch": epoch, "train_seconds": train_seconds, "config": cfg,
        "model": {k: v.to(torch.bfloat16) if bf16 else v for k, v in model.state_dict().items()},
        "ema": {k: v.to(torch.bfloat16) if bf16 else v for k, v in ema_model.state_dict().items()},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
        state["scheduler"] = scheduler.state_dict() if scheduler is not None else None
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rank, world_size, device = setup_distributed()
    is_main = rank == 0

    tcfg, dcfg, mcfg = cfg["training"], cfg["data"], cfg["model"]
    run_dir = abs_path(args.run_dir or cfg["run_dir"])
    cfg["run_dir"] = run_dir
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    if dist.is_initialized():
        dist.barrier()
    logger = create_logger(run_dir if is_main else None)
    logger.info(f"config {args.config} -> {run_dir}; world_size {world_size}")

    seed = int(tcfg.get("seed", 0))
    torch.manual_seed(seed * world_size + rank)
    global_batch = int(tcfg["global_batch_size"])
    if global_batch % world_size:
        raise ValueError("global_batch_size must be divisible by world_size")
    micro_bs = global_batch // world_size
    smoke = args.smoke > 0

    # ---------------------------------------------------------------- data
    image_size = int(dcfg.get("image_size", 256))
    sources = resolve_sources(dcfg["sources"], dcfg.get("root"))
    train_ds = build_train_dataset(sources, image_size)
    logger.info(f"train pool: {len(train_ds):,} samples from {len(sources)} sources")
    train_loader, train_sampler = make_train_loader(train_ds, micro_bs, int(dcfg.get("num_workers", 8)), rank, world_size, seed)
    val_subsets = build_val_subsets(sources, image_size, int(dcfg.get("val_per_source", 256)) if not smoke else 16, seed=seed)
    val_loaders = [(n, make_val_loader(s, int(dcfg.get("val_batch_size", 128)), 4, rank, world_size)) for n, s in val_subsets]
    samples_per_epoch = int(tcfg.get("samples_per_epoch", 1_000_000))
    steps_per_epoch = max(1, min(len(train_loader), samples_per_epoch // global_batch))
    logger.info(f"epoch = {steps_per_epoch} steps x {global_batch} (pool provides {len(train_loader)} batches/rank)")

    # ---------------------------------------------------------------- models
    if cfg.get("dinov3_repo_dir"):
        os.environ.setdefault("DINOV3_REPO_DIR", abs_path(cfg["dinov3_repo_dir"]))
    s1 = dict(cfg["stage_1"])
    for k in ("decoder_config_path", "pretrained_decoder_path", "normalization_stat_path"):
        s1[k] = abs_path(s1[k])
    rae = RAE(**s1).to(device).eval()
    rae.requires_grad_(False)
    text_encoder = FrozenTextEncoder(model_path(cfg["text_encoder"]["model_path"]), int(cfg["text_encoder"]["max_length"])).to(device)
    latent_size = tuple(cfg["misc"]["latent_size"])

    model = build_dit(mcfg, text_encoder.feature_dim, text_encoder.max_length).to(device)
    init = cfg.get("init", {})
    if init.get("stage2_ckpt"):
        load_stage2_init(model, abs_path(init["stage2_ckpt"]), init.get("which", "ema"), int(init.get("n_enc_ckpt", 28)), logger)
    logger.info(f"DiT params: {count_params(model)/1e6:.1f}M (enc depth {model.num_enc_blocks}, dec {model.num_dec_blocks}); "
                f"RAE encoder {count_params(rae.encoder)/1e6:.0f}M, decoder {count_params(rae.decoder)/1e6:.0f}M (frozen); "
                f"text encoder {count_params(text_encoder)/1e6:.0f}M (frozen)")
    ema_model = deepcopy(model).to(device).eval()
    ema_model.requires_grad_(False)
    if world_size > 1:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
    fwd = torch.compile(model) if args.compile else model
    train_params = [p for p in model.parameters() if p.requires_grad]

    optimizer, omsg = build_optimizer(train_params, tcfg["optimizer"])
    logger.info(omsg)
    max_steps = int(tcfg["max_steps"]) if not smoke else args.smoke
    warmup = int(tcfg.get("warmup_steps", 2000))
    schedule_steps = int(tcfg.get("schedule_steps", max_steps)) if not smoke else max_steps  # LR-decay anchor
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(warmup, schedule_steps, float(tcfg.get("final_lr_ratio", 0.1))))

    transport = Transport(
        prediction=cfg["transport"].get("prediction", "x"),
        time_dist_type=cfg["transport"].get("time_dist_type", "logit-normal_0_1"),
        time_dist_shift=time_shift(cfg["misc"], latent_size),
        t_eps=float(cfg["transport"].get("t_eps", 0.05)),
    )
    base_coeff = float(cfg.get("internal_guidance", {}).get("base_model_coeff", 1.0))
    text_dropout_p = float(cfg.get("conditioning", {}).get("text_dropout_p", 0.0))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        null = text_encoder([""])
    null_text, null_mask = null["tokens"].float(), null["attention_mask"]

    # ---------------------------------------------------------------- resume
    step, epoch, train_seconds = 0, 0, 0.0
    resume_path = args.resume or os.path.join(ckpt_dir, "latest.pt")
    if not args.no_resume and os.path.isfile(resume_path) and not smoke:
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema_model.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scheduler"):
            sched.load_state_dict(ck["scheduler"])
        step, epoch, train_seconds = ck["step"], ck.get("epoch", 0), ck.get("train_seconds", 0.0)
        logger.info(f"resumed {resume_path}: step {step}, epoch {epoch}, {train_seconds/3600:.2f} h trained")
        del ck
    elif args.resume_snapshot and not smoke:
        ck = torch.load(args.resume_snapshot, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
        ema_model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["ema"].items()})
        step, epoch, train_seconds = ck["step"], ck.get("epoch", 0), ck.get("train_seconds", 0.0)
        for _ in range(step):
            sched.step()
        logger.info(f"continued from snapshot {args.resume_snapshot}: step {step}, {train_seconds/3600:.2f} h trained; "
                    f"fresh optimizer, lr {optimizer.param_groups[0]['lr']:.2e}")
        del ck

    use_wandb = is_main and not args.no_wandb and cfg.get("logging", {}).get("use_wandb", False) and not smoke
    if use_wandb:
        import wandb
        wid_path = os.path.join(run_dir, "wandb_run_id.txt")
        wid = open(wid_path).read().strip() if os.path.exists(wid_path) else None
        run = wandb.init(project=cfg["logging"].get("project", "LaLa-World"), name=cfg["logging"].get("run_name") or os.path.basename(run_dir),
                         id=wid, resume="allow", config=cfg, dir=run_dir)
        with open(wid_path, "w") as f:
            f.write(run.id)

    log_every = int(tcfg.get("log_every", 50))
    val_every = int(tcfg.get("val_every", 5000)) if not smoke else max(1, args.smoke)
    ckpt_every = int(tcfg.get("ckpt_every", 5000)) if not smoke else args.smoke
    max_hours = float(tcfg.get("max_hours", 1e9))
    clip_grad = float(tcfg.get("clip_grad", 1.0))
    ema_decay = float(tcfg.get("ema_decay", 0.9995))
    rel_every = int(tcfg.get("rel_l2_every", log_every))
    csv_path = os.path.join(run_dir, "train.csv")
    if is_main and step == 0:
        with open(csv_path, "w") as f:
            f.write("step,epoch,loss,loss_base,rel_l2,lr,train_seconds,samples_per_sec\n")
    val_path = os.path.join(run_dir, "val.jsonl")

    def do_val(tag_step):
        model.eval()
        t0 = time.time()
        res = run_val({"ema": ema_model, "model": model}, rae, text_encoder, transport, val_loaders, device, world_size, latent_size, seed=1234)
        res.update({"step": tag_step, "train_seconds": train_seconds, "val_seconds": time.time() - t0})
        if is_main:
            with open(val_path, "a") as f:
                f.write(json.dumps(res) + "\n")
            logger.info(f"[val step {tag_step}] ema rel_l2 all {res['val_ema/all/rel_l2']:.4f} dommean {res['val_ema/dommean/rel_l2']:.4f} "
                        f"r1 {res['val_ema/all/r1']:.3f} (id {res['val_ema/all/r1_id']:.3f}) | model rel_l2 all {res['val_model/all/rel_l2']:.4f} "
                        f"fm {res['val_model/all/fm_loss']:.4f} | {res['val_seconds']:.0f}s")
            if use_wandb:
                wandb.log(res, step=tag_step)
        model.train()

    def params_finite():
        with torch.no_grad():
            return bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all())

    if step == 0:
        do_val(0)

    # ---------------------------------------------------------------- train
    model.train()
    done = step >= max_steps
    t_log = time.time()
    loss_acc, base_acc, n_acc = torch.zeros((), device=device), torch.zeros((), device=device), 0
    rel_acc, rel_n = torch.zeros(2, device=device), 0
    skipped, skipped_run = 0, 0
    while not done:
        train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            if i >= steps_per_epoch:
                break
            t_step = time.time()
            z_src, z_tgt, txt, mask0 = encode_batch(rae, text_encoder, batch, device)
            context0 = cond_mod.make_context(z_src, txt)
            context, mask = cond_mod.apply_text_dropout(context0, mask0, null_text, null_mask, text_dropout_p)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                terms = fm_losses(transport, fwd, z_tgt, dict(context=context, attn_mask=mask), base_coeff)
                loss = terms["loss"].mean()
            loss.backward()
            if world_size > 1:
                all_reduce_grads(train_params, world_size)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad if clip_grad > 0 else float("inf"))
            # non-finite gate: every rank skips the step together; abort after 20 in a row
            ok_t = (torch.isfinite(gnorm) & torch.isfinite(loss.detach())).to(torch.int32)
            if world_size > 1:
                dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
            ok = bool(ok_t.item())
            if not ok:
                skipped += 1
                skipped_run += 1
                logger.info(f"step {step+1}: NON-FINITE (loss {loss.detach().item():.4g}, grad-norm {gnorm.item():.4g}) -> skipped "
                            f"({skipped} total, {skipped_run} in a row)")
                optimizer.zero_grad(set_to_none=True)
                if skipped_run >= 20:
                    raise RuntimeError(f"{skipped_run} consecutive non-finite steps at step {step+1}: aborting")
            else:
                skipped_run = 0
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema_model, model, decay=ema_decay)
            sched.step()
            step += 1
            torch.cuda.synchronize()
            train_seconds += time.time() - t_step
            if ok:
                loss_acc += loss.detach(); n_acc += 1
                if "loss_base" in terms:
                    base_acc += terms["loss_base"].detach().float().mean()
            if step % rel_every == 0 and ok:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    x_hat = predict_x_t1(model, context0, mask0, latent_size).float()
                rel_acc[0] += ((x_hat - z_tgt) ** 2).sum()
                rel_acc[1] += ((z_src - z_tgt) ** 2).sum()
                rel_n += 1

            if step % log_every == 0:
                la = all_reduce_mean(loss_acc / max(n_acc, 1), world_size).item() if n_acc else float("nan")
                ba = all_reduce_mean(base_acc / max(n_acc, 1), world_size).item() if n_acc else float("nan")
                if world_size > 1:
                    dist.all_reduce(rel_acc, op=dist.ReduceOp.SUM)
                rel = (rel_acc[0] / rel_acc[1].clamp_min(1e-12)).item() if rel_n > 0 else float("nan")
                sps = n_acc * global_batch / (time.time() - t_log)
                lr = optimizer.param_groups[0]["lr"]
                logger.info(f"[ep {epoch} step {step}/{max_steps}] loss {la:.4f} base {ba:.4f} rel_l2 {rel:.4f} lr {lr:.2e} "
                            f"{sps:.0f} samples/s  trained {train_seconds/3600:.2f} h" + (f"  skipped {skipped}" if skipped else ""))
                if not math.isfinite(la) and not params_finite():
                    raise RuntimeError(f"non-finite parameters at step {step}: aborting (resume from the last good latest.pt)")
                if is_main:
                    with open(csv_path, "a") as f:
                        f.write(f"{step},{epoch},{la:.5f},{ba:.5f},{rel:.5f},{lr:.3e},{train_seconds:.1f},{sps:.1f}\n")
                    if use_wandb:
                        wandb.log({"train/loss": la, "train/loss_base": ba, "train/rel_l2": rel, "train/lr": lr,
                                   "train/samples_per_sec": sps, "train/hours": train_seconds / 3600, "epoch": epoch,
                                   "train/skipped_steps": skipped}, step=step)
                loss_acc.zero_(); base_acc.zero_(); rel_acc.zero_(); n_acc = 0; rel_n = 0; t_log = time.time()

            if step % val_every == 0:
                do_val(step)
            if step % ckpt_every == 0 and not smoke:
                if not params_finite():
                    raise RuntimeError(f"non-finite parameters at step {step}: not saving, aborting")
                if is_main:
                    save_snapshot(os.path.join(ckpt_dir, f"step-{step:07d}.pt"), model, ema_model, step, train_seconds, cfg, epoch=epoch)
                    save_snapshot(os.path.join(ckpt_dir, "latest.pt"), model, ema_model, step, train_seconds, cfg, optimizer, sched, epoch)
                    logger.info(f"saved step-{step:07d}.pt + latest.pt ({train_seconds/3600:.2f} h, skipped {skipped})")
                if dist.is_initialized():
                    dist.barrier()
            # the stop decision must agree on every rank (train_seconds drifts slightly across ranks)
            stop = (step >= max_steps) or (train_seconds / 3600 >= max_hours)
            if world_size > 1:
                flag = torch.tensor(int(stop), device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                stop = bool(flag.item())
            if stop:
                done = True
                break
        epoch += 1

    if step % val_every != 0:
        do_val(step)
    if is_main and not smoke:
        save_snapshot(os.path.join(ckpt_dir, f"step-{step:07d}.pt"), model, ema_model, step, train_seconds, cfg, epoch=epoch)
        save_snapshot(os.path.join(ckpt_dir, "latest.pt"), model, ema_model, step, train_seconds, cfg, optimizer, sched, epoch)
    logger.info(f"done: step {step}, {train_seconds/3600:.2f} h of training")
    if use_wandb:
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
