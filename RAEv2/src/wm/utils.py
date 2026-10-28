import logging
import os
import sys
from collections import OrderedDict

import torch
import torch.distributed as dist


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        return rank, world_size, torch.device("cuda", local_rank)
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir, name="rae-wm"):
    """Rank 0 logs to stdout and <logging_dir>/log.txt; other ranks get a silent logger."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger = logging.getLogger(name)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.propagate = False
    if rank != 0:
        logger.setLevel(logging.CRITICAL + 1)
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logging_dir:
        os.makedirs(logging_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(logging_dir, "log.txt"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
