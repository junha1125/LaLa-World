# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import datetime
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.elastic.utils.distributed import get_free_port

from src.utils.logging import get_logger

logger = get_logger()


def _get_port(world_size, default_port=37129):
    # app/main.py exports a free MASTER_PORT per launch so concurrent jobs on one node do not clash
    if os.environ.get("MASTER_PORT"):
        return int(os.environ["MASTER_PORT"])
    return get_free_port() if world_size == 1 else default_port


def init_distributed(
    port=None,
    rank_and_world_size=(None, None),
    nccl_timeout_minutes=None,
):
    # Set all environment variables *before* calling `torch.distributed.init_process_group`. `init_process_group` may
    # reallocate environment variables; modifying them after could trigger a race condition leading to a segfault.
    if "SLURM_JOB_ID" in os.environ:
        # Use the slurm_tmpdir (if it exists) instead of /tmp
        tmpdir = Path(f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}")
        if tmpdir.exists():
            os.environ["TMPDIR"] = str(tmpdir)

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    # defaults
    rank, world_size = rank_and_world_size
    os.environ["MASTER_ADDR"] = "localhost"

    # torchrun
    dist_keys = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
    dist_env_set = all([key in os.environ for key in dist_keys])

    # If rank and world_size are explicitly provided, set env vars for compatibility
    # Fix local launcher on interactive node
    if (rank is not None) and (world_size is not None) and not dist_env_set:
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        dist_env_set = True

    # submitit / hydra.submitit
    if not dist_env_set and ((rank is None) or (world_size is None)):
        try:
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            # $HOSTNAME is not always exported to os.environ
            os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"] if "HOSTNAME" in os.environ else socket.gethostname()
        except Exception as e:
            logger.info(f"SLURM vars not set (distributed training not available): {e}")
            world_size, rank = 1, 0
            return world_size, rank

    try:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        if port is None:
            port = _get_port(world_size)
        os.environ["MASTER_PORT"] = str(port)
        # Increase timeout for large-scale multi-node jobs
        # Also need longer timeout for mixed video+image training where different loaders
        # (e.g., Instagram video loader) can take a very long time to fetch the first sample
        nccl_timeout = None
        if nccl_timeout_minutes is not None:
            logger.info(f"Initializing distributed with timeout={nccl_timeout_minutes} minutes")
            nccl_timeout = datetime.timedelta(minutes=nccl_timeout_minutes)
        torch.distributed.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            world_size=world_size,
            rank=rank,
            timeout=nccl_timeout,
        )
    except Exception as e:
        logger.error(f"Rank {rank}: Distributed training initialization FAILED: {e}")
        logger.error("This is a fatal error for multi-GPU training. Check network connectivity.")
        # Re-raise the exception instead of silently continuing with world_size=1
        # This prevents confusing errors later when DDP fails
        raise RuntimeError(
            f"Failed to initialize distributed training: {e}. "
            f"Rank={rank}, World={world_size}, Master={os.environ.get('MASTER_ADDR')}"
        ) from e

    return world_size, rank


def is_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
        if key not in os.environ:
            return False
    return True


def get_local_rank() -> int:
    assert is_initialized()
    return int(os.environ["LOCAL_RANK"])


def get_global_rank() -> int:
    assert is_initialized()
    return int(os.environ["RANK"])


def get_world_size() -> int:
    assert is_initialized()
    return int(os.environ["WORLD_SIZE"])


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
