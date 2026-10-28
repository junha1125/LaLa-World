"""Launch one training process per device:  python -m app.main --fname <config.yaml> --devices cuda:0 cuda:1 ...
`--debug` runs a single in-process rank. `${VAR}` in the config is expanded from the environment
(e.g. ${LALA_DATA_ROOT}).
"""

import argparse
import multiprocessing as mp
import os
import pprint
from pathlib import Path

import yaml

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed
from src.utils.yaml_utils import expand_env_vars

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, required=True, help="config file")
parser.add_argument("--devices", type=str, nargs="+", default=["cuda:0"], help="one process per device")
parser.add_argument("--debug", action="store_true", help="single in-process rank (no spawn)")


def load_config(fname):
    os.environ.setdefault("LALA_DATA_ROOT", "data")
    with open(fname, "r") as f:
        return expand_env_vars(yaml.load(f, Loader=yaml.FullLoader))


def process_main(rank, fname, world_size, devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])
    import logging

    from src.utils.logging import get_logger

    logger = get_logger(force=True)
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    params = load_config(fname)
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        Path(params["folder"]).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(params["folder"], "params-pretrain.yaml"), "w") as f:
            yaml.dump(params, f)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f"Running... (rank: {rank}/{world_size})")
    app_main(params["app"], args=params)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        process_main(rank=0, fname=args.fname, world_size=1, devices=args.devices[:1])
    else:
        mp.set_start_method("spawn")
        from torch.distributed.elastic.utils.distributed import get_free_port

        os.environ.setdefault("MASTER_PORT", str(get_free_port()))
        procs = [
            mp.Process(target=process_main, args=(rank, args.fname, len(args.devices), args.devices))
            for rank in range(len(args.devices))
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        raise SystemExit(max(p.exitcode or 0 for p in procs))
