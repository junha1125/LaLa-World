import importlib
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(app, args, resume_preempt=False):
    logger.info(f"Running app: {app}")
    return importlib.import_module(f"app.{app}.train").main(args=args, resume_preempt=resume_preempt)
