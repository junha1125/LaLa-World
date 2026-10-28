"""Data locations used by the planning evaluations ($LALA_DATA_ROOT = the data root)."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA_ROOT = os.environ.get("LALA_DATA_ROOT", "data")
TRAJ_JSONL = os.path.join(DATA_ROOT, "jsonl", "traj")
SIM_RAW = os.path.join(DATA_ROOT, "sim_raw")
EVAL_ONLY = os.path.join(DATA_ROOT, "eval_only")
SPEC_DIR = os.environ.get("LALA_SPEC_DIR", os.path.join(REPO_ROOT, "data_preparation", "specs"))
LANG_BINS_DIR = os.environ.get("LALA_LANG_BINS_DIR", os.path.join(REPO_ROOT, "data_preparation", "lang_bins"))
TARGETS_DIR = os.path.join(HERE, "targets")
DEFAULT_OUT = os.environ.get("LALA_PLANNING_OUT", "logs/planning")
