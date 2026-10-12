import os


def data_root(path=None):
    root = path or os.environ.get("LALA_DATA_ROOT") or "data"
    return os.path.abspath(os.path.expanduser(root))


def add_root_arg(ap):
    ap.add_argument("--root", default=None, help="data root (default: $LALA_DATA_ROOT or ./data)")
