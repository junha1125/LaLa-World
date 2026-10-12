import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import data_root  # noqa: E402


def openvid_root():
    """<root>/openvid_raw; --root is consumed here so every script shares one flag."""
    root = None
    if "--root" in sys.argv:
        i = sys.argv.index("--root")
        root = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    return os.environ.get("OPENVID_ROOT") or os.path.join(data_root(root), "openvid_raw")
