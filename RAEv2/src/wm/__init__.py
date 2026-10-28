import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # RAEv2/
LALA_ROOT = os.path.dirname(ROOT)  # the LaLa-World repository


def abs_path(path):
    """Paths in the configs are relative to the RAEv2 directory."""
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)
