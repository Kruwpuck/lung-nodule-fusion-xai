"""Cache-skip helper for stage scripts."""
import os


def cached(path: str) -> bool:
    """Return True if path exists and is non-empty."""
    return os.path.exists(path) and os.path.getsize(path) > 0
