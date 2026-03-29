"""
Cache layer for managing local file caches, models, and graph DBs.
"""

import logging
from pathlib import Path
from typing import Callable, Any

from .config import NIMEM_CACHE_DIR

logger = logging.getLogger(__name__)


def ensure_directory(path: Path | str) -> Path:
    """Ensure the directory exists."""
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p


def get_cached_resource(
    resource_id: str,
    target_path: Path | str,
    loader_func: Callable[[], Any],
) -> Any:
    """
    Checks if a target_path exists.
    If it exists, ensures its parent directory exists (just in case) and returns it.
    If not, ensures the parent directory exists and executes the loader_func to create/fetch it.
    """
    p = Path(target_path)

    # Always ensure the parent directory of the target path exists
    ensure_directory(p.parent)

    if p.exists():
        logger.debug(f"Cache hit for {resource_id} at {p}")
        return p

    logger.info(f"Cache miss for {resource_id}, executing loader...")
    result = loader_func()

    # We return the target path assuming the loader function populated it.
    return p


def setup_cache_directories():
    """Setup core cache directories."""
    ensure_directory(NIMEM_CACHE_DIR / "conceptnet")
    ensure_directory(NIMEM_CACHE_DIR / "huggingface")
    ensure_directory(NIMEM_CACHE_DIR / "infinity")


# Setup immediately upon import
setup_cache_directories()
