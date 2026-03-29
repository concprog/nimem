import logging
from functools import lru_cache
from typing import Callable, Any

from .cache_layer import get_cached_resource, setup_cache_directories

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: dict[str, Callable[[], Any]] = {}


def register(name: str) -> Callable[[], Any]:
    """Decorator to register a model loader."""

    def decorator(loader: Callable[[], Any]) -> Callable[[], Any]:
        _MODEL_REGISTRY[name] = loader
        return loader

    return decorator


def _load_spacy():
    import spacy
    from .config import SPACY_MODEL

    logger.info(f"Loading spaCy model: {SPACY_MODEL}")
    try:
        return spacy.load(SPACY_MODEL)
    except OSError:
        logger.warning(f"spaCy model {SPACY_MODEL} not found, downloading...")
        from spacy.cli import download

        download(SPACY_MODEL)
        return spacy.load(SPACY_MODEL)


def _load_gliner():
    from gliner2 import GLiNER2
    from .config import GLINER_MODEL

    logger.info(f"Loading GLiNER model: {GLINER_MODEL}")
    return GLiNER2.from_pretrained(GLINER_MODEL)


def _load_fastcoref():
    from fastcoref import FCoref

    logger.info("Loading FastCoref model")
    return FCoref(device="cpu")


# ---------------------------------------------------------------------------
# ConceptNet local graph  (loaded from CSV → FalkorDB)
# ---------------------------------------------------------------------------


def _download_conceptnet_csv() -> str:
    """Download the ConceptNet assertions CSV if not already cached.

    Returns the local file path.
    """
    from .config import CONCEPTNET_CSV_URL, CONCEPTNET_CSV_PATH

    setup_cache_directories()

    csv_path = CONCEPTNET_CSV_PATH
    if csv_path.exists():
        logger.info("ConceptNet CSV already cached at %s", csv_path)
        return str(csv_path)

    logger.info("Downloading ConceptNet CSV from %s ...", CONCEPTNET_CSV_URL)

    import urllib.request
    import shutil

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(".tmp")

    try:
        with (
            urllib.request.urlopen(CONCEPTNET_CSV_URL) as resp,
            open(tmp_path, "wb") as out,
        ):
            shutil.copyfileobj(resp, out)
        tmp_path.rename(csv_path)
        logger.info("ConceptNet CSV saved to %s", csv_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return str(csv_path)


def _load_conceptnet():
    """Load (and optionally download + import) the local ConceptNet graph.

    Returns the FalkorDB graph handle used by ``conceptnet.py``.
    """
    from pathlib import Path
    from .config import (
        CONCEPTNET_DB_PATH,
        CONCEPTNET_GRAPH_NAME,
        CONCEPTNET_LANGUAGE_FILTER,
        CONCEPTNET_MAX_EDGES,
    )

    db_file = Path(CONCEPTNET_DB_PATH)

    # If the DB already exists we just open it.
    if db_file.exists():
        logger.info("Opening existing ConceptNet DB at %s", CONCEPTNET_DB_PATH)
        from .conceptnet_loader import get_conceptnet_graph

        return get_conceptnet_graph(CONCEPTNET_DB_PATH, CONCEPTNET_GRAPH_NAME)

    # Otherwise download the CSV (if needed) and import it.
    csv_path = _download_conceptnet_csv()

    logger.info("Importing ConceptNet CSV into FalkorDB at %s ...", CONCEPTNET_DB_PATH)
    setup_cache_directories()
    db_file.parent.mkdir(parents=True, exist_ok=True)

    from .conceptnet_loader import load_conceptnet_csv, get_conceptnet_graph

    load_conceptnet_csv(
        csv_path=csv_path,
        db_path=CONCEPTNET_DB_PATH,
        graph_name=CONCEPTNET_GRAPH_NAME,
        language_filter=CONCEPTNET_LANGUAGE_FILTER,
        max_edges=CONCEPTNET_MAX_EDGES,
    )

    return get_conceptnet_graph(CONCEPTNET_DB_PATH, CONCEPTNET_GRAPH_NAME)


MODEL_LOADERS = {
    "spacy": _load_spacy,
    "gliner": _load_gliner,
    "fastcoref": _load_fastcoref,
    "conceptnet": _load_conceptnet,
}
_MODEL_REGISTRY.update(MODEL_LOADERS)


@lru_cache(maxsize=1)
def get_model(name: str) -> Any:
    """Factory - returns cached model instance."""
    loader = _MODEL_REGISTRY.get(name)
    if loader is None:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return loader()


def list_models() -> list[str]:
    """List all registered model names."""
    return list(_MODEL_REGISTRY.keys())
