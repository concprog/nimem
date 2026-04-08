import logging
from functools import lru_cache
from typing import Callable, Any

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
    from .schema import SPACY_MODEL

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

    logger.info("Loading GLiNER model: fastino/gliner2-multi-v1")
    return GLiNER2.from_pretrained("fastino/gliner2-multi-v1")


def _load_fastcoref():
    from fastcoref import FCoref

    logger.info("Loading FastCoref model")
    return FCoref(device="cpu")


MODEL_LOADERS = {
    "spacy": _load_spacy,
    "gliner": _load_gliner,
    "fastcoref": _load_fastcoref,
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
