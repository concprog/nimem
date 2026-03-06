"""
Nimem - A memory system with NLP-powered fact extraction and graph storage.
"""

__version__ = "0.1.0"

__all__ = [
    "ingest_text",
    "add_memory",
    "recall_memory",
    "consolidate_topics",
]


def __getattr__(name):
    """Lazy import to keep 'import nimem' fast."""
    if name in __all__:
        from . import memory

        return getattr(memory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
