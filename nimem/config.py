"""
Centralized configuration for nimem.

All paths, URLs, and tunable defaults live here.
Environment variables override defaults where noted.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache root  (Task 2)
# ---------------------------------------------------------------------------
# Everything nimem downloads/caches goes under this directory.
# Override with NIMEM_CACHE_DIR env var.
NIMEM_CACHE_DIR: Path = Path(
    os.environ.get("NIMEM_CACHE_DIR", ".nimem_cache")
).resolve()

# Set HF_HOME *before* any HuggingFace / torch import has a chance to read it.
_HF_HOME = str(NIMEM_CACHE_DIR / "huggingface")
os.environ.setdefault("HF_HOME", _HF_HOME)

# infinity-emb vector disk cache
INFINITY_CACHE_DIR: str = str(NIMEM_CACHE_DIR / "infinity")

# ---------------------------------------------------------------------------
# ConceptNet  (Task 1)
# ---------------------------------------------------------------------------
CONCEPTNET_CSV_URL: str = os.environ.get(
    "CONCEPTNET_CSV_URL",
    "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
    "conceptnet-assertions-5.7.0.csv.gz",
)

# Where the downloaded CSV.gz is stored
CONCEPTNET_CSV_PATH: Path = (
    NIMEM_CACHE_DIR / "conceptnet" / "conceptnet-assertions-5.7.0.csv.gz"
)

# FalkorDB paths for the local ConceptNet graph
CONCEPTNET_DB_PATH: str = os.environ.get(
    "CONCEPTNET_DB_PATH",
    str(NIMEM_CACHE_DIR / "conceptnet" / "conceptnet.db"),
)
CONCEPTNET_GRAPH_NAME: str = os.environ.get("CONCEPTNET_GRAPH_NAME", "conceptnet")

# Language filter for CSV import (set to "" to load all languages)
CONCEPTNET_LANGUAGE_FILTER: str = os.environ.get("CONCEPTNET_LANGUAGE_FILTER", "en")

# Maximum edges to load (None = all).  Useful for testing.
_max = os.environ.get("CONCEPTNET_MAX_EDGES")
CONCEPTNET_MAX_EDGES: int | None = int(_max) if _max else None

# Whether to fall back to the public ConceptNet API when the local DB
# returns no results.
CONCEPTNET_API_FALLBACK: bool = (
    os.environ.get("CONCEPTNET_API_FALLBACK", "true").lower() == "true"
)
CONCEPTNET_API_URL: str = os.environ.get(
    "CONCEPTNET_API_URL", "http://api.conceptnet.io"
)

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------
SPACY_MODEL: str = os.environ.get("NIMEM_SPACY_MODEL", "en_core_web_md")
GLINER_MODEL: str = os.environ.get("NIMEM_GLINER_MODEL", "fastino/gliner2-multi-v1")
EMBEDDING_MODEL: str = os.environ.get(
    "NIMEM_EMBEDDING_MODEL", "michaelfeil/bge-small-en-v1.5"
)

# ---------------------------------------------------------------------------
# Graph store
# ---------------------------------------------------------------------------
NIMEM_DB_PATH: str = os.environ.get("NIMEM_DB_PATH", "./nimem.db")
NIMEM_GRAPH_NAME: str = os.environ.get("NIMEM_GRAPH_NAME", "nimem_memory")
