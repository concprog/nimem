import asyncio
import logging
from typing import List

import numpy as np
from infinity_emb import AsyncEmbeddingEngine, EngineArgs
from infinity_emb.primitives import InferenceEngine
from returns.result import safe

from nimem.config import EMBEDDING_MODEL, INFINITY_CACHE_DIR
from nimem.cache_layer import setup_cache_directories

logger = logging.getLogger(__name__)

_engine_instance = None
_loop = None


def get_engine():
    global _engine_instance
    if _engine_instance is None:
        setup_cache_directories()
        logger.info("Initializing embedding engine (%s)", EMBEDDING_MODEL)
        engine_args = EngineArgs(
            model_name_or_path=EMBEDDING_MODEL,
            engine=InferenceEngine.torch,
            bettertransformer=False,
            vector_disk_cache_path=INFINITY_CACHE_DIR,
        )
        _engine_instance = AsyncEmbeddingEngine.from_args(engine_args)
    return _engine_instance


def _get_or_create_loop():
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop


async def _start_engine_async():
    engine = get_engine()
    await engine.astart()
    logger.info("Embedding engine started")


def _start_engine():
    loop = _get_or_create_loop()
    if not loop.is_running():
        loop.run_until_complete(_start_engine_async())


async def _embed_async(texts: List[str]) -> np.ndarray:
    engine = get_engine()
    embeddings, _ = await engine.embed(texts)
    return np.array(embeddings)


@safe
def embed(texts: List[str]) -> np.ndarray:
    """Embeds a list of texts using Infinity-emb."""
    _start_engine()
    loop = _get_or_create_loop()
    return loop.run_until_complete(_embed_async(texts))
