from typing import List
import numpy as np
from returns.result import Result, safe
from functools import lru_cache
import logging

import asyncio

try:
    # Depending on how infinity is installed, it might be used as a library or server.
    # The user notes mention "Infinity-emb" library usage.
    # We'll assume a direct library usage via `infinity_emb.Infinity` or similar if available,
    # or fall back to sentence-transformers if strictly needed, but NOTES says Infinity.
    from infinity_emb import EngineArgs, AsyncEmbeddingEngine
except ImportError:
    logging.warning("infinity_emb not found.")
    AsyncEmbeddingEngine = None
    EngineArgs = None

# --- Lazy Loading ---

class EmbeddingService:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            if AsyncEmbeddingEngine is None:
               raise ImportError("Infinity-emb not installed")
            
            # Using synchronous wrapper logic or running loop for the async engine
            # Ideally we run this in an async context, but for a simple library function
            # we might block.
            engine_args = EngineArgs(model_name_or_path="michaelfeil/bge-small-en-v1.5", engine="optimum") 
            cls._instance = AsyncEmbeddingEngine.from_args(engine_args)
        return cls._instance

# --- Functional Interface ---

# Since Infinity is async native, we need to decide how to expose it to sync code.
# The user asked for pragmatic. Using asyncio.run inside a library function is a bit anti-pattern
# if the caller is async, but for a script it's fine.

async def _embed_async(texts: List[str]) -> np.ndarray:
    engine = EmbeddingService.get_instance()
    async with engine: 
        embeddings, _ = await engine.embed(texts)
    return embeddings

@safe
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embeds a list of texts using Infinity-emb.
    Returns Result[np.ndarray, Exception].
    """
    # Pragmatic sync wrapper
    return asyncio.run(_embed_async(texts))
