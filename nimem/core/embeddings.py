import asyncio
import logging
from typing import List

import numpy as np
from infinity_emb import AsyncEmbeddingEngine, EngineArgs
from infinity_emb.primitives import InferenceEngine
from returns.result import safe

logger = logging.getLogger(__name__)


class EmbeddingService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Initializing embedding engine (michaelfeil/bge-small-en-v1.5)")
            engine_args = EngineArgs(
                model_name_or_path="michaelfeil/bge-small-en-v1.5",
                engine=InferenceEngine.torch,
                bettertransformer=False,
            )
            cls._instance = AsyncEmbeddingEngine.from_args(engine_args)
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None


async def _embed_async(texts: List[str]) -> np.ndarray:
    engine = EmbeddingService.get_instance()
    async with engine:
        embeddings, _ = await engine.embed(texts)
    return np.array(embeddings)


@safe
def embed_texts(texts: List[str]) -> np.ndarray:
    """Embeds a list of texts using Infinity-emb."""
    return asyncio.run(_embed_async(texts))
