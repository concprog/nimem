import pytest
from unittest.mock import patch, AsyncMock
import numpy as np
from nimem.embeddings import infinity as embeddings


@pytest.fixture
def mock_infinity():
    embeddings._engine_instance = None
    with (
        patch("nimem.embeddings.infinity.AsyncEmbeddingEngine") as mock_engine,
        patch("nimem.embeddings.infinity.EngineArgs"),
    ):
        mock_instance = AsyncMock()
        mock_engine.from_args.return_value = mock_instance

        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.__aexit__.return_value = None

        fake_vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_instance.embed.return_value = (fake_vectors, 10)

        yield mock_instance
    embeddings._engine_instance = None


def test_embed_texts(mock_infinity):
    texts = ["hello", "world"]
    res = embeddings.embed(texts).unwrap()

    assert res.shape == (2, 3)
    assert res[0, 0] == 0.1
    mock_infinity.embed.assert_called()
