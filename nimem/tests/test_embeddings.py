import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
from nimem.core import embeddings

@pytest.fixture
def mock_infinity():
    with patch('nimem.core.embeddings.AsyncEmbeddingEngine') as mock_engine, \
         patch('nimem.core.embeddings.EngineArgs') as mock_args:
        
        mock_instance = AsyncMock()
        mock_engine.from_args.return_value = mock_instance
        
        # Mock async context manager: async with engine -> returns engine
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.__aexit__.return_value = None
        
        # Mock embed return: (embeddings, usage)
        # 2 texts -> 2 vectors of size 3
        fake_vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_instance.embed.return_value = (fake_vectors, 10)
        
        yield mock_instance

def test_embed_texts(mock_infinity):
    texts = ["hello", "world"]
    try:
        res = embeddings.embed_texts(texts).unwrap()
    except Exception as e:
        print(f"\nUNWRAP FAILED WITH: {type(e)}")
        # If it's UnwrapFailedError, accessing .failure() needs the original Result object if we had it,
        # but here we can't easily access it unless we split step.
        # But wait, unwrap() raises UnwrapFailedError which has a .halted_container attribute usually
        # or we just print e.
        if hasattr(e, 'halted_container'):
             print(f"FAILURE CAUSE: {e.halted_container.failure()}")
        raise e
    
    assert res.shape == (2, 3)
    assert res[0, 0] == 0.1
    # Verify singleton behavior
    mock_infinity.embed.assert_called()
