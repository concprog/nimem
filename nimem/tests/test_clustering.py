import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from nimem.core import clustering

@pytest.fixture
def mock_hdbscan():
    with patch('nimem.core.clustering.HDBSCAN') as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        # Mock fit_predict: return labels [0, 0, 1, -1] for 4 items
        instance.fit_predict.return_value = np.array([0, 0, 1, -1])
        yield instance

@pytest.fixture
def mock_embeddings():
    with patch('nimem.core.embeddings.embed_texts') as mock_emb:
        from returns.result import Success
        # Return Success containing the array
        mock_emb.return_value = Success(np.zeros((4, 10)))
        yield mock_emb

def test_perform_clustering(mock_hdbscan, mock_embeddings):
    texts = ["a", "b", "c", "noise"]
    res = clustering.perform_clustering(texts).unwrap()
    
    # Debug print
    print(f"Result type: {type(res)}")
    print(f"Result value: {res}")
    
    # Handle potential double-wrapping in mocks
    from returns.result import Success
    if isinstance(res, Success):
        res = res.unwrap()
        
    assert len(res) == 2
    assert 0 in res
    assert 1 in res
    assert -1 not in res
    assert len(res[0]) == 2
    
def test_topic_naming():
    name = clustering.generate_topic_name(["apple", "banana", "cherry"])
    assert "Topic:" in name
    assert "apple" in name
