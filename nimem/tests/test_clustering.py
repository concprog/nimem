import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from nimem.domain import graph_ops as clustering


@pytest.fixture
def mock_hdbscan():
    with patch("nimem.domain.graph_ops.HDBSCAN") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        instance.fit_predict.return_value = np.array([0, 0, 1, -1])
        yield instance


def test_perform_clustering(mock_hdbscan):
    texts = ["a", "b", "c", "noise"]
    vectors = np.zeros((4, 10))
    res = clustering.perform_clustering(vectors, texts).unwrap()

    assert len(res) == 2
    assert 0 in res
    assert 1 in res
    assert -1 not in res
    assert len(res[0]) == 2


def test_perform_clustering_empty(mock_hdbscan):
    res = clustering.perform_clustering(np.array([]), []).unwrap()
    assert res == {}


def test_topic_naming():
    name = clustering.generate_topic_name(["apple", "banana", "cherry"])
    assert "Topic:" in name
    assert "apple" in name
