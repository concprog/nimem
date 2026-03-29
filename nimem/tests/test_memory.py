import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from returns.result import Success
from nimem.domain.schema import Triple
from nimem import memory


@pytest.fixture
def mock_text_pipeline():
    with patch("nimem.pipelines.ingest.process_text_pipeline") as mock:
        triplets = [
            Triple("Alice", "works_for", "Google"),
            Triple("Bob", "knows", "Alice"),
        ]
        mock.return_value = Success(("Alice works...", triplets))
        yield mock


@pytest.fixture
def mock_graph_add():
    with patch("nimem.storage.graph_store.add_fact") as mock:
        mock.return_value = Success(True)
        yield mock


@pytest.fixture
def mock_graph_expire():
    with patch("nimem.storage.graph_store.expire_facts") as mock:
        mock.return_value = Success(0)
        yield mock


@pytest.fixture
def mock_consolidate_deps():
    with (
        patch("nimem.storage.graph_store.get_all_entities") as mock_ents,
        patch("nimem.embeddings.infinity.embed") as mock_embed,
        patch("nimem.domain.graph_ops.perform_clustering") as mock_cluster,
        patch("nimem.domain.graph_ops.generate_topic_name") as mock_topic,
    ):
        mock_ents.return_value = Success(["Alice", "Bob"])
        mock_embed.return_value = Success(np.zeros((2, 10)))
        mock_cluster.return_value = Success({0: ["Alice", "Bob"]})
        mock_topic.return_value = "Topic: Friends"
        yield {
            "ents": mock_ents,
            "embed": mock_embed,
            "cluster": mock_cluster,
            "topic": mock_topic,
        }


def test_ingest_text_flow(mock_text_pipeline, mock_graph_add):
    res = memory.ingest_text("Source Text").unwrap()
    assert "Ingested 2 facts" in res

    mock_text_pipeline.assert_called_with("Source Text", use_coref=False)
    assert mock_graph_add.call_count == 2


def test_ingest_cardinality_one(mock_text_pipeline, mock_graph_add, mock_graph_expire):
    mock_text_pipeline.return_value = Success(
        ("Txt", [Triple("Alice", "located_in", "Paris")])
    )

    res = memory.ingest_text("Alice is in Paris").unwrap()

    mock_graph_expire.assert_called_with("Alice", "located_in")
    mock_graph_add.assert_called_with("Alice", "located_in", "Paris")


def test_consolidate_topics(mock_graph_add, mock_consolidate_deps):
    res = memory.consolidate_topics().unwrap()
    assert "Consolidated" in res

    mock_graph_add.assert_any_call("Alice", "BELONGS_TO", "Topic: Friends")
