import pytest
from unittest.mock import patch, MagicMock
from returns.result import Success
from nimem import memory, schema

@pytest.fixture
def mock_deps():
    with patch('nimem.text_processing.process_text_pipeline') as mock_proc, \
         patch('nimem.graph_store.add_fact') as mock_add, \
         patch('nimem.graph_store.expire_facts') as mock_expire, \
         patch('nimem.graph_store.get_all_entities') as mock_get_ents, \
         patch('nimem.clustering.perform_clustering') as mock_cluster, \
         patch('nimem.clustering.generate_topic_name') as mock_topic_name:
         
         # Setup Text Processing
         from nimem.text_processing import Triple
         triplets = [
             Triple("Alice", "works_for", "Google"),
             Triple("Bob", "knows", "Alice")
         ]
         mock_proc.return_value = Success(("Alice works...", triplets))
         
         # Setup Graph Store
         mock_add.return_value = Success(True)
         mock_expire.return_value = Success(0)
         mock_get_ents.return_value = Success(["Alice", "Bob"])
         
         # Setup Clustering
         mock_cluster.return_value = Success({0: ["Alice", "Bob"]})
         mock_topic_name.return_value = "Topic: Friends"
         
         yield {
             'proc': mock_proc,
             'add': mock_add,
             'expire': mock_expire
         }

def test_ingest_text_flow(mock_deps):
    res = memory.ingest_text("Source Text").unwrap()
    assert "Ingested 2 facts" in res
    
    # Verify calls
    mock_deps['proc'].assert_called_with("Source Text")
    assert mock_deps['add'].call_count == 2
    
    # Verify cardinality check for works_for (MANY) - should NOT expire
    # schema.CARDINALITY['works_for'] is MANY.
    # Wait, 'works_for' is MANY. 'located_in' is ONE.
    # Let's validte logic. API says: checks schema.
    mock_deps['expire'].assert_not_called()

def test_ingest_cardinality_one(mock_deps):
    # Change mock to deliver a 'located_in' relation
    from nimem.text_processing import Triple
    mock_deps['proc'].return_value = Success(("Txt", [Triple("Alice", "located_in", "Paris")]))
    
    res = memory.ingest_text("Alice is in Paris").unwrap()
    
    # Should call expire
    mock_deps['expire'].assert_called_with("Alice", "located_in")
    mock_deps['add'].assert_called_with("Alice", "located_in", "Paris")

def test_consolidate_topics(mock_deps):
    res = memory.consolidate_topics().unwrap()
    assert "Consolidated" in res
    # Topic name is not in the return string, only summary
    # assert "Topic: Friends" in res
    
    # Verify it added weak relations
    mock_deps['add'].assert_any_call("Alice", "BELONGS_TO", "Topic: Friends")
