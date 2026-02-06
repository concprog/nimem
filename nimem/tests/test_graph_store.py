import pytest
import os
import time
from unittest.mock import patch
from nimem import graph_store

FAKE_DB = './test_nimem.db'

@pytest.fixture
def clean_db():
    if os.path.exists(FAKE_DB):
        os.remove(FAKE_DB)
    yield
    if os.path.exists(FAKE_DB):
        os.remove(FAKE_DB)

@pytest.fixture
def patch_db_path():
    with patch('nimem.graph_store.get_graph_client') as mock_get:
        # We perform the real logic but with our path
        # We need to import FalkorDB inside the patch scope or use the original reference
        from redislite.falkordb_client import FalkorDB
        def fake_get(db_path=None, graph_name=None):
            db = FalkorDB(FAKE_DB)
            return db.select_graph('test_memory')
        mock_get.side_effect = fake_get
        yield

def test_add_and_query_fact(clean_db, patch_db_path):
    # Add
    res = graph_store.add_fact("Alice", "works_for", "Google").unwrap()
    assert res is True
    
    # Query
    facts = graph_store.query_valid_facts("Alice").unwrap()
    assert len(facts) == 1
    assert facts[0]['relation'] == 'WORKS_FOR'
    assert facts[0]['object'] == 'Google'

def test_expire_facts(clean_db, patch_db_path):
    # Use explicit valid_at to avoid clock skew issues
    res = graph_store.add_fact("Alice", "located_in", "London", valid_at=0)
    
    # Ensure persistence/indexing
    time.sleep(0.1)
    
    # Expire
    count = graph_store.expire_facts("Alice", "located_in").unwrap()
    assert count == 1
    
    # Verify expired (should return nothing for current time)
    facts = graph_store.query_valid_facts("Alice").unwrap()
    assert len(facts) == 0
    
    # Verify soft-delete property (manual check)
    g = graph_store.get_graph_client()
    # Check that invalidated_at is NOT NULL
    res = g.query("MATCH ()-[r]->() RETURN r.invalidated_at")
    val = res.result_set[0][0]
    assert val is not None

def test_bitemporality_query(clean_db, patch_db_path):
    # Add fact valid in the past
    past_time = time.time() - 1000
    graph_store.add_fact("Bob", "knows", "Alice", valid_at=past_time)
    
    # Query at past time
    facts = graph_store.query_valid_facts("Bob", at_time=past_time + 1).unwrap()
    assert len(facts) == 1
    
    # Query before valid time
    facts_too_early = graph_store.query_valid_facts("Bob", at_time=past_time - 1).unwrap()
    assert len(facts_too_early) == 0
