import pytest
import os
import uuid
import time
from nimem.storage import graph_store

TEST_GRAPH = "test_memory"


@pytest.fixture
def clean_db():
    fake_db = f"./test_nimem_{uuid.uuid4().hex}.db"
    yield fake_db
    if os.path.exists(fake_db):
        try:
            os.remove(fake_db)
        except OSError:
            pass


def test_add_and_query_fact(clean_db):
    FAKE_DB = clean_db
    res = graph_store.add_fact(
        "Alice", "works_for", "Google", db_path=FAKE_DB, graph_name=TEST_GRAPH
    ).unwrap()
    assert res is True

    facts = graph_store.query_valid_facts(
        "Alice", db_path=FAKE_DB, graph_name=TEST_GRAPH
    ).unwrap()
    assert len(facts) == 1
    assert facts[0]["relation"] == "WORKS_FOR"
    assert facts[0]["object"] == "Google"


def test_expire_facts(clean_db):
    FAKE_DB = clean_db
    graph_store.add_fact(
        "Alice",
        "located_in",
        "London",
        valid_at=0,
        db_path=FAKE_DB,
        graph_name=TEST_GRAPH,
    )

    time.sleep(0.1)

    count = graph_store.expire_facts(
        "Alice", "located_in", db_path=FAKE_DB, graph_name=TEST_GRAPH
    ).unwrap()
    assert count == 1

    facts = graph_store.query_valid_facts(
        "Alice", db_path=FAKE_DB, graph_name=TEST_GRAPH
    ).unwrap()
    assert len(facts) == 0

    g = graph_store.get_graph_client(db_path=FAKE_DB, graph_name=TEST_GRAPH)
    res = g.query("MATCH ()-[r]->() RETURN r.invalidated_at")
    val = res.result_set[0][0]
    assert val is not None


def test_bitemporality_query(clean_db):
    FAKE_DB = clean_db
    past_time = time.time() - 1000
    graph_store.add_fact(
        "Bob",
        "knows",
        "Alice",
        valid_at=past_time,
        db_path=FAKE_DB,
        graph_name=TEST_GRAPH,
    )

    facts = graph_store.query_valid_facts(
        "Bob", at_time=past_time + 1, db_path=FAKE_DB, graph_name=TEST_GRAPH
    ).unwrap()
    assert len(facts) == 1

    facts_too_early = graph_store.query_valid_facts(
        "Bob", at_time=past_time - 1, db_path=FAKE_DB, graph_name=TEST_GRAPH
    ).unwrap()
    assert len(facts_too_early) == 0
