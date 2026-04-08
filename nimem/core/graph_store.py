import logging
import re
import time
import uuid
from typing import List, Any, Dict

from redislite.falkordb_client import FalkorDB
from returns.result import safe

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "./nimem.db"
DEFAULT_GRAPH_NAME = "nimem_memory"


def get_graph_client(
    db_path: str = DEFAULT_DB_PATH, graph_name: str = DEFAULT_GRAPH_NAME
):
    db = FalkorDB(db_path)
    return db.select_graph(graph_name)


_RELATION_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _sanitize_relation(relation: str) -> str:
    upper = relation.upper()
    if not _RELATION_RE.match(upper):
        raise ValueError(f"Invalid relation name: {relation}")
    return upper


@safe
def add_fact(
    subject: str,
    relation: str,
    obj: str,
    valid_at: float | None = None,
    db_path: str = DEFAULT_DB_PATH,
    graph_name: str = DEFAULT_GRAPH_NAME,
) -> bool:
    """Adds a fact to the graph with soft-delete metadata."""
    g = get_graph_client(db_path, graph_name)
    safe_rel = _sanitize_relation(relation)

    if valid_at is None:
        valid_at = time.time()

    query = f"""
    MERGE (s:Entity {{name: $subject}})
    MERGE (o:Entity {{name: $obj}})
    CREATE (s)-[r:{safe_rel} {{
        valid_at: {valid_at},
        id: $edge_id
    }}]->(o)
    RETURN count(r)
    """

    params = {"subject": subject, "obj": obj, "edge_id": str(uuid.uuid4())}
    result = g.query(query, params)
    return len(result.result_set) > 0


@safe
def expire_facts(
    subject: str,
    relation: str,
    invalidated_at: float | None = None,
    db_path: str = DEFAULT_DB_PATH,
    graph_name: str = DEFAULT_GRAPH_NAME,
) -> int:
    """Expires existing active facts by setting invalidated_at."""
    g = get_graph_client(db_path, graph_name)
    safe_rel = _sanitize_relation(relation)

    if invalidated_at is None:
        invalidated_at = time.time()

    query = f"""
    MATCH (s:Entity {{name: $subject}})-[r:{safe_rel}]->(o)
    WHERE r.invalidated_at IS NULL
    SET r.invalidated_at = {invalidated_at}
    RETURN count(r)
    """

    params = {"subject": subject}
    res = g.query(query, params)

    count = 0
    if res.result_set and len(res.result_set) > 0:
        count = res.result_set[0][0]

    logger.debug(f"Expired {count} facts for {subject}/{relation}")
    return count


@safe
def query_valid_facts(
    subject: str,
    at_time: float | None = None,
    db_path: str = DEFAULT_DB_PATH,
    graph_name: str = DEFAULT_GRAPH_NAME,
) -> List[Dict[str, Any]]:
    """Queries facts about a subject that are active (not invalidated)."""
    g = get_graph_client(db_path, graph_name)

    if at_time is None:
        query = """
        MATCH (s:Entity {name: $subject})-[r]->(o:Entity)
        WHERE r.invalidated_at IS NULL
        RETURN type(r) as relation, o.name as object
        """
    else:
        query = f"""
        MATCH (s:Entity {{name: $subject}})-[r]->(o:Entity)
        WHERE r.valid_at <= {at_time} 
          AND (r.invalidated_at IS NULL OR r.invalidated_at > {at_time})
        RETURN type(r) as relation, o.name as object
        """

    params = {"subject": subject}
    res = g.query(query, params)
    output = []
    for record in res.result_set:
        output.append({"relation": record[0], "object": record[1]})

    return output


@safe
def get_all_entities(
    db_path: str = DEFAULT_DB_PATH, graph_name: str = DEFAULT_GRAPH_NAME
) -> List[str]:
    """Retrieves all unique entity names from the graph."""
    g = get_graph_client(db_path, graph_name)
    query = "MATCH (n:Entity) RETURN n.name"
    res = g.query(query)
    return [record[0] for record in res.result_set]
