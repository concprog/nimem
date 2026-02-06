from typing import List, Any, Dict
from returns.result import Result, safe
import time
import uuid
import logging

# FalkorDBLite python client
try:
    from redislite.falkordb_client import FalkorDB
except ImportError:
    FalkorDB = None

# --- Connection Management ---

def get_graph_client(db_path='./nimem.db', graph_name='nimem_memory'):
    if FalkorDB is None:
        raise ImportError("falkordblite library not installed")
    
    # FalkorDBLite works with a local file
    db = FalkorDB(db_path)
    return db.select_graph(graph_name)

# --- Bitemporal Logic ---

@safe
def add_fact(
    subject: str, 
    relation: str, 
    obj: str, 
    valid_at: float = None,
    graph_name: str = 'nimem_memory'
) -> bool:
    """
    Adds a fact to the graph with simple soft-delete metadata.
    """
    g = get_graph_client()
    
    if valid_at is None:
        valid_at = time.time()
    
    # Simple Soft-Delete Model
    # valid_at: when the fact became true
    # invalidated_at: when it ceased to be true (NULL if currently true)
    
    query = f"""
    MERGE (s:Entity {{name: $subject}})
    MERGE (o:Entity {{name: $obj}})
    CREATE (s)-[r:{relation.upper()} {{
        valid_at: {valid_at},
        id: $edge_id
    }}]->(o)
    RETURN count(r)
    """
    
    params = {
        'subject': subject,
        'obj': obj,
        'edge_id': str(uuid.uuid4())
    }
    
    result = g.query(query, params)
    return len(result.result_set) > 0

@safe
def expire_facts(
    subject: str,
    relation: str,
    invalidated_at: float = None,
    graph_name: str = 'nimem_memory'
) -> int:
    """
    Expires existing active facts by setting invalidated_at.
    """
    g = get_graph_client()
    if invalidated_at is None:
        invalidated_at = time.time()
    
    query = f"""
    MATCH (s:Entity {{name: $subject}})-[r:{relation.upper()}]->(o)
    WHERE r.invalidated_at IS NULL
    SET r.invalidated_at = {invalidated_at}
    RETURN count(r)
    """
    
    params = {
        'subject': subject
    }
    
    res = g.query(query, params)
    
    # Debug logging
    logging.info(f"EXPIRE DEBUG: Subject={subject}, Relation={relation}, InvalidatedAt={invalidated_at}")
    logging.info(f"EXPIRE DEBUG: Query=\n{query}")
    
    count = 0
    if res.result_set and len(res.result_set) > 0:
        count = res.result_set[0][0]
    
    logging.info(f"EXPIRE DEBUG: Result Count={count}")
    return count

@safe
def query_valid_facts(
    subject: str, 
    at_time: float = None,
    graph_name: str = 'nimem_memory'
) -> List[Dict[str, Any]]:
    """
    Queries facts about a subject that are active (not invalidated).
    """
    g = get_graph_client()
    
    # Simple current state query: invalidated_at must be NULL
    # If at_time is provided, we can also check history, but primarily we query current state.
    
    if at_time is None:
        query = f"""
        MATCH (s:Entity {{name: $subject}})-[r]->(o:Entity)
        WHERE r.invalidated_at IS NULL
        RETURN type(r) as relation, o.name as object
        """
    else:
        # Time travel query
        query = f"""
        MATCH (s:Entity {{name: $subject}})-[r]->(o:Entity)
        WHERE r.valid_at <= {at_time} 
          AND (r.invalidated_at IS NULL OR r.invalidated_at > {at_time})
        RETURN type(r) as relation, o.name as object
        """
    
    params = {
        'subject': subject
    }
    
    res = g.query(query, params)
    output = []
    for record in res.result_set:
        output.append({'relation': record[0], 'object': record[1]})
        
    return output

@safe
def get_all_entities(graph_name: str = 'nimem_memory') -> List[str]:
    """
    Retrieves all unique entity names from the graph.
    """
    g = get_graph_client()
    query = "MATCH (n:Entity) RETURN n.name"
    res = g.query(query)
    
    return [record[0] for record in res.result_set]
