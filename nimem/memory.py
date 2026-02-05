from returns.result import Result, Success, Failure
from .core import text_processing, embeddings, graph_store, clustering, schema
import logging
from typing import Dict, List

def ingest_text(text: str) -> Result[str, Exception]:
    processed: Result = text_processing.process_text_pipeline(text)
    def store_triplets(data) -> Result[str, Exception]:
        resolved_text, triplets = data
        logging.info(f"Resolved Text: {resolved_text}")
        logging.info(f"Found Triplets: {len(triplets)}")
        
        count = 0
        errors = []
        for tri in triplets:
             logging.info(f"Adding: {tri.subject} -[{tri.relation}]-> {tri.object}")
             
             # Check Cardinality using Schema
             cardinality = schema.CARDINALITY.get(tri.relation, "MANY")
             if cardinality == "ONE":
                 logging.info(f"Relation '{tri.relation}' is 1-to-1. Expiring old facts.")
                 graph_store.expire_facts(tri.subject, tri.relation)
             
             # We execute safely here. 
             res = graph_store.add_fact(tri.subject, tri.relation, tri.object)
             if isinstance(res, Success):
                 count += 1
             else:
                 errors.append(str(res.failure()))
        
        if errors:
            logging.warning(f"Some facts failed to store: {errors}")
            
        return Success(f"Ingested {count} facts. (Resolved text: {resolved_text[:50]}...)")

    return processed.bind(store_triplets)

def add_memory(subject: str, relation: str, obj: str) -> Result[bool, Exception]:
    """
    Directly adds a memory fact.
    """
    return graph_store.add_fact(subject, relation, obj)

def recall_memory(subject: str) -> Result[list, Exception]:
    """
    Recalls facts about a subject.
    """
    return graph_store.query_valid_facts(subject)

def consolidate_topics() -> Result[str, Exception]:
    """
    Performs clustering on all entity names in the graph to find 'weak' relations (Topics).
    This creates 'BELONGS_TO' edges from Entities to new Topic nodes.
    """
    entities_res = graph_store.get_all_entities()
    
    # Composition: graph_entities -> clustering
    return entities_res.bind(clustering.perform_clustering).map(
        lambda clusters: _process_clusters(clusters)
    )

def _process_clusters(clusters: Dict[int, List[str]]) -> str:
    count = 0
    topic_count = 0
    for label, items in clusters.items():
        if label == -1: continue # Noise
        topic_name = clustering.generate_topic_name(items)
        logging.info(f"Found Cluster '{topic_name}': {items}")
        topic_count += 1
        
        for item in items:
            # Add weak relation - safe to ignore specific errors here for aggregate report
            graph_store.add_fact(item, "BELONGS_TO", topic_name)
            count += 1
            
    return f"Consolidated {count} weak relations into {topic_count} topics."
