import logging
from typing import Dict, List

from returns.result import Result, Success, Failure

from .core import text_processing
from .core import embeddings
from .core import graph_store
from .core import clustering
from .core import schema

logger = logging.getLogger(__name__)


def ingest_text(text: str, use_coref: bool = False) -> Result[str, Exception]:
    """
    Ingest text by extracting triplets and storing them in the graph.

    Args:
        text: Input text to process
        use_coref: If True, resolve coreferences before extraction (slower, requires FastCoref)
    """
    processed: Result = text_processing.process_text_pipeline(text, use_coref=use_coref)

    def store_triplets(data) -> Result[str, Exception]:
        resolved_text, triplets = data
        logger.info(f"Resolved Text: {resolved_text}")
        logger.info(f"Found Triplets: {len(triplets)}")

        count = 0
        errors = []
        for tri in triplets:
            logger.info(f"Adding: {tri.subject} -[{tri.relation}]-> {tri.object}")

            cardinality = schema.CARDINALITY.get(tri.relation, "MANY")
            if cardinality == "ONE":
                logger.info(f"Relation '{tri.relation}' is 1-to-1. Expiring old facts.")
                res_expire = graph_store.expire_facts(tri.subject, tri.relation)
                if isinstance(res_expire, Failure):
                    logger.warning(f"Failed to expire facts: {res_expire}")

            res = graph_store.add_fact(tri.subject, tri.relation, tri.object)
            if isinstance(res, Success):
                count += 1
            else:
                errors.append(str(res.failure()))

        if errors and count == 0:
            return Failure(RuntimeError(f"All {len(errors)} facts failed: {errors}"))

        if errors:
            logger.warning(f"Some facts failed to store: {errors}")

        return Success(
            f"Ingested {count} facts. (Resolved text: {resolved_text[:50]}...)"
        )

    return processed.bind(store_triplets)


def add_memory(subject: str, relation: str, obj: str) -> Result[bool, Exception]:
    """Directly adds a memory fact."""
    return graph_store.add_fact(subject, relation, obj)


def recall_memory(subject: str) -> Result[list, Exception]:
    """Recalls facts about a subject."""
    return graph_store.query_valid_facts(subject)


def consolidate_topics() -> Result[str, Exception]:
    """
    Clusters entity names in the graph to find topics.
    Creates 'BELONGS_TO' edges from Entities to Topic nodes.
    """
    entities_res = graph_store.get_all_entities()

    def embed_and_cluster(entities: List[str]):
        return embeddings.embed_texts(entities).bind(
            lambda vectors: clustering.perform_clustering(vectors, entities)
        )

    return entities_res.bind(embed_and_cluster).map(_process_clusters)


def _process_clusters(clusters: Dict[int, List[str]]) -> str:
    count = 0
    topic_count = 0
    for label, items in clusters.items():
        if label == -1:
            continue
        topic_name = clustering.generate_topic_name(items)
        logger.info(f"Found Cluster '{topic_name}': {items}")
        topic_count += 1

        for item in items:
            res = graph_store.add_fact(item, "BELONGS_TO", topic_name)
            if isinstance(res, Success):
                count += 1
            else:
                logger.warning(f"Failed to add cluster relation for {item}: {res}")

    return f"Consolidated {count} weak relations into {topic_count} topics."
