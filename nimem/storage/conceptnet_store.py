import logging
from functools import lru_cache
from typing import List, Optional, Tuple

from nimem.config import (
    CONCEPTNET_API_FALLBACK,
    CONCEPTNET_API_URL,
    CONCEPTNET_DB_PATH,
    CONCEPTNET_GRAPH_NAME,
)
from nimem.cache_layer import setup_cache_directories
from nimem.domain.schema import Triple, Entity
from nimem.domain.graph_ops import _get_best_relation_via_similarity

logger = logging.getLogger(__name__)


def get_conceptnet_graph(
    db_path: str = CONCEPTNET_DB_PATH,
    graph_name: str = CONCEPTNET_GRAPH_NAME,
):
    """Get ConceptNet graph client."""
    from redislite.falkordb_client import FalkorDB

    setup_cache_directories()
    db = FalkorDB(db_path)
    return db.select_graph(graph_name)


def _normalize(text: str) -> str:
    """Normalize text for ConceptNet lookup."""
    return text.lower().replace(" ", "_").replace(".", "")


def _query_local_db(node: str, other: str, limit: int = 30) -> List[dict]:
    """Query local ConceptNet database."""
    try:
        g = get_conceptnet_graph()

        query = """
        MATCH (s:Concept)-[r]->(e:Concept)
        WHERE s.uri = $node AND e.uri = $other
        RETURN 
            type(r) as relation,
            r.weight as weight,
            s.uri as start_uri,
            e.uri as end_uri,
            r.dataset as dataset
        ORDER BY r.weight DESC
        LIMIT $limit
        """

        result = g.query(query, {"node": node, "other": other, "limit": limit})

        edges = []
        for row in result.result_set:
            relation_type, weight, start_uri, end_uri, dataset = row

            edge = {
                "rel": {"@id": f"/r/{relation_type}"},
                "start": {"@id": start_uri},
                "end": {"@id": end_uri},
                "weight": weight or 1.0,
                "dataset": dataset or "",
            }
            edges.append(edge)

        return edges

    except Exception as e:
        logger.warning(f"Local DB query failed: {e}")
        return []


def _query_api_fallback(node: str, other: str, limit: int = 30) -> List[dict]:
    """Fallback to ConceptNet API if local DB unavailable."""
    if not CONCEPTNET_API_FALLBACK:
        return []

    try:
        import requests

        response = requests.get(
            f"{CONCEPTNET_API_URL}/query",
            params={"node": node, "other": other, "limit": limit},
            timeout=5,
        )
        if response.status_code == 200:
            return response.json().get("edges", [])
    except Exception as e:
        logger.warning(f"API fallback failed: {e}")

    return []


@lru_cache(maxsize=1000)
def _query_conceptnet_cached(node: str, other: str, limit: int = 30) -> List[dict]:
    """Query ConceptNet (local DB first, API fallback)."""
    edges = _query_local_db(node, other, limit)

    if not edges:
        edges = _query_api_fallback(node, other, limit)

    return edges


def get_conceptnet_edges(type1: str, type2: str) -> List[dict]:
    """Get ConceptNet edges between two entity types."""
    node = f"/c/en/{_normalize(type1)}"
    other = f"/c/en/{_normalize(type2)}"

    edges = _query_conceptnet_cached(node, other)

    if not edges:
        edges = _query_conceptnet_cached(other, node)

    return edges


def get_conceptnet_edges_for_entities(
    entity1_text: str,
    entity1_type: str,
    entity2_text: str,
    entity2_type: str,
) -> List[dict]:
    """Get ConceptNet edges between two specific entities."""
    node = f"/c/en/{_normalize(entity1_text)}"
    other = f"/c/en/{_normalize(entity2_text)}"

    edges = _query_conceptnet_cached(node, other, limit=50)

    if not edges:
        edges = _query_conceptnet_cached(other, node, limit=50)

    return edges


def resolve_relation_with_conceptnet(
    original_text: str,
    head_text: str,
    head_type: str,
    tail_text: str,
    tail_type: str,
    threshold: float = 0.5,
) -> Optional[Tuple[str, float]]:
    """Resolve relation using ConceptNet + semantic similarity."""

    edges = get_conceptnet_edges_for_entities(
        head_text, head_type, tail_text, tail_type
    )

    if not edges:
        edges = get_conceptnet_edges(head_type, tail_type)

    if not edges:
        return None

    edges.sort(key=lambda e: e.get("weight", 0), reverse=True)
    top_edges = edges[:10]

    return _get_best_relation_via_similarity(
        original_text, top_edges, head_text, tail_text, threshold
    )


def preload_common_types(
    types: List[str],
    db_path: str = CONCEPTNET_DB_PATH,
    graph_name: str = CONCEPTNET_GRAPH_NAME,
):
    """
    Preload edges for common entity types into cache.
    Call this at startup to warm the cache.
    """
    logger.info(f"Preloading edges for {len(types)} entity types...")

    for i, type1 in enumerate(types):
        for type2 in types[i:]:
            get_conceptnet_edges(type1, type2)

    logger.info("Preload complete")


def search_concepts_by_embedding(
    query_text: str,
    embedding_func,
    top_k: int = 10,
    db_path: str = CONCEPTNET_DB_PATH,
    graph_name: str = CONCEPTNET_GRAPH_NAME,
) -> List[dict]:
    import numpy as np

    try:
        g = get_conceptnet_graph(db_path, graph_name)

        query_emb = embedding_func([query_text])[0]

        if hasattr(query_emb, "tolist"):
            query_emb = query_emb.tolist()

        query = """
        MATCH (c:Concept)
        WHERE c.embedding IS NOT NULL
        RETURN c.uri, c.name, c.embedding
        LIMIT 1000
        """

        result = g.query(query)

        similarities = []
        for row in result.result_set:
            uri, name, embedding = row
            if embedding:
                sim = np.dot(query_emb, embedding) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(embedding) + 1e-8
                )
                similarities.append(
                    {"uri": uri, "name": name, "similarity": float(sim)}
                )

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]

    except Exception as e:
        logger.warning(f"Embedding search failed: {e}")
        return []


def _disambiguate_with_conceptnet(
    text: str, entities: List[Entity], triplets: List[Triple], threshold: float = 0.5
) -> List[Triple]:
    """Disambiguate relations using ConceptNet + semantic similarity."""

    if not triplets:
        return triplets

    disambiguated = []
    entity_dicts = {e.text: e for e in entities}

    for triple in triplets:
        head_ent = entity_dicts.get(triple.subject)
        tail_ent = entity_dicts.get(triple.object)

        if head_ent and tail_ent:
            result = resolve_relation_with_conceptnet(
                original_text=text,
                head_text=triple.subject,
                head_type=head_ent.label,
                tail_text=triple.object,
                tail_type=tail_ent.label,
                threshold=threshold,
            )

            if result:
                relation, confidence = result
                disambiguated.append(Triple(triple.subject, relation, triple.object))
                logger.debug(
                    f"Disambiguated: {triple.subject} --[{relation}]--> {triple.object} (confidence: {confidence:.2f})"
                )
            else:
                disambiguated.append(triple)
        else:
            disambiguated.append(triple)

    return disambiguated


def has_named_entity(entities: List[Entity]) -> bool:
    """Check if any entity is a named entity (not 'unseen')."""
    return any(e.label != "unseen" for e in entities)


def extract_triplets_conceptnet(text: str, threshold: float = 0.5) -> List[Triple]:
    """Extract triplets using dependency-based pair finding + ConceptNet disambiguation."""
    from nimem.nlp.spacy import extract_entities_and_pairs

    entities, pairs = extract_entities_and_pairs(text)
    logger.debug(f"Entities: {entities}, pairs: {pairs}")

    if not pairs:
        return []

    seen = set()
    triplets = []

    for head_ent, tail_ent in pairs:
        key = (head_ent.text, tail_ent.text)
        if key in seen:
            continue
        seen.add(key)

        result = resolve_relation_with_conceptnet(
            original_text=text,
            head_text=head_ent.text,
            head_type=head_ent.label,
            tail_text=tail_ent.text,
            tail_type=tail_ent.label,
            threshold=threshold,
        )

        if result:
            relation, confidence = result
            triplets.append(Triple(head_ent.text, relation, tail_ent.text))
            logger.debug(
                f"ConceptNet: {head_ent.text} --[{relation}]--> {tail_ent.text} "
                f"(confidence: {confidence:.2f})"
            )

    return triplets


@lru_cache(maxsize=1000)
def entity_exists_in_conceptnet(entity_text: str) -> bool:
    """Check if an entity exists in ConceptNet by checking for any outgoing edge."""
    node = f"/c/en/{_normalize(entity_text)}"
    try:
        g = get_conceptnet_graph()
        query = """
        MATCH (s:Concept)-[r]->(e:Concept)
        WHERE s.uri = $node OR e.uri = $node
        RETURN count(r) LIMIT 1
        """
        result = g.query(query, {"node": node})
        for row in result.result_set:
            return row[0] > 0
        return False
    except Exception:
        return False
