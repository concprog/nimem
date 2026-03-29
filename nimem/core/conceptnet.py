"""
ConceptNet integration using local FalkorDB.

Refactored to query local ConceptNet database instead of API.
Maintains same function signatures for backward compatibility.
"""

import logging
import re
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

from .config import (
    CONCEPTNET_API_FALLBACK,
    CONCEPTNET_API_URL,
    CONCEPTNET_DB_PATH,
    CONCEPTNET_GRAPH_NAME,
)
from .schema import (
    CONCEPTNET_RELATION_TEMPLATES,
    CONCEPTNET_TO_RELATION,
    RELATIONS,
    Triple,
)
from .embeddings import embed_texts
from .cache_layer import setup_cache_directories

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

        # Query for edges between two nodes
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

            # Convert to API-compatible format
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
    # Try local database first
    edges = _query_local_db(node, other, limit)

    # Fallback to API if local query returns nothing
    if not edges:
        edges = _query_api_fallback(node, other, limit)

    return edges


def get_conceptnet_edges(type1: str, type2: str) -> List[dict]:
    """Get ConceptNet edges between two entity types."""
    node = f"/c/en/{_normalize(type1)}"
    other = f"/c/en/{_normalize(type2)}"

    edges = _query_conceptnet_cached(node, other)

    # Try reverse direction if no results
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


def _reconstruct_sentences(edge: dict, head_text: str, tail_text: str) -> List[str]:
    """Reconstruct sentences from ConceptNet edge using templates."""
    relation = edge.get("rel", {})
    rel_uri = relation.get("@id", "") if isinstance(relation, dict) else ""

    templates = CONCEPTNET_RELATION_TEMPLATES.get(rel_uri, [])

    if not templates:
        return []

    sentences = []
    for template in templates:
        try:
            sentences.append(template.format(head=head_text, tail=tail_text))
        except (KeyError, ValueError):
            continue

    return sentences


def _get_best_relation_via_similarity(
    original_text: str,
    candidate_edges: List[dict],
    head_text: str,
    tail_text: str,
    threshold: float = 0.5,
) -> Optional[Tuple[str, float]]:
    """Pick best relation using semantic similarity."""

    all_reconstructed = []
    edge_to_relation = {}

    for edge in candidate_edges:
        sentences = _reconstruct_sentences(edge, head_text, tail_text)
        for sentence in sentences:
            all_reconstructed.append(sentence)
            edge_to_relation[sentence] = edge

    if not all_reconstructed:
        return None

    try:
        original_emb = embed_texts([original_text])
        reconstructed_embs = embed_texts(all_reconstructed)

        if isinstance(original_emb, np.ndarray):
            original_vec = original_emb[0]
        else:
            original_vec = np.array(original_emb[0])

        similarities = []
        for i, recon in enumerate(all_reconstructed):
            if isinstance(reconstructed_embs, np.ndarray):
                recon_vec = reconstructed_embs[i]
            else:
                recon_vec = np.array(reconstructed_embs[i])

            sim = np.dot(original_vec, recon_vec) / (
                np.linalg.norm(original_vec) * np.linalg.norm(recon_vec) + 1e-8
            )
            similarities.append(
                (all_reconstructed[i], sim, edge_to_relation[all_reconstructed[i]])
            )

        if not similarities:
            return None

        similarities.sort(key=lambda x: x[1], reverse=True)
        best_sentence, best_sim, best_edge = similarities[0]

        if best_sim < threshold:
            return None

        rel_uri = best_edge.get("rel", {})
        rel_id = rel_uri.get("@id", "") if isinstance(rel_uri, dict) else ""

        relation = CONCEPTNET_TO_RELATION.get(rel_id)

        if relation and relation in RELATIONS:
            return relation, float(best_sim)

    except Exception as e:
        logger.warning(f"Similarity computation failed: {e}")

    return None


def resolve_relation_with_conceptnet(
    original_text: str,
    head_text: str,
    head_type: str,
    tail_text: str,
    tail_type: str,
    threshold: float = 0.5,
) -> Optional[Tuple[str, float]]:
    """Resolve relation using ConceptNet + semantic similarity."""

    # Try specific entities first
    edges = get_conceptnet_edges_for_entities(
        head_text, head_type, tail_text, tail_type
    )

    # Fallback to generic types
    if not edges:
        edges = get_conceptnet_edges(head_type, tail_type)

    if not edges:
        return None

    # Sort by weight and take top candidates
    edges.sort(key=lambda e: e.get("weight", 0), reverse=True)
    top_edges = edges[:10]

    return _get_best_relation_via_similarity(
        original_text, top_edges, head_text, tail_text, threshold
    )


# Additional functions for local DB optimization


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
    """
    Search for similar concepts using vector embeddings.
    Requires embeddings to be pre-computed via conceptnet_loader.add_embeddings_to_concepts()

    Returns:
        List of {uri, name, similarity_score}
    """
    try:
        g = get_conceptnet_graph(db_path, graph_name)

        # Get query embedding
        query_emb = embedding_func([query_text])[0]

        if hasattr(query_emb, "tolist"):
            query_emb = query_emb.tolist()

        # Note: FalkorDB vector similarity would be done via extension
        # For now, we fetch concepts and compute similarity in Python
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
