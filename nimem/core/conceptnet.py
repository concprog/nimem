import logging
import re
from functools import lru_cache
from typing import List, Optional, Tuple

import requests
import numpy as np
from returns.result import safe

from .schema import (
    CONCEPTNET_RELATION_TEMPLATES,
    CONCEPTNET_TO_RELATION,
    RELATIONS,
    Triple,
)
from .embeddings import embed_texts

logger = logging.getLogger(__name__)

CONCEPTNET_API = "http://api.conceptnet.io"


@lru_cache(maxsize=1000)
def _query_conceptnet_cached(node: str, other: str, limit: int = 30) -> List[dict]:
    """Cached ConceptNet API query."""
    try:
        response = requests.get(
            f"{CONCEPTNET_API}/query",
            params={"node": node, "other": other, "limit": limit},
            timeout=5,
        )
        if response.status_code == 200:
            return response.json().get("edges", [])
    except requests.RequestException as e:
        logger.warning(f"ConceptNet query failed: {e}")
    return []


def _normalize(text: str) -> str:
    """Normalize text for ConceptNet lookup."""
    return text.lower().replace(" ", "_").replace(".", "")


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


@safe
def extract_triplets_with_conceptnet(
    text: str,
    entities: List[dict],
    threshold: float = 0.5,
) -> List[Triple]:
    """Extract triplets using ConceptNet for relation disambiguation."""

    triplets = []

    for i, ent1 in enumerate(entities):
        for ent2 in entities[i + 1 :]:
            result = resolve_relation_with_conceptnet(
                original_text=text,
                head_text=ent1["text"],
                head_type=ent1["label"],
                tail_text=ent2["text"],
                tail_type=ent2["label"],
                threshold=threshold,
            )

            if result:
                relation, confidence = result
                triplets.append(Triple(ent1["text"], relation, ent2["text"]))

            result_rev = resolve_relation_with_conceptnet(
                original_text=text,
                head_text=ent2["text"],
                head_type=ent2["label"],
                tail_text=ent1["text"],
                tail_type=ent1["label"],
                threshold=threshold,
            )

            if result_rev:
                relation, confidence = result_rev
                if relation not in {
                    t.relation
                    for t in triplets
                    if t.subject == ent2["text"] and t.object == ent1["text"]
                }:
                    triplets.append(Triple(ent2["text"], relation, ent1["text"]))

    return triplets
