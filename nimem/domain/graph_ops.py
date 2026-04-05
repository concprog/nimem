import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
from fast_hdbscan import HDBSCAN
from returns.result import safe

from nimem.domain.schema import RELATIONS
from nimem.domain.conceptnet_vocab import (
    CONCEPTNET_RELATION_TEMPLATES,
    CONCEPTNET_TO_RELATION,
)
from nimem.domain.result_utils import unwrap_result
from nimem.embeddings.infinity import embed

logger = logging.getLogger(__name__)


@safe
def perform_clustering(
    vectors: np.ndarray, texts: List[str], min_cluster_size: int = 2
) -> Dict[int, List[str]]:
    """Clusters embedding vectors and maps them back to text labels."""
    if not texts:
        return {}

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(vectors)

    clusters: Dict[int, List[str]] = {}
    for text, label in zip(texts, labels):
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(text)
    return clusters


def generate_topic_name(texts: List[str]) -> str:
    """Simple heuristic to name a cluster."""
    return "Topic: " + ", ".join(list(set(texts))[:3])


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
        original_emb_result = embed([original_text])
        original_emb = unwrap_result(
            original_emb_result, "Failed to embed original text"
        )
        if original_emb is None:
            return _fallback_relation(candidate_edges)

        reconstructed_embs_result = embed(all_reconstructed)
        reconstructed_embs = unwrap_result(
            reconstructed_embs_result, "Failed to embed candidates"
        )
        if reconstructed_embs is None:
            return _fallback_relation(candidate_edges)

        original_vec = (
            original_emb[0]
            if isinstance(original_emb, np.ndarray)
            else np.array(original_emb[0])
        )

        similarities = []
        for i, recon in enumerate(all_reconstructed):
            recon_vec = (
                reconstructed_embs[i]
                if isinstance(reconstructed_embs, np.ndarray)
                else np.array(reconstructed_embs[i])
            )
            sim = np.dot(original_vec, recon_vec) / (
                np.linalg.norm(original_vec) * np.linalg.norm(recon_vec) + 1e-8
            )
            similarities.append(
                (all_reconstructed[i], sim, edge_to_relation[all_reconstructed[i]])
            )

        if not similarities:
            return _fallback_relation(candidate_edges)

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

    return _fallback_relation(candidate_edges)


def _fallback_relation(candidate_edges: List[dict]) -> Optional[Tuple[str, float]]:
    """Fallback: return best relation from candidate edges sorted by weight."""
    valid = []
    for edge in candidate_edges:
        rel_uri = edge.get("rel", {})
        rel_id = rel_uri.get("@id", "") if isinstance(rel_uri, dict) else ""
        relation = CONCEPTNET_TO_RELATION.get(rel_id)
        if relation and relation in RELATIONS:
            weight = edge.get("weight", 0.5)
            valid.append((relation, weight))

    if not valid:
        return None

    valid.sort(key=lambda x: x[1], reverse=True)
    return valid[0][0], float(valid[0][1])
