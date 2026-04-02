import logging
from typing import List, Dict, Any, Tuple
from returns.result import Result, safe

from nimem.storage.graph_store import query_valid_facts
from nimem.embeddings.infinity import embed
from nimem.domain.graph_ops import perform_clustering
from nimem.domain.result_utils import unwrap_result, result_to_tuple

logger = logging.getLogger(__name__)


def search_and_cluster(
    subject: str,
    at_time: float | None = None,
    min_cluster_size: int = 2,
) -> Result[Dict[int, List[str]], Exception]:
    """Query facts, embed them, and cluster them."""
    return query_valid_facts(subject, at_time=at_time).bind(
        lambda facts: _cluster_facts(facts, min_cluster_size)
    )


@safe
def _cluster_facts(facts: List[Dict[str, Any]], min_cluster_size: int):
    if not facts:
        return {}

    facts_list = [f"{f['relation']} {f['object']}" for f in facts]
    vectors = unwrap_result(embed(facts_list), "Failed to embed facts")
    if vectors is None:
        return {}

    clusters_result = perform_clustering(
        vectors, facts_list, min_cluster_size=min_cluster_size
    )
    clusters = unwrap_result(clusters_result, "Failed to cluster")
    if clusters is None:
        return {}

    return clusters
