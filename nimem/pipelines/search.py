import logging
from typing import List, Dict, Any, Tuple
from returns.result import Result, safe

from nimem.storage.graph_store import query_valid_facts
from nimem.embeddings.infinity import embed
from nimem.domain.graph_ops import perform_clustering

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
    vectors = embed(facts_list).unwrap()
    clusters = perform_clustering(
        vectors, facts_list, min_cluster_size=min_cluster_size
    ).unwrap()

    return clusters
