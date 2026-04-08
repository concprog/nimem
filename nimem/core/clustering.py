import logging
from typing import List, Dict

import numpy as np
from fast_hdbscan import HDBSCAN
from returns.result import safe

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
