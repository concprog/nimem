from typing import List, Dict, Tuple
from returns.result import Result, safe
import numpy as np
import logging

try:
    from fast_hdbscan import HDBSCAN
except ImportError:
    HDBSCAN = None

from . import embeddings

@safe
def perform_clustering(texts: List[str], min_cluster_size: int = 2) -> Result[Dict[int, List[str]], Exception]:
    """
    Clusters a list of texts using Infinity Embeddings + FastHDBSCAN.
    """
    if not texts:
        return {}
        
    if HDBSCAN is None:
        raise ImportError("fast_hdbscan is not installed.")

    def cluster_vectors(vectors: np.ndarray) -> Dict[int, List[str]]:
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

    return embeddings.embed_texts(texts).map(cluster_vectors)

def generate_topic_name(texts: List[str]) -> str:
    """
    Simple heuristic or LLM call to name the cluster.
    For non-LLM, we can use the most central term or just 'Topic {hash}'.
    Let's use a concatenation of top 3 terms or just 'Topic: A, B, C'
    """
    # Simply join the first 3 unique items
    return "Topic: " + ", ".join(list(set(texts))[:3])
