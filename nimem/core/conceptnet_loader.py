"""
ConceptNet CSV loader for FalkorDB.

Loads ConceptNet assertions from the downloaded CSV into a local graph database.
CSV format: URI | relation | start_node | end_node | JSON_metadata
"""

import csv
import gzip
import json
import logging
from pathlib import Path
from typing import Optional

from redislite.falkordb_client import FalkorDB

from .config import CONCEPTNET_DB_PATH, CONCEPTNET_GRAPH_NAME

logger = logging.getLogger(__name__)


def get_conceptnet_graph(
    db_path: str = CONCEPTNET_DB_PATH,
    graph_name: str = CONCEPTNET_GRAPH_NAME,
):
    """Get ConceptNet graph client."""
    db = FalkorDB(db_path)
    return db.select_graph(graph_name)


def _parse_uri(uri: str) -> Optional[str]:
    """Extract concept name from ConceptNet URI."""
    # /c/en/dog -> dog
    # /c/en/apple_inc -> apple_inc
    parts = uri.split("/")
    if len(parts) >= 4 and parts[1] == "c":
        return parts[3]  # concept name
    return None


def _extract_language(uri: str) -> Optional[str]:
    """Extract language code from URI."""
    # /c/en/dog -> en
    parts = uri.split("/")
    if len(parts) >= 3 and parts[1] == "c":
        return parts[2]
    return None


def _extract_relation(uri: str) -> Optional[str]:
    """Extract relation from URI."""
    # /r/RelatedTo -> RelatedTo
    # /r/AtLocation -> AtLocation
    parts = uri.split("/")
    if len(parts) >= 3 and parts[1] == "r":
        return parts[2]
    return None


def load_conceptnet_csv(
    csv_path: str,
    db_path: str = CONCEPTNET_DB_PATH,
    graph_name: str = CONCEPTNET_GRAPH_NAME,
    max_edges: Optional[int] = None,
    language_filter: str = "en",
    batch_size: int = 1000,
):
    """
    Load ConceptNet CSV into FalkorDB.

    Args:
        csv_path: Path to conceptnet-assertions-5.7.0.csv.gz
        db_path: FalkorDB database path
        graph_name: Graph name in database
        max_edges: Maximum edges to load (for testing)
        language_filter: Only load concepts from this language (default: en)
        batch_size: Number of edges per transaction
    """
    g = get_conceptnet_graph(db_path, graph_name)

    # Create indices for fast lookups
    logger.info("Creating indices...")
    try:
        g.query("CREATE INDEX FOR (c:Concept) ON (c.uri)")
        g.query("CREATE INDEX FOR (c:Concept) ON (c.name)")
    except Exception as e:
        logger.debug(f"Index creation note: {e}")

    logger.info(f"Loading ConceptNet from {csv_path}...")

    open_func = gzip.open if csv_path.endswith(".gz") else open

    edges_loaded = 0
    skipped = 0
    batch_queries = []

    with open_func(csv_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if max_edges and edges_loaded >= max_edges:
                break

            if len(row) < 5:
                skipped += 1
                continue

            edge_uri, relation_uri, start_uri, end_uri, metadata_json = row

            # Filter by language
            start_lang = _extract_language(start_uri)
            end_lang = _extract_language(end_uri)

            if language_filter:
                if start_lang != language_filter or end_lang != language_filter:
                    skipped += 1
                    continue

            # Extract components
            start_name = _parse_uri(start_uri)
            end_name = _parse_uri(end_uri)
            relation = _extract_relation(relation_uri)

            if not (start_name and end_name and relation):
                skipped += 1
                continue

            # Parse metadata
            try:
                metadata = json.loads(metadata_json)
                weight = metadata.get("weight", 1.0)
                dataset = metadata.get("dataset", "")
            except (json.JSONDecodeError, KeyError):
                weight = 1.0
                dataset = ""

            # Build query for this edge
            query = f"""
            MERGE (s:Concept {{uri: $start_uri}})
            ON CREATE SET s.name = $start_name, s.language = $start_lang
            MERGE (e:Concept {{uri: $end_uri}})
            ON CREATE SET e.name = $end_name, e.language = $end_lang
            CREATE (s)-[r:{relation} {{
                weight: $weight,
                dataset: $dataset,
                edge_uri: $edge_uri
            }}]->(e)
            """

            params = {
                "start_uri": start_uri,
                "start_name": start_name,
                "start_lang": start_lang,
                "end_uri": end_uri,
                "end_name": end_name,
                "end_lang": end_lang,
                "weight": weight,
                "dataset": dataset,
                "edge_uri": edge_uri,
            }

            batch_queries.append((query, params))

            # Execute batch
            if len(batch_queries) >= batch_size:
                _execute_batch(g, batch_queries)
                edges_loaded += len(batch_queries)
                batch_queries = []

                if edges_loaded % 10000 == 0:
                    logger.info(f"Loaded {edges_loaded} edges, skipped {skipped}")

        # Execute remaining batch
        if batch_queries:
            _execute_batch(g, batch_queries)
            edges_loaded += len(batch_queries)

    logger.info(f"Completed: {edges_loaded} edges loaded, {skipped} skipped")
    return edges_loaded


def _execute_batch(graph, batch_queries):
    """Execute a batch of queries."""
    for query, params in batch_queries:
        try:
            graph.query(query, params)
        except Exception as e:
            logger.warning(f"Failed to execute query: {e}")


def add_embeddings_to_concepts(
    embedding_func,
    db_path: str = DEFAULT_CONCEPTNET_DB_PATH,
    graph_name: str = DEFAULT_CONCEPTNET_GRAPH,
    batch_size: int = 100,
):
    """
    Add vector embeddings to concept nodes.

    Args:
        embedding_func: Function that takes text and returns embedding vector
        db_path: Database path
        graph_name: Graph name
        batch_size: Concepts to embed per batch
    """
    g = get_conceptnet_graph(db_path, graph_name)

    # Get all concepts without embeddings
    query = """
    MATCH (c:Concept)
    WHERE c.embedding IS NULL
    RETURN c.uri, c.name
    LIMIT $batch_size
    """

    total_embedded = 0

    while True:
        result = g.query(query, {"batch_size": batch_size})

        if not result.result_set:
            break

        concepts_to_embed = [(row[0], row[1]) for row in result.result_set]

        # Generate embeddings
        texts = [name for _, name in concepts_to_embed]
        embeddings = embedding_func(texts)

        # Update concepts
        for (uri, name), embedding in zip(concepts_to_embed, embeddings):
            update_query = """
            MATCH (c:Concept {uri: $uri})
            SET c.embedding = $embedding
            """
            g.query(
                update_query,
                {
                    "uri": uri,
                    "embedding": embedding.tolist()
                    if hasattr(embedding, "tolist")
                    else embedding,
                },
            )

        total_embedded += len(concepts_to_embed)
        logger.info(f"Embedded {total_embedded} concepts")

        if len(concepts_to_embed) < batch_size:
            break

    logger.info(f"Total concepts embedded: {total_embedded}")


def get_stats(
    db_path: str = DEFAULT_CONCEPTNET_DB_PATH,
    graph_name: str = DEFAULT_CONCEPTNET_GRAPH,
):
    """Get database statistics."""
    g = get_conceptnet_graph(db_path, graph_name)

    # Count concepts
    concept_result = g.query("MATCH (c:Concept) RETURN count(c)")
    concept_count = concept_result.result_set[0][0] if concept_result.result_set else 0

    # Count edges by relation type
    relation_result = g.query("""
        MATCH ()-[r]->()
        RETURN type(r) as relation, count(r) as count
        ORDER BY count DESC
        LIMIT 20
    """)

    relations = [(row[0], row[1]) for row in relation_result.result_set]

    logger.info(f"Total concepts: {concept_count}")
    logger.info("Top relations:")
    for rel, count in relations:
        logger.info(f"  {rel}: {count}")

    return {"concepts": concept_count, "relations": relations}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python conceptnet_loader.py <csv_path> [max_edges]")
        sys.exit(1)

    csv_path = sys.argv[1]
    max_edges = int(sys.argv[2]) if len(sys.argv) > 2 else None

    load_conceptnet_csv(csv_path, max_edges=max_edges)
    get_stats()
