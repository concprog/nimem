# Nimem: A Non-LLM Episodic Memory System

Nimem is a lightweight, low-latency, and privacy-focused episodic memory system designed to provide agents with "infinite memory" without relying on expensive generative LLMs for ingestion. It leverages a **Neuro-Symbolic architecture**, combining the semantic flexibility of neural networks (GLiNER, FastCoref) with the structural rigidity and temporal reasoning of a bitemporal knowledge graph (FalkorDBLite).

## Quick Start

Nimem is designed to be used as a simple Python library. Here is the "Hello World" of memory.

```python
from nimem import memory

# 1. Ingest information from raw text
# The system automatically handles coreference ("She" -> "Alice") and extracts relations.
result = memory.ingest_text("Alice works at Google. She founded a new AI startup last week.")

if result.is_success:
    print(result.unwrap()) 
    # Output: Ingested 2 facts. (Resolved text: Alice works at Google...)

# 2. Recall memory about an entity
facts = memory.recall_memory("Alice")
# Returns a list of dicts: [{'relation': 'WORKS_FOR', 'object': 'Google'}, ...]

# 3. Consolidate memory (find hidden patterns)
# This runs clustering on embeddings to find shared topics.
memory.consolidate_topics()
```

---

## Core Concepts

### 1. The Neuro-Symbolic Ingestion Pipeline

Unlike RAG systems that chunk text blindly, Nimem "understands" the structure of data before storing it. The pipeline consists of two deterministic neural steps:

1.  **Coreference Resolution (`FastCoref`)**: Maps pronouns to their entities to ensure context is preserved.
2.  **Triplet Extraction (`GLiNER2`)**: Extracts `(Subject, Relation, Object)` tuples based on a predefined schema.

> [!NOTE]
> This happens entirely locally on the CPU, typically in under 200ms suitable for real-time agents.

**Code Look:** The pipeline acts as a functional chain in `nimem/core/text_processing.py`:

```python
def process_text_pipeline(text: str) -> Result[Tuple[str, List[Triple]], Exception]:
    """
    Chains coreference resolution -> Triplet Extraction.
    """
    return resolve_coreferences(text).bind(
        lambda resolved: extract_triplets(resolved).map(
            lambda triplets: (resolved, triplets)
        )
    )
```

### 2. Schema-Driven Memory

Nimem doesn't guess what's important; it looks for specific types of information defined in your `schema.py`. This ensures high precision and allows for "Cardinality Constraints".

**Cardinality Handling**: 
If a relation is defined as `"ONE"` (e.g., `located_in`), learning a new fact automatically invalidates the old one (updating the `valid_to` timestamp).

```python
# nimem/core/schema.py
CARDINALITY = {
    "works_for": "MANY",   # Can have multiple jobs
    "located_in": "ONE",   # Only one location at a time
    "knows": "MANY",
}
```

In `nimem/memory.py`, this logic is applied during ingestion:

```python
# Check Cardinality using Schema
cardinality = schema.CARDINALITY.get(tri.relation, "MANY")
if cardinality == "ONE":
     logging.info(f"Relation '{tri.relation}' is 1-to-1. Expiring old facts.")
     graph_store.expire_facts(tri.subject, tri.relation)
```

### 3. Bitemporal Graph Storage

Nimem stores facts in a **Bitemporal Graph**. Every edge has two time dimensions:

1.  **Valid Time (`valid_from`/`valid_to`)**: When the fact is true in the real world.
2.  **Transaction Time (`tx_from`/`tx_to`)**: When the system learned the fact.

This allows for "Time Travel"—you can see what the agent *knew* at a specific point in the past, or correct false memories without erasing the history of the mistake.

**Structure in `nimem/core/graph_store.py`**:

```cypher
CREATE (s)-[r:{relation.upper()} {
    valid_from: $valid_from,
    valid_to: 9999999999.0,  # "Infinity" (Open-ended)
    tx_from: $tx_time,
    tx_to: 9999999999.0,
    id: $edge_id
}]->(o)
```

### 4. Memory Consolidation (Weak Relations)

Facts are "Strong Relations" (explicitly stated). Nimem also finds "Weak Relations" (implicit connections) by clustering entities based on their semantic embeddings.

If "Alice" and "Bob" frequently appear in similar contexts, their vector embeddings will be close. `consolidate_topics()` uses HDBSCAN to group them into a Topic, creating new edges in the graph:
`(Alice)-[BELONGS_TO]->(Topic: Semantic_Cluster_1)`

```python
# nimem/core/clustering.py - The logic behind topic discovery
def perform_clustering(texts: List[str], min_cluster_size: int = 2):
    return embeddings.embed_texts(texts).map(cluster_vectors)
```

---

## Deployment & Configuration

Nimem is designed to be zero-config for development (`./nimem.db` is created automatically).

### Dependencies
- `gliner2`: For extraction.
- `fastcoref`: For coreference.
- `falkordb-lite`: For the embedded graph database.
- `infinity-emb`: For embeddings.
- `fast_hdbscan`: For clustering.

Ensure you have the required models downloaded (happens automatically on first run, cached to `~/.cache/huggingface`).
