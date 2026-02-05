# Nimem

**A Non-LLM Episodic Memory System**

Nimem is a lightweight, non-intelligent memory system for AI agents. It uses neural models for extraction and a knowledge graph for storage—no generative LLMs required for ingestion.

## Quick Start

```bash
pip install nimem
```

```python
from nimem import memory

# Ingest information
memory.ingest_text("Alice works at Google. She lives in London.")

# Recall facts
facts = memory.recall_memory("Alice")
# Returns: [{'relation': 'WORKS_FOR', 'object': 'Google'}, ...]

# Find hidden patterns
memory.consolidate_topics()
```

## Core Concepts

### Schema-Driven Extraction
Nimem extracts structured facts based on predefined relations, not open-ended generation.

```python
# nimem/core/schema.py
RELATIONS = {
    "works_for": "Employment or professional affiliation",
    "located_in": "Geographic containment",
    "knows": "Social relationship between people"
}
```

### Coreference Resolution
Pronouns are resolved to entities before extraction, preserving context.

```python
# Input: "Alice went home. She was tired."
# Resolved: "Alice went home. Alice was tired."
```

### Cardinality Constraints
Some facts are exclusive—learning a new location invalidates the old one.

```python
CARDINALITY = {
    "located_in": "ONE",   # Only one location at a time
    "works_for": "MANY"    # Can have multiple jobs
}
```

When you tell the system "Alice moved to Paris," it automatically invalidates "Alice lives in London."

### Bitemporal Graph
Every fact has two timestamps: when it was true, and when the system learned it.

```python
# Query memory at a specific time
facts = memory.recall_memory("Alice", at_time=timestamp)
```

This enables "time travel" and correcting false memories without losing history.

### Memory Consolidation
The system finds implicit connections by clustering entities with similar embeddings.

```python
# If "Alice" and "Bob" appear in similar contexts,
# the system infers: (Alice)-[BELONGS_TO]->(Topic_1)
#                    (Bob)-[BELONGS_TO]->(Topic_1)
```

## Getting Started

Run the example to see all features in action:

```bash
python example.py
```

### Example Walkthrough

```python
import logging
from nimem import memory
from returns.result import Success

# 1. INGESTION
# The system extracts structured facts from natural language
text = "Alice works for Google. She lives in London. Bob is Alice's friend."
result = memory.ingest_text(text)

if isinstance(result, Success):
    print(result.unwrap())
    # Output: Ingested 3 facts. (Resolved text: Alice works for Google...)

# 2. RECALL
# Query what the system knows about an entity
facts = memory.recall_memory("Alice")
for fact in facts.unwrap():
    print(f"Alice [{fact['relation']}] {fact['object']}")
# Output:
#   Alice [WORKS_FOR] Google
#   Alice [LOCATED_IN] London
#   Alice [KNOWS] Bob

# 3. UPDATING FACTS (Cardinality)
# "located_in" is a ONE-to-ONE relation
memory.ingest_text("Alice moved to Paris.")

# The old location is automatically invalidated
facts = memory.recall_memory("Alice")
locations = [f['object'] for f in facts.unwrap() if f['relation'] == 'LOCATED_IN']
print(locations)  # ['Paris'] - London is gone

# 4. CONSOLIDATION
# Find implicit topic clusters among entities
result = memory.consolidate_topics()
print(result.unwrap())
# Output: Consolidated 5 weak relations into 2 topics.
```

## Architecture

```
┌─────────────┐
│ Raw Text    │
└──────┬──────┘
       │
       ├─► FastCoref (Coreference Resolution)
       │   "She" → "Alice"
       │
       ├─► GLiNER (Relation Extraction)
       │   Extract: (Alice, WORKS_FOR, Google)
       │
       └─► FalkorDB (Graph Storage)
           Store with bitemporal metadata
```

### Components

| Layer | Library | Purpose |
|-------|---------|---------|
| **Coreference** | FastCoref | Resolve pronouns to entities |
| **Extraction** | GLiNER2 | Extract (Subject, Relation, Object) triples |
| **Storage** | FalkorDBLite | Bitemporal knowledge graph |
| **Embeddings** | Infinity-emb | Semantic vector generation |
| **Clustering** | FastHDBSCAN | Topic discovery |

## Installation

```bash
pip install nimem
```

### Dependencies
- `gliner2`: Relation extraction
- `fastcoref`: Coreference resolution
- `falkordb-lite`: Graph database
- `infinity-emb`: Embeddings
- `fast_hdbscan`: Clustering
- `returns`: Functional error handling

Models are automatically downloaded and cached on first run (~500MB).

## Configuration

Nimem is zero-config for development. The database is created automatically at `./nimem.db`.

### Custom Schema

Define your own relations:

```python
# nimem/core/schema.py
RELATIONS = {
    "owns": "Ownership relationship",
    "created": "Creation relationship"
}

CARDINALITY = {
    "owns": "MANY",
    "created": "ONE"
}
```

### Custom Database Path

```python
from nimem.core import graph_store

# Use a different database file
graph_store.get_graph_client(db_path='/path/to/custom.db')
```

## API Reference

### `memory.ingest_text(text: str) -> Result[str, Exception]`
Extract facts from text and store them in the graph.

### `memory.recall_memory(subject: str, at_time: float = None) -> Result[list, Exception]`
Retrieve all facts about an entity. Optionally query at a specific timestamp.

### `memory.add_memory(subject: str, relation: str, obj: str) -> Result[bool, Exception]`
Manually add a fact to the graph.

### `memory.consolidate_topics() -> Result[str, Exception]`
Run clustering to discover implicit topic relationships.

## Performance

All processing happens locally on CPU:
- **Coreference**: ~50-100ms per document
- **Extraction**: ~100-200ms per document
- **Storage**: ~10-50ms per fact
- **Total ingestion**: <300ms for typical conversational turns

## Comparison

| Feature | Nimem | RAG | Graphiti |
|---------|-------|-----|----------|
| **LLM for ingestion** | ❌ | ❌ | ✅ |
| **Temporal reasoning** | ✅ | ❌ | ✅ |
| **Fact updates** | ✅ | ❌ | ✅ |
| **Privacy** | ✅ Local | ⚠️ Varies | ⚠️ Varies |
| **Latency** | <300ms | <100ms | >1000ms |
| **Cost** | Free | Free | $$$ |

## Advanced Usage

### Time Travel Queries

```python
import time

# Record a fact
memory.ingest_text("Alice lives in London.")
t1 = time.time()

# Update the fact
time.sleep(1)
memory.ingest_text("Alice moved to Paris.")
t2 = time.time()

# Query at different points in time
past_facts = memory.recall_memory("Alice", at_time=t1)
current_facts = memory.recall_memory("Alice")
```

### Custom Processing Pipeline

```python
from nimem.core import text_processing
from returns.result import Success

# Override the default pipeline
def custom_pipeline(text):
    # Your custom logic here
    resolved = text_processing.resolve_coreferences(text)
    if isinstance(resolved, Success):
        return text_processing.extract_triplets(resolved.unwrap())
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use Nimem in research, please cite:

```bibtex
@software{nimem2026,
  title={Nimem: A Non-LLM Episodic Memory System},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/nimem}
}
```

## Acknowledgments

- **GLiNER**: [urchade/GLiNER](https://github.com/urchade/GLiNER)
- **FastCoref**: [shon-otmazgin/fastcoref](https://github.com/shon-otmazgin/fastcoref)
- **FalkorDB**: [FalkorDB/falkordb-lite](https://github.com/FalkorDB/falkordb-lite)
- **Graphiti**: Conceptual inspiration for bitemporal architecture
