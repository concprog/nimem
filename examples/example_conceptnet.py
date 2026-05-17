"""
Example demonstrating ConceptNet download, ingestion into FalkorDB,
and using the spaCy + ConceptNet pipeline.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

from nimem.storage.conceptnet_loader import _load_conceptnet, get_stats
from nimem.storage.conceptnet_store import (
    extract_triplets_conceptnet,
    get_conceptnet_edges_for_entities,
    resolve_relation_with_conceptnet,
)
from nimem.nlp.spacy import extract_entities_and_pairs


def main():
    print("\n" + "=" * 70)
    print("STEP 1: Load/Import ConceptNet into FalkorDB")
    print("=" * 70)
    print("This will download ConceptNet CSV if not cached, then import into FalkorDB.")

    graph = _load_conceptnet()

    print("\nConceptNet Stats:")
    stats = get_stats()

    print("\n" + "=" * 70)
    print("STEP 2: Test ConceptNet queries")
    print("=" * 70)

    edges = get_conceptnet_edges_for_entities("cat", "animal", "dog", "animal")
    print(f"\nEdges between 'cat' and 'dog':")
    for edge in edges[:5]:
        print(
            f"  {edge['start']['@id']} --[{edge['rel']['@id']}]--> {edge['end']['@id']} (weight: {edge.get('weight', 'N/A')})"
        )

    result = resolve_relation_with_conceptnet(
        original_text="A cat is a kind of animal",
        head_text="cat",
        head_type="animal",
        tail_text="animal",
        tail_type="animal",
        threshold=0.5,
    )
    if result:
        relation, confidence = result
        print(
            f"\nResolved relation: cat --[{relation}]--> animal (confidence: {confidence:.2f})"
        )

    print("\n" + "=" * 70)
    print("STEP 3: spaCy + ConceptNet pipeline (entity pairs)")
    print("=" * 70)

    texts = [
        "The cat sat on the mat.",
        "John gave Mary a book.",
        "The chef cooked dinner with fresh ingredients.",
        "Alice works at Google in Mountain View.",
    ]

    for text in texts:
        print(f"\n--- Input: {text} ---")

        entities, pairs = extract_entities_and_pairs(text)
        print(f"Entities: {[(e.text, e.label) for e in entities]}")
        print(f"Pairs: {[(e1.text, e2.text) for e1, e2 in pairs]}")

        triplets = extract_triplets_conceptnet(text, threshold=0.3)
        print(f"Triplets (ConceptNet):")
        for t in triplets:
            print(f"  {t.subject} --[{t.relation}]--> {t.object}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
