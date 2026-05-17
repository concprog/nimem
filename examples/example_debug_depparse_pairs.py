"""
Debug example for dependency parsing and entity pair extraction.
"""

import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

from nimem.nlp.spacy import extract_entities_and_pairs, find_entity_pairs


def main():
    texts = [
        "The cat sat on the mat.",
        "John gave Mary a book.",
        "The chef cooked dinner with fresh ingredients.",
        "Alice works at Google in Mountain View.",
        "Bob is a software engineer.",
        "The dog chased the cat across the yard.",
    ]

    for text in texts:
        print(f"\n{'=' * 60}")
        print(f"Text: {text}")
        print("=" * 60)

        nlp = None
        from nimem.nlp.spacy import get_model

        nlp = get_model()
        doc = nlp(text)

        print("\nTokens:")
        for token in doc:
            print(
                f"  {token.text:15} | pos: {token.pos_:6} | dep: {token.dep_:10} | head: {token.head.text}"
            )

        print("\nEntities (NER):")
        entities, pairs = extract_entities_and_pairs(text)
        for ent in entities:
            print(f"  {ent.text:20} | label: {ent.label}")

        print("\nEntity Pairs (via dependency parsing):")
        if pairs:
            for e1, e2 in pairs:
                print(f"  ({e1.text}, {e1.label})  --  ({e2.text}, {e2.label})")
        else:
            print("  (no pairs found)")


if __name__ == "__main__":
    main()
