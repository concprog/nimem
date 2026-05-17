import logging
from typing import List

from nimem.domain.schema import Entity, Triple, ENTITIES, RELATIONS
from nimem.config import GLINER_MODEL

logger = logging.getLogger(__name__)

_model_instance = None


def get_model():
    global _model_instance
    if _model_instance is None:
        from gliner2 import GLiNER2

        logger.info(f"Loading GLiNER model: {GLINER_MODEL}")
        _model_instance = GLiNER2.from_pretrained(GLINER_MODEL)
    return _model_instance


def extract_entities(
    text: str,
    include_confidence: bool = False,
    include_spans: bool = False,
) -> List[Entity]:
    """Extract entities using GLiNER. Uses ENTITIES dict as labels."""
    model = get_model()
    labels = list(ENTITIES.keys())

    result = model.extract_entities(
        text,
        labels,
        include_confidence=include_confidence,
        include_spans=include_spans,
    )

    entities_dict = result.get("entities", {})
    entities = [
        Entity(
            text=item["text"],
            label=label,
            start=item.get("start", 0),
            end=item.get("end", 0),
            confidence=item.get("confidence", 1.0),
        )
        for label, items in entities_dict.items()
        for item in items
    ]
    logger.debug(f"GLiNER extracted entities: {entities}")
    return entities


def extract_relations(text: str) -> List[Triple]:
    """Extract relations using GLiNER (joint extraction)."""
    model = get_model()
    relation_labels = list(RELATIONS.keys())
    result = model.extract_relations(text, relation_labels)

    triplets = []
    extractions = result.get("relation_extraction", {})
    for relation, pairs in extractions.items():
        if relation not in RELATIONS:
            continue
        for pair in pairs:
            if isinstance(pair, tuple):
                triplets.append(Triple(pair[0], relation, pair[1]))
            elif isinstance(pair, dict):
                head = pair.get("head", {}).get("text", "")
                tail = pair.get("tail", {}).get("text", "")
                if head and tail:
                    triplets.append(Triple(head, relation, tail))

    return triplets
