import logging
from typing import List

from .schema import SPACY_LABEL_MAP, ENTITIES, Entity
from .model_loader import get_model

logger = logging.getLogger(__name__)


def extract_entities_spacy(text: str) -> List[Entity]:
    """Extract entities using spaCy NER. Uses SPACY_LABEL_MAP for filtering."""
    nlp = get_model("spacy")
    doc = nlp(text)

    entities = [
        Entity(
            text=ent.text,
            label=SPACY_LABEL_MAP.get(ent.label_, ent.label_.lower()),
            start=ent.start_char,
            end=ent.end_char,
        )
        for ent in doc.ents
        if ent.label_ in SPACY_LABEL_MAP
    ]
    logger.debug(f"spaCy extracted entities: {entities}")
    return entities


def extract_entities_gliner(
    text: str,
    include_confidence: bool = False,
    include_spans: bool = False,
) -> List[Entity]:
    """Extract entities using GLiNER. Uses ENTITIES dict as labels."""
    model = get_model("gliner")
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
