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


def _build_entity_token_map(doc, entities: List[Entity]) -> dict:
    """Map token indices to their containing entity."""
    token_to_entity = {}
    for ent in entities:
        span = doc.char_span(ent.start, ent.end)
        if span is not None:
            for token in span:
                token_to_entity[token.i] = ent
    return token_to_entity


def _get_entity_for_token(token, token_to_entity: dict):
    """Find entity for a token, checking children/ancestors for compounds."""
    if token.i in token_to_entity:
        return token_to_entity[token.i]
    for child in token.children:
        if child.i in token_to_entity:
            return token_to_entity[child.i]
    for ancestor in token.ancestors:
        if ancestor.i in token_to_entity:
            return token_to_entity[ancestor.i]
    return None


def find_entity_pairs(doc, entities: List[Entity]):
    """Find syntactically connected entity pairs via dependency parsing.

    Connects entities through:
    - Verb subjects to verb objects (nsubj/nsubjpass -> dobj/attr/pobj)
    - Noun prepositional objects (entity -> prep -> entity)
    """
    token_to_entity = _build_entity_token_map(doc, entities)
    pairs = set()

    for token in doc:
        if token.pos_ == "VERB":
            subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
            objects = [c for c in token.children if c.dep_ in ("dobj", "attr")]
            for prep in [c for c in token.children if c.dep_ == "prep"]:
                objects.extend(c for c in prep.children if c.dep_ == "pobj")
            for subj in subjects:
                for obj in objects:
                    e1 = _get_entity_for_token(subj, token_to_entity)
                    e2 = _get_entity_for_token(obj, token_to_entity)
                    if e1 and e2 and e1 != e2:
                        pairs.add((e1, e2))

        elif token.pos_ == "NOUN":
            for prep in [c for c in token.children if c.dep_ == "prep"]:
                for pobj in [c for c in prep.children if c.dep_ == "pobj"]:
                    e1 = _get_entity_for_token(token, token_to_entity)
                    e2 = _get_entity_for_token(pobj, token_to_entity)
                    if e1 and e2 and e1 != e2:
                        pairs.add((e1, e2))

    return list(pairs)


def extract_entities_and_pairs(text: str):
    """Extract entities and syntactically connected pairs in one spaCy pass.

    Returns:
        Tuple of (entities, pairs) where pairs are (Entity, Entity) tuples
        connected via dependency structure.
    """
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

    pairs = find_entity_pairs(doc, entities)
    logger.debug(f"Entities: {entities}, pairs: {pairs}")
    return entities, pairs
