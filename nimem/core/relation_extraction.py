import logging
import re
from typing import List, Set, Tuple, Optional, Dict, Any

from returns.result import safe

from .schema import (
    ENTITY_RELATION_MAP,
    RELATIONS,
    VERB_TO_RELATION,
    WITH_PREPOSITIONS,
    Triple,
    Entity,
)
from .model_loader import get_model

logger = logging.getLogger(__name__)

SUBJECT_DEPS = {"nsubj", "nsubjpass"}
OBJECT_DEPS = {"dobj", "attr", "oprd", "pobj"}
ALL_OBJECT_DEPS = OBJECT_DEPS | {"pobj"}


def _infer_relation(entity1_label: str, entity2_label: str) -> str | None:
    key = (entity1_label.lower(), entity2_label.lower())
    return ENTITY_RELATION_MAP.get(key)


def _build_entity_lookup(doc, entities: List[Entity]) -> Dict[int, Entity]:
    """Build token index -> entity mapping using spaCy's char_span."""
    token_to_entity = {}
    for ent in entities:
        span = doc.char_span(ent.start, ent.end)
        if span is not None:
            for token in span:
                token_to_entity[token.i] = ent
    return token_to_entity


def _get_entity_for_token(
    token, token_to_entity: Dict[int, Entity]
) -> Optional[Entity]:
    """Find entity containing this token."""
    if token.i in token_to_entity:
        return token_to_entity[token.i]
    for child in token.children:
        if child.i in token_to_entity:
            return token_to_entity[child.i]
    for ancestor in token.ancestors:
        if ancestor.i in token_to_entity:
            return token_to_entity[ancestor.i]
    return None


def _extract_verb_relations(
    doc, entities: List[Entity], token_to_entity: Dict[int, Entity]
) -> List[Triple]:
    """Extract relations based on verb parsing."""
    triplets = []

    for token in doc:
        if token.pos_ != "VERB":
            continue

        verb_lemma = token.lemma_.lower()
        relation = VERB_TO_RELATION.get(verb_lemma)

        subjects = [c for c in token.children if c.dep_ in SUBJECT_DEPS]
        direct_objects = [c for c in token.children if c.dep_ in OBJECT_DEPS]

        prep_objects = []
        with_objects = []

        for child in token.children:
            if child.dep_ == "prep":
                prep_text = child.text.lower()
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        if prep_text in WITH_PREPOSITIONS:
                            with_objects.append(pobj)
                        else:
                            prep_objects.append(pobj)

        all_objects = direct_objects + prep_objects

        for subj in subjects:
            subj_ent = _get_entity_for_token(subj, token_to_entity)
            if not subj_ent:
                continue

            for obj in all_objects:
                obj_ent = _get_entity_for_token(obj, token_to_entity)
                if obj_ent:
                    if relation:
                        triplets.append(Triple(subj_ent.text, relation, obj_ent.text))
                else:
                    descriptive_name = f"{subj_ent.text}'s {obj.text}"
                    if relation:
                        triplets.append(
                            Triple(subj_ent.text, relation, descriptive_name)
                        )

            for with_obj in with_objects:
                with_ent = _get_entity_for_token(with_obj, token_to_entity)
                if with_ent:
                    if relation:
                        triplets.append(
                            Triple(subj_ent.text, "worked_with", with_ent.text)
                        )
                    for obj in all_objects:
                        obj_ent = _get_entity_for_token(obj, token_to_entity)
                        if obj_ent and relation:
                            triplets.append(
                                Triple(with_ent.text, relation, obj_ent.text)
                            )

    return triplets


def _extract_prep_relations(
    doc, entities: List[Entity], token_to_entity: Dict[int, Entity]
) -> List[Triple]:
    """Extract relations from prepositional phrases."""
    triplets = []

    prep_relation_map = {
        "in": "located_in",
        "at": "located_in",
        "on": "located_in",
        "with": "worked_with",
        "for": "works_for",
        "to": "related_to",
        "from": "related_to",
    }

    for ent in entities:
        span = doc.char_span(ent.start, ent.end)
        if span is None:
            continue
        for token in span:
            for child in token.children:
                if child.dep_ == "prep":
                    prep_text = child.text.lower()
                    relation = prep_relation_map.get(prep_text)

                    if relation and relation in RELATIONS:
                        for pobj in child.children:
                            if pobj.dep_ == "pobj":
                                obj_ent = _get_entity_for_token(pobj, token_to_entity)
                                if obj_ent and obj_ent != ent:
                                    triplets.append(
                                        Triple(ent.text, relation, obj_ent.text)
                                    )

    return triplets


def _extract_copula_relations(
    doc, entities: List[Entity], token_to_entity: Dict[int, Entity]
) -> List[Triple]:
    """Extract relations from copula constructions (is, was, became, etc.)."""
    triplets = []

    for token in doc:
        if token.lemma_ not in {"be", "become", "remain", "stay"}:
            continue

        subjects = [c for c in token.children if c.dep_ in SUBJECT_DEPS]
        attributes = [c for c in token.children if c.dep_ == "attr"]

        for subj in subjects:
            subj_ent = _get_entity_for_token(subj, token_to_entity)
            if not subj_ent:
                continue

            for attr in attributes:
                attr_ent = _get_entity_for_token(attr, token_to_entity)
                if attr_ent:
                    relation = _infer_relation(subj_ent.label, attr_ent.label)
                    if relation and relation in RELATIONS:
                        triplets.append(Triple(subj_ent.text, relation, attr_ent.text))

    return triplets


def _extract_entity_pair_relations(
    doc, entities: List[Entity], token_to_entity: Dict[int, Entity]
) -> List[Triple]:
    """Extract relations from entity pairs in same sentence using dependency paths."""
    triplets = []

    sentences = list(doc.sents)

    for sent in sentences:
        sent_entities = [
            ent
            for ent in entities
            if ent.start >= sent.start_char and ent.end <= sent.end_char
        ]

        for i, e1 in enumerate(sent_entities):
            for e2 in sent_entities[i + 1 :]:
                relation = _infer_relation(e1.label, e2.label)
                if relation and relation in RELATIONS:
                    triplets.append(Triple(e1.text, relation, e2.text))

                relation_rev = _infer_relation(e2.label, e1.label)
                if (
                    relation_rev
                    and relation_rev in RELATIONS
                    and relation_rev != relation
                ):
                    triplets.append(Triple(e2.text, relation_rev, e1.text))

    return triplets


def extract_relations_spacy(text: str, entities: List[Entity]) -> List[Triple]:
    """Extract relations from pre-extracted entities using spaCy."""
    nlp = get_model("spacy")
    doc = nlp(text)

    token_to_entity = _build_entity_lookup(doc, entities)

    verb_triplets = _extract_verb_relations(doc, entities, token_to_entity)
    prep_triplets = _extract_prep_relations(doc, entities, token_to_entity)
    copula_triplets = _extract_copula_relations(doc, entities, token_to_entity)
    entity_pair_triplets = _extract_entity_pair_relations(
        doc, entities, token_to_entity
    )

    all_triplets = (
        verb_triplets + prep_triplets + copula_triplets + entity_pair_triplets
    )

    seen = set()
    combined = [
        t
        for t in all_triplets
        if not (
            (key := (t.subject.lower(), t.relation.lower(), t.object.lower())) in seen
            or seen.add(key)
        )
    ]

    logger.debug(f"spaCy relation triplets: {combined}")
    return combined


def _extract_gliner_relations(text: str) -> List[Triple]:
    """Extract relations using GLiNER (joint extraction)."""
    model = get_model("gliner")
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


def extract_triplets_spacy(
    text: str, use_conceptnet: bool = False, threshold: float = 0.5
) -> List[Triple]:
    """Full spaCy pipeline: entity extraction + relation extraction.

    Args:
        text: Input text
        use_conceptnet: If True, use ConceptNet for relation disambiguation
        threshold: Confidence threshold for ConceptNet disambiguation
    """
    from .entity_recognition import extract_entities_spacy

    entities = extract_entities_spacy(text)
    logger.debug(f"Extracted entities: {entities}")
    triplets = extract_relations_spacy(text, entities)
    logger.debug(f"spaCy triplets: {triplets}")

    if use_conceptnet:
        triplets = _disambiguate_with_conceptnet(text, entities, triplets, threshold)

    return triplets


def _disambiguate_with_conceptnet(
    text: str, entities: List[Entity], triplets: List[Triple], threshold: float = 0.5
) -> List[Triple]:
    """Disambiguate relations using ConceptNet + semantic similarity."""
    from . import conceptnet

    if not triplets:
        return triplets

    disambiguated = []
    entity_dicts = {e.text: e for e in entities}

    for triple in triplets:
        head_ent = entity_dicts.get(triple.subject)
        tail_ent = entity_dicts.get(triple.object)

        if head_ent and tail_ent:
            result = conceptnet.resolve_relation_with_conceptnet(
                original_text=text,
                head_text=triple.subject,
                head_type=head_ent.label,
                tail_text=triple.object,
                tail_type=tail_ent.label,
                threshold=threshold,
            )

            if result:
                relation, confidence = result
                disambiguated.append(Triple(triple.subject, relation, triple.object))
                logger.debug(
                    f"Disambiguated: {triple.subject} --[{relation}]--> {triple.object} (confidence: {confidence:.2f})"
                )
            else:
                disambiguated.append(triple)
        else:
            disambiguated.append(triple)

    return disambiguated


def extract_triplets_gliner(text: str) -> List[Triple]:
    """GLiNER joint extraction (entities + relations)."""
    triplets = _extract_gliner_relations(text)
    logger.debug(f"GLiNER triplets: {triplets}")
    return triplets


def extract_triplets_conceptnet(text: str, threshold: float = 0.5) -> List[Triple]:
    """Extract triplets using ConceptNet + semantic similarity (no spaCy)."""
    from .entity_recognition import extract_entities_spacy
    from . import conceptnet

    entities = extract_entities_spacy(text)
    logger.debug(f"Entities for ConceptNet: {entities}")

    entity_dicts = [
        {"text": e.text, "label": e.label, "start": e.start, "end": e.end}
        for e in entities
    ]

    triplets = conceptnet.extract_triplets_with_conceptnet(
        text, entity_dicts, threshold=threshold
    )
    logger.debug(f"ConceptNet triplets: {triplets}")
    return triplets


@safe
def extract_triplets(
    text: str, use_gliner2: bool = False, use_conceptnet: bool = False
) -> List[Triple]:
    """Extract triplets using specified method.

    Args:
        text: Input text
        use_gliner2: Use GLiNER joint extraction
        use_conceptnet: Use spaCy + ConceptNet disambiguation (hybrid)

    Returns:
        List of Triple (subject, relation, object)
    """
    if use_gliner2:
        return extract_triplets_gliner(text)
    if use_conceptnet:
        return extract_triplets_spacy(text, use_conceptnet=True)
    return extract_triplets_spacy(text)
