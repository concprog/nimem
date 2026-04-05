import logging
from typing import List, Dict, Optional
import spacy

from nimem.domain.schema import (
    Entity,
    Triple,
    ENTITY_RELATION_MAP,
    RELATIONS,
)
from nimem.domain.conceptnet_vocab import (
    SPACY_LABEL_MAP,
    VERB_TO_RELATION,
    WITH_PREPOSITIONS,
)

logger = logging.getLogger(__name__)

SPACY_MODEL = "en_core_web_md"
SUBJECT_DEPS = {"nsubj", "nsubjpass"}
OBJECT_DEPS = {"dobj", "attr", "oprd", "pobj", "dative", "iobj"}

_model_instance = None


def get_model():
    global _model_instance
    if _model_instance is None:
        logger.info(f"Loading spaCy model: {SPACY_MODEL}")
        try:
            _model_instance = spacy.load(SPACY_MODEL)
        except OSError:
            logger.warning(f"spaCy model {SPACY_MODEL} not found, downloading...")
            from spacy.cli import download

            download(SPACY_MODEL)
            _model_instance = spacy.load(SPACY_MODEL)
    return _model_instance


def extract_entities(text: str) -> List[Entity]:
    """Extract entities using spaCy NER. Uses SPACY_LABEL_MAP for filtering."""
    nlp = get_model()
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


def _build_entity_token_map(doc, entities: List[Entity]) -> dict:
    """Map token indices to their containing entity."""
    token_to_entity = {}
    for ent in entities:
        span = doc.char_span(ent.start, ent.end)
        if span is not None:
            for token in span:
                token_to_entity[token.i] = ent
    return token_to_entity


def _get_entity_for_token(token, token_to_entity: dict) -> Optional[Entity]:
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
    """Find all entities (named + unseen) and generate pairs from them."""
    token_to_entity = _build_entity_token_map(doc, entities)
    all_entities = list(entities)
    seen_starts = {e.start for e in entities}

    for token in doc:
        if token.i not in token_to_entity:
            if token.pos_ in {"NOUN", "PROPN", "VERB"}:
                all_entities.append(
                    Entity(
                        text=token.text,
                        label="unseen",
                        start=token.idx,
                        end=token.idx + len(token.text),
                    )
                )
                seen_starts.add(token.idx)

    sentences = list(doc.sents)
    pairs = []
    for sent in sentences:
        sent_entities = [
            e
            for e in all_entities
            if e.start >= sent.start_char and e.end <= sent.end_char
        ]
        for i, e1 in enumerate(sent_entities):
            for e2 in sent_entities[i + 1 :]:
                pairs.append((e1, e2))

    return all_entities, pairs


def extract_entities_and_pairs(text: str):
    """Extract entities and syntactically connected pairs in one spaCy pass."""
    nlp = get_model()
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

    all_entities, pairs = find_entity_pairs(doc, entities)
    logger.debug(f"Entities: {all_entities}, pairs: {pairs}")
    return all_entities, pairs


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


def extract_relations(text: str, entities: List[Entity]) -> List[Triple]:
    """Extract relations from pre-extracted entities using spaCy."""
    nlp = get_model()
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


def extract_triplets_spacy(
    text: str, use_conceptnet: bool = False, threshold: float = 0.5
) -> List[Triple]:
    """Full spaCy pipeline: entity extraction + relation extraction."""
    entities = extract_entities(text)
    logger.debug(f"Extracted entities: {entities}")
    triplets = extract_relations(text, entities)
    logger.debug(f"spaCy triplets: {triplets}")

    if use_conceptnet:
        from nimem.storage.conceptnet_store import _disambiguate_with_conceptnet

        triplets = _disambiguate_with_conceptnet(text, entities, triplets, threshold)

    return triplets
