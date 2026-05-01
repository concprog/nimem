import logging
import re
from typing import List, Set

from returns.result import safe

from .schema import (
    ENTITY_RELATION_MAP,
    RELATIONS,
    VERB_TO_RELATION,
    VERB_RULES,
    WITH_PREPOSITIONS,
    Triple,
    Entity,
)
from .model_loader import get_model

logger = logging.getLogger(__name__)


def _infer_relation(entity1_label: str, entity2_label: str) -> str | None:
    key = (entity1_label.lower(), entity2_label.lower())
    return ENTITY_RELATION_MAP.get(key)


def _extract_relations_from_entities(text: str, entities: List[Entity]) -> List[Triple]:
    """Extract relations from entity pairs in same sentence."""
    triplets = []
    sentences = re.split(r"[.!?]+", text)
    sentence_starts = [0]
    pos = 0
    for sent in sentences[:-1]:
        pos += len(sent) + 1
        sentence_starts.append(pos)

    def get_sentence_idx(entity_start: int) -> int:
        for i, start in enumerate(sentence_starts):
            if i + 1 < len(sentence_starts):
                if sentence_starts[i] <= entity_start < sentence_starts[i + 1]:
                    return i
            else:
                return i
        return len(sentence_starts) - 1

    sentence_entities: dict[int, List[Entity]] = {}
    for entity in entities:
        sent_idx = get_sentence_idx(entity.start)
        sentence_entities.setdefault(sent_idx, []).append(entity)

    for sent_entities in sentence_entities.values():
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


def _get_noun_phrase(token) -> str:
    parts = []
    for child in token.lefts:
        if child.dep_ in ("compound", "amod", "poss") and child.dep_ != "det":
            parts.append(child.text)
    parts.append(token.text)
    return " ".join(parts)

def _get_entity_label(token, doc):
    for ent in doc.ents:
        if ent.start <= token.i < ent.end:
            return ent.label_
    return None

def _extract_verb_relations(text: str, known_entities: Set[str]) -> List[Triple]:
    """Extract relations based on verb parsing."""
    nlp = get_model("spacy")
    doc = nlp(text)
    triplets = []

    for token in doc:
        if token.pos_ != "VERB":
            continue

        verb_lemma = token.lemma_.lower()
        rule = VERB_RULES.get(verb_lemma)
        if not rule:
            continue

        relation = rule["relation"]
        subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
        direct_objects = [
            c for c in token.children if c.dep_ in ("dobj", "attr", "oprd")
        ]

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
            subj_label = _get_entity_label(subj, doc)

            if subj_label not in rule.get("subject_types", set()):
                continue
            subj_text = subj.text
            if subj_text not in known_entities:
                subj_ent = None
                for ent in doc.ents:
                    if subj.i >= ent.start and subj.i < ent.end:
                        subj_ent = ent
                        break
                if not subj_ent:
                    continue

            for obj in all_objects:
                obj_label = _get_entity_label(obj, doc)
                if obj_label not in rule.get("object_types", set()):
                    continue
                obj_text = obj.text
                if obj_text in known_entities:
                    triplets.append(Triple(subj_text, relation, obj_text))
                else:
                    obj_ent = None
                    for ent in doc.ents:
                        if obj.i >= ent.start and obj.i < ent.end:
                            obj_ent = ent
                            obj_text = ent.text
                            break

                    if obj_ent:
                        if rule.get("reverse"):
                            triplets.append(Triple(obj_text, relation, subj_text))
                        else:
                            triplets.append(Triple(subj_text, relation, obj_text))
                        triplets.append(Triple(subj_text, relation, obj_text))
                    else:
                        descriptive_name = f"{subj_text}'s {obj.text}"
                        if rule.get("reverse"):
                            triplets.append(Triple(obj_text, relation, subj_text))
                        else:
                            triplets.append(Triple(subj_text, relation, obj_text))
                        triplets.append(Triple(subj_text, relation, descriptive_name))

            for with_obj in with_objects:
                with_text = with_obj.text
                if with_text in known_entities or any(
                    with_obj.i >= ent.start and with_obj.i < ent.end for ent in doc.ents
                ):
                    triplets.append(Triple(subj_text, "worked_with", with_text))
                    for obj in all_objects:
                        if obj.text in known_entities:
                            triplets.append(Triple(with_text, relation, obj.text))

    return triplets


def extract_relations_spacy(text: str, entities: List[Entity]) -> List[Triple]:
    """Extract relations from pre-extracted entities using spaCy."""
    known_entities = {e.text for e in entities}
    triplets_heuristic = _extract_relations_from_entities(text, entities)
    triplets_verb = _extract_verb_relations(text, known_entities)

    seen = set()
    combined = [
        t
        for t in triplets_heuristic + triplets_verb
        if not (
            (key := (t.subject.lower(), t.relation.lower(), t.object.lower())) in seen
            or seen.add(key)
        )
    ]
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


def extract_triplets_spacy(text: str) -> List[Triple]:
    """Full spaCy pipeline: entity extraction + relation extraction."""
    from .entity_recognition import extract_entities_spacy

    entities = extract_entities_spacy(text)
    logger.debug(f"Extracted entities: {entities}")
    triplets = extract_relations_spacy(text, entities)
    logger.debug(f"spaCy triplets: {triplets}")
    return triplets


def extract_triplets_gliner(text: str) -> List[Triple]:
    """GLiNER joint extraction (entities + relations)."""
    triplets = _extract_gliner_relations(text)
    logger.debug(f"GLiNER triplets: {triplets}")
    return triplets


# Backward compatibility
@safe
def extract_triplets(text: str, use_gliner2: bool = False) -> List[Triple]:
    """Legacy function - dispatches to appropriate implementation."""
    if use_gliner2:
        return extract_triplets_gliner(text)
    return extract_triplets_spacy(text)
