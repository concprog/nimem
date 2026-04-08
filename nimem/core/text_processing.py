import logging
import re
from typing import List, Tuple, NamedTuple, Set
from functools import lru_cache

import spacy
from gliner2 import GLiNER2
from returns.result import Result, safe

from . import schema

logger = logging.getLogger(__name__)

SPACY_MODEL = "en_core_web_sm"
GLINER2_MODEL = "fastino/gliner2-multi-v1"


class Triple(NamedTuple):
    subject: str
    relation: str
    object: str


ENTITY_RELATION_MAP = {
    ("person", "organization"): "works_for",
    ("person", "location"): "located_in",
    ("person", "person"): "knows",
    ("organization", "location"): "located_in",
    ("event", "location"): "happened_at",
}


@lru_cache(maxsize=1)
def get_spacy_model():
    logger.info(f"Loading spaCy model: {SPACY_MODEL}")
    try:
        return spacy.load(SPACY_MODEL)
    except OSError:
        logger.warning(f"spaCy model {SPACY_MODEL} not found, downloading...")
        from spacy.cli import download

        download(SPACY_MODEL)
        return spacy.load(SPACY_MODEL)


@lru_cache(maxsize=1)
def get_gliner_model():
    logger.info(f"Loading GLiNER2 model: {GLINER2_MODEL}")
    return GLiNER2.from_pretrained(GLINER2_MODEL)


@lru_cache(maxsize=1)
def get_fastcoref_model():
    from fastcoref import FCoref

    logger.info("Loading FastCoref model")
    return FCoref(device="cpu")


def _infer_relation(entity1_label: str, entity2_label: str) -> str | None:
    key = (entity1_label.lower(), entity2_label.lower())
    return ENTITY_RELATION_MAP.get(key)


def _find_entity_positions(text: str, entities: List[dict]) -> List[dict]:
    """Find character positions for entities in text."""
    result = []
    for entity in entities:
        name = entity["text"]
        label = entity["label"]
        start = 0
        while True:
            pos = text.find(name, start)
            if pos == -1:
                break
            result.append(
                {
                    "text": name,
                    "label": label,
                    "start": pos,
                    "end": pos + len(name),
                }
            )
            start = pos + len(name)
    return result


def _extract_relations_from_entities(text: str, entities: List[dict]) -> List[Triple]:
    entities_with_pos = _find_entity_positions(text, entities)

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

    sentence_entities: dict[int, List[dict]] = {}
    for entity in entities_with_pos:
        sent_idx = get_sentence_idx(entity["start"])
        if sent_idx not in sentence_entities:
            sentence_entities[sent_idx] = []
        sentence_entities[sent_idx].append(entity)

    for sent_idx, sent_entities in sentence_entities.items():
        for i, e1 in enumerate(sent_entities):
            for e2 in sent_entities[i + 1 :]:
                relation = _infer_relation(e1["label"], e2["label"])
                if relation and relation in schema.RELATIONS:
                    triplets.append(Triple(e1["text"], relation, e2["text"]))

                relation_rev = _infer_relation(e2["label"], e1["label"])
                if (
                    relation_rev
                    and relation_rev in schema.RELATIONS
                    and relation_rev != relation
                ):
                    triplets.append(Triple(e2["text"], relation_rev, e1["text"]))

    return triplets


def _extract_verb_relations(text: str, known_entities: Set[str]) -> List[Triple]:
    nlp = get_spacy_model()
    doc = nlp(text)
    triplets = []

    for token in doc:
        if token.pos_ != "VERB":
            continue

        verb_lemma = token.lemma_.lower()
        relation = schema.VERB_TO_RELATION.get(verb_lemma)
        if not relation:
            continue

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
                        if prep_text in schema.WITH_PREPOSITIONS:
                            with_objects.append(pobj)
                        else:
                            prep_objects.append(pobj)

        all_objects = direct_objects + prep_objects

        for subj in subjects:
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
                        triplets.append(Triple(subj_text, relation, obj_text))
                    else:
                        descriptive_name = f"{subj_text}'s {obj.text}"
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


@safe
def extract_triplets(text: str) -> List[Triple]:
    model = get_gliner_model()
    labels = list(schema.ENTITIES.keys())
    result = model.extract_entities(text, labels)

    entities = []
    for label, values in result.get("entities", {}).items():
        for v in values:
            entities.append({"text": v, "label": label})

    logger.debug(f"Extracted entities: {entities}")
    known_entities = {e["text"] for e in entities}
    triplets_heuristic = _extract_relations_from_entities(text, entities)
    triplets_verb = _extract_verb_relations(text, known_entities)

    seen = set()
    combined = []
    for t in triplets_heuristic + triplets_verb:
        key = (t.subject.lower(), t.relation.lower(), t.object.lower())
        if key not in seen:
            seen.add(key)
            combined.append(t)

    logger.debug(f"Combined triplets: {combined}")
    return combined


@safe
def resolve_coreferences(text: str) -> str:
    model = get_fastcoref_model()
    preds = model.predict(texts=[text])
    return preds[0].get_resolved_text()


def process_text_pipeline(
    text: str, use_coref: bool = False
) -> Result[Tuple[str, List[Triple]], Exception]:
    if use_coref:
        return resolve_coreferences(text).bind(
            lambda resolved: extract_triplets(resolved).map(
                lambda triplets: (resolved, triplets)
            )
        )
    else:
        return extract_triplets(text).map(lambda triplets: (text, triplets))
