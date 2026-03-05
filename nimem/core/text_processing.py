import logging
import re
from typing import List, Tuple, NamedTuple
from functools import lru_cache

from gliner import GLiNER
from returns.result import Result, safe

from . import schema

logger = logging.getLogger(__name__)


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
def get_gliner_model():
    logger.info("Loading GLiNER model: urchade/gliner_small-v2.1")
    return GLiNER.from_pretrained("urchade/gliner_small-v2.1")


@lru_cache(maxsize=1)
def get_fastcoref_model():
    from fastcoref import FCoref

    logger.info("Loading FastCoref model")
    return FCoref(device="cpu")


def _infer_relation(entity1_label: str, entity2_label: str) -> str | None:
    key = (entity1_label.lower(), entity2_label.lower())
    return ENTITY_RELATION_MAP.get(key)


def _extract_relations_from_entities(text: str, entities: List[dict]) -> List[Triple]:
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
    for entity in entities:
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


@safe
def extract_triplets(text: str) -> List[Triple]:
    model = get_gliner_model()
    labels = list(schema.ENTITIES.keys())
    entities = model.predict_entities(text, labels, threshold=0.5)
    logger.debug(f"Extracted entities: {entities}")
    triplets = _extract_relations_from_entities(text, entities)
    logger.debug(f"Inferred triplets: {triplets}")
    return triplets


@safe
def resolve_coreferences(text: str) -> str:
    model = get_fastcoref_model()
    preds = model.predict(texts=[text])
    return preds[0].get_resolved_text()


def process_text_pipeline(
    text: str, use_coref: bool = False
) -> Result[Tuple[str, List[Triple]], Exception]:
    """
    Text processing pipeline: optionally resolve coreferences, then extract triplets.

    Args:
        text: Input text to process
        use_coref: If True, resolve coreferences before extraction (requires FastCoref)
    """
    if use_coref:
        return resolve_coreferences(text).bind(
            lambda resolved: extract_triplets(resolved).map(
                lambda triplets: (resolved, triplets)
            )
        )
    else:
        return extract_triplets(text).map(lambda triplets: (text, triplets))
