import logging
from typing import Tuple, List

from returns.result import Result

from .schema import Triple, Entity
from . import relation_extraction
from . import coreference
from . import entity_recognition

logger = logging.getLogger(__name__)


def extract_entities_spacy(text: str) -> List[Entity]:
    return entity_recognition.extract_entities_spacy(text)


def extract_entities_gliner(
    text: str,
    include_confidence: bool = False,
    include_spans: bool = False,
) -> List[Entity]:
    return entity_recognition.extract_entities_gliner(
        text,
        include_confidence=include_confidence,
        include_spans=include_spans,
    )


def extract_relations_spacy(text: str, entities: List[Entity]) -> List[Triple]:
    return relation_extraction.extract_relations_spacy(text, entities)


def extract_triplets(
    text: str, use_gliner2: bool = False
) -> Result[List[Triple], Exception]:
    return relation_extraction.extract_triplets(text, use_gliner2=use_gliner2)


def resolve_coreferences(text: str) -> Result[str, Exception]:
    return coreference.resolve_coreferences(text)


def process_text_pipeline(
    text: str, use_coref: bool = False, use_gliner2: bool = False
) -> Result[Tuple[str, List[Triple]], Exception]:
    if use_coref:
        return resolve_coreferences(text).bind(
            lambda resolved: extract_triplets(resolved, use_gliner2=use_gliner2).map(
                lambda triplets: (resolved, triplets)
            )
        )
    else:
        return extract_triplets(text, use_gliner2=use_gliner2).map(
            lambda triplets: (text, triplets)
        )


Triple = Triple
Entity = Entity
