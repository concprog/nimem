import logging
from typing import Tuple, List
from returns.result import Result, safe

from nimem.domain.schema import Triple, Entity
from nimem.nlp.spacy import extract_triplets_spacy
from nimem.nlp.gliner import extract_relations as extract_triplets_gliner
from nimem.storage.conceptnet_store import extract_triplets_conceptnet
from nimem.nlp.fastcoref import resolve as resolve_coreferences

logger = logging.getLogger(__name__)


@safe
def extract_triplets(
    text: str, use_gliner2: bool = False, use_conceptnet: bool = False
) -> List[Triple]:
    """Extract triplets using specified method."""
    if use_gliner2:
        return extract_triplets_gliner(text)
    if use_conceptnet:
        return extract_triplets_spacy(text, use_conceptnet=True)
    return extract_triplets_spacy(text)


def process_text_pipeline(
    text: str,
    use_coref: bool = False,
    use_gliner2: bool = False,
    use_conceptnet: bool = False,
) -> Result[Tuple[str, List[Triple]], Exception]:
    if use_coref:
        return resolve_coreferences(text).bind(
            lambda resolved: extract_triplets(
                resolved, use_gliner2=use_gliner2, use_conceptnet=use_conceptnet
            ).map(lambda triplets: (resolved, triplets))
        )
    else:
        return extract_triplets(
            text, use_gliner2=use_gliner2, use_conceptnet=use_conceptnet
        ).map(lambda triplets: (text, triplets))
