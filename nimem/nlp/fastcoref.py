import logging
import spacy
from returns.result import safe

logger = logging.getLogger(__name__)

_nlp_coref = None

def get_coref_pipeline():
    global _nlp_coref
    if _nlp_coref is None:
        from fastcoref import spacy_component 
        
        logger.info("Loading FastCoref spaCy pipeline")
        _nlp_coref = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])
        _nlp_coref.add_pipe("fastcoref")
    return _nlp_coref

@safe
def resolve(text: str) -> str:
    """
    Resolve coreferences using the official spaCy component method.
    """
    nlp = get_coref_pipeline()
    
    doc = nlp(
        text, 
        component_cfg={"fastcoref": {'resolve_text': True}}
    )
    
    if hasattr(doc._, 'resolved_text'):
        return doc._.resolved_text
        
    logger.warning("Coreference resolution attribute not found; returning original text.")
    return text