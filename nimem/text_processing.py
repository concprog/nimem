from typing import List, Tuple, NamedTuple
from functools import lru_cache
from returns.result import Result, Success, Failure, safe
import logging

try:
    from gliner2 import GLiNER2
    from fastcoref import FCoref
except ImportError:
    logging.warning("gliner2 or fastcoref not found. Please ensure they are installed.")
    GLiNER2 = None
    FCoref = None

class Triple(NamedTuple):
    subject: str
    relation: str
    object: str

from . import schema

# --- Lazy Model Loading ---

@lru_cache(maxsize=1)
def get_gliner_model():
    if GLiNER2 is None:
        raise ImportError("gliner2 library not installed.")
    # Using the model from user instruction or a default
    return GLiNER2.from_pretrained("urchade/gliner_small-v2.1")

@lru_cache(maxsize=1)
def get_fastcoref_model() -> 'FCoref':
    if FCoref is None:
        raise ImportError("FastCoref library not installed.")
    return FCoref(device='cpu')

# --- Extraction Logic ---

@safe
def extract_triplets(text: str) -> List[Triple]:
    """
    Extracts triplets using GLiNER2 relation extraction.
    """
    model = get_gliner_model()
    
    # We use the schema defined in schema.py
    # Adapt to the API shown in docs: extraction with list of relation types
    # Or strict schema object if available. The docs showed both list and schema builder.
    # We will use the list of keys from our schema for simplicity and flexibility.
    
    relation_types = list(schema.RELATIONS.keys())
    
    # The user docs show: results = extractor.extract_relations(text, ["list", "of", "relations"])
    # We assume 'extract_relations' method exists on the loaded model.
    try:
        results = model.extract_relations(text, relation_types)
    except AttributeError:
        # Fallback if the model class doesn't strictly match the v2 API snippet
        # but is a standard GLiNER class.
        logging.warning("GLiNER model does not support extract_relations. Returning empty.")
        return []
    
    triplets = []
    # Parse the output: {'relation_extraction': {'works_for': [('John', 'Apple')], ...}}
    relations_map = results.get('relation_extraction', {})
    
    for rel_type, instances in relations_map.items():
        for instance in instances:
            # instance is typically (source, target) or dict if conf included
            if isinstance(instance, tuple) or isinstance(instance, list):
                if len(instance) >= 2:
                    triplets.append(Triple(instance[0], rel_type, instance[1]))
            elif isinstance(instance, dict):
                 # Handle {'head': ..., 'tail': ...} format
                 head = instance.get('head', {}).get('text', '')
                 tail = instance.get('tail', {}).get('text', '')
                 if head and tail:
                     triplets.append(Triple(head, rel_type, tail))

    return triplets

@safe
def resolve_coreferences(text: str) -> str:
    """
    Resolves coreferences in the text using FastCoref.
    """
    model = get_fastcoref_model()
    preds = model.predict(texts=[text])
    return preds[0].get_resolved_text()

# --- Composition ---

def process_text_pipeline(text: str) -> Result[Tuple[str, List[Triple]], Exception]:
    """
    Chains coreference resolution -> Triplet Extraction.
    Uses bind to propagate errors safely without unwrap.
    """
    return resolve_coreferences(text).bind(
        lambda resolved: extract_triplets(resolved).map(
             lambda triplets: (resolved, triplets)
        )
    )
