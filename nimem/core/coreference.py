import logging

from returns.result import safe

from .model_loader import get_model

logger = logging.getLogger(__name__)


@safe
def resolve_coreferences(text: str) -> str:
    model = get_model("fastcoref")
    preds = model.predict(texts=[text])
    return preds[0].get_resolved_text()
