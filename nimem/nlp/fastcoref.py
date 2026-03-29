import logging
from returns.result import safe

logger = logging.getLogger(__name__)

_model_instance = None


def get_model():
    global _model_instance
    if _model_instance is None:
        from fastcoref import FCoref

        logger.info("Loading FastCoref model")
        _model_instance = FCoref(device="cpu")
    return _model_instance


@safe
def resolve(text: str) -> str:
    model = get_model()
    preds = model.predict(texts=[text])
    return preds[0].get_resolved_text()
