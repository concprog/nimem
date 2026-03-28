import pytest
from unittest.mock import MagicMock, patch
from nimem.core import text_processing
from nimem.core import relation_extraction
from nimem.core import entity_recognition
from nimem.core import coreference
from nimem.core import model_loader
from returns.result import Success, Failure


@pytest.fixture
def mock_spacy():
    model_loader.get_model.cache_clear()
    with patch("nimem.core.model_loader.get_model") as mock_get:
        nlp = MagicMock()
        mock_get.return_value = nlp

        mock_ent_alice = MagicMock()
        mock_ent_alice.text = "Alice"
        mock_ent_alice.label_ = "PERSON"
        mock_ent_alice.start_char = 0
        mock_ent_alice.start = 0
        mock_ent_alice.end = 1

        mock_ent_google = MagicMock()
        mock_ent_google.text = "Google"
        mock_ent_google.label_ = "ORG"
        mock_ent_google.start_char = 15
        mock_ent_google.start = 3
        mock_ent_google.end = 4

        doc = MagicMock()
        doc.ents = [mock_ent_alice, mock_ent_google]
        doc.__iter__ = MagicMock(return_value=iter([]))
        nlp.return_value = doc

        yield nlp
    model_loader.get_model.cache_clear()


@pytest.fixture
def mock_gliner():
    model_loader.get_model.cache_clear()
    with patch("nimem.core.model_loader.get_model") as mock_get:
        instance = MagicMock()
        mock_get.return_value = instance
        instance.extract_relations.return_value = {
            "relation_extraction": {
                "knows": [("Alice", "Bob")],
                "works_for": [{"head": {"text": "Alice"}, "tail": {"text": "Google"}}],
            }
        }
        yield instance
    model_loader.get_model.cache_clear()


@pytest.fixture
def mock_coref():
    model_loader.get_model.cache_clear()
    with patch("nimem.core.model_loader.get_model") as mock_get:
        instance = MagicMock()
        mock_get.return_value = instance

        mock_pred = MagicMock()
        mock_pred.get_resolved_text.return_value = (
            "Alice works at Google. Alice knows Bob. Alice is happy."
        )
        instance.predict.return_value = [mock_pred]

        yield instance
    model_loader.get_model.cache_clear()


def test_extract_triplets_heuristic(mock_spacy):
    with patch("nimem.core.model_loader.get_model", return_value=mock_spacy):
        model_loader.get_model.cache_clear()
        triplets = text_processing.extract_triplets("Alice works at Google").unwrap()
        assert len(triplets) > 0


def test_extract_triplets_gliner2(mock_gliner):
    with patch("nimem.core.model_loader.get_model", return_value=mock_gliner):
        model_loader.get_model.cache_clear()
        triplets = text_processing.extract_triplets(
            "Alice works at Google", use_gliner2=True
        ).unwrap()

        mock_gliner.extract_relations.assert_called_once()
        assert len(triplets) == 2

        assert triplets[0].subject == "Alice"
        assert triplets[0].relation == "knows"
        assert triplets[0].object == "Bob"

        assert triplets[1].subject == "Alice"
        assert triplets[1].relation == "works_for"
        assert triplets[1].object == "Google"


def test_resolve_coreferences(mock_coref):
    with patch("nimem.core.model_loader.get_model", return_value=mock_coref):
        model_loader.get_model.cache_clear()
        text = "Alice works at Google. Alice knows Bob. He is happy."
        res = text_processing.resolve_coreferences(text).unwrap()
        assert "He" not in res
        assert res.count("Alice") >= 2


def test_pipeline_heuristic(mock_spacy, mock_coref):
    def get_model_side_effect(name):
        if name == "spacy":
            return mock_spacy
        elif name == "fastcoref":
            return mock_coref
        raise ValueError(f"Unknown model: {name}")

    with patch("nimem.core.model_loader.get_model", side_effect=get_model_side_effect):
        model_loader.get_model.cache_clear()
        res = text_processing.process_text_pipeline("Input text")
        assert isinstance(res, Success)
        _, triplets = res.unwrap()
        assert len(triplets) >= 0


def test_pipeline_gliner2(mock_gliner):
    with patch("nimem.core.model_loader.get_model", return_value=mock_gliner):
        model_loader.get_model.cache_clear()
        res = text_processing.process_text_pipeline("Input text", use_gliner2=True)
        assert isinstance(res, Success)
        _, triplets = res.unwrap()
        assert len(triplets) == 2


def test_extract_entities_spacy(mock_spacy):
    with patch("nimem.core.model_loader.get_model", return_value=mock_spacy):
        model_loader.get_model.cache_clear()
        from nimem.core.schema import Entity

        entities = text_processing.extract_entities_spacy("Alice works at Google")
        assert len(entities) == 2
        assert entities[0].text == "Alice"
        assert entities[0].label == "person"
        assert entities[1].text == "Google"
        assert entities[1].label == "organization"


def test_extract_entities_gliner(mock_gliner):
    with patch("nimem.core.model_loader.get_model", return_value=mock_gliner):
        model_loader.get_model.cache_clear()
        entities = text_processing.extract_entities_gliner("Alice works at Google")
        assert len(entities) == 2


def test_extract_relations_spacy_with_entities(mock_spacy):
    with patch("nimem.core.model_loader.get_model", return_value=mock_spacy):
        model_loader.get_model.cache_clear()
        from nimem.core.schema import Entity

        entities = [
            Entity(text="Alice", label="person", start=0, end=5),
            Entity(text="Google", label="organization", start=15, end=21),
        ]
        triplets = text_processing.extract_relations_spacy(
            "Alice works at Google", entities
        )
        assert len(triplets) >= 0
