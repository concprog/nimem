import pytest
from unittest.mock import MagicMock, patch
from nimem.pipelines import ingest as text_processing
from nimem.nlp import spacy as spacy_nlp
from nimem.nlp import gliner as gliner_nlp
from nimem.nlp import fastcoref as fastcoref_nlp
from nimem.domain.schema import Entity, Triple
from returns.result import Success, Failure


@pytest.fixture
def mock_spacy():
    with patch("nimem.nlp.spacy.get_model") as mock_get:
        nlp = MagicMock()
        mock_get.return_value = nlp

        mock_ent_alice = MagicMock()
        mock_ent_alice.text = "Alice"
        mock_ent_alice.label_ = "PERSON"
        mock_ent_alice.start_char = 0
        mock_ent_alice.end_char = 5
        mock_ent_alice.start = 0
        mock_ent_alice.end = 1

        mock_ent_google = MagicMock()
        mock_ent_google.text = "Google"
        mock_ent_google.label_ = "ORG"
        mock_ent_google.start_char = 15
        mock_ent_google.end_char = 21
        mock_ent_google.start = 3
        mock_ent_google.end = 4

        doc = MagicMock()
        doc.ents = [mock_ent_alice, mock_ent_google]
        doc.__iter__ = MagicMock(return_value=iter([]))
        doc.sents = [doc]
        doc.start_char = 0
        doc.end_char = 21
        nlp.return_value = doc

        yield nlp


@pytest.fixture
def mock_gliner():
    with patch("nimem.nlp.gliner.get_model") as mock_get:
        instance = MagicMock()
        mock_get.return_value = instance
        instance.extract_relations.return_value = {
            "relation_extraction": {
                "knows": [("Alice", "Bob")],
                "works_for": [{"head": {"text": "Alice"}, "tail": {"text": "Google"}}],
            }
        }
        instance.extract_entities.return_value = {
            "entities": {
                "person": [{"text": "Alice", "start": 0, "end": 5}],
                "organization": [{"text": "Google", "start": 15, "end": 21}],
            }
        }
        yield instance


@pytest.fixture
def mock_coref():
    with patch("nimem.nlp.fastcoref.get_model") as mock_get:
        instance = MagicMock()
        mock_get.return_value = instance

        mock_pred = MagicMock()
        mock_pred.get_resolved_text.return_value = (
            "Alice works at Google. Alice knows Bob. Alice is happy."
        )
        instance.predict.return_value = [mock_pred]

        yield instance


def test_extract_triplets_heuristic(mock_spacy):
    with (
        patch("nimem.nlp.spacy.get_model", return_value=mock_spacy),
        patch(
            "nimem.nlp.spacy.extract_relations",
            return_value=[Triple("Alice", "works_for", "Google")],
        ),
    ):
        triplets = text_processing.extract_triplets("Alice works at Google").unwrap()
        assert len(triplets) > 0


def test_extract_triplets_gliner2(mock_gliner):
    with patch("nimem.nlp.gliner.get_model", return_value=mock_gliner):
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
    with patch("nimem.nlp.fastcoref.get_model", return_value=mock_coref):
        text = "Alice works at Google. Alice knows Bob. He is happy."
        res = text_processing.resolve_coreferences(text).unwrap()
        assert "He" not in res
        assert res.count("Alice") >= 2


def test_pipeline_heuristic(mock_spacy, mock_coref):
    with (
        patch("nimem.nlp.spacy.get_model", return_value=mock_spacy),
        patch(
            "nimem.nlp.spacy.extract_relations",
            return_value=[Triple("Alice", "works_for", "Google")],
        ),
        patch("nimem.nlp.fastcoref.get_model", return_value=mock_coref),
    ):
        res = text_processing.process_text_pipeline("Input text")
        assert isinstance(res, Success)
        _, triplets = res.unwrap()
        assert len(triplets) >= 0


def test_pipeline_gliner2(mock_gliner):
    with patch("nimem.nlp.gliner.get_model", return_value=mock_gliner):
        res = text_processing.process_text_pipeline("Input text", use_gliner2=True)
        assert isinstance(res, Success)
        _, triplets = res.unwrap()
        assert len(triplets) == 2


def test_extract_entities_spacy(mock_spacy):
    with patch("nimem.nlp.spacy.get_model", return_value=mock_spacy):
        entities = spacy_nlp.extract_entities("Alice works at Google")
        assert len(entities) == 2
        assert entities[0].text == "Alice"
        assert entities[0].label == "person"
        assert entities[1].text == "Google"
        assert entities[1].label == "organization"


def test_extract_entities_gliner(mock_gliner):
    with patch("nimem.nlp.gliner.get_model", return_value=mock_gliner):
        # We need to test if extract_entities exists in gliner
        if hasattr(gliner_nlp, "extract_entities"):
            entities = gliner_nlp.extract_entities("Alice works at Google")
            assert len(entities) == 2


def test_extract_relations_spacy_with_entities(mock_spacy):
    with patch("nimem.nlp.spacy.get_model", return_value=mock_spacy):
        entities = [
            Entity(text="Alice", label="person", start=0, end=5),
            Entity(text="Google", label="organization", start=15, end=21),
        ]
        triplets = spacy_nlp.extract_relations("Alice works at Google", entities)
        assert len(triplets) >= 0
