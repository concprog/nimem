import pytest
from unittest.mock import MagicMock, patch
from nimem.core import text_processing, schema
from returns.result import Success, Failure

# Mock heavy dependencies
@pytest.fixture(autouse=True)
def mock_models():
    with patch('nimem.core.text_processing.GLiNER2') as mock_gliner, \
         patch('nimem.core.text_processing.FCoref') as mock_fcoref:
        
        # Setup GLiNER mock
        mock_model_instance = MagicMock()
        mock_gliner.from_pretrained.return_value = mock_model_instance
        # Mock API: extract_relations returns dict
        mock_model_instance.extract_relations.return_value = {
            'relation_extraction': {
                'knows': [('Alice', 'Bob')],
                'works_for': [{'head': {'text': 'Alice'}, 'tail': {'text': 'Google'}}]
            }
        }
        
        # Setup FCoref mock
        mock_fcoref_instance = MagicMock()
        mock_fcoref.return_value = mock_fcoref_instance
        mock_preds = MagicMock()
        mock_preds.get_resolved_text.return_value = "Alice works at Google. Alice knows Bob."
        mock_fcoref_instance.predict.return_value = [mock_preds]
        
        yield mock_gliner, mock_fcoref

def test_extract_triplets_success():
    triplets = text_processing.extract_triplets("Alice works at Google").unwrap()
    assert len(triplets) == 2
    
    # Check Tuple format
    assert triplets[0].subject == 'Alice'
    assert triplets[0].relation == 'knows'
    assert triplets[0].object == 'Bob'
    
    # Check Dict format conversion
    assert triplets[1].subject == 'Alice'
    assert triplets[1].relation == 'works_for'
    assert triplets[1].object == 'Google'

def test_resolve_coreferences():
    res = text_processing.resolve_coreferences("She works at Google.").unwrap()
    assert res == "Alice works at Google. Alice knows Bob."

def test_pipeline_integration():
    # Test the bind logic
    res = text_processing.process_text_pipeline("Input text")
    assert isinstance(res, Success)
    resolved, triplets = res.unwrap()
    assert resolved == "Alice works at Google. Alice knows Bob."
    assert len(triplets) == 2
