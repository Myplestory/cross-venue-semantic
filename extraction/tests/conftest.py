"""Pytest configuration for extraction tests."""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, UTC

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def sample_canonical_text():
    """Sample canonical text for testing."""
    return """Market Statement:
Will Bitcoin close above $60,000 on Coinbase by Dec 31, 2024?

Resolution Criteria:
Market resolves based on Coinbase BTC/USD closing price at 11:59 PM ET on Dec 31, 2024.
Price must be verified from Coinbase API.

Clarifications:
- Uses Coinbase BTC/USD pair
- Closing price at end of day

End Date: 2024-12-31

Outcomes: Yes, No"""


@pytest.fixture
def sample_canonical_text_simple():
    """Simple canonical text without all sections."""
    return """Market Statement:
Will the price exceed $50k?

End Date: 2024-12-31"""


@pytest.fixture
def sample_canonical_text_with_negation():
    """Canonical text with negation patterns."""
    return """Market Statement:
Will Bitcoin not exceed $100,000 by end of year?

Resolution Criteria:
Market resolves unless price goes above $100k before Dec 31, 2024.

End Date: 2024-12-31"""


@pytest.fixture
def sample_canonical_text_multi_threshold():
    """Canonical text with multiple thresholds."""
    return """Market Statement:
Will Bitcoin be above $60k and below $80k by Dec 31?

Resolution Criteria:
Price must be between $60,000 and $80,000 at close.

End Date: 2024-12-31"""


@pytest.fixture
def mock_spacy_model():
    """Mock spaCy model for entity extraction."""
    mock_doc = MagicMock()
    mock_ent1 = MagicMock()
    mock_ent1.text = "Bitcoin"
    mock_ent1.label_ = "ORG"
    
    mock_ent2 = MagicMock()
    mock_ent2.text = "Coinbase"
    mock_ent2.label_ = "ORG"
    
    mock_doc.ents = [mock_ent1, mock_ent2]
    
    def nlp_callable(text):
        return mock_doc
    
    mock_nlp = MagicMock(side_effect=nlp_callable)
    
    return mock_nlp


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for LLM fallback."""
    from unittest.mock import AsyncMock, MagicMock
    
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"statement": "Test", "confidence": 0.9}'
    
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    return mock_client


@pytest.fixture
def sample_contract_spec():
    """Sample ContractSpec for testing."""
    from canonicalization.contract_spec import ContractSpec, DateSpec, EntitySpec, ThresholdSpec
    from datetime import datetime
    
    return ContractSpec(
        statement="Will Bitcoin close above $60,000 on Coinbase by Dec 31, 2024?",
        resolution_date=DateSpec(
            date=datetime(2024, 12, 31),
            is_deadline=True
        ),
        entities=[
            EntitySpec(name="Bitcoin", entity_type="other"),
            EntitySpec(name="Coinbase", entity_type="organization")
        ],
        thresholds=[
            ThresholdSpec(value=60000.0, unit="dollars", comparison=">")
        ],
        resolution_criteria="Market resolves based on Coinbase BTC/USD closing price",
        data_source="Coinbase",
        outcome_labels=["Yes", "No"],
        confidence=0.95
    )

