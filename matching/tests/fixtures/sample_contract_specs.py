"""
Sample ContractSpec fixtures for testing.
"""

from datetime import datetime
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from canonicalization.contract_spec import (
    ContractSpec,
    DateSpec,
    EntitySpec,
    ThresholdSpec
)


def create_bitcoin_spec_60k() -> ContractSpec:
    """Create Bitcoin $60k ContractSpec."""
    return ContractSpec(
        statement="Will Bitcoin close above $60,000 on Coinbase by Dec 31, 2024?",
        resolution_date=DateSpec(
            date=datetime(2024, 12, 31),
            is_deadline=True
        ),
        entities=[
            EntitySpec(name="Bitcoin", entity_type="other", aliases=["BTC"]),
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


def create_bitcoin_spec_80k() -> ContractSpec:
    """Create Bitcoin $80k ContractSpec."""
    return ContractSpec(
        statement="Will Bitcoin close above $80,000 on Coinbase by Dec 31, 2024?",
        resolution_date=DateSpec(
            date=datetime(2024, 12, 31),
            is_deadline=True
        ),
        entities=[
            EntitySpec(name="Bitcoin", entity_type="other", aliases=["BTC"]),
            EntitySpec(name="Coinbase", entity_type="organization")
        ],
        thresholds=[
            ThresholdSpec(value=80000.0, unit="dollars", comparison=">")
        ],
        resolution_criteria="Market resolves based on Coinbase BTC/USD closing price",
        data_source="Coinbase",
        outcome_labels=["Yes", "No"],
        confidence=0.95
    )


def create_ethereum_spec_10k() -> ContractSpec:
    """Create Ethereum $10k ContractSpec."""
    return ContractSpec(
        statement="Will Ethereum reach $10,000 by Dec 31, 2024?",
        resolution_date=DateSpec(
            date=datetime(2024, 12, 31),
            is_deadline=True
        ),
        entities=[
            EntitySpec(name="Ethereum", entity_type="other", aliases=["ETH"])
        ],
        thresholds=[
            ThresholdSpec(value=10000.0, unit="dollars", comparison=">")
        ],
        outcome_labels=["Yes", "No"],
        confidence=0.9
    )


def create_simple_spec_no_entities() -> ContractSpec:
    """Create simple ContractSpec without entities."""
    return ContractSpec(
        statement="Will the price exceed $50k by end of year?",
        resolution_date=DateSpec(
            date=datetime(2024, 12, 31),
            is_deadline=True
        ),
        entities=[],
        thresholds=[
            ThresholdSpec(value=50000.0, unit="dollars", comparison=">")
        ],
        outcome_labels=["Yes", "No"],
        confidence=0.85
    )


def create_multi_outcome_spec() -> ContractSpec:
    """Create ContractSpec with multiple outcomes."""
    return ContractSpec(
        statement="Which cryptocurrency will have the highest market cap by 2025?",
        resolution_date=DateSpec(
            date=datetime(2025, 12, 31),
            is_deadline=True
        ),
        entities=[
            EntitySpec(name="Bitcoin", entity_type="other"),
            EntitySpec(name="Ethereum", entity_type="other")
        ],
        thresholds=[],
        outcome_labels=["Bitcoin", "Ethereum", "Other"],
        confidence=0.8
    )

