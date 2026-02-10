"""
Golden dataset for pair verification testing.

Contains known equivalent/non-equivalent pairs for validation.
For production, manually curate real market pairs from Kalshi/Polymarket.

How to create a real golden dataset:
1. Identify known equivalent markets across venues (manually verified)
2. Identify known non-equivalent markets (different events)
3. Identify ambiguous cases (needs review)
4. Extract ContractSpecs for each pair
5. Add to this dataset with expected verdicts
"""

from datetime import datetime
from typing import List, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from canonicalization.contract_spec import (
    ContractSpec,
    DateSpec,
    EntitySpec,
    ThresholdSpec
)


def get_golden_dataset() -> List[Dict[str, Any]]:
    """
    Get golden dataset of known test pairs.
    
    Returns:
        List of test cases with expected verdicts
    """
    return [
        # EQUIVALENT PAIRS
        {
            "name": "equivalent_bitcoin_price_same_threshold",
            "spec_a": ContractSpec(
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
            ),
            "spec_b": ContractSpec(
                statement="Will BTC exceed $60,000 USD on Coinbase before December 31, 2024?",
                resolution_date=DateSpec(
                    date=datetime(2024, 12, 31),
                    is_deadline=True
                ),
                entities=[
                    EntitySpec(name="BTC", entity_type="other", aliases=["Bitcoin"]),
                    EntitySpec(name="Coinbase", entity_type="organization")
                ],
                thresholds=[
                    ThresholdSpec(value=60000.0, unit="dollars", comparison=">")
                ],
                resolution_criteria="Resolves using Coinbase BTC/USD closing price",
                data_source="Coinbase",
                outcome_labels=["YES", "NO"],
                confidence=0.92
            ),
            "expected_verdict": "equivalent",
            "expected_confidence_min": 0.85,
            "expected_outcome_mapping": {"Yes": "YES", "No": "NO"}
        },
        
        # NOT EQUIVALENT PAIRS
        {
            "name": "not_equivalent_different_threshold",
            "spec_a": ContractSpec(
                statement="Will Bitcoin close above $60,000 by Dec 31, 2024?",
                resolution_date=DateSpec(
                    date=datetime(2024, 12, 31),
                    is_deadline=True
                ),
                entities=[
                    EntitySpec(name="Bitcoin", entity_type="other")
                ],
                thresholds=[
                    ThresholdSpec(value=60000.0, unit="dollars", comparison=">")
                ],
                outcome_labels=["Yes", "No"],
                confidence=0.9
            ),
            "spec_b": ContractSpec(
                statement="Will Bitcoin close above $80,000 by Dec 31, 2024?",
                resolution_date=DateSpec(
                    date=datetime(2024, 12, 31),
                    is_deadline=True
                ),
                entities=[
                    EntitySpec(name="Bitcoin", entity_type="other")
                ],
                thresholds=[
                    ThresholdSpec(value=80000.0, unit="dollars", comparison=">")
                ],
                outcome_labels=["Yes", "No"],
                confidence=0.9
            ),
            "expected_verdict": "not_equivalent",
            "expected_confidence_max": 0.55
        },
        
        {
            "name": "not_equivalent_different_entity",
            "spec_a": ContractSpec(
                statement="Will Bitcoin reach $100k by 2025?",
                resolution_date=DateSpec(
                    date=datetime(2025, 12, 31),
                    is_deadline=True
                ),
                entities=[
                    EntitySpec(name="Bitcoin", entity_type="other")
                ],
                thresholds=[
                    ThresholdSpec(value=100000.0, unit="dollars", comparison=">")
                ],
                outcome_labels=["Yes", "No"],
                confidence=0.9
            ),
            "spec_b": ContractSpec(
                statement="Will Ethereum reach $10k by 2025?",
                resolution_date=DateSpec(
                    date=datetime(2025, 12, 31),
                    is_deadline=True
                ),
                entities=[
                    EntitySpec(name="Ethereum", entity_type="other")
                ],
                thresholds=[
                    ThresholdSpec(value=10000.0, unit="dollars", comparison=">")
                ],
                outcome_labels=["Yes", "No"],
                confidence=0.9
            ),
            "expected_verdict": "not_equivalent",
            "expected_confidence_max": 0.4
        },
        
        # NEEDS REVIEW (AMBIGUOUS)
        {
            "name": "needs_review_similar_but_different_dates",
            "spec_a": ContractSpec(
                statement="Will Bitcoin reach $100k by end of 2024?",
                resolution_date=DateSpec(
                    date=datetime(2024, 12, 31),
                    is_deadline=True
                ),
                entities=[
                    EntitySpec(name="Bitcoin", entity_type="other")
                ],
                thresholds=[
                    ThresholdSpec(value=100000.0, unit="dollars", comparison=">")
                ],
                outcome_labels=["Yes", "No"],
                confidence=0.9
            ),
            "spec_b": ContractSpec(
                statement="Will Bitcoin reach $100k by end of 2025?",
                resolution_date=DateSpec(
                    date=datetime(2025, 12, 31),
                    is_deadline=True
                ),
                entities=[
                    EntitySpec(name="Bitcoin", entity_type="other")
                ],
                thresholds=[
                    ThresholdSpec(value=100000.0, unit="dollars", comparison=">")
                ],
                outcome_labels=["Yes", "No"],
                confidence=0.9
            ),
            "expected_verdict": "needs_review",
            "expected_confidence_min": 0.5,
            "expected_confidence_max": 0.9
        },
        
        {
            "name": "needs_review_close_thresholds",
            "spec_a": ContractSpec(
                statement="Will Bitcoin close above $60,000 by Dec 31, 2024?",
                resolution_date=DateSpec(
                    date=datetime(2024, 12, 31),
                    is_deadline=True
                ),
                entities=[
                    EntitySpec(name="Bitcoin", entity_type="other")
                ],
                thresholds=[
                    ThresholdSpec(value=60000.0, unit="dollars", comparison=">")
                ],
                outcome_labels=["Yes", "No"],
                confidence=0.9
            ),
            "spec_b": ContractSpec(
                statement="Will Bitcoin close above $61,000 by Dec 31, 2024?",
                resolution_date=DateSpec(
                    date=datetime(2024, 12, 31),
                    is_deadline=True
                ),
                entities=[
                    EntitySpec(name="Bitcoin", entity_type="other")
                ],
                thresholds=[
                    ThresholdSpec(value=61000.0, unit="dollars", comparison=">")
                ],
                outcome_labels=["Yes", "No"],
                confidence=0.9
            ),
            "expected_verdict": "needs_review",
            "expected_confidence_min": 0.5,
            "expected_confidence_max": 0.9
        },
        
        # EDGE CASES
        {
            "name": "equivalent_no_entities",
            "spec_a": ContractSpec(
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
            ),
            "spec_b": ContractSpec(
                statement="Will price go above $50,000 by December 31, 2024?",
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
            ),
            "expected_verdict": "equivalent",
            "expected_confidence_min": 0.8
        },
        
        {
            "name": "equivalent_with_aliases",
            "spec_a": ContractSpec(
                statement="Will Bitcoin reach $100k?",
                resolution_date=DateSpec(
                    date=datetime(2025, 12, 31),
                    is_deadline=True
                ),
                entities=[
                    EntitySpec(name="Bitcoin", entity_type="other", aliases=["BTC"])
                ],
                thresholds=[
                    ThresholdSpec(value=100000.0, unit="dollars", comparison=">")
                ],
                outcome_labels=["Yes", "No"],
                confidence=0.9
            ),
            "spec_b": ContractSpec(
                statement="Will BTC reach $100k?",
                resolution_date=DateSpec(
                    date=datetime(2025, 12, 31),
                    is_deadline=True
                ),
                entities=[
                    EntitySpec(name="BTC", entity_type="other", aliases=["Bitcoin"])
                ],
                thresholds=[
                    ThresholdSpec(value=100000.0, unit="dollars", comparison=">")
                ],
                outcome_labels=["Yes", "No"],
                confidence=0.9
            ),
            "expected_verdict": "equivalent",
            "expected_confidence_min": 0.85
        }
    ]


def get_edge_case_dataset() -> List[Dict[str, Any]]:
    """
    Get edge case dataset for testing boundary conditions.
    
    Returns:
        List of edge case test scenarios
    """
    return [
        {
            "name": "edge_case_missing_dates",
            "spec_a": ContractSpec(
                statement="Will Bitcoin reach $100k?",
                resolution_date=None,
                entities=[
                    EntitySpec(name="Bitcoin", entity_type="other")
                ],
                thresholds=[
                    ThresholdSpec(value=100000.0, unit="dollars", comparison=">")
                ],
                outcome_labels=["Yes", "No"],
                confidence=0.8
            ),
            "spec_b": ContractSpec(
                statement="Will Bitcoin reach $100k?",
                resolution_date=DateSpec(
                    date=datetime(2025, 12, 31),
                    is_deadline=True
                ),
                entities=[
                    EntitySpec(name="Bitcoin", entity_type="other")
                ],
                thresholds=[
                    ThresholdSpec(value=100000.0, unit="dollars", comparison=">")
                ],
                outcome_labels=["Yes", "No"],
                confidence=0.8
            ),
            "cross_encoder_score": 0.7,
            "expected_verdict": "needs_review"
        },
        
        {
            "name": "edge_case_empty_entities",
            "spec_a": ContractSpec(
                statement="Will the price exceed $50k?",
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
            ),
            "spec_b": ContractSpec(
                statement="Will the price exceed $50k?",
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
            ),
            "cross_encoder_score": 0.9,
            "expected_verdict": "equivalent"
        },
        
        {
            "name": "edge_case_different_data_sources",
            "spec_a": ContractSpec(
                statement="Will Bitcoin reach $100k?",
                resolution_date=DateSpec(
                    date=datetime(2025, 12, 31),
                    is_deadline=True
                ),
                entities=[
                    EntitySpec(name="Bitcoin", entity_type="other")
                ],
                thresholds=[
                    ThresholdSpec(value=100000.0, unit="dollars", comparison=">")
                ],
                data_source="Coinbase",
                outcome_labels=["Yes", "No"],
                confidence=0.9
            ),
            "spec_b": ContractSpec(
                statement="Will Bitcoin reach $100k?",
                resolution_date=DateSpec(
                    date=datetime(2025, 12, 31),
                    is_deadline=True
                ),
                entities=[
                    EntitySpec(name="Bitcoin", entity_type="other")
                ],
                thresholds=[
                    ThresholdSpec(value=100000.0, unit="dollars", comparison=">")
                ],
                data_source="Binance",
                outcome_labels=["Yes", "No"],
                confidence=0.9
            ),
            "expected_verdict": "needs_review"
        }
    ]

