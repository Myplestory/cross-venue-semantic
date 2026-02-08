"""Tests for ContractSpec model."""

import pytest
import json
from datetime import datetime
from canonicalization.contract_spec import (
    ContractSpec,
    DateSpec,
    EntitySpec,
    ThresholdSpec,
)


class TestDateSpec:
    """Tests for DateSpec model."""
    
    def test_create_date_spec(self):
        """Test creating DateSpec."""
        date = datetime(2024, 12, 31)
        spec = DateSpec(date=date, timezone="UTC", is_deadline=True)
        
        assert spec.date == date
        assert spec.timezone == "UTC"
        assert spec.is_deadline is True
    
    def test_date_spec_defaults(self):
        """Test DateSpec defaults."""
        date = datetime(2024, 12, 31)
        spec = DateSpec(date=date)
        
        assert spec.timezone is None
        assert spec.is_deadline is False


class TestEntitySpec:
    """Tests for EntitySpec model."""
    
    def test_create_entity_spec(self):
        """Test creating EntitySpec."""
        entity = EntitySpec(
            name="Bitcoin",
            entity_type="other",
            aliases=["BTC", "Bitcoin"]
        )
        
        assert entity.name == "Bitcoin"
        assert entity.entity_type == "other"
        assert len(entity.aliases) == 2
    
    def test_entity_spec_defaults(self):
        """Test EntitySpec defaults."""
        entity = EntitySpec(name="Trump", entity_type="person")
        
        assert entity.aliases == []


class TestThresholdSpec:
    """Tests for ThresholdSpec model."""
    
    def test_create_threshold_spec(self):
        """Test creating ThresholdSpec."""
        threshold = ThresholdSpec(
            value=100000.0,
            unit="dollars",
            comparison=">="
        )
        
        assert threshold.value == 100000.0
        assert threshold.unit == "dollars"
        assert threshold.comparison == ">="
    
    def test_threshold_spec_defaults(self):
        """Test ThresholdSpec defaults."""
        threshold = ThresholdSpec(value=50.0)
        
        assert threshold.unit is None
        assert threshold.comparison == ">="


class TestContractSpec:
    """Tests for ContractSpec model."""
    
    def test_create_minimal_contract_spec(self):
        """Test creating minimal ContractSpec."""
        spec = ContractSpec(statement="Will Bitcoin reach $100,000?")
        
        assert spec.statement == "Will Bitcoin reach $100,000?"
        assert spec.entities == []
        assert spec.thresholds == []
        assert spec.confidence == 0.0
    
    def test_create_full_contract_spec(self):
        """Test creating full ContractSpec."""
        resolution_date = DateSpec(
            date=datetime(2024, 12, 31),
            is_deadline=True
        )
        entity = EntitySpec(name="Bitcoin", entity_type="other")
        threshold = ThresholdSpec(value=100000.0, unit="dollars")
        
        spec = ContractSpec(
            statement="Will Bitcoin reach $100,000 by Dec 31, 2024?",
            resolution_date=resolution_date,
            entities=[entity],
            thresholds=[threshold],
            resolution_criteria="Resolves YES if Bitcoin closes above $100,000",
            data_source="Coinbase",
            outcome_labels=["Yes", "No"],
            confidence=0.95,
        )
        
        assert spec.statement == "Will Bitcoin reach $100,000 by Dec 31, 2024?"
        assert spec.resolution_date is not None
        assert len(spec.entities) == 1
        assert len(spec.thresholds) == 1
        assert spec.confidence == 0.95
    
    def test_contract_spec_validation(self):
        """Test ContractSpec validation."""
        # Valid entity type
        spec = ContractSpec(
            statement="Test",
            entities=[EntitySpec(name="Test", entity_type="person")]
        )
        assert spec is not None
        
        # Invalid entity type
        with pytest.raises(Exception):  # Pydantic validation error
            ContractSpec(
                statement="Test",
                entities=[EntitySpec(name="Test", entity_type="invalid")]
            )
    
    def test_contract_spec_json_serialization(self):
        """Test ContractSpec JSON serialization."""
        spec = ContractSpec(
            statement="Will Bitcoin reach $100,000?",
            confidence=0.9
        )
        
        json_str = spec.model_dump_json()
        data = json.loads(json_str)
        
        assert data["statement"] == "Will Bitcoin reach $100,000?"
        assert data["confidence"] == 0.9
    
    def test_contract_spec_json_deserialization(self):
        """Test ContractSpec JSON deserialization."""
        json_str = '{"statement": "Will Bitcoin reach $100,000?", "confidence": 0.9}'
        
        spec = ContractSpec.parse_raw(json_str)
        
        assert spec.statement == "Will Bitcoin reach $100,000?"
        assert spec.confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_contract_spec_async_methods(self):
        """Test async ContractSpec methods."""
        spec = ContractSpec(statement="Test statement")
        
        # Test async JSON serialization
        json_str = await spec.to_json_async()
        assert isinstance(json_str, str)
        assert "Test statement" in json_str
        
        # Test async JSON deserialization
        spec2 = await ContractSpec.from_json_async(json_str)
        assert spec2.statement == "Test statement"
    
    def test_contract_spec_confidence_bounds(self):
        """Test confidence score bounds validation."""
        # Valid confidence
        spec = ContractSpec(statement="Test", confidence=0.5)
        assert spec.confidence == 0.5
        
        # Confidence too high
        with pytest.raises(Exception):  # Pydantic validation error
            ContractSpec(statement="Test", confidence=1.5)
        
        # Confidence too low
        with pytest.raises(Exception):  # Pydantic validation error
            ContractSpec(statement="Test", confidence=-0.1)

