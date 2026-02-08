"""Edge case tests for ContractSpec model."""

import pytest
from datetime import datetime
from canonicalization.contract_spec import (
    ContractSpec,
    DateSpec,
    EntitySpec,
    ThresholdSpec,
)


class TestContractSpecEdgeCases:
    """Tests for edge cases in ContractSpec."""
    
    def test_minimal_contract_spec(self):
        """Test creating ContractSpec with only required field."""
        spec = ContractSpec(statement="")
        
        assert spec.statement == ""
        assert spec.entities == []
        assert spec.thresholds == []
        assert spec.confidence == 0.0
    
    def test_very_long_statement(self):
        """Test ContractSpec with very long statement."""
        long_statement = "A" * 10000
        spec = ContractSpec(statement=long_statement)
        
        assert spec.statement == long_statement
        assert len(spec.statement) == 10000
    
    def test_multiline_statement(self):
        """Test ContractSpec with multiline statement."""
        multiline = "Line 1\nLine 2\nLine 3"
        spec = ContractSpec(statement=multiline)
        
        assert spec.statement == multiline
        assert "\n" in spec.statement
    
    def test_unicode_statement(self):
        """Test ContractSpec with unicode statement."""
        unicode_statement = "Will Bitcoin reach $100,000? 🚀 Élection 2024 中文"
        spec = ContractSpec(statement=unicode_statement)
        
        assert "🚀" in spec.statement
        assert "Élection" in spec.statement
        assert "中文" in spec.statement
    
    def test_special_characters_statement(self):
        """Test ContractSpec with special characters."""
        special = "Will $BTC reach $100,000? (Yes/No) [2024] {test}"
        spec = ContractSpec(statement=special)
        
        assert "$BTC" in spec.statement
        assert "(Yes/No)" in spec.statement
        assert "[2024]" in spec.statement
    
    def test_confidence_boundary_values(self):
        """Test ContractSpec confidence boundary values."""
        # Minimum confidence
        spec1 = ContractSpec(statement="Test", confidence=0.0)
        assert spec1.confidence == 0.0
        
        # Maximum confidence
        spec2 = ContractSpec(statement="Test", confidence=1.0)
        assert spec2.confidence == 1.0
        
        # Middle confidence
        spec3 = ContractSpec(statement="Test", confidence=0.5)
        assert spec3.confidence == 0.5
    
    def test_many_entities(self):
        """Test ContractSpec with many entities."""
        entities = [
            EntitySpec(name=f"Entity {i}", entity_type="person")
            for i in range(50)
        ]
        spec = ContractSpec(statement="Test", entities=entities)
        
        assert len(spec.entities) == 50
        assert all(isinstance(e, EntitySpec) for e in spec.entities)
    
    def test_many_thresholds(self):
        """Test ContractSpec with many thresholds."""
        thresholds = [
            ThresholdSpec(value=float(i), unit="dollars")
            for i in range(50)
        ]
        spec = ContractSpec(statement="Test", thresholds=thresholds)
        
        assert len(spec.thresholds) == 50
        assert all(isinstance(t, ThresholdSpec) for t in spec.thresholds)
    
    def test_empty_lists(self):
        """Test ContractSpec with empty lists."""
        spec = ContractSpec(
            statement="Test",
            entities=[],
            thresholds=[],
            outcome_labels=[],
        )
        
        assert spec.entities == []
        assert spec.thresholds == []
        assert spec.outcome_labels == []
    
    def test_very_long_resolution_criteria(self):
        """Test ContractSpec with very long resolution criteria."""
        long_criteria = "A" * 5000
        spec = ContractSpec(
            statement="Test",
            resolution_criteria=long_criteria,
        )
        
        assert spec.resolution_criteria == long_criteria
        assert len(spec.resolution_criteria) == 5000


class TestDateSpecEdgeCases:
    """Tests for edge cases in DateSpec."""
    
    def test_far_future_date(self):
        """Test DateSpec with far future date."""
        future_date = datetime(2099, 12, 31, 23, 59, 59)
        spec = DateSpec(date=future_date, is_deadline=True)
        
        assert spec.date == future_date
        assert spec.is_deadline is True
    
    def test_far_past_date(self):
        """Test DateSpec with far past date."""
        past_date = datetime(1900, 1, 1, 0, 0, 0)
        spec = DateSpec(date=past_date, is_deadline=False)
        
        assert spec.date == past_date
        assert spec.is_deadline is False
    
    def test_date_with_timezone(self):
        """Test DateSpec with timezone."""
        date = datetime(2024, 12, 31)
        spec = DateSpec(date=date, timezone="America/New_York")
        
        assert spec.date == date
        assert spec.timezone == "America/New_York"
    
    def test_date_with_unicode_timezone(self):
        """Test DateSpec with unicode timezone."""
        date = datetime(2024, 12, 31)
        spec = DateSpec(date=date, timezone="UTC+5:30")
        
        assert spec.timezone == "UTC+5:30"


class TestEntitySpecEdgeCases:
    """Tests for edge cases in EntitySpec."""
    
    def test_entity_with_many_aliases(self):
        """Test EntitySpec with many aliases."""
        aliases = [f"Alias {i}" for i in range(100)]
        entity = EntitySpec(
            name="Bitcoin",
            entity_type="other",
            aliases=aliases,
        )
        
        assert len(entity.aliases) == 100
        assert all(isinstance(a, str) for a in entity.aliases)
    
    def test_entity_with_unicode_name(self):
        """Test EntitySpec with unicode name."""
        entity = EntitySpec(name="比特币", entity_type="other")
        
        assert entity.name == "比特币"
        assert "比特币" in entity.name
    
    def test_entity_with_special_characters(self):
        """Test EntitySpec with special characters."""
        entity = EntitySpec(name="Bitcoin (BTC)", entity_type="other")
        
        assert "(BTC)" in entity.name
    
    def test_entity_validation_invalid_type(self):
        """Test EntitySpec validation with invalid type."""
        with pytest.raises(Exception):  # Pydantic validation error
            EntitySpec(name="Test", entity_type="invalid_type")


class TestThresholdSpecEdgeCases:
    """Tests for edge cases in ThresholdSpec."""
    
    def test_threshold_very_large_value(self):
        """Test ThresholdSpec with very large value."""
        threshold = ThresholdSpec(value=1e15, unit="dollars")
        
        assert threshold.value == 1e15
        assert threshold.unit == "dollars"
    
    def test_threshold_very_small_value(self):
        """Test ThresholdSpec with very small value."""
        threshold = ThresholdSpec(value=1e-10, unit="percentage")
        
        assert threshold.value == 1e-10
        assert threshold.unit == "percentage"
    
    def test_threshold_negative_value(self):
        """Test ThresholdSpec with negative value."""
        threshold = ThresholdSpec(value=-100.0, unit="dollars")
        
        assert threshold.value == -100.0
    
    def test_threshold_zero_value(self):
        """Test ThresholdSpec with zero value."""
        threshold = ThresholdSpec(value=0.0, unit="count")
        
        assert threshold.value == 0.0
    
    def test_threshold_unicode_unit(self):
        """Test ThresholdSpec with unicode unit."""
        threshold = ThresholdSpec(value=100.0, unit="元")  # Chinese yuan
        
        assert threshold.unit == "元"
    
    def test_threshold_all_comparison_operators(self):
        """Test ThresholdSpec with all comparison operators."""
        operators = [">=", "<=", "==", ">", "<"]
        
        for op in operators:
            threshold = ThresholdSpec(value=100.0, comparison=op)
            assert threshold.comparison == op
    
    def test_threshold_validation_invalid_operator(self):
        """Test ThresholdSpec validation with invalid operator."""
        # This should be validated at ContractSpec level
        # But let's test that invalid operator raises error
        with pytest.raises(Exception):  # Pydantic validation error
            ThresholdSpec(value=100.0, comparison="!=")


class TestContractSpecSerializationEdgeCases:
    """Tests for edge cases in ContractSpec serialization."""
    
    @pytest.mark.asyncio
    async def test_serialize_with_none_values(self):
        """Test serialization with None values."""
        spec = ContractSpec(
            statement="Test",
            resolution_criteria=None,
            data_source=None,
            extraction_notes=None,
        )
        
        json_str = await spec.to_json_async()
        assert isinstance(json_str, str)
        assert len(json_str) > 0
    
    @pytest.mark.asyncio
    async def test_serialize_with_empty_lists(self):
        """Test serialization with empty lists."""
        spec = ContractSpec(
            statement="Test",
            entities=[],
            thresholds=[],
            outcome_labels=[],
        )
        
        json_str = await spec.to_json_async()
        assert isinstance(json_str, str)
        # Should serialize empty lists
        assert "entities" in json_str or "[]" in json_str
    
    @pytest.mark.asyncio
    async def test_deserialize_with_missing_fields(self):
        """Test deserialization with missing optional fields."""
        json_str = '{"statement": "Test"}'
        
        spec = await ContractSpec.from_json_async(json_str)
        
        assert spec.statement == "Test"
        assert spec.entities == []
        assert spec.thresholds == []
        assert spec.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_round_trip_serialization(self):
        """Test round-trip serialization preserves data."""
        original = ContractSpec(
            statement="Will Bitcoin reach $100,000?",
            resolution_criteria="Test criteria",
            confidence=0.95,
            entities=[
                EntitySpec(name="Bitcoin", entity_type="other")
            ],
        )
        
        json_str = await original.to_json_async()
        restored = await ContractSpec.from_json_async(json_str)
        
        assert restored.statement == original.statement
        assert restored.resolution_criteria == original.resolution_criteria
        assert restored.confidence == original.confidence
        assert len(restored.entities) == len(original.entities)
        assert restored.entities[0].name == original.entities[0].name

