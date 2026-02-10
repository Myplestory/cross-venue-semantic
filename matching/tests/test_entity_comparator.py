"""Unit tests for EntityComparator."""

import pytest
from unittest.mock import AsyncMock

from matching.comparators.entity_comparator import EntityComparator
from canonicalization.contract_spec import EntitySpec


@pytest.mark.unit
@pytest.mark.asyncio
async def test_entity_comparator_initialization():
    """Test EntityComparator initialization."""
    comparator = EntityComparator()
    await comparator.initialize()
    
    assert comparator._initialized is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_entities_both_empty():
    """Test comparing empty entity lists."""
    comparator = EntityComparator()
    await comparator.initialize()
    
    score, details = await comparator.compare_entities([], [])
    
    assert score == 1.0
    assert details["match"] is True
    assert details["matched_count"] == 0
    assert details["total_count"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_entities_one_empty():
    """Test comparing when one list is empty."""
    comparator = EntityComparator()
    await comparator.initialize()
    
    entities_a = [
        EntitySpec(name="Bitcoin", entity_type="other")
    ]
    
    score, details = await comparator.compare_entities(entities_a, [])
    
    assert score == 0.5
    assert details["match"] is False
    assert details["entities_a_count"] == 1
    assert details["entities_b_count"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_entities_exact_match():
    """Test exact entity name matching."""
    comparator = EntityComparator()
    await comparator.initialize()
    
    entities_a = [
        EntitySpec(name="Bitcoin", entity_type="other"),
        EntitySpec(name="Coinbase", entity_type="organization")
    ]
    entities_b = [
        EntitySpec(name="Bitcoin", entity_type="other"),
        EntitySpec(name="Coinbase", entity_type="organization")
    ]
    
    score, details = await comparator.compare_entities(entities_a, entities_b)
    
    assert score == 1.0
    assert details["match"] is True
    assert details["matched_count"] == 2
    assert details["total_count"] == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_entities_alias_match():
    """Test entity matching with aliases."""
    comparator = EntityComparator()
    await comparator.initialize()
    
    entities_a = [
        EntitySpec(name="Bitcoin", entity_type="other", aliases=["BTC"])
    ]
    entities_b = [
        EntitySpec(name="BTC", entity_type="other", aliases=["Bitcoin"])
    ]
    
    score, details = await comparator.compare_entities(entities_a, entities_b)
    
    assert score == 1.0
    assert details["match"] is True
    assert details["matched_count"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_entities_type_mismatch():
    """Test entity matching with type mismatch."""
    comparator = EntityComparator()
    await comparator.initialize()
    
    entities_a = [
        EntitySpec(name="Bitcoin", entity_type="other")
    ]
    entities_b = [
        EntitySpec(name="Bitcoin", entity_type="organization")
    ]
    
    score, details = await comparator.compare_entities(entities_a, entities_b)
    
    # Should match on name but penalize for type mismatch
    assert score < 1.0
    assert details["type_mismatches"] > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_entities_partial_match():
    """Test partial entity matching."""
    comparator = EntityComparator()
    await comparator.initialize()
    
    entities_a = [
        EntitySpec(name="Bitcoin", entity_type="other"),
        EntitySpec(name="Coinbase", entity_type="organization")
    ]
    entities_b = [
        EntitySpec(name="Bitcoin", entity_type="other")
    ]
    
    score, details = await comparator.compare_entities(entities_a, entities_b)
    
    assert 0.0 < score < 1.0
    assert details["matched_count"] == 1
    assert details["total_count"] == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_entities_no_match():
    """Test when no entities match."""
    comparator = EntityComparator()
    await comparator.initialize()
    
    entities_a = [
        EntitySpec(name="Bitcoin", entity_type="other")
    ]
    entities_b = [
        EntitySpec(name="Ethereum", entity_type="other")
    ]
    
    score, details = await comparator.compare_entities(entities_a, entities_b)
    
    assert score == 0.0
    assert details["match"] is False
    assert details["matched_count"] == 0

