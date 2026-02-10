"""Unit tests for OutcomeMapper."""

import pytest

from matching.comparators.outcome_mapper import OutcomeMapper


@pytest.mark.unit
@pytest.mark.asyncio
async def test_outcome_mapper_initialization():
    """Test OutcomeMapper initialization."""
    mapper = OutcomeMapper()
    await mapper.initialize()
    
    assert mapper._initialized is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_map_binary_fast_path_yes_no():
    """Test binary fast path with Yes/No."""
    mapper = OutcomeMapper()
    await mapper.initialize()
    
    outcomes_a = ["Yes", "No"]
    outcomes_b = ["YES", "NO"]
    
    mapping = await mapper.map_binary_fast_path(outcomes_a, outcomes_b)
    
    assert "Yes" in mapping
    assert "No" in mapping
    assert mapping["Yes"] == "YES"
    assert mapping["No"] == "NO"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_map_binary_fast_path_case_insensitive():
    """Test binary fast path with case variations."""
    mapper = OutcomeMapper()
    await mapper.initialize()
    
    outcomes_a = ["yes", "no"]
    outcomes_b = ["YES", "NO"]
    
    mapping = await mapper.map_binary_fast_path(outcomes_a, outcomes_b)
    
    assert len(mapping) == 2
    assert outcomes_a[0] in mapping
    assert outcomes_a[1] in mapping


@pytest.mark.unit
@pytest.mark.asyncio
async def test_map_binary_fast_path_alternatives():
    """Test binary fast path with alternative labels."""
    mapper = OutcomeMapper()
    await mapper.initialize()
    
    outcomes_a = ["Y", "N"]
    outcomes_b = ["True", "False"]
    
    mapping = await mapper.map_binary_fast_path(outcomes_a, outcomes_b)
    
    # Should map Y -> True, N -> False
    assert len(mapping) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_map_outcomes_binary():
    """Test outcome mapping for binary markets."""
    mapper = OutcomeMapper()
    await mapper.initialize()
    
    outcomes_a = ["Yes", "No"]
    outcomes_b = ["YES", "NO"]
    
    mapping = await mapper.map_outcomes(outcomes_a, outcomes_b)
    
    assert len(mapping) == 2
    assert mapping["Yes"] == "YES"
    assert mapping["No"] == "NO"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_map_outcomes_multi_outcome():
    """Test outcome mapping for multi-outcome markets."""
    mapper = OutcomeMapper()
    await mapper.initialize()
    
    outcomes_a = ["Bitcoin", "Ethereum", "Other"]
    outcomes_b = ["BTC", "ETH", "Other"]
    
    mapping = await mapper.map_outcomes(outcomes_a, outcomes_b)
    
    # Should use order-based mapping for multi-outcome
    assert len(mapping) == 3
    assert mapping["Bitcoin"] == "BTC"
    assert mapping["Ethereum"] == "ETH"
    assert mapping["Other"] == "Other"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_map_outcomes_empty():
    """Test outcome mapping with empty lists."""
    mapper = OutcomeMapper()
    await mapper.initialize()
    
    mapping = await mapper.map_outcomes([], [])
    
    assert mapping == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_map_outcomes_different_lengths():
    """Test outcome mapping with different list lengths."""
    mapper = OutcomeMapper()
    await mapper.initialize()
    
    outcomes_a = ["Yes", "No", "Maybe"]
    outcomes_b = ["YES", "NO"]
    
    mapping = await mapper.map_outcomes(outcomes_a, outcomes_b)
    
    # Should map up to min length
    assert len(mapping) == 2

