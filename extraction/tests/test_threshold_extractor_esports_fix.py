"""Unit tests to verify esports fix doesn't break normal spread arb thresholds."""

import pytest
from extraction.parsers.threshold_extractor import ThresholdExtractor


@pytest.mark.unit
@pytest.mark.asyncio
async def test_legitimate_game_thresholds_still_work():
    """Verify legitimate 'game' thresholds are still extracted."""
    extractor = ThresholdExtractor()
    
    # Legitimate thresholds that should still work
    test_cases = [
        ("Will the team win over 3 games this season?", 1, "Should extract '3 games' threshold"),
        ("Will the team score above 5 games in the championship?", 1, "Should extract '5 games' threshold"),
        ("Will the team reach at least 2 games?", 1, "Should extract '2 games' threshold"),
        ("Will Bitcoin reach over $60k?", 1, "Should extract '$60k' threshold"),
        ("Will the price be above 50%?", 1, "Should extract '50%' threshold"),
        ("Will the team score over 100 points?", 1, "Should extract '100 points' threshold"),
    ]
    
    for text, expected_count, description in test_cases:
        thresholds = await extractor.extract_thresholds(text, None)
        assert len(thresholds) >= expected_count, f"{description}: {text}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_esports_game_identifiers_filtered():
    """Verify esports game identifiers are correctly filtered out."""
    extractor = ThresholdExtractor()
    
    # False positives that should be filtered
    test_cases = [
        ("LoL: KT Rolster Challengers vs T1 Academy - Game 3 Winner", 0, "Should filter 'Game 3' identifier"),
        ("Will T1 Academy win map 3 in the match?", 0, "Should filter 'map 3' if followed by winner context"),
        ("Game 1 Winner: Team A vs Team B", 0, "Should filter 'Game 1' identifier"),
        ("Map 2 Winner in the tournament", 0, "Should filter 'Map 2' identifier"),
        ("Match 3 Result: Team A wins", 0, "Should filter 'Match 3' identifier"),
    ]
    
    for text, expected_count, description in test_cases:
        thresholds = await extractor.extract_thresholds(text, None)
        assert len(thresholds) == expected_count, f"{description}: {text} - found {len(thresholds)} thresholds"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_edge_cases_game_thresholds():
    """Test edge cases where 'game' appears but should still be extracted."""
    extractor = ThresholdExtractor()
    
    # Edge cases that should still work
    test_cases = [
        ("Will the team win over 3 games to qualify?", 1, "Should extract '3 games' (no 'winner' keyword)"),
        ("Will the team score above 5 games in the playoffs?", 1, "Should extract '5 games' (no 'winner' keyword)"),
        ("Will the team reach at least 2 games this season?", 1, "Should extract '2 games' (no 'winner' keyword)"),
    ]
    
    for text, expected_count, description in test_cases:
        thresholds = await extractor.extract_thresholds(text, None)
        assert len(thresholds) >= expected_count, f"{description}: {text} - found {len(thresholds)} thresholds"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_currency_and_percentage_unaffected():
    """Verify currency and percentage thresholds are unaffected by the fix."""
    extractor = ThresholdExtractor()
    
    test_cases = [
        ("Will Bitcoin reach over $60k?", 1),
        ("Will the price be above 50%?", 1),
        ("Will the stock exceed $100,000?", 1),
        ("Will the rate be below 25%?", 1),
    ]
    
    for text, expected_count in test_cases:
        thresholds = await extractor.extract_thresholds(text, None)
        assert len(thresholds) >= expected_count, f"Failed for: {text}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_points_goals_yards_unaffected():
    """Verify other unit words (points, goals, yards) are unaffected."""
    extractor = ThresholdExtractor()
    
    test_cases = [
        ("Will the team score over 100 points?", 1),
        ("Will the team score above 5 goals?", 1),
        ("Will the player rush over 1000 yards?", 1),
        ("Will the team score at least 50 runs?", 1),
    ]
    
    for text, expected_count in test_cases:
        thresholds = await extractor.extract_thresholds(text, None)
        assert len(thresholds) >= expected_count, f"Failed for: {text}"

