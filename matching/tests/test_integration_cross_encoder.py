"""Integration tests for CrossEncoder with real models."""

import pytest
import asyncio
import torch
from matching.cross_encoder import CrossEncoder
from matching.reranker import CandidateReranker


@pytest.fixture(scope="session")
def real_cross_encoder():
    """Create real CrossEncoder instance (session-scoped for reuse)."""
    try:
        encoder = CrossEncoder()
        return encoder
    except Exception as e:
        pytest.skip(f"Failed to create CrossEncoder: {e}")


@pytest.fixture(scope="session")
@pytest.mark.asyncio
async def initialized_cross_encoder(real_cross_encoder):
    """Initialize real CrossEncoder (session-scoped)."""
    try:
        await real_cross_encoder.initialize()
        return real_cross_encoder
    except Exception as e:
        pytest.skip(f"Failed to initialize CrossEncoder: {e}")


@pytest.mark.integration
@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.asyncio
async def test_integration_real_model_loading(real_cross_encoder):
    """Test that real model loads successfully."""
    assert not real_cross_encoder._initialized
    
    await real_cross_encoder.initialize()
    
    assert real_cross_encoder._initialized
    assert real_cross_encoder._pipeline is not None
    assert real_cross_encoder._model is not None
    assert real_cross_encoder._tokenizer is not None


@pytest.mark.integration
@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.asyncio
async def test_integration_real_model_scoring(initialized_cross_encoder):
    """Test real NLI scoring with actual model."""
    text1 = "Will Bitcoin reach $100,000 by December 31, 2025?"
    text2 = "Bitcoin will hit $100k before the end of 2025."
    
    scores = await initialized_cross_encoder.score_equivalence_async(text1, text2)
    
    # Verify scores structure
    assert "entailment" in scores
    assert "neutral" in scores
    assert "contradiction" in scores
    
    # Verify scores are valid
    assert 0.0 <= scores["entailment"] <= 1.0
    assert 0.0 <= scores["neutral"] <= 1.0
    assert 0.0 <= scores["contradiction"] <= 1.0
    
    # Verify scores sum to approximately 1.0 (allowing for floating point)
    total = sum(scores.values())
    assert abs(total - 1.0) < 0.01


@pytest.mark.integration
@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.asyncio
async def test_integration_real_model_batch_processing(initialized_cross_encoder):
    """Test real batch processing."""
    pairs = [
        ("Bitcoin will reach $100k", "BTC hits $100,000"),
        ("S&P 500 above 6000", "S&P 500 exceeds 6000"),
        ("Rain tomorrow", "Sunny weather expected"),
    ]
    
    scores = await initialized_cross_encoder.score_batch_async(pairs)
    
    assert len(scores) == 3
    assert all("entailment" in s for s in scores)
    assert all("neutral" in s for s in scores)
    assert all("contradiction" in s for s in scores)
    
    # First pair should have high entailment (similar)
    assert scores[0]["entailment"] > 0.5
    
    # Third pair should have lower entailment (different)
    assert scores[2]["entailment"] < scores[0]["entailment"]


@pytest.mark.integration
@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.asyncio
async def test_integration_real_model_device_mps():
    """Test MPS device detection and usage (M4 Mac)."""
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")
    
    encoder = CrossEncoder(device="mps")
    await encoder.initialize()
    
    assert encoder.device == "mps"
    
    # Test scoring works on MPS
    scores = await encoder.score_equivalence_async("Text 1", "Text 2")
    assert scores is not None


@pytest.mark.integration
@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.asyncio
async def test_integration_real_model_device_cpu():
    """Test CPU device fallback."""
    encoder = CrossEncoder(device="cpu")
    await encoder.initialize()
    
    assert encoder.device == "cpu"
    
    # Test scoring works on CPU
    scores = await encoder.score_equivalence_async("Text 1", "Text 2")
    assert scores is not None


@pytest.mark.integration
@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.asyncio
async def test_integration_text_extraction_accuracy(initialized_cross_encoder, sample_canonical_texts):
    """Test that text extraction works correctly with real canonical text."""
    encoder = initialized_cross_encoder
    
    # Test primary event extraction
    primary = encoder.extract_primary_event(sample_canonical_texts["standard"])
    assert "Bitcoin" in primary
    assert "$100,000" in primary
    assert "Market Statement:" not in primary
    
    # Test secondary clause extraction
    clauses = encoder.extract_secondary_clauses(sample_canonical_texts["standard"])
    assert len(clauses) > 0
    assert any("Bitcoin" in clause or "BTC" in clause for clause in clauses)


@pytest.mark.integration
@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.asyncio
async def test_integration_confidence_mapping(initialized_cross_encoder):
    """Test confidence mapping with real NLI scores."""
    encoder = initialized_cross_encoder
    
    # High entailment should map to full_match
    high_entailment = {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}
    confidence, match_type = encoder.map_nli_to_confidence(high_entailment)
    assert match_type == "full_match"
    assert confidence >= encoder.entailment_threshold
    
    # Neutral should map to partial_match
    neutral = {"entailment": 0.3, "neutral": 0.6, "contradiction": 0.1}
    confidence, match_type = encoder.map_nli_to_confidence(neutral)
    assert match_type == "partial_match"
    
    # Contradiction should map to no_match
    contradiction = {"entailment": 0.1, "neutral": 0.2, "contradiction": 0.7}
    confidence, match_type = encoder.map_nli_to_confidence(contradiction)
    assert match_type == "no_match"


@pytest.mark.integration
@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.asyncio
async def test_integration_secondary_clause_scoring(initialized_cross_encoder):
    """Test secondary clause scoring with real model."""
    encoder = initialized_cross_encoder
    
    query_clauses = [
        "This market resolves based on CoinMarketCap data.",
        "Price must be sustained for at least 1 minute."
    ]
    
    candidate_clauses = [
        "Resolution uses CoinMarketCap as data source.",
        "Price must hold for 60 seconds minimum."
    ]
    
    score = await encoder.score_secondary_clauses_async(query_clauses, candidate_clauses)
    
    assert 0.0 <= score <= 1.0
    # Should be relatively high for similar clauses
    assert score > 0.5


@pytest.mark.integration
@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.asyncio
async def test_integration_reranker_with_real_model(initialized_cross_encoder, sample_canonical_event, sample_candidate_matches):
    """Test CandidateReranker with real CrossEncoder."""
    reranker = CandidateReranker(initialized_cross_encoder, score_threshold=0.5)
    await reranker.initialize()
    
    # Use first 3 candidates
    results = await reranker.rerank_async(sample_canonical_event, sample_candidate_matches[:3])
    
    # Should return some results (depending on actual similarity)
    assert isinstance(results, list)
    assert all(isinstance(r, type(results[0])) for r in results) if results else True
    
    # If results exist, verify structure
    if results:
        for result in results:
            assert result.cross_encoder_score >= 0.5
            assert result.match_type in ["full_match", "partial_match", "no_match"]
            assert result.primary_event_score is not None

