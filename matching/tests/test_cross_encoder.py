"""Unit tests for CrossEncoder with mocked models."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import torch

from matching.cross_encoder import CrossEncoder


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cross_encoder_initialization_defaults():
    """Test cross-encoder initialization with default parameters."""
    encoder = CrossEncoder()
    
    assert encoder.model_name == "microsoft/deberta-v3-large"
    assert encoder.batch_size == 8
    assert encoder.max_length == 512
    assert encoder.use_quantization is False
    assert encoder.entailment_threshold == 0.7
    assert encoder.neutral_threshold == 0.3
    assert not encoder._initialized


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cross_encoder_initialization_custom_params():
    """Test cross-encoder with custom parameters."""
    encoder = CrossEncoder(
        model_name="custom/model",
        batch_size=16,
        max_length=256,
        use_quantization=True,
        entailment_threshold=0.8,
        neutral_threshold=0.4
    )
    
    assert encoder.model_name == "custom/model"
    assert encoder.batch_size == 16
    assert encoder.max_length == 256
    assert encoder.entailment_threshold == 0.8
    assert encoder.neutral_threshold == 0.4


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cross_encoder_device_detection_cuda(monkeypatch):
    """Test CUDA device detection."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends, "mps", MagicMock())
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    
    encoder = CrossEncoder(device=None)
    assert encoder.device == "cuda"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cross_encoder_device_detection_mps(monkeypatch):
    """Test MPS device detection (M4 Mac)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", MagicMock())
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    
    encoder = CrossEncoder(device=None)
    assert encoder.device == "mps"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cross_encoder_device_detection_cpu(monkeypatch):
    """Test CPU fallback when no GPU available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", MagicMock())
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    
    encoder = CrossEncoder(device=None)
    assert encoder.device == "cpu"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cross_encoder_device_explicit():
    """Test explicit device override."""
    encoder = CrossEncoder(device="cpu")
    assert encoder.device == "cpu"
    
    encoder2 = CrossEncoder(device="mps")
    assert encoder2.device == "mps"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cross_encoder_lazy_initialization(mock_nli_pipeline):
    """Test lazy model loading pattern."""
    with patch('matching.cross_encoder.pipeline', return_value=mock_nli_pipeline):
        with patch('matching.cross_encoder.AutoTokenizer') as mock_tokenizer:
            with patch('matching.cross_encoder.AutoModelForSequenceClassification') as mock_model:
                mock_model.return_value.to.return_value = mock_model.return_value
                mock_model.return_value.eval = MagicMock()
                
                encoder = CrossEncoder()
                assert not encoder._initialized
                
                await encoder.initialize()
                assert encoder._initialized
                assert encoder._pipeline is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cross_encoder_singleton_initialization():
    """Test that initialization is idempotent (singleton pattern)."""
    encoder = CrossEncoder()
    
    # Mock pipeline directly on instance (works in executor threads)
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [
        {"label": "ENTAILMENT", "score": 0.9},
        {"label": "NEUTRAL", "score": 0.05},
        {"label": "CONTRADICTION", "score": 0.05}
    ]
    
    # Set pipeline and mark as initialized to test singleton behavior
    encoder._pipeline = mock_pipeline
    encoder._initialized = True
    
    # Initialize multiple times - should not reload
    await encoder.initialize()
    await encoder.initialize()
    await encoder.initialize()
    
    # Verify pipeline wasn't recreated (singleton pattern)
    assert encoder._initialized
    assert encoder._pipeline is mock_pipeline


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cross_encoder_model_loading_failure():
    """Test error handling on model loading failure."""
    encoder = CrossEncoder()
    
    # Mock _load_model_sync to raise an error (tests error handling directly)
    def failing_load():
        raise Exception("Model not found")
    
    encoder._load_model_sync = failing_load
    
    with pytest.raises(Exception, match="Model not found"):
        await encoder.initialize()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_score_equivalence_async_single_pair(mock_nli_pipeline):
    """Test single pair scoring."""
    with patch('matching.cross_encoder.pipeline', return_value=mock_nli_pipeline):
        with patch('matching.cross_encoder.AutoTokenizer'):
            with patch('matching.cross_encoder.AutoModelForSequenceClassification') as mock_model:
                mock_model.return_value.to.return_value = mock_model.return_value
                mock_model.return_value.eval = MagicMock()
                
                encoder = CrossEncoder()
                await encoder.initialize()
                
                # Configure mock to return specific scores (flat list format)
                mock_nli_pipeline.return_value = [
                    {"label": "ENTAILMENT", "score": 0.9},
                    {"label": "NEUTRAL", "score": 0.05},
                    {"label": "CONTRADICTION", "score": 0.05}
                ]
                
                scores = await encoder.score_equivalence_async("Text 1", "Text 2")
                
                assert "entailment" in scores
                assert "neutral" in scores
                assert "contradiction" in scores
                assert scores["entailment"] == 0.9
                assert scores["neutral"] == 0.05
                assert scores["contradiction"] == 0.05


@pytest.mark.unit
@pytest.mark.asyncio
async def test_score_equivalence_async_normalized_scores(mock_nli_pipeline):
    """Test that scores are normalized to sum to 1.0."""
    with patch('matching.cross_encoder.pipeline', return_value=mock_nli_pipeline):
        with patch('matching.cross_encoder.AutoTokenizer'):
            with patch('matching.cross_encoder.AutoModelForSequenceClassification') as mock_model:
                mock_model.return_value.to.return_value = mock_model.return_value
                mock_model.return_value.eval = MagicMock()
                
                encoder = CrossEncoder()
                await encoder.initialize()
                
                # Mock scores that don't sum to 1.0 (should be normalized)
                mock_nli_pipeline.return_value = [
                    {"label": "ENTAILMENT", "score": 0.8},
                    {"label": "NEUTRAL", "score": 0.15},
                    {"label": "CONTRADICTION", "score": 0.1}
                ]
                
                scores = await encoder.score_equivalence_async("Text 1", "Text 2")
                
                total = sum(scores.values())
                assert abs(total - 1.0) < 1e-6  # Should sum to 1.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_score_batch_async_multiple_pairs(mock_nli_pipeline):
    """Test batch processing of multiple pairs."""
    with patch('matching.cross_encoder.pipeline', return_value=mock_nli_pipeline):
        with patch('matching.cross_encoder.AutoTokenizer'):
            with patch('matching.cross_encoder.AutoModelForSequenceClassification') as mock_model:
                mock_model.return_value.to.return_value = mock_model.return_value
                mock_model.return_value.eval = MagicMock()
                
                encoder = CrossEncoder(batch_size=2)
                await encoder.initialize()
                
                # Mock batch results using side_effect for multiple batch calls
                # Batch 1: 2 pairs, Batch 2: 1 pair
                # For batch input, pipeline returns list of lists (one per input)
                mock_nli_pipeline.side_effect = [
                    [  # First batch call (2 pairs)
                        [{"label": "ENTAILMENT", "score": 0.9}, {"label": "NEUTRAL", "score": 0.05}, {"label": "CONTRADICTION", "score": 0.05}],
                        [{"label": "ENTAILMENT", "score": 0.7}, {"label": "NEUTRAL", "score": 0.2}, {"label": "CONTRADICTION", "score": 0.1}]
                    ],
                    [  # Second batch call (1 pair)
                        [{"label": "ENTAILMENT", "score": 0.5}, {"label": "NEUTRAL", "score": 0.3}, {"label": "CONTRADICTION", "score": 0.2}]
                    ]
                ]
                
                pairs = [("Text 1", "Text 2"), ("Text 3", "Text 4"), ("Text 5", "Text 6")]
                scores = await encoder.score_batch_async(pairs)
                
                assert len(scores) == 3
                assert all("entailment" in s for s in scores)
                assert all("neutral" in s for s in scores)
                assert all("contradiction" in s for s in scores)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_score_batch_async_empty_list(mock_nli_pipeline):
    """Test batch processing with empty list."""
    with patch('matching.cross_encoder.pipeline', return_value=mock_nli_pipeline):
        with patch('matching.cross_encoder.AutoTokenizer'):
            with patch('matching.cross_encoder.AutoModelForSequenceClassification') as mock_model:
                mock_model.return_value.to.return_value = mock_model.return_value
                mock_model.return_value.eval = MagicMock()
                
                encoder = CrossEncoder()
                await encoder.initialize()
                
                scores = await encoder.score_batch_async([])
                assert scores == []


@pytest.mark.unit
def test_map_nli_to_confidence_full_match(sample_nli_scores):
    """Test confidence mapping for full match (high entailment)."""
    encoder = CrossEncoder(entailment_threshold=0.7)
    
    confidence, match_type = encoder.map_nli_to_confidence(sample_nli_scores["high_entailment"])
    
    assert match_type == "full_match"
    assert confidence >= 0.7
    assert 0.0 <= confidence <= 1.0


@pytest.mark.unit
def test_map_nli_to_confidence_partial_match(sample_nli_scores):
    """Test confidence mapping for partial match (neutral/moderate)."""
    encoder = CrossEncoder(entailment_threshold=0.7, neutral_threshold=0.3)
    
    confidence, match_type = encoder.map_nli_to_confidence(sample_nli_scores["neutral"])
    
    assert match_type == "partial_match"
    assert 0.0 <= confidence <= 1.0


@pytest.mark.unit
def test_map_nli_to_confidence_no_match(sample_nli_scores):
    """Test confidence mapping for no match (contradiction)."""
    encoder = CrossEncoder(entailment_threshold=0.7)
    
    confidence, match_type = encoder.map_nli_to_confidence(sample_nli_scores["contradiction"])
    
    assert match_type == "no_match"
    assert 0.0 <= confidence <= 1.0


@pytest.mark.unit
def test_map_nli_to_confidence_threshold_edge_cases():
    """Test confidence mapping at threshold boundaries."""
    encoder = CrossEncoder(entailment_threshold=0.7, neutral_threshold=0.3)
    
    # Exactly at entailment threshold
    scores = {"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1}
    confidence, match_type = encoder.map_nli_to_confidence(scores)
    assert match_type == "full_match"
    
    # Just below entailment threshold
    scores = {"entailment": 0.69, "neutral": 0.2, "contradiction": 0.11}
    confidence, match_type = encoder.map_nli_to_confidence(scores)
    assert match_type in ["partial_match", "no_match"]


@pytest.mark.unit
def test_map_nli_to_confidence_score_clamping():
    """Test that confidence scores are clamped to [0.0, 1.0]."""
    encoder = CrossEncoder()
    
    # Scores that would produce out-of-range confidence
    scores = {"entailment": 1.1, "neutral": -0.1, "contradiction": 0.0}
    confidence, _ = encoder.map_nli_to_confidence(scores)
    
    assert 0.0 <= confidence <= 1.0


@pytest.mark.unit
def test_extract_primary_event_standard_format(sample_canonical_texts):
    """Test primary event extraction from standard canonical text."""
    encoder = CrossEncoder()
    
    primary = encoder.extract_primary_event(sample_canonical_texts["standard"])
    
    assert "Bitcoin" in primary
    assert "$100,000" in primary
    assert "December 31, 2025" in primary
    assert "Market Statement:" not in primary  # Should not include section header


@pytest.mark.unit
def test_extract_primary_event_missing_section(sample_canonical_texts):
    """Test primary event extraction when Market Statement section is missing."""
    encoder = CrossEncoder()
    
    # Use minimal text without explicit section
    primary = encoder.extract_primary_event(sample_canonical_texts["minimal"])
    
    assert primary is not None
    assert len(primary) > 0


@pytest.mark.unit
def test_extract_secondary_clauses_standard(sample_canonical_texts):
    """Test secondary clause extraction from standard text."""
    encoder = CrossEncoder()
    
    clauses = encoder.extract_secondary_clauses(sample_canonical_texts["standard"])
    
    assert len(clauses) > 0
    assert all(len(clause) > 10 for clause in clauses)  # Substantial clauses
    assert any("Bitcoin" in clause or "BTC" in clause for clause in clauses)


@pytest.mark.unit
def test_extract_secondary_clauses_empty(sample_canonical_texts):
    """Test secondary clause extraction when no clauses exist."""
    encoder = CrossEncoder()
    
    clauses = encoder.extract_secondary_clauses(sample_canonical_texts["no_clauses"])
    
    assert clauses == []


@pytest.mark.unit
def test_extract_secondary_clauses_minimal(sample_canonical_texts):
    """Test secondary clause extraction from minimal text."""
    encoder = CrossEncoder()
    
    clauses = encoder.extract_secondary_clauses(sample_canonical_texts["minimal"])
    
    # Should extract "Based on closing price" or similar
    assert isinstance(clauses, list)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_score_secondary_clauses_async_matching(mock_nli_pipeline):
    """Test secondary clause scoring with matching clauses."""
    with patch('matching.cross_encoder.pipeline', return_value=mock_nli_pipeline):
        with patch('matching.cross_encoder.AutoTokenizer'):
            with patch('matching.cross_encoder.AutoModelForSequenceClassification') as mock_model:
                mock_model.return_value.to.return_value = mock_model.return_value
                mock_model.return_value.eval = MagicMock()
                
                encoder = CrossEncoder()
                await encoder.initialize()
                
                # Mock high entailment for matching clauses (flat list format)
                mock_nli_pipeline.return_value = [
                    {"label": "ENTAILMENT", "score": 0.9},
                    {"label": "NEUTRAL", "score": 0.05},
                    {"label": "CONTRADICTION", "score": 0.05}
                ]
                
                query_clauses = ["Clause 1", "Clause 2"]
                candidate_clauses = ["Clause 1", "Clause 2"]
                
                score = await encoder.score_secondary_clauses_async(query_clauses, candidate_clauses)
                
                assert 0.0 <= score <= 1.0
                assert score > 0.5  # Should be high for matching clauses


@pytest.mark.unit
@pytest.mark.asyncio
async def test_score_secondary_clauses_async_empty(mock_nli_pipeline):
    """Test secondary clause scoring with empty clauses."""
    with patch('matching.cross_encoder.pipeline', return_value=mock_nli_pipeline):
        with patch('matching.cross_encoder.AutoTokenizer'):
            with patch('matching.cross_encoder.AutoModelForSequenceClassification') as mock_model:
                mock_model.return_value.to.return_value = mock_model.return_value
                mock_model.return_value.eval = MagicMock()
                
                encoder = CrossEncoder()
                await encoder.initialize()
                
                # Empty clauses should return neutral score
                score = await encoder.score_secondary_clauses_async([], [])
                assert score == 0.5
                
                score2 = await encoder.score_secondary_clauses_async(["Clause 1"], [])
                assert score2 == 0.5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_score_equivalence_async_auto_initialization(mock_nli_pipeline):
    """Test that scoring auto-initializes model if not initialized."""
    with patch('matching.cross_encoder.pipeline', return_value=mock_nli_pipeline):
        with patch('matching.cross_encoder.AutoTokenizer'):
            with patch('matching.cross_encoder.AutoModelForSequenceClassification') as mock_model:
                mock_model.return_value.to.return_value = mock_model.return_value
                mock_model.return_value.eval = MagicMock()
                
                encoder = CrossEncoder()
                assert not encoder._initialized
                
                # Score without explicit initialization (flat list format)
                mock_nli_pipeline.return_value = [
                    {"label": "ENTAILMENT", "score": 0.9},
                    {"label": "NEUTRAL", "score": 0.05},
                    {"label": "CONTRADICTION", "score": 0.05}
                ]
                
                scores = await encoder.score_equivalence_async("Text 1", "Text 2")
                
                # Should have auto-initialized
                assert encoder._initialized
                assert scores is not None

