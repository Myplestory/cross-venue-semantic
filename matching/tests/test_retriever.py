"""Unit tests for CandidateRetriever with mocks."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from matching.retriever import CandidateRetriever
from matching.types import CandidateMatch
from embedding.index import QdrantIndex
from discovery.types import VenueType


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retriever_initialization(mock_qdrant_index):
    """Test retriever initialization with default parameters."""
    retriever = CandidateRetriever(mock_qdrant_index)
    
    assert retriever.index == mock_qdrant_index
    assert retriever.default_top_k == 20
    assert retriever.default_score_threshold == 0.5
    assert retriever.max_retries == 3
    assert retriever.retry_backoff_factor == 2.0
    assert retriever.query_timeout == 5.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retriever_custom_parameters(mock_qdrant_index):
    """Test retriever with custom parameters."""
    retriever = CandidateRetriever(
        mock_qdrant_index,
        default_top_k=20,
        default_score_threshold=0.8,
        max_retries=5,
        retry_backoff_factor=1.5,
        query_timeout=10.0
    )
    
    assert retriever.default_top_k == 20
    assert retriever.default_score_threshold == 0.8
    assert retriever.max_retries == 5
    assert retriever.retry_backoff_factor == 1.5
    assert retriever.query_timeout == 10.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retriever_initialize(mock_qdrant_index):
    """Test that initialize calls index.initialize()."""
    retriever = CandidateRetriever(mock_qdrant_index)
    
    await retriever.initialize()
    
    mock_qdrant_index.initialize.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_candidates_success(mock_qdrant_index, sample_embedded_event, mock_qdrant_search_results):
    """Test successful candidate retrieval."""
    mock_qdrant_index.search_async = AsyncMock(return_value=mock_qdrant_search_results)
    
    retriever = CandidateRetriever(mock_qdrant_index)
    await retriever.initialize()
    
    candidates = await retriever.retrieve_candidates(sample_embedded_event)
    
    assert len(candidates) == 2
    assert all(isinstance(c, CandidateMatch) for c in candidates)
    assert candidates[0].similarity_score == 0.95
    assert candidates[1].similarity_score == 0.85
    assert candidates[0].canonical_event.event.venue == VenueType.POLYMARKET
    assert candidates[1].canonical_event.event.venue == VenueType.KALSHI


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_candidates_default_params(mock_qdrant_index, sample_embedded_event):
    """Test that default parameters are used when not specified."""
    mock_qdrant_index.search_async = AsyncMock(return_value=[])
    
    retriever = CandidateRetriever(mock_qdrant_index, default_top_k=15, default_score_threshold=0.75)
    await retriever.initialize()
    
    await retriever.retrieve_candidates(sample_embedded_event)
    
    call_args = mock_qdrant_index.search_async.call_args
    assert call_args[1]["top_k"] == 15
    assert call_args[1]["score_threshold"] == 0.75


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_candidates_custom_params(mock_qdrant_index, sample_embedded_event):
    """Test that custom parameters override defaults."""
    mock_qdrant_index.search_async = AsyncMock(return_value=[])
    
    retriever = CandidateRetriever(mock_qdrant_index, default_top_k=10, default_score_threshold=0.7)
    await retriever.initialize()
    
    await retriever.retrieve_candidates(
        sample_embedded_event,
        top_k=25,
        score_threshold=0.85
    )
    
    call_args = mock_qdrant_index.search_async.call_args
    assert call_args[1]["top_k"] == 25
    assert call_args[1]["score_threshold"] == 0.85


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_candidates_cross_venue(mock_qdrant_index, sample_embedded_event):
    """Test cross-venue matching excludes source venue."""
    mock_qdrant_index.search_async = AsyncMock(return_value=[])
    
    retriever = CandidateRetriever(mock_qdrant_index)
    await retriever.initialize()
    
    await retriever.retrieve_candidates(
        sample_embedded_event,
        exclude_venue=VenueType.KALSHI
    )
    
    call_args = mock_qdrant_index.search_async.call_args
    assert call_args[1]["exclude_venue"] == "kalshi"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_candidates_self_exclusion(mock_qdrant_index, sample_embedded_event):
    """Test that own identity_hash is excluded."""
    mock_qdrant_index.search_async = AsyncMock(return_value=[])
    
    retriever = CandidateRetriever(mock_qdrant_index)
    await retriever.initialize()
    
    await retriever.retrieve_candidates(sample_embedded_event)
    
    call_args = mock_qdrant_index.search_async.call_args
    assert call_args[1]["exclude_identity_hash"] == sample_embedded_event.canonical_event.identity_hash


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_candidates_no_exclude_venue(mock_qdrant_index, sample_embedded_event):
    """Test that no venue filter is applied when exclude_venue is None."""
    mock_qdrant_index.search_async = AsyncMock(return_value=[])
    
    retriever = CandidateRetriever(mock_qdrant_index)
    await retriever.initialize()
    
    await retriever.retrieve_candidates(sample_embedded_event, exclude_venue=None)
    
    call_args = mock_qdrant_index.search_async.call_args
    assert call_args[1]["exclude_venue"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_candidates_empty_results(mock_qdrant_index, sample_embedded_event):
    """Test handling of empty results."""
    mock_qdrant_index.search_async = AsyncMock(return_value=[])
    
    retriever = CandidateRetriever(mock_qdrant_index)
    await retriever.initialize()
    
    candidates = await retriever.retrieve_candidates(sample_embedded_event)
    
    assert candidates == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_candidates_score_sorting(mock_qdrant_index, sample_embedded_event):
    """Test that results are sorted by score (highest first)."""
    results = [
        {"id": "1", "score": 0.7, "payload": {"venue": "kalshi", "venue_market_id": "k1", "identity_hash": "h1", "canonical_text": "Text 1"}},
        {"id": "2", "score": 0.9, "payload": {"venue": "kalshi", "venue_market_id": "k2", "identity_hash": "h2", "canonical_text": "Text 2"}},
        {"id": "3", "score": 0.8, "payload": {"venue": "kalshi", "venue_market_id": "k3", "identity_hash": "h3", "canonical_text": "Text 3"}},
    ]
    mock_qdrant_index.search_async = AsyncMock(return_value=results)
    
    retriever = CandidateRetriever(mock_qdrant_index)
    await retriever.initialize()
    
    candidates = await retriever.retrieve_candidates(sample_embedded_event)
    
    assert len(candidates) == 3
    assert candidates[0].similarity_score == 0.9
    assert candidates[1].similarity_score == 0.8
    assert candidates[2].similarity_score == 0.7


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_candidates_timeout_retry(mock_qdrant_index, sample_embedded_event):
    """Test retry logic on timeout."""
    mock_qdrant_index.search_async = AsyncMock(side_effect=asyncio.TimeoutError())
    
    retriever = CandidateRetriever(mock_qdrant_index, max_retries=2, retry_backoff_factor=0.1)
    await retriever.initialize()
    
    with pytest.raises(RuntimeError, match="Retrieval timeout"):
        await retriever.retrieve_candidates(sample_embedded_event)
    
    assert mock_qdrant_index.search_async.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_candidates_timeout_exhausted(mock_qdrant_index, sample_embedded_event):
    """Test that timeout raises RuntimeError after max retries."""
    mock_qdrant_index.search_async = AsyncMock(side_effect=asyncio.TimeoutError())
    
    retriever = CandidateRetriever(mock_qdrant_index, max_retries=3, retry_backoff_factor=0.01)
    await retriever.initialize()
    
    with pytest.raises(RuntimeError, match="Retrieval timeout"):
        await retriever.retrieve_candidates(sample_embedded_event)
    
    assert mock_qdrant_index.search_async.call_count == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_candidates_network_error_retry(mock_qdrant_index, sample_embedded_event):
    """Test retry logic on network errors."""
    mock_qdrant_index.search_async = AsyncMock(side_effect=ConnectionError("Network error"))
    
    retriever = CandidateRetriever(mock_qdrant_index, max_retries=2, retry_backoff_factor=0.1)
    await retriever.initialize()
    
    with pytest.raises(RuntimeError, match="Retrieval failed"):
        await retriever.retrieve_candidates(sample_embedded_event)
    
    assert mock_qdrant_index.search_async.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_candidates_network_error_exhausted(mock_qdrant_index, sample_embedded_event):
    """Test that network errors raise RuntimeError after max retries."""
    mock_qdrant_index.search_async = AsyncMock(side_effect=Exception("Network error"))
    
    retriever = CandidateRetriever(mock_qdrant_index, max_retries=3, retry_backoff_factor=0.01)
    await retriever.initialize()
    
    with pytest.raises(RuntimeError, match="Retrieval failed"):
        await retriever.retrieve_candidates(sample_embedded_event)
    
    assert mock_qdrant_index.search_async.call_count == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_candidates_retry_backoff_timing(mock_qdrant_index, sample_embedded_event):
    """Test that exponential backoff timing is correct."""
    call_times = []
    
    async def mock_search_with_timing(*args, **kwargs):
        call_times.append(asyncio.get_event_loop().time())
        if len(call_times) < 3:
            raise asyncio.TimeoutError()
        return []
    
    mock_qdrant_index.search_async = AsyncMock(side_effect=mock_search_with_timing)
    
    retriever = CandidateRetriever(mock_qdrant_index, max_retries=3, retry_backoff_factor=2.0)
    await retriever.initialize()
    
    await retriever.retrieve_candidates(sample_embedded_event)
    
    if len(call_times) >= 3:
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        
        assert 0.9 <= delay1 <= 2.1
        assert 1.9 <= delay2 <= 4.1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_candidates_query_timeout(mock_qdrant_index, sample_embedded_event):
    """Test that query timeout is applied."""
    async def slow_search(*args, **kwargs):
        await asyncio.sleep(10)
        return []
    
    mock_qdrant_index.search_async = AsyncMock(side_effect=slow_search)
    
    retriever = CandidateRetriever(mock_qdrant_index, query_timeout=0.1)
    await retriever.initialize()
    
    with pytest.raises(RuntimeError, match="Retrieval timeout"):
        await retriever.retrieve_candidates(sample_embedded_event)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_format_results_success(mock_qdrant_index, mock_qdrant_search_results):
    """Test successful result formatting."""
    retriever = CandidateRetriever(mock_qdrant_index)
    
    candidates = retriever._format_results(mock_qdrant_search_results)
    
    assert len(candidates) == 2
    assert all(isinstance(c, CandidateMatch) for c in candidates)
    assert candidates[0].similarity_score == 0.95
    assert candidates[1].similarity_score == 0.85
    assert candidates[0].canonical_event.event.venue == VenueType.POLYMARKET
    assert candidates[1].canonical_event.event.venue == VenueType.KALSHI
    assert "qdrant_id" in candidates[0].retrieval_metadata


@pytest.mark.unit
@pytest.mark.asyncio
async def test_format_results_missing_venue(mock_qdrant_index):
    """Test that results with missing venue are skipped."""
    results = [
        {"id": "1", "score": 0.9, "payload": {}},
    ]
    
    retriever = CandidateRetriever(mock_qdrant_index)
    candidates = retriever._format_results(results)
    
    assert len(candidates) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_format_results_invalid_venue(mock_qdrant_index):
    """Test that results with invalid venue are skipped."""
    results = [
        {"id": "1", "score": 0.9, "payload": {"venue": "invalid"}},
    ]
    
    retriever = CandidateRetriever(mock_qdrant_index)
    candidates = retriever._format_results(results)
    
    assert len(candidates) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_format_results_missing_required_fields(mock_qdrant_index):
    """Test that results missing required fields are skipped."""
    results = [
        {"id": "1", "score": 0.9, "payload": {"venue": "kalshi"}},
        {"id": "2", "score": 0.8, "payload": {"venue": "kalshi", "venue_market_id": "k1"}},
        {"id": "3", "score": 0.7, "payload": {"venue": "kalshi", "venue_market_id": "k2", "identity_hash": "h1"}},
    ]
    
    retriever = CandidateRetriever(mock_qdrant_index)
    candidates = retriever._format_results(results)
    
    assert len(candidates) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_format_results_malformed_payload(mock_qdrant_index, mock_qdrant_search_results_malformed):
    """Test that malformed payloads are handled gracefully."""
    retriever = CandidateRetriever(mock_qdrant_index)
    candidates = retriever._format_results(mock_qdrant_search_results_malformed)
    
    assert len(candidates) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_format_results_error_in_formatting(mock_qdrant_index):
    """Test that individual formatting errors don't crash."""
    results = [
        {"id": "1", "score": 0.9, "payload": {"venue": "kalshi", "venue_market_id": "k1", "identity_hash": "h1", "canonical_text": "Valid"}},
        {"id": "2", "score": 0.8, "invalid": "structure"},
    ]
    
    retriever = CandidateRetriever(mock_qdrant_index)
    candidates = retriever._format_results(results)
    
    assert len(candidates) == 1
    assert candidates[0].similarity_score == 0.9


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_candidates_zero_top_k(mock_qdrant_index, sample_embedded_event):
    """Test handling of top_k=0."""
    mock_qdrant_index.search_async = AsyncMock(return_value=[])
    
    retriever = CandidateRetriever(mock_qdrant_index)
    await retriever.initialize()
    
    candidates = await retriever.retrieve_candidates(sample_embedded_event, top_k=0)
    
    assert candidates == []
    call_args = mock_qdrant_index.search_async.call_args
    assert call_args[1]["top_k"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_candidates_high_score_threshold(mock_qdrant_index, sample_embedded_event):
    """Test that high score threshold returns no results."""
    mock_qdrant_index.search_async = AsyncMock(return_value=[])
    
    retriever = CandidateRetriever(mock_qdrant_index)
    await retriever.initialize()
    
    candidates = await retriever.retrieve_candidates(
        sample_embedded_event,
        score_threshold=0.99
    )
    
    assert candidates == []
    call_args = mock_qdrant_index.search_async.call_args
    assert call_args[1]["score_threshold"] == 0.99

