"""Tests for QdrantIndex (with mocks + real Qdrant for integration)."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from qdrant_client.models import Distance, ScoredPoint

from embedding.index import QdrantIndex
from embedding.types import EmbeddedEvent


@pytest.mark.asyncio
async def test_collection_creation(mock_qdrant_client):
    """Test collection creation."""
    with patch('embedding.index.AsyncQdrantClient', return_value=mock_qdrant_client):
        index = QdrantIndex(
            url="http://localhost:6333",
            collection_name="test_collection",
            vector_size=2048
        )
        
        await index.initialize()
        
        mock_qdrant_client.create_collection.assert_called_once()
        assert index._initialized


@pytest.mark.asyncio
async def test_collection_already_exists(mock_qdrant_client):
    """Test handling when collection already exists."""
    # Mock existing collection
    existing_collection = MagicMock()
    existing_collection.name = "test_collection"
    mock_qdrant_client.get_collections = AsyncMock(
        return_value=MagicMock(collections=[existing_collection])
    )
    
    with patch('embedding.index.AsyncQdrantClient', return_value=mock_qdrant_client):
        index = QdrantIndex(collection_name="test_collection")
        await index.initialize()
        
        # Should not create collection
        mock_qdrant_client.create_collection.assert_not_called()
        assert index._initialized


@pytest.mark.asyncio
async def test_upsert_batch(mock_qdrant_client, sample_canonical_event, mock_embedding):
    """Test batch upsert."""
    with patch('embedding.index.AsyncQdrantClient', return_value=mock_qdrant_client):
        index = QdrantIndex()
        await index.initialize()
        
        embedded_events = [
            EmbeddedEvent(
                canonical_event=sample_canonical_event,
                embedding=mock_embedding,
                embedding_model="Qwen/Qwen3-Embedding-4B",
                embedding_dim=2048
            )
        ]
        
        await index.upsert_async(embedded_events)
        
        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args
        assert call_args[1]["collection_name"] == "market_embeddings"
        assert len(call_args[1]["points"]) == 1


@pytest.mark.asyncio
async def test_upsert_empty_list(mock_qdrant_client):
    """Test upsert with empty list."""
    with patch('embedding.index.AsyncQdrantClient', return_value=mock_qdrant_client):
        index = QdrantIndex()
        await index.initialize()
        
        await index.upsert_async([])
        
        # Should not call upsert with empty list
        mock_qdrant_client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_search_similarity(mock_qdrant_client, mock_embedding):
    """Test similarity search."""
    # Mock search results
    mock_result = MagicMock()
    mock_result.id = "test-id"
    mock_result.score = 0.95
    mock_result.payload = {"venue": "kalshi", "identity_hash": "xyz789"}
    
    mock_qdrant_client.search = AsyncMock(return_value=[mock_result])
    
    with patch('embedding.index.AsyncQdrantClient', return_value=mock_qdrant_client):
        index = QdrantIndex()
        await index.initialize()
        
        results = await index.search_async(
            query_vector=mock_embedding,
            top_k=10,
            score_threshold=0.7
        )
        
        assert len(results) == 1
        assert results[0]["score"] == 0.95
        assert results[0]["id"] == "test-id"
        assert results[0]["payload"]["venue"] == "kalshi"


@pytest.mark.asyncio
async def test_search_with_exclude_venue(mock_qdrant_client, mock_embedding):
    """Test search with venue exclusion filter."""
    mock_result = MagicMock()
    mock_result.id = "test-id"
    mock_result.score = 0.95
    mock_result.payload = {"venue": "polymarket"}
    
    mock_qdrant_client.search = AsyncMock(return_value=[mock_result])
    
    with patch('embedding.index.AsyncQdrantClient', return_value=mock_qdrant_client):
        index = QdrantIndex()
        await index.initialize()
        
        results = await index.search_async(
            query_vector=mock_embedding,
            exclude_venue="kalshi"
        )
        
        # Verify filter was applied
        call_args = mock_qdrant_client.search.call_args
        assert call_args[1]["query_filter"] is not None


@pytest.mark.asyncio
async def test_search_with_exclude_identity_hash(mock_qdrant_client, mock_embedding):
    """Test search with identity hash exclusion filter."""
    mock_result = MagicMock()
    mock_result.id = "other-id"
    mock_result.score = 0.95
    mock_result.payload = {"identity_hash": "other-hash"}
    
    mock_qdrant_client.search = AsyncMock(return_value=[mock_result])
    
    with patch('embedding.index.AsyncQdrantClient', return_value=mock_qdrant_client):
        index = QdrantIndex()
        await index.initialize()
        
        results = await index.search_async(
            query_vector=mock_embedding,
            exclude_identity_hash="xyz789"
        )
        
        # Verify filter was applied
        call_args = mock_qdrant_client.search.call_args
        assert call_args[1]["query_filter"] is not None


@pytest.mark.asyncio
async def test_get_by_identity_hash(mock_qdrant_client):
    """Test retrieving by identity hash."""
    # Mock scroll result (returns tuple: (results, next_page_offset))
    mock_point = MagicMock()
    mock_point.id = "xyz789"
    mock_point.vector = [0.1] * 2048
    mock_point.payload = {"venue": "kalshi", "identity_hash": "xyz789"}
    
    mock_qdrant_client.scroll = AsyncMock(return_value=([mock_point], None))
    
    with patch('embedding.index.AsyncQdrantClient', return_value=mock_qdrant_client):
        index = QdrantIndex()
        await index.initialize()
        
        result = await index.get_by_identity_hash("xyz789")
        
        assert result is not None
        assert result["id"] == "xyz789"
        assert len(result["vector"]) == 2048


@pytest.mark.asyncio
async def test_get_by_identity_hash_not_found(mock_qdrant_client):
    """Test retrieving non-existent identity hash."""
    mock_qdrant_client.retrieve = AsyncMock(return_value=[])
    
    with patch('embedding.index.AsyncQdrantClient', return_value=mock_qdrant_client):
        index = QdrantIndex()
        await index.initialize()
        
        result = await index.get_by_identity_hash("nonexistent")
        
        assert result is None


@pytest.mark.asyncio
async def test_api_key_authentication(mock_qdrant_client):
    """Test Qdrant Cloud API key authentication."""
    with patch('embedding.index.AsyncQdrantClient', return_value=mock_qdrant_client) as mock_client_class:
        index = QdrantIndex(
            url="https://cloud.qdrant.io",
            api_key="test-api-key-123"
        )
        
        # Verify client was created with API key
        mock_client_class.assert_called_once_with(
            url="https://cloud.qdrant.io",
            api_key="test-api-key-123"
        )


@pytest.mark.asyncio
async def test_no_api_key_local(mock_qdrant_client):
    """Test local Qdrant without API key."""
    with patch('embedding.index.AsyncQdrantClient', return_value=mock_qdrant_client) as mock_client_class:
        index = QdrantIndex(url="http://localhost:6333")
        
        # Verify client was created without API key
        mock_client_class.assert_called_once_with(url="http://localhost:6333")


@pytest.mark.asyncio
async def test_custom_vector_size(mock_qdrant_client):
    """Test custom vector size."""
    with patch('embedding.index.AsyncQdrantClient', return_value=mock_qdrant_client):
        index = QdrantIndex(vector_size=512)
        await index.initialize()
        
        assert index.vector_size == 512
        # Verify collection created with correct vector size
        call_args = mock_qdrant_client.create_collection.call_args
        assert call_args[1]["vectors_config"].size == 512


@pytest.mark.asyncio
async def test_custom_distance_metric(mock_qdrant_client):
    """Test custom distance metric."""
    from qdrant_client.models import Distance
    
    with patch('embedding.index.AsyncQdrantClient', return_value=mock_qdrant_client):
        index = QdrantIndex(distance=Distance.EUCLID)
        await index.initialize()
        
        assert index.distance == Distance.EUCLID


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_qdrant_connection(real_qdrant_config):
    """Test connection to real Qdrant (from .env)."""
    # Skip if no API key configured
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    
    await index.initialize()
    
    # Verify collection exists
    collections = await index._client.get_collections()
    assert real_qdrant_config["collection_name"] in [
        c.name for c in collections.collections
    ]

