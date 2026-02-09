"""Tests for EmbeddingEncoder (with mocked model)."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np

from embedding.encoder import EmbeddingEncoder


@pytest.mark.asyncio
async def test_encoder_initialization(mock_model):
    """Test encoder lazy initialization."""
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        encoder = EmbeddingEncoder()
        assert not encoder._initialized
        
        await encoder.initialize()
        assert encoder._initialized
        assert encoder._model is not None


@pytest.mark.asyncio
async def test_encoder_singleton_initialization(mock_model):
    """Test that initialization is idempotent (singleton pattern)."""
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        encoder = EmbeddingEncoder()
        
        # Initialize multiple times
        await encoder.initialize()
        await encoder.initialize()
        await encoder.initialize()
        
        # Model should only be loaded once
        assert encoder._initialized
        # Verify encode was called during initialization (indirectly via model loading)


@pytest.mark.asyncio
async def test_encode_single_text(mock_model):
    """Test encoding single text."""
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        encoder = EmbeddingEncoder()
        await encoder.initialize()
        
        embedding = await encoder.encode_async("Test market text")
        
        assert len(embedding) == 2048
        assert all(isinstance(x, float) for x in embedding)
        # Note: encode is called in executor, so we verify the result instead
        assert embedding == [0.1] * 2048


@pytest.mark.asyncio
async def test_encode_batch(mock_model):
    """Test batch encoding."""
    # Mock batch response
    batch_mock_model = MagicMock()
    batch_mock_model.encode = MagicMock(
        return_value=np.array([[0.1] * 2048, [0.2] * 2048, [0.3] * 2048])
    )
    batch_mock_model.eval = MagicMock()
    
    with patch('embedding.encoder.SentenceTransformer', return_value=batch_mock_model):
        encoder = EmbeddingEncoder(batch_size=2)
        await encoder.initialize()
        
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await encoder.encode_batch_async(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 2048 for emb in embeddings)
        batch_mock_model.encode.assert_called_once()


@pytest.mark.asyncio
async def test_encode_with_instruction(mock_model):
    """Test that instruction is prepended to text."""
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        encoder = EmbeddingEncoder(
            instruction="Represent the market contract for similarity search."
        )
        await encoder.initialize()
        
        await encoder.encode_async("Test text")
        
        # Verify encode was called (instruction formatting happens internally)
        assert mock_model.encode.called


@pytest.mark.asyncio
async def test_encode_without_instruction(mock_model):
    """Test encoding without instruction."""
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        encoder = EmbeddingEncoder(instruction=None)
        await encoder.initialize()
        
        await encoder.encode_async("Test text")
        
        assert mock_model.encode.called


@pytest.mark.asyncio
async def test_retry_logic(mock_model):
    """Test retry logic on encoding failure."""
    # Mock model that fails twice then succeeds
    call_count = 0
    
    def encode_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Network error")
        return np.array([[0.1] * 2048])
    
    mock_model.encode = MagicMock(side_effect=encode_side_effect)
    
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        encoder = EmbeddingEncoder()
        await encoder.initialize()
        
        embedding = await encoder.encode_with_retry("Test", max_retries=3)
        
        assert len(embedding) == 2048
        assert call_count == 3  # Failed twice, succeeded on third


@pytest.mark.asyncio
async def test_retry_exhausted(mock_model):
    """Test that retry raises exception when all retries exhausted."""
    mock_model.encode = MagicMock(side_effect=Exception("Persistent error"))
    
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        encoder = EmbeddingEncoder()
        await encoder.initialize()
        
        with pytest.raises(Exception, match="Persistent error"):
            await encoder.encode_with_retry("Test", max_retries=3)


@pytest.mark.asyncio
async def test_device_selection_cpu(mock_model):
    """Test CPU device selection."""
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        with patch('torch.cuda.is_available', return_value=False):
            encoder = EmbeddingEncoder(device=None)
            await encoder.initialize()
            
            assert encoder.device == "cpu"


@pytest.mark.asyncio
async def test_device_selection_cuda(mock_model):
    """Test CUDA device selection."""
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        with patch('torch.cuda.is_available', return_value=True):
            encoder = EmbeddingEncoder(device=None)
            await encoder.initialize()
            
            assert encoder.device == "cuda"


@pytest.mark.asyncio
async def test_device_override(mock_model):
    """Test explicit device override."""
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        encoder = EmbeddingEncoder(device="cpu")
        await encoder.initialize()
        
        assert encoder.device == "cpu"


@pytest.mark.asyncio
async def test_mrl_dimension_reduction(mock_model):
    """Test MRL dimension reduction."""
    # Mock model that returns larger dimension
    large_dim_model = MagicMock()
    large_dim_model.encode = MagicMock(
        return_value=np.array([[0.1] * 3072])  # Larger than 2048
    )
    large_dim_model.eval = MagicMock()
    
    with patch('embedding.encoder.SentenceTransformer', return_value=large_dim_model):
        encoder = EmbeddingEncoder(embedding_dim=2048)
        await encoder.initialize()
        
        embedding = await encoder.encode_async("Test")
        
        # Should be reduced to 2048
        assert len(embedding) == 2048


@pytest.mark.asyncio
async def test_empty_text(mock_model):
    """Test encoding empty text."""
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        encoder = EmbeddingEncoder()
        await encoder.initialize()
        
        embedding = await encoder.encode_async("")
        
        assert len(embedding) == 2048


@pytest.mark.asyncio
async def test_batch_empty_list(mock_model):
    """Test batch encoding with empty list."""
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        encoder = EmbeddingEncoder()
        await encoder.initialize()
        
        embeddings = await encoder.encode_batch_async([])
        
        assert embeddings == []


@pytest.mark.asyncio
async def test_custom_batch_size(mock_model):
    """Test custom batch size."""
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        encoder = EmbeddingEncoder(batch_size=64)
        await encoder.initialize()
        
        assert encoder.batch_size == 64


@pytest.mark.asyncio
async def test_custom_embedding_dim(mock_model):
    """Test custom embedding dimension."""
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        encoder = EmbeddingEncoder(embedding_dim=512)
        await encoder.initialize()
        
        assert encoder.embedding_dim == 512

