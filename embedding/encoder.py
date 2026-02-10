"""
Async embedding encoder using Qwen3-Embedding-4B.

Industry standards:
- Singleton model instance (loaded once, reused)
- Batch processing for GPU efficiency
- Async wrapper for CPU-bound operations
- Error handling with retry logic
- Quantization support for memory efficiency
"""

import asyncio
import logging
from typing import List, Optional
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """
    Async embedding encoder with model reuse and batch processing.
    
    Uses Qwen3-Embedding-4B (or equivalent 4B variant) for semantic embeddings.
    Optimized for batch processing and GPU utilization.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        device: Optional[str] = None,
        batch_size: int = 48,
        max_length: int = 512,
        embedding_dim: int = 2048,
        instruction: Optional[str] = "Represent the market contract for similarity search.",
        use_quantization: bool = False,
    ):
        """
        Initialize encoder.
        
        Args:
            model_name: Hugging Face model identifier
            device: 'cuda', 'cpu', or None (auto-detect)
            batch_size: Batch size for encoding (48-64 optimal for 4B)
            max_length: Max token length (512 sufficient for market text)
            embedding_dim: Output dimension (2048 default, 32-2560 via MRL)
            instruction: Task-specific instruction for better embeddings
            use_quantization: Use 8-bit quantization (reduces VRAM by ~50%)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.instruction = instruction
        
        # Device selection (supports CUDA, MPS for M4 Mac, or CPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"  # Metal Performance Shaders for M4 Mac
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Model instance (singleton pattern)
        self._model: Optional[SentenceTransformer] = None
        self._model_lock = asyncio.Lock()
        self._initialized = False
        
        # Quantization config
        self.use_quantization = use_quantization and self.device == "cuda"
        
        logger.info(
            f"EmbeddingEncoder initialized: model={model_name}, "
            f"device={self.device}, batch_size={batch_size}, "
            f"embedding_dim={embedding_dim}, quantization={self.use_quantization}"
        )
    
    async def initialize(self) -> None:
        """
        Lazy model loading (async).
        
        Loads model once and reuses for all subsequent calls.
        Thread-safe initialization.
        """
        if self._initialized:
            return
        
        async with self._model_lock:
            if self._initialized:  # Double-check
                return
            
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                
                # Load model (CPU-bound, run in executor)
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None,
                    self._load_model_sync
                )
                
                self._initialized = True
                logger.info(f"Model loaded successfully on {self.device}")
                
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
    def _load_model_sync(self) -> SentenceTransformer:
        """Synchronous model loading (called from executor)."""
        # Load with quantization if requested
        if self.use_quantization:
            try:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.bfloat16
                )
                
                # Note: sentence-transformers may need custom loading for quantization
                # This is a placeholder - actual implementation depends on model support
                model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    model_kwargs={"quantization_config": quantization_config}
                )
            except ImportError:
                logger.warning("bitsandbytes not available, loading without quantization")
                model = SentenceTransformer(
                    self.model_name,
                    device=self.device
                )
        else:
            model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
        
        # Set model to eval mode
        model.eval()
        
        return model
    
    async def encode_async(self, text: str) -> List[float]:
        """
        Encode single text (async wrapper).
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector (list of floats)
        """
        if not self._initialized:
            await self.initialize()
        
        # Format with instruction if provided
        if self.instruction:
            formatted_text = f"{self.instruction}\n{text}"
        else:
            formatted_text = text
        
        # Run encoding in executor (CPU-bound GPU operations)
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                formatted_text,
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
            )
        )
        
        # Convert to list and ensure correct dimension
        embedding_list = embedding.tolist()
        
        # Handle 2D array (batch of 1) - take first element
        if isinstance(embedding_list[0], list):
            embedding_list = embedding_list[0]
        
        # Apply MRL dimension reduction if needed
        if len(embedding_list) != self.embedding_dim:
            embedding_list = self._apply_mrl(embedding_list)
        
        return embedding_list
    
    async def encode_batch_async(self, texts: List[str]) -> List[List[float]]:
        """
        Encode batch of texts (GPU-optimized).
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self.initialize()
        
        if not texts:
            return []
        
        # Format with instruction if provided
        if self.instruction:
            formatted_texts = [f"{self.instruction}\n{text}" for text in texts]
        else:
            formatted_texts = texts
        
        # Run batch encoding in executor
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                formatted_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )
        
        # Convert to list of lists
        embeddings_list = embeddings.tolist()
        
        # Apply MRL dimension reduction if needed
        if self.embedding_dim != len(embeddings_list[0]):
            embeddings_list = [
                self._apply_mrl(emb) for emb in embeddings_list
            ]
        
        return embeddings_list
    
    def _apply_mrl(self, embedding: List[float]) -> List[float]:
        """
        Apply Matryoshka Representation Learning dimension reduction.
        
        If model outputs larger dimension than requested, truncate to requested size.
        This maintains most of the semantic information in lower dimensions.
        """
        if len(embedding) > self.embedding_dim:
            return embedding[:self.embedding_dim]
        elif len(embedding) < self.embedding_dim:
            # Pad with zeros (shouldn't happen, but handle gracefully)
            logger.warning(
                f"Embedding dimension ({len(embedding)}) smaller than "
                f"requested ({self.embedding_dim}), padding with zeros"
            )
            return embedding + [0.0] * (self.embedding_dim - len(embedding))
        return embedding
    
    async def encode_with_retry(
        self,
        text: str,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ) -> List[float]:
        """
        Encode with retry logic and exponential backoff.
        
        Args:
            text: Text to encode
            max_retries: Maximum retry attempts
            backoff_factor: Exponential backoff multiplier
            
        Returns:
            Embedding vector
        """
        for attempt in range(max_retries):
            try:
                return await self.encode_async(text)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to encode after {max_retries} attempts: {e}")
                    raise
                
                delay = backoff_factor ** attempt
                logger.warning(
                    f"Encoding failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay}s: {e}"
                )
                await asyncio.sleep(delay)
        
        raise RuntimeError("Should not reach here")
