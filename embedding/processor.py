"""
Main embedding processor that orchestrates encoding, caching, and indexing.

Follows async-first, queue-based architecture from discovery phase.
"""

import asyncio
import logging
from typing import AsyncIterator, List, Optional
from datetime import datetime, UTC

# Import from parent module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from canonicalization.types import CanonicalEvent
from .types import EmbeddedEvent
from .encoder import EmbeddingEncoder
from .index import QdrantIndex
from .cache.in_memory import InMemoryCache

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """
    Main embedding processor.
    
    Orchestrates:
    1. Batch accumulation from canonicalization queue
    2. Cache lookup (content_hash → embedding)
    3. Batch encoding (uncached items)
    4. Qdrant upsert
    5. Cache update
    """
    
    def __init__(
        self,
        encoder: EmbeddingEncoder,
        index: QdrantIndex,
        cache: Optional[InMemoryCache] = None,
        batch_size: int = 48,
        batch_timeout: float = 2.0,
    ):
        """
        Initialize processor.
        
        Args:
            encoder: Embedding encoder instance
            index: Qdrant index instance
            cache: Optional embedding cache
            batch_size: Batch size for encoding
            batch_timeout: Timeout for batch accumulation (seconds)
        """
        self.encoder = encoder
        self.index = index
        self.cache = cache
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Batch accumulation queue
        self._batch_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(
            f"EmbeddingProcessor initialized: batch_size={batch_size}, "
            f"batch_timeout={batch_timeout}"
        )
    
    async def initialize(self) -> None:
        """Initialize all components."""
        await self.encoder.initialize()
        await self.index.initialize()
        if self.cache:
            await self.cache.initialize()
        logger.info("EmbeddingProcessor initialized")
    
    async def start(self) -> None:
        """Start background processing task."""
        if self._processing_task is not None and not self._processing_task.done():
            return
        
        self._running = True
        self._processing_task = asyncio.create_task(self._process_batches())
        logger.info("EmbeddingProcessor started")
    
    async def stop(self) -> None:
        """Stop background processing."""
        self._running = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("EmbeddingProcessor stopped")
    
    async def process_async(
        self,
        canonical_event: CanonicalEvent
    ) -> EmbeddedEvent:
        """
        Process single canonical event (synchronous interface).
        
        Args:
            canonical_event: Canonical event to embed
            
        Returns:
            Embedded event
        """
        # Check cache first
        cached_embedding = None
        if self.cache:
            cached_embedding = await self.cache.get(
                canonical_event.content_hash
            )
        
        if cached_embedding:
            logger.debug(
                f"Cache hit for content_hash: {canonical_event.content_hash[:8]}"
            )
            embedded_event = EmbeddedEvent(
                canonical_event=canonical_event,
                embedding=cached_embedding,
                embedding_model=self.encoder.model_name,
                embedding_dim=self.encoder.embedding_dim,
            )
        else:
            # Encode (cache miss)
            embedding = await self.encoder.encode_async(
                canonical_event.canonical_text
            )
            
            # Create embedded event
            embedded_event = EmbeddedEvent(
                canonical_event=canonical_event,
                embedding=embedding,
                embedding_model=self.encoder.model_name,
                embedding_dim=self.encoder.embedding_dim,
            )
            
            # Update cache
            if self.cache:
                await self.cache.set(
                    canonical_event.content_hash,
                    embedding
                )
        
        # Upsert to Qdrant (even for cache hits, to ensure it's stored)
        await self.index.upsert_async([embedded_event])
        
        return embedded_event
    
    async def process_batch_async(
        self,
        canonical_events: List[CanonicalEvent],
    ) -> List[EmbeddedEvent]:
        """
        Process a batch of canonical events and return embedded events.

        Public batch interface for the orchestrator's micro-batch worker.
        Performs: cache lookup -> batch encode misses -> cache update ->
        Qdrant upsert -> return.

        This is the batch equivalent of ``process_async()`` -- same logic,
        but encodes all cache misses in one GPU call via
        ``encode_batch_async()``.

        Args:
            canonical_events: List of canonical events to embed.

        Returns:
            List of EmbeddedEvent objects (same order as input).
        """
        if not canonical_events:
            return []

        # ── Cache lookup ─────────────────────────────────────────────
        embedded_events: List[EmbeddedEvent] = []
        cache_misses: List[CanonicalEvent] = []
        # Preserve insertion order: map content_hash -> slot index
        order_map: dict = {}
        result_slots: List[Optional[EmbeddedEvent]] = [
            None
        ] * len(canonical_events)

        for idx, event in enumerate(canonical_events):
            cached_embedding = None
            if self.cache:
                cached_embedding = await self.cache.get(event.content_hash)

            if cached_embedding:
                ee = EmbeddedEvent(
                    canonical_event=event,
                    embedding=cached_embedding,
                    embedding_model=self.encoder.model_name,
                    embedding_dim=self.encoder.embedding_dim,
                )
                result_slots[idx] = ee
                embedded_events.append(ee)
            else:
                cache_misses.append(event)
                order_map[event.content_hash] = idx

        # ── Batch encode cache misses (1 GPU call) ───────────────────
        if cache_misses:
            texts = [e.canonical_text for e in cache_misses]
            embeddings = await self.encoder.encode_batch_async(texts)

            for event, embedding in zip(cache_misses, embeddings):
                ee = EmbeddedEvent(
                    canonical_event=event,
                    embedding=embedding,
                    embedding_model=self.encoder.model_name,
                    embedding_dim=self.encoder.embedding_dim,
                )
                result_slots[order_map[event.content_hash]] = ee
                embedded_events.append(ee)

                if self.cache:
                    await self.cache.set(event.content_hash, embedding)

        # ── Batch upsert to Qdrant ───────────────────────────────────
        if embedded_events:
            await self.index.upsert_async(embedded_events)

        cache_hits = len(canonical_events) - len(cache_misses)
        logger.info(
            "Batch processed: %d events (%d cache hits, %d encoded)",
            len(canonical_events),
            cache_hits,
            len(cache_misses),
        )

        # Return in original input order
        return [slot for slot in result_slots if slot is not None]

    async def enqueue(
        self,
        canonical_event: CanonicalEvent
    ) -> None:
        """
        Enqueue canonical event for batch processing.
        
        Args:
            canonical_event: Canonical event to queue
        """
        try:
            await self._batch_queue.put(canonical_event)
        except Exception as e:
            logger.error(f"Failed to enqueue canonical event: {e}")
    
    async def _process_batches(self) -> None:
        """Background task: process batches from queue."""
        batch = []
        last_batch_time = datetime.now(UTC)
        
        while self._running or not self._batch_queue.empty():
            try:
                # Accumulate batch
                try:
                    event = await asyncio.wait_for(
                        self._batch_queue.get(),
                        timeout=0.1
                    )
                    batch.append(event)
                except asyncio.TimeoutError:
                    # Check if batch should be processed (timeout or size)
                    if batch and (
                        len(batch) >= self.batch_size or
                        (datetime.now(UTC) - last_batch_time).total_seconds() >= self.batch_timeout
                    ):
                        await self._process_batch(batch)
                        batch = []
                        last_batch_time = datetime.now(UTC)
                    continue
                
                # Process batch if full
                if len(batch) >= self.batch_size:
                    await self._process_batch(batch)
                    batch = []
                    last_batch_time = datetime.now(UTC)
                
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                await asyncio.sleep(0.1)
        
        # Process remaining batch
        if batch:
            await self._process_batch(batch)
    
    async def _process_batch(
        self,
        canonical_events: List[CanonicalEvent]
    ) -> None:
        """Process a batch of canonical events."""
        if not canonical_events:
            return
        
        logger.info(f"Processing batch of {len(canonical_events)} events")
        
        # Check cache for all events
        cache_hits = {}
        cache_misses = []
        
        if self.cache:
            content_hashes = [e.content_hash for e in canonical_events]
            cache_results = await self.cache.get_batch(content_hashes)
            
            for event in canonical_events:
                if event.content_hash in cache_results:
                    cache_hits[event.content_hash] = cache_results[event.content_hash]
                else:
                    cache_misses.append(event)
        else:
            cache_misses = canonical_events
        
        # Encode cache misses
        embedded_events = []
        
        if cache_hits:
            # Create embedded events from cache
            for event in canonical_events:
                if event.content_hash in cache_hits:
                    embedded_events.append(EmbeddedEvent(
                        canonical_event=event,
                        embedding=cache_hits[event.content_hash],
                        embedding_model=self.encoder.model_name,
                        embedding_dim=self.encoder.embedding_dim,
                    ))
        
        if cache_misses:
            # Batch encode cache misses
            texts = [e.canonical_text for e in cache_misses]
            embeddings = await self.encoder.encode_batch_async(texts)
            
            # Create embedded events
            for event, embedding in zip(cache_misses, embeddings):
                embedded_event = EmbeddedEvent(
                    canonical_event=event,
                    embedding=embedding,
                    embedding_model=self.encoder.model_name,
                    embedding_dim=self.encoder.embedding_dim,
                )
                embedded_events.append(embedded_event)
                
                # Update cache
                if self.cache:
                    await self.cache.set(
                        event.content_hash,
                        embedding
                    )
        
        # Batch upsert to Qdrant
        if embedded_events:
            await self.index.upsert_async(embedded_events)
            logger.info(
                f"Processed batch: {len(cache_hits)} cache hits, "
                f"{len(cache_misses)} encoded, {len(embedded_events)} upserted"
            )

