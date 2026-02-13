"""
Candidate retriever for embedding-based market matching.

Top-K candidate retrieval from Qdrant vector database.
Filters by venue (cross-venue or intra-venue matching).
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, UTC

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from discovery.types import VenueType
from embedding.types import EmbeddedEvent
from embedding.index import QdrantIndex
from .types import CandidateMatch

logger = logging.getLogger(__name__)


class CandidateRetriever:
    """
    Retrieves candidate matches from Qdrant using vector similarity search.
    
    Handles:
    - Single and batch retrieval
    - Venue filtering (cross-venue matching)
    - Score threshold filtering
    - Error handling with retries
    """
    
    def __init__(
        self,
        index: QdrantIndex,
        default_top_k: int = 10,
        default_score_threshold: float = 0.7,
        max_retries: int = 3,
        retry_backoff_factor: float = 2.0,
        query_timeout: float = 5.0,
    ):
        """
        Initialize candidate retriever.
        
        Args:
            index: Qdrant index instance for vector search
            default_top_k: Default number of candidates to return
            default_score_threshold: Default minimum similarity score (0-1)
            max_retries: Maximum retry attempts for failed queries
            retry_backoff_factor: Exponential backoff multiplier
            query_timeout: Query timeout in seconds
        """
        self.index = index
        self.default_top_k = default_top_k
        self.default_score_threshold = default_score_threshold
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.query_timeout = query_timeout
        
        logger.info(
            f"CandidateRetriever initialized: top_k={default_top_k}, "
            f"score_threshold={default_score_threshold}, "
            f"max_retries={max_retries}, timeout={query_timeout}s"
        )
    
    async def initialize(self) -> None:
        """Initialize Qdrant index connection."""
        await self.index.initialize()
        logger.debug("CandidateRetriever initialized")
    
    async def retrieve_candidates(
        self,
        embedded_event: EmbeddedEvent,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        exclude_venue: Optional[VenueType] = None,
    ) -> List[CandidateMatch]:
        """
        Retrieve candidate matches for a single embedded event.
        
        Args:
            embedded_event: The event to find matches for
            top_k: Number of candidates to return (default: self.default_top_k)
            score_threshold: Minimum similarity score (default: self.default_score_threshold)
            exclude_venue: Exclude markets from this venue (for cross-venue matching)
            
        Returns:
            List of candidate matches sorted by similarity (highest first)
            
        Raises:
            ValueError: If embedded_event is invalid
            RuntimeError: If retrieval fails after max retries
        """
        if top_k is None:
            top_k = self.default_top_k
        if score_threshold is None:
            score_threshold = self.default_score_threshold
        
        exclude_venue_str = exclude_venue.value if exclude_venue else None
        
        return await self._retrieve_with_retry(
            query_vector=embedded_event.embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            exclude_venue=exclude_venue_str,
            exclude_identity_hash=embedded_event.canonical_event.identity_hash,
        )
    
    async def _retrieve_with_retry(
        self,
        query_vector: List[float],
        top_k: int,
        score_threshold: float,
        exclude_venue: Optional[str] = None,
        exclude_identity_hash: Optional[str] = None,
    ) -> List[CandidateMatch]:
        """
        Retrieve candidates with retry logic and exponential backoff.
        
        Retries on network errors, timeouts, and 5xx errors.
        Does not retry on 4xx errors or invalid queries.
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                results = await asyncio.wait_for(
                    self.index.search_async(
                        query_vector=query_vector,
                        top_k=top_k,
                        score_threshold=score_threshold,
                        exclude_venue=exclude_venue,
                        exclude_identity_hash=exclude_identity_hash,
                    ),
                    timeout=self.query_timeout,
                )
                
                return self._format_results(results)
                
            except asyncio.TimeoutError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_backoff_factor ** attempt
                    logger.warning(
                        f"Retrieval timeout (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Retrieval failed after {self.max_retries} attempts: {e}")
                    raise RuntimeError(f"Retrieval timeout after {self.max_retries} attempts") from e
                    
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_backoff_factor ** attempt
                    logger.warning(
                        f"Retrieval error (attempt {attempt + 1}/{self.max_retries}): {e}, "
                        f"retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Retrieval failed after {self.max_retries} attempts: {e}")
                    raise RuntimeError(f"Retrieval failed after {self.max_retries} attempts") from e
        
        raise RuntimeError("Should not reach here") from last_error
    
    def _format_results(
        self,
        qdrant_results: List[Dict[str, Any]]
    ) -> List[CandidateMatch]:
        """
        Format Qdrant search results into CandidateMatch objects.
        
        Extracts canonical event data from Qdrant payload and reconstructs
        CanonicalEvent objects for downstream processing.
        """
        candidates = []
        
        for result in qdrant_results:
            try:
                payload = result["payload"]
                score = result["score"]
                
                from discovery.types import MarketEvent, EventType
                
                venue_str = payload.get("venue")
                if not venue_str:
                    logger.warning(f"Missing venue in payload: {payload}")
                    continue
                
                try:
                    venue = VenueType(venue_str)
                except ValueError:
                    logger.warning(f"Invalid venue '{venue_str}' in payload: {payload}")
                    continue
                
                canonical_text = payload.get("canonical_text", "")
                venue_market_id = payload.get("venue_market_id", "")
                identity_hash = payload.get("identity_hash", "")
                content_hash = payload.get("content_hash", "")
                
                if not canonical_text or not venue_market_id or not identity_hash:
                    logger.warning(f"Missing required fields in payload: {payload}")
                    continue
                
                # Prefer stored title; fall back to line 2 of canonical_text
                # (line 1 is always the "Market Statement:" header)
                title = payload.get("title") or ""
                if not title:
                    lines = canonical_text.split('\n')
                    if len(lines) > 1 and lines[0].startswith("Market Statement"):
                        title = lines[1]
                    else:
                        title = lines[0] if lines else ""
                
                market_event = MarketEvent(
                    venue=venue,
                    venue_market_id=venue_market_id,
                    event_type=EventType.CREATED,
                    title=title,
                    description=canonical_text,
                    received_at=datetime.now(UTC),
                )
                
                from canonicalization.types import CanonicalEvent
                canonical_event = CanonicalEvent(
                    event=market_event,
                    canonical_text=canonical_text,
                    content_hash=content_hash,
                    identity_hash=identity_hash,
                )
                
                candidate = CandidateMatch(
                    canonical_event=canonical_event,
                    similarity_score=score,
                    embedding=[],  # Vector not included in search results by default
                    retrieval_metadata={
                        "qdrant_id": str(result.get("id", "")),
                        "embedding_model": payload.get("embedding_model", ""),
                        "embedding_dim": payload.get("embedding_dim", 0),
                    },
                )
                
                candidates.append(candidate)
                
            except Exception as e:
                logger.error(f"Error formatting result: {e}, result: {result}")
                continue
        
        candidates.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return candidates
