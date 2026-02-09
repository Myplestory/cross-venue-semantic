"""
Qdrant vector database wrapper for market embeddings.

Industry standards:
- Connection pooling
- Batch upserts for efficiency
- Payload filtering for venue/date queries
- Async operations
"""

import logging
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

# Import from parent module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from canonicalization.types import CanonicalEvent
from .types import EmbeddedEvent

logger = logging.getLogger(__name__)


class QdrantIndex:
    """
    Qdrant vector database wrapper for market embeddings.
    
    Handles collection creation, batch upserts, and similarity search.
    Optimized for 4B model embeddings (2048-3072 dimensions).
    """
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "market_embeddings",
        vector_size: int = 2048,
        distance: Distance = Distance.COSINE,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Qdrant index.
        
        Args:
            url: Qdrant server URL
            collection_name: Collection name for embeddings
            vector_size: Vector dimension (must match embedding_dim)
            distance: Distance metric (COSINE for semantic similarity)
            api_key: Optional API key for Qdrant Cloud
        """
        self.url = url
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        
        # Initialize async client
        if api_key:
            self._client = AsyncQdrantClient(
                url=url,
                api_key=api_key,
            )
        else:
            self._client = AsyncQdrantClient(url=url)
        
        self._initialized = False
        logger.info(
            f"QdrantIndex initialized: url={url}, "
            f"collection={collection_name}, vector_size={vector_size}"
        )
    
    async def initialize(self) -> None:
        """Create collection if it doesn't exist."""
        if self._initialized:
            return
        
        try:
            # Check if collection exists
            collections = await self._client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                await self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance,
                    ),
                )
                logger.info(f"Collection '{self.collection_name}' created")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
            
            # Create indexes on payload fields for efficient filtering
            # Required for get_by_identity_hash, search filtering, and cross-venue matching
            try:
                from qdrant_client.models import PayloadSchemaType
                
                # Index on identity_hash for retrieval by identity
                await self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="identity_hash",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.debug("Created index on 'identity_hash' field")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Failed to create index on 'identity_hash': {e}")
            
            try:
                from qdrant_client.models import PayloadSchemaType
                
                # Index on venue for cross-venue matching filters
                await self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="venue",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.debug("Created index on 'venue' field")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Failed to create index on 'venue': {e}")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")
            raise
    
    async def upsert_async(
        self,
        embedded_events: List[EmbeddedEvent]
    ) -> None:
        """
        Batch upsert embeddings to Qdrant.
        
        Args:
            embedded_events: List of embedded events to upsert
        """
        if not self._initialized:
            await self.initialize()
        
        if not embedded_events:
            return
        
        # Prepare points for batch upsert
        points = []
        for event in embedded_events:
            # Convert identity_hash (hex string) to UUID for Qdrant point ID
            # Qdrant requires point IDs to be either unsigned integers or UUIDs
            # identity_hash is a 64-char hex string, we use first 32 chars for UUID
            identity_hash_hex = event.canonical_event.identity_hash
            try:
                # Convert hex string to UUID (first 32 hex chars = 16 bytes)
                point_id = uuid.UUID(hex=identity_hash_hex[:32])
            except ValueError:
                # Fallback: if not valid hex, create UUID5 from the hash string
                point_id = uuid.uuid5(uuid.NAMESPACE_DNS, identity_hash_hex)
            
            point = PointStruct(
                id=point_id,  # UUID format required by Qdrant
                vector=event.embedding,
                payload={
                    "venue": event.canonical_event.event.venue.value,
                    "venue_market_id": event.canonical_event.event.venue_market_id,
                    "identity_hash": event.canonical_event.identity_hash,
                    "content_hash": event.canonical_event.content_hash,
                    "canonical_text": event.canonical_event.canonical_text,
                    "embedding_model": event.embedding_model,
                    "embedding_dim": event.embedding_dim,
                    "created_at": event.created_at.isoformat(),
                }
            )
            points.append(point)
        
        # Batch upsert (Qdrant recommends 100-1000 points per batch)
        try:
            await self._client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.debug(
                f"Upserted {len(points)} embeddings to Qdrant collection "
                f"'{self.collection_name}'"
            )
        except Exception as e:
            logger.error(f"Failed to upsert embeddings to Qdrant: {e}")
            raise
    
    async def search_async(
        self,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: float = 0.7,
        exclude_venue: Optional[str] = None,
        exclude_identity_hash: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar markets.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            exclude_venue: Exclude results from this venue (for cross-venue matching)
            exclude_identity_hash: Exclude this specific market (self-exclusion)
            
        Returns:
            List of search results with scores and payloads
        """
        if not self._initialized:
            await self.initialize()
        
        # Build filter conditions
        filter_conditions = []
        
        if exclude_venue:
            # Cross-venue matching: exclude same venue
            filter_conditions.append(
                FieldCondition(
                    key="venue",
                    match=MatchValue(value=exclude_venue)
                )
            )
        
        if exclude_identity_hash:
            # Exclude self
            filter_conditions.append(
                FieldCondition(
                    key="identity_hash",
                    match=MatchValue(value=exclude_identity_hash)
                )
            )
        
        # Build query filter
        query_filter = None
        if filter_conditions:
            query_filter = Filter(
                must_not=filter_conditions
            )
        
        # Perform search
        try:
            # Try search first (works for mocks and sync clients)
            # If search fails, fall back to query_points (for real AsyncQdrantClient)
            results = None
            search_error = None
            
            if hasattr(self._client, 'search'):
                try:
                    results = await self._client.search(
                        collection_name=self.collection_name,
                        query_vector=query_vector,
                        query_filter=query_filter,
                        limit=top_k,
                        score_threshold=score_threshold,
                    )
                except (AttributeError, TypeError) as e:
                    # search doesn't work (real AsyncQdrantClient doesn't have search)
                    search_error = e
                    results = None
            
            # If search didn't work, try query_points (for real AsyncQdrantClient)
            if results is None:
                if hasattr(self._client, 'query_points'):
                    # Use query_points - query can be a vector directly or a Query object
                    # For AsyncQdrantClient, query can be the vector list directly
                    try:
                        # Try passing vector directly as query
                        response = await self._client.query_points(
                            collection_name=self.collection_name,
                            query=query_vector,  # Vector as query
                            query_filter=query_filter,
                            limit=top_k,
                            score_threshold=score_threshold,
                        )
                    except (TypeError, ValueError):
                        # If that doesn't work, try using NamedVector
                        from qdrant_client.models import NamedVector
                        named_vector = NamedVector(
                            vector=query_vector,
                            name="",  # Empty name for default vector
                        )
                        response = await self._client.query_points(
                            collection_name=self.collection_name,
                            query=named_vector,
                            query_filter=query_filter,
                            limit=top_k,
                            score_threshold=score_threshold,
                        )
                    
                    # Response is a QueryResponse object with .points attribute
                    if hasattr(response, 'points'):
                        results = response.points
                    elif isinstance(response, list):
                        results = response
                    else:
                        results = []
                else:
                    # Neither method works
                    if search_error:
                        raise AttributeError(
                            f"Qdrant client {type(self._client)} search method failed: {search_error}. "
                            f"query_points also not available."
                        )
                    else:
                        raise AttributeError(
                            f"Qdrant client {type(self._client)} has no search or query_points method."
                        )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search Qdrant: {e}")
            raise
    
    async def get_by_identity_hash(
        self,
        identity_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve embedding by identity hash.
        
        Uses payload filtering since we store identity_hash in payload
        (point ID is UUID derived from identity_hash, but we can't reverse it reliably).
        
        Args:
            identity_hash: Market identity hash
            
        Returns:
            Point data if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Search by identity_hash in payload (more reliable than UUID conversion)
            # Use scroll to find by payload filter
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="identity_hash",
                        match=MatchValue(value=identity_hash)
                    )
                ]
            )
            
            results, _ = await self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=1,
            )
            
            if results and len(results) > 0:
                point = results[0]
                return {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload,
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve from Qdrant: {e}")
            return None
