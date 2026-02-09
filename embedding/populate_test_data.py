"""
Utility script to populate Qdrant with test embeddings for retrieval testing.

This script:
1. Creates sample market events (Kalshi and Polymarket)
2. Canonicalizes them
3. Embeds them (using mocked encoder for speed, or real model)
4. Inserts them into Qdrant
5. Provides a simple query interface to test retrieval

Usage:
    python -m embedding.populate_test_data [--use-real-model] [--num-markets N]
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, UTC
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from discovery.types import MarketEvent, VenueType, EventType, OutcomeSpec
from canonicalization.types import CanonicalEvent
from canonicalization.text_builder import get_builder
from canonicalization.hasher import ContentHasher
from embedding.types import EmbeddedEvent
from embedding.encoder import EmbeddingEncoder
from embedding.index import QdrantIndex
from embedding.processor import EmbeddingProcessor
from embedding.cache.in_memory import InMemoryCache
import config
from unittest.mock import MagicMock, patch
import numpy as np


# Sample market data for testing - based on real Kalshi/Polymarket market structures
# Includes resolution criteria, secondary clauses, and proper formatting
SAMPLE_MARKETS = [
    # Kalshi markets (with detailed resolution criteria)
    {
        "venue": VenueType.KALSHI,
        "venue_market_id": "kalshi-001",
        "title": "Will Bitcoin reach $100,000 by December 31, 2025?",
        "description": "This market tracks whether Bitcoin (BTC) will reach or exceed $100,000 USD.",
        "resolution_criteria": (
            "This market resolves to Yes if Bitcoin (BTC) reaches or exceeds $100,000 USD "
            "on any of the following exchanges: Coinbase, Binance, Kraken, or Bitstamp, "
            "at any point before 11:59 PM ET on December 31, 2025. The price must be "
            "sustained for at least 1 minute. If Bitcoin reaches $100,000 but then falls "
            "below before the end date, the market resolves to No. In the event of a "
            "dispute, Coinbase closing price will be used as the authoritative source."
        ),
        "end_date": datetime(2025, 12, 31, 23, 59, 0, tzinfo=UTC),
        "outcomes": [
            OutcomeSpec(outcome_id="yes", label="Yes"),
            OutcomeSpec(outcome_id="no", label="No")
        ]
    },
    {
        "venue": VenueType.KALSHI,
        "venue_market_id": "kalshi-002",
        "title": "Will the S&P 500 close above 6000 in 2025?",
        "description": "This market tracks the S&P 500 index closing price for 2025.",
        "resolution_criteria": (
            "This market resolves to Yes if the S&P 500 index (SPX) closes at or above "
            "6000.00 on any trading day in 2025. The closing price is determined by the "
            "official S&P 500 index value at market close (4:00 PM ET) as reported by "
            "S&P Dow Jones Indices. If the index closes above 6000 on any single day in "
            "2025, the market resolves to Yes. If the index never closes at or above 6000 "
            "in 2025, the market resolves to No."
        ),
        "end_date": datetime(2025, 12, 31, 16, 0, 0, tzinfo=UTC),
        "outcomes": [
            OutcomeSpec(outcome_id="yes", label="Yes"),
            OutcomeSpec(outcome_id="no", label="No")
        ]
    },
    {
        "venue": VenueType.KALSHI,
        "venue_market_id": "kalshi-003",
        "title": "Will Ethereum reach $5,000 by December 31, 2025?",
        "description": "This market tracks whether Ethereum (ETH) will reach $5,000 USD.",
        "resolution_criteria": (
            "This market resolves to Yes if Ethereum (ETH) reaches or exceeds $5,000 USD "
            "on Coinbase, Binance, or Kraken at any point before 11:59 PM ET on December 31, "
            "2025. The price must be sustained for at least 1 minute. In case of exchange "
            "outages or data discrepancies, Coinbase will be used as the primary source. "
            "If Ethereum reaches $5,000 but then falls below before the end date, the "
            "market resolves to No."
        ),
        "end_date": datetime(2025, 12, 31, 23, 59, 0, tzinfo=UTC),
        "outcomes": [
            OutcomeSpec(outcome_id="yes", label="Yes"),
            OutcomeSpec(outcome_id="no", label="No")
        ]
    },
    {
        "venue": VenueType.KALSHI,
        "venue_market_id": "kalshi-004",
        "title": "Will the US enter a recession in 2025?",
        "description": "This market tracks whether the United States will experience a recession in 2025.",
        "resolution_criteria": (
            "This market resolves to Yes if the United States experiences two consecutive "
            "quarters of negative real GDP growth in 2025, as officially reported by the "
            "Bureau of Economic Analysis (BEA). The GDP data must be from the BEA's "
            "advance, preliminary, or final estimates. If the BEA revises data after the "
            "market closes, the most recent revision as of the market end date will be used. "
            "A recession is defined as two consecutive quarters of negative real GDP growth."
        ),
        "end_date": datetime(2025, 12, 31, 23, 59, 0, tzinfo=UTC),
        "outcomes": [
            OutcomeSpec(outcome_id="yes", label="Yes"),
            OutcomeSpec(outcome_id="no", label="No")
        ]
    },
    # Polymarket markets (similar topics, different wording for cross-venue matching)
    {
        "venue": VenueType.POLYMARKET,
        "venue_market_id": "poly-001",
        "title": "Bitcoin to $100k by 2025?",
        "description": "Will Bitcoin reach $100,000 USD before 2026?",
        "resolution_criteria": (
            "This market resolves to Yes if Bitcoin (BTC) trades at or above $100,000 USD "
            "on Coinbase, Binance, or Kraken at any time before January 1, 2026. The price "
            "must be maintained for at least 60 seconds. Coinbase will serve as the primary "
            "data source in case of discrepancies. If Bitcoin reaches $100k but drops below "
            "before year end, the market resolves to No."
        ),
        "end_date": datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC),
        "outcomes": [
            OutcomeSpec(outcome_id="yes", label="Yes"),
            OutcomeSpec(outcome_id="no", label="No")
        ]
    },
    {
        "venue": VenueType.POLYMARKET,
        "venue_market_id": "poly-002",
        "title": "S&P 500 above 6000 in 2025",
        "description": "Will the S&P 500 index close above 6000 points during 2025?",
        "resolution_criteria": (
            "This market resolves to Yes if the S&P 500 index closes at or above 6000 on "
            "any trading day in 2025. The official closing value from S&P Dow Jones Indices "
            "at 4:00 PM ET will be used. If the index closes above 6000 on any day in 2025, "
            "the market resolves to Yes. Otherwise, it resolves to No."
        ),
        "end_date": datetime(2025, 12, 31, 16, 0, 0, tzinfo=UTC),
        "outcomes": [
            OutcomeSpec(outcome_id="yes", label="Yes"),
            OutcomeSpec(outcome_id="no", label="No")
        ]
    },
    {
        "venue": VenueType.POLYMARKET,
        "venue_market_id": "poly-003",
        "title": "ETH $5000 by end of 2025",
        "description": "Ethereum price reaches $5000 USD by December 31, 2025.",
        "resolution_criteria": (
            "This market resolves to Yes if Ethereum (ETH) trades at or above $5,000 USD on "
            "Coinbase, Binance, or Kraken before 11:59 PM ET on December 31, 2025. The price "
            "must hold for at least 1 minute. Coinbase is the primary data source. If ETH "
            "hits $5k but falls below before year end, resolves to No."
        ),
        "end_date": datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC),
        "outcomes": [
            OutcomeSpec(outcome_id="yes", label="Yes"),
            OutcomeSpec(outcome_id="no", label="No")
        ]
    },
    {
        "venue": VenueType.POLYMARKET,
        "venue_market_id": "poly-004",
        "title": "US Recession in 2025?",
        "description": "Will the United States experience a recession in 2025?",
        "resolution_criteria": (
            "This market resolves to Yes if the US experiences two consecutive quarters of "
            "negative real GDP growth in 2025, as reported by the Bureau of Economic Analysis. "
            "BEA advance, preliminary, or final estimates will be used. If BEA revises data "
            "after market close, the most recent revision as of the end date applies. "
            "Recession = two consecutive quarters of negative real GDP growth."
        ),
        "end_date": datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC),
        "outcomes": [
            OutcomeSpec(outcome_id="yes", label="Yes"),
            OutcomeSpec(outcome_id="no", label="No")
        ]
    },
]


async def create_canonical_events(market_data: List[dict]) -> List[CanonicalEvent]:
    """Create canonical events from market data."""
    canonical_events = []
    
    for market in market_data:
        # Create MarketEvent with all fields including resolution_criteria and end_date
        market_event = MarketEvent(
            venue=market["venue"],
            venue_market_id=market["venue_market_id"],
            event_type=EventType.CREATED,
            title=market["title"],
            description=market.get("description", ""),
            resolution_criteria=market.get("resolution_criteria"),
            end_date=market.get("end_date"),
            outcomes=market["outcomes"],
            received_at=datetime.now(UTC)
        )
        
        # Canonicalize
        builder = get_builder(market_event.venue)
        canonical_text = builder.build(market_event)
        content_hash = ContentHasher.hash_content(canonical_text)
        identity_hash = ContentHasher.identity_hash(
            market_event.venue, market_event.venue_market_id
        )
        
        canonical_event = CanonicalEvent(
            event=market_event,
            canonical_text=canonical_text,
            content_hash=content_hash,
            identity_hash=identity_hash
        )
        canonical_events.append(canonical_event)
    
    return canonical_events


async def populate_qdrant(
    use_real_model: bool = False,
    num_markets: int = None
) -> None:
    """Populate Qdrant with test embeddings."""
    print("🚀 Starting Qdrant population with test data...")
    
    # Select markets
    markets_to_use = SAMPLE_MARKETS[:num_markets] if num_markets else SAMPLE_MARKETS
    print(f"📊 Creating {len(markets_to_use)} test markets...")
    
    # Create canonical events
    canonical_events = await create_canonical_events(markets_to_use)
    print(f"✅ Created {len(canonical_events)} canonical events")
    
    # Initialize components
    index = QdrantIndex(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
        vector_size=config.QDRANT_VECTOR_SIZE
    )
    
    encoder = EmbeddingEncoder(
        model_name=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    cache = InMemoryCache(max_size=config.EMBEDDING_CACHE_MAX_SIZE)
    
    # Initialize
    await index.initialize()
    print("✅ Qdrant index initialized")
    
    if use_real_model:
        print("🤖 Using REAL embedding model (this will download if not cached)...")
        await encoder.initialize()
    else:
        print("🎭 Using MOCKED embedding model (fast, for testing)...")
        mock_model = MagicMock()
        mock_model.encode = MagicMock(
            return_value=np.array([[0.1] * config.EMBEDDING_DIM])
        )
        mock_model.eval = MagicMock()
        
        with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
            await encoder.initialize()
    
    processor = EmbeddingProcessor(encoder, index, cache)
    await processor.initialize()
    print("✅ Embedding processor initialized")
    
    # Process and embed all events
    print(f"\n📝 Processing {len(canonical_events)} events...")
    embedded_events = []
    
    for i, canonical_event in enumerate(canonical_events, 1):
        print(f"  [{i}/{len(canonical_events)}] Processing: {canonical_event.event.title[:50]}...")
        embedded_event = await processor.process_async(canonical_event)
        embedded_events.append(embedded_event)
    
    print(f"\n✅ Successfully embedded and stored {len(embedded_events)} markets in Qdrant!")
    print(f"\n📊 Summary:")
    print(f"   - Collection: {config.QDRANT_COLLECTION_NAME}")
    print(f"   - Vector size: {config.QDRANT_VECTOR_SIZE}")
    print(f"   - Total markets: {len(embedded_events)}")
    
    # Show sample query
    print(f"\n🔍 You can now test retrieval with:")
    print(f"   from embedding.index import QdrantIndex")
    print(f"   index = QdrantIndex(...)")
    print(f"   results = await index.search_async(query_vector=..., top_k=5)")


async def test_retrieval():
    """Test retrieval with the populated data."""
    print("\n🧪 Testing retrieval...")
    
    index = QdrantIndex(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
        vector_size=config.QDRANT_VECTOR_SIZE
    )
    await index.initialize()
    
    # Create a test query vector (similar to Bitcoin market)
    test_vector = [0.1] * config.EMBEDDING_DIM
    
    # Search for similar markets
    results = await index.search_async(
        query_vector=test_vector,
        top_k=5,
        score_threshold=0.0
    )
    
    print(f"✅ Found {len(results)} results")
    for i, result in enumerate(results, 1):
        payload = result["payload"]
        print(f"  {i}. Score: {result['score']:.4f}")
        print(f"     Title: {payload.get('canonical_text', 'N/A')[:60]}...")
        print(f"     Venue: {payload.get('venue')}")
        print(f"     Identity Hash: {payload.get('identity_hash', 'N/A')[:16]}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Populate Qdrant with test embeddings")
    parser.add_argument(
        "--use-real-model",
        action="store_true",
        help="Use real embedding model (slower, requires model download)"
    )
    parser.add_argument(
        "--num-markets",
        type=int,
        default=None,
        help="Number of markets to create (default: all)"
    )
    parser.add_argument(
        "--test-retrieval",
        action="store_true",
        help="Test retrieval after populating"
    )
    
    args = parser.parse_args()
    
    async def main():
        await populate_qdrant(
            use_real_model=args.use_real_model,
            num_markets=args.num_markets
        )
        
        if args.test_retrieval:
            await test_retrieval()
    
    asyncio.run(main())

