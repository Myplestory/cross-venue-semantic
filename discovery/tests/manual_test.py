"""
Manual testing script for venue connectors.

Connect to real venues (or mock server) and stream events.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discovery.kalshi_poller import KalshiConnector
from discovery.polymarket_poller import PolymarketConnector
from discovery.dedup import MarketDeduplicator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_kalshi_connector(use_mock: bool = False):
    """Test Kalshi connector."""
    if use_mock:
        ws_url = "ws://localhost:8765"
        logger.info("Using mock server at ws://localhost:8765")
    else:
        ws_url = "wss://api.elections.kalshi.com/trade-api/ws/v2"
        logger.info("Connecting to real Kalshi WebSocket")
    
    connector = KalshiConnector(ws_url=ws_url, reconnect_delay=5.0)
    dedup = MarketDeduplicator()
    
    try:
        logger.info("Connecting...")
        await connector.connect()
        logger.info("Connected! Streaming events...")
        
        event_count = 0
        async for event in connector.stream_events():
            if dedup.is_duplicate(event):
                logger.debug(f"Skipping duplicate: {event.venue_market_id}")
                continue
            
            event_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Event #{event_count}")
            logger.info(f"{'='*60}")
            logger.info(f"Venue: {event.venue.value}")
            logger.info(f"Market ID: {event.venue_market_id}")
            logger.info(f"Event Type: {event.event_type.value}")
            logger.info(f"Title: {event.title}")
            if event.description:
                logger.info(f"Description: {event.description[:100]}...")
            if event.outcomes:
                logger.info(f"Outcomes: {[o.label for o in event.outcomes]}")
            logger.info(f"Received At: {event.received_at}")
            
            # Exit after 10 events for testing
            if event_count >= 10:
                logger.info("\nReached 10 events, stopping...")
                break
                
    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        await connector.disconnect()
        logger.info("Disconnected")


async def test_polymarket_connector(use_mock: bool = False):
    """Test Polymarket connector."""
    if use_mock:
        ws_url = "ws://localhost:8766"
        logger.info("Using mock server at ws://localhost:8766")
    else:
        ws_url = "wss://gamma-api.polymarket.com/ws"
        logger.info("Connecting to real Polymarket WebSocket")
    
    connector = PolymarketConnector(ws_url=ws_url, reconnect_delay=5.0)
    dedup = MarketDeduplicator()
    
    try:
        logger.info("Connecting...")
        await connector.connect()
        logger.info("Connected! Streaming events...")
        
        event_count = 0
        async for event in connector.stream_events():
            if dedup.is_duplicate(event):
                logger.debug(f"Skipping duplicate: {event.venue_market_id}")
                continue
            
            event_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Event #{event_count}")
            logger.info(f"{'='*60}")
            logger.info(f"Venue: {event.venue.value}")
            logger.info(f"Market ID: {event.venue_market_id}")
            logger.info(f"Event Type: {event.event_type.value}")
            logger.info(f"Title: {event.title}")
            if event.description:
                logger.info(f"Description: {event.description[:100]}...")
            if event.outcomes:
                logger.info(f"Outcomes: {[o.label for o in event.outcomes]}")
            logger.info(f"Received At: {event.received_at}")
            
            # Exit after 10 events for testing
            if event_count >= 10:
                logger.info("\nReached 10 events, stopping...")
                break
                
    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        await connector.disconnect()
        logger.info("Disconnected")


async def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test venue connectors")
    parser.add_argument(
        "--venue",
        choices=["kalshi", "polymarket", "both"],
        default="kalshi",
        help="Which venue to test"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock WebSocket server instead of real venue"
    )
    
    args = parser.parse_args()
    
    if args.venue == "kalshi":
        await test_kalshi_connector(use_mock=args.mock)
    elif args.venue == "polymarket":
        await test_polymarket_connector(use_mock=args.mock)
    elif args.venue == "both":
        # Run both in parallel
        await asyncio.gather(
            test_kalshi_connector(use_mock=args.mock),
            test_polymarket_connector(use_mock=args.mock),
            return_exceptions=True
        )


if __name__ == "__main__":
    asyncio.run(main())

