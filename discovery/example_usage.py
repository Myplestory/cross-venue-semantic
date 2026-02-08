"""
Example usage of venue connectors.

Demonstrates how to use the modular venue connector system.
"""

import asyncio
import logging
from discovery.venue_factory import create_connector, list_available_venues
from discovery.types import VenueType
from discovery.dedup import MarketDeduplicator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def stream_single_venue(venue_type: VenueType):
    """Stream events from a single venue."""
    logger.info(f"Connecting to {venue_type.value}...")
    
    # Create connector using factory
    connector = create_connector(venue_type, reconnect_delay=5.0)
    dedup = MarketDeduplicator()
    
    try:
        await connector.connect()
        logger.info(f"Connected to {venue_type.value}! Streaming events...")
        
        event_count = 0
        async for event in connector.stream_events():
            # Deduplicate
            if dedup.is_duplicate(event):
                continue
            
            event_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Event #{event_count} from {event.venue.value}")
            logger.info(f"Market ID: {event.venue_market_id}")
            logger.info(f"Title: {event.title}")
            logger.info(f"Event Type: {event.event_type.value}")
            if event.outcomes:
                logger.info(f"Outcomes: {[o.label for o in event.outcomes]}")
            
            # Stop after 10 events for demo
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


async def stream_multiple_venues():
    """Stream events from multiple venues in parallel."""
    venues = [VenueType.POLYMARKET, VenueType.KALSHI]
    
    logger.info(f"Connecting to {len(venues)} venues...")
    
    # Create connectors for each venue
    connectors = {
        venue: create_connector(venue, reconnect_delay=5.0)
        for venue in venues
    }
    
    dedup = MarketDeduplicator()
    
    try:
        # Connect all
        for venue, connector in connectors.items():
            await connector.connect()
            logger.info(f"Connected to {venue.value}")
        
        # Stream from all venues
        tasks = []
        for venue, connector in connectors.items():
            task = asyncio.create_task(
                _stream_venue(connector, venue, dedup)
            )
            tasks.append(task)
        
        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        
    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    finally:
        for connector in connectors.values():
            await connector.disconnect()


async def _stream_venue(connector, venue: VenueType, dedup: MarketDeduplicator):
    """Helper to stream from a single venue."""
    event_count = 0
    try:
        async for event in connector.stream_events():
            if dedup.is_duplicate(event):
                continue
            
            event_count += 1
            logger.info(
                f"[{venue.value}] Event #{event_count}: {event.title[:50]}"
            )
            
            if event_count >= 10:
                break
    except Exception as e:
        logger.error(f"[{venue.value}] Error: {e}")


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Venue connector example")
    parser.add_argument(
        "--venue",
        choices=["kalshi", "polymarket", "both"],
        default="polymarket",
        help="Which venue(s) to connect to"
    )
    
    args = parser.parse_args()
    
    # List available venues
    available = list_available_venues()
    logger.info(f"Available venues: {[v.value for v in available]}")
    
    if args.venue == "both":
        await stream_multiple_venues()
    else:
        venue_type = VenueType.POLYMARKET if args.venue == "polymarket" else VenueType.KALSHI
        await stream_single_venue(venue_type)


if __name__ == "__main__":
    asyncio.run(main())

