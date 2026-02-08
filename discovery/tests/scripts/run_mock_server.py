"""
Standalone mock WebSocket server for testing.

Run this to start a mock server that simulates venue responses.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from discovery.tests.test_websocket_server import KalshiMockServer, PolymarketMockServer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_kalshi_mock():
    """Run Kalshi mock server."""
    server = KalshiMockServer(port=8765)
    await server.start()
    logger.info("Kalshi mock server running on ws://localhost:8765")
    logger.info("Press Ctrl+C to stop")
    
    try:
        counter = 0
        while True:
            await asyncio.sleep(5)
            counter += 1
            await server.send_to_all({
                "type": "market_created",
                "data": {
                    "event_ticker": f"TEST-KALSHI-{counter}",
                    "title": f"Test Market #{counter}",
                    "description": "This is a periodic test market from mock server",
                    "resolution_criteria": "Resolves YES if test passes",
                    "end_time": "2024-12-31T23:59:59Z",
                    "outcomes": [
                        {"ticker": "YES", "name": "Yes"},
                        {"ticker": "NO", "name": "No"}
                    ]
                }
            })
            logger.info(f"Sent test market #{counter}")
    except KeyboardInterrupt:
        logger.info("\nStopping server...")
    finally:
        await server.stop()


async def run_polymarket_mock():
    """Run Polymarket mock server."""
    server = PolymarketMockServer(port=8766)
    await server.start()
    logger.info("Polymarket mock server running on ws://localhost:8766")
    logger.info("Press Ctrl+C to stop")
    
    try:
        counter = 0
        while True:
            await asyncio.sleep(5)
            counter += 1
            await server.send_to_all({
                "type": "market_created",
                "data": {
                    "id": f"0xtest{counter:04d}",
                    "question": f"Test Market #{counter}",
                    "description": "This is a periodic test market from mock server",
                    "resolutionSource": "Resolves YES if test passes",
                    "endDate": "2024-12-31T23:59:59Z",
                    "outcomes": [
                        {"token": f"0xyes{counter:04d}", "name": "Yes"},
                        {"token": f"0xno{counter:04d}", "name": "No"}
                    ]
                }
            })
            logger.info(f"Sent test market #{counter}")
    except KeyboardInterrupt:
        logger.info("\nStopping server...")
    finally:
        await server.stop()


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run mock WebSocket server")
    parser.add_argument(
        "--venue",
        choices=["kalshi", "polymarket", "both"],
        default="kalshi",
        help="Which venue to simulate"
    )
    
    args = parser.parse_args()
    
    if args.venue == "kalshi":
        await run_kalshi_mock()
    elif args.venue == "polymarket":
        await run_polymarket_mock()
    elif args.venue == "both":
        await asyncio.gather(
            run_kalshi_mock(),
            run_polymarket_mock(),
            return_exceptions=True
        )


if __name__ == "__main__":
    asyncio.run(main())

