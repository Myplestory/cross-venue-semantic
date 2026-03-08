"""
Real-time esports arbitrage monitor with game state integration.

Monitors a single market pair (DK vs T1) across Kalshi and Polymarket,
correlating game events with odds updates to detect frontrunning opportunities.

Usage:
    python monitor_dk_vs_t1.py

Environment Variables:
    KALSHI_API_KEY_ID: Kalshi API key ID
    KALSHI_PRIVATE_KEY_PATH: Path to Kalshi private key
    RIOT_API_KEY: Riot Games API key (optional, for game event integration)
    RIOT_MATCH_ID: Riot match identifier (optional)
    DATABASE_URL: PostgreSQL connection string (optional, for pair lookup)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from datetime import datetime, UTC
from typing import Optional

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import aiohttp
import config
from spread_scanner import EquivalentPair
from monitoring.core import (
    SingleMarketOrderbookManager,
    LatencyCorrelationEngine,
    FrontrunningDetector,
)
from monitoring.feeds import (
    KalshiWSFeed,
    PolymarketWSFeed,
    PolymarketSportsFeed,
    RiotAPIClient,
    RiotEventPoller,
    GameEventBridge,
    GameEvent,
)
from monitoring.compliance import (
    CircuitBreaker,
    CircuitBreakerConfig,
    AuditLogger,
    SystemMetrics,
)
from monitoring.utils.market_resolver import (
    resolve_polymarket_id,
    resolve_polymarket_tokens,
)
from monitoring.utils.match_discovery import (
    discover_series_games,
    MatchDiscovery,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Market identifiers (hardcoded for DK vs T1)
KALSHI_TICKER = "KXLOLGAME-26FEB22DKT1"
POLYMARKET_SLUG = "lol-t1-dk-2026-02-22"


class EsportsArbitrageMonitor:
    """
    Main orchestrator for esports arbitrage monitoring.
    
    Coordinates WebSocket feeds, orderbook management, latency correlation,
    and frontrunning detection for a single market pair.
    """
    
    def __init__(
        self,
        kalshi_ticker: str,
        polymarket_slug: str,
        riot_match_id: Optional[str] = None,
    ):
        """
        Initialize esports arbitrage monitor.
        
        Args:
            kalshi_ticker: Kalshi market ticker
            polymarket_slug: Polymarket market slug
            riot_match_id: Optional Riot match ID for game event polling
        """
        self.kalshi_ticker = kalshi_ticker
        self.polymarket_slug = polymarket_slug
        self.riot_match_id = riot_match_id
        
        # Initialize compliance components
        self.metrics = SystemMetrics()
        self.audit_logger = AuditLogger()
        self.circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                name="monitor_api",
            )
        )
        
        # Initialize core components
        self.pair = self._create_pair()
        self.orderbook_manager = SingleMarketOrderbookManager(
            pair=self.pair,
            on_opportunity=self._on_arbitrage_opportunity,
        )
        self.latency_engine = LatencyCorrelationEngine(
            correlation_window_ms=5000.0,
            metrics=self.metrics,
            audit_logger=self.audit_logger,
        )
        self.frontrunning_detector = FrontrunningDetector(
            min_price_move_pct=0.02,
            max_latency_ms=2000.0,
        )
        
        # Feed references (initialized in run())
        self.kalshi_feed: Optional[KalshiWSFeed] = None
        self.poly_feed: Optional[PolymarketWSFeed] = None
        self.sports_feed: Optional[PolymarketSportsFeed] = None
        self.riot_pollers: List[RiotEventPoller] = []  # Legacy pollers (deprecated)
        self.game_event_bridge: Optional[GameEventBridge] = None  # High-frequency bridge
        
        # State
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self.poly_market_id: Optional[str] = None
        self.poly_tokens: dict = {}
        self.series_games: List[Dict[str, Any]] = []  # Discovered games in series
        self.match_discovery: Optional[MatchDiscovery] = None
    
    def _create_pair(self) -> EquivalentPair:
        """Create EquivalentPair for the monitored markets."""
        return EquivalentPair(
            pair_key=f"{self.kalshi_ticker}-{self.polymarket_slug}",
            venue_a="kalshi",
            vmid_a=self.kalshi_ticker,
            title_a=f"DK vs T1 (Kalshi)",
            venue_b="polymarket",
            vmid_b=self.polymarket_slug,  # Will be replaced with condition_id
            title_b=f"DK vs T1 (Polymarket)",
            outcome_mapping={},  # Assume non-inverted (Yes=Yes)
            confidence=1.0,
            verdict="equivalent",
        )
    
    async def _on_arbitrage_opportunity(self, opportunity):
        """Handle detected arbitrage opportunity."""
        self.metrics.arb_opportunities_detected += 1
        
        # Log to audit trail
        self.audit_logger.log_arbitrage_opportunity(opportunity)
        
        # Alert
        logger.info(
            f"🚨 ARBITRAGE OPPORTUNITY: "
            f"gross_edge={opportunity.gross_edge:.2%}, "
            f"net_profit=${opportunity.net_profit_1:.4f}, "
            f"optimal_qty={opportunity.optimal_qty:.0f} contracts"
        )
    
    async def _on_orderbook_update(self, venue: str, book):
        """Handle orderbook update from WebSocket feed."""
        if venue == "kalshi":
            await self.orderbook_manager.update_kalshi_book(book)
        elif venue == "polymarket":
            await self.orderbook_manager.update_poly_book(book)
        
        # Record for latency correlation
        await self.latency_engine.record_orderbook_update(
            venue, self.pair.vmid_b if venue == "polymarket" else self.pair.vmid_a, book
        )
    
    async def _on_game_state_change(self, event: dict):
        """Handle game state change from Polymarket Sports or Riot API."""
        await self.latency_engine.record_game_event(event)
    
    async def _on_game_event_from_bridge(self, game_event: GameEvent):
        """
        Handle game event from high-frequency bridge.
        
        This callback receives events with precise timestamps for latency measurement.
        """
        # Convert GameEvent to dict format expected by LatencyCorrelationEngine
        event_dict = {
            "timestamp": game_event.event_timestamp,  # When event actually occurred
            "ingestion_timestamp": game_event.ingestion_timestamp,  # When we received it
            "event_type": game_event.event_type,
            "source": game_event.source,
            "match_id": game_event.match_id,
            "event_data": game_event.event_data,
        }
        
        # Record in latency engine
        await self.latency_engine.record_game_event(event_dict)
        
        # Log ingestion latency (time from event to our ingestion)
        ingestion_latency_ms = (
            (game_event.ingestion_timestamp - game_event.event_timestamp).total_seconds() * 1000
        )
        
        logger.debug(
            f"[Monitor] Game event ingested: {game_event.event_type} | "
            f"Match: {game_event.match_id} | "
            f"Ingestion latency: {ingestion_latency_ms:.1f}ms"
        )
    
    async def _on_game_state_change_with_update(self, event: dict):
        """Handle game state change and update discovered games."""
        # Update game status from Sports WS
        if self.match_discovery:
            self.series_games = await self.match_discovery.update_game_status_from_sports_ws(
                event, self.series_games
            )
        
        # Record for latency correlation
        await self._on_game_state_change(event)
    
    async def run(self):
        """Start monitoring - runs until stopped."""
        self._running = True
        
        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
        
        async with aiohttp.ClientSession() as session:
            self._session = session
            
            # Resolve Polymarket market ID and tokens
            logger.info(f"Resolving Polymarket market: {self.polymarket_slug}")
            self.poly_market_id = await resolve_polymarket_id(self.polymarket_slug)
            if not self.poly_market_id:
                logger.error(f"Could not resolve Polymarket market ID for: {self.polymarket_slug}")
                return
            
            logger.info(f"Resolved to condition_id: {self.poly_market_id}")
            
            # Resolve token IDs (with slug fallback)
            yes_token, no_token, _, _ = await resolve_polymarket_tokens(
                session, self.poly_market_id, slug=self.polymarket_slug
            )
            if not yes_token or not no_token:
                logger.error(f"Could not resolve Polymarket token IDs for: {self.poly_market_id}")
                return
            
            self.poly_tokens = {"yes_token": yes_token, "no_token": no_token}
            logger.info(f"Resolved tokens: yes={yes_token[:20]}..., no={no_token[:20]}...")
            
            # Update pair with resolved market ID
            self.pair.vmid_b = self.poly_market_id
            
            # Initialize WebSocket feeds
            self.kalshi_feed = KalshiWSFeed(
                ticker=self.kalshi_ticker,
                on_orderbook_update=lambda book: self._on_orderbook_update("kalshi", book),
                metrics=self.metrics,
            )
            
            self.poly_feed = PolymarketWSFeed(
                market_id=self.poly_market_id,
                token_map=self.poly_tokens,
                on_orderbook_update=lambda book: self._on_orderbook_update("polymarket", book),
                metrics=self.metrics,
            )
            
            self.sports_feed = PolymarketSportsFeed(
                market_slug=self.polymarket_slug,
                on_game_state_change=self._on_game_state_change_with_update,
                metrics=self.metrics,
            )
            
            # Discover series games and initialize game event bridge
            riot_tasks = []
            riot_api_key = os.getenv("RIOT_API_KEY")
            poll_interval_ms = float(os.getenv("GAME_EVENT_POLL_INTERVAL_MS", "2000.0"))  # Default 2000ms (safe for dev keys)
            
            # Check for manual Riot match IDs (comma-separated)
            manual_match_ids = os.getenv("RIOT_MATCH_IDS", "")
            if manual_match_ids:
                match_ids = {mid.strip() for mid in manual_match_ids.split(",") if mid.strip()}
                logger.info(f"Using manual Riot match IDs: {list(match_ids)}")
            else:
                # Discover all games in the series from Polymarket
                logger.info("Discovering games in series from Polymarket...")
                self.series_games = await discover_series_games(
                    market_slug=self.polymarket_slug,
                    team_a="DK",
                    team_b="T1",
                    session=session,
                    riot_client=None,  # Will be created below if API key available
                    metrics=self.metrics,
                )
                
                logger.info(
                    f"Discovered {len(self.series_games)} game(s) in series: "
                    f"{[g['game_number'] for g in self.series_games]}"
                )
                
                # Extract Riot match IDs from discovered games
                match_ids = {
                    game["riot_match_id"]
                    for game in self.series_games
                    if game.get("riot_match_id")
                }
            
            # Initialize high-frequency game event bridge (preferred for latency measurement)
            if riot_api_key and match_ids:
                logger.info(
                    f"Initializing GameEventBridge with {len(match_ids)} match ID(s) "
                    f"(poll interval: {poll_interval_ms}ms)"
                )
                
                self.game_event_bridge = GameEventBridge(
                    riot_api_key=riot_api_key,
                    match_ids=match_ids,
                    poll_interval_ms=poll_interval_ms,
                    on_event=self._on_game_event_from_bridge,  # Direct callback for latency
                    metrics=self.metrics,
                    audit_logger=self.audit_logger,
                    circuit_breaker=self.circuit_breaker,
                )
                
                await self.game_event_bridge.start()
                logger.info(
                    "✅ GameEventBridge started - high-frequency polling active "
                    f"({1000/poll_interval_ms:.1f} polls/second)"
                )
            elif riot_api_key and not match_ids:
                logger.warning(
                    "No Riot match IDs found. To enable game event polling, either:\n"
                    "  1. Set RIOT_MATCH_IDS environment variable (comma-separated)\n"
                    "  2. Or ensure discover_series_games finds games with match IDs\n"
                    "  Will use Polymarket Sports WS only for game state"
                )
            else:
                logger.info("No RIOT_API_KEY provided - using Polymarket Sports WS only")
            
            # Initialize match discovery (for game status tracking)
            self.match_discovery = MatchDiscovery(metrics=self.metrics)
            
            # Start all feeds
            logger.info("🚀 Starting all WebSocket feeds...")
            tasks = [
                self.kalshi_feed.run(session),
                self.poly_feed.run(session),
                self.sports_feed.run(),
                *riot_tasks,
                self._status_printer(),
            ]
            
            # Note: GameEventBridge runs its own polling loop, no need to add to tasks
            
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}", exc_info=True)
            finally:
                await self.stop()
    
    async def _status_printer(self):
        """Periodic status printing."""
        while self._running:
            await asyncio.sleep(10)  # Every 10 seconds
            
            stats = self.orderbook_manager.get_stats()
            prices = stats.get("current_prices", {})
            
            kalshi_yes = prices.get("kalshi", {}).get("yes")
            poly_yes = prices.get("polymarket", {}).get("yes")
            
            logger.info(
                f"[Status] Kalshi YES: ${kalshi_yes:.4f} | "
                f"Poly YES: ${poly_yes:.4f} | "
                f"Updates: {stats['update_count']} | "
                f"Opportunities: {self.metrics.arb_opportunities_detected}"
            )
            
            # Print latency stats
            latency_stats = self.metrics.get_latency_stats()
            if latency_stats["count"] > 0:
                logger.info(
                    f"[Latency] Samples: {latency_stats['count']} | "
                    f"Mean: {latency_stats['mean_ms']:.1f}ms | "
                    f"Min: {latency_stats['min_ms']:.1f}ms | "
                    f"Max: {latency_stats['max_ms']:.1f}ms"
                )
    
    async def stop(self):
        """Stop all feeds and cleanup."""
        if not self._running:
            return
        
        logger.info("🛑 Stopping monitor...")
        self._running = False
        
        # Stop game event bridge first (stops polling)
        if self.game_event_bridge:
            await self.game_event_bridge.stop()
            bridge_stats = self.game_event_bridge.get_stats()
            logger.info(
                f"[GameEventBridge] Final stats: "
                f"events_polled={bridge_stats['total_events_polled']}, "
                f"events_pushed={bridge_stats['total_events_pushed']}, "
                f"avg_poll_latency={bridge_stats['avg_poll_latency_ms']:.1f}ms"
            )
        
        # Stop all feeds
        if self.kalshi_feed:
            await self.kalshi_feed.stop()
        if self.poly_feed:
            await self.poly_feed.stop()
        if self.sports_feed:
            await self.sports_feed.stop()
        for poller in self.riot_pollers:
            await poller.stop()
        
        # Print final summary
        summary = self.metrics.get_summary()
        logger.info(f"📊 Final Summary: {summary}")


async def main():
    """Main entry point for DK vs T1 monitoring."""
    logger.info("🚀 Starting esports arbitrage monitor (DK vs T1)")
    logger.info(f"   Kalshi: {KALSHI_TICKER}")
    logger.info(f"   Polymarket: {POLYMARKET_SLUG}")
    
    riot_match_id = os.getenv("RIOT_MATCH_ID")
    if riot_match_id:
        logger.info(f"   Riot Match ID: {riot_match_id}")
    
    monitor = EsportsArbitrageMonitor(
        kalshi_ticker=KALSHI_TICKER,
        polymarket_slug=POLYMARKET_SLUG,
        riot_match_id=riot_match_id,
    )
    
    try:
        await monitor.run()
    except KeyboardInterrupt:
        logger.info("👋 Monitor stopped by user")
        await monitor.stop()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        await monitor.stop()
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Monitor stopped")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)

