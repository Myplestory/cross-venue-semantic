"""
Single-market orderbook manager for real-time arbitrage monitoring.

Manages orderbooks for a single market pair across multiple venues,
recalculating arbitrage opportunities on every update.
"""

import asyncio
import logging
from typing import Optional, Callable, List
from datetime import datetime, UTC

import sys
from pathlib import Path

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from spread_scanner import (
    VenueBook,
    ArbOpportunity,
    EquivalentPair,
    find_arb_opportunities,
)

logger = logging.getLogger(__name__)


class SingleMarketOrderbookManager:
    """
    Manages orderbooks for a single market pair.
    
    Tracks both venues' books and recalculates arbitrage on every update.
    Thread-safe for concurrent orderbook updates from multiple WebSocket feeds.
    """
    
    def __init__(
        self,
        pair: EquivalentPair,
        on_opportunity: Optional[Callable[[ArbOpportunity], None]] = None,
    ):
        """
        Initialize orderbook manager for a single market pair.
        
        Args:
            pair: EquivalentPair defining the market pair
            on_opportunity: Optional callback when arbitrage opportunity is detected
        """
        self.pair = pair
        self.on_opportunity = on_opportunity
        self.kalshi_book: Optional[VenueBook] = None
        self.poly_book: Optional[VenueBook] = None
        self.last_opportunities: List[ArbOpportunity] = []
        self.update_count = 0
        self.last_update_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
    
    async def update_kalshi_book(self, book: VenueBook):
        """
        Update Kalshi orderbook and recalculate arbitrage.
        
        Args:
            book: VenueBook for Kalshi market
        """
        async with self._lock:
            self.kalshi_book = book
            self.update_count += 1
            self.last_update_time = datetime.now(UTC)
            await self._recalculate()
    
    async def update_poly_book(self, book: VenueBook):
        """
        Update Polymarket orderbook and recalculate arbitrage.
        
        Args:
            book: VenueBook for Polymarket market
        """
        async with self._lock:
            self.poly_book = book
            self.update_count += 1
            self.last_update_time = datetime.now(UTC)
            await self._recalculate()
    
    async def _recalculate(self):
        """
        Calculate arbitrage opportunities from current orderbooks.
        
        Called automatically after each orderbook update.
        """
        # Need both books to calculate arbitrage
        if not self.kalshi_book or not self.poly_book:
            return
        
        # Determine which book is venue_a and which is venue_b
        # Based on the pair definition
        if self.kalshi_book.venue == self.pair.venue_a:
            book_a = self.kalshi_book
            book_b = self.poly_book
        elif self.poly_book.venue == self.pair.venue_a:
            book_a = self.poly_book
            book_b = self.kalshi_book
        else:
            # Fallback: use order of pair definition
            if self.kalshi_book.venue == "kalshi":
                book_a = self.kalshi_book
                book_b = self.poly_book
            else:
                book_a = self.poly_book
                book_b = self.kalshi_book
        
        # Find arbitrage opportunities
        try:
            opportunities = find_arb_opportunities(self.pair, book_a, book_b)
            self.last_opportunities = opportunities
            
            # Call callback for each opportunity
            if self.on_opportunity and opportunities:
                for opp in opportunities:
                    try:
                        if asyncio.iscoroutinefunction(self.on_opportunity):
                            await self.on_opportunity(opp)
                        else:
                            self.on_opportunity(opp)
                    except Exception as e:
                        logger.error(
                            f"Error in opportunity callback: {e}",
                            exc_info=True
                        )
            
            # Log if opportunities found
            if opportunities:
                best = max(opportunities, key=lambda o: o.net_profit_1)
                logger.info(
                    f"[OrderbookManager] Arbitrage detected: "
                    f"gross_edge={best.gross_edge:.2%}, "
                    f"net_profit_1=${best.net_profit_1:.4f}, "
                    f"optimal_qty={best.optimal_qty:.0f}"
                )
        
        except Exception as e:
            logger.error(
                f"[OrderbookManager] Error calculating arbitrage: {e}",
                exc_info=True
            )
    
    def get_current_prices(self) -> dict:
        """
        Get current top-of-book prices from both venues.
        
        Returns:
            Dict with yes/no prices for both venues, or None if unavailable
        """
        prices = {
            "kalshi": {
                "yes": self.kalshi_book.yes_ask_top if self.kalshi_book else None,
                "no": self.kalshi_book.no_ask_top if self.kalshi_book else None,
            },
            "polymarket": {
                "yes": self.poly_book.yes_ask_top if self.poly_book else None,
                "no": self.poly_book.no_ask_top if self.poly_book else None,
            },
        }
        return prices
    
    def has_both_books(self) -> bool:
        """Check if both orderbooks are available."""
        return self.kalshi_book is not None and self.poly_book is not None
    
    def get_stats(self) -> dict:
        """
        Get manager statistics.
        
        Returns:
            Dict with update count, last update time, and opportunity count
        """
        return {
            "update_count": self.update_count,
            "last_update_time": (
                self.last_update_time.isoformat() if self.last_update_time else None
            ),
            "has_kalshi_book": self.kalshi_book is not None,
            "has_poly_book": self.poly_book is not None,
            "opportunity_count": len(self.last_opportunities),
            "current_prices": self.get_current_prices(),
        }

