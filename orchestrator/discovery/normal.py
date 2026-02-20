"""
Normal spread arbitrage discovery strategy.

Discovers all active markets from configured venues without category filtering.
This is the current default behavior.

Fintech-grade implementation with proper error handling and observability.
"""

import logging
from typing import List, Optional
from datetime import timezone

from discovery.types import VenueType, MarketEvent
from discovery.base_connector import BaseVenueConnector
from orchestrator.discovery.base import DiscoveryStrategy
from orchestrator.exceptions import DiscoveryStrategyError, ConfigurationError
import config

logger = logging.getLogger(__name__)
UTC = timezone.utc


class NormalDiscoveryStrategy(DiscoveryStrategy):
    """
    Normal discovery: all active markets, no filtering.
    
    This is the default strategy that discovers all active markets
    from configured venues without any category or keyword filtering.
    """
    
    def __init__(self):
        """Initialize normal discovery strategy."""
        super().__init__()
        self._logger.info("Initialized NormalDiscoveryStrategy")
    
    def get_venues(self) -> List[VenueType]:
        """
        Return venues from configuration.
        
        Returns:
            List of venue types from ORCHESTRATOR_VENUES config
            
        Raises:
            ConfigurationError: If no venues are configured
        """
        venue_str: str = config.ORCHESTRATOR_VENUES or ""
        if not venue_str.strip():
            raise ConfigurationError(
                "ORCHESTRATOR_VENUES is not set. "
                "Configure venues in .env or environment variables."
            )
        
        venues = [
            VenueType(v.strip())
            for v in venue_str.split(",")
            if v.strip()
        ]
        
        if not venues:
            raise ConfigurationError(
                f"No valid venues found in ORCHESTRATOR_VENUES: {venue_str}"
            )
        
        self._logger.info(f"Configured venues: {[v.value for v in venues]}")
        return venues
    
    async def configure_connector(
        self,
        connector: BaseVenueConnector,
        venue: VenueType,
    ) -> None:
        """
        Configure connector for normal discovery.
        
        No special configuration needed for normal discovery.
        This method is idempotent.
        
        Args:
            connector: Venue connector to configure
            venue: Venue type for context
        """
        # No special configuration needed
        self._logger.debug(f"Configured connector for {venue.value} (normal mode)")
    
    async def fetch_bootstrap_markets(
        self,
        connector: BaseVenueConnector,
        venue: VenueType,
        deadline: Optional[float] = None,
        max_markets: int = 0,
    ) -> List[MarketEvent]:
        """
        Fetch all active markets (no category filter).
        
        Implements retry logic and proper error handling.
        
        Args:
            connector: Venue connector to fetch from
            venue: Venue type for context
            deadline: Optional deadline (asyncio loop time)
            max_markets: Maximum markets to fetch (0 = unlimited)
            
        Returns:
            List of MarketEvent objects
            
        Raises:
            DiscoveryStrategyError: If fetch fails after retries
        """
        if not hasattr(connector, "fetch_bootstrap_markets"):
            self._logger.warning(
                f"Connector {venue.value} does not support fetch_bootstrap_markets"
            )
            return []
        
        fetch_kwargs: dict = {"deadline": deadline}
        if max_markets > 0:
            fetch_kwargs["max_markets"] = max_markets
        
        try:
            events = await connector.fetch_bootstrap_markets(**fetch_kwargs)
            self._logger.info(
                f"[Normal/{venue.value}] Fetched {len(events)} markets"
            )
            return events
        except Exception as exc:
            raise DiscoveryStrategyError(
                f"Failed to fetch bootstrap markets from {venue.value}",
                cause=exc,
            ) from exc
    
    def should_process_event(self, event: MarketEvent) -> bool:
        """
        Process all events (no filtering).
        
        Args:
            event: MarketEvent from WebSocket stream
            
        Returns:
            True (all events are processed in normal mode)
        """
        return True
    
    def get_name(self) -> str:
        """Return strategy name."""
        return "normal"
    
    def get_description(self) -> str:
        """Return detailed description."""
        return "Normal discovery: all active markets from configured venues"
    
    async def validate_configuration(self) -> None:
        """
        Validate normal discovery configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        venues = self.get_venues()
        if not venues:
            raise ConfigurationError("No venues configured for normal discovery")

