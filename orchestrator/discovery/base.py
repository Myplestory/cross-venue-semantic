"""
Base discovery strategy interface.

Fintech-grade abstraction for market discovery strategies with proper
error handling, retry logic, and observability.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import timezone

from discovery.types import VenueType, MarketEvent
from discovery.base_connector import BaseVenueConnector
from orchestrator.exceptions import DiscoveryStrategyError

logger = logging.getLogger(__name__)
UTC = timezone.utc


class DiscoveryStrategy(ABC):
    """
    Abstract base class for discovery strategies.
    
    Each strategy defines how markets are discovered and filtered.
    Implements retry logic, error handling, and observability.
    """
    
    def __init__(self):
        """Initialize strategy with logging context."""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def get_venues(self) -> List[VenueType]:
        """
        Return list of venues to connect to.
        
        Returns:
            List of venue types for this strategy
            
        Raises:
            DiscoveryStrategyError: If venue configuration is invalid
        """
        pass
    
    @abstractmethod
    async def configure_connector(
        self,
        connector: BaseVenueConnector,
        venue: VenueType,
    ) -> None:
        """
        Configure a connector with strategy-specific settings.
        
        Called during orchestrator initialization for each connector.
        Should be idempotent (safe to call multiple times).
        
        Args:
            connector: Venue connector to configure
            venue: Venue type for context
            
        Raises:
            DiscoveryStrategyError: If configuration fails
        """
        pass
    
    @abstractmethod
    async def fetch_bootstrap_markets(
        self,
        connector: BaseVenueConnector,
        venue: VenueType,
        deadline: Optional[float] = None,
        max_markets: int = 0,
    ) -> List[MarketEvent]:
        """
        Fetch bootstrap markets for a connector.
        
        Strategy-specific filtering is applied here.
        Implements retry logic and error handling.
        
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
        pass
    
    @abstractmethod
    def should_process_event(self, event: MarketEvent) -> bool:
        """
        Filter WebSocket events in real-time.
        
        Returns True if event should be processed, False to skip.
        Should be fast (<1ms) as it's called for every event.
        
        Args:
            event: MarketEvent from WebSocket stream
            
        Returns:
            True if event should be processed, False to skip
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Return human-readable strategy name.
        
        Returns:
            Strategy name (e.g., "normal", "esports", "hybrid")
        """
        pass
    
    def get_description(self) -> str:
        """
        Return detailed strategy description.
        
        Default implementation returns name. Override for more details.
        """
        return self.get_name()
    
    async def validate_configuration(self) -> None:
        """
        Validate strategy configuration.
        
        Called during orchestrator initialization.
        Raises ConfigurationError if configuration is invalid.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Default: no validation. Override in subclasses.
        pass

