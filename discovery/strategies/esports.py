"""
Esports-only discovery strategy.

Discovers only esports/gaming markets from configured venues using:
- Polymarket: Client-side category and keyword filtering
- Kalshi: Client-side keyword filtering

Fintech-grade implementation with proper error handling, observability,
and performance optimizations.

Design principles:
- Non-blocking async operations
- Efficient regex pattern matching (compiled at init)
- Comprehensive error handling with context
- Structured logging for observability
- Configuration validation at initialization
- Memory-efficient filtering for large datasets
"""

import asyncio
import logging
import re
from typing import List, Optional, Set, Pattern
from datetime import timezone

from discovery.types import VenueType, MarketEvent
from discovery.base_connector import BaseVenueConnector
from orchestrator.discovery.base import DiscoveryStrategy
from orchestrator.exceptions import DiscoveryStrategyError, ConfigurationError
import config

logger = logging.getLogger(__name__)
UTC = timezone.utc


class EsportsDiscoveryStrategy(DiscoveryStrategy):
    """
    Esports discovery: only esports/gaming markets.
    
    Uses venue-specific filtering:
    - Polymarket: Client-side category and keyword filtering
    - Kalshi: Client-side keyword filtering (LOL, LEAGUE, DOTA, VALORANT, CSGO, ESPORT, ESPORTS, GAMING)
    """
    
    def __init__(self):
        """
        Initialize esports discovery strategy.
        
        Parses configuration, compiles regex patterns for efficient matching,
        and validates configuration at initialization.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        super().__init__()
        
        # Parse esports configuration
        self._polymarket_categories: List[str] = self._parse_categories(
            config.ESPORTS_POLYMARKET_CATEGORIES
        )
        self._kalshi_keywords: List[str] = self._parse_keywords(
            config.ESPORTS_KALSHI_KEYWORDS
        )
        
        # Compile regex patterns for efficient matching (word boundaries prevent false positives)
        self._kalshi_patterns: List[Pattern[str]] = [
            re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE)
            for kw in self._kalshi_keywords
        ]
        
        # Compile Polymarket keyword patterns for fallback matching
        # Use word boundaries to avoid false positives (e.g., "game" in "program")
        self._polymarket_keyword_patterns: List[Pattern[str]] = self._compile_polymarket_keywords()
        
        # Validate configuration
        self._validate_init_config()
        
        self._logger.info(
            "Initialized EsportsDiscoveryStrategy: "
            "Polymarket categories=%s (%d), Kalshi keywords=%d",
            self._polymarket_categories,
            len(self._polymarket_categories),
            len(self._kalshi_keywords),
        )
    
    def _compile_polymarket_keywords(self) -> List[Pattern[str]]:
        """
        Compile Polymarket fallback keyword patterns.
        
        Returns:
            List of compiled regex patterns for esports keywords
        """
        # Esports-related keywords for Polymarket fallback matching
        # Using word boundaries to prevent false positives
        keywords = [
            r"\besport\b", r"\besports\b", r"\bgaming\b", r"\bvideo\s+game\b",
            r"\bleague\s+of\s+legends\b", r"\blol\b", r"\bdota\b", r"\bvalorant\b",
            r"\bcsgo\b", r"\bcounter[-\s]?strike\b", r"\boverwatch\b",
            r"\bfortnite\b", r"\bapex\b", r"\bcall\s+of\s+duty\b", r"\bcod\b",
            r"\brainbow\s+six\b", r"\br6\b", r"\brocket\s+league\b", r"\bfifa\b",
            r"\bnba\s+2k\b", r"\bmadden\b", r"\btournament\b", r"\bchampionship\b",
            r"\bpro\s+player\b", r"\bstreamer\b", r"\bstreaming\b",
        ]
        
        return [re.compile(kw, re.IGNORECASE) for kw in keywords]
    
    def _validate_init_config(self) -> None:
        """
        Validate configuration at initialization.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Basic validation - full validation happens in validate_configuration()
        # which is called after venues are known
        if not self._polymarket_categories and not self._kalshi_keywords:
            self._logger.warning(
                "No esports filtering criteria configured. "
                "Set ESPORTS_POLYMARKET_CATEGORIES and/or ESPORTS_KALSHI_KEYWORDS"
            )
    
    def _parse_categories(self, categories_str: Optional[str]) -> List[str]:
        """
        Parse comma-separated category list.
        
        Args:
            categories_str: Comma-separated category names (may be None)
            
        Returns:
            List of category names (stripped, lowercased, non-empty)
        """
        if not categories_str or not categories_str.strip():
            return []
        
        categories = [
            cat.strip().lower()
            for cat in categories_str.split(",")
            if cat.strip()
        ]
        
        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique_categories = []
        for cat in categories:
            if cat not in seen:
                seen.add(cat)
                unique_categories.append(cat)
        
        return unique_categories
    
    def _parse_keywords(self, keywords_str: Optional[str]) -> List[str]:
        """
        Parse comma-separated keyword list.
        
        Args:
            keywords_str: Comma-separated keywords (may be None)
            
        Returns:
            List of keywords (stripped, uppercased, non-empty)
        """
        if not keywords_str or not keywords_str.strip():
            return []
        
        keywords = [
            kw.strip().upper()
            for kw in keywords_str.split(",")
            if kw.strip()
        ]
        
        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords
    
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
        
        self._logger.info(
            f"Configured venues for esports discovery: {[v.value for v in venues]}"
        )
        return venues
    
    async def configure_connector(
        self,
        connector: BaseVenueConnector,
        venue: VenueType,
    ) -> None:
        """
        Configure connector for esports discovery.
        
        For Polymarket, we'll pass category filters during bootstrap.
        For Kalshi, we'll filter client-side during bootstrap and streaming.
        
        Args:
            connector: Venue connector to configure
            venue: Venue type for context
        """
        # Store venue-specific configuration on connector if needed
        # For now, filtering happens in fetch_bootstrap_markets and should_process_event
        self._logger.debug(
            f"Configured connector for {venue.value} (esports mode)"
        )
    
    async def fetch_bootstrap_markets(
        self,
        connector: BaseVenueConnector,
        venue: VenueType,
        deadline: Optional[float] = None,
        max_markets: int = 0,
    ) -> List[MarketEvent]:
        """
        Fetch esports markets for a connector.
        
        Implements venue-specific filtering:
        - Polymarket: Client-side category and keyword filtering
        - Kalshi: Client-side keyword filter
        
        Args:
            connector: Venue connector to fetch from
            venue: Venue type for context
            deadline: Optional deadline (asyncio loop time)
            max_markets: Maximum markets to fetch (0 = unlimited)
            
        Returns:
            List of MarketEvent objects (esports/gaming only)
            
        Raises:
            DiscoveryStrategyError: If fetch fails after retries
        """
        if not hasattr(connector, "fetch_bootstrap_markets"):
            self._logger.warning(
                f"Connector {venue.value} does not support fetch_bootstrap_markets"
            )
            return []
        
        try:
            if venue == VenueType.POLYMARKET:
                return await self._fetch_polymarket_esports(
                    connector, deadline, max_markets
                )
            elif venue == VenueType.KALSHI:
                return await self._fetch_kalshi_esports(
                    connector, deadline, max_markets
                )
            else:
                self._logger.warning(
                    f"Esports discovery not implemented for venue: {venue.value}. "
                    f"Falling back to normal fetch with client-side filtering."
                )
                # Fallback: fetch all and filter client-side
                all_events = await connector.fetch_bootstrap_markets(
                    deadline=deadline,
                    max_markets=max_markets,
                )
                return [
                    event for event in all_events
                    if self.should_process_event(event)
                ]
        except Exception as exc:
            raise DiscoveryStrategyError(
                f"Failed to fetch esports markets from {venue.value}",
                cause=exc,
            ) from exc
    
    async def _fetch_polymarket_esports(
        self,
        connector: BaseVenueConnector,
        deadline: Optional[float] = None,
        max_markets: int = 0,
    ) -> List[MarketEvent]:
        """
        Fetch esports markets from Polymarket using client-side filtering.
        
        Fetches all active markets and filters client-side by checking:
        1. Category field in raw_payload (if available)
        2. Keywords in title/description using compiled regex patterns
        
        This approach ensures compatibility and reuses existing tested code
        while providing efficient filtering.
        
        Args:
            connector: Polymarket connector
            deadline: Optional deadline (asyncio loop time)
            max_markets: Maximum markets to fetch (0 = unlimited)
            
        Returns:
            List of esports MarketEvent objects
            
        Raises:
            DiscoveryStrategyError: If fetch fails
        """
        try:
            # Fetch all active markets using connector's existing method
            # This is async and non-blocking
            all_events = await connector.fetch_bootstrap_markets(
                deadline=deadline,
                max_markets=0,  # Fetch all, filter client-side
            )
            
            if not all_events:
                self._logger.info(
                    "[Esports/Polymarket] No markets fetched from connector"
                )
                return []
            
            # Filter by esports criteria (synchronous but fast - O(n) with compiled patterns)
            # Using generator for memory efficiency with large datasets
            esports_events = [
                event for event in all_events
                if self._matches_polymarket_esports(event)
            ]
            
            # Apply max_markets limit after filtering
            if max_markets > 0 and len(esports_events) > max_markets:
                esports_events = esports_events[:max_markets]
                self._logger.debug(
                    "[Esports/Polymarket] Limited to %d markets (from %d filtered)",
                    max_markets,
                    len(esports_events),
                )
            
            filter_ratio = (
                len(esports_events) / len(all_events) * 100
                if all_events
                else 0.0
            )
            
            self._logger.info(
                "[Esports/Polymarket] Fetched %d esports markets "
                "(from %d total active markets, %.1f%% match rate)",
                len(esports_events),
                len(all_events),
                filter_ratio,
            )
            
            return esports_events
            
        except asyncio.CancelledError:
            self._logger.warning(
                "[Esports/Polymarket] Fetch cancelled (deadline reached or shutdown)"
            )
            raise
        except Exception as exc:
            raise DiscoveryStrategyError(
                f"Failed to fetch esports markets from Polymarket: {exc}",
                cause=exc,
            ) from exc
    
    def _matches_polymarket_esports(self, event: MarketEvent) -> bool:
        """
        Check if a Polymarket market matches esports criteria.
        
        Uses efficient matching strategies:
        1. Category in raw_payload (fastest - O(1) lookup)
        2. Compiled regex patterns for keywords (prevents false positives)
        
        Args:
            event: MarketEvent to check
            
        Returns:
            True if event matches esports criteria, False otherwise
        """
        # Fast path: Check category in raw payload (most reliable)
        if event.raw_payload:
            category = (
                event.raw_payload.get("category")
                or event.raw_payload.get("categories")
                or event.raw_payload.get("category_slug")
            )
            
            if category:
                # Normalize category to lowercase for comparison
                category_str = str(category).lower()
                if isinstance(category, list):
                    category_str = ",".join(str(c).lower() for c in category)
                
                # Check if any configured category matches (case-insensitive)
                for cat in self._polymarket_categories:
                    if cat.lower() in category_str:
                        return True
        
        # Fallback: Keyword matching using compiled regex patterns
        # This prevents false positives (e.g., "game" in "program")
        search_text = " ".join([
            event.title or "",
            event.description or "",
        ])
        
        # Use compiled patterns for efficient matching
        for pattern in self._polymarket_keyword_patterns:
            if pattern.search(search_text):
                return True
        
        return False
    
    async def _fetch_kalshi_esports(
        self,
        connector: BaseVenueConnector,
        deadline: Optional[float] = None,
        max_markets: int = 0,
    ) -> List[MarketEvent]:
        """
        Fetch esports markets from Kalshi using client-side keyword filtering.
        
        Kalshi API doesn't support category filtering, so we fetch all active
        markets and filter client-side based on keywords in title/ticker using
        compiled regex patterns for efficient matching.
        
        Args:
            connector: Kalshi connector
            deadline: Optional deadline (asyncio loop time)
            max_markets: Maximum markets to fetch (0 = unlimited)
            
        Returns:
            List of esports MarketEvent objects
            
        Raises:
            DiscoveryStrategyError: If fetch fails
        """
        if not self._kalshi_keywords:
            self._logger.warning(
                "No Kalshi keywords configured for esports discovery"
            )
            return []
        
        try:
            # Fetch all active markets (async, non-blocking)
            all_events = await connector.fetch_bootstrap_markets(
                deadline=deadline,
                max_markets=0,  # Fetch all, filter client-side
            )
            
            if not all_events:
                self._logger.info(
                    "[Esports/Kalshi] No markets fetched from connector"
                )
                return []
            
            # Filter by keywords using compiled patterns (synchronous but fast)
            esports_events = [
                event for event in all_events
                if self._matches_kalshi_keywords(event)
            ]
            
            # Apply max_markets limit after filtering
            if max_markets > 0 and len(esports_events) > max_markets:
                esports_events = esports_events[:max_markets]
                self._logger.debug(
                    "[Esports/Kalshi] Limited to %d markets (from %d filtered)",
                    max_markets,
                    len(esports_events),
                )
            
            filter_ratio = (
                len(esports_events) / len(all_events) * 100
                if all_events
                else 0.0
            )
            
            self._logger.info(
                "[Esports/Kalshi] Fetched %d esports markets "
                "(from %d total active markets, %.1f%% match rate)",
                len(esports_events),
                len(all_events),
                filter_ratio,
            )
            
            return esports_events
            
        except asyncio.CancelledError:
            self._logger.warning(
                "[Esports/Kalshi] Fetch cancelled (deadline reached or shutdown)"
            )
            raise
        except Exception as exc:
            raise DiscoveryStrategyError(
                f"Failed to fetch esports markets from Kalshi: {exc}",
                cause=exc,
            ) from exc
    
    def _matches_kalshi_keywords(self, event: MarketEvent) -> bool:
        """
        Check if a Kalshi market matches esports keywords.
        
        Searches in title, description, and ticker (venue_market_id).
        
        Args:
            event: MarketEvent to check
            
        Returns:
            True if event matches esports keywords
        """
        search_text = " ".join([
            event.title or "",
            event.description or "",
            event.venue_market_id or "",
        ]).upper()
        
        # Check if any keyword pattern matches
        for pattern in self._kalshi_patterns:
            if pattern.search(search_text):
                return True
        
        return False
    
    def should_process_event(self, event: MarketEvent) -> bool:
        """
        Filter WebSocket events in real-time for esports markets.
        
        Args:
            event: MarketEvent from WebSocket stream
            
        Returns:
            True if event is an esports market, False otherwise
        """
        if event.venue == VenueType.POLYMARKET:
            # Filter Polymarket events by esports criteria
            return self._matches_polymarket_esports(event)
        
        elif event.venue == VenueType.KALSHI:
            # Filter Kalshi events by keywords
            return self._matches_kalshi_keywords(event)
        
        else:
            # Unknown venue: reject by default
            return False
    
    def get_name(self) -> str:
        """Return strategy name."""
        return "esports"
    
    def get_description(self) -> str:
        """Return detailed description."""
        return (
            f"Esports discovery: "
            f"Polymarket categories={self._polymarket_categories}, "
            f"Kalshi keywords={len(self._kalshi_keywords)} keywords"
        )
    
    async def validate_configuration(self) -> None:
        """
        Validate esports discovery configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        venues = self.get_venues()
        if not venues:
            raise ConfigurationError("No venues configured for esports discovery")
        
        # Check that we have filtering criteria for configured venues
        has_polymarket = VenueType.POLYMARKET in venues
        has_kalshi = VenueType.KALSHI in venues
        
        if has_polymarket and not self._polymarket_categories:
            raise ConfigurationError(
                "Polymarket is configured but ESPORTS_POLYMARKET_CATEGORIES is empty. "
                "Set ESPORTS_POLYMARKET_CATEGORIES in .env (e.g., 'esports,gaming,video-games')"
            )
        
        if has_kalshi and not self._kalshi_keywords:
            raise ConfigurationError(
                "Kalshi is configured but ESPORTS_KALSHI_KEYWORDS is empty. "
                "Set ESPORTS_KALSHI_KEYWORDS in .env (e.g., 'LOL,LEAGUE,DOTA,VALORANT,CSGO,ESPORT,ESPORTS,GAMING')"
            )

