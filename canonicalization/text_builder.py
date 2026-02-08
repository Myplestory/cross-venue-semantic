"""
Canonical text builder.

Converts venue-specific market data into normalized canonical format.
Venue-specific builders (Kalshi, Polymarket, etc.) normalize their API responses.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from datetime import datetime

from discovery.types import MarketEvent, VenueType

logger = logging.getLogger(__name__)


class CanonicalTextBuilder(ABC):
    """Base class for venue-specific text builders."""
    
    @abstractmethod
    def build(self, event: MarketEvent) -> str:
        """
        Convert MarketEvent to canonical markdown (synchronous).
        
        This is CPU-bound but fast (<5ms), so synchronous is acceptable.
        Wrapped in async function for consistency with pipeline.
        
        Args:
            event: Market event to canonicalize
            
        Returns:
            Canonical markdown text
        """
        ...
    
    async def build_async(self, event: MarketEvent) -> str:
        """
        Async wrapper for build() - allows integration with async pipeline.
        
        Args:
            event: Market event to canonicalize
            
        Returns:
            Canonical markdown text
        """
        # For fast operations, direct call is fine
        # For slower operations, use: return await asyncio.to_thread(self.build, event)
        return self.build(event)
    
    async def build_batch(
        self, 
        events: List[MarketEvent]
    ) -> List[Tuple[MarketEvent, str]]:
        """
        Build canonical text for multiple events in parallel (non-blocking).
        
        Uses asyncio.gather() for parallelization.
        
        Args:
            events: List of market events to canonicalize
            
        Returns:
            List of (event, canonical_text) tuples
        """
        tasks = [self.build_async(event) for event in events]
        texts = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for event, text in zip(events, texts):
            if isinstance(text, Exception):
                logger.error(
                    f"Error building text for {event.venue.value}:{event.venue_market_id}: {text}"
                )
                continue
            results.append((event, text))
        
        return results


class KalshiTextBuilder(CanonicalTextBuilder):
    """Kalshi-specific canonical text builder."""
    
    def build(self, event: MarketEvent) -> str:
        """
        Convert Kalshi MarketEvent to canonical markdown.
        
        Args:
            event: Kalshi market event
            
        Returns:
            Canonical markdown text
        """
        parts = []
        
        # Market Statement (required)
        parts.append(f"Market Statement:\n{event.title}")
        
        # Resolution Criteria (optional)
        if event.resolution_criteria:
            parts.append(f"\nResolution Criteria:\n{event.resolution_criteria}")
        
        # Clarifications (optional)
        clarifications = []
        if event.description:
            clarifications.append(event.description)
        
        if clarifications:
            bullets = '\n'.join(f"- {c}" for c in clarifications)
            parts.append(f"\nClarifications:\n{bullets}")
        
        # End Date (optional)
        if event.end_date:
            date_str = event.end_date.strftime('%Y-%m-%d')
            parts.append(f"\nEnd Date: {date_str}")
        
        # Outcomes (optional, only if multi-outcome)
        if event.outcomes and len(event.outcomes) > 2:
            outcome_labels = [outcome.label for outcome in event.outcomes]
            parts.append(f"\nOutcomes: {', '.join(outcome_labels)}")
        
        return '\n'.join(parts)


class PolymarketTextBuilder(CanonicalTextBuilder):
    """Polymarket-specific canonical text builder."""
    
    def build(self, event: MarketEvent) -> str:
        """
        Convert Polymarket MarketEvent to canonical markdown.
        
        Args:
            event: Polymarket market event
            
        Returns:
            Canonical markdown text
        """
        parts = []
        
        # Market Statement (required)
        parts.append(f"Market Statement:\n{event.title}")
        
        # Resolution Criteria (optional)
        if event.resolution_criteria:
            parts.append(f"\nResolution Criteria:\n{event.resolution_criteria}")
        
        # Clarifications (optional)
        clarifications = []
        if event.description:
            clarifications.append(event.description)
        
        if clarifications:
            bullets = '\n'.join(f"- {c}" for c in clarifications)
            parts.append(f"\nClarifications:\n{bullets}")
        
        # End Date (optional)
        if event.end_date:
            date_str = event.end_date.strftime('%Y-%m-%d')
            parts.append(f"\nEnd Date: {date_str}")
        
        # Outcomes (optional, only if multi-outcome)
        if event.outcomes and len(event.outcomes) > 2:
            outcome_labels = [outcome.label for outcome in event.outcomes]
            parts.append(f"\nOutcomes: {', '.join(outcome_labels)}")
        
        return '\n'.join(parts)


def get_builder(venue: VenueType) -> CanonicalTextBuilder:
    """
    Get venue-specific canonical text builder.
    
    Args:
        venue: Venue type
        
    Returns:
        CanonicalTextBuilder instance for the venue
        
    Raises:
        ValueError: If venue is not supported
    """
    builders = {
        VenueType.KALSHI: KalshiTextBuilder(),
        VenueType.POLYMARKET: PolymarketTextBuilder(),
    }
    
    if venue not in builders:
        raise ValueError(
            f"Unsupported venue: {venue.value}. "
            f"Supported venues: {[v.value for v in builders.keys()]}"
        )
    
    return builders[venue]
