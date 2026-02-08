"""
Example: How to add a new venue connector.

This demonstrates the modular architecture for adding new venues.
"""

from datetime import datetime
from typing import Optional
import json
import logging

from .base_connector import BaseVenueConnector
from .types import VenueType, MarketEvent, EventType, OutcomeSpec
from .venue_factory import register_venue

logger = logging.getLogger(__name__)


class OpinionConnector(BaseVenueConnector):
    """
    Example: Opinion Markets connector.
    
    This is a template for adding new venues.
    """
    
    def __init__(
        self,
        ws_url: str = "wss://api.opinion.markets/ws",
        **kwargs
    ):
        """
        Initialize Opinion connector.
        
        Args:
            ws_url: Opinion WebSocket URL
        """
        super().__init__(
            venue_name=VenueType.OPINION,
            ws_url=ws_url,
            **kwargs
        )
    
    def _build_subscription_message(self) -> Optional[dict]:
        """
        Build venue-specific subscription message.
        
        Replace with actual Opinion API format.
        """
        return {
            "action": "subscribe",
            "channel": "markets"
        }
    
    async def _parse_message(self, message: str) -> Optional[MarketEvent]:
        """
        Parse venue-specific message into MarketEvent.
        
        Replace with actual Opinion message format.
        """
        try:
            data = json.loads(message)
            
            # Extract market data from Opinion's format
            market_data = data.get("market", {})
            if not market_data:
                return None
            
            # Map Opinion's event types
            event_type = EventType.UPDATED
            if data.get("event") == "created":
                event_type = EventType.CREATED
            elif data.get("event") == "resolved":
                event_type = EventType.RESOLVED
            
            # Parse outcomes
            outcomes = []
            for outcome in market_data.get("options", []):
                outcomes.append(OutcomeSpec(
                    outcome_id=outcome.get("id", ""),
                    label=outcome.get("label", "")
                ))
            
            return MarketEvent(
                venue=VenueType.OPINION,
                venue_market_id=market_data.get("id", ""),
                event_type=event_type,
                title=market_data.get("title", ""),
                description=market_data.get("description"),
                resolution_criteria=market_data.get("rules"),
                end_date=None,  # Parse if available
                outcomes=outcomes,
                raw_payload=data,
                received_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"[Opinion] Error parsing message: {e}")
            return None


# Register the new venue
register_venue(VenueType.OPINION, OpinionConnector)

# Now you can use it:
# from discovery.venue_factory import create_connector
# connector = create_connector(VenueType.OPINION)

