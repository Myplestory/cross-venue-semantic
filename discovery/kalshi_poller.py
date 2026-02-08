"""
Kalshi venue connector for market discovery.

WebSocket-only market discovery from Kalshi.
Handles connection, reconnection, and event normalization.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from .base_connector import BaseVenueConnector
from .types import VenueType, MarketEvent, EventType, OutcomeSpec


logger = logging.getLogger(__name__)


class KalshiConnector(BaseVenueConnector):
    """Kalshi WebSocket connector for market discovery."""
    
    def __init__(
        self,
        ws_url: str = "wss://api.kalshi.com/trade-api/v2/websocket",
        **kwargs
    ):
        """
        Initialize Kalshi connector.
        
        Args:
            ws_url: Kalshi WebSocket URL
        """
        super().__init__(
            venue_name=VenueType.KALSHI,
            ws_url=ws_url,
            **kwargs
        )
    
    def _build_subscription_message(self) -> Optional[dict]:
        """
        Build Kalshi subscription message.
        
        Kalshi WebSocket format for market events:
        {
            "action": "subscribe",
            "channel": "markets",
            "params": {}
        }
        
        Or for specific markets:
        {
            "action": "subscribe",
            "channel": "markets",
            "params": {
                "event_ticker": "MARKET-123"
            }
        }
        """
        return {
            "action": "subscribe",
            "channel": "markets",
            "params": {}  # Empty = all markets
        }
    
    async def _parse_message(self, message: str) -> Optional[MarketEvent]:
        """
        Parse Kalshi WebSocket message into MarketEvent.
        
        Kalshi message formats:
        
        1. Subscription confirmation:
        {
            "action": "subscribe",
            "channel": "markets",
            "status": "subscribed"
        }
        
        2. Market event:
        {
            "channel": "markets",
            "type": "event",
            "data": {
                "event_ticker": "MARKET-123",
                "title": "Market title",
                "description": "Market description",
                "resolution_criteria": "...",
                "end_time": "2024-01-01T00:00:00Z",
                "status": "open" | "closed" | "resolved",
                "outcomes": [
                    {"ticker": "YES", "name": "Yes"},
                    {"ticker": "NO", "name": "No"}
                ]
            }
        }
        
        3. Market update:
        {
            "channel": "markets",
            "type": "update",
            "data": {
                "event_ticker": "MARKET-123",
                ...
            }
        }
        """
        try:
            data = json.loads(message)
            
            # Skip subscription confirmations
            if data.get("action") == "subscribe" and data.get("status") == "subscribed":
                logger.debug("[Kalshi] Subscription confirmed")
                return None
            
            # Handle market events
            if data.get("channel") == "markets":
                market_data = data.get("data", {})
                if not market_data:
                    return None
                
                # Determine event type from message type and status
                msg_type = data.get("type", "")
                status = market_data.get("status", "")
                
                event_type = EventType.UPDATED
                if msg_type == "event" and status == "open":
                    event_type = EventType.CREATED
                elif status == "resolved":
                    event_type = EventType.RESOLVED
                elif status == "closed":
                    event_type = EventType.CLOSED
                
                # Parse end date
                end_date = None
                if "end_time" in market_data:
                    try:
                        end_time_str = market_data["end_time"]
                        if isinstance(end_time_str, str):
                            end_date = datetime.fromisoformat(
                                end_time_str.replace("Z", "+00:00")
                            )
                    except (ValueError, AttributeError):
                        pass
                
                # Parse outcomes
                outcomes = []
                outcomes_data = market_data.get("outcomes")
                if outcomes_data is not None and isinstance(outcomes_data, list):
                    for outcome in outcomes_data:
                        if isinstance(outcome, dict):
                            outcomes.append(OutcomeSpec(
                                outcome_id=outcome.get("ticker", ""),
                                label=outcome.get("name", "")
                            ))
                
                return MarketEvent(
                    venue=VenueType.KALSHI,
                    venue_market_id=market_data.get("event_ticker", ""),
                    event_type=event_type,
                    title=market_data.get("title", ""),
                    description=market_data.get("description"),
                    resolution_criteria=market_data.get("resolution_criteria"),
                    end_date=end_date,
                    outcomes=outcomes,
                    raw_payload=data,
                    received_at=datetime.utcnow()
                )
            
            # Unknown message format
            logger.debug(f"[Kalshi] Unknown message format: {data.get('channel', 'unknown')}")
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"[Kalshi] Failed to parse JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"[Kalshi] Error parsing message: {e}")
            return None
