"""
Polymarket venue connector for market discovery.

WebSocket-only market discovery from Polymarket CLOB API.
Handles connection, reconnection, and event normalization.

Based on Polymarket CLOB WebSocket API:
https://docs.polymarket.com/developers/CLOB/websocket/wss-overview
"""

import json
import logging
from datetime import datetime, UTC
from typing import Optional, List

from .base_connector import BaseVenueConnector
from .types import VenueType, MarketEvent, EventType, OutcomeSpec


logger = logging.getLogger(__name__)


class PolymarketConnector(BaseVenueConnector):
    """
    Polymarket WebSocket connector for market discovery.
    
    Connects to Polymarket CLOB WebSocket for real-time market events.
    Uses subscription-based model to receive market updates.
    """
    
    def __init__(
        self,
        ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        **kwargs
    ):
        """
        Initialize Polymarket connector.
        
        Args:
            ws_url: Polymarket CLOB WebSocket URL
        """
        super().__init__(
            venue_name=VenueType.POLYMARKET,
            ws_url=ws_url,
            **kwargs
        )
        self._subscribed_markets: set = set()
    
    def _build_subscription_message(self) -> Optional[dict]:
        """
        Build Polymarket subscription message.
        
        Polymarket CLOB WebSocket format for market discovery:
        {
            "type": "market",
            "assets_ids": []  # Empty = subscribe to all markets
        }
        
        For specific markets:
        {
            "type": "market",
            "assets_ids": ["0x123...", "0x456..."]
        }
        """
        # Subscribe to all markets for discovery
        return {
            "type": "market",
            "assets_ids": []  # Empty list = all markets
        }
    
    async def _parse_message(self, message: str) -> Optional[MarketEvent]:
        """
        Parse Polymarket CLOB WebSocket message into MarketEvent.
        
        Polymarket CLOB message formats:
        
        1. Market data (orderbook/price updates):
        {
            "type": "orderbook" | "price",
            "market": "0x123...",
            "token_id": "0xabc...",
            ...
        }
        
        2. Market creation/update (from subscription):
        {
            "type": "market",
            "data": {
                "id": "0x123...",
                "question": "Market title",
                "description": "...",
                "resolutionSource": "...",
                "endDate": "2024-01-01T00:00:00Z",
                "outcomes": [...]
            }
        }
        
        Note: Polymarket CLOB primarily sends price/orderbook updates.
        For market discovery, we may need to combine with REST API polling
        or use a different endpoint. This implementation handles both cases.
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")
            
            # Handle orderbook/price updates - these indicate market activity
            if msg_type in ["orderbook", "price"]:
                # Extract market ID from orderbook/price message
                market_id = (
                    data.get("market") or 
                    data.get("market_id") or 
                    data.get("condition_id") or
                    ""
                )
                
                if not market_id:
                    return None
                
                # This is a market activity event (update)
                return MarketEvent(
                    venue=VenueType.POLYMARKET,
                    venue_market_id=market_id,
                    event_type=EventType.UPDATED,
                    title="",  # Not available in orderbook message
                    description=None,
                    resolution_criteria=None,
                    end_date=None,
                    outcomes=[],  # Not available in orderbook message
                    raw_payload=data,
                    received_at=datetime.now(UTC)
                )
            
            # Handle market data messages (if available)
            if msg_type == "market" and "data" in data:
                market_data = data.get("data", {})
                
                # Parse end date
                end_date = None
                if "endDate" in market_data:
                    try:
                        end_date_str = market_data["endDate"]
                        if isinstance(end_date_str, str):
                            end_date = datetime.fromisoformat(
                                end_date_str.replace("Z", "+00:00")
                            )
                    except (ValueError, AttributeError):
                        pass
                
                # Parse outcomes
                outcomes = []
                for outcome in market_data.get("outcomes", []):
                    if isinstance(outcome, dict):
                        outcomes.append(OutcomeSpec(
                            outcome_id=outcome.get("token", ""),
                            label=outcome.get("name", "")
                        ))
                
                # Determine event type
                event_type = EventType.UPDATED
                if market_data.get("status") == "resolved":
                    event_type = EventType.RESOLVED
                elif market_data.get("status") == "closed":
                    event_type = EventType.CLOSED
                elif not market_data.get("id") in self._subscribed_markets:
                    # New market we haven't seen
                    event_type = EventType.CREATED
                    self._subscribed_markets.add(market_data.get("id", ""))
                
                return MarketEvent(
                    venue=VenueType.POLYMARKET,
                    venue_market_id=market_data.get("id", ""),
                    event_type=event_type,
                    title=market_data.get("question", ""),
                    description=market_data.get("description"),
                    resolution_criteria=market_data.get("resolutionSource"),
                    end_date=end_date,
                    outcomes=outcomes,
                    raw_payload=data,
                    received_at=datetime.now(UTC)
                )
            
            # Unknown message type
            logger.debug(f"[Polymarket] Unknown message type: {msg_type}")
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"[Polymarket] Failed to parse JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"[Polymarket] Error parsing message: {e}")
            return None
