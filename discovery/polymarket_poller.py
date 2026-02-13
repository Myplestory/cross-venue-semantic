"""
Polymarket venue connector for market discovery.

REST bootstrap + WebSocket streaming from Polymarket.

Bootstrap:
  Gamma API: https://gamma-api.polymarket.com/markets
  Docs: https://docs.polymarket.com/developers/gamma-markets-api/fetch-markets-guide
  No authentication required.
  Server-side filters: ``closed=false`` + ``active=true`` → only actively-trading markets.
  Rate limit: ~100 requests/minute.

WebSocket (real-time updates):
  CLOB API: wss://ws-subscriptions-clob.polymarket.com/ws/market
  Docs: https://docs.polymarket.com/developers/CLOB/websocket/wss-overview
  Primarily delivers orderbook/price updates (not new-market creation).
"""

import asyncio
import json
import logging
from datetime import datetime, UTC
from typing import Optional, List

from .base_connector import BaseVenueConnector
from .types import VenueType, MarketEvent, EventType, OutcomeSpec


logger = logging.getLogger(__name__)

# Gamma API (REST) — no auth, public endpoint
_GAMMA_API_BASE = "https://gamma-api.polymarket.com"
_GAMMA_MARKETS_PATH = "/markets"
_GAMMA_PAGE_SIZE = 100  # Max per page
_GAMMA_RATE_LIMIT_DELAY = 0.7  # ~85 req/min (under 100 req/min limit)


class PolymarketConnector(BaseVenueConnector):
    """
    Polymarket connector: REST bootstrap + WebSocket streaming.

    Bootstrap (startup):
      Fetches all active markets via the Gamma REST API with server-side
      filters ``closed=false`` and ``active=true``.  Offset-based
      pagination, rate-limited to stay under 100 req/min.

    WebSocket (steady-state):
      Streams orderbook/price updates from the CLOB WebSocket.
      Note: the WebSocket does NOT reliably deliver new-market creation
      events with full metadata (title, description, outcomes).  New
      market discovery requires periodic REST polling (Phase 2).
    """

    def __init__(
        self,
        ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        gamma_api_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Polymarket connector.

        Args:
            ws_url: Polymarket CLOB WebSocket URL.
            gamma_api_url: Override for Gamma REST API base URL
                (default: ``https://gamma-api.polymarket.com``).
        """
        super().__init__(
            venue_name=VenueType.POLYMARKET,
            ws_url=ws_url,
            **kwargs,
        )
        self._gamma_api_url: str = (gamma_api_url or _GAMMA_API_BASE).rstrip("/")
        self._subscribed_markets: set = set()
    
    # ═══════════════════════════════════════════════════════════════════
    #  REST Bootstrap (Gamma API)
    # ═══════════════════════════════════════════════════════════════════

    async def fetch_bootstrap_markets(
        self,
        deadline: Optional[float] = None,
        max_markets: int = 0,
    ) -> List[MarketEvent]:
        """
        Fetch currently active (open) markets from Polymarket Gamma API.

        Uses ``GET /markets`` with server-side filters so **every** returned
        market is actively trading — no wasted requests or bandwidth:

        - ``closed=false`` — excludes resolved / closed markets
        - ``active=true``  — only markets currently accepting trades

        Pagination is offset-based (Gamma returns a flat JSON array).
        Rate-limited to ~85 req/min to stay under the 100 req/min cap.

        Args:
            deadline: ``asyncio`` loop time after which to return partial
                results (same semantics as Kalshi bootstrap).
            max_markets: Cap on total markets returned (0 = unlimited).
                Use for development / testing to avoid bootstrapping
                thousands of markets.

        Returns:
            List of ``MarketEvent`` objects, one per active market.
        """
        import aiohttp

        url = f"{self._gamma_api_url}{_GAMMA_MARKETS_PATH}"
        events: List[MarketEvent] = []
        offset = 0
        page = 0
        timeout = aiohttp.ClientTimeout(connect=10, total=20)
        _REQUEST_TIMEOUT = 25.0

        logger.info(
            "[Polymarket] Bootstrap: starting REST fetch from %s "
            "(max_markets=%s)",
            self._gamma_api_url,
            max_markets or "unlimited",
        )

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                while True:
                    # ── Deadline check ────────────────────────────────
                    if deadline is not None:
                        now = asyncio.get_running_loop().time()
                        if now >= deadline:
                            logger.warning(
                                "[Polymarket] Bootstrap: deadline reached, "
                                "returning %d market(s) from %d page(s)",
                                len(events),
                                page,
                            )
                            break

                    # ── Market cap check ──────────────────────────────
                    if max_markets > 0 and len(events) >= max_markets:
                        logger.info(
                            "[Polymarket] Bootstrap: market cap (%d) reached",
                            max_markets,
                        )
                        break

                    page += 1
                    params = {
                        "closed": "false",
                        "active": "true",
                        "order": "volume24hr",
                        "ascending": "false",
                        "limit": str(_GAMMA_PAGE_SIZE),
                        "offset": str(offset),
                    }

                    logger.debug(
                        "[Polymarket] Bootstrap: page %d (offset=%d)",
                        page,
                        offset,
                    )

                    # ── HTTP request with per-page timeout ────────────
                    try:
                        async def _do_request():
                            async with session.get(
                                url, params=params
                            ) as resp:
                                if resp.status != 200:
                                    text = await resp.text()
                                    return resp.status, text, None
                                data = await resp.json()
                                return resp.status, None, data

                        status, err_text, data = await asyncio.wait_for(
                            _do_request(),
                            timeout=_REQUEST_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "[Polymarket] Bootstrap: page %d timed out "
                            "after %.0fs — returning %d market(s) so far",
                            page,
                            _REQUEST_TIMEOUT,
                            len(events),
                        )
                        break
                    except Exception as exc:
                        logger.warning(
                            "[Polymarket] Bootstrap: page %d request "
                            "failed: %s",
                            page,
                            exc,
                        )
                        break

                    if status != 200:
                        logger.warning(
                            "[Polymarket] Bootstrap: GET markets "
                            "status=%s: %s",
                            status,
                            (err_text or "")[:200],
                        )
                        break

                    # Gamma API returns a flat JSON array for /markets
                    markets_list: list = (
                        data
                        if isinstance(data, list)
                        else data.get("data", [])
                    )
                    if not markets_list:
                        break

                    page_accepted = 0
                    for m in markets_list:
                        # ── Defensive client-side filter ──────────────
                        # Server already filters, but protect against
                        # stale/inconsistent API responses.
                        if m.get("closed") is True:
                            continue
                        if m.get("active") is not True:
                            # "active" can be bool or stringified bool
                            active_val = m.get("active")
                            if not (
                                active_val is True
                                or str(active_val).lower() in ("true", "1")
                            ):
                                continue

                        # ── Market ID ─────────────────────────────────
                        market_id = (
                            m.get("condition_id")
                            or m.get("id")
                            or ""
                        )
                        if not market_id:
                            continue

                        # ── Title (required for canonicalization) ──────
                        title = m.get("question") or ""
                        if not title:
                            continue

                        # ── End date ──────────────────────────────────
                        end_date = None
                        end_date_str = (
                            m.get("endDate")
                            or m.get("end_date_iso")
                        )
                        if end_date_str and isinstance(end_date_str, str):
                            try:
                                end_date = datetime.fromisoformat(
                                    end_date_str.replace("Z", "+00:00")
                                )
                            except (ValueError, TypeError):
                                pass

                        # ── Outcomes ──────────────────────────────────
                        outcomes = self._parse_outcomes(m)

                        events.append(
                            MarketEvent(
                                venue=VenueType.POLYMARKET,
                                venue_market_id=str(market_id),
                                event_type=EventType.UPDATED,
                                title=title,
                                description=m.get("description"),
                                resolution_criteria=m.get(
                                    "resolutionSource"
                                ),
                                end_date=end_date,
                                outcomes=outcomes,
                                raw_payload=m,
                                received_at=datetime.now(UTC),
                            )
                        )
                        page_accepted += 1

                        # Cap check inside inner loop for early exit
                        if max_markets > 0 and len(events) >= max_markets:
                            break

                    logger.debug(
                        "[Polymarket] Bootstrap: page %d → %d/%d "
                        "markets accepted (total so far: %d)",
                        page,
                        page_accepted,
                        len(markets_list),
                        len(events),
                    )

                    # ── Pagination: next page or done ─────────────────
                    if len(markets_list) < _GAMMA_PAGE_SIZE:
                        # Last page (fewer results than requested)
                        break
                    offset += _GAMMA_PAGE_SIZE

                    # ── Rate limiting ─────────────────────────────────
                    await asyncio.sleep(_GAMMA_RATE_LIMIT_DELAY)

        except asyncio.CancelledError:
            logger.info(
                "[Polymarket] Bootstrap cancelled, returning %d market(s)",
                len(events),
            )
        except Exception as exc:
            logger.warning(
                "[Polymarket] Bootstrap failed: %s", exc, exc_info=True
            )

        logger.info(
            "[Polymarket] Bootstrap: fetched %d active market(s) "
            "in %d page(s)",
            len(events),
            page,
        )
        return events

    # ── Outcome parsing helper ───────────────────────────────────────

    @staticmethod
    def _parse_outcomes(market_data: dict) -> List[OutcomeSpec]:
        """
        Parse Polymarket outcomes from Gamma API response.

        The Gamma API returns outcomes as a JSON-encoded string array
        (e.g. ``'["Yes","No"]'``) or a native list.  Token IDs are in
        ``clobTokenIds`` as a parallel array.

        Args:
            market_data: Raw market dict from Gamma API.

        Returns:
            List of ``OutcomeSpec`` with label + token ID.
        """
        outcome_names = market_data.get("outcomes") or []
        token_ids = market_data.get("clobTokenIds") or []

        # Gamma sometimes returns outcomes as a JSON string
        if isinstance(outcome_names, str):
            try:
                outcome_names = json.loads(outcome_names)
            except (json.JSONDecodeError, TypeError):
                outcome_names = []

        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except (json.JSONDecodeError, TypeError):
                token_ids = []

        if not isinstance(outcome_names, list):
            return []

        specs: List[OutcomeSpec] = []
        for idx, name in enumerate(outcome_names):
            tok_id = token_ids[idx] if idx < len(token_ids) else ""
            specs.append(
                OutcomeSpec(
                    outcome_id=str(tok_id),
                    label=str(name),
                )
            )
        return specs

    # ═══════════════════════════════════════════════════════════════════
    #  WebSocket (real-time streaming)
    # ═══════════════════════════════════════════════════════════════════

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
