"""
Kalshi venue connector for market discovery.

WebSocket-only market discovery from Kalshi.
Handles connection, reconnection, and event normalization.

Kalshi requires authentication for WebSocket connections. Set KALSHI_API_KEY_ID
and KALSHI_PRIVATE_KEY_PATH (or KALSHI_PRIVATE_KEY PEM string) in config.

Docs:
  API keys & signing: https://docs.kalshi.com/getting_started/api_keys
  WebSocket connection: https://docs.kalshi.com/websockets/websocket-connection
  Quick start WebSockets: https://docs.kalshi.com/getting_started/quick_start_websockets
"""

import asyncio
import base64
import json
import logging
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from .base_connector import BaseVenueConnector
from .types import VenueType, MarketEvent, EventType, OutcomeSpec


logger = logging.getLogger(__name__)

# Path to sign for WebSocket handshake (per Quick Start: timestamp + "GET" + "/trade-api/ws/v2")
KALSHI_WS_SIGN_PATH = "/trade-api/ws/v2"
# REST path for listing markets (sign path without query)
KALSHI_MARKETS_PATH = "/trade-api/v2/markets"


def _load_kalshi_private_key(path: Optional[str] = None, pem: Optional[str] = None):
    """Load RSA private key from file path or PEM string. Returns None on failure."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
        if path:
            path = path.strip().replace("\\", "/")
            key_path = Path(path).expanduser().resolve()
            logger.debug("[Kalshi] Trying to load private key from: %s", key_path)
            if not key_path.exists():
                logger.warning(
                    "[Kalshi] Private key path does not exist: %s (check KALSHI_PRIVATE_KEY_PATH)",
                    key_path,
                )
                return None
            if not key_path.is_file():
                logger.warning("[Kalshi] Private key path is not a file: %s", key_path)
                return None
            with open(key_path, "rb") as f:
                data = f.read()
            if not data.strip():
                logger.warning("[Kalshi] Private key file is empty: %s", key_path)
                return None
            return serialization.load_pem_private_key(
                data, password=None, backend=default_backend()
            )
        if pem:
            pem = pem.strip()
            if isinstance(pem, str) and "\\n" in pem:
                pem = pem.replace("\\n", "\n")
            if not pem:
                logger.warning("[Kalshi] KALSHI_PRIVATE_KEY is empty")
                return None
            return serialization.load_pem_private_key(
                pem.encode() if isinstance(pem, str) else pem,
                password=None,
                backend=default_backend(),
            )
    except FileNotFoundError as e:
        logger.warning("[Kalshi] Private key file not found: %s", e)
    except ValueError as e:
        logger.warning(
            "[Kalshi] Private key format invalid (bad PEM or encrypted key?): %s",
            e,
        )
    except Exception as e:
        logger.warning(
            "[Kalshi] Failed to load private key: %s (%s)",
            e,
            type(e).__name__,
            exc_info=True,
        )
    return None


def _sign_kalshi_request(private_key, method: str, path: str) -> Optional[tuple]:
    """Sign message for Kalshi API: timestamp + method + path (RSA-PSS SHA256, base64).
    Path must be without query params (per https://docs.kalshi.com/getting_started/api_keys).
    Returns (signature_b64, timestamp_ms) or None."""
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        timestamp_ms = str(int(time.time() * 1000))
        path_without_query = path.split("?")[0]
        message = timestamp_ms + method + path_without_query
        signature = private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("ascii"), timestamp_ms
    except Exception as e:
        logger.warning("[Kalshi] Signing failed: %s", e)
        return None


class KalshiConnector(BaseVenueConnector):
    """Kalshi WebSocket connector for market discovery."""
    
    def __init__(
        self,
        ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2",
        api_key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        private_key_pem: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Kalshi connector.
        
        Args:
            ws_url: Kalshi WebSocket URL
            api_key_id: Kalshi API key ID (required for auth)
            private_key_path: Path to PEM file (optional if private_key_pem set)
            private_key_pem: PEM string of private key (optional if private_key_path set)
        """
        super().__init__(
            venue_name=VenueType.KALSHI,
            ws_url=ws_url,
            **kwargs
        )
        self._api_key_id = api_key_id
        self._private_key_path = private_key_path
        self._private_key_pem = private_key_pem
        self._private_key = None
        if api_key_id and (private_key_path or private_key_pem):
            self._private_key = _load_kalshi_private_key(
                path=private_key_path.strip() if private_key_path else None,
                pem=private_key_pem.strip() if private_key_pem else None,
            )
            if not self._private_key:
                logger.warning(
                    "[Kalshi] API key ID set but private key could not be loaded; "
                    "WebSocket connection will likely get HTTP 401"
                )
            else:
                logger.info("[Kalshi] Auth enabled (API key + private key loaded)")

    def get_connection_headers(self) -> Optional[dict]:
        """Kalshi requires KALSHI-ACCESS-KEY, KALSHI-ACCESS-SIGNATURE, KALSHI-ACCESS-TIMESTAMP on handshake (see docs)."""
        if not self._api_key_id or not self._private_key:
            return None
        result = _sign_kalshi_request(self._private_key, "GET", KALSHI_WS_SIGN_PATH)
        if not result:
            return None
        signature_b64, timestamp_ms = result
        headers = {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }
        logger.info(
            "[Kalshi] Sending auth headers (key id ...%s)",
            self._api_key_id[-4:] if len(self._api_key_id or "") >= 4 else "****",
        )
        return headers

    def _rest_base_url(self) -> str:
        """Derive REST base URL from WebSocket URL (wss://host/path -> https://host)."""
        parsed = urlparse(self.ws_url)
        netloc = parsed.netloc or parsed.path.split("/")[0]
        return f"https://{netloc}"

    def _auth_headers_for_path(self, method: str, path: str) -> Optional[dict]:
        """Build KALSHI-ACCESS-* headers for a REST request (path without query)."""
        if not self._api_key_id or not self._private_key:
            return None
        result = _sign_kalshi_request(self._private_key, method, path)
        if not result:
            return None
        sig_b64, timestamp_ms = result
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }

    async def fetch_bootstrap_markets(
        self,
        deadline: Optional[float] = None,
        max_markets: int = 0,
    ) -> List[MarketEvent]:
        """
        Fetch currently active markets via REST.

        Uses ``GET /trade-api/v2/markets?status=open&limit=1000`` with
        cursor pagination.  The Kalshi API accepts ``status=open`` as
        the query filter, but the response ``status`` field returns
        ``"active"`` for tradeable markets.  We query with
        ``status=open`` and accept both ``"active"`` and ``"open"``
        client-side for safety.

        Args:
            deadline: ``asyncio`` loop time after which to return partial
                results (prevents the orchestrator from losing all
                bootstrap data on timeout).
            max_markets: Cap on total markets returned (0 = unlimited).
                Useful for dev/testing to limit bootstrap scope.

        Returns:
            List of ``MarketEvent`` objects, one per active market.
        """
        import aiohttp

        base = self._rest_base_url()
        path = KALSHI_MARKETS_PATH
        url = f"{base}{path}"
        events: List[MarketEvent] = []
        cursor: Optional[str] = None
        page = 0
        # Per-request: connect 10s, total 20s; outer asyncio.wait_for 25s so we never hang
        timeout = aiohttp.ClientTimeout(connect=10, total=20)
        _REQUEST_TIMEOUT = 25.0

        logger.info("[Kalshi] Bootstrap: starting REST fetch %s%s", base, path)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                while True:
                    if deadline is not None:
                        now = asyncio.get_running_loop().time()
                        if now >= deadline:
                            logger.warning(
                                "[Kalshi] Bootstrap: deadline reached, "
                                "returning %d market(s) from %d page(s)",
                                len(events),
                                page,
                            )
                            break
                    if max_markets > 0 and len(events) >= max_markets:
                        logger.info(
                            "[Kalshi] Bootstrap: market cap (%d) reached",
                            max_markets,
                        )
                        break
                    page += 1
                    params: dict = {"status": "open", "limit": 1000}
                    if cursor:
                        params["cursor"] = cursor
                    headers = self._auth_headers_for_path("GET", path) or {}

                    logger.debug(
                        "[Kalshi] Bootstrap: page %d request (cursor=%s)",
                        page,
                        cursor[:20] + "..." if cursor and len(cursor) > 20 else cursor,
                    )

                    try:
                        async def _do_request():
                            async with session.get(url, params=params, headers=headers or None) as resp:
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
                            "[Kalshi] Bootstrap: page %d timed out after %.0fs — returning %d market(s) so far",
                            page,
                            _REQUEST_TIMEOUT,
                            len(events),
                        )
                        break
                    except Exception as e:
                        logger.warning("[Kalshi] Bootstrap request failed (page %d): %s", page, e)
                        break

                    if status != 200:
                        logger.warning(
                            "[Kalshi] Bootstrap GET markets status=%s: %s",
                            status,
                            (err_text or "")[:200],
                        )
                        break

                    assert data is not None
                    markets = data.get("markets") or []
                    # Kalshi API docs list query param as "open" but the
                    # response status field returns "active" for tradeable
                    # markets.  Accept both to be safe.
                    _ACTIVE_STATUSES = {"active", "open"}
                    active_count = sum(
                        1 for m in markets
                        if (m.get("status") or "").lower() in _ACTIVE_STATUSES
                    )

                    logger.debug(
                        "[Kalshi] Bootstrap: page %d got %d markets (%d active), cursor=%s",
                        page,
                        len(markets),
                        active_count,
                        "yes" if data.get("cursor") else "no",
                    )

                    for m in markets:
                        # Only include active markets; skip closed/settled/unopened
                        if (m.get("status") or "").lower() not in _ACTIVE_STATUSES:
                            continue
                        ticker = m.get("ticker") or ""
                        if not ticker:
                            continue
                        title = m.get("title") or m.get("subtitle") or ticker
                        description = m.get("rules_primary") or m.get("rules_secondary")
                        end_date = None
                        for key in ("close_time", "expiration_time", "expected_expiration_time"):
                            raw = m.get(key)
                            if not raw:
                                continue
                            try:
                                if isinstance(raw, str):
                                    end_date = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                                elif isinstance(raw, (int, float)):
                                    end_date = datetime.fromtimestamp(int(raw), tz=UTC)
                                if end_date:
                                    break
                            except (ValueError, TypeError, OSError):
                                pass
                        events.append(
                            MarketEvent(
                                venue=VenueType.KALSHI,
                                venue_market_id=ticker,
                                event_type=EventType.UPDATED,
                                title=title,
                                description=description,
                                resolution_criteria=m.get("rules_primary"),
                                end_date=end_date,
                                outcomes=[
                                    OutcomeSpec(outcome_id="YES", label="Yes"),
                                    OutcomeSpec(outcome_id="NO", label="No"),
                                ],
                                raw_payload=m,
                                received_at=datetime.now(UTC),
                            )
                        )
                        if max_markets > 0 and len(events) >= max_markets:
                            break

                    cursor = data.get("cursor") if isinstance(data.get("cursor"), str) and data.get("cursor") else None
                    if not cursor or not markets:
                        break
        except asyncio.TimeoutError:
            logger.warning("[Kalshi] Bootstrap HTTP timeout")
        except Exception as e:
            logger.warning("[Kalshi] Bootstrap failed: %s", e)

        logger.info(
            "[Kalshi] Bootstrap: fetched %d active market(s) in %d page(s)",
            len(events),
            page,
        )
        return events

    def _build_subscription_message(self) -> Optional[dict]:
        """
        Build Kalshi WebSocket v2 subscription message.
        Uses cmd/subscribe and params.channels per docs.
        market_lifecycle_v2 streams all market/event lifecycle (created, activated, etc.).
        """
        return {
            "id": 1,
            "cmd": "subscribe",
            "params": {"channels": ["market_lifecycle_v2"]},
        }
    
    async def _parse_message(self, message: str) -> Optional[MarketEvent]:
        """
        Parse Kalshi WebSocket v2 message into MarketEvent.
        Handles: type=subscribed (ack), type=market_lifecycle_v2, type=event_lifecycle, type=error.
        See https://docs.kalshi.com/websockets/market-&-event-lifecycle
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            # Subscription confirmation
            if msg_type == "subscribed":
                logger.info("[Kalshi] Subscription confirmed: %s", data.get("msg"))
                return None

            # Server error (e.g. unknown command)
            if msg_type == "error":
                err = data.get("msg", {})
                logger.warning("[Kalshi] Server error: code=%s msg=%s", err.get("code"), err.get("msg"))
                return None

            # Market lifecycle: only emit for active/open markets (created, activated, close_date_updated)
            # Skip deactivated, determined, settled so we don't ingest closed/settled markets.
            if msg_type == "market_lifecycle_v2":
                msg = data.get("msg") or {}
                market_ticker = msg.get("market_ticker") or ""
                if not market_ticker:
                    return None
                event_type_str = (msg.get("event_type") or "").lower()
                if event_type_str in ("deactivated", "determined", "settled"):
                    logger.debug("[Kalshi] Skipping non-active market %s (event_type=%s)", market_ticker, event_type_str)
                    return None
                event_type = EventType.CREATED if event_type_str == "created" else EventType.UPDATED
                meta = msg.get("additional_metadata") or {}
                title = meta.get("title") or meta.get("name") or market_ticker
                description = meta.get("rules_primary") or meta.get("rules_secondary")
                end_date = None
                if msg.get("close_ts") is not None:
                    try:
                        end_date = datetime.fromtimestamp(int(msg["close_ts"]), tz=UTC)
                    except (ValueError, TypeError):
                        pass
                return MarketEvent(
                    venue=VenueType.KALSHI,
                    venue_market_id=market_ticker,
                    event_type=event_type,
                    title=title,
                    description=description,
                    resolution_criteria=meta.get("rules_primary"),
                    end_date=end_date,
                    outcomes=[OutcomeSpec(outcome_id="YES", label="Yes"), OutcomeSpec(outcome_id="NO", label="No")],
                    raw_payload=data,
                    received_at=datetime.now(UTC),
                )

            # Event lifecycle (new event created)
            if msg_type == "event_lifecycle":
                msg = data.get("msg") or {}
                event_ticker = msg.get("event_ticker") or ""
                if not event_ticker:
                    return None
                title = msg.get("title") or msg.get("subtitle") or event_ticker
                return MarketEvent(
                    venue=VenueType.KALSHI,
                    venue_market_id=event_ticker,
                    event_type=EventType.CREATED,
                    title=title,
                    description=msg.get("subtitle"),
                    outcomes=[OutcomeSpec(outcome_id="YES", label="Yes"), OutcomeSpec(outcome_id="NO", label="No")],
                    raw_payload=data,
                    received_at=datetime.now(UTC),
                )

            logger.debug("[Kalshi] Unhandled message type: %s", msg_type)
            return None

        except json.JSONDecodeError as e:
            logger.error("[Kalshi] Failed to parse JSON: %s", e)
            return None
        except Exception as e:
            logger.error("[Kalshi] Error parsing message: %s", e)
            return None
