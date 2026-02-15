"""
Cross-venue spread scanner for equivalent verified pairs.

Fetches live orderbooks from Polymarket (CLOB) and Kalshi for all
equivalent pairs and identifies arbitrage opportunities where
buying complementary outcomes (Yes + No) across venues costs less
than the guaranteed $1 payout, net of fees.

The correct arb is always:
    Buy YES on Venue A  +  Buy NO on Venue B
    (or vice versa)
This guarantees exactly one side pays $1 regardless of outcome.

Fee model:
    - Kalshi: ~7% of profit on the winning contract
    - Polymarket CLOB: 0% taker fee (current)

Sizing PnL curve:
    Walks both orderbooks simultaneously to compute guaranteed
    profit at each cumulative contract quantity.

Usage:
    python spread_scanner.py

Requires:
    - DATABASE_URL in .env (Supabase PostgreSQL)
    - KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY_PATH in .env (Kalshi REST auth)
    - Polymarket APIs are public (no auth)
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

import aiohttp
import asyncpg

# ── Project imports ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from discovery.kalshi_poller import (
    _load_kalshi_private_key,
    _sign_kalshi_request,
)


# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("spread_scanner")


# ── Fee constants ────────────────────────────────────────────────
# Kalshi charges ~7% of profit on the winning side of a contract.
# If you buy at price P and the contract resolves to $1,
# fee = KALSHI_FEE_RATE * (1 - P).
KALSHI_FEE_RATE = 0.070

# Polymarket CLOB: 0% taker fee (as of 2025/2026).
POLYMARKET_FEE_RATE = 0.000

VENUE_FEES: dict[str, float] = {
    "kalshi": KALSHI_FEE_RATE,
    "polymarket": POLYMARKET_FEE_RATE,
}


# ── API constants ────────────────────────────────────────────────
# NOTE: Despite the "elections" in the hostname, api.elections.kalshi.com
# is Kalshi's ONLY production REST API and serves ALL market categories
# (sports, crypto, entertainment, politics, etc.).  Kalshi explicitly
# rejects trading-api.kalshi.com with a 401 redirect here.
KALSHI_REST_BASE = (
    "https://demo-api.kalshi.co"
    if config.KALSHI_USE_DEMO
    else "https://api.elections.kalshi.com"
)
KALSHI_MARKETS_PATH = "/trade-api/v2/markets"
GAMMA_API_BASE = config.POLYMARKET_GAMMA_API_URL or "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

# Rate-limit semaphores
_KALSHI_SEM = asyncio.Semaphore(5)
_POLY_SEM = asyncio.Semaphore(10)

# PnL curve sample points
PNL_CURVE_STEPS = 20


# ═══════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BookLevel:
    """Single price level in an orderbook."""
    price: float   # [0, 1]
    size: float    # number of contracts


@dataclass
class VenueBook:
    """
    Full orderbook for one venue's binary market.

    Contains ask levels for both Yes and No outcomes, which is
    all we need for the arb (we only buy, never sell).
    """
    venue: str
    venue_market_id: str
    yes_asks: list[BookLevel] = field(default_factory=list)  # ascending price
    yes_bids: list[BookLevel] = field(default_factory=list)  # descending price
    no_asks: list[BookLevel] = field(default_factory=list)   # ascending price
    no_bids: list[BookLevel] = field(default_factory=list)   # descending price
    source: str = "unknown"  # "orderbook", "top_of_book", "gamma_mid"
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def yes_ask_top(self) -> Optional[float]:
        """Best (lowest) ask to buy Yes."""
        return self.yes_asks[0].price if self.yes_asks else None

    @property
    def no_ask_top(self) -> Optional[float]:
        """Best (lowest) ask to buy No."""
        return self.no_asks[0].price if self.no_asks else None

    @property
    def yes_depth(self) -> float:
        """Total contracts available on Yes asks."""
        return sum(lvl.size for lvl in self.yes_asks)

    @property
    def no_depth(self) -> float:
        """Total contracts available on No asks."""
        return sum(lvl.size for lvl in self.no_asks)


@dataclass
class PnLPoint:
    """One point on the sizing PnL curve."""
    quantity: float           # contracts
    total_cost: float         # USD (sum of both legs)
    profit_if_a_wins: float   # net profit if side-A outcome wins
    profit_if_b_wins: float   # net profit if side-B outcome wins
    guaranteed_profit: float  # min of the two
    roi_pct: float            # guaranteed_profit / total_cost * 100


@dataclass
class ArbOpportunity:
    """A detected cross-venue arbitrage opportunity with PnL curve."""
    pair_key: str
    title_a: str
    title_b: str
    buy_a_side: str       # "yes" or "no" — what we buy on venue A
    buy_b_side: str       # "yes" or "no" — what we buy on venue B
    venue_a: str
    venue_b: str
    vmid_a: str
    vmid_b: str
    ask_a: float          # top-of-book ask for our side on A
    ask_b: float          # top-of-book ask for our side on B
    total_cost_1: float   # cost per contract (top of book)
    gross_edge: float     # 1.0 - total_cost_1 (before fees)
    fee_a: float          # fee rate on venue A
    fee_b: float          # fee rate on venue B
    net_profit_1: float   # guaranteed profit per contract (after fees)
    net_roi_pct: float    # net_profit_1 / total_cost_1 * 100
    pnl_curve: list[PnLPoint] = field(default_factory=list)
    optimal_qty: float = 0.0
    max_profit: float = 0.0
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class EquivalentPair:
    """An equivalent pair from the database."""
    pair_key: str
    venue_a: str
    vmid_a: str
    title_a: str
    venue_b: str
    vmid_b: str
    title_b: str
    outcome_mapping: dict
    confidence: float


# ═══════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════

async def fetch_equivalent_pairs(dsn: str) -> list[EquivalentPair]:
    """
    Fetch all current equivalent pairs from the database.

    Joins verified_pairs with markets to get venue + venue_market_id
    for both sides.

    Args:
        dsn: PostgreSQL connection string.

    Returns:
        List of EquivalentPair objects.
    """
    conn = await asyncpg.connect(dsn=dsn, statement_cache_size=0)
    try:
        rows = await conn.fetch(
            """
            SELECT
                vp.pair_key,
                vp.outcome_mapping,
                vp.confidence_score,
                ma.venue   AS venue_a,
                ma.venue_market_id AS vmid_a,
                ma.title   AS title_a,
                mb.venue   AS venue_b,
                mb.venue_market_id AS vmid_b,
                mb.title   AS title_b
            FROM verified_pairs vp
            JOIN markets ma ON ma.id = vp.market_a_id
            JOIN markets mb ON mb.id = vp.market_b_id
            WHERE vp.verdict = 'equivalent'
              AND vp.is_current = true
              AND vp.is_active  = true
            ORDER BY vp.confidence_score DESC
            """
        )

        pairs: list[EquivalentPair] = []
        for r in rows:
            outcome_raw = r["outcome_mapping"]
            if isinstance(outcome_raw, str):
                outcome_map = json.loads(outcome_raw)
            elif isinstance(outcome_raw, dict):
                outcome_map = outcome_raw
            else:
                outcome_map = {}

            pairs.append(EquivalentPair(
                pair_key=r["pair_key"],
                venue_a=r["venue_a"],
                vmid_a=r["vmid_a"],
                title_a=r["title_a"] or "",
                venue_b=r["venue_b"],
                vmid_b=r["vmid_b"],
                title_b=r["title_b"] or "",
                outcome_mapping=outcome_map,
                confidence=float(r["confidence_score"] or 0),
            ))

        return pairs
    finally:
        await conn.close()


# ═══════════════════════════════════════════════════════════════════
# KALSHI FETCHER
# ═══════════════════════════════════════════════════════════════════

def _build_kalshi_auth() -> tuple:
    """
    Load Kalshi private key and return (api_key_id, private_key).

    Returns:
        Tuple of (api_key_id, private_key_object) or (None, None).
    """
    key_id = config.KALSHI_API_KEY_ID
    if not key_id:
        return None, None

    pk = _load_kalshi_private_key(
        path=config.KALSHI_PRIVATE_KEY_PATH,
        pem=config.KALSHI_PRIVATE_KEY,
    )
    return key_id, pk


def _kalshi_headers(
    api_key_id: Optional[str],
    private_key,
    method: str,
    path: str,
) -> dict:
    """Build authenticated headers for a Kalshi REST request."""
    headers: dict = {"Accept": "application/json"}
    if api_key_id and private_key:
        result = _sign_kalshi_request(private_key, method, path)
        if result:
            sig_b64, ts_ms = result
            headers["KALSHI-ACCESS-KEY"] = api_key_id
            headers["KALSHI-ACCESS-SIGNATURE"] = sig_b64
            headers["KALSHI-ACCESS-TIMESTAMP"] = ts_ms
    return headers


async def fetch_kalshi_book(
    session: aiohttp.ClientSession,
    ticker: str,
    api_key_id: Optional[str],
    private_key,
) -> Optional[VenueBook]:
    """
    Fetch orderbook for a Kalshi market.

    Always fetches the market endpoint first for reliable top-of-book
    (yes_bid, yes_ask, no_bid, no_ask).  Then tries the full orderbook
    endpoint for depth.  The orderbook is validated against the market
    endpoint's top-of-book before being used.

    Kalshi prices are in cents (1-99); normalised to [0, 1].

    Args:
        session: aiohttp session.
        ticker: Kalshi market ticker.
        api_key_id: Kalshi API key ID.
        private_key: Loaded RSA private key.

    Returns:
        VenueBook or None on failure.
    """
    # ── 1. Always fetch the market endpoint for reliable prices ──
    tob = await _fetch_kalshi_market_data(session, ticker, api_key_id, private_key)
    if tob is None:
        return None

    # ── 2. Try orderbook endpoint for depth ──────────────────────
    depth_book = await _fetch_kalshi_orderbook_depth(
        session, ticker, api_key_id, private_key, tob,
    )
    if depth_book is not None:
        return depth_book

    # ── 3. Fallback: build book from top-of-book only ────────────
    return _build_kalshi_tob_book(ticker, tob)


@dataclass
class _KalshiTopOfBook:
    """Parsed Kalshi market endpoint data (top-of-book)."""
    yes_bid: Optional[float]
    yes_ask: Optional[float]
    no_bid: Optional[float]
    no_ask: Optional[float]
    volume: float


async def _fetch_kalshi_market_data(
    session: aiohttp.ClientSession,
    ticker: str,
    api_key_id: Optional[str],
    private_key,
) -> Optional[_KalshiTopOfBook]:
    """
    Fetch ``GET /trade-api/v2/markets/{ticker}`` for reliable top-of-book.

    Returns:
        _KalshiTopOfBook or None on failure.
    """
    path = f"{KALSHI_MARKETS_PATH}/{ticker}"
    url = f"{KALSHI_REST_BASE}{path}"
    headers = _kalshi_headers(api_key_id, private_key, "GET", path)

    async with _KALSHI_SEM:
        try:
            async with session.get(
                url, headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "[Kalshi] %s -> %d: %s", ticker, resp.status, body[:200],
                    )
                    return None
                data = await resp.json()
        except Exception as exc:
            logger.warning("[Kalshi] Failed to fetch %s: %s", ticker, exc)
            return None

    market = data.get("market") or data
    yes_bid_raw = market.get("yes_bid")
    yes_ask_raw = market.get("yes_ask")
    no_bid_raw = market.get("no_bid")
    no_ask_raw = market.get("no_ask")

    if yes_bid_raw is None and yes_ask_raw is None:
        logger.debug("[Kalshi] No prices for %s", ticker)
        return None

    # Kalshi prices are in cents → [0, 1]
    yes_bid = int(yes_bid_raw) / 100.0 if yes_bid_raw is not None else None
    yes_ask = int(yes_ask_raw) / 100.0 if yes_ask_raw is not None else None

    # Derive No from Yes if not directly available
    if no_bid_raw is not None:
        no_bid = int(no_bid_raw) / 100.0
    elif yes_ask is not None:
        no_bid = round(1.0 - yes_ask, 4)
    else:
        no_bid = None

    if no_ask_raw is not None:
        no_ask = int(no_ask_raw) / 100.0
    elif yes_bid is not None:
        no_ask = round(1.0 - yes_bid, 4)
    else:
        no_ask = None

    volume = float(market.get("volume", 0) or 0)

    return _KalshiTopOfBook(
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        volume=volume,
    )


async def _fetch_kalshi_orderbook_depth(
    session: aiohttp.ClientSession,
    ticker: str,
    api_key_id: Optional[str],
    private_key,
    tob: _KalshiTopOfBook,
) -> Optional[VenueBook]:
    """
    Fetch full depth from ``GET /trade-api/v2/markets/{ticker}/orderbook``.

    Kalshi's orderbook response contains **resting buy orders** (bids)
    for each outcome::

        {
          "orderbook": {
            "yes": [[price_cents, qty], ...],   ← YES bids
            "no":  [[price_cents, qty], ...]    ← NO bids
          }
        }

    In a binary market, buying one outcome = selling the other:
        - YES ask at price P  ←  NO bid at price (100 - P)
        - NO  ask at price P  ←  YES bid at price (100 - P)

    So we derive executable asks by inverting the opposite side's bids.

    The derived asks are validated against the market endpoint's
    top-of-book to ensure correct interpretation.

    Args:
        session: aiohttp session.
        ticker: Kalshi market ticker.
        api_key_id: API key.
        private_key: RSA key.
        tob: Reliable top-of-book from the market endpoint.

    Returns:
        VenueBook or None if orderbook unavailable or inconsistent.
    """
    path = f"{KALSHI_MARKETS_PATH}/{ticker}/orderbook"
    url = f"{KALSHI_REST_BASE}{path}"
    headers = _kalshi_headers(api_key_id, private_key, "GET", path)

    async with _KALSHI_SEM:
        try:
            async with session.get(
                url, headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.debug(
                        "[Kalshi] Orderbook %s -> %d", ticker, resp.status,
                    )
                    return None
                data = await resp.json()
        except Exception as exc:
            logger.debug("[Kalshi] Orderbook fetch failed %s: %s", ticker, exc)
            return None

    ob = data.get("orderbook") or data
    yes_raw = ob.get("yes") or []
    no_raw = ob.get("no") or []

    if not yes_raw and not no_raw:
        return None

    # ── Parse raw levels ─────────────────────────────────────────
    def _parse_raw(raw: list) -> list[BookLevel]:
        """Parse [[price_cents, qty], ...] into BookLevel list (ascending)."""
        levels: list[BookLevel] = []
        for entry in raw:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                price = float(entry[0]) / 100.0
                size = float(entry[1])
                if 0 < price < 1 and size > 0:
                    levels.append(BookLevel(price=price, size=size))
        levels.sort(key=lambda l: l.price)
        return levels

    yes_levels = _parse_raw(yes_raw)   # YES bids (ascending)
    no_levels = _parse_raw(no_raw)     # NO bids (ascending)

    # ── Derive ASKS by inverting opposite-side BIDS ──────────────
    # YES asks ← inverted NO bids  (to buy Yes, match a No buyer)
    # NO  asks ← inverted YES bids (to buy No, match a Yes buyer)
    def _invert_to_asks(bids: list[BookLevel]) -> list[BookLevel]:
        """Convert bids into asks: ask_price = 1 - bid_price, ascending."""
        asks = [
            BookLevel(price=round(1.0 - lvl.price, 6), size=lvl.size)
            for lvl in bids
            if 0 < round(1.0 - lvl.price, 6) < 1
        ]
        asks.sort(key=lambda l: l.price)  # ascending (cheapest first)
        return asks

    yes_asks = _invert_to_asks(no_levels)
    no_asks = _invert_to_asks(yes_levels)

    # ── Sanity check: derived top-of-book vs market endpoint ─────
    # The derived yes_ask should be close to the market endpoint's yes_ask
    if tob.yes_ask is not None and yes_asks:
        delta = abs(yes_asks[0].price - tob.yes_ask)
        if delta > 0.05:
            logger.warning(
                "[Kalshi] Orderbook yes_ask (%.4f) disagrees with market "
                "endpoint yes_ask (%.4f) for %s — falling back to top-of-book",
                yes_asks[0].price, tob.yes_ask, ticker,
            )
            return None

    if tob.no_ask is not None and no_asks:
        delta = abs(no_asks[0].price - tob.no_ask)
        if delta > 0.05:
            logger.warning(
                "[Kalshi] Orderbook no_ask (%.4f) disagrees with market "
                "endpoint no_ask (%.4f) for %s — falling back to top-of-book",
                no_asks[0].price, tob.no_ask, ticker,
            )
            return None

    # YES/NO bids directly from the raw data (descending for bids)
    yes_bids = list(reversed(yes_levels))  # descending (best bid first)
    no_bids = list(reversed(no_levels))

    return VenueBook(
        venue="kalshi",
        venue_market_id=ticker,
        yes_asks=yes_asks,
        yes_bids=yes_bids,
        no_asks=no_asks,
        no_bids=no_bids,
        source="orderbook",
    )


def _build_kalshi_tob_book(
    ticker: str,
    tob: _KalshiTopOfBook,
) -> Optional[VenueBook]:
    """Build a VenueBook from top-of-book data (single-level depth)."""
    # Conservative depth estimate from volume
    depth_estimate = max(100.0, min(tob.volume * 0.01, 10000.0))

    yes_asks = [BookLevel(price=tob.yes_ask, size=depth_estimate)] if tob.yes_ask else []
    yes_bids = [BookLevel(price=tob.yes_bid, size=depth_estimate)] if tob.yes_bid else []
    no_asks = [BookLevel(price=tob.no_ask, size=depth_estimate)] if tob.no_ask else []
    no_bids = [BookLevel(price=tob.no_bid, size=depth_estimate)] if tob.no_bid else []

    return VenueBook(
        venue="kalshi",
        venue_market_id=ticker,
        yes_asks=yes_asks,
        yes_bids=yes_bids,
        no_asks=no_asks,
        no_bids=no_bids,
        source="top_of_book",
    )


# ═══════════════════════════════════════════════════════════════════
# POLYMARKET FETCHER
# ═══════════════════════════════════════════════════════════════════

async def _fetch_gamma_tokens(
    session: aiohttp.ClientSession,
    market_id: str,
) -> tuple[Optional[str], Optional[str], Optional[list[float]]]:
    """
    Fetch Polymarket market data (token IDs + indicative prices) via Gamma API.

    Uses ``GET /markets/{id}`` which works for both numeric Gamma IDs
    and condition_id hex strings.

    Returns:
        Tuple of (yes_token_id, no_token_id, outcome_prices) or (None, None, None).
    """
    url = f"{GAMMA_API_BASE}/markets/{market_id}"
    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                if isinstance(data, dict):
                    return _parse_gamma_market(data)
    except Exception as exc:
        logger.debug("[Polymarket] Gamma /%s failed: %s", market_id[:20], exc)

    # Fallback: GET /markets?id={id}
    url2 = f"{GAMMA_API_BASE}/markets"
    try:
        async with session.get(
            url2, params={"id": market_id},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                if isinstance(data, list) and data:
                    return _parse_gamma_market(data[0])
    except Exception as exc:
        logger.debug("[Polymarket] Gamma ?id=%s failed: %s", market_id[:20], exc)

    return None, None, None


def _parse_gamma_market(
    market: dict,
) -> tuple[Optional[str], Optional[str], Optional[list[float]]]:
    """Extract token IDs and outcome prices from a Gamma API market dict."""
    token_ids_raw = market.get("clobTokenIds") or []
    if isinstance(token_ids_raw, str):
        try:
            token_ids_raw = json.loads(token_ids_raw)
        except (json.JSONDecodeError, TypeError):
            token_ids_raw = []

    yes_token = token_ids_raw[0] if len(token_ids_raw) > 0 else None
    no_token = token_ids_raw[1] if len(token_ids_raw) > 1 else None

    prices_raw = market.get("outcomePrices") or []
    if isinstance(prices_raw, str):
        try:
            prices_raw = json.loads(prices_raw)
        except (json.JSONDecodeError, TypeError):
            prices_raw = []

    prices: list[float] = []
    for p in prices_raw:
        try:
            prices.append(float(p))
        except (ValueError, TypeError):
            pass

    return yes_token, no_token, prices if prices else None


async def _fetch_clob_book_full(
    session: aiohttp.ClientSession,
    token_id: str,
) -> tuple[list[BookLevel], list[BookLevel]]:
    """
    Fetch full orderbook from Polymarket CLOB for a single token.

    Returns:
        Tuple of (bids_descending, asks_ascending) as BookLevel lists.
    """
    url = f"{CLOB_API_BASE}/book"
    params = {"token_id": token_id}

    try:
        async with session.get(
            url, params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return [], []
            data = await resp.json()
    except Exception as exc:
        logger.debug("[Polymarket] CLOB book failed for %s: %s", token_id[:20], exc)
        return [], []

    raw_bids = data.get("bids") or []
    raw_asks = data.get("asks") or []

    bids: list[BookLevel] = []
    for entry in raw_bids:
        try:
            price = float(entry.get("price", 0))
            size = float(entry.get("size", 0))
            if 0 < price < 1 and size > 0:
                bids.append(BookLevel(price=price, size=size))
        except (ValueError, TypeError, AttributeError):
            continue
    bids.sort(key=lambda l: l.price, reverse=True)  # descending

    asks: list[BookLevel] = []
    for entry in raw_asks:
        try:
            price = float(entry.get("price", 0))
            size = float(entry.get("size", 0))
            if 0 < price < 1 and size > 0:
                asks.append(BookLevel(price=price, size=size))
        except (ValueError, TypeError, AttributeError):
            continue
    asks.sort(key=lambda l: l.price)  # ascending

    return bids, asks


async def fetch_polymarket_book(
    session: aiohttp.ClientSession,
    market_id: str,
) -> Optional[VenueBook]:
    """
    Fetch orderbook for a Polymarket market.

    Strategy:
    1. Gamma API → get clobTokenIds (Yes token + No token)
    2. CLOB /book → full orderbook for BOTH Yes and No tokens
    3. Fall back to Gamma mid-prices if CLOB unavailable

    Args:
        session: aiohttp session.
        market_id: Polymarket venue_market_id.

    Returns:
        VenueBook or None.
    """
    async with _POLY_SEM:
        yes_token, no_token, gamma_prices = await _fetch_gamma_tokens(
            session, market_id,
        )

        yes_bids: list[BookLevel] = []
        yes_asks: list[BookLevel] = []
        no_bids: list[BookLevel] = []
        no_asks: list[BookLevel] = []
        source = "gamma_mid"

        # ── Fetch Yes CLOB book ──────────────────────────────────
        if yes_token:
            y_bids, y_asks = await _fetch_clob_book_full(session, yes_token)
            if y_asks:
                yes_asks = y_asks
                yes_bids = y_bids
                source = "orderbook"

        # ── Fetch No CLOB book ───────────────────────────────────
        if no_token:
            n_bids, n_asks = await _fetch_clob_book_full(session, no_token)
            if n_asks:
                no_asks = n_asks
                no_bids = n_bids
                source = "orderbook"

        # ── Fallback: derive from Gamma mid-prices ───────────────
        if not yes_asks and gamma_prices and len(gamma_prices) >= 1:
            mid = gamma_prices[0]
            if mid is not None and 0 < mid < 1:
                yes_asks = [BookLevel(price=mid, size=100.0)]
                yes_bids = [BookLevel(price=mid, size=100.0)]
                source = "gamma_mid"

        if not no_asks and gamma_prices and len(gamma_prices) >= 2:
            mid = gamma_prices[1]
            if mid is not None and 0 < mid < 1:
                no_asks = [BookLevel(price=mid, size=100.0)]
                no_bids = [BookLevel(price=mid, size=100.0)]
                source = "gamma_mid" if source != "orderbook" else source

        # If we still have no No book, derive from Yes bids:
        # no_ask = 1 - yes_bid (synthetic, less reliable for Polymarket)
        if not no_asks and yes_bids:
            no_asks = [
                BookLevel(
                    price=round(1.0 - lvl.price, 6),
                    size=lvl.size,
                )
                for lvl in reversed(yes_bids)  # cheapest no_ask first
            ]
            no_asks.sort(key=lambda l: l.price)
            if source == "orderbook":
                source = "orderbook_partial"

        # If we still have no Yes book, derive from No bids:
        if not yes_asks and no_bids:
            yes_asks = [
                BookLevel(
                    price=round(1.0 - lvl.price, 6),
                    size=lvl.size,
                )
                for lvl in reversed(no_bids)
            ]
            yes_asks.sort(key=lambda l: l.price)
            if source == "orderbook":
                source = "orderbook_partial"

        if not yes_asks and not no_asks:
            return None

        return VenueBook(
            venue="polymarket",
            venue_market_id=market_id,
            yes_asks=yes_asks,
            yes_bids=yes_bids,
            no_asks=no_asks,
            no_bids=no_bids,
            source=source,
        )


# ═══════════════════════════════════════════════════════════════════
# ARB LOGIC — FEES + PNL CURVE
# ═══════════════════════════════════════════════════════════════════

def _build_cumulative_schedule(
    asks: list[BookLevel],
) -> list[tuple[float, float]]:
    """
    Convert ascending ask levels into a cumulative cost schedule.

    Returns:
        [(cum_qty, cum_cost), ...] starting with (0, 0).
        Monotonically increasing by quantity.
    """
    schedule: list[tuple[float, float]] = [(0.0, 0.0)]
    cum_qty = 0.0
    cum_cost = 0.0
    for lvl in asks:
        cum_qty += lvl.size
        cum_cost += lvl.price * lvl.size
        schedule.append((cum_qty, cum_cost))
    return schedule


def _interpolate_cost(
    schedule: list[tuple[float, float]],
    qty: float,
) -> Optional[float]:
    """
    Interpolate cumulative cost at a given quantity from the schedule.

    Returns None if qty exceeds total available depth.
    """
    if qty <= 0:
        return 0.0
    if qty > schedule[-1][0]:
        return None  # exceeds book depth

    for i in range(1, len(schedule)):
        prev_qty, prev_cost = schedule[i - 1]
        curr_qty, curr_cost = schedule[i]
        if qty <= curr_qty + 1e-9:
            if curr_qty == prev_qty:
                return curr_cost
            frac = (qty - prev_qty) / (curr_qty - prev_qty)
            return prev_cost + frac * (curr_cost - prev_cost)

    return schedule[-1][1]


def _compute_profit_at_qty(
    qty: float,
    cost_a: float,
    cost_b: float,
    fee_a: float,
    fee_b: float,
) -> PnLPoint:
    """
    Compute guaranteed profit at a given contract quantity.

    Position: Q contracts of side-A on venue A + Q contracts of side-B on venue B.
    Exactly one side pays $1 per contract regardless of outcome.

    If side-A wins:
        revenue = Q - fee_a * (Q - cost_a)
                = (Q - cost_a) * (1 - fee_a) + cost_a - cost_a  ... simplify
                = (Q - cost_a) * (1 - fee_a)
        profit  = revenue - cost_a - cost_b
                = (Q - cost_a) * (1 - fee_a) - cost_b

    If side-B wins:
        profit  = (Q - cost_b) * (1 - fee_b) - cost_a

    Args:
        qty: Number of contracts.
        cost_a: Cumulative cost of side-A contracts.
        cost_b: Cumulative cost of side-B contracts.
        fee_a: Fee rate on venue A (applied to profit on winning contracts).
        fee_b: Fee rate on venue B.

    Returns:
        PnLPoint with profit figures.
    """
    total_cost = cost_a + cost_b

    profit_a_wins = (qty - cost_a) * (1.0 - fee_a) - cost_b
    profit_b_wins = (qty - cost_b) * (1.0 - fee_b) - cost_a

    guaranteed = min(profit_a_wins, profit_b_wins)
    roi = (guaranteed / total_cost * 100.0) if total_cost > 0 else 0.0

    return PnLPoint(
        quantity=round(qty, 2),
        total_cost=round(total_cost, 4),
        profit_if_a_wins=round(profit_a_wins, 4),
        profit_if_b_wins=round(profit_b_wins, 4),
        guaranteed_profit=round(guaranteed, 4),
        roi_pct=round(roi, 2),
    )


def compute_pnl_curve(
    asks_a: list[BookLevel],
    asks_b: list[BookLevel],
    fee_a: float,
    fee_b: float,
    steps: int = PNL_CURVE_STEPS,
) -> list[PnLPoint]:
    """
    Walk two orderbooks and compute the PnL curve at each size step.

    We buy ``asks_a`` on venue A and ``asks_b`` on venue B in
    equal quantities. At each quantity point we compute the
    guaranteed (worst-case) profit after fees.

    Args:
        asks_a: Ask levels for side-A (ascending price).
        asks_b: Ask levels for side-B (ascending price).
        fee_a: Fee rate on venue A.
        fee_b: Fee rate on venue B.
        steps: Number of sample points on the curve.

    Returns:
        List of PnLPoint objects (ascending quantity).
    """
    if not asks_a or not asks_b:
        return []

    sched_a = _build_cumulative_schedule(asks_a)
    sched_b = _build_cumulative_schedule(asks_b)

    max_qty_a = sched_a[-1][0]
    max_qty_b = sched_b[-1][0]
    max_qty = min(max_qty_a, max_qty_b)

    if max_qty <= 0:
        return []

    curve: list[PnLPoint] = []

    # Sample at regular intervals
    for i in range(1, steps + 1):
        qty = max_qty * i / steps
        cost_a = _interpolate_cost(sched_a, qty)
        cost_b = _interpolate_cost(sched_b, qty)

        if cost_a is None or cost_b is None:
            break

        point = _compute_profit_at_qty(qty, cost_a, cost_b, fee_a, fee_b)
        curve.append(point)

    return curve


def _is_inverted(outcome_mapping: dict) -> bool:
    """
    Check if the outcome mapping is inverted (Yes->No or No->Yes).

    Args:
        outcome_mapping: Dict from verified_pairs.

    Returns:
        True if Yes on A maps to No on B (inverted).
    """
    for key, val in outcome_mapping.items():
        k = str(key).lower().strip()
        v = str(val).lower().strip()
        if k == "yes" and v == "no":
            return True
        if k == "no" and v == "yes":
            return True
    return False


def _venue_book_sane(book: VenueBook) -> bool:
    """
    Sanity-check a VenueBook: yes_ask + no_ask should be >= $0.85.

    In a real binary market the asks always sum to >= $1.00 (the
    spread is the market-maker's edge).  A sum far below $1 means
    the prices are bogus (e.g. stale, illiquid, or misinterpreted).
    We use 0.85 as a lenient floor to allow for wide spreads.
    """
    ya = book.yes_ask_top
    na = book.no_ask_top
    if ya is None or na is None:
        return True  # can't check — allow through
    total = ya + na
    if total < 0.85:
        logger.debug(
            "[Sanity] %s %s: yes_ask(%.4f) + no_ask(%.4f) = %.4f < 0.85 — skipping",
            book.venue, book.venue_market_id, ya, na, total,
        )
        return False
    return True


def find_arb_opportunities(
    pair: EquivalentPair,
    book_a: VenueBook,
    book_b: VenueBook,
) -> list[ArbOpportunity]:
    """
    Find arbitrage opportunities for a pair by checking both directions.

    For non-inverted pairs (Yes=Yes):
        Direction 1: Buy YES on A + Buy NO on B
        Direction 2: Buy NO on A  + Buy YES on B

    For inverted pairs (Yes=No):
        Direction 1: Buy YES on A + Buy YES on B  (complementary because A_yes=B_no)
        Direction 2: Buy NO on A  + Buy NO on B

    Each venue's book is sanity-checked first: yes_ask + no_ask must be
    >= $0.85 (a real binary market always sums >= $1.00; anything below
    $0.85 indicates bad data).

    Args:
        pair: EquivalentPair with outcome mapping.
        book_a: VenueBook for market A.
        book_b: VenueBook for market B.

    Returns:
        List of ArbOpportunity (0, 1, or 2 opportunities).
    """
    # ── Sanity-check each venue's book ───────────────────────────
    if not _venue_book_sane(book_a) or not _venue_book_sane(book_b):
        return []

    inverted = _is_inverted(pair.outcome_mapping)

    fee_a = VENUE_FEES.get(book_a.venue, 0.0)
    fee_b = VENUE_FEES.get(book_b.venue, 0.0)

    # Define the two directions to check
    if not inverted:
        # Normal: Yes=Yes → arb buys opposite sides
        directions = [
            ("yes", "no",  book_a.yes_asks, book_b.no_asks, fee_a, fee_b),
            ("no",  "yes", book_a.no_asks,  book_b.yes_asks, fee_a, fee_b),
        ]
    else:
        # Inverted: Yes=No → arb buys same-named sides (which are actually complementary)
        directions = [
            ("yes", "yes", book_a.yes_asks, book_b.yes_asks, fee_a, fee_b),
            ("no",  "no",  book_a.no_asks,  book_b.no_asks, fee_a, fee_b),
        ]

    opportunities: list[ArbOpportunity] = []

    for side_a, side_b, asks_a, asks_b, f_a, f_b in directions:
        if not asks_a or not asks_b:
            continue

        ask_top_a = asks_a[0].price
        ask_top_b = asks_b[0].price
        total_cost_1 = ask_top_a + ask_top_b

        gross_edge = 1.0 - total_cost_1

        # Per-contract profit at top-of-book with fees
        pt = _compute_profit_at_qty(
            qty=1.0, cost_a=ask_top_a, cost_b=ask_top_b,
            fee_a=f_a, fee_b=f_b,
        )
        net_profit_1 = pt.guaranteed_profit

        # Only report if there's at least marginal gross edge
        # (we report even small negatives so the user can see near-misses)
        if gross_edge < -0.10:
            continue  # way too expensive, skip

        # Build PnL curve
        pnl_curve = compute_pnl_curve(asks_a, asks_b, f_a, f_b)

        # Find optimal size (max guaranteed profit on the curve)
        optimal_qty = 0.0
        max_profit = 0.0
        for pt_c in pnl_curve:
            if pt_c.guaranteed_profit > max_profit:
                max_profit = pt_c.guaranteed_profit
                optimal_qty = pt_c.quantity

        opportunities.append(ArbOpportunity(
            pair_key=pair.pair_key,
            title_a=pair.title_a,
            title_b=pair.title_b,
            buy_a_side=side_a,
            buy_b_side=side_b,
            venue_a=book_a.venue,
            venue_b=book_b.venue,
            vmid_a=book_a.venue_market_id,
            vmid_b=book_b.venue_market_id,
            ask_a=ask_top_a,
            ask_b=ask_top_b,
            total_cost_1=round(total_cost_1, 4),
            gross_edge=round(gross_edge, 4),
            fee_a=f_a,
            fee_b=f_b,
            net_profit_1=round(net_profit_1, 4),
            net_roi_pct=round(
                (net_profit_1 / total_cost_1 * 100) if total_cost_1 > 0 else 0.0,
                2,
            ),
            pnl_curve=pnl_curve,
            optimal_qty=round(optimal_qty, 2),
            max_profit=round(max_profit, 4),
        ))

    return opportunities


# ═══════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════

def _trunc(text: str, maxlen: int = 50) -> str:
    """Truncate text with ellipsis."""
    return text[:maxlen - 3] + "..." if len(text) > maxlen else text


def _side_label(side: str) -> str:
    """Format side for display."""
    return side.upper()


def print_results(
    pairs: list[EquivalentPair],
    books: dict[str, VenueBook],
    opportunities: list[ArbOpportunity],
) -> None:
    """
    Print a formatted summary of all pairs, prices, and opportunities.

    Args:
        pairs: All equivalent pairs queried.
        books: Dict of "venue:vmid" -> VenueBook.
        opportunities: Detected arb opportunities.
    """
    print()
    print("=" * 115)
    print(
        f"  SPREAD SCANNER RESULTS  |  "
        f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}  |  "
        f"Kalshi fee: {KALSHI_FEE_RATE:.0%}  Poly fee: {POLYMARKET_FEE_RATE:.0%}"
    )
    print(
        f"  Equivalent pairs: {len(pairs)}  |  "
        f"Books fetched: {len(books)}  |  "
        f"Opportunities: {len(opportunities)}"
    )
    print("=" * 115)

    # ── All pairs with prices ────────────────────────────────────
    print()
    header = (
        f"{'#':>3}  {'Pair':45}  "
        f"{'A Yes Ask':>9}  {'A No Ask':>9}  "
        f"{'B Yes Ask':>9}  {'B No Ask':>9}  "
        f"{'Source'}"
    )
    print("-" * 115)
    print(header)
    print("-" * 115)

    for i, pair in enumerate(pairs, 1):
        key_a = f"{pair.venue_a}:{pair.vmid_a}"
        key_b = f"{pair.venue_b}:{pair.vmid_b}"
        ba = books.get(key_a)
        bb = books.get(key_b)

        label = _trunc(pair.title_a, 45)

        a_ya = f"{ba.yes_ask_top:.4f}" if ba and ba.yes_ask_top is not None else "  N/A"
        a_na = f"{ba.no_ask_top:.4f}" if ba and ba.no_ask_top is not None else "  N/A"
        b_ya = f"{bb.yes_ask_top:.4f}" if bb and bb.yes_ask_top is not None else "  N/A"
        b_na = f"{bb.no_ask_top:.4f}" if bb and bb.no_ask_top is not None else "  N/A"

        src_a = ba.source[:3] if ba else "---"
        src_b = bb.source[:3] if bb else "---"

        print(
            f"{i:3}  {label:45}  "
            f"{a_ya:>9}  {a_na:>9}  "
            f"{b_ya:>9}  {b_na:>9}  "
            f"{src_a}/{src_b}"
        )

    # ── Opportunities ────────────────────────────────────────────
    # Filter to positive net profit for the "actionable" section
    positive = [o for o in opportunities if o.net_profit_1 > 0]
    negative = [o for o in opportunities if o.net_profit_1 <= 0 and o.gross_edge > -0.03]

    if not positive and not negative:
        print()
        print("  No spread opportunities detected (all pairs cost >= $1.00).")
        print()
        return

    if positive:
        positive.sort(key=lambda o: o.net_roi_pct, reverse=True)

        print()
        print("=" * 115)
        print("  PROFITABLE OPPORTUNITIES (after fees, sorted by ROI)")
        print("=" * 115)
        print()
        print(
            f"{'#':>3}  {'Market':40}  "
            f"{'Buy A':>8}  {'Buy B':>8}  "
            f"{'Cost':>6}  {'Gross':>7}  {'Net':>7}  "
            f"{'ROI%':>6}  {'OptQty':>7}  {'MaxPnL':>8}  "
            f"{'Action'}"
        )
        print("-" * 115)

        for i, opp in enumerate(positive, 1):
            label = _trunc(opp.title_a, 40)
            gross_c = opp.gross_edge * 100
            net_c = opp.net_profit_1 * 100

            action = (
                f"{_side_label(opp.buy_a_side)}@{opp.venue_a[:4].upper()} + "
                f"{_side_label(opp.buy_b_side)}@{opp.venue_b[:4].upper()}"
            )

            print(
                f"{i:3}  {label:40}  "
                f"${opp.ask_a:.4f}  ${opp.ask_b:.4f}  "
                f"${opp.total_cost_1:.2f}  "
                f"{gross_c:+6.2f}c  {net_c:+6.2f}c  "
                f"{opp.net_roi_pct:+5.2f}%  "
                f"{opp.optimal_qty:>7.0f}  "
                f"${opp.max_profit:>7.2f}  "
                f"{action}"
            )

            # Print PnL curve for top opportunities
            if i <= 5 and opp.pnl_curve:
                _print_pnl_curve(opp)

    # ── Near-miss opportunities ──────────────────────────────────
    if negative:
        negative.sort(key=lambda o: o.gross_edge, reverse=True)
        print()
        print(f"  Near-miss opportunities (gross edge > 0 but net negative after fees): {len(negative)}")
        for opp in negative[:5]:
            label = _trunc(opp.title_a, 40)
            action = (
                f"{_side_label(opp.buy_a_side)}@{opp.venue_a[:4].upper()} + "
                f"{_side_label(opp.buy_b_side)}@{opp.venue_b[:4].upper()}"
            )
            print(
                f"    {label:40}  "
                f"gross {opp.gross_edge*100:+.2f}c  net {opp.net_profit_1*100:+.2f}c  "
                f"{action}"
            )

    # ── Summary ──────────────────────────────────────────────────
    print()
    print(f"  Total profitable: {len(positive)}")
    if positive:
        best = positive[0]
        print(
            f"  Best: {best.net_profit_1*100:.2f}c/contract "
            f"({best.net_roi_pct:+.2f}% ROI) on {_trunc(best.title_a, 40)}"
        )
        if best.optimal_qty > 0:
            print(
                f"  Optimal size: {best.optimal_qty:.0f} contracts, "
                f"max profit: ${best.max_profit:.2f}"
            )
        total_optimal_pnl = sum(o.max_profit for o in positive if o.max_profit > 0)
        total_optimal_capital = sum(
            o.optimal_qty * o.total_cost_1
            for o in positive
            if o.optimal_qty > 0 and o.max_profit > 0
        )
        print(
            f"  Total optimal PnL (all opportunities): "
            f"${total_optimal_pnl:,.2f}"
            f"  (capital required: ${total_optimal_capital:,.2f})"
        )
    print()


def _print_pnl_curve(opp: ArbOpportunity) -> None:
    """Print the PnL curve for an opportunity."""
    curve = opp.pnl_curve
    if not curve:
        return

    print()
    print(
        f"        PnL Curve: {_side_label(opp.buy_a_side)}@{opp.venue_a[:4]} + "
        f"{_side_label(opp.buy_b_side)}@{opp.venue_b[:4]}  "
        f"(fees: {opp.venue_a}={opp.fee_a:.0%}, {opp.venue_b}={opp.fee_b:.0%})"
    )
    print(
        f"        {'Qty':>8}  {'Cost':>10}  "
        f"{'If A wins':>10}  {'If B wins':>10}  "
        f"{'Guaranteed':>10}  {'ROI%':>7}"
    )
    print(f"        {'---':>8}  {'---':>10}  {'---':>10}  {'---':>10}  {'---':>10}  {'---':>7}")

    # Show a subset of the curve (every other point for brevity)
    shown = 0
    for j, pt in enumerate(curve):
        # Show first, last, optimal, and every ~4th point
        is_optimal = abs(pt.quantity - opp.optimal_qty) < 1.0
        is_key = (j == 0 or j == len(curve) - 1 or is_optimal or j % 4 == 0)
        if not is_key and shown > 10:
            continue

        marker = " <-- optimal" if is_optimal and pt.guaranteed_profit > 0 else ""
        if pt.guaranteed_profit < 0 and shown > 0:
            # Show the first negative point to indicate the limit
            marker = " <-- breakeven"

        print(
            f"        {pt.quantity:>8.0f}  "
            f"${pt.total_cost:>9.2f}  "
            f"${pt.profit_if_a_wins:>9.2f}  "
            f"${pt.profit_if_b_wins:>9.2f}  "
            f"${pt.guaranteed_profit:>9.2f}  "
            f"{pt.roi_pct:>6.2f}%"
            f"{marker}"
        )
        shown += 1

        # Stop showing after first negative guaranteed profit
        if pt.guaranteed_profit < -0.01 and shown > 3:
            print(f"        {'...':>8}  (remaining {len(curve) - j - 1} points omitted)")
            break

    print()


# ═══════════════════════════════════════════════════════════════════
# MAIN SCAN
# ═══════════════════════════════════════════════════════════════════

async def scan() -> list[ArbOpportunity]:
    """
    Execute a single spread scan.

    Steps:
    1. Fetch equivalent pairs from database
    2. Fetch live orderbooks from both venues (Yes + No) concurrently
    3. Find arbitrage opportunities with fee-adjusted PnL curves
    4. Print and return results

    Returns:
        List of detected ArbOpportunity objects.
    """
    dsn = config.DATABASE_URL
    if not dsn:
        logger.error("DATABASE_URL not set. Add it to .env")
        return []

    # ── 1. Fetch equivalent pairs ────────────────────────────────
    logger.info("Fetching equivalent pairs from database...")
    pairs = await fetch_equivalent_pairs(dsn)
    logger.info("Found %d equivalent pairs", len(pairs))

    if not pairs:
        print("\nNo equivalent pairs found in the database.")
        return []

    # ── 2. Collect unique markets to price ───────────────────────
    markets_to_price: dict[str, tuple[str, str]] = {}
    for pair in pairs:
        key_a = f"{pair.venue_a}:{pair.vmid_a}"
        key_b = f"{pair.venue_b}:{pair.vmid_b}"
        markets_to_price[key_a] = (pair.venue_a, pair.vmid_a)
        markets_to_price[key_b] = (pair.venue_b, pair.vmid_b)

    logger.info(
        "Fetching orderbooks for %d unique markets...",
        len(markets_to_price),
    )

    # ── 3. Fetch books concurrently ──────────────────────────────
    kalshi_key_id, kalshi_pk = _build_kalshi_auth()
    if not kalshi_key_id:
        logger.warning("Kalshi auth not configured -- Kalshi prices may fail")

    books: dict[str, VenueBook] = {}
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks: dict[str, asyncio.Task] = {}
        for key, (venue, vmid) in markets_to_price.items():
            if venue == "kalshi":
                tasks[key] = fetch_kalshi_book(
                    session, vmid, kalshi_key_id, kalshi_pk,
                )
            elif venue == "polymarket":
                tasks[key] = fetch_polymarket_book(session, vmid)
            else:
                logger.debug("Skipping unsupported venue: %s", venue)

        keys = list(tasks.keys())
        results = await asyncio.gather(
            *[tasks[k] for k in keys],
            return_exceptions=True,
        )

        for key, result in zip(keys, results):
            if isinstance(result, Exception):
                logger.warning("Book fetch failed for %s: %s", key, result)
            elif result is not None:
                books[key] = result

    logger.info(
        "Fetched %d/%d orderbooks successfully",
        len(books), len(markets_to_price),
    )

    # ── 4. Find arbitrage opportunities ──────────────────────────
    all_opps: list[ArbOpportunity] = []
    for pair in pairs:
        key_a = f"{pair.venue_a}:{pair.vmid_a}"
        key_b = f"{pair.venue_b}:{pair.vmid_b}"
        ba = books.get(key_a)
        bb = books.get(key_b)
        if not ba or not bb:
            continue

        opps = find_arb_opportunities(pair, ba, bb)
        all_opps.extend(opps)

    # ── 5. Display results ───────────────────────────────────────
    print_results(pairs, books, all_opps)

    # Return only profitable ones
    profitable = [o for o in all_opps if o.net_profit_1 > 0]
    return profitable


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    """Entry point for the spread scanner."""
    from dotenv import load_dotenv
    load_dotenv()

    logger.info("Starting spread scanner...")
    start = time.monotonic()
    opportunities = asyncio.run(scan())
    elapsed = time.monotonic() - start
    logger.info(
        "Scan completed in %.1fs -- %d profitable opportunities found",
        elapsed, len(opportunities),
    )


if __name__ == "__main__":
    main()
