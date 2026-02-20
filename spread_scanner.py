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

Fee model (per Kalshi official fee schedule, effective Feb 5 2026):
    - Kalshi taker:  fee = ceil(0.07  × C × P × (1-P))  (entry cost)
    - Kalshi maker:  fee = ceil(0.0175 × C × P × (1-P))  (entry cost)
    - Kalshi S&P/NDX: fee = ceil(0.035 × C × P × (1-P))  (entry cost)
    - Settlement fee: $0 (no fee on payout)
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
# Kalshi fee (per official fee schedule, effective Feb 5 2026):
#   fee = ceil(rate × C × P × (1-P))
# where P = contract price ($), C = contracts, ceil = round-up to 1¢.
# Fee is an ENTRY cost (charged when the trade matches), NOT at settlement.
# Settlement fee is $0.
KALSHI_TAKER_RATE = 0.070    # General taker (market orders)
KALSHI_MAKER_RATE = 0.0175   # Maker (resting orders)
KALSHI_INDEX_RATE = 0.035    # S&P 500 (INX*) / NASDAQ-100 (NASDAQ100*)

# Polymarket CLOB: 0% taker fee (as of 2025/2026).
POLYMARKET_FEE_RATE = 0.000

# Default fee rate per venue (taker = conservative worst-case).
VENUE_FEE_RATES: dict[str, float] = {
    "kalshi": KALSHI_TAKER_RATE,
    "polymarket": POLYMARKET_FEE_RATE,
}

# Legacy alias kept for WS scanner import compatibility.
KALSHI_FEE_RATE = KALSHI_TAKER_RATE


def kalshi_entry_fee(
    contracts: float,
    price: float,
    rate: float = KALSHI_TAKER_RATE,
) -> float:
    """
    Compute Kalshi entry fee per the official formula (Feb 5 2026).

    fee = rate × C × P × (1 - P)

    For simulation with fractional quantities we skip the ceil
    (rounding impact < 1¢ per price level, negligible at scale).
    The ceil is applied per-order on the exchange, not per-level.

    Args:
        contracts: Number of contracts (may be fractional for interpolation).
        price: Contract price in dollars [0, 1].
        rate: Fee-schedule rate (0.07 taker, 0.0175 maker, 0.035 index).

    Returns:
        Fee in USD.
    """
    if contracts <= 0 or price <= 0.0 or price >= 1.0 or rate <= 0.0:
        return 0.0
    return rate * contracts * price * (1.0 - price)


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

# Capital ladder interval (USD)
CAPITAL_LADDER_STEP = 10_000.0

# ── Daily-yield / capital-budget constants ────────────────────
# Default capital budget for a small retail trader (USD).
TRADER_CAPITAL = 5_000.0

# Opportunities yielding less than this absolute profit at
# TRADER_CAPITAL are demoted to the "dust" section.
MIN_PROFIT_THRESHOLD = 5.0


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
    resolution_date: Optional[datetime] = None
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
class CapitalLadderPoint:
    """PnL at a fixed capital investment level."""
    capital: float            # USD invested
    quantity: float           # contracts purchasable at this capital
    guaranteed_profit: float  # net guaranteed profit after fees
    roi_pct: float            # guaranteed_profit / capital * 100
    profit_if_a_wins: float   # profit if side-A outcome wins
    profit_if_b_wins: float   # profit if side-B outcome wins


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
    # Capital at optimal sizing (total cost to purchase optimal_qty)
    optimal_capital: float = 0.0
    optimal_roi_pct: float = 0.0
    # Capital ladder: PnL at $10k investment intervals
    capital_ladder: list[CapitalLadderPoint] = field(default_factory=list)
    # Resolution dates from venue APIs
    resolution_date_a: Optional[datetime] = None
    resolution_date_b: Optional[datetime] = None
    # ── Daily-yield metrics (populated by find_arb_opportunities) ──
    days_to_resolution: int = 0
    pnl_at_budget: float = 0.0       # guaranteed PnL at TRADER_CAPITAL
    daily_yield_bps: float = 0.0     # bps per day of held capital
    annualized_roi_pct: float = 0.0  # (pnl / capital) * (365 / days) * 100
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
    verdict: str = "equivalent"


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
                vp.verdict,
                ma.venue   AS venue_a,
                ma.venue_market_id AS vmid_a,
                ma.title   AS title_a,
                mb.venue   AS venue_b,
                mb.venue_market_id AS vmid_b,
                mb.title   AS title_b
            FROM verified_pairs vp
            JOIN markets ma ON ma.id = vp.market_a_id
            JOIN markets mb ON mb.id = vp.market_b_id
            WHERE vp.verdict IN ('equivalent', 'needs_review')
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
                verdict=r["verdict"] or "equivalent",
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
    close_time: Optional[datetime] = None


async def _fetch_kalshi_market_data(
    session: aiohttp.ClientSession,
    ticker: str,
    api_key_id: Optional[str],
    private_key,
    _max_retries: int = 3,
) -> Optional[_KalshiTopOfBook]:
    """
    Fetch ``GET /trade-api/v2/markets/{ticker}`` for reliable top-of-book.

    Retries up to ``_max_retries`` times on HTTP 429 with exponential
    backoff (1 s → 2 s → 4 s).

    Returns:
        _KalshiTopOfBook or None on failure.
    """
    path = f"{KALSHI_MARKETS_PATH}/{ticker}"
    url = f"{KALSHI_REST_BASE}{path}"
    headers = _kalshi_headers(api_key_id, private_key, "GET", path)

    for attempt in range(_max_retries + 1):
        async with _KALSHI_SEM:
            try:
                async with session.get(
                    url, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 429:
                        retry_after = float(
                            resp.headers.get("Retry-After", 1 * (2 ** attempt))
                        )
                        if attempt < _max_retries:
                            logger.debug(
                                "[Kalshi] %s -> 429, retry %d/%d in %.1fs",
                                ticker, attempt + 1, _max_retries, retry_after,
                            )
                            await asyncio.sleep(retry_after)
                            continue
                        body = await resp.text()
                        logger.warning(
                            "[Kalshi] %s -> 429 (exhausted retries): %s",
                            ticker, body[:200],
                        )
                        return None
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning(
                            "[Kalshi] %s -> %d: %s", ticker, resp.status, body[:200],
                        )
                        return None
                    data = await resp.json()
                    break  # success
            except Exception as exc:
                logger.warning("[Kalshi] Failed to fetch %s: %s", ticker, exc)
                return None
    else:
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

    # Extract resolution / close time
    close_time = None
    close_time_raw = market.get("close_time") or market.get("expiration_time")
    if close_time_raw:
        try:
            close_time = datetime.fromisoformat(
                str(close_time_raw).replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            pass

    return _KalshiTopOfBook(
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        volume=volume,
        close_time=close_time,
    )


async def _fetch_kalshi_orderbook_depth(
    session: aiohttp.ClientSession,
    ticker: str,
    api_key_id: Optional[str],
    private_key,
    tob: _KalshiTopOfBook,
    _max_retries: int = 3,
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

    Retries up to ``_max_retries`` times on HTTP 429 with exponential
    backoff (1 s → 2 s → 4 s).

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

    for attempt in range(_max_retries + 1):
        async with _KALSHI_SEM:
            try:
                async with session.get(
                    url, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 429:
                        retry_after = float(
                            resp.headers.get("Retry-After", 1 * (2 ** attempt))
                        )
                        if attempt < _max_retries:
                            logger.debug(
                                "[Kalshi] Orderbook %s -> 429, retry %d/%d in %.1fs",
                                ticker, attempt + 1, _max_retries, retry_after,
                            )
                            await asyncio.sleep(retry_after)
                            continue
                        logger.debug(
                            "[Kalshi] Orderbook %s -> 429 (exhausted retries)",
                            ticker,
                        )
                        return None
                    if resp.status != 200:
                        logger.debug(
                            "[Kalshi] Orderbook %s -> %d", ticker, resp.status,
                        )
                        return None
                    data = await resp.json()
                    break  # success
            except Exception as exc:
                logger.debug("[Kalshi] Orderbook fetch failed %s: %s", ticker, exc)
                return None
    else:
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
        resolution_date=tob.close_time,
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
        resolution_date=tob.close_time,
    )


# ═══════════════════════════════════════════════════════════════════
# POLYMARKET FETCHER
# ═══════════════════════════════════════════════════════════════════

async def _fetch_gamma_tokens(
    session: aiohttp.ClientSession,
    market_id: str,
) -> tuple[Optional[str], Optional[str], Optional[list[float]], Optional[datetime]]:
    """
    Fetch Polymarket market data (token IDs + indicative prices) via Gamma API.

    Uses ``GET /markets/{id}`` which works for both numeric Gamma IDs
    and condition_id hex strings.

    Returns:
        Tuple of (yes_token_id, no_token_id, outcome_prices, resolution_date).
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

    return None, None, None, None


def _parse_gamma_market(
    market: dict,
) -> tuple[Optional[str], Optional[str], Optional[list[float]], Optional[datetime]]:
    """Extract token IDs, outcome prices, and resolution date from a Gamma API market dict."""
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

    # Extract resolution date
    end_date: Optional[datetime] = None
    end_date_raw = market.get("endDate") or market.get("end_date_iso")
    if end_date_raw:
        try:
            end_date = datetime.fromisoformat(
                str(end_date_raw).replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            pass

    return yes_token, no_token, prices if prices else None, end_date


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
        yes_token, no_token, gamma_prices, end_date = await _fetch_gamma_tokens(
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
            resolution_date=end_date,
        )


# ═══════════════════════════════════════════════════════════════════
# ARB LOGIC — FEES + PNL CURVE
# ═══════════════════════════════════════════════════════════════════

def _build_cumulative_schedule(
    asks: list[BookLevel],
    fee_rate: float = 0.0,
) -> list[tuple[float, float, float]]:
    """
    Convert ascending ask levels into a cumulative cost+fee schedule.

    The fee at each level is computed using the Kalshi formula:
        fee = rate × C_level × P_level × (1 - P_level)
    For venues with 0% fees (Polymarket), pass fee_rate=0.0.

    Args:
        asks: Ask levels sorted ascending by price.
        fee_rate: Venue fee-schedule rate (0.07 for Kalshi taker, etc.).

    Returns:
        [(cum_qty, cum_cost, cum_fee), ...] starting with (0, 0, 0).
        Monotonically increasing by quantity.
    """
    schedule: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    cum_qty = 0.0
    cum_cost = 0.0
    cum_fee = 0.0
    for lvl in asks:
        cum_qty += lvl.size
        cum_cost += lvl.price * lvl.size
        cum_fee += kalshi_entry_fee(lvl.size, lvl.price, rate=fee_rate)
        schedule.append((cum_qty, cum_cost, cum_fee))
    return schedule


def _interpolate_cost(
    schedule: list[tuple[float, float, float]],
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
        prev_qty, prev_cost, _pf = schedule[i - 1]
        curr_qty, curr_cost, _cf = schedule[i]
        if qty <= curr_qty + 1e-9:
            if curr_qty == prev_qty:
                return curr_cost
            frac = (qty - prev_qty) / (curr_qty - prev_qty)
            return prev_cost + frac * (curr_cost - prev_cost)

    return schedule[-1][1]


def _interpolate_fee(
    schedule: list[tuple[float, float, float]],
    qty: float,
) -> float:
    """
    Interpolate cumulative entry fee at a given quantity from the schedule.

    Returns 0.0 if qty <= 0 or exceeds depth.
    """
    if qty <= 0:
        return 0.0
    if qty > schedule[-1][0]:
        return schedule[-1][2]  # cap to max

    for i in range(1, len(schedule)):
        prev_qty, _pc, prev_fee = schedule[i - 1]
        curr_qty, _cc, curr_fee = schedule[i]
        if qty <= curr_qty + 1e-9:
            if curr_qty == prev_qty:
                return curr_fee
            frac = (qty - prev_qty) / (curr_qty - prev_qty)
            return prev_fee + frac * (curr_fee - prev_fee)

    return schedule[-1][2]


def _compute_profit_at_qty(
    qty: float,
    cost_a: float,
    cost_b: float,
    fee_a: float,
    fee_b: float,
) -> PnLPoint:
    """
    Compute guaranteed profit at a given contract quantity.

    Position: Q contracts of side-A on venue A + Q contracts of side-B
    on venue B.  Exactly one side pays $1 per contract regardless of
    outcome.

    Fee model (entry-based):
        Both venue fees are charged UP FRONT when the trade matches.
        There is NO settlement fee.  Therefore:

            total_outlay = cost_a + fee_a + cost_b + fee_b
            payout       = Q      (one side always pays $1/contract)
            profit       = Q - total_outlay   (same for BOTH outcomes)

    Args:
        qty: Number of contracts.
        cost_a: Cumulative cost of side-A contracts (USD).
        cost_b: Cumulative cost of side-B contracts (USD).
        fee_a: Cumulative entry fee on venue A (USD, NOT a rate).
        fee_b: Cumulative entry fee on venue B (USD, NOT a rate).

    Returns:
        PnLPoint with profit figures.
    """
    total_outlay = cost_a + cost_b + fee_a + fee_b

    # Entry-based fees → profit is symmetric regardless of outcome.
    guaranteed = qty - total_outlay
    roi = (guaranteed / total_outlay * 100.0) if total_outlay > 0 else 0.0

    return PnLPoint(
        quantity=round(qty, 2),
        total_cost=round(total_outlay, 4),
        profit_if_a_wins=round(guaranteed, 4),
        profit_if_b_wins=round(guaranteed, 4),
        guaranteed_profit=round(guaranteed, 4),
        roi_pct=round(roi, 2),
    )


def compute_pnl_curve(
    asks_a: list[BookLevel],
    asks_b: list[BookLevel],
    venue_a: str = "polymarket",
    venue_b: str = "kalshi",
    steps: int = PNL_CURVE_STEPS,
) -> list[PnLPoint]:
    """
    Walk two orderbooks and compute the PnL curve at each size step.

    We buy ``asks_a`` on venue A and ``asks_b`` on venue B in
    equal quantities. At each quantity point we compute the
    guaranteed (worst-case) profit after entry fees.

    Args:
        asks_a: Ask levels for side-A (ascending price).
        asks_b: Ask levels for side-B (ascending price).
        venue_a: Venue name for side-A (determines fee rate).
        venue_b: Venue name for side-B (determines fee rate).
        steps: Number of sample points on the curve.

    Returns:
        List of PnLPoint objects (ascending quantity).
    """
    if not asks_a or not asks_b:
        return []

    rate_a = VENUE_FEE_RATES.get(venue_a, 0.0)
    rate_b = VENUE_FEE_RATES.get(venue_b, 0.0)

    sched_a = _build_cumulative_schedule(asks_a, fee_rate=rate_a)
    sched_b = _build_cumulative_schedule(asks_b, fee_rate=rate_b)

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

        fee_a_usd = _interpolate_fee(sched_a, qty)
        fee_b_usd = _interpolate_fee(sched_b, qty)

        point = _compute_profit_at_qty(qty, cost_a, cost_b, fee_a_usd, fee_b_usd)
        curve.append(point)

    return curve


def compute_pnl_at_capital(
    asks_a: list[BookLevel],
    asks_b: list[BookLevel],
    venue_a: str = "polymarket",
    venue_b: str = "kalshi",
    capital: float = TRADER_CAPITAL,
) -> Optional[CapitalLadderPoint]:
    """
    Compute the guaranteed PnL when deploying exactly *capital* dollars.

    This is a single-point version of :func:`compute_capital_ladder`,
    used to evaluate the trader-budget metric without generating a full
    ladder.

    Capital includes contract cost + entry fees.

    Args:
        asks_a: Ask levels for side-A (ascending price).
        asks_b: Ask levels for side-B (ascending price).
        venue_a: Venue name for side-A (determines fee rate).
        venue_b: Venue name for side-B (determines fee rate).
        capital: USD to deploy.

    Returns:
        A single :class:`CapitalLadderPoint`, or ``None`` if the books
        don't have enough depth.
    """
    if not asks_a or not asks_b or capital <= 0:
        return None

    rate_a = VENUE_FEE_RATES.get(venue_a, 0.0)
    rate_b = VENUE_FEE_RATES.get(venue_b, 0.0)
    sched_a = _build_cumulative_schedule(asks_a, fee_rate=rate_a)
    sched_b = _build_cumulative_schedule(asks_b, fee_rate=rate_b)

    max_qty_a = sched_a[-1][0]
    max_qty_b = sched_b[-1][0]
    max_qty = min(max_qty_a, max_qty_b)
    if max_qty <= 0:
        return None

    max_cost_a = _interpolate_cost(sched_a, max_qty)
    max_cost_b = _interpolate_cost(sched_b, max_qty)
    if max_cost_a is None or max_cost_b is None:
        return None
    max_fee_a = _interpolate_fee(sched_a, max_qty)
    max_fee_b = _interpolate_fee(sched_b, max_qty)
    max_capital = (max_cost_a or 0) + (max_cost_b or 0) + max_fee_a + max_fee_b
    if capital > max_capital:
        capital = max_capital  # cap to available depth

    # Binary search for qty purchasable at *capital* (cost + fees)
    lo, hi = 0.0, max_qty
    for _ in range(60):
        mid = (lo + hi) / 2.0
        ca = _interpolate_cost(sched_a, mid)
        cb = _interpolate_cost(sched_b, mid)
        if ca is None or cb is None:
            hi = mid
            continue
        fa = _interpolate_fee(sched_a, mid)
        fb = _interpolate_fee(sched_b, mid)
        if ca + cb + fa + fb <= capital:
            lo = mid
        else:
            hi = mid

    qty = lo
    if qty < 0.5:
        return None

    cost_a = _interpolate_cost(sched_a, qty)
    cost_b = _interpolate_cost(sched_b, qty)
    if cost_a is None or cost_b is None:
        return None
    fee_a_usd = _interpolate_fee(sched_a, qty)
    fee_b_usd = _interpolate_fee(sched_b, qty)

    actual_cap = cost_a + cost_b + fee_a_usd + fee_b_usd
    pt = _compute_profit_at_qty(qty, cost_a, cost_b, fee_a_usd, fee_b_usd)
    roi = (pt.guaranteed_profit / actual_cap * 100) if actual_cap > 0 else 0.0

    return CapitalLadderPoint(
        capital=round(actual_cap, 2),
        quantity=round(qty, 2),
        guaranteed_profit=round(pt.guaranteed_profit, 4),
        roi_pct=round(roi, 4),
        profit_if_a_wins=round(pt.profit_if_a_wins, 4),
        profit_if_b_wins=round(pt.profit_if_b_wins, 4),
    )


def compute_capital_ladder(
    asks_a: list[BookLevel],
    asks_b: list[BookLevel],
    venue_a: str = "polymarket",
    venue_b: str = "kalshi",
    step: float = CAPITAL_LADDER_STEP,
    max_rungs: int = 50,
    neg_rungs_after_peak: int = 3,
) -> list[CapitalLadderPoint]:
    """
    Compute guaranteed PnL at fixed capital investment intervals.

    Uses binary search on the cumulative cost+fee schedules to find
    the exact quantity purchasable at each capital level (where capital
    covers contract cost + entry fees), then computes fee-adjusted PnL.

    Early termination: stops after ``neg_rungs_after_peak`` consecutive
    rungs with declining PnL past the best rung, or after ``max_rungs``
    total rungs — whichever comes first.

    Args:
        asks_a: Ask levels for side-A (ascending price).
        asks_b: Ask levels for side-B (ascending price).
        venue_a: Venue name for side-A (determines fee rate).
        venue_b: Venue name for side-B (determines fee rate).
        step: Capital interval in USD (default $10,000).
        max_rungs: Hard cap on number of ladder entries (default 50).
        neg_rungs_after_peak: Stop after this many consecutive rungs
            with PnL below the peak (default 3).

    Returns:
        List of CapitalLadderPoint (ascending by capital).
    """
    if not asks_a or not asks_b:
        return []

    rate_a = VENUE_FEE_RATES.get(venue_a, 0.0)
    rate_b = VENUE_FEE_RATES.get(venue_b, 0.0)
    sched_a = _build_cumulative_schedule(asks_a, fee_rate=rate_a)
    sched_b = _build_cumulative_schedule(asks_b, fee_rate=rate_b)

    max_qty_a = sched_a[-1][0]
    max_qty_b = sched_b[-1][0]
    max_qty = min(max_qty_a, max_qty_b)

    if max_qty <= 0:
        return []

    # Find max capital available at max_qty (cost + fees)
    max_cost_a = _interpolate_cost(sched_a, max_qty)
    max_cost_b = _interpolate_cost(sched_b, max_qty)
    if max_cost_a is None or max_cost_b is None:
        return []
    max_fee_a = _interpolate_fee(sched_a, max_qty)
    max_fee_b = _interpolate_fee(sched_b, max_qty)
    max_capital = max_cost_a + max_cost_b + max_fee_a + max_fee_b

    ladder: list[CapitalLadderPoint] = []
    target_capital = step

    best_profit = -float("inf")
    consecutive_below_peak = 0

    while target_capital <= max_capital + 1e-9:
        # Binary search for quantity Q where cost+fee ≈ target_capital
        lo, hi = 0.0, max_qty
        for _ in range(60):  # ~18 digits of precision
            mid = (lo + hi) / 2
            ca = _interpolate_cost(sched_a, mid)
            cb = _interpolate_cost(sched_b, mid)
            if ca is None or cb is None:
                hi = mid
                continue
            fa = _interpolate_fee(sched_a, mid)
            fb = _interpolate_fee(sched_b, mid)
            if ca + cb + fa + fb < target_capital:
                lo = mid
            else:
                hi = mid

        qty = (lo + hi) / 2
        cost_a = _interpolate_cost(sched_a, qty)
        cost_b = _interpolate_cost(sched_b, qty)

        if cost_a is None or cost_b is None:
            break

        fee_a_usd = _interpolate_fee(sched_a, qty)
        fee_b_usd = _interpolate_fee(sched_b, qty)

        actual_capital = cost_a + cost_b + fee_a_usd + fee_b_usd
        pt = _compute_profit_at_qty(qty, cost_a, cost_b, fee_a_usd, fee_b_usd)
        roi = (pt.guaranteed_profit / actual_capital * 100) if actual_capital > 0 else 0.0

        ladder.append(CapitalLadderPoint(
            capital=round(target_capital, 2),
            quantity=round(qty, 2),
            guaranteed_profit=round(pt.guaranteed_profit, 2),
            roi_pct=round(roi, 2),
            profit_if_a_wins=round(pt.profit_if_a_wins, 2),
            profit_if_b_wins=round(pt.profit_if_b_wins, 2),
        ))

        # ── Early termination logic ──────────────────────────────
        if pt.guaranteed_profit > best_profit:
            best_profit = pt.guaranteed_profit
            consecutive_below_peak = 0
        else:
            consecutive_below_peak += 1

        if consecutive_below_peak >= neg_rungs_after_peak:
            break
        if len(ladder) >= max_rungs:
            break

        target_capital += step

    return ladder


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


def _pick_resolution_date(
    date_a: Optional[datetime],
    date_b: Optional[datetime],
) -> Optional[datetime]:
    """Return the earliest non-None resolution date (markets should agree)."""
    if date_a and date_b:
        return min(date_a, date_b)
    return date_a or date_b


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

    rate_a = VENUE_FEE_RATES.get(book_a.venue, 0.0)
    rate_b = VENUE_FEE_RATES.get(book_b.venue, 0.0)

    # Define the two directions to check
    if not inverted:
        # Normal: Yes=Yes → arb buys opposite sides
        directions = [
            ("yes", "no",  book_a.yes_asks, book_b.no_asks),
            ("no",  "yes", book_a.no_asks,  book_b.yes_asks),
        ]
    else:
        # Inverted: Yes=No → arb buys same-named sides (which are actually complementary)
        directions = [
            ("yes", "yes", book_a.yes_asks, book_b.yes_asks),
            ("no",  "no",  book_a.no_asks,  book_b.no_asks),
        ]

    opportunities: list[ArbOpportunity] = []

    for side_a, side_b, asks_a, asks_b in directions:
        if not asks_a or not asks_b:
            continue

        ask_top_a = asks_a[0].price
        ask_top_b = asks_b[0].price
        total_cost_1 = ask_top_a + ask_top_b

        gross_edge = 1.0 - total_cost_1

        # Per-contract entry fees at top-of-book price (USD)
        fee_1_a = kalshi_entry_fee(1.0, ask_top_a, rate=rate_a)
        fee_1_b = kalshi_entry_fee(1.0, ask_top_b, rate=rate_b)

        # Per-contract profit at top-of-book with entry fees
        pt = _compute_profit_at_qty(
            qty=1.0, cost_a=ask_top_a, cost_b=ask_top_b,
            fee_a=fee_1_a, fee_b=fee_1_b,
        )
        net_profit_1 = pt.guaranteed_profit

        # Only report if there's at least marginal gross edge
        # (we report even small negatives so the user can see near-misses)
        if gross_edge < -0.10:
            continue  # way too expensive, skip

        # Build PnL curve (uses venue names to look up fee rates)
        pnl_curve = compute_pnl_curve(
            asks_a, asks_b,
            venue_a=book_a.venue, venue_b=book_b.venue,
        )

        # Find optimal size (max guaranteed profit on the curve)
        optimal_qty = 0.0
        max_profit = 0.0
        optimal_capital = 0.0
        optimal_roi_pct = 0.0
        for pt_c in pnl_curve:
            if pt_c.guaranteed_profit > max_profit:
                max_profit = pt_c.guaranteed_profit
                optimal_qty = pt_c.quantity
                optimal_capital = pt_c.total_cost
                optimal_roi_pct = pt_c.roi_pct

        # Build capital ladder at $10k intervals
        capital_ladder = compute_capital_ladder(
            asks_a, asks_b,
            venue_a=book_a.venue, venue_b=book_b.venue,
        )

        # ── Daily-yield / capital-budget metrics ──────────────────
        res_dt = _pick_resolution_date(
            book_a.resolution_date, book_b.resolution_date,
        )
        days_to = max((res_dt - datetime.now(UTC)).days, 1) if res_dt else 365

        budget_pt = compute_pnl_at_capital(
            asks_a, asks_b,
            venue_a=book_a.venue, venue_b=book_b.venue,
            capital=TRADER_CAPITAL,
        )
        pnl_at_budget = budget_pt.guaranteed_profit if budget_pt else 0.0

        # Daily yield in basis points:  (pnl / capital) / days * 10_000
        if pnl_at_budget > 0 and days_to > 0:
            daily_yield = (pnl_at_budget / TRADER_CAPITAL) / days_to * 10_000
            ann_roi = (pnl_at_budget / TRADER_CAPITAL) * (365.0 / days_to) * 100
        else:
            daily_yield = 0.0
            ann_roi = 0.0

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
            fee_a=rate_a,
            fee_b=rate_b,
            net_profit_1=round(net_profit_1, 4),
            net_roi_pct=round(
                (net_profit_1 / total_cost_1 * 100) if total_cost_1 > 0 else 0.0,
                2,
            ),
            pnl_curve=pnl_curve,
            optimal_qty=round(optimal_qty, 2),
            max_profit=round(max_profit, 4),
            optimal_capital=round(optimal_capital, 2),
            optimal_roi_pct=round(optimal_roi_pct, 2),
            capital_ladder=capital_ladder,
            resolution_date_a=book_a.resolution_date,
            resolution_date_b=book_b.resolution_date,
            days_to_resolution=days_to,
            pnl_at_budget=round(pnl_at_budget, 2),
            daily_yield_bps=round(daily_yield, 4),
            annualized_roi_pct=round(ann_roi, 2),
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
        f"Kalshi fee: {KALSHI_TAKER_RATE:.0%}·P·(1-P)  Poly fee: {POLYMARKET_FEE_RATE:.0%}"
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
        f"{'#':>3}  {'Pair':40}  "
        f"{'A Yes Ask':>9}  {'A No Ask':>9}  "
        f"{'B Yes Ask':>9}  {'B No Ask':>9}  "
        f"{'Resolves':>12}  {'Verdict':>10}  "
        f"{'Src'}"
    )
    print("-" * 130)
    print(header)
    print("-" * 130)

    for i, pair in enumerate(pairs, 1):
        key_a = f"{pair.venue_a}:{pair.vmid_a}"
        key_b = f"{pair.venue_b}:{pair.vmid_b}"
        ba = books.get(key_a)
        bb = books.get(key_b)

        label = _trunc(pair.title_a, 40)

        a_ya = f"{ba.yes_ask_top:.4f}" if ba and ba.yes_ask_top is not None else "  N/A"
        a_na = f"{ba.no_ask_top:.4f}" if ba and ba.no_ask_top is not None else "  N/A"
        b_ya = f"{bb.yes_ask_top:.4f}" if bb and bb.yes_ask_top is not None else "  N/A"
        b_na = f"{bb.no_ask_top:.4f}" if bb and bb.no_ask_top is not None else "  N/A"

        # Resolution date: use the earliest of the two venues
        res_a = ba.resolution_date if ba else None
        res_b = bb.resolution_date if bb else None
        res_date = _pick_resolution_date(res_a, res_b)
        res_str = res_date.strftime("%Y-%m-%d") if res_date else "N/A"

        src_a = ba.source[:3] if ba else "---"
        src_b = bb.source[:3] if bb else "---"

        print(
            f"{i:3}  {label:40}  "
            f"{a_ya:>9}  {a_na:>9}  "
            f"{b_ya:>9}  {b_na:>9}  "
            f"{res_str:>12}  {pair.verdict:>10}  "
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
        # Primary sort: daily yield (profit-per-dollar-per-day).
        # Demote opportunities below MIN_PROFIT_THRESHOLD at TRADER_CAPITAL.
        actionable = [o for o in positive if o.pnl_at_budget >= MIN_PROFIT_THRESHOLD]
        dust = [o for o in positive if o.pnl_at_budget < MIN_PROFIT_THRESHOLD]
        actionable.sort(key=lambda o: o.daily_yield_bps, reverse=True)
        dust.sort(key=lambda o: o.daily_yield_bps, reverse=True)

        print()
        print("=" * 155)
        budget_k = TRADER_CAPITAL / 1000
        print(
            f"  PROFITABLE OPPORTUNITIES  (after fees, ranked by daily yield)  "
            f"|  Capital budget: ${TRADER_CAPITAL:,.0f}"
        )
        print("=" * 155)
        print()
        print(
            f"{'#':>3}  {'Market':35}  "
            f"{'Buy A':>8}  {'Buy B':>8}  "
            f"{'Cost':>6}  {'Net':>7}  {'ROI%':>6}  "
            f"{'Days':>5}  "
            f"{'PnL@$' + f'{budget_k:.0f}k':>9}  "
            f"{'Ann.ROI':>8}  {'DailyYld':>9}  "
            f"{'Resolves':>12}  "
            f"{'Action'}"
        )
        print("-" * 155)

        ranked = actionable + dust
        for i, opp in enumerate(ranked, 1):
            label = _trunc(opp.title_a, 35)
            net_c = opp.net_profit_1 * 100

            res_date = _pick_resolution_date(
                opp.resolution_date_a, opp.resolution_date_b,
            )
            res_str = res_date.strftime("%Y-%m-%d") if res_date else "N/A"

            action = (
                f"{_side_label(opp.buy_a_side)}@{opp.venue_a[:4].upper()} + "
                f"{_side_label(opp.buy_b_side)}@{opp.venue_b[:4].upper()}"
            )

            # Badges for time-to-resolution
            badge = ""
            if opp.days_to_resolution <= 7:
                badge = " ⚡"       # resolves within a week
            elif opp.days_to_resolution <= 30:
                badge = " 🔄"      # resolves within a month

            pnl_str = f"${opp.pnl_at_budget:>+7.2f}" if opp.pnl_at_budget != 0 else "  $0.00"
            ann_str = f"{opp.annualized_roi_pct:>+7.2f}%" if opp.annualized_roi_pct else "   N/A"
            dy_str = f"{opp.daily_yield_bps:>8.4f}" if opp.daily_yield_bps else "  0.0000"

            # Separator line between actionable and dust
            if i == len(actionable) + 1 and dust:
                print(f"  {'─' * 150}")
                print(
                    f"  Below: PnL < ${MIN_PROFIT_THRESHOLD:.0f} "
                    f"at ${TRADER_CAPITAL:,.0f} capital (dust)"
                )
                print(f"  {'─' * 150}")

            print(
                f"{i:3}  {label:35}  "
                f"${opp.ask_a:.4f}  ${opp.ask_b:.4f}  "
                f"${opp.total_cost_1:.2f}  "
                f"{net_c:+6.2f}c  "
                f"{opp.net_roi_pct:+5.2f}%  "
                f"{opp.days_to_resolution:>5}  "
                f"{pnl_str}  "
                f"{ann_str}  "
                f"{dy_str}  "
                f"{res_str:>12}  "
                f"{action}{badge}"
            )

            # Print PnL curve + capital ladder for top opportunities
            if i <= 5 and opp.pnl_curve:
                _print_pnl_curve(opp)
                _print_optimal_sizing(opp)
                _print_capital_ladder(opp)

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
    print(f"  Actionable (PnL ≥ ${MIN_PROFIT_THRESHOLD:.0f} at ${TRADER_CAPITAL:,.0f}): {len(actionable)}")
    if positive:
        # Best by daily yield (first in ranked)
        best = ranked[0]
        print(
            f"  Top daily yield: {best.daily_yield_bps:.4f} bps/day"
            f"  ({best.annualized_roi_pct:+.2f}% ann.) on"
            f"  {_trunc(best.title_a, 40)}"
            f"  — resolves in {best.days_to_resolution}d"
        )

        # Find the best viable opportunity with optimal_qty > 0 for the
        # "Optimal size" summary line.  The top-ranked opp by daily yield
        # may have zero depth at optimal sizing; fall back to the first
        # that does.
        best_viable = next(
            (o for o in ranked if o.optimal_qty > 0 and o.max_profit > 0),
            None,
        )
        if best_viable:
            print(
                f"  Optimal size: {best_viable.optimal_qty:.0f} contracts, "
                f"capital: ${best_viable.optimal_capital:,.0f}, "
                f"max profit: ${best_viable.max_profit:.2f} "
                f"({best_viable.optimal_roi_pct:+.2f}% ROI)"
            )
        else:
            print(
                f"  Optimal size: no executable sizing found "
                f"(insufficient orderbook depth)"
            )

        total_optimal_pnl = sum(o.max_profit for o in positive if o.max_profit > 0)
        total_optimal_capital = sum(
            o.optimal_capital
            for o in positive
            if o.optimal_qty > 0 and o.max_profit > 0
        )
        print(
            f"  Total optimal PnL (all opportunities): "
            f"${total_optimal_pnl:,.2f}"
            f"  (capital required: ${total_optimal_capital:,.2f})"
        )

        # Capital recyclable within 30 days
        short_dated = [o for o in actionable if o.days_to_resolution <= 30]
        if short_dated:
            recyclable_pnl = sum(o.pnl_at_budget for o in short_dated)
            print(
                f"  Capital recyclable ≤30d: "
                f"{len(short_dated)} opps, "
                f"${recyclable_pnl:,.2f} PnL at ${TRADER_CAPITAL:,.0f} budget"
            )

        # Resolution date summary
        res_dates = []
        for o in positive:
            rd = _pick_resolution_date(o.resolution_date_a, o.resolution_date_b)
            if rd:
                res_dates.append(rd)
        if res_dates:
            earliest = min(res_dates)
            latest = max(res_dates)
            days_to_earliest = (earliest - datetime.now(UTC)).days
            print(
                f"  Resolution window: "
                f"{earliest.strftime('%Y-%m-%d')} → {latest.strftime('%Y-%m-%d')}"
                f"  ({days_to_earliest}d to nearest)"
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
        f"(fees: {opp.venue_a}={'0%' if opp.fee_a == 0 else f'{opp.fee_a:.0%}·P·(1-P)'}, "
        f"{opp.venue_b}={'0%' if opp.fee_b == 0 else f'{opp.fee_b:.0%}·P·(1-P)'})"
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


def _print_optimal_sizing(opp: ArbOpportunity) -> None:
    """Print optimal capital sizing analysis."""
    if opp.optimal_qty <= 0 or opp.max_profit <= 0:
        return

    res_date = _pick_resolution_date(opp.resolution_date_a, opp.resolution_date_b)
    res_str = ""
    if res_date:
        days_to = (res_date - datetime.now(UTC)).days
        res_str = f"  |  Resolves: {res_date.strftime('%Y-%m-%d')} ({days_to}d)"

    avg_price = opp.optimal_capital / opp.optimal_qty if opp.optimal_qty > 0 else 0

    print(
        f"        ┌─ OPTIMAL SIZING ──────────────────────────────────"
        f"──────────────────────────────────────┐"
    )
    print(
        f"        │  {opp.optimal_qty:,.0f} contracts @ ${avg_price:.4f} avg"
        f"  →  ${opp.optimal_capital:,.0f} capital"
        f"  →  ${opp.max_profit:,.2f} profit"
        f"  ({opp.optimal_roi_pct:+.2f}% ROI){res_str}"
    )

    # Show diminishing returns: where 50%, 80%, 95% of max profit is captured
    curve = opp.pnl_curve
    if curve:
        thresholds = [0.50, 0.80, 0.95]
        captures: list[str] = []
        for pct in thresholds:
            target = opp.max_profit * pct
            for pt in curve:
                if pt.guaranteed_profit >= target - 1e-6:
                    cap_pct = (
                        (pt.total_cost / opp.optimal_capital * 100)
                        if opp.optimal_capital > 0
                        else 0
                    )
                    captures.append(
                        f"{pct:.0%} profit @ ${pt.total_cost:,.0f} "
                        f"({cap_pct:.0f}% of optimal capital)"
                    )
                    break
        if captures:
            print(f"        │  Diminishing returns:  {' │ '.join(captures)}")

    print(
        f"        └──────────────────────────────────────────────────"
        f"──────────────────────────────────────┘"
    )


def _print_capital_ladder(opp: ArbOpportunity) -> None:
    """Print PnL at fixed capital investment intervals."""
    ladder = opp.capital_ladder
    if not ladder:
        return

    print()
    print(
        f"        Capital Ladder (PnL at ${CAPITAL_LADDER_STEP/1000:.0f}k intervals):"
    )
    print(
        f"        {'Capital':>12}  {'Contracts':>10}  "
        f"{'Profit':>10}  {'ROI%':>7}  "
        f"{'If A Wins':>10}  {'If B Wins':>10}  "
        f"{'Note'}"
    )
    print(
        f"        {'─' * 12}  {'─' * 10}  "
        f"{'─' * 10}  {'─' * 7}  "
        f"{'─' * 10}  {'─' * 10}  "
        f"{'─' * 10}"
    )

    for rung in ladder:
        # Mark the rung closest to optimal capital
        note = ""
        if (
            opp.optimal_capital > 0
            and abs(rung.capital - opp.optimal_capital) < CAPITAL_LADDER_STEP * 0.6
        ):
            note = "<-- near optimal"
        elif rung.guaranteed_profit <= 0:
            note = "<-- negative PnL"

        print(
            f"        ${rung.capital:>11,.0f}  "
            f"{rung.quantity:>10,.0f}  "
            f"${rung.guaranteed_profit:>9,.2f}  "
            f"{rung.roi_pct:>6.2f}%  "
            f"${rung.profit_if_a_wins:>9,.2f}  "
            f"${rung.profit_if_b_wins:>9,.2f}  "
            f"{note}"
        )

    # Summary line
    if ladder:
        best_rung = max(ladder, key=lambda r: r.guaranteed_profit)
        last = ladder[-1]
        if best_rung.guaranteed_profit > 0:
            print(
                f"        ── Best on ladder: "
                f"${best_rung.capital:,.0f} → "
                f"${best_rung.guaranteed_profit:,.2f} profit "
                f"({best_rung.roi_pct:.2f}% ROI)"
            )
        else:
            print(
                f"        ── No profitable capital level "
                f"(best: ${best_rung.guaranteed_profit:,.2f} "
                f"at ${best_rung.capital:,.0f})"
            )
        if last is not best_rung:
            print(
                f"        ── Ladder truncated after "
                f"${last.capital:,.0f} (PnL declining)"
            )
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

    # ── 3. Fetch books (sequential Kalshi, concurrent Poly) ─────
    kalshi_key_id, kalshi_pk = _build_kalshi_auth()
    if not kalshi_key_id:
        logger.warning("Kalshi auth not configured -- Kalshi prices may fail")

    books: dict[str, VenueBook] = {}
    timeout = aiohttp.ClientTimeout(total=180)

    # Split markets by venue
    kalshi_markets: list[tuple[str, str]] = []   # (cache_key, ticker)
    poly_markets: list[tuple[str, str]] = []     # (cache_key, vmid)

    for key, (venue, vmid) in markets_to_price.items():
        if venue == "kalshi":
            kalshi_markets.append((key, vmid))
        elif venue == "polymarket":
            poly_markets.append((key, vmid))
        else:
            logger.debug("Skipping unsupported venue: %s", venue)

    KALSHI_GAP = 0.20  # seconds between Kalshi request starts

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # ── Polymarket: fetch concurrently (no rate-limit issues) ──
        async def _fetch_poly(key: str, vmid: str) -> tuple[str, Optional[VenueBook]]:
            result = await fetch_polymarket_book(session, vmid)
            return key, result

        poly_coros = [_fetch_poly(k, v) for k, v in poly_markets]
        poly_results = await asyncio.gather(*poly_coros, return_exceptions=True)

        for result in poly_results:
            if isinstance(result, Exception):
                logger.warning("Poly book fetch failed: %s", result)
            elif result is not None:
                key, book = result
                if book is not None:
                    books[key] = book

        # ── Kalshi: fetch sequentially with 200 ms gap ────────────
        for idx, (key, ticker) in enumerate(kalshi_markets):
            try:
                book = await fetch_kalshi_book(
                    session, ticker, kalshi_key_id, kalshi_pk,
                )
                if book is not None:
                    books[key] = book
            except Exception as exc:
                logger.warning("Kalshi book fetch failed for %s: %s", ticker, exc)
            # Pause between tickers (skip after the last one)
            if idx < len(kalshi_markets) - 1:
                await asyncio.sleep(KALSHI_GAP)

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
