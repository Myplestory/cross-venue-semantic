"""
Cross-venue spread scanner for equivalent verified pairs.

Fetches live prices from Polymarket (CLOB) and Kalshi for all
equivalent pairs and identifies spread opportunities where
cross-venue bid/ask differences create potential profit.

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
# Ensure project root is on sys.path so config + discovery are importable
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


# ── Constants ────────────────────────────────────────────────────
KALSHI_FEE_RATE = 0.07       # Kalshi takes ~7% of profit on winning contracts
POLYMARKET_FEE_RATE = 0.00   # Polymarket: 0% taker fee on CLOB
MIN_GROSS_SPREAD = 0.0       # Report all (even tiny) spreads; filter at display
KALSHI_REST_BASE = (
    "https://demo-api.kalshi.co"
    if config.KALSHI_USE_DEMO
    else "https://api.elections.kalshi.com"
)
KALSHI_MARKETS_PATH = "/trade-api/v2/markets"
GAMMA_API_BASE = config.POLYMARKET_GAMMA_API_URL or "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

# Rate-limit semaphores
_KALSHI_SEM = asyncio.Semaphore(5)    # 5 concurrent Kalshi requests
_POLY_SEM = asyncio.Semaphore(10)     # 10 concurrent Polymarket requests


# ── Data models ──────────────────────────────────────────────────

@dataclass
class LivePrice:
    """Normalised price for one side of a binary market (Yes/No)."""
    venue: str
    venue_market_id: str
    yes_bid: Optional[float] = None   # Best bid for Yes [0, 1]
    yes_ask: Optional[float] = None   # Best ask for Yes [0, 1]
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def mid(self) -> Optional[float]:
        """Mid-price for Yes."""
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2.0
        return self.yes_bid or self.yes_ask

    @property
    def spread_width(self) -> Optional[float]:
        """Venue-internal bid-ask spread."""
        if self.yes_bid is not None and self.yes_ask is not None:
            return self.yes_ask - self.yes_bid
        return None


@dataclass
class SpreadOpportunity:
    """Actionable cross-venue spread."""
    pair_key: str
    title_a: str
    title_b: str
    venue_buy: str          # Buy Yes here (at ask)
    venue_sell: str         # Sell Yes here (their bid)
    buy_price: float        # ask price on buy venue
    sell_price: float       # bid price on sell venue
    gross_spread: float     # sell_price - buy_price
    roi_pct: float          # gross_spread / buy_price * 100
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


# ── Database ─────────────────────────────────────────────────────

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
    # statement_cache_size=0 required for Supabase pgbouncer compatibility
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


# ── Kalshi price fetcher ─────────────────────────────────────────

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


async def fetch_kalshi_price(
    session: aiohttp.ClientSession,
    ticker: str,
    api_key_id: Optional[str],
    private_key,
) -> Optional[LivePrice]:
    """
    Fetch live price for a Kalshi market by ticker.

    Uses GET /trade-api/v2/markets/{ticker}.
    Kalshi prices are in cents (0-99); we normalise to [0, 1].

    Args:
        session: aiohttp session.
        ticker: Kalshi market ticker (e.g. "KXBTC-26FEB14-T95250").
        api_key_id: Kalshi API key ID.
        private_key: Loaded RSA private key.

    Returns:
        LivePrice or None on failure.
    """
    path = f"{KALSHI_MARKETS_PATH}/{ticker}"
    url = f"{KALSHI_REST_BASE}{path}"
    headers: dict = {"Accept": "application/json"}

    if api_key_id and private_key:
        result = _sign_kalshi_request(private_key, "GET", path)
        if result:
            sig_b64, ts_ms = result
            headers["KALSHI-ACCESS-KEY"] = api_key_id
            headers["KALSHI-ACCESS-SIGNATURE"] = sig_b64
            headers["KALSHI-ACCESS-TIMESTAMP"] = ts_ms

    async with _KALSHI_SEM:
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("[Kalshi] %s → %d: %s", ticker, resp.status, body[:200])
                    return None
                data = await resp.json()
        except Exception as exc:
            logger.warning("[Kalshi] Failed to fetch %s: %s", ticker, exc)
            return None

    market = data.get("market") or data
    yes_bid_raw = market.get("yes_bid")
    yes_ask_raw = market.get("yes_ask")

    if yes_bid_raw is None and yes_ask_raw is None:
        logger.debug("[Kalshi] No bid/ask for %s", ticker)
        return None

    # Kalshi prices are in cents (0-99) → normalise to [0, 1]
    yes_bid = int(yes_bid_raw) / 100.0 if yes_bid_raw is not None else None
    yes_ask = int(yes_ask_raw) / 100.0 if yes_ask_raw is not None else None

    return LivePrice(
        venue="kalshi",
        venue_market_id=ticker,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
    )


# ── Polymarket price fetcher ─────────────────────────────────────

async def _fetch_gamma_market(
    session: aiohttp.ClientSession,
    market_id: str,
) -> tuple[Optional[str], Optional[str], Optional[list[float]]]:
    """
    Fetch Polymarket market data (token IDs + indicative prices) via Gamma API.

    Uses ``GET /markets/{id}`` (path parameter) which works for both
    numeric Gamma IDs and condition_id hex strings.

    Args:
        session: aiohttp session.
        market_id: Polymarket venue_market_id (Gamma internal ID or condition_id).

    Returns:
        Tuple of (yes_token_id, no_token_id, outcome_prices) or (None, None, None).
    """
    # Primary: GET /markets/{id} (single-object response)
    url = f"{GAMMA_API_BASE}/markets/{market_id}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                if isinstance(data, dict):
                    return _parse_gamma_market(data)
    except Exception as exc:
        logger.debug("[Polymarket] Gamma /%s failed: %s", market_id[:20], exc)

    # Fallback: GET /markets?id={id} (list response)
    url2 = f"{GAMMA_API_BASE}/markets"
    try:
        async with session.get(url2, params={"id": market_id}, timeout=aiohttp.ClientTimeout(total=10)) as resp:
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
    """
    Extract token IDs and outcome prices from a Gamma API market object.

    Args:
        market: Single market dict from the Gamma API.

    Returns:
        Tuple of (yes_token_id, no_token_id, outcome_prices).
    """
    # Parse token IDs
    token_ids_raw = market.get("clobTokenIds") or []
    if isinstance(token_ids_raw, str):
        try:
            token_ids_raw = json.loads(token_ids_raw)
        except (json.JSONDecodeError, TypeError):
            token_ids_raw = []

    yes_token = token_ids_raw[0] if len(token_ids_raw) > 0 else None
    no_token = token_ids_raw[1] if len(token_ids_raw) > 1 else None

    # Parse indicative prices
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


async def _fetch_clob_book(
    session: aiohttp.ClientSession,
    token_id: str,
) -> tuple[Optional[float], Optional[float]]:
    """
    Fetch best bid and best ask from Polymarket CLOB orderbook.

    Args:
        session: aiohttp session.
        token_id: CLOB token ID.

    Returns:
        Tuple of (best_bid, best_ask) or (None, None).
    """
    url = f"{CLOB_API_BASE}/book"
    params = {"token_id": token_id}

    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None, None
            data = await resp.json()
    except Exception as exc:
        logger.debug("[Polymarket] CLOB book failed for %s: %s", token_id, exc)
        return None, None

    bids = data.get("bids") or []
    asks = data.get("asks") or []

    best_bid = float(bids[0]["price"]) if bids else None
    best_ask = float(asks[0]["price"]) if asks else None

    return best_bid, best_ask


async def fetch_polymarket_price(
    session: aiohttp.ClientSession,
    market_id: str,
) -> Optional[LivePrice]:
    """
    Fetch live price for a Polymarket market.

    Strategy:
    1. Gamma API ``GET /markets/{id}`` → get clobTokenIds + indicative mid-prices
    2. CLOB ``/book`` → get real bid/ask for the Yes token
    3. If CLOB spread is unreasonably wide (>30%), use Gamma mid-prices instead
    4. Fall back to Gamma mid-prices if CLOB unavailable

    Args:
        session: aiohttp session.
        market_id: Polymarket venue_market_id (Gamma ID or condition_id).

    Returns:
        LivePrice or None.
    """
    async with _POLY_SEM:
        yes_token, _, gamma_prices = await _fetch_gamma_market(session, market_id)

        gamma_mid: Optional[float] = None
        if gamma_prices and len(gamma_prices) >= 1 and gamma_prices[0] is not None:
            gamma_mid = gamma_prices[0]

        if yes_token:
            # Try CLOB book for precise bid/ask
            best_bid, best_ask = await _fetch_clob_book(session, yes_token)
            if best_bid is not None and best_ask is not None:
                clob_spread = best_ask - best_bid
                # If CLOB spread is reasonable (<30 cents), use it
                if clob_spread < 0.30:
                    return LivePrice(
                        venue="polymarket",
                        venue_market_id=market_id,
                        yes_bid=best_bid,
                        yes_ask=best_ask,
                    )
                # CLOB spread too wide — fall through to Gamma mid

        # Fall back to Gamma mid-prices (treat as both bid and ask)
        if gamma_mid is not None:
            return LivePrice(
                venue="polymarket",
                venue_market_id=market_id,
                yes_bid=gamma_mid,
                yes_ask=gamma_mid,
            )

        return None


# ── Spread calculation ───────────────────────────────────────────

def calculate_spread(
    pair: EquivalentPair,
    price_a: LivePrice,
    price_b: LivePrice,
) -> Optional[SpreadOpportunity]:
    """
    Calculate cross-venue spread for an equivalent pair.

    Checks both directions:
    - Buy Yes on A, sell Yes on B (B_yes_bid - A_yes_ask)
    - Buy Yes on B, sell Yes on A (A_yes_bid - B_yes_ask)

    Takes the better direction. Accounts for outcome mapping
    (if Yes on A maps to No on B, prices are inverted).

    Args:
        pair: EquivalentPair with outcome mapping.
        price_a: LivePrice for market A.
        price_b: LivePrice for market B.

    Returns:
        SpreadOpportunity or None if no positive spread.
    """
    # ── Outcome alignment ────────────────────────────────────────
    # Check if Yes→Yes or Yes→No mapping
    mapping = pair.outcome_mapping or {}
    inverted = False

    # Detect inversion: if "Yes" maps to "No" on the other side
    for key, val in mapping.items():
        k = str(key).lower().strip()
        v = str(val).lower().strip()
        if k == "yes" and v == "no":
            inverted = True
            break
        if k == "no" and v == "yes":
            inverted = True
            break

    a_yes_bid = price_a.yes_bid
    a_yes_ask = price_a.yes_ask
    b_yes_bid = price_b.yes_bid
    b_yes_ask = price_b.yes_ask

    if inverted:
        # Flip B's Yes to be B's No (= 1 - B's Yes)
        if b_yes_bid is not None and b_yes_ask is not None:
            b_yes_bid, b_yes_ask = 1.0 - b_yes_ask, 1.0 - b_yes_bid
        else:
            return None

    # Need all four prices
    if any(p is None for p in [a_yes_bid, a_yes_ask, b_yes_bid, b_yes_ask]):
        return None

    # ── Direction 1: Buy Yes on A (at ask), sell Yes on B (at bid) ─
    spread_1 = b_yes_bid - a_yes_ask
    roi_1 = (spread_1 / a_yes_ask * 100) if a_yes_ask > 0 else 0.0

    # ── Direction 2: Buy Yes on B (at ask), sell Yes on A (at bid) ─
    spread_2 = a_yes_bid - b_yes_ask
    roi_2 = (spread_2 / b_yes_ask * 100) if b_yes_ask > 0 else 0.0

    # Pick the better direction
    if spread_1 >= spread_2 and spread_1 > MIN_GROSS_SPREAD:
        return SpreadOpportunity(
            pair_key=pair.pair_key,
            title_a=pair.title_a,
            title_b=pair.title_b,
            venue_buy=price_a.venue,
            venue_sell=price_b.venue,
            buy_price=a_yes_ask,
            sell_price=b_yes_bid,
            gross_spread=round(spread_1, 4),
            roi_pct=round(roi_1, 2),
        )
    elif spread_2 > MIN_GROSS_SPREAD:
        return SpreadOpportunity(
            pair_key=pair.pair_key,
            title_a=pair.title_a,
            title_b=pair.title_b,
            venue_buy=price_b.venue,
            venue_sell=price_a.venue,
            buy_price=b_yes_ask,
            sell_price=a_yes_bid,
            gross_spread=round(spread_2, 4),
            roi_pct=round(roi_2, 2),
        )

    return None


# ── Display ──────────────────────────────────────────────────────

def _trunc(text: str, maxlen: int = 50) -> str:
    """Truncate text with ellipsis."""
    return text[:maxlen - 3] + "..." if len(text) > maxlen else text


def print_results(
    pairs: list[EquivalentPair],
    prices: dict[str, LivePrice],
    opportunities: list[SpreadOpportunity],
) -> None:
    """
    Print a formatted summary of all equivalent pairs, their prices,
    and any spread opportunities.

    Args:
        pairs: All equivalent pairs queried.
        prices: Dict of "venue:venue_market_id" → LivePrice.
        opportunities: Detected spread opportunities.
    """
    print()
    print("=" * 100)
    print(f"  SPREAD SCANNER RESULTS  |  {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Equivalent pairs: {len(pairs)}  |  Prices fetched: {len(prices)}  |  Opportunities: {len(opportunities)}")
    print("=" * 100)

    # ── All pairs with prices ────────────────────────────────────
    print()
    print("-" * 100)
    print(f"{'#':>3}  {'Pair (A vs B)':50}  {'A Yes Bid':>9}  {'A Yes Ask':>9}  {'B Yes Bid':>9}  {'B Yes Ask':>9}")
    print("-" * 100)

    for i, pair in enumerate(pairs, 1):
        key_a = f"{pair.venue_a}:{pair.vmid_a}"
        key_b = f"{pair.venue_b}:{pair.vmid_b}"
        pa = prices.get(key_a)
        pb = prices.get(key_b)

        label = _trunc(f"{pair.title_a}", 50)

        a_bid = f"{pa.yes_bid:.4f}" if pa and pa.yes_bid is not None else "  N/A"
        a_ask = f"{pa.yes_ask:.4f}" if pa and pa.yes_ask is not None else "  N/A"
        b_bid = f"{pb.yes_bid:.4f}" if pb and pb.yes_bid is not None else "  N/A"
        b_ask = f"{pb.yes_ask:.4f}" if pb and pb.yes_ask is not None else "  N/A"

        print(f"{i:3}  {label:50}  {a_bid:>9}  {a_ask:>9}  {b_bid:>9}  {b_ask:>9}")

    # ── Opportunities ────────────────────────────────────────────
    if not opportunities:
        print()
        print("  No positive spread opportunities detected.")
        print()
        return

    # Sort by ROI descending
    opportunities.sort(key=lambda o: o.roi_pct, reverse=True)

    print()
    print("=" * 100)
    print("  SPREAD OPPORTUNITIES (sorted by ROI)")
    print("=" * 100)
    print()
    print(f"{'#':>3}  {'Market':45}  {'Buy @':>8}  {'Sell @':>8}  {'Spread':>8}  {'ROI %':>7}  {'Action'}")
    print("-" * 100)

    for i, opp in enumerate(opportunities, 1):
        label = _trunc(opp.title_a, 45)
        action = f"BUY {opp.venue_buy.upper()} -> SELL {opp.venue_sell.upper()}"
        spread_cents = opp.gross_spread * 100

        print(
            f"{i:3}  {label:45}  "
            f"${opp.buy_price:.4f}  ${opp.sell_price:.4f}  "
            f"{spread_cents:+7.2f}c  {opp.roi_pct:+6.2f}%  {action}"
        )

    print()
    print(f"  Total opportunities: {len(opportunities)}")
    positive = [o for o in opportunities if o.gross_spread > 0.01]
    print(f"  Opportunities > 1 cent spread: {len(positive)}")
    if positive:
        best = positive[0]
        print(f"  Best spread: {best.gross_spread*100:.2f} cents ({best.roi_pct:+.2f}% ROI) on {_trunc(best.title_a, 40)}")
    print()


# ── Main scan ────────────────────────────────────────────────────

async def scan() -> list[SpreadOpportunity]:
    """
    Execute a single spread scan.

    Steps:
    1. Fetch equivalent pairs from database
    2. Fetch live prices from both venues concurrently
    3. Calculate spreads
    4. Print and return results

    Returns:
        List of detected SpreadOpportunity objects.
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
    markets_to_price: dict[str, tuple[str, str]] = {}  # "venue:vmid" → (venue, vmid)
    for pair in pairs:
        key_a = f"{pair.venue_a}:{pair.vmid_a}"
        key_b = f"{pair.venue_b}:{pair.vmid_b}"
        markets_to_price[key_a] = (pair.venue_a, pair.vmid_a)
        markets_to_price[key_b] = (pair.venue_b, pair.vmid_b)

    logger.info(
        "Fetching live prices for %d unique markets...",
        len(markets_to_price),
    )

    # ── 3. Fetch prices concurrently ─────────────────────────────
    kalshi_key_id, kalshi_pk = _build_kalshi_auth()
    if not kalshi_key_id:
        logger.warning("Kalshi auth not configured — Kalshi prices may fail")

    prices: dict[str, LivePrice] = {}
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = {}
        for key, (venue, vmid) in markets_to_price.items():
            if venue == "kalshi":
                tasks[key] = fetch_kalshi_price(session, vmid, kalshi_key_id, kalshi_pk)
            elif venue == "polymarket":
                tasks[key] = fetch_polymarket_price(session, vmid)
            else:
                logger.debug("Skipping unsupported venue: %s", venue)

        # Gather all concurrently
        keys = list(tasks.keys())
        results = await asyncio.gather(
            *[tasks[k] for k in keys],
            return_exceptions=True,
        )

        for key, result in zip(keys, results):
            if isinstance(result, Exception):
                logger.warning("Price fetch failed for %s: %s", key, result)
            elif result is not None:
                prices[key] = result

    logger.info(
        "Fetched %d/%d prices successfully",
        len(prices),
        len(markets_to_price),
    )

    # ── 4. Calculate spreads ─────────────────────────────────────
    opportunities: list[SpreadOpportunity] = []
    for pair in pairs:
        key_a = f"{pair.venue_a}:{pair.vmid_a}"
        key_b = f"{pair.venue_b}:{pair.vmid_b}"
        pa = prices.get(key_a)
        pb = prices.get(key_b)
        if not pa or not pb:
            continue

        opp = calculate_spread(pair, pa, pb)
        if opp is not None:
            opportunities.append(opp)

    # ── 5. Display results ───────────────────────────────────────
    print_results(pairs, prices, opportunities)

    return opportunities


# ── Entry point ──────────────────────────────────────────────────

def main() -> None:
    """Entry point for the spread scanner."""
    from dotenv import load_dotenv
    load_dotenv()

    logger.info("Starting spread scanner...")
    start = time.monotonic()
    opportunities = asyncio.run(scan())
    elapsed = time.monotonic() - start
    logger.info("Scan completed in %.1fs — %d opportunities found", elapsed, len(opportunities))


if __name__ == "__main__":
    main()

