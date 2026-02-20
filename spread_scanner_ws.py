"""
Real-time WebSocket spread scanner for cross-venue arbitrage monitoring.

Connects to Kalshi and Polymarket WebSocket feeds for live orderbook
updates, tracks spread evolution over time, identifies arbitrage windows,
and generates matplotlib graphs showing PnL / price divergence.

Architecture:
    1. Bootstrap — REST fetch of initial orderbooks + equivalent pairs
    2. Live feed — Kalshi WS (orderbook_delta) + Polymarket WS (CLOB)
    3. On every update — recalculate spreads, record time-series snapshot
    4. Periodic — REST refresh as safety net (every 30 s)
    5. On exit — generate matplotlib graph + print final report

Kalshi WS: wss://api.elections.kalshi.com/trade-api/ws/v2
    Requires RSA-PSS auth headers on handshake.
    Channel: orderbook_delta — sends incremental [[price_cents, qty], ...]
    Subscription: {"id":N,"cmd":"subscribe","params":{"channels":["orderbook_delta"],"market_tickers":[...]}}

Polymarket WS: wss://ws-subscriptions-clob.polymarket.com/ws/market
    No auth required (public).
    Subscription: {"type":"market","assets_ids":["token1","token2",...]}
    Sends price_change events per asset — we refetch full CLOB book on each.

Fee model:
    Kalshi  ~7 % of profit on the winning contract.
    Polymarket CLOB  0 % taker fee (current).

Usage:
    python spread_scanner_ws.py                          # 1-hour default
    python spread_scanner_ws.py --duration 86400         # 24-hour session
    python spread_scanner_ws.py -d 3600 -o report.png   # custom output

Requires:
    DATABASE_URL, KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH in .env
    pip: aiohttp asyncpg websockets matplotlib
"""

import asyncio
import argparse
import json
import logging
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import Optional

import aiohttp
import asyncpg
import websockets
from websockets.exceptions import ConnectionClosed

# Headless matplotlib backend — must be set before pyplot import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.dates as mdates        # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# ── Project imports ──────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402
from discovery.kalshi_poller import (  # noqa: E402
    _load_kalshi_private_key,
    _sign_kalshi_request,
)
from spread_scanner import (  # noqa: E402
    # Data models
    BookLevel,
    VenueBook,
    ArbOpportunity,
    EquivalentPair,
    # Functions
    fetch_equivalent_pairs,
    fetch_kalshi_book,
    fetch_polymarket_book,
    find_arb_opportunities,
    _build_kalshi_auth,
    _pick_resolution_date,
    _fetch_gamma_tokens,
    # Constants
    KALSHI_FEE_RATE,
    KALSHI_TAKER_RATE,
    POLYMARKET_FEE_RATE,
    TRADER_CAPITAL,
    MIN_PROFIT_THRESHOLD,
)


# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("spread_ws")


# ── WS endpoint constants ───────────────────────────────────
KALSHI_WS_URL = (
    "wss://demo-api.kalshi.co/trade-api/ws/v2"
    if config.KALSHI_USE_DEMO
    else "wss://api.elections.kalshi.com/trade-api/ws/v2"
)
KALSHI_WS_SIGN_PATH = "/trade-api/ws/v2"

POLY_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# REST refresh interval — keeps books fresh even if WS drops a delta
REST_REFRESH_INTERVAL = 30  # seconds

# Minimum interval between spread recalculations to avoid CPU churn
# on markets with high tick frequency
RECALC_THROTTLE = 0.25  # seconds


# ═══════════════════════════════════════════════════════════════
# TIME-SERIES DATA MODELS
# ═══════════════════════════════════════════════════════════════

REFERENCE_CAPITAL = 10_000.0  # USD — the capital level shown in displays


@dataclass
class SpreadSnapshot:
    """Point-in-time measurement of a cross-venue spread."""
    timestamp: datetime
    pair_key: str
    cost_per_contract: float  # ask_a + ask_b
    gross_edge: float         # 1.0 − cost
    net_edge: float           # guaranteed profit per contract after fees
    ask_a: float
    ask_b: float
    venue_a: str
    venue_b: str
    buy_a_side: str
    buy_b_side: str
    source: str               # ws_kalshi | ws_poly | rest_refresh
    pnl_10k: float = 0.0     # guaranteed profit at $10 k capital
    pnl_budget: float = 0.0  # guaranteed profit at TRADER_CAPITAL
    daily_yield_bps: float = 0.0  # bps per day at TRADER_CAPITAL


@dataclass
class PairTimeSeries:
    """Rolling time-series for one equivalent pair."""
    pair_key: str
    title_a: str
    title_b: str
    venue_a: str
    venue_b: str
    snapshots: list[SpreadSnapshot] = field(default_factory=list)
    max_edge: float = -999.0
    max_edge_time: Optional[datetime] = None
    min_edge: float = 999.0
    min_edge_time: Optional[datetime] = None
    max_pnl_10k: float = -999_999.0
    max_pnl_10k_time: Optional[datetime] = None

    def record(self, snap: SpreadSnapshot) -> None:
        self.snapshots.append(snap)
        if snap.gross_edge > self.max_edge:
            self.max_edge = snap.gross_edge
            self.max_edge_time = snap.timestamp
        if snap.gross_edge < self.min_edge:
            self.min_edge = snap.gross_edge
            self.min_edge_time = snap.timestamp
        if snap.pnl_10k > self.max_pnl_10k:
            self.max_pnl_10k = snap.pnl_10k
            self.max_pnl_10k_time = snap.timestamp


# ═══════════════════════════════════════════════════════════════
# LIVE ORDERBOOK MANAGER
# ═══════════════════════════════════════════════════════════════

class LiveOrderbookManager:
    """
    Maintains in-memory orderbooks for all tracked markets.

    On every book update, recalculates spreads and records snapshots.
    Throttles recalculation to avoid CPU churn from high-frequency ticks.
    """

    def __init__(self, pairs: list[EquivalentPair]):
        self.pairs = pairs
        self.books: dict[str, VenueBook] = {}
        self.series: dict[str, PairTimeSeries] = {}
        self._update_count = 0
        self._last_recalc_mono = 0.0
        self._lock = asyncio.Lock()

        for pair in pairs:
            self.series[pair.pair_key] = PairTimeSeries(
                pair_key=pair.pair_key,
                title_a=pair.title_a,
                title_b=pair.title_b,
                venue_a=pair.venue_a,
                venue_b=pair.venue_b,
            )

    async def update_book(self, key: str, book: VenueBook) -> None:
        async with self._lock:
            self.books[key] = book
            self._update_count += 1

    async def recalculate(self, source: str = "unknown") -> list[ArbOpportunity]:
        """Recalculate all spreads and record snapshots (throttled)."""
        now_mono = time.monotonic()
        if now_mono - self._last_recalc_mono < RECALC_THROTTLE:
            return []
        self._last_recalc_mono = now_mono

        async with self._lock:
            return self._recalculate_locked(source)

    def _recalculate_locked(self, source: str) -> list[ArbOpportunity]:
        now = datetime.now(UTC)
        all_opps: list[ArbOpportunity] = []

        for pair in self.pairs:
            key_a = f"{pair.venue_a}:{pair.vmid_a}"
            key_b = f"{pair.venue_b}:{pair.vmid_b}"
            ba = self.books.get(key_a)
            bb = self.books.get(key_b)
            if not ba or not bb:
                continue

            opps = find_arb_opportunities(pair, ba, bb)
            all_opps.extend(opps)

            ts = self.series.get(pair.pair_key)
            if not ts:
                continue

            if opps:
                best = max(opps, key=lambda o: o.gross_edge)
                # Extract $10k PnL from the capital ladder (first rung)
                pnl_10k = 0.0
                if best.capital_ladder:
                    pnl_10k = best.capital_ladder[0].guaranteed_profit
                ts.record(SpreadSnapshot(
                    timestamp=now,
                    pair_key=pair.pair_key,
                    cost_per_contract=best.total_cost_1,
                    gross_edge=best.gross_edge,
                    net_edge=best.net_profit_1,
                    ask_a=best.ask_a,
                    ask_b=best.ask_b,
                    venue_a=best.venue_a,
                    venue_b=best.venue_b,
                    buy_a_side=best.buy_a_side,
                    buy_b_side=best.buy_b_side,
                    source=source,
                    pnl_10k=pnl_10k,
                    pnl_budget=best.pnl_at_budget,
                    daily_yield_bps=best.daily_yield_bps,
                ))
            else:
                # No opportunity; record current state with zero edge
                ya = ba.yes_ask_top
                nb = bb.no_ask_top
                if ya is not None and nb is not None:
                    cost = ya + nb
                    ts.record(SpreadSnapshot(
                        timestamp=now,
                        pair_key=pair.pair_key,
                        cost_per_contract=cost,
                        gross_edge=1.0 - cost,
                        net_edge=0.0,
                        ask_a=ya,
                        ask_b=nb,
                        venue_a=pair.venue_a,
                        venue_b=pair.venue_b,
                        buy_a_side="yes",
                        buy_b_side="no",
                        source=source,
                        pnl_10k=0.0,
                        pnl_budget=0.0,
                        daily_yield_bps=0.0,
                    ))

        return all_opps


# ═══════════════════════════════════════════════════════════════
# KALSHI WEBSOCKET FEED
# ═══════════════════════════════════════════════════════════════

class KalshiWSFeed:
    """
    WebSocket feed for Kalshi orderbook deltas.

    Subscribes to ``orderbook_delta`` for specific tickers.
    On each delta, applies changes to the in-memory book, then triggers
    a spread recalculation.  A parallel REST loop refreshes full books
    every REST_REFRESH_INTERVAL seconds as a consistency safety net.
    """

    def __init__(
        self,
        tickers: list[str],
        manager: LiveOrderbookManager,
        api_key_id: Optional[str] = None,
        private_key=None,
    ):
        self.tickers = tickers
        self.manager = manager
        self.api_key_id = api_key_id
        self.private_key = private_key
        self._running = False
        self._ws = None
        self.update_count = 0

    # ── lifecycle ─────────────────────────────────────────────

    async def run(self, session: aiohttp.ClientSession) -> None:
        """Start WS listener + REST refresh. Runs until stop() is called."""
        self._running = True
        self._session = session
        await asyncio.gather(
            self._ws_loop(),
            self._rest_refresh_loop(),
            return_exceptions=True,
        )

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    # ── authentication ────────────────────────────────────────

    def _auth_headers(self) -> dict:
        """RSA-PSS signed headers for the WS handshake."""
        if not self.api_key_id or not self.private_key:
            return {}
        result = _sign_kalshi_request(self.private_key, "GET", KALSHI_WS_SIGN_PATH)
        if not result:
            return {}
        sig, ts = result
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    # ── WebSocket loop ────────────────────────────────────────

    async def _ws_loop(self) -> None:
        while self._running:
            try:
                headers = self._auth_headers()
                kw = {"ping_interval": 20, "ping_timeout": 10}
                if headers:
                    kw["additional_headers"] = headers

                logger.info("[Kalshi WS] Connecting to %s (%d tickers)", KALSHI_WS_URL, len(self.tickers))
                async with websockets.connect(KALSHI_WS_URL, **kw) as ws:
                    self._ws = ws
                    logger.info("[Kalshi WS] Connected — subscribing to orderbook_delta")

                    # Kalshi WS v2: subscribe to orderbook deltas
                    await ws.send(json.dumps({
                        "id": 1,
                        "cmd": "subscribe",
                        "params": {
                            "channels": ["orderbook_delta"],
                            "market_tickers": self.tickers,
                        },
                    }))

                    async for raw in ws:
                        if not self._running:
                            break
                        await self._handle(raw)

            except ConnectionClosed as e:
                logger.warning("[Kalshi WS] Closed (%s). Reconnecting in 5 s…", e)
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("[Kalshi WS] Error: %s. Reconnecting in 10 s…", e)
                await asyncio.sleep(10)

    async def _handle(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type", "")

        if msg_type == "subscribed":
            logger.info("[Kalshi WS] Subscription confirmed: %s", data.get("msg", {}).get("channel", ""))
            return

        if msg_type == "error":
            logger.warning("[Kalshi WS] Server error: %s", data.get("msg"))
            return

        if msg_type != "orderbook_delta":
            return

        msg = data.get("msg", {})
        ticker = msg.get("market_ticker", "")
        if not ticker:
            return

        key = f"kalshi:{ticker}"
        yes_delta = msg.get("yes", [])
        no_delta = msg.get("no", [])

        existing = self.manager.books.get(key)
        if existing:
            updated = self._apply_delta(existing, yes_delta, no_delta)
            await self.manager.update_book(key, updated)
        else:
            # First delta with no bootstrap — fall back to REST snapshot
            book = await fetch_kalshi_book(
                self._session, ticker, self.api_key_id, self.private_key,
            )
            if book:
                await self.manager.update_book(key, book)

        self.update_count += 1
        await self.manager.recalculate(source="ws_kalshi")

    def _apply_delta(
        self, existing: VenueBook, yes_delta: list, no_delta: list,
    ) -> VenueBook:
        """
        Apply Kalshi orderbook delta to an existing book.

        Kalshi ``orderbook_delta`` sends raw bid-side updates for YES and NO.
        The delta format is [[price_cents, new_qty], …].  A qty of 0 means
        "remove that price level".

        Since our arb logic only needs asks, and Kalshi's orderbook semantics
        are YES asks = inverted NO bids and vice-versa, we apply deltas to
        the opposite side's asks.
        """

        def _apply(asks: list[BookLevel], delta: list, invert: bool) -> list[BookLevel]:
            level_map = {round(a.price, 6): a.size for a in asks}
            for entry in delta:
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                raw_price = float(entry[0]) / 100.0
                size = float(entry[1])
                price = round(1.0 - raw_price, 6) if invert else raw_price
                if size <= 0:
                    level_map.pop(price, None)
                elif 0 < price < 1:
                    level_map[price] = size
            result = [BookLevel(price=p, size=s) for p, s in level_map.items() if s > 0]
            result.sort(key=lambda l: l.price)
            return result

        # YES asks are derived from inverted NO bids → apply no_delta
        # NO asks are derived from inverted YES bids → apply yes_delta
        return VenueBook(
            venue=existing.venue,
            venue_market_id=existing.venue_market_id,
            yes_asks=_apply(existing.yes_asks, no_delta, invert=True),
            yes_bids=existing.yes_bids,
            no_asks=_apply(existing.no_asks, yes_delta, invert=True),
            no_bids=existing.no_bids,
            source="ws_delta",
            resolution_date=existing.resolution_date,
        )

    # ── periodic REST safety net ──────────────────────────────

    async def _rest_refresh_loop(self) -> None:
        while self._running:
            await asyncio.sleep(REST_REFRESH_INTERVAL)
            if not self._running:
                break
            logger.debug("[Kalshi REST] Periodic refresh (%d tickers)", len(self.tickers))
            for ticker in self.tickers:
                if not self._running:
                    break
                try:
                    book = await fetch_kalshi_book(
                        self._session, ticker, self.api_key_id, self.private_key,
                    )
                    if book:
                        await self.manager.update_book(f"kalshi:{ticker}", book)
                except Exception as exc:
                    logger.debug("[Kalshi REST] Refresh %s failed: %s", ticker, exc)
                await asyncio.sleep(0.05)  # 50 ms gap to avoid 429s
            await self.manager.recalculate(source="rest_kalshi")


# ═══════════════════════════════════════════════════════════════
# POLYMARKET WEBSOCKET FEED
# ═══════════════════════════════════════════════════════════════

class PolymarketWSFeed:
    """
    WebSocket feed for Polymarket CLOB price changes.

    Subscribes with specific asset (token) IDs.  On each price-change
    event, re-fetches the full CLOB book via REST (the WS message itself
    is lightweight and doesn't contain depth).  A parallel REST loop
    refreshes all books every REST_REFRESH_INTERVAL seconds.
    """

    def __init__(
        self,
        market_ids: list[str],
        token_map: dict[str, tuple[Optional[str], Optional[str]]],
        manager: LiveOrderbookManager,
    ):
        self.market_ids = market_ids
        self.token_map = token_map  # market_id → (yes_token, no_token)
        self.manager = manager
        self._running = False
        self._ws = None
        self.update_count = 0

        # Reverse lookup: token_id → market_id
        self._tok_to_mid: dict[str, str] = {}
        for mid, (yt, nt) in token_map.items():
            if yt:
                self._tok_to_mid[yt] = mid
            if nt:
                self._tok_to_mid[nt] = mid

    # ── lifecycle ─────────────────────────────────────────────

    async def run(self, session: aiohttp.ClientSession) -> None:
        self._running = True
        self._session = session
        await asyncio.gather(
            self._ws_loop(),
            self._rest_refresh_loop(),
            return_exceptions=True,
        )

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    # ── WebSocket loop ────────────────────────────────────────

    async def _ws_loop(self) -> None:
        all_tokens = [
            t
            for mid, (yt, nt) in self.token_map.items()
            for t in (yt, nt)
            if t
        ]
        if not all_tokens:
            logger.warning("[Poly WS] No token IDs — WS feed inactive")
            return

        while self._running:
            try:
                logger.info("[Poly WS] Connecting to %s (%d tokens)", POLY_WS_URL, len(all_tokens))
                async with websockets.connect(
                    POLY_WS_URL, ping_interval=25, ping_timeout=15,
                ) as ws:
                    self._ws = ws
                    logger.info("[Poly WS] Connected — subscribing")

                    # Polymarket CLOB subscription format
                    await ws.send(json.dumps({
                        "type": "market",
                        "assets_ids": all_tokens,
                    }))

                    async for raw in ws:
                        if not self._running:
                            break
                        await self._handle(raw)

            except ConnectionClosed as e:
                logger.warning("[Poly WS] Closed (%s). Reconnecting in 5 s…", e)
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("[Poly WS] Error: %s. Reconnecting in 10 s…", e)
                await asyncio.sleep(10)

    async def _handle(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Polymarket CLOB WS sends either a single dict or a list of dicts
        items: list[dict] = []
        if isinstance(payload, list):
            items = [d for d in payload if isinstance(d, dict)]
        elif isinstance(payload, dict):
            items = [payload]
        else:
            return

        # Collect unique market IDs that changed so we refetch each once
        changed_mids: set[str] = set()
        for data in items:
            asset_id = (
                data.get("asset_id")
                or data.get("token_id")
                or data.get("market")
                or ""
            )
            mid = self._tok_to_mid.get(asset_id)
            if mid:
                changed_mids.add(mid)

        for mid in changed_mids:
            try:
                book = await fetch_polymarket_book(self._session, mid)
                if book:
                    await self.manager.update_book(f"polymarket:{mid}", book)
                    self.update_count += 1
            except Exception as exc:
                logger.debug("[Poly WS] Refetch failed for %s: %s", mid[:20], exc)

        if changed_mids:
            await self.manager.recalculate(source="ws_poly")

    # ── periodic REST safety net ──────────────────────────────

    async def _rest_refresh_loop(self) -> None:
        while self._running:
            await asyncio.sleep(REST_REFRESH_INTERVAL)
            if not self._running:
                break
            logger.debug("[Poly REST] Periodic refresh (%d markets)", len(self.market_ids))
            for mid in self.market_ids:
                if not self._running:
                    break
                try:
                    book = await fetch_polymarket_book(self._session, mid)
                    if book:
                        await self.manager.update_book(f"polymarket:{mid}", book)
                except Exception as exc:
                    logger.debug("[Poly REST] Refresh %s failed: %s", mid[:20], exc)
            await self.manager.recalculate(source="rest_poly")


# ═══════════════════════════════════════════════════════════════
# GRAPH GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_graph(
    series: dict[str, PairTimeSeries],
    output_path: str = "spread_divergence.png",
    duration_label: str = "",
) -> str:
    """
    Generate a three-panel matplotlib figure:

      1. Gross edge over time (1 − cost per contract)
      2. Guaranteed PnL on $10 k capital — green fill when positive
      3. Venue ask prices (solid = venue A, dashed = venue B)

    Returns the output file path, or "" if nothing to plot.
    """
    active = {k: v for k, v in series.items() if v.snapshots}
    if not active:
        logger.warning("No time-series data to graph")
        return ""

    n = min(len(active), 8)
    fig, axes = plt.subplots(3, 1, figsize=(16, max(10, 3 * n + 2)), sharex=True)

    fig.suptitle(
        f"Cross-Venue Spread Scanner — Live WebSocket Monitor\n"
        f"{duration_label}  |  {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=13, fontweight="bold",
    )

    palette = plt.cm.tab10.colors

    # ── Panel 1: gross edge ──────────────────────────────────
    ax1 = axes[0]
    ax1.set_title("Gross Edge (1 − cost per contract)", fontsize=10)
    ax1.set_ylabel("Edge ($)")
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    for idx, (_, ts) in enumerate(active.items()):
        times = [s.timestamp for s in ts.snapshots]
        edges = [s.gross_edge for s in ts.snapshots]
        colour = palette[idx % len(palette)]
        label = f"{ts.title_a[:28]} ({ts.venue_a[:4]}/{ts.venue_b[:4]})"
        ax1.plot(times, edges, label=label, color=colour, linewidth=1.1, alpha=0.85)
        if ts.max_edge_time:
            ax1.annotate(
                f"max {ts.max_edge:.4f}",
                xy=(ts.max_edge_time, ts.max_edge),
                fontsize=6, color=colour,
                xytext=(5, 5), textcoords="offset points",
            )

    ax1.legend(fontsize=6, loc="upper left", ncol=2)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: PnL on $10k capital ─────────────────────────
    ax2 = axes[1]
    ax2.set_title("Guaranteed PnL on $10k Capital (after fees)", fontsize=10)
    ax2.set_ylabel("PnL ($)")
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

    for idx, (_, ts) in enumerate(active.items()):
        times = [s.timestamp for s in ts.snapshots]
        pnls = [s.pnl_10k for s in ts.snapshots]
        colour = palette[idx % len(palette)]
        ax2.plot(times, pnls, color=colour, linewidth=1.1, alpha=0.85)
        ax2.fill_between(times, pnls, 0, where=[p > 0 for p in pnls], color="green", alpha=0.08)
        ax2.fill_between(times, pnls, 0, where=[p <= 0 for p in pnls], color="red", alpha=0.08)
        # Annotate max PnL@10k
        if ts.max_pnl_10k > 0 and ts.max_pnl_10k_time:
            ax2.annotate(
                f"${ts.max_pnl_10k:+,.0f}",
                xy=(ts.max_pnl_10k_time, ts.max_pnl_10k),
                fontsize=6, color=colour,
                xytext=(5, 5), textcoords="offset points",
            )

    ax2.grid(True, alpha=0.3)

    # ── Panel 3: venue ask prices (divergence) ───────────────
    ax3 = axes[2]
    ax3.set_title("Ask Prices per Venue (price divergence)", fontsize=10)
    ax3.set_ylabel("Price ($)")
    ax3.set_xlabel("Time (UTC)")

    for idx, (_, ts) in enumerate(active.items()):
        times = [s.timestamp for s in ts.snapshots]
        aa = [s.ask_a for s in ts.snapshots]
        ab = [s.ask_b for s in ts.snapshots]
        colour = palette[idx % len(palette)]
        ax3.plot(times, aa, color=colour, linewidth=1.0, alpha=0.7, linestyle="-",
                 label=f"{ts.title_a[:18]} {ts.venue_a[:4]}")
        ax3.plot(times, ab, color=colour, linewidth=1.0, alpha=0.7, linestyle="--",
                 label=f"{ts.title_a[:18]} {ts.venue_b[:4]}")

    ax3.legend(fontsize=5, loc="upper left", ncol=2)
    ax3.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=24))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Graph saved to %s", output_path)
    return output_path


# ═══════════════════════════════════════════════════════════════
# CONSOLE OUTPUT
# ═══════════════════════════════════════════════════════════════

def _trunc(text: str, n: int = 40) -> str:
    return text[:n - 3] + "…" if len(text) > n else text


def print_live_summary(
    manager: LiveOrderbookManager,
    elapsed: float,
    k_updates: int,
    p_updates: int,
) -> None:
    """Periodic console summary while the monitor runs."""
    now = datetime.now(UTC)
    up = str(timedelta(seconds=int(elapsed)))

    print()
    print("=" * 120)
    print(
        f"  LIVE SPREAD MONITOR  |  "
        f"{now.strftime('%Y-%m-%d %H:%M:%S UTC')}  |  "
        f"Uptime: {up}  |  "
        f"WS updates: Kalshi={k_updates}  Poly={p_updates}"
    )
    budget_k = TRADER_CAPITAL / 1000
    print("=" * 140)

    header = (
        f"  {'Status':8}  {'Pair':40}  "
        f"{'Edge':>8}  {'Net':>8}  "
        f"{'PnL@10k':>10}  "
        f"{'PnL@$' + f'{budget_k:.0f}k':>9}  "
        f"{'DyYld':>8}  {'Max Edge':>9}  {'Snaps':>6}"
    )
    print(header)
    print("  " + "-" * 136)

    for ts in manager.series.values():
        if not ts.snapshots:
            continue
        latest = ts.snapshots[-1]
        if latest.net_edge > 0:
            status = "ARB"
        elif latest.gross_edge > 0:
            status = "EDGE"
        else:
            status = "---"

        mx = f"{ts.max_edge:+.4f}" if ts.max_edge_time else "N/A"
        pnl = f"${latest.pnl_10k:>+,.0f}" if latest.pnl_10k != 0 else "$0"
        pnl_b = f"${latest.pnl_budget:>+,.0f}" if latest.pnl_budget != 0 else "$0"
        dy = f"{latest.daily_yield_bps:>7.3f}" if latest.daily_yield_bps else " 0.000"
        print(
            f"  {status:8}  {_trunc(ts.title_a, 40):40}  "
            f"{latest.gross_edge:>+8.4f}  {latest.net_edge:>+8.4f}  "
            f"{pnl:>10}  {pnl_b:>9}  "
            f"{dy:>8}  {mx:>9}  {len(ts.snapshots):>6}"
        )

    print("=" * 140)


def print_final_report(
    manager: LiveOrderbookManager,
    duration: float,
    k_updates: int,
    p_updates: int,
    graph_path: str,
) -> None:
    """Session-end summary with per-pair statistics."""
    total_snaps = sum(len(ts.snapshots) for ts in manager.series.values())
    up = str(timedelta(seconds=int(duration)))

    print()
    print("=" * 130)
    print("  FINAL SESSION REPORT")
    print("=" * 130)
    print(f"  Duration: {up}")
    print(f"  WebSocket updates: Kalshi={k_updates}  Polymarket={p_updates}")
    print(f"  Total snapshots recorded: {total_snaps}")
    if graph_path:
        print(f"  Graph: {graph_path}")
    print()

    budget_k = TRADER_CAPITAL / 1000
    header = (
        f"  {'Pair':40}  {'Snaps':>6}  "
        f"{'Max Edge':>9}  {'Max Time':>10}  "
        f"{'Min Edge':>9}  {'Avg Edge':>9}  "
        f"{'MaxPnL@10k':>11}  {'AvgPnL@10k':>11}  "
        f"{'AvgDyYld':>9}"
    )
    print(header)
    print("  " + "-" * 140)

    for ts in manager.series.values():
        if not ts.snapshots:
            continue
        avg_edge = sum(s.gross_edge for s in ts.snapshots) / len(ts.snapshots)
        avg_pnl = sum(s.pnl_10k for s in ts.snapshots) / len(ts.snapshots)
        avg_dy = sum(s.daily_yield_bps for s in ts.snapshots) / len(ts.snapshots)
        mt = ts.max_edge_time.strftime("%H:%M:%S") if ts.max_edge_time else "N/A"
        max_pnl_str = f"${ts.max_pnl_10k:>+,.0f}" if ts.max_pnl_10k > -999_000 else "N/A"
        avg_pnl_str = f"${avg_pnl:>+,.0f}"
        avg_dy_str = f"{avg_dy:>8.3f}" if avg_dy else "   0.000"
        print(
            f"  {_trunc(ts.title_a, 40):40}  {len(ts.snapshots):>6}  "
            f"{ts.max_edge:>+9.4f}  {mt:>10}  "
            f"{ts.min_edge:>+9.4f}  {avg_edge:>+9.4f}  "
            f"{max_pnl_str:>11}  {avg_pnl_str:>11}  "
            f"{avg_dy_str:>9}"
        )

    # Aggregate arb windows — periods where net_edge > 0
    arb_pairs = 0
    profitable_10k = 0
    profitable_budget = 0
    for ts in manager.series.values():
        if any(s.net_edge > 0 for s in ts.snapshots):
            arb_pairs += 1
        if ts.max_pnl_10k > 0:
            profitable_10k += 1
        if any(s.pnl_budget > MIN_PROFIT_THRESHOLD for s in ts.snapshots):
            profitable_budget += 1

    print()
    print(f"  Pairs with at least one ARB window: {arb_pairs}/{len(manager.series)}")
    print(f"  Pairs profitable at $10k capital:    {profitable_10k}/{len(manager.series)}")
    print(
        f"  Pairs actionable at ${TRADER_CAPITAL:,.0f} "
        f"(PnL ≥ ${MIN_PROFIT_THRESHOLD:.0f}):  "
        f"{profitable_budget}/{len(manager.series)}"
    )
    print("=" * 130)
    print()


# ═══════════════════════════════════════════════════════════════
# POLYMARKET TOKEN RESOLUTION
# ═══════════════════════════════════════════════════════════════

async def _resolve_poly_tokens(
    session: aiohttp.ClientSession,
    market_ids: list[str],
) -> dict[str, tuple[Optional[str], Optional[str]]]:
    """Resolve Polymarket condition IDs → (yes_token, no_token) via Gamma API."""
    result: dict[str, tuple[Optional[str], Optional[str]]] = {}
    for mid in market_ids:
        yt, nt, _, _ = await _fetch_gamma_tokens(session, mid)
        result[mid] = (yt, nt)
    return result


# ═══════════════════════════════════════════════════════════════
# MAIN SCANNER LOOP
# ═══════════════════════════════════════════════════════════════

async def run_ws_scanner(
    duration_seconds: int = 3600,
    output_path: str = "spread_divergence.png",
    summary_interval: int = 60,
) -> None:
    """
    Full lifecycle of the WebSocket spread scanner.

    1. Fetch equivalent pairs from PostgreSQL.
    2. Bootstrap orderbooks via REST.
    3. Resolve Polymarket token IDs.
    4. Start Kalshi + Polymarket WS feeds as background tasks.
    5. Print periodic summaries.
    6. On duration expiry or SIGINT, generate graph + print report.
    """
    dsn = config.DATABASE_URL
    if not dsn:
        logger.error("DATABASE_URL not set — add it to .env")
        return

    # ── 1. equivalent pairs ──────────────────────────────────
    logger.info("Fetching equivalent pairs from database…")
    pairs = await fetch_equivalent_pairs(dsn)
    logger.info("Found %d equivalent pairs", len(pairs))
    if not pairs:
        print("\nNo equivalent pairs in the database.")
        return

    manager = LiveOrderbookManager(pairs)

    # ── 2. partition by venue ────────────────────────────────
    kalshi_tickers: list[str] = []
    poly_mids: list[str] = []
    for p in pairs:
        for venue, vmid in [(p.venue_a, p.vmid_a), (p.venue_b, p.vmid_b)]:
            if venue == "kalshi" and vmid not in kalshi_tickers:
                kalshi_tickers.append(vmid)
            elif venue == "polymarket" and vmid not in poly_mids:
                poly_mids.append(vmid)

    logger.info("Tracking %d Kalshi tickers + %d Polymarket markets", len(kalshi_tickers), len(poly_mids))

    # ── 3. bootstrap + WS feeds ──────────────────────────────
    kalshi_key_id, kalshi_pk = _build_kalshi_auth()
    if not kalshi_key_id:
        logger.warning("Kalshi auth not configured — Kalshi WS/REST will likely fail")

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        # Bootstrap: concurrent REST fetch of all books
        logger.info("Bootstrapping %d orderbooks via REST…", len(kalshi_tickers) + len(poly_mids))
        coros: dict[str, asyncio.Task] = {}
        for t in kalshi_tickers:
            coros[f"kalshi:{t}"] = fetch_kalshi_book(session, t, kalshi_key_id, kalshi_pk)
        for mid in poly_mids:
            coros[f"polymarket:{mid}"] = fetch_polymarket_book(session, mid)

        keys_list = list(coros.keys())
        results = await asyncio.gather(*[coros[k] for k in keys_list], return_exceptions=True)
        for key, res in zip(keys_list, results):
            if isinstance(res, Exception):
                logger.warning("Bootstrap %s failed: %s", key, res)
            elif res is not None:
                await manager.update_book(key, res)

        logger.info("Bootstrapped %d/%d books", len(manager.books), len(coros))
        initial = await manager.recalculate(source="rest_bootstrap")
        profitable = [o for o in initial if o.net_profit_1 > 0]
        logger.info("Initial scan: %d opps (%d profitable)", len(initial), len(profitable))

        # Resolve Polymarket token IDs for WS subscriptions
        poly_tokens = await _resolve_poly_tokens(session, poly_mids)

        # ── 4. create feed objects ───────────────────────────
        k_feed = KalshiWSFeed(
            tickers=kalshi_tickers,
            manager=manager,
            api_key_id=kalshi_key_id,
            private_key=kalshi_pk,
        )
        p_feed = PolymarketWSFeed(
            market_ids=poly_mids,
            token_map=poly_tokens,
            manager=manager,
        )

        # ── 5. run ───────────────────────────────────────────
        start = time.monotonic()
        stop_event = asyncio.Event()

        def _on_signal():
            logger.info("Shutdown signal received")
            stop_event.set()

        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, _on_signal)
        except (NotImplementedError, RuntimeError):
            pass  # Windows / nested event loops

        k_task = asyncio.create_task(k_feed.run(session))
        p_task = asyncio.create_task(p_feed.run(session))

        async def _printer():
            while not stop_event.is_set():
                await asyncio.sleep(summary_interval)
                if stop_event.is_set():
                    break
                print_live_summary(
                    manager,
                    time.monotonic() - start,
                    k_feed.update_count,
                    p_feed.update_count,
                )

        printer_task = asyncio.create_task(_printer())

        # Block until duration or signal
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=duration_seconds)
        except asyncio.TimeoutError:
            logger.info("Duration (%d s) elapsed — shutting down", duration_seconds)

        # ── 6. shutdown ──────────────────────────────────────
        await k_feed.stop()
        await p_feed.stop()
        for t in (k_task, p_task, printer_task):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    # ── 7. graph + report ────────────────────────────────────
    elapsed = time.monotonic() - start
    dur_label = f"Duration: {timedelta(seconds=int(elapsed))}"

    graph_path = generate_graph(
        manager.series,
        output_path=output_path,
        duration_label=dur_label,
    )

    print_final_report(
        manager,
        duration=elapsed,
        k_updates=k_feed.update_count,
        p_updates=p_feed.update_count,
        graph_path=graph_path,
    )


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time WebSocket spread scanner for cross-venue arbitrage.",
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=3600,
        help="Monitoring duration in seconds (default: 3600 = 1 h)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="spread_divergence.png",
        help="Output graph file path (default: spread_divergence.png)",
    )
    parser.add_argument(
        "-s", "--summary-interval",
        type=int,
        default=60,
        help="Seconds between live console summaries (default: 60)",
    )
    args = parser.parse_args()

    logger.info(
        "Starting WebSocket spread scanner (duration=%d s, output=%s)",
        args.duration, args.output,
    )
    asyncio.run(run_ws_scanner(
        duration_seconds=args.duration,
        output_path=args.output,
        summary_interval=args.summary_interval,
    ))


if __name__ == "__main__":
    main()
