# Esports Arbitrage Monitor

Real-time cross-venue arbitrage monitor for esports prediction markets. Measures latency between game events (via Riot API) and market odds updates across Kalshi and Polymarket.

## What It Does

1. Connects to Kalshi + Polymarket orderbooks via WebSocket
2. Polls Riot Games API for live game events (kills, objectives, etc.)
3. Correlates game events with odds movements to measure market reaction latency
4. Detects arbitrage opportunities when cross-venue spreads exceed fee thresholds

## Prerequisites

- Python 3.11+
- Kalshi API key + private key (`.pem`)
- Riot API key (optional, for game event latency measurement)

## Configuration

Set in `.env`:

```env
# Required
KALSHI_API_KEY_ID=your_kalshi_key_id
KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi.pem

# Optional (enables game event → odds latency measurement)
RIOT_API_KEY=your_riot_api_key
GAME_EVENT_POLL_INTERVAL_MS=1000  # ms between Riot API polls

# Optional
DATABASE_URL=postgresql://user:pass@host:port/database
```

## Running

```bash
python monitor_dk_vs_t1.py
```

## Output

```
[Status] Kalshi YES: $0.5234 | Poly YES: $0.5210 | Updates: 142 | Opportunities: 3
[Latency] Samples: 15 | Mean: 1247.2ms | Min: 856.3ms | Max: 2103.1ms
ARBITRAGE OPPORTUNITY: gross_edge=1.23%, net_profit=$0.0123, optimal_qty=100 contracts
[GameEventBridge] Event: CHAMPION_KILL | Ingestion latency: 523.4ms
[LatencyEngine] Correlated: CHAMPION_KILL -> polymarket odds update, latency=1247.2ms
```

## Riot API Rate Limits

See [RIOT_API_RATE_LIMITS.md](RIOT_API_RATE_LIMITS.md) for details.

| Key Type | Limit | Recommended Polling |
|----------|-------|-------------------|
| Development | 100 req / 2 min | `GAME_EVENT_POLL_INTERVAL_MS=2000` |
| Production | ~500 req / 10 sec | `GAME_EVENT_POLL_INTERVAL_MS=500` |

The monitor handles 429s with circuit breaker + exponential backoff automatically.

## Extending

To monitor a different match, edit the market identifiers in `monitor_dk_vs_t1.py`:

```python
KALSHI_TICKER = "YOUR_KALSHI_TICKER"
POLYMARKET_SLUG = "your-polymarket-slug"
```

## Findings

Market reaction latency to game events averaged 1-2 seconds. Cross-venue spreads rarely exceeded fee thresholds (~2% round-trip on Kalshi + Polymarket). Conclusion: esports prediction market arb is not viable at current liquidity levels.
