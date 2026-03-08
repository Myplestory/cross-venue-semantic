# Esports Arbitrage Monitor - Quick Start Guide

## Overview

The `monitor_dk_vs_t1.py` script monitors real-time arbitrage opportunities between Kalshi and Polymarket for the DK vs T1 League of Legends match, while measuring latency between game events and market odds updates.

## Prerequisites

1. **Python 3.11+** (tested with 3.14)
2. **Required packages** (already installed):
   - `aiohttp`
   - `websockets`
   - `asyncpg` (for database, optional)

3. **Environment Variables** (set in `.env` file or export):

## Required Environment Variables

### Minimum (for basic monitoring without Riot API):

```bash
# Kalshi API (required for orderbook data)
KALSHI_API_KEY_ID=your_kalshi_api_key_id
KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi_private_key.pem

# Optional: Database for pair lookup
DATABASE_URL=postgresql://user:password@host:port/database
```

### Full Setup (with game event latency measurement):

```bash
# Kalshi API (required)
KALSHI_API_KEY_ID=your_kalshi_api_key_id
KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi_private_key.pem

# Riot API (optional, for game event polling)
RIOT_API_KEY=your_riot_api_key

# Optional: Manual Riot match IDs (comma-separated) if auto-discovery doesn't find them
# Example: RIOT_MATCH_IDS=ESPORTSTMNT01_123456,ESPORTSTMNT01_123457
RIOT_MATCH_IDS=

# Optional: Customize polling frequency (default: 2000ms for dev keys, 500ms for production)
GAME_EVENT_POLL_INTERVAL_MS=2000

# Optional: Database
DATABASE_URL=postgresql://user:password@host:port/database
```

## Getting API Keys

### Kalshi API
1. Sign up at [kalshi.com](https://kalshi.com)
2. Go to Settings → API Keys
3. Generate API key ID and download private key (`.pem` file)
4. Save private key to a secure location

### Riot API (Optional)
1. Sign up at [Riot Developer Portal](https://developer.riotgames.com/)
2. Create an API key
3. Note: Rate limits apply (100 requests per 2 minutes for development keys)

## Running the Monitor

### Step 1: Navigate to semantic_pipeline directory

```bash
cd /Users/charles/Desktop/PolyEdge/PolyEdge/semantic_pipeline
```

### Step 2: Set environment variables

**Option A: Export in terminal (temporary)**
```bash
export KALSHI_API_KEY_ID="your_key_id"
export KALSHI_PRIVATE_KEY_PATH="/path/to/key.pem"
export RIOT_API_KEY="your_riot_key"  # Optional
export GAME_EVENT_POLL_INTERVAL_MS=500  # Optional
```

**Option B: Add to `.env` file (persistent)**
```bash
# Edit .env file in semantic_pipeline directory
nano .env
```

Add:
```env
KALSHI_API_KEY_ID=your_key_id
KALSHI_PRIVATE_KEY_PATH=/path/to/key.pem
RIOT_API_KEY=your_riot_key
GAME_EVENT_POLL_INTERVAL_MS=500
```

### Step 3: Run the monitor

```bash
python3 monitor_dk_vs_t1.py
```

Or if using Python directly:
```bash
python monitor_dk_vs_t1.py
```

## What You'll See

### Startup Output
```
2026-01-16 10:00:00 INFO     🚀 Starting esports arbitrage monitor (DK vs T1)
2026-01-16 10:00:00 INFO        Kalshi: KXLOLGAME-26FEB22DKT1
2026-01-16 10:00:00 INFO        Polymarket: lol-t1-dk-2026-02-22
2026-01-16 10:00:01 INFO     Resolving Polymarket market: lol-t1-dk-2026-02-22
2026-01-16 10:00:02 INFO     Resolved to condition_id: 0x1234...
2026-01-16 10:00:03 INFO     ✅ GameEventBridge started - high-frequency polling active (2.0 polls/second)
2026-01-16 10:00:04 INFO     🚀 Starting all WebSocket feeds...
```

### Real-time Output
```
2026-01-16 10:05:23 INFO     [Status] Kalshi YES: $0.5234 | Poly YES: $0.5210 | Updates: 142 | Opportunities: 3
2026-01-16 10:05:23 INFO     [Latency] Samples: 15 | Mean: 1247.2ms | Min: 856.3ms | Max: 2103.1ms
2026-01-16 10:05:25 INFO     🚨 ARBITRAGE OPPORTUNITY: gross_edge=1.23%, net_profit=$0.0123, optimal_qty=100 contracts
2026-01-16 10:05:26 DEBUG    [GameEventBridge] Event: CHAMPION_KILL | Match: match_123 | Ingestion latency: 523.4ms
2026-01-16 10:05:27 DEBUG    [LatencyEngine] Correlated event: CHAMPION_KILL -> polymarket odds update, latency=1247.2ms
```

## Stopping the Monitor

Press `Ctrl+C` to gracefully stop the monitor. It will:
- Stop all WebSocket feeds
- Stop game event polling
- Print final statistics
- Clean up connections

## Troubleshooting

### Error: "Kalshi auth not configured"
- Check `KALSHI_API_KEY_ID` is set
- Check `KALSHI_PRIVATE_KEY_PATH` points to valid `.pem` file
- Verify private key file has correct permissions (readable)

### Error: "Could not resolve Polymarket market ID"
- Market may be expired or slug may be incorrect
- Check Polymarket URL: `https://polymarket.com/sports/league-of-legends/lol-t1-dk-2026-02-22`
- Verify market is still active

### Error: "No Riot match IDs found"
- This is normal if Riot API key is not provided
- Monitor will still work using Polymarket Sports WebSocket only
- For full latency measurement, provide `RIOT_API_KEY`

### WebSocket Connection Errors
- Check internet connection
- Verify Kalshi API credentials are valid
- Check if markets are still active (may have expired)

## Advanced Configuration

### Custom Polling Interval

Set faster polling (more API calls, lower latency):
```bash
export GAME_EVENT_POLL_INTERVAL_MS=250  # 4 polls/second
```

Or slower polling (fewer API calls, higher latency):
```bash
export GAME_EVENT_POLL_INTERVAL_MS=1000  # 1 poll/second
```

**Note:** Riot API has rate limits. Default 500ms (2 polls/second) is safe for development keys.

### Custom Market Pair

Edit `monitor_dk_vs_t1.py` and change:
```python
KALSHI_TICKER = "YOUR_KALSHI_TICKER"
POLYMARKET_SLUG = "your-polymarket-slug"
```

## Understanding the Output

### Latency Metrics
- **Ingestion latency**: Time from game event to your system receiving it (API polling delay)
- **Market reaction latency**: Time from game event to market odds update (what you're measuring)
- **Frontrunning window**: Time window where you can act before market fully reacts

### Arbitrage Opportunities
- **gross_edge**: Profit before fees
- **net_profit**: Profit after fees
- **optimal_qty**: Best contract quantity to trade

## Next Steps

1. **Monitor during live match**: Run during actual DK vs T1 game to see real latency
2. **Analyze patterns**: Look for consistent latency differences between venues
3. **Tune thresholds**: Adjust `min_price_move_pct` and `max_latency_ms` in `FrontrunningDetector`
4. **Extend to other matches**: Modify market identifiers for different games

## Support

For issues or questions:
- Check logs for detailed error messages
- Verify all environment variables are set correctly
- Ensure API keys are valid and not expired

