# Riot Games API Rate Limits

## Overview

Riot Games API has strict rate limits that vary by API key type and endpoint. Exceeding these limits results in HTTP 429 (Too Many Requests) responses.

## Rate Limit Types

### 1. **Application Rate Limits** (Per API Key)
- **Development/Personal API Key**: 
  - **20 requests per 1 second** (per region)
  - **100 requests per 2 minutes** (per region)
- **Production API Key**:
  - Higher limits (contact Riot for production access)
  - Typically: 500 requests per 10 seconds per region

### 2. **Method Rate Limits** (Per Endpoint)
Some endpoints have additional per-method limits:
- **Match Timeline**: 100 requests per 2 minutes
- **Match Details**: 100 requests per 2 minutes
- **Spectator**: 20 requests per 1 second

## Rate Limit Headers

Riot API returns rate limit information in response headers:

```
X-Method-Rate-Limit: 100:2
X-Method-Rate-Limit-Count: 50:2
X-App-Rate-Limit: 20:1,100:120
X-App-Rate-Limit-Count: 10:1,50:120
Retry-After: 5
```

- `X-Method-Rate-Limit`: Method-specific limits (format: `limit:window_seconds`)
- `X-Method-Rate-Limit-Count`: Current method usage
- `X-App-Rate-Limit`: Application-wide limits
- `X-App-Rate-Limit-Count`: Current application usage
- `Retry-After`: Seconds to wait before retrying (on 429)

## Current Implementation

### Rate Limit Handling

The `GameEventBridge` and `RiotAPIClient` handle rate limits as follows:

1. **429 Response Detection**: Catches `aiohttp.ClientResponseError` with status 429
2. **Retry-After Support**: Reads `Retry-After` header and waits before retry
3. **Circuit Breaker**: Opens circuit after 5 consecutive failures (default)
4. **Automatic Backoff**: Waits for recovery timeout (60s default) before retrying

### Default Polling Configuration

**Current Default**: `GAME_EVENT_POLL_INTERVAL_MS=500` (2 polls/second)

**Rate Limit Analysis**:
- 2 polls/second = 120 polls/minute
- **⚠️ EXCEEDS Development Key Limit** (100 requests per 2 minutes)

### Recommended Polling Intervals

#### For Development/Personal API Keys:

```bash
# Safe: 1 poll per 1.2 seconds (50 requests/minute)
export GAME_EVENT_POLL_INTERVAL_MS=1200

# Conservative: 1 poll per 2 seconds (30 requests/minute)
export GAME_EVENT_POLL_INTERVAL_MS=2000
```

#### For Production API Keys:

```bash
# Aggressive: 500ms (120 requests/minute) - OK for production
export GAME_EVENT_POLL_INTERVAL_MS=500

# Very aggressive: 250ms (240 requests/minute) - Check your limits
export GAME_EVENT_POLL_INTERVAL_MS=250
```

## Rate Limit Best Practices

### 1. **Respect Retry-After Header**
Always wait for the duration specified in `Retry-After` header before retrying.

### 2. **Use Circuit Breaker**
The circuit breaker automatically opens after repeated failures, preventing API key suspension.

### 3. **Monitor Rate Limit Headers**
Track `X-App-Rate-Limit-Count` and `X-Method-Rate-Limit-Count` to avoid hitting limits.

### 4. **Implement Exponential Backoff**
On 429 errors, wait with exponential backoff:
- First retry: 1 second
- Second retry: 2 seconds
- Third retry: 4 seconds
- Max: 60 seconds

### 5. **Batch Requests When Possible**
If polling multiple matches, stagger requests to avoid bursts.

## Current Code Behavior

### GameEventBridge Polling

```python
# Default: 500ms interval
poll_interval_ms = 500  # 2 polls/second

# For 1 match:
# - 2 requests/second = 120 requests/minute
# - ⚠️ Exceeds development key limit (100/2min)
```

### Rate Limit Handling

```python
# In game_event_bridge.py
except aiohttp.ClientResponseError as e:
    if e.status == 429:
        retry_after = int(e.headers.get("Retry-After", "1"))
        await asyncio.sleep(retry_after)  # Wait before retry
```

## Recommendations

### For Development/Testing:

1. **Use conservative polling**:
   ```bash
   export GAME_EVENT_POLL_INTERVAL_MS=2000  # 1 poll per 2 seconds
   ```

2. **Monitor for 429 errors** in logs:
   ```
   [GameEventBridge] Rate limited for match_123. Waiting 5s before retry
   ```

3. **Check circuit breaker state**:
   - Circuit opens after 5 consecutive failures
   - Automatically attempts recovery after 60 seconds

### For Production:

1. **Request production API key** from Riot for higher limits
2. **Monitor rate limit headers** to track usage
3. **Implement request queuing** to smooth out bursts
4. **Use multiple API keys** (if allowed) for load distribution

## Testing Rate Limits

To test rate limit handling:

```bash
# Set very aggressive polling (will hit limits quickly)
export GAME_EVENT_POLL_INTERVAL_MS=100  # 10 polls/second

# Run monitor and watch for 429 errors
python3 monitor_dk_vs_t1.py
```

Expected behavior:
- Initial requests succeed
- After ~10 requests, 429 errors appear
- Circuit breaker opens after 5 consecutive failures
- Automatic recovery after 60 seconds

## References

- [Riot Developer Portal](https://developer.riotgames.com/)
- [Riot API Documentation](https://developer.riotgames.com/apis)
- [Rate Limiting Guide](https://developer.riotgames.com/docs/rate-limits)

## Summary

**Development Key Limits**:
- 20 requests/second
- 100 requests/2 minutes

**Current Default (500ms polling)**:
- 2 requests/second = 120 requests/minute
- **⚠️ Exceeds 100/2min limit**

**Recommended for Development**:
- Use `GAME_EVENT_POLL_INTERVAL_MS=2000` (1 poll per 2 seconds)
- This gives 30 requests/minute, well under the limit

**Rate Limit Handling**:
- ✅ Automatic Retry-After support
- ✅ Circuit breaker protection
- ✅ Graceful degradation on errors

