#!/usr/bin/env python3
"""
Verify if two markets on Kalshi and Polymarket are the same event by comparing resolution dates.

Fetches market details from both APIs and compares:
- Resolution/close dates
- Market titles
- Event details
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, UTC
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import aiohttp
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Market identifiers from logs
KALSHI_TICKER = "KXLOLMAP-26FEB23T1AKTC-3-T1A"
POLYMARKET_SEARCH_TITLE = "LoL: KT Rolster Challengers vs T1 Academy - Game 3 Winner"


async def fetch_kalshi_market(ticker: str) -> dict:
    """Fetch market details from Kalshi API."""
    try:
        # Kalshi REST API endpoint
        url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("market", {})
                else:
                    logger.error(f"Kalshi API error: {resp.status} - {await resp.text()}")
                    return {}
    except Exception as e:
        logger.error(f"Failed to fetch Kalshi market: {e}")
        return {}


async def fetch_polymarket_market(search_title: str) -> dict:
    """Fetch market details from Polymarket Gamma API by searching for title."""
    try:
        # Polymarket Gamma API
        base_url = "https://gamma-api.polymarket.com/markets"
        
        # Extract key terms for broader search
        search_terms = ["T1 Academy", "KT Rolster", "Game 3", "LoL"]
        
        # Search multiple pages to find the market (try both active and closed)
        async with aiohttp.ClientSession() as session:
            for closed_status in ["false", "true"]:  # Try both active and closed markets
                for page in range(10):  # Search first 10 pages (1000 markets)
                    offset = page * 100
                    params = {
                        "closed": closed_status,
                        "active": "true" if closed_status == "false" else "false",
                        "limit": "100",
                        "offset": str(offset)
                    }
                
                async with session.get(base_url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        markets = data if isinstance(data, list) else []
                        
                        if not markets:
                            break
                        
                        # Search for matching title with multiple criteria
                        for market in markets:
                            title = market.get("question", "") or market.get("title", "")
                            title_lower = title.lower()
                            
                            # Check if all key terms are present
                            matches = sum(1 for term in search_terms if term.lower() in title_lower)
                            
                            # Also check direct title match or partial matches
                            if (search_title.lower() in title_lower or 
                                title_lower in search_title.lower() or
                                matches >= 3 or  # At least 3 key terms match
                                ("t1" in title_lower and "kt rolster" in title_lower and ("game 3" in title_lower or "map 3" in title_lower))):
                                logger.info(f"Found Polymarket market on page {page + 1} (closed={closed_status})")
                                return market
                    else:
                        logger.error(f"Polymarket API error: {resp.status} - {await resp.text()}")
                        break
            
            logger.warning(f"No Polymarket market found matching: {search_title}")
            # Return a sample market for debugging
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params={"closed": "false", "active": "true", "limit": "10"}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        markets = data if isinstance(data, list) else []
                        if markets:
                            logger.info(f"Sample Polymarket markets (first 3):")
                            for i, m in enumerate(markets[:3], 1):
                                title = m.get("question", "") or m.get("title", "N/A")
                                logger.info(f"  {i}. {title[:80]}")
            return {}
    except Exception as e:
        logger.error(f"Failed to fetch Polymarket market: {e}")
        return {}


def parse_kalshi_date(market: dict) -> Optional[datetime]:
    """Extract resolution date from Kalshi market."""
    # Kalshi uses close_ts (timestamp in seconds)
    if "close_ts" in market and market["close_ts"]:
        try:
            return datetime.fromtimestamp(int(market["close_ts"]), tz=UTC)
        except (ValueError, TypeError):
            pass
    
    # Also check close_time
    if "close_time" in market and market["close_time"]:
        try:
            # Try parsing ISO format
            return datetime.fromisoformat(market["close_time"].replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass
    
    return None


def parse_polymarket_date(market: dict) -> Optional[datetime]:
    """Extract resolution date from Polymarket market."""
    # Polymarket uses endDate_iso or endDate
    if "endDate_iso" in market and market["endDate_iso"]:
        try:
            return datetime.fromisoformat(market["endDate_iso"].replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass
    
    if "endDate" in market and market["endDate"]:
        try:
            return datetime.fromisoformat(market["endDate"].replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass
    
    return None


async def main():
    """Fetch and compare markets."""
    print("=" * 80)
    print("MARKET EQUIVALENCE VERIFICATION")
    print("=" * 80)
    
    # Fetch Kalshi market
    print(f"\n📊 Fetching Kalshi market: {KALSHI_TICKER}")
    kalshi_market = await fetch_kalshi_market(KALSHI_TICKER)
    
    if not kalshi_market:
        print("❌ Failed to fetch Kalshi market")
        return
    
    kalshi_title = kalshi_market.get("title", "N/A")
    kalshi_date = parse_kalshi_date(kalshi_market)
    
    print(f"   Title: {kalshi_title}")
    print(f"   Resolution Date: {kalshi_date.isoformat() if kalshi_date else 'N/A'}")
    print(f"   Status: {kalshi_market.get('status', 'N/A')}")
    print(f"   Event Ticker: {kalshi_market.get('event_ticker', 'N/A')}")
    
    # Fetch Polymarket market
    print(f"\n📊 Fetching Polymarket market: {POLYMARKET_SEARCH_TITLE}")
    polymarket_market = await fetch_polymarket_market(POLYMARKET_SEARCH_TITLE)
    
    if not polymarket_market:
        print("❌ Failed to fetch Polymarket market")
        return
    
    polymarket_title = polymarket_market.get("question", "") or polymarket_market.get("title", "N/A")
    polymarket_date = parse_polymarket_date(polymarket_market)
    
    print(f"   Title: {polymarket_title}")
    print(f"   Resolution Date: {polymarket_date.isoformat() if polymarket_date else 'N/A'}")
    print(f"   Slug: {polymarket_market.get('slug', 'N/A')}")
    print(f"   Active: {polymarket_market.get('active', 'N/A')}")
    
    # Compare dates
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    if kalshi_date and polymarket_date:
        date_diff = abs((kalshi_date - polymarket_date).total_seconds())
        date_diff_hours = date_diff / 3600
        date_diff_days = date_diff / 86400
        
        print(f"\n📅 Date Comparison:")
        print(f"   Kalshi:    {kalshi_date.isoformat()}")
        print(f"   Polymarket: {polymarket_date.isoformat()}")
        print(f"   Difference: {date_diff_days:.2f} days ({date_diff_hours:.2f} hours)")
        
        # Tolerance: same day or within 24 hours
        if date_diff_days < 1.0:
            print(f"\n✅ SAME EVENT: Resolution dates are within 24 hours")
            print(f"   These markets target the same event.")
        elif date_diff_days < 7.0:
            print(f"\n⚠️  POSSIBLY SAME: Resolution dates differ by {date_diff_days:.1f} days")
            print(f"   Could be the same event with different resolution timing.")
        else:
            print(f"\n❌ DIFFERENT EVENTS: Resolution dates differ by {date_diff_days:.1f} days")
            print(f"   These are likely different events.")
    else:
        print(f"\n⚠️  Cannot compare dates:")
        if not kalshi_date:
            print(f"   Kalshi date: Missing")
        if not polymarket_date:
            print(f"   Polymarket date: Missing")
    
    # Compare titles
    print(f"\n📝 Title Comparison:")
    print(f"   Kalshi:    {kalshi_title}")
    print(f"   Polymarket: {polymarket_title}")
    
    # Check for same teams
    teams_match = (
        "T1 Academy" in kalshi_title and "T1 Academy" in polymarket_title and
        "KT Rolster" in kalshi_title and "KT Rolster" in polymarket_title
    )
    
    # Check for same game number
    game_match = (
        ("map 3" in kalshi_title.lower() or "game 3" in kalshi_title.lower()) and
        ("game 3" in polymarket_title.lower() or "map 3" in polymarket_title.lower())
    )
    
    print(f"\n🔍 Content Analysis:")
    print(f"   Same Teams: {'✅' if teams_match else '❌'}")
    print(f"   Same Game/Map: {'✅' if game_match else '❌'}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if kalshi_date and polymarket_date and date_diff_days < 1.0:
        print("\n✅ VERIFIED: These markets target the SAME EVENT")
        print("   Resolution dates match within 24 hours.")
    elif teams_match and game_match:
        print("\n✅ LIKELY SAME EVENT: Teams and game number match")
        print("   Even if dates differ slightly, these appear to be the same match.")
    else:
        print("\n❓ UNCERTAIN: Cannot definitively verify equivalence")
        print("   Manual review recommended.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

