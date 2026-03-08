#!/usr/bin/env python3
"""
Fetch the specific Game 3 market from Polymarket using the match URL structure.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Match URL: https://polymarket.com/sports/league-of-legends/games/week/1/lol-ktc-t1a-2026-02-23
MATCH_SLUG = "lol-ktc-t1a-2026-02-23"


async def fetch_polymarket_match_markets():
    """Fetch all markets for the match, including Game 3."""
    try:
        # Try to fetch from Gamma API using the slug
        # Polymarket uses slugs in their URLs
        base_url = "https://gamma-api.polymarket.com/markets"
        
        # Search for markets with the slug or related terms
        async with aiohttp.ClientSession() as session:
            # Try searching for markets with T1 Academy and KT Rolster
            params = {
                "closed": "false",
                "active": "true",
                "limit": "200",
                "offset": "0"
            }
            
            async with session.get(base_url, params=params) as resp:
                if resp.status == 200:
                    markets = await resp.json()
                    if not isinstance(markets, list):
                        markets = []
                    
                    # Filter for Game 3 markets
                    game3_markets = []
                    for market in markets:
                        title = market.get("question", "") or market.get("title", "")
                        slug = market.get("slug", "")
                        
                        # Check if it's Game 3 for this match
                        if (("t1 academy" in title.lower() or "t1a" in title.lower()) and
                            ("kt rolster" in title.lower() or "ktc" in title.lower()) and
                            ("game 3" in title.lower() or "game3" in title.lower())):
                            game3_markets.append(market)
                    
                    return game3_markets
        return []
    except Exception as e:
        logger.error(f"Error fetching Polymarket markets: {e}")
        return []


async def fetch_polymarket_by_slug():
    """Try to fetch market directly using slug pattern."""
    try:
        # Polymarket might have a direct endpoint for slugs
        # Try the CLOB API or check if there's a slug endpoint
        async with aiohttp.ClientSession() as session:
            # Try searching with the date and teams
            search_terms = ["2026-02-23", "T1 Academy", "KT Rolster", "Game 3"]
            
            base_url = "https://gamma-api.polymarket.com/markets"
            params = {
                "closed": "false",
                "active": "true",
                "limit": "500",
                "offset": "0"
            }
            
            async with session.get(base_url, params=params) as resp:
                if resp.status == 200:
                    markets = await resp.json()
                    if not isinstance(markets, list):
                        markets = []
                    
                    # Look for Game 3 market
                    for market in markets:
                        title = market.get("question", "") or market.get("title", "")
                        slug = market.get("slug", "")
                        end_date = market.get("endDate_iso") or market.get("endDate")
                        
                        # Check multiple criteria
                        has_date = "2026-02-23" in str(end_date) if end_date else False
                        has_teams = ("t1" in title.lower() and "kt rolster" in title.lower())
                        has_game3 = "game 3" in title.lower() or "game3" in title.lower()
                        
                        if has_teams and has_game3:
                            return market
        return None
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


async def main():
    """Fetch and display Game 3 market details."""
    print("=" * 80)
    print("FETCHING POLYMARKET GAME 3 MARKET")
    print("=" * 80)
    print(f"\nMatch: KT Rolster Challengers vs T1 Academy")
    print(f"Date: February 23, 2026")
    print(f"URL: https://polymarket.com/sports/league-of-legends/games/week/1/{MATCH_SLUG}\n")
    
    # Try fetching by slug/date
    print("Searching for Game 3 market...")
    market = await fetch_polymarket_by_slug()
    
    if market:
        title = market.get("question", "") or market.get("title", "N/A")
        slug = market.get("slug", "N/A")
        end_date = market.get("endDate_iso") or market.get("endDate", "N/A")
        active = market.get("active", "N/A")
        closed = market.get("closed", "N/A")
        
        print(f"\n✅ Found Game 3 Market:")
        print(f"   Title: {title}")
        print(f"   Slug: {slug}")
        print(f"   End Date: {end_date}")
        print(f"   Active: {active}")
        print(f"   Closed: {closed}")
        
        # Parse date
        if end_date:
            try:
                if isinstance(end_date, str):
                    if "T" in end_date:
                        pm_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    else:
                        pm_date = datetime.fromisoformat(end_date)
                else:
                    pm_date = end_date
                
                print(f"\n📅 Resolution Date: {pm_date.isoformat()}")
                
                # Compare with Kalshi date
                kalshi_date = datetime(2026, 3, 9, 5, 0, 0, tzinfo=UTC)
                date_diff = abs((pm_date - kalshi_date).total_seconds())
                date_diff_days = date_diff / 86400
                
                print(f"\n📊 Date Comparison:")
                print(f"   Kalshi:    {kalshi_date.isoformat()}")
                print(f"   Polymarket: {pm_date.isoformat()}")
                print(f"   Difference: {date_diff_days:.2f} days")
                
                if date_diff_days < 1.0:
                    print(f"\n✅ SAME EVENT: Resolution dates match within 24 hours")
                elif date_diff_days < 7.0:
                    print(f"\n⚠️  POSSIBLY SAME: Dates differ by {date_diff_days:.1f} days")
                    print(f"   (Match is Feb 23, but resolution might be after match completion)")
                else:
                    print(f"\n❌ DIFFERENT: Dates differ by {date_diff_days:.1f} days")
            except Exception as e:
                print(f"   Error parsing date: {e}")
    else:
        print("\n❌ Game 3 market not found via API")
        print("   The market may be closed or the API doesn't expose it directly.")
        print("   However, based on the URL structure, Game 3 markets exist for this match.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

