#!/usr/bin/env python3
"""
Investigate why esports markets didn't produce verified pairs.

Checks:
1. If esports markets were written to the markets table
2. If any esports markets exist but weren't matched
3. Matching/verification thresholds
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import asyncpg
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Esports keywords for filtering
ESPORTS_KEYWORDS = [
    'esports', 'esport', 'lol', 'league of legends', 'dota', 'dota 2',
    'valorant', 'csgo', 'cs:go', 'counter-strike', 'overwatch',
    'apex legends', 'fortnite', 'rocket league', 'rainbow six',
    'call of duty', 'cod', 'fifa', 'nba 2k', 'madden',
    'worlds', 'championship', 'tournament', 'tourney',
    'team liquid', 'fnatic', 'g2', 'tsm', 'cloud9', 'c9',
    'faker', 'doublelift', 's1mple', 'zywoo'
]


async def investigate():
    """Investigate esports markets and matching."""
    dsn = config.DATABASE_URL
    if not dsn:
        logger.error("DATABASE_URL not set in .env file")
        return

    logger.info("Connecting to database...")
    try:
        conn = await asyncpg.connect(dsn=dsn, statement_cache_size=0)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return

    try:
        # ── Check 1: Total markets in database ────────────────────────
        total_markets = await conn.fetchval("SELECT COUNT(*) FROM markets")
        logger.info(f"\n{'='*80}")
        logger.info(f"Total markets in database: {total_markets}")
        logger.info(f"{'='*80}\n")

        # ── Check 2: Esports markets in database ───────────────────────
        all_markets = await conn.fetch(
            """
            SELECT venue, venue_market_id, title, description, created_at
            FROM markets
            ORDER BY created_at DESC
            LIMIT 1000
            """
        )

        esports_markets = []
        for market in all_markets:
            title = (market["title"] or "").lower()
            desc = (market["description"] or "").lower()
            text = f"{title} {desc}"
            
            if any(keyword in text for keyword in ESPORTS_KEYWORDS):
                esports_markets.append(market)

        logger.info(f"Esports markets found in database: {len(esports_markets)}")
        logger.info(f"Non-esports markets: {len(all_markets) - len(esports_markets)}\n")

        if esports_markets:
            logger.info("Sample esports markets:")
            for i, market in enumerate(esports_markets[:10], 1):
                logger.info(f"  {i}. [{market['venue']}] {market['title'][:70]}")
                logger.info(f"     ID: {market['venue_market_id'][:50]}")
        else:
            logger.warning("⚠️  NO ESPORTS MARKETS FOUND IN DATABASE!")
            logger.warning("This means esports markets were discovered but never written.")
            logger.warning("Markets are only written when a verified pair is found.")
            logger.warning("\nPossible reasons:")
            logger.warning("  1. Esports markets from Kalshi and Polymarket didn't match each other")
            logger.warning("  2. Matching candidates were found but verification failed")
            logger.warning("  3. No matching candidates were found (different events/games)")

        # ── Check 3: Markets by venue ──────────────────────────────────
        venue_counts = await conn.fetch(
            """
            SELECT venue, COUNT(*) as count
            FROM markets
            GROUP BY venue
            ORDER BY count DESC
            """
        )
        
        logger.info(f"\n{'='*80}")
        logger.info("Markets by venue:")
        for row in venue_counts:
            logger.info(f"  {row['venue']}: {row['count']} markets")

        # ── Check 4: Recent markets (last 24 hours) ────────────────────
        recent_markets = await conn.fetch(
            """
            SELECT venue, venue_market_id, title, created_at
            FROM markets
            WHERE created_at > NOW() - INTERVAL '24 hours'
            ORDER BY created_at DESC
            LIMIT 50
            """
        )
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Markets created in last 24 hours: {len(recent_markets)}")
        
        recent_esports = [
            m for m in recent_markets
            if any(kw in (m["title"] or "").lower() for kw in ESPORTS_KEYWORDS)
        ]
        logger.info(f"Recent esports markets: {len(recent_esports)}")
        
        if recent_markets:
            logger.info("\nSample recent markets:")
            for i, market in enumerate(recent_markets[:5], 1):
                is_esports = market in recent_esports
                tag = " [ESPORTS]" if is_esports else ""
                logger.info(f"  {i}. [{market['venue']}]{tag} {market['title'][:70]}")

        # ── Check 5: Analysis ──────────────────────────────────────────
        logger.info(f"\n{'='*80}")
        logger.info("ANALYSIS")
        logger.info(f"{'='*80}")
        
        if len(esports_markets) == 0:
            logger.warning("\n❌ PROBLEM IDENTIFIED:")
            logger.warning("No esports markets were written to the database.")
            logger.warning("\nThis means:")
            logger.warning("  1. Esports markets WERE discovered (logs showed 1454 from Kalshi, 3038 from Polymarket)")
            logger.warning("  2. But they were NOT matched/verified as pairs")
            logger.warning("  3. Markets are only written when a verified pair is found")
            logger.warning("\nPossible causes:")
            logger.warning("  • Esports markets from Kalshi and Polymarket are about different games/events")
            logger.warning("  • Matching candidates were found but verification confidence was too low")
            logger.warning("  • No matching candidates were found (embedding similarity too low)")
            logger.warning("\nNext steps:")
            logger.warning("  • Check pipeline logs for matching attempts")
            logger.warning("  • Lower verification confidence threshold temporarily")
            logger.warning("  • Check if esports markets from both venues are actually similar")
        else:
            logger.info(f"\n✅ Found {len(esports_markets)} esports markets in database")
            logger.info("But they didn't form verified pairs. Possible reasons:")
            logger.info("  • Markets from different venues weren't similar enough")
            logger.info("  • Verification confidence was below threshold")
            logger.info("  • Outcome mappings didn't align")

    except Exception as e:
        logger.error(f"Error querying database: {e}", exc_info=True)
    finally:
        await conn.close()
        logger.info("\nDatabase connection closed")


async def main():
    """Main entry point."""
    logger.info("Investigating esports market matching...")
    await investigate()


if __name__ == "__main__":
    asyncio.run(main())

