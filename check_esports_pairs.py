#!/usr/bin/env python3
"""
Check if esports pairs were written to the database.

Queries verified_pairs table and shows:
- Total pairs found
- Esports-specific pairs (filtered by keywords)
- Recent pairs (by verified_at timestamp)
- Pair details with confidence scores
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
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


async def check_pairs():
    """Query database for verified pairs."""
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
        # ── Query 1: Total pairs ────────────────────────────────────────
        total_count = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM verified_pairs vp
            WHERE vp.verdict IN ('equivalent', 'needs_review')
              AND vp.is_current = true
              AND vp.is_active = true
            """
        )
        logger.info(f"Total verified pairs: {total_count}")

        if total_count == 0:
            logger.warning("No verified pairs found in database!")
            logger.info("This could mean:")
            logger.info("  1. Pipeline hasn't completed matching yet")
            logger.info("  2. No pairs were verified as equivalent")
            logger.info("  3. Pairs were written but marked inactive")
            return

        # ── Query 2: All pairs with details ─────────────────────────────
        rows = await conn.fetch(
            """
            SELECT
                vp.pair_key,
                vp.outcome_mapping,
                vp.confidence_score,
                vp.verdict,
                vp.verified_at,
                ma.venue   AS venue_a,
                ma.venue_market_id AS vmid_a,
                ma.title   AS title_a,
                ma.description AS desc_a,
                mb.venue   AS venue_b,
                mb.venue_market_id AS vmid_b,
                mb.title   AS title_b,
                mb.description AS desc_b
            FROM verified_pairs vp
            JOIN markets ma ON ma.id = vp.market_a_id
            JOIN markets mb ON mb.id = vp.market_b_id
            WHERE vp.verdict IN ('equivalent', 'needs_review')
              AND vp.is_current = true
              AND vp.is_active  = true
            ORDER BY vp.verified_at DESC, vp.confidence_score DESC
            LIMIT 100
            """
        )

        logger.info(f"\n{'='*80}")
        logger.info(f"Found {len(rows)} pairs (showing up to 100 most recent)")
        logger.info(f"{'='*80}\n")

        # ── Filter for esports pairs ────────────────────────────────────
        esports_pairs = []
        for row in rows:
            title_a = (row["title_a"] or "").lower()
            title_b = (row["title_b"] or "").lower()
            desc_a = (row["desc_a"] or "").lower()
            desc_b = (row["desc_b"] or "").lower()
            
            text = f"{title_a} {title_b} {desc_a} {desc_b}"
            
            if any(keyword in text for keyword in ESPORTS_KEYWORDS):
                esports_pairs.append(row)

        logger.info(f"Esports-related pairs: {len(esports_pairs)}")
        logger.info(f"Non-esports pairs: {len(rows) - len(esports_pairs)}\n")

        # ── Display all pairs ────────────────────────────────────────────
        if rows:
            logger.info("All Verified Pairs:")
            logger.info("-" * 80)
            
            for i, row in enumerate(rows, 1):
                outcome_map = row["outcome_mapping"]
                if isinstance(outcome_map, str):
                    outcome_map = json.loads(outcome_map)
                elif outcome_map is None:
                    outcome_map = {}
                
                outcome_str = ", ".join([
                    f"{k} → {v}" for k, v in outcome_map.items()
                ]) if outcome_map else "N/A"
                
                verified_at = row["verified_at"]
                if isinstance(verified_at, datetime):
                    verified_str = verified_at.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    verified_str = str(verified_at)
                
                is_esports = row in esports_pairs
                esports_tag = " [ESPORTS]" if is_esports else ""
                
                logger.info(f"\n{i}. Pair Key: {row['pair_key']}{esports_tag}")
                logger.info(f"   Confidence: {row['confidence_score']:.3f} | Verdict: {row['verdict']}")
                logger.info(f"   Verified: {verified_str}")
                logger.info(f"   Outcome Mapping: {outcome_str}")
                logger.info(f"   {row['venue_a'].upper()}: {row['title_a'][:70]}")
                logger.info(f"   {row['venue_b'].upper()}: {row['title_b'][:70]}")

        # ── Display esports pairs separately ─────────────────────────────
        if esports_pairs:
            logger.info(f"\n{'='*80}")
            logger.info(f"ESPORTS PAIRS ({len(esports_pairs)} found):")
            logger.info(f"{'='*80}\n")
            
            for i, row in enumerate(esports_pairs, 1):
                outcome_map = row["outcome_mapping"]
                if isinstance(outcome_map, str):
                    outcome_map = json.loads(outcome_map)
                elif outcome_map is None:
                    outcome_map = {}
                
                outcome_str = ", ".join([
                    f"{k} → {v}" for k, v in outcome_map.items()
                ]) if outcome_map else "N/A"
                
                verified_at = row["verified_at"]
                if isinstance(verified_at, datetime):
                    verified_str = verified_at.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    verified_str = str(verified_at)
                
                logger.info(f"\n{i}. {row['pair_key']}")
                logger.info(f"   Confidence: {row['confidence_score']:.3f} | Verdict: {row['verdict']}")
                logger.info(f"   Verified: {verified_str}")
                logger.info(f"   Outcome: {outcome_str}")
                logger.info(f"   {row['venue_a'].upper()}: {row['title_a']}")
                logger.info(f"   {row['venue_b'].upper()}: {row['title_b']}")
                if row['desc_a']:
                    logger.info(f"   Desc A: {row['desc_a'][:100]}")
                if row['desc_b']:
                    logger.info(f"   Desc B: {row['desc_b'][:100]}")

        # ── Summary statistics ───────────────────────────────────────────
        logger.info(f"\n{'='*80}")
        logger.info("SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total verified pairs: {total_count}")
        logger.info(f"Esports pairs: {len(esports_pairs)}")
        logger.info(f"Non-esports pairs: {len(rows) - len(esports_pairs)}")
        
        if rows:
            avg_confidence = sum(r["confidence_score"] for r in rows) / len(rows)
            logger.info(f"Average confidence: {avg_confidence:.3f}")
            
            verdict_counts = {}
            for row in rows:
                verdict = row["verdict"]
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            logger.info(f"Verdicts: {verdict_counts}")

    except Exception as e:
        logger.error(f"Error querying database: {e}", exc_info=True)
    finally:
        await conn.close()
        logger.info("\nDatabase connection closed")


async def main():
    """Main entry point."""
    logger.info("Checking esports pairs in database...")
    logger.info(f"Database URL: {config.DATABASE_URL[:50]}..." if config.DATABASE_URL else "Not set")
    await check_pairs()


if __name__ == "__main__":
    asyncio.run(main())

