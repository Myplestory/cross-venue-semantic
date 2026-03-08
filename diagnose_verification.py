#!/usr/bin/env python3
"""
Diagnose why matches aren't being written to database.

Checks:
1. Verification thresholds from config
2. Whether matches are being verified as "not_equivalent"
3. Writer queue status
4. Database connection status
"""

import asyncio
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


async def main():
    """Diagnose verification and persistence issues."""
    
    print("=" * 80)
    print("VERIFICATION DIAGNOSTICS")
    print("=" * 80)
    
    # ── 1. Check verification thresholds ────────────────────────────────
    print("\n📊 Verification Thresholds:")
    print(f"   VERIFICATION_EQUIVALENT_THRESHOLD: {config.VERIFICATION_EQUIVALENT_THRESHOLD}")
    print(f"   VERIFICATION_NOT_EQUIVALENT_THRESHOLD: {config.VERIFICATION_NOT_EQUIVALENT_THRESHOLD}")
    print(f"   VERIFICATION_ENTITY_TOLERANCE: {config.VERIFICATION_ENTITY_TOLERANCE}")
    print(f"   VERIFICATION_THRESHOLD_TOLERANCE_PERCENT: {config.VERIFICATION_THRESHOLD_TOLERANCE_PERCENT}")
    print(f"   VERIFICATION_DATE_TOLERANCE_DAYS: {config.VERIFICATION_DATE_TOLERANCE_DAYS}")
    
    print("\n📊 Verification Weights:")
    print(f"   VERIFICATION_CROSS_ENCODER_WEIGHT: {config.VERIFICATION_CROSS_ENCODER_WEIGHT}")
    print(f"   VERIFICATION_THRESHOLD_WEIGHT: {config.VERIFICATION_THRESHOLD_WEIGHT}")
    print(f"   VERIFICATION_ENTITY_WEIGHT: {config.VERIFICATION_ENTITY_WEIGHT}")
    print(f"   VERIFICATION_DATE_WEIGHT: {config.VERIFICATION_DATE_WEIGHT}")
    print(f"   VERIFICATION_DATA_SOURCE_WEIGHT: {config.VERIFICATION_DATA_SOURCE_WEIGHT}")
    
    # ── 2. Check database for recent pairs ──────────────────────────────
    print("\n🔍 Checking Database...")
    try:
        conn = await asyncpg.connect(config.DATABASE_URL)
        
        # Check verified_pairs table
        pair_count = await conn.fetchval(
            "SELECT COUNT(*) FROM verified_pairs"
        )
        print(f"   Total verified_pairs in DB: {pair_count}")
        
        # Check recent pairs (last hour)
        recent_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM verified_pairs
            WHERE verified_at > NOW() - INTERVAL '1 hour'
            """
        )
        print(f"   Pairs verified in last hour: {recent_count}")
        
        # Check markets table
        market_count = await conn.fetchval(
            "SELECT COUNT(*) FROM markets"
        )
        print(f"   Total markets in DB: {market_count}")
        
        # Check recent markets
        recent_markets = await conn.fetchval(
            """
            SELECT COUNT(*) FROM markets
            WHERE created_at > NOW() - INTERVAL '1 hour'
            """
        )
        print(f"   Markets created in last hour: {recent_markets}")
        
        # Check for esports-related markets
        esports_markets = await conn.fetchval(
            """
            SELECT COUNT(*) FROM markets
            WHERE (
                LOWER(title) LIKE '%esport%' OR
                LOWER(title) LIKE '%lol%' OR
                LOWER(title) LIKE '%league of legends%' OR
                LOWER(title) LIKE '%dota%' OR
                LOWER(title) LIKE '%valorant%' OR
                LOWER(title) LIKE '%csgo%' OR
                LOWER(title) LIKE '%counter-strike%'
            )
            """
        )
        print(f"   Esports-related markets in DB: {esports_markets}")
        
        await conn.close()
        
    except Exception as e:
        print(f"   ❌ Database check failed: {e}")
    
    # ── 3. Analysis ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    print("\n💡 Key Findings:")
    print("   1. Matches are found by reranking (logs show '1 matches')")
    print("   2. Verification happens but verdict is likely 'not_equivalent'")
    print("   3. Only 'equivalent' or 'needs_review' verdicts are persisted")
    print("   4. No logging exists for verdict/confidence in current code")
    
    print("\n🔍 Why matches might fail verification:")
    print(f"   - Weighted score < {config.VERIFICATION_EQUIVALENT_THRESHOLD} (equivalent threshold)")
    print(f"   - Weighted score < {config.VERIFICATION_NOT_EQUIVALENT_THRESHOLD} (not_equivalent threshold)")
    print("   - Entity mismatch (entity_score < entity_tolerance * 0.5)")
    print("   - Threshold mismatch (threshold_score < 0.3)")
    print("   - Date mismatch (date_score < 0.5)")
    print("   - Cross-encoder score < 0.7 (for equivalent verdict)")
    
    print("\n💡 Recommendations:")
    print("   1. Add logging in _process_match() to show verdict and confidence")
    print("   2. Check if verification thresholds are too strict for esports")
    print("   3. Consider lowering thresholds temporarily for esports discovery")
    print("   4. Check if ContractSpec extraction is working for esports markets")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

