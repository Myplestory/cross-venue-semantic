#!/usr/bin/env python3
"""Check database for Polymarket market to get resolution date."""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import asyncpg
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Check database for Polymarket market."""
    conn = await asyncpg.connect(config.DATABASE_URL)
    
    # Search for Polymarket market with T1 Academy and KT Rolster
    query = """
    SELECT 
        venue_market_id,
        title,
        description,
        end_date,
        created_at
    FROM markets
    WHERE venue = 'polymarket'
    AND (
        LOWER(title) LIKE '%t1 academy%' 
        AND LOWER(title) LIKE '%kt rolster%'
        AND (LOWER(title) LIKE '%game 3%' OR LOWER(title) LIKE '%map 3%')
    )
    ORDER BY created_at DESC
    LIMIT 10
    """
    
    rows = await conn.fetch(query)
    
    print(f"Found {len(rows)} Polymarket markets matching criteria:\n")
    for i, row in enumerate(rows, 1):
        print(f"{i}. {row['title']}")
        print(f"   Market ID: {row['venue_market_id']}")
        print(f"   End Date: {row['end_date']}")
        print(f"   Created: {row['created_at']}")
        print()
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())

