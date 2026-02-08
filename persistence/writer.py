"""
Database writer for semantic pipeline outputs.

Writes to:
- markets table (upsert by venue + venue_market_id)
- contract_specs table (idempotent by market_id + text_hash + model + prompt)
- verified_pairs table (transactional versioning with is_current flag)
- market_snapshots table (point-in-time captures)
"""

