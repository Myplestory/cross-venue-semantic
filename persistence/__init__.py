"""
Persistence Module

Writes markets, contract_specs, and verified_pairs to PostgreSQL.
Handles transactional versioning and idempotency.

Fires pg_notify('pair_changes') to wake the Rust data plane for
config refresh (new/updated verified pairs).
"""

from .writer import PipelineWriter, PairWriteRequest

__all__ = ["PipelineWriter", "PairWriteRequest"]
