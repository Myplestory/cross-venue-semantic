"""
Market ID and token resolution utilities.
"""

import aiohttp
import logging
from typing import Optional, Tuple
from datetime import datetime

import sys
from pathlib import Path

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from spread_scanner import _fetch_gamma_tokens, GAMMA_API_BASE
import config

logger = logging.getLogger(__name__)


async def resolve_polymarket_id(slug: str) -> Optional[str]:
    """
    Resolve Polymarket slug to condition_id.
    
    Searches Polymarket Gamma API by slug to find the market condition_id.
    
    Args:
        slug: Market slug (e.g., "lol-t1-dk-2026-02-22")
    
    Returns:
        Condition ID (condition_id) or None if not found
    """
    gamma_base = config.POLYMARKET_GAMMA_API_URL or GAMMA_API_BASE
    url = f"{gamma_base}/markets"
    
    # Try searching by slug
    params = {"slug": slug, "closed": "false", "active": "true"}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # Handle both list and dict responses
                    markets = []
                    if isinstance(data, list):
                        markets = data
                    elif isinstance(data, dict):
                        markets = data.get("data", []) or [data]
                    
                    # Find matching slug
                    for market in markets:
                        if isinstance(market, dict):
                            market_slug = market.get("slug") or market.get("questionSlug")
                            if market_slug == slug:
                                # Return condition_id (preferred) or id
                                condition_id = (
                                    market.get("condition_id")
                                    or market.get("conditionId")
                                    or market.get("id")
                                )
                                if condition_id:
                                    logger.info(
                                        f"[MarketResolver] Resolved slug '{slug}' to condition_id: {condition_id}"
                                    )
                                    return str(condition_id)
                    
                    logger.warning(f"[MarketResolver] No market found for slug: {slug}")
                    return None
                else:
                    logger.warning(
                        f"[MarketResolver] API returned status {resp.status} for slug: {slug}"
                    )
                    return None
    except Exception as e:
        logger.error(f"[MarketResolver] Error resolving slug '{slug}': {e}", exc_info=True)
        return None


async def resolve_polymarket_tokens(
    session: aiohttp.ClientSession,
    market_id: str,
    slug: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], Optional[list], Optional[datetime]]:
    """
    Resolve Polymarket market ID to token IDs.
    
    Uses _fetch_gamma_tokens from spread_scanner to get yes/no token IDs.
    Falls back to slug-based lookup if condition_id lookup fails.
    
    Args:
        session: aiohttp ClientSession
        market_id: Polymarket condition_id or market ID
        slug: Optional market slug for fallback lookup
    
    Returns:
        Tuple of (yes_token_id, no_token_id, outcome_prices, resolution_date)
    """
    try:
        # First try: Direct market_id lookup
        yes_token, no_token, outcome_prices, resolution_date = await _fetch_gamma_tokens(
            session, market_id
        )
        
        if yes_token and no_token:
            logger.info(
                f"[MarketResolver] Resolved market_id '{market_id}' to tokens: "
                f"yes={yes_token[:20]}..., no={no_token[:20]}..."
            )
            return yes_token, no_token, outcome_prices, resolution_date
        
        # Fallback: Try slug-based lookup if slug provided
        if slug:
            logger.debug(
                f"[MarketResolver] Direct lookup failed, trying slug-based lookup: {slug}"
            )
            gamma_base = config.POLYMARKET_GAMMA_API_URL or GAMMA_API_BASE
            url = f"{gamma_base}/markets"
            params = {"slug": slug, "closed": "false", "active": "true"}
            
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # Handle both list and dict responses
                    markets = []
                    if isinstance(data, list):
                        markets = data
                    elif isinstance(data, dict):
                        markets = data.get("data", []) or [data]
                    
                    # Find matching market
                    for market in markets:
                        if isinstance(market, dict):
                            market_slug = market.get("slug") or market.get("questionSlug")
                            if market_slug == slug:
                                # Extract tokens directly from market data
                                from spread_scanner import _parse_gamma_market
                                yes_token, no_token, outcome_prices, resolution_date = _parse_gamma_market(market)
                                
                                if yes_token and no_token:
                                    logger.info(
                                        f"[MarketResolver] Resolved via slug '{slug}' to tokens: "
                                        f"yes={yes_token[:20]}..., no={no_token[:20]}..."
                                    )
                                    return yes_token, no_token, outcome_prices, resolution_date
        
        logger.warning(
            f"[MarketResolver] Could not resolve tokens for market_id: {market_id}"
            + (f" (slug: {slug})" if slug else "")
        )
        return None, None, None, None
        
    except Exception as e:
        logger.error(
            f"[MarketResolver] Error resolving tokens for '{market_id}': {e}",
            exc_info=True
        )
        return None, None, None, None

