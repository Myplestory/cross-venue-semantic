"""
Automatic match ID discovery for esports series.

Discovers Riot match IDs for a series (e.g., best-of-5) by:
1. Monitoring Polymarket Sports WebSocket for game state (period info)
2. Querying Polymarket API for game-specific markets
3. Correlating with Riot API when available

For a best-of-5 series, automatically discovers match IDs for:
- Game 1, Game 2, Game 3, Game 4, Game 5 (as they become available)
"""

import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Set
from datetime import datetime, UTC, timedelta
from collections import defaultdict

import aiohttp

import sys
from pathlib import Path

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from ..feeds.riot_api import RiotAPIClient
from ..compliance.metrics import SystemMetrics

logger = logging.getLogger(__name__)

GAMMA_API_BASE = config.POLYMARKET_GAMMA_API_URL or "https://gamma-api.polymarket.com"


class MatchDiscovery:
    """
    Discovers Riot match IDs for esports series automatically.
    
    Uses Polymarket Sports WebSocket game state to identify active games,
    then attempts to correlate with Riot match IDs.
    """
    
    def __init__(
        self,
        riot_client: Optional[RiotAPIClient] = None,
        metrics: Optional[SystemMetrics] = None,
    ):
        """
        Initialize match discovery.
        
        Args:
            riot_client: Optional RiotAPIClient for API queries
            metrics: Optional SystemMetrics for tracking
        """
        self.riot_client = riot_client
        self.metrics = metrics
        self.discovered_games: Dict[int, Dict[str, Any]] = {}  # game_number -> match_info
    
    async def discover_series_from_polymarket_slug(
        self,
        market_slug: str,
        session: aiohttp.ClientSession,
    ) -> List[Dict[str, Any]]:
        """
        Discover all games in a series from Polymarket market slug.
        
        Queries Polymarket API for markets related to the series and extracts
        game numbers from market titles (e.g., "Game 1 Winner", "Game 2 Winner").
        
        Args:
            market_slug: Polymarket market slug (e.g., "lol-t1-dk-2026-02-22")
            session: aiohttp ClientSession
        
        Returns:
            List of game info dicts with game_number, market_id, etc.
        """
        games = []
        
        try:
            # Search Polymarket for markets containing the slug/teams
            # Extract game numbers from market titles
            url = f"{GAMMA_API_BASE}/markets"
            
            # Try searching by slug
            params = {"slug": market_slug, "closed": "false", "active": "true", "limit": 100}
            
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    markets = data if isinstance(data, list) else data.get("data", [])
                    
                    # Extract game numbers from market titles
                    game_pattern = re.compile(r'game\s+(\d+)', re.IGNORECASE)
                    game_markets: Dict[int, Dict] = defaultdict(dict)
                    
                    for market in markets:
                        if not isinstance(market, dict):
                            continue
                        
                        title = market.get("question") or market.get("title", "")
                        market_id = market.get("condition_id") or market.get("id")
                        
                        # Check for game number in title
                        match = game_pattern.search(title)
                        if match:
                            game_num = int(match.group(1))
                            if 1 <= game_num <= 5:  # Valid for Bo5
                                game_markets[game_num][market_id] = {
                                    "game_number": game_num,
                                    "market_id": market_id,
                                    "title": title,
                                    "slug": market.get("slug"),
                                }
                    
                    # Build game list
                    for game_num in sorted(game_markets.keys()):
                        # Take first market for each game (or could aggregate)
                        market_info = list(game_markets[game_num].values())[0]
                        games.append({
                            "game_number": game_num,
                            "polymarket_market_id": market_info["market_id"],
                            "polymarket_title": market_info["title"],
                            "polymarket_slug": market_info.get("slug"),
                            "riot_match_id": None,  # To be populated if Riot API available
                            "status": "scheduled",  # Will be updated from Sports WS
                        })
                    
                    logger.info(
                        f"[MatchDiscovery] Found {len(games)} game(s) from Polymarket: "
                        f"{[g['game_number'] for g in games]}"
                    )
        
        except Exception as e:
            logger.error(
                f"[MatchDiscovery] Error discovering games from Polymarket: {e}",
                exc_info=True
            )
            if self.metrics:
                await self.metrics.increment_api_error("polymarket_match_discovery")
        
        return games
    
    async def update_game_status_from_sports_ws(
        self,
        game_state: Dict[str, Any],
        discovered_games: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Update game status from Polymarket Sports WebSocket game state.
        
        Extracts period information (e.g., "2/5" means Game 2 of Bo5) and
        updates the discovered games list.
        
        Args:
            game_state: Game state dict from Polymarket Sports WS
            discovered_games: List of discovered games to update
        
        Returns:
            Updated games list
        """
        period = game_state.get("period")  # e.g., "2/5"
        status = game_state.get("status")  # e.g., "InProgress", "Final"
        
        if period:
            # Parse period (e.g., "2/5" -> game_number=2, total=5)
            period_match = re.match(r'(\d+)/(\d+)', str(period))
            if period_match:
                current_game = int(period_match.group(1))
                total_games = int(period_match.group(2))
                
                # Update status for current game
                for game in discovered_games:
                    if game["game_number"] == current_game:
                        game["status"] = status.lower() if status else "unknown"
                        game["is_current"] = True
                    elif game["game_number"] < current_game:
                        game["status"] = "completed"
                        game["is_current"] = False
                    else:
                        game["status"] = "scheduled"
                        game["is_current"] = False
        
        return discovered_games
    
    async def correlate_with_riot_api(
        self,
        discovered_games: List[Dict[str, Any]],
        team_a: str,
        team_b: str,
    ) -> List[Dict[str, Any]]:
        """
        Attempt to correlate discovered games with Riot match IDs.
        
        For each game, tries to find corresponding Riot match ID using
        team names and game number.
        
        Args:
            discovered_games: List of discovered games
            team_a: First team name
            team_b: Second team name
        
        Returns:
            Updated games list with Riot match IDs if found
        """
        if not self.riot_client:
            return discovered_games
        
        # Note: This requires Riot esports API endpoints
        # Implementation depends on available API structure
        # For now, games will have riot_match_id=None until API is available
        
        logger.info(
            f"[MatchDiscovery] Attempting Riot API correlation for {len(discovered_games)} game(s)"
        )
        
        return discovered_games


async def discover_series_games(
    market_slug: str,
    team_a: str,
    team_b: str,
    session: aiohttp.ClientSession,
    riot_client: Optional[RiotAPIClient] = None,
    metrics: Optional[SystemMetrics] = None,
) -> List[Dict[str, Any]]:
    """
    High-level function to discover all games in a series.
    
    Combines Polymarket market discovery with Riot API correlation.
    
    Args:
        market_slug: Polymarket market slug
        team_a: First team name
        team_b: Second team name
        session: aiohttp ClientSession
        riot_client: Optional RiotAPIClient
        metrics: Optional SystemMetrics
    
    Returns:
        List of game info dicts with game_number, market_id, match_id, etc.
    """
    discovery = MatchDiscovery(riot_client=riot_client, metrics=metrics)
    
    # Discover games from Polymarket
    games = await discovery.discover_series_from_polymarket_slug(market_slug, session)
    
    # Attempt Riot API correlation
    if riot_client:
        games = await discovery.correlate_with_riot_api(games, team_a, team_b)
    
    return games

