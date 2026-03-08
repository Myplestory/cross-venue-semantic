"""
Riot Games API client for live League of Legends match data.

Supports:
- Live game data (spectator mode)
- Match timeline (events: kills, objectives, etc.)
- Tournament match data (via esports API)

Note: Riot API requires API key and has rate limits.
For production use, consider using Polymarket Sports WebSocket as primary
source and Riot API as supplementary for detailed event data.
"""

import os
import logging
from typing import Optional, Dict, Callable, Any
from datetime import datetime, UTC

import aiohttp

from ..compliance.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from ..compliance.metrics import SystemMetrics

logger = logging.getLogger(__name__)

RIOT_API_BASE = "https://{region}.api.riotgames.com"


class RiotAPIClient:
    """
    Riot Games API client for live League of Legends match data.
    
    Supports:
    - Live game data (spectator mode)
    - Match timeline (events: kills, objectives, etc.)
    - Tournament match data (via esports API)
    
    Note: Riot API has strict rate limits. Use circuit breaker and
    respect rate limits to avoid API key suspension.
    """
    
    def __init__(
        self,
        api_key: str,
        region: str = "americas",  # americas, asia, europe
        circuit_breaker: Optional[CircuitBreaker] = None,
        metrics: Optional[SystemMetrics] = None,
    ):
        """
        Initialize Riot API client.
        
        Args:
            api_key: Riot Games API key
            region: API region (americas, asia, europe)
            circuit_breaker: Optional circuit breaker for resilience
            metrics: Optional SystemMetrics for tracking
        """
        self.api_key = api_key
        self.region = region
        self.circuit_breaker = circuit_breaker
        self.metrics = metrics
        self.base_url = RIOT_API_BASE.format(region=region)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def get_live_match(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get live match data via spectator API.
        
        Args:
            match_id: Match identifier from tournament API or active game detection
        
        Returns:
            Match data dict or None if not found
        """
        # Note: This endpoint may require different match_id format
        # Implementation depends on Riot API version and match type
        url = f"{self.base_url}/lol/spectator/v5/active-games/by-summoner/{match_id}"
        headers = {"X-Riot-Token": self.api_key}
        
        try:
            if self.circuit_breaker:
                result = await self.circuit_breaker.call(
                    self._fetch_json, url, headers
                )
            else:
                result = await self._fetch_json(url, headers)
            
            if self.metrics and result:
                await self.metrics.increment_api_success("riot_api")
            elif self.metrics:
                await self.metrics.increment_api_error("riot_api")
            
            return result
        except CircuitBreakerOpenError:
            if self.metrics:
                await self.metrics.increment_api_error("riot_api")
            raise
        except Exception as e:
            logger.debug(f"[Riot API] Error fetching live match: {e}")
            if self.metrics:
                await self.metrics.increment_api_error("riot_api")
            return None
    
    async def get_match_timeline(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get match timeline with detailed events.
        
        Events include: kills, turret destruction, dragon/baron, etc.
        
        Args:
            match_id: Match identifier
        
        Returns:
            Timeline data dict or None if not found
        """
        url = f"{self.base_url}/lol/match/v5/matches/{match_id}/timeline"
        headers = {"X-Riot-Token": self.api_key}
        
        try:
            if self.circuit_breaker:
                result = await self.circuit_breaker.call(
                    self._fetch_json, url, headers
                )
            else:
                result = await self._fetch_json(url, headers)
            
            if self.metrics and result:
                await self.metrics.increment_api_success("riot_api")
            elif self.metrics:
                await self.metrics.increment_api_error("riot_api")
            
            return result
        except CircuitBreakerOpenError:
            if self.metrics:
                await self.metrics.increment_api_error("riot_api")
            raise
        except Exception as e:
            logger.debug(f"[Riot API] Error fetching timeline: {e}")
            if self.metrics:
                await self.metrics.increment_api_error("riot_api")
            return None
    
    async def get_tournament_match(self, tournament_id: str, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get tournament match data (official esports matches).
        
        More reliable than spectator API for professional matches.
        Note: Endpoint may vary by Riot API version.
        
        Args:
            tournament_id: Tournament identifier
            match_id: Match identifier
        
        Returns:
            Tournament match data dict or None if not found
        """
        # Note: Tournament API endpoint structure may vary
        # This is a placeholder - actual endpoint depends on Riot's esports API
        url = f"{self.base_url}/lol/esports/match/v1/matches/{match_id}"
        headers = {"X-Riot-Token": self.api_key}
        
        try:
            if self.circuit_breaker:
                result = await self.circuit_breaker.call(
                    self._fetch_json, url, headers
                )
            else:
                result = await self._fetch_json(url, headers)
            
            if self.metrics and result:
                await self.metrics.increment_api_success("riot_api")
            elif self.metrics:
                await self.metrics.increment_api_error("riot_api")
            
            return result
        except CircuitBreakerOpenError:
            if self.metrics:
                await self.metrics.increment_api_error("riot_api")
            raise
        except Exception as e:
            logger.debug(f"[Riot API] Error fetching tournament match: {e}")
            if self.metrics:
                await self.metrics.increment_api_error("riot_api")
            return None
    
    async def _fetch_json(self, url: str, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Internal method to fetch JSON from Riot API.
        
        Handles rate limits (429) with Retry-After header support.
        """
        if not self.session:
            return None
        
        async with self.session.get(
            url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            elif resp.status == 404:
                return None  # Not found (no active game, etc.)
            elif resp.status == 429:
                # Rate limit exceeded - check Retry-After header
                retry_after = resp.headers.get("Retry-After", "1")
                try:
                    retry_seconds = int(retry_after)
                except ValueError:
                    retry_seconds = 1
                
                logger.warning(
                    f"[Riot API] Rate limited (429) for {url}. "
                    f"Retry-After: {retry_seconds}s"
                )
                
                if self.metrics:
                    await self.metrics.increment_api_error("riot_api_rate_limit")
                
                # Raise exception so caller can handle retry logic
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info,
                    history=resp.history,
                    status=resp.status,
                    message=f"Rate limit exceeded. Retry after {retry_seconds}s",
                    headers=resp.headers,
                )
            else:
                logger.warning(f"[Riot API] Status {resp.status} for {url}")
                if self.metrics:
                    await self.metrics.increment_api_error(f"riot_api_http_{resp.status}")
                return None


class RiotEventPoller:
    """
    Polls Riot API for game events and correlates with odds updates.
    
    Polls match timeline periodically to extract significant events
    (kills, objectives, etc.) and notifies the latency correlation engine.
    """
    
    def __init__(
        self,
        riot_client: RiotAPIClient,
        match_id: str,
        on_event: Callable[[Dict[str, Any]], None],
        poll_interval: float = 2.0,  # Poll every 2 seconds
    ):
        """
        Initialize Riot event poller.
        
        Args:
            riot_client: RiotAPIClient instance
            match_id: Match identifier to poll
            on_event: Callback when game event is detected
            poll_interval: Polling interval in seconds
        """
        self.riot_client = riot_client
        self.match_id = match_id
        self.on_event = on_event
        self.poll_interval = poll_interval
        self._running = False
        self.last_event_timestamp: Optional[datetime] = None
        self.processed_event_ids: set = set()  # Track processed events to avoid duplicates
    
    async def run(self):
        """
        Poll for game events continuously.
        
        Extracts events from match timeline and calls on_event callback
        for each new significant event (kills, objectives, etc.).
        """
        self._running = True
        
        while self._running:
            try:
                timeline = await self.riot_client.get_match_timeline(self.match_id)
                if timeline:
                    await self._process_timeline(timeline)
                
                await asyncio.sleep(self.poll_interval)
            
            except Exception as e:
                logger.error(f"[Riot Poller] Error: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)
    
    async def _process_timeline(self, timeline: Dict[str, Any]):
        """Process timeline data and extract events."""
        try:
            frames = timeline.get("info", {}).get("frames", [])
            for frame in frames:
                events = frame.get("events", [])
                for event in events:
                    event_type = event.get("type", "")
                    
                    # Filter significant events
                    if event_type in ["CHAMPION_KILL", "BUILDING_KILL", "ELITE_MONSTER_KILL"]:
                        event_id = event.get("eventId") or event.get("timestamp")
                        
                        # Skip if already processed
                        if event_id in self.processed_event_ids:
                            continue
                        
                        # Extract timestamp
                        timestamp_ms = event.get("timestamp", 0)
                        event_ts = datetime.fromtimestamp(timestamp_ms / 1000, UTC)
                        
                        # Only process new events
                        if not self.last_event_timestamp or event_ts > self.last_event_timestamp:
                            self.last_event_timestamp = event_ts
                            self.processed_event_ids.add(event_id)
                            
                            # Notify callback
                            event_data = {
                                "source": "riot_api",
                                "timestamp": event_ts,
                                "event_type": event_type,
                                "event_data": event,
                            }
                            
                            try:
                                if asyncio.iscoroutinefunction(self.on_event):
                                    await self.on_event(event_data)
                                else:
                                    self.on_event(event_data)
                            except Exception as e:
                                logger.error(
                                    f"[Riot Poller] Error in event callback: {e}",
                                    exc_info=True
                                )
        except Exception as e:
            logger.error(f"[Riot Poller] Error processing timeline: {e}", exc_info=True)
    
    async def stop(self):
        """Stop the event poller."""
        self._running = False

