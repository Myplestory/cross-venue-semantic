"""Pytest configuration for discovery tests."""

import pytest
import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def mock_websocket_connect():
    """Helper fixture to properly mock websockets.connect."""
    def _mock_connect(mock_ws=None):
        """Create a mock for websockets.connect that returns an awaitable."""
        if mock_ws is None:
            mock_ws = AsyncMock()
            mock_ws.closed = False
            mock_ws.send = AsyncMock()
            mock_ws.close = AsyncMock()
        
        async def mock_connect_func(*args, **kwargs):
            return mock_ws
        
        with patch("websockets.connect") as mock_connect:
            mock_connect.side_effect = mock_connect_func
            yield mock_connect, mock_ws
    
    return _mock_connect
