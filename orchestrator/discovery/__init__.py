"""
Discovery strategy factory.

Creates discovery strategies based on configuration.
Implements proper error handling and validation.
"""

import logging
from typing import Optional

from orchestrator.discovery.base import DiscoveryStrategy
from orchestrator.discovery.normal import NormalDiscoveryStrategy
from discovery.strategies.esports import EsportsDiscoveryStrategy
from orchestrator.exceptions import ConfigurationError
import config

logger = logging.getLogger(__name__)


def create_discovery_strategy(
    mode: Optional[str] = None,
) -> DiscoveryStrategy:
    """
    Create discovery strategy based on DISCOVERY_MODE env var.
    
    Args:
        mode: Optional mode override (default: from config.DISCOVERY_MODE)
        
    Returns:
        DiscoveryStrategy instance
        
    Raises:
        ConfigurationError: If DISCOVERY_MODE is invalid or strategy
            creation fails
    """
    if mode is None:
        mode = (config.DISCOVERY_MODE or "normal").lower().strip()
    else:
        mode = mode.lower().strip()
    
    try:
        if mode == "normal":
            strategy = NormalDiscoveryStrategy()
        elif mode == "esports":
            strategy = EsportsDiscoveryStrategy()
        else:
            raise ConfigurationError(
                f"Invalid DISCOVERY_MODE: {mode}. "
                f"Must be one of: normal, esports (hybrid coming soon)"
            )
        
        logger.info(
            f"Created discovery strategy: {strategy.get_name()} "
            f"({strategy.get_description()})"
        )
        
        return strategy
        
    except Exception as exc:
        raise ConfigurationError(
            f"Failed to create discovery strategy (mode={mode}): {exc}",
            cause=exc,
        ) from exc


# Export for convenience
__all__ = [
    "DiscoveryStrategy",
    "NormalDiscoveryStrategy",
    "EsportsDiscoveryStrategy",
    "create_discovery_strategy",
]

