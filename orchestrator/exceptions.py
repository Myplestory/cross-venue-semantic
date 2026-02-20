"""
Custom exceptions for orchestrator module.

Fintech-grade error handling with clear exception hierarchy.
"""

from typing import Optional


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    
    def __init__(
        self,
        message: str,
        *,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.correlation_id = correlation_id
        self.cause = cause


class DiscoveryStrategyError(OrchestratorError):
    """Error in discovery strategy execution."""
    pass


class BootstrapError(OrchestratorError):
    """Error during bootstrap phase."""
    pass


class ConnectorError(OrchestratorError):
    """Error in venue connector."""
    pass


class PipelineError(OrchestratorError):
    """Error in pipeline processing."""
    pass


class ConfigurationError(OrchestratorError):
    """Configuration validation error."""
    pass

