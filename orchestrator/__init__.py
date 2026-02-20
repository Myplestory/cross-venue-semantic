"""
Orchestrator module for semantic pipeline.

Modular orchestrator with discovery strategy pattern.
"""

from orchestrator.core import SemanticPipelineOrchestrator
from orchestrator.metrics import PipelineMetrics, StageMetrics, STAGE_NAMES
from orchestrator.discovery import create_discovery_strategy

__all__ = [
    "SemanticPipelineOrchestrator",
    "PipelineMetrics",
    "StageMetrics",
    "STAGE_NAMES",
    "create_discovery_strategy",
]

