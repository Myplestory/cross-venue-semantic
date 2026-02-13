"""
Semantic Pipeline — entrypoint.

Usage::

    # From the PolyEdge directory:
    python -m semantic_pipeline

    # Or from semantic_pipeline directory:
    python __main__.py

    # Override venues / workers via env:
    ORCHESTRATOR_VENUES=kalshi ORCHESTRATOR_NUM_WORKERS=2 python -m semantic_pipeline
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# ── Path bootstrapping ──────────────────────────────────────────────────
_PIPELINE_ROOT = str(Path(__file__).resolve().parent)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)


def _setup_logging() -> None:
    """
    Configure structured logging.

    Format: ``[TIMESTAMP] LEVEL  module  message``

    Levels:
    - Root logger: from LOG_LEVEL env (default INFO)
    - ``sentence_transformers``, ``transformers``, ``torch``: WARNING
    - ``qdrant_client``: WARNING
    - ``asyncpg``: WARNING
    - ``websockets``: DEBUG when LOG_LEVEL=DEBUG or WEBSOCKET_DEBUG=true (handshake request/response)
    """
    fmt = (
        "[%(asctime)s] %(levelname)-7s %(name)-30s %(message)s"
    )
    level_name = (os.getenv("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Quieten noisy libraries
    for lib in (
        "sentence_transformers",
        "transformers",
        "torch",
        "qdrant_client",
        "asyncpg",
        "httpx",
        "httpcore",
        "urllib3",
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)

    # WebSocket handshake debugging: see GET line and headers sent/received
    if level <= logging.DEBUG or os.getenv("WEBSOCKET_DEBUG", "").lower() in ("1", "true", "yes"):
        logging.getLogger("websockets").setLevel(logging.DEBUG)


async def _run() -> None:
    """Initialize and run the orchestrator."""
    from orchestrator import SemanticPipelineOrchestrator

    orchestrator = SemanticPipelineOrchestrator()
    await orchestrator.initialize()

    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        # Windows: SIGINT arrives as KeyboardInterrupt
        pass
    finally:
        await orchestrator.shutdown()


def main() -> None:
    """CLI entrypoint."""
    _setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Semantic Pipeline...")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("Interrupted — exiting")


if __name__ == "__main__":
    main()




