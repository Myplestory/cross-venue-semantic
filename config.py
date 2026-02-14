"""
Configuration module for Semantic Pipeline

Loads configuration from environment variables and database runtime_config.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file from semantic_pipeline directory
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

# Log if .env was loaded
if _env_path.exists():
    print(f"[OK] Loaded .env from: {_env_path}")
else:
    print(f"[WARN] .env file not found at: {_env_path}")
    print("       Using environment variables and defaults")


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable with optional default.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        
    Returns:
        Environment variable value or default
    """
    return os.getenv(key, default)


def get_env_int(key: str, default: int) -> int:
    """
    Get environment variable as integer.
    
    Args:
        key: Environment variable name
        default: Default integer value
        
    Returns:
        Integer value from environment or default
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get environment variable as boolean.
    
    Args:
        key: Environment variable name
        default: Default boolean value
        
    Returns:
        Boolean value from environment or default
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


# Qdrant Configuration
QDRANT_URL = get_env("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = get_env("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = get_env("QDRANT_COLLECTION_NAME", "market_embeddings")
QDRANT_VECTOR_SIZE = get_env_int("QDRANT_VECTOR_SIZE", 2048)

# Embedding Model Configuration
EMBEDDING_MODEL = get_env("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
EMBEDDING_DEVICE = get_env("EMBEDDING_DEVICE")  # None = auto-detect
EMBEDDING_BATCH_SIZE = get_env_int("EMBEDDING_BATCH_SIZE", 48)
EMBEDDING_MAX_LENGTH = get_env_int("EMBEDDING_MAX_LENGTH", 512)
EMBEDDING_DIM = get_env_int("EMBEDDING_DIM", 2048)
EMBEDDING_INSTRUCTION = get_env(
    "EMBEDDING_INSTRUCTION",
    "Given a prediction market contract, represent its core event, resolution "
    "conditions, and timeframe for matching equivalent contracts across "
    "different prediction market platforms"
)
EMBEDDING_QUANTIZATION = get_env_bool("EMBEDDING_QUANTIZATION", False)

# Cache Configuration
EMBEDDING_CACHE_MAX_SIZE = get_env_int("EMBEDDING_CACHE_MAX_SIZE", 10000)

# Cross-Encoder Configuration
# Default: cross-encoder/nli-deberta-v3-large (NLI model for semantic equivalence)
CROSS_ENCODER_MODEL = get_env("CROSS_ENCODER_MODEL", "cross-encoder/nli-deberta-v3-large")
CROSS_ENCODER_USE_SENTENCE_TRANSFORMERS = get_env_bool("CROSS_ENCODER_USE_SENTENCE_TRANSFORMERS", True)
CROSS_ENCODER_DEVICE = get_env("CROSS_ENCODER_DEVICE")  # None = auto-detect
CROSS_ENCODER_BATCH_SIZE = get_env_int("CROSS_ENCODER_BATCH_SIZE", 8)
CROSS_ENCODER_MAX_LENGTH = get_env_int("CROSS_ENCODER_MAX_LENGTH", 512)
CROSS_ENCODER_QUANTIZATION = get_env_bool("CROSS_ENCODER_QUANTIZATION", False)

# Confidence Scoring Thresholds
def get_env_float(key: str, default: float) -> float:
    """Get environment variable as float."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default

CROSS_ENCODER_ENTAILMENT_THRESHOLD = get_env_float("CROSS_ENCODER_ENTAILMENT_THRESHOLD", 0.7)
CROSS_ENCODER_NEUTRAL_THRESHOLD = get_env_float("CROSS_ENCODER_NEUTRAL_THRESHOLD", 0.3)
CROSS_ENCODER_SCORE_THRESHOLD = get_env_float("CROSS_ENCODER_SCORE_THRESHOLD", 0.7)
CROSS_ENCODER_PRIMARY_WEIGHT = get_env_float("CROSS_ENCODER_PRIMARY_WEIGHT", 0.7)
CROSS_ENCODER_SECONDARY_WEIGHT = get_env_float("CROSS_ENCODER_SECONDARY_WEIGHT", 0.3)
CROSS_ENCODER_TOP_K = get_env_int("CROSS_ENCODER_TOP_K", 10)

# Retrieval Configuration
# Lower threshold = higher recall (more candidates for cross-encoder to filter).
# Industry standard for two-stage retrieve-then-rerank is 0.45–0.55.
RETRIEVAL_SCORE_THRESHOLD = get_env_float("RETRIEVAL_SCORE_THRESHOLD", 0.5)
RETRIEVAL_TOP_K = get_env_int("RETRIEVAL_TOP_K", 20)

# ContractSpec Extraction Configuration
EXTRACTION_USE_LLM_FALLBACK = get_env_bool("EXTRACTION_USE_LLM_FALLBACK", False)
EXTRACTION_CONFIDENCE_THRESHOLD = get_env_float("EXTRACTION_CONFIDENCE_THRESHOLD", 0.7)
EXTRACTION_HIGH_CONFIDENCE_THRESHOLD = get_env_float("EXTRACTION_HIGH_CONFIDENCE_THRESHOLD", 0.9)
EXTRACTION_LLM_MODEL = get_env("EXTRACTION_LLM_MODEL", "gpt-4o-mini")
EXTRACTION_LLM_API_KEY = get_env("OPENAI_API_KEY")
EXTRACTION_NER_MODEL = get_env("EXTRACTION_NER_MODEL", "en_core_web_sm")
EXTRACTION_TRACK_EVIDENCE_SPANS = get_env_bool("EXTRACTION_TRACK_EVIDENCE_SPANS", False)
EXTRACTION_CACHE_MAX_SIZE = get_env_int("EXTRACTION_CACHE_MAX_SIZE", 1000)

# Circuit Breaker Configuration
EXTRACTION_CIRCUIT_BREAKER_FAILURE_THRESHOLD = get_env_int("EXTRACTION_CIRCUIT_BREAKER_FAILURE_THRESHOLD", 5)
EXTRACTION_CIRCUIT_BREAKER_RECOVERY_TIMEOUT = get_env_float("EXTRACTION_CIRCUIT_BREAKER_RECOVERY_TIMEOUT", 60.0)
EXTRACTION_CIRCUIT_BREAKER_SUCCESS_THRESHOLD = get_env_int("EXTRACTION_CIRCUIT_BREAKER_SUCCESS_THRESHOLD", 2)
EXTRACTION_CIRCUIT_BREAKER_TIMEOUT = get_env_float("EXTRACTION_CIRCUIT_BREAKER_TIMEOUT", 30.0)

# Pair Verification Configuration
VERIFICATION_ENTITY_TOLERANCE = get_env_float("VERIFICATION_ENTITY_TOLERANCE", 0.8)
VERIFICATION_THRESHOLD_TOLERANCE_PERCENT = get_env_float("VERIFICATION_THRESHOLD_TOLERANCE_PERCENT", 0.01)
VERIFICATION_DATE_TOLERANCE_DAYS = get_env_int("VERIFICATION_DATE_TOLERANCE_DAYS", 1)
VERIFICATION_EQUIVALENT_THRESHOLD = get_env_float("VERIFICATION_EQUIVALENT_THRESHOLD", 0.9)
VERIFICATION_NOT_EQUIVALENT_THRESHOLD = get_env_float("VERIFICATION_NOT_EQUIVALENT_THRESHOLD", 0.5)
VERIFICATION_CACHE_MAX_SIZE = get_env_int("VERIFICATION_CACHE_MAX_SIZE", 10000)

# Configurable weights (research-backed defaults)
VERIFICATION_CROSS_ENCODER_WEIGHT = get_env_float("VERIFICATION_CROSS_ENCODER_WEIGHT", 0.50)
VERIFICATION_THRESHOLD_WEIGHT = get_env_float("VERIFICATION_THRESHOLD_WEIGHT", 0.20)
VERIFICATION_ENTITY_WEIGHT = get_env_float("VERIFICATION_ENTITY_WEIGHT", 0.15)
VERIFICATION_DATE_WEIGHT = get_env_float("VERIFICATION_DATE_WEIGHT", 0.10)
VERIFICATION_DATA_SOURCE_WEIGHT = get_env_float("VERIFICATION_DATA_SOURCE_WEIGHT", 0.05)

# Persistence Writer Configuration
DATABASE_URL = get_env("DATABASE_URL")
WRITER_BATCH_SIZE = get_env_int("WRITER_BATCH_SIZE", 20)
WRITER_BATCH_TIMEOUT = get_env_float("WRITER_BATCH_TIMEOUT", 2.0)
WRITER_QUEUE_SIZE = get_env_int("WRITER_QUEUE_SIZE", 500)
WRITER_MAX_RETRIES = get_env_int("WRITER_MAX_RETRIES", 3)
WRITER_NOTIFY_CHANNEL = get_env("WRITER_NOTIFY_CHANNEL", "pair_changes")

# GPU Concurrency Control
# Semaphore permits per model — controls how many workers can run
# GPU forward passes concurrently on the same model.
# 1 = serialize (safest, zero extra VRAM).  2 = allow overlap (~+1 GB peak).
EMBEDDING_GPU_CONCURRENCY = get_env_int("EMBEDDING_GPU_CONCURRENCY", 1)
CROSS_ENCODER_GPU_CONCURRENCY = get_env_int("CROSS_ENCODER_GPU_CONCURRENCY", 1)

# Dynamic Batch Sizing
# When true, batch sizes are computed at startup from free VRAM.
# EMBEDDING_BATCH_SIZE / CROSS_ENCODER_BATCH_SIZE become ceilings (never exceeded).
# When false, config values are used as-is.
AUTO_BATCH_SIZE = get_env_bool("AUTO_BATCH_SIZE", True)

# Micro-batch accumulation timeout (seconds).
# Worker waits up to this long to fill a batch before flushing.
# Lower = less latency for sparse event streams. Higher = fuller batches.
WORKER_BATCH_TIMEOUT = get_env_float("WORKER_BATCH_TIMEOUT", 1.0)

# Logging
LOG_LEVEL = get_env("LOG_LEVEL", "INFO")

# Orchestrator Configuration
ORCHESTRATOR_VENUES = get_env("ORCHESTRATOR_VENUES", "kalshi,polymarket")
ORCHESTRATOR_INGESTION_QUEUE_SIZE = get_env_int("ORCHESTRATOR_INGESTION_QUEUE_SIZE", 1000)
ORCHESTRATOR_NUM_WORKERS = get_env_int("ORCHESTRATOR_NUM_WORKERS", 1)
ORCHESTRATOR_MODEL_ID = get_env("ORCHESTRATOR_MODEL_ID", "rule-based-v1")
ORCHESTRATOR_PROMPT_VERSION = get_env("ORCHESTRATOR_PROMPT_VERSION", "v1.0")
ORCHESTRATOR_DEDUP_TTL = get_env_int("ORCHESTRATOR_DEDUP_TTL", 3600)

# Bootstrap Configuration
# On startup, fetch all currently-active markets via REST before switching
# to WebSocket streaming.  Server-side filters (Kalshi: status=open,
# Polymarket: closed=false + active=true) ensure every fetched market is
# actively trading — no wasted requests.
BOOTSTRAP_ENABLED = get_env_bool("BOOTSTRAP_ENABLED", True)
# Cap per venue (0 = unlimited).  Useful for dev/testing to avoid
# bootstrapping thousands of markets (~25+ hours on an 8 GB GPU).
BOOTSTRAP_MAX_MARKETS_PER_VENUE = get_env_int("BOOTSTRAP_MAX_MARKETS_PER_VENUE", 0)
# Hard timeout for the REST fetch phase per venue (seconds).
BOOTSTRAP_FETCH_TIMEOUT = get_env_float("BOOTSTRAP_FETCH_TIMEOUT", 120.0)
# Deadline is slightly shorter than timeout so connectors can return
# partial results before the outer wait_for cancels them.
BOOTSTRAP_FETCH_DEADLINE = get_env_float("BOOTSTRAP_FETCH_DEADLINE", 110.0)
# Per-event enqueue timeout (seconds).  await queue.put() blocks if the
# queue is full; if it stays full for this long, drop the event.
BOOTSTRAP_ENQUEUE_TIMEOUT = get_env_float("BOOTSTRAP_ENQUEUE_TIMEOUT", 10.0)

# Polymarket Gamma REST API (no auth required)
POLYMARKET_GAMMA_API_URL = get_env(
    "POLYMARKET_GAMMA_API_URL", "https://gamma-api.polymarket.com"
)

# Kalshi WebSocket auth (required for wss://api.elections.kalshi.com/trade-api/ws/v2)
# API keys: https://docs.kalshi.com/getting_started/api_keys
# WebSocket: https://docs.kalshi.com/websockets/websocket-connection
KALSHI_API_KEY_ID = get_env("KALSHI_API_KEY_ID")
KALSHI_PRIVATE_KEY_PATH = get_env("KALSHI_PRIVATE_KEY_PATH")
KALSHI_PRIVATE_KEY = get_env("KALSHI_PRIVATE_KEY")  # PEM string (alternative to path)
# If your key was created at demo.kalshi.com, set KALSHI_USE_DEMO=true (uses wss://demo-api.kalshi.co/...)
KALSHI_USE_DEMO = get_env_bool("KALSHI_USE_DEMO", False)
# Optional: override WebSocket URL (default: wss://api.elections.kalshi.com/trade-api/ws/v2 or demo)
KALSHI_WS_URL = get_env("KALSHI_WS_URL")


def print_config_summary() -> None:
    """
    Print configuration summary (for verification).
    
    Masks sensitive values like API keys.
    """
    print("\n📋 Semantic Pipeline Configuration:")
    print(f"  LOG_LEVEL: {LOG_LEVEL}")
    print(f"  QDRANT_URL: {QDRANT_URL}")
    print(f"  QDRANT_API_KEY: {'***' + QDRANT_API_KEY[-4:] if QDRANT_API_KEY else 'Not set'}")
    print(f"  QDRANT_COLLECTION: {QDRANT_COLLECTION_NAME}")
    print(f"  QDRANT_VECTOR_SIZE: {QDRANT_VECTOR_SIZE}")
    print(f"  EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    print(f"  EMBEDDING_DEVICE: {EMBEDDING_DEVICE or 'auto-detect'}")
    print(f"  EMBEDDING_BATCH_SIZE: {EMBEDDING_BATCH_SIZE}")
    print(f"  EMBEDDING_DIM: {EMBEDDING_DIM}")
    print(f"  EMBEDDING_QUANTIZATION: {EMBEDDING_QUANTIZATION}")
    print(f"  CACHE_MAX_SIZE: {EMBEDDING_CACHE_MAX_SIZE}")
    print(f"  CROSS_ENCODER_MODEL: {CROSS_ENCODER_MODEL}")
    print(f"  CROSS_ENCODER_DEVICE: {CROSS_ENCODER_DEVICE or 'auto-detect'}")
    print(f"  CROSS_ENCODER_BATCH_SIZE: {CROSS_ENCODER_BATCH_SIZE}")
    print(f"  CROSS_ENCODER_QUANTIZATION: {CROSS_ENCODER_QUANTIZATION}")
    print(f"  CROSS_ENCODER_SCORE_THRESHOLD: {CROSS_ENCODER_SCORE_THRESHOLD}")
    print(f"  RETRIEVAL_SCORE_THRESHOLD: {RETRIEVAL_SCORE_THRESHOLD}")
    print(f"  RETRIEVAL_TOP_K: {RETRIEVAL_TOP_K}")
    print(f"  DATABASE_URL: {'***' + DATABASE_URL[-20:] if DATABASE_URL else 'Not set'}")
    print(f"  WRITER_BATCH_SIZE: {WRITER_BATCH_SIZE}")
    print(f"  WRITER_BATCH_TIMEOUT: {WRITER_BATCH_TIMEOUT}")
    print(f"  WRITER_QUEUE_SIZE: {WRITER_QUEUE_SIZE}")
    print(f"  WRITER_NOTIFY_CHANNEL: {WRITER_NOTIFY_CHANNEL}")
    print(f"  ORCHESTRATOR_VENUES: {ORCHESTRATOR_VENUES}")
    print(f"  ORCHESTRATOR_INGESTION_QUEUE_SIZE: {ORCHESTRATOR_INGESTION_QUEUE_SIZE}")
    print(f"  ORCHESTRATOR_NUM_WORKERS: {ORCHESTRATOR_NUM_WORKERS}")
    print(f"  EMBEDDING_GPU_CONCURRENCY: {EMBEDDING_GPU_CONCURRENCY}")
    print(f"  CROSS_ENCODER_GPU_CONCURRENCY: {CROSS_ENCODER_GPU_CONCURRENCY}")
    print(f"  AUTO_BATCH_SIZE: {AUTO_BATCH_SIZE}")
    print(f"  WORKER_BATCH_TIMEOUT: {WORKER_BATCH_TIMEOUT}")
    print(f"  BOOTSTRAP_ENABLED: {BOOTSTRAP_ENABLED}")
    print(f"  BOOTSTRAP_MAX_MARKETS_PER_VENUE: {BOOTSTRAP_MAX_MARKETS_PER_VENUE or 'unlimited'}")
    print(f"  BOOTSTRAP_FETCH_TIMEOUT: {BOOTSTRAP_FETCH_TIMEOUT}")
    print(f"  BOOTSTRAP_ENQUEUE_TIMEOUT: {BOOTSTRAP_ENQUEUE_TIMEOUT}")
    print(f"  POLYMARKET_GAMMA_API_URL: {POLYMARKET_GAMMA_API_URL}")
    print(f"  ORCHESTRATOR_MODEL_ID: {ORCHESTRATOR_MODEL_ID}")
    print(f"  ORCHESTRATOR_PROMPT_VERSION: {ORCHESTRATOR_PROMPT_VERSION}")
    print(f"  KALSHI_API_KEY_ID: {'***' + KALSHI_API_KEY_ID[-4:] if KALSHI_API_KEY_ID else 'Not set'}")
    print(f"  KALSHI_PRIVATE_KEY_PATH: {KALSHI_PRIVATE_KEY_PATH or 'Not set'}")
    print(f"  KALSHI_USE_DEMO: {KALSHI_USE_DEMO}")
    print(f"  KALSHI_WS_URL: {KALSHI_WS_URL or '(default)'}")
    print()
