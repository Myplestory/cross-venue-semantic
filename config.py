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
    print(f"✅ Loaded .env from: {_env_path}")
else:
    print(f"⚠️  .env file not found at: {_env_path}")
    print("   Using environment variables and defaults")


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
    "Represent the market contract for similarity search."
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

# ContractSpec Extraction Configuration
EXTRACTION_USE_LLM_FALLBACK = get_env_bool("EXTRACTION_USE_LLM_FALLBACK", True)
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


def print_config_summary() -> None:
    """
    Print configuration summary (for verification).
    
    Masks sensitive values like API keys.
    """
    print("\n📋 Semantic Pipeline Configuration:")
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
    print()
