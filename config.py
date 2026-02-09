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
    print()
