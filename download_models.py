#!/usr/bin/env python3
"""
Standalone script to download models used by the semantic pipeline.

Usage:
    python download_models.py [--model qwen|cross-encoder|both] [--force]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_qwen(model_name: str = "Qwen/Qwen3-Embedding-4B", force: bool = False):
    """Download Qwen embedding model."""
    logger.info(f"Downloading Qwen embedding model: {model_name}")
    logger.info("This may take several minutes depending on your connection...")
    
    try:
        # SentenceTransformer will download and cache the model
        model = SentenceTransformer(model_name)
        logger.info(f"✓ Successfully downloaded/cached: {model_name}")
        logger.info(f"  Model device: {model.device}")
        logger.info(f"  Max sequence length: {model.max_seq_length}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download {model_name}: {e}")
        return False


def download_cross_encoder(model_name: str = "cross-encoder/nli-deberta-v3-large", force: bool = False):
    """Download cross-encoder NLI model."""
    logger.info(f"Downloading cross-encoder model: {model_name}")
    logger.info("This may take several minutes depending on your connection...")
    
    try:
        # Use transformers to download tokenizer and model
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("✓ Tokenizer downloaded")
        
        logger.info("Downloading model (this is the large file ~1.5GB)...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logger.info(f"✓ Successfully downloaded/cached: {model_name}")
        logger.info(f"  Model device: {model.device if hasattr(model, 'device') else 'CPU'}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download models for semantic pipeline")
    parser.add_argument(
        "--model",
        choices=["qwen", "cross-encoder", "both"],
        default="both",
        help="Which model(s) to download (default: both)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached (not implemented, always checks cache first)"
    )
    
    args = parser.parse_args()
    
    success = True
    
    if args.model in ["qwen", "both"]:
        success &= download_qwen(force=args.force)
        print()  # Blank line for readability
    
    if args.model in ["cross-encoder", "both"]:
        success &= download_cross_encoder(force=args.force)
    
    if success:
        logger.info("✓ All requested models downloaded successfully!")
        sys.exit(0)
    else:
        logger.error("✗ Some models failed to download")
        sys.exit(1)


if __name__ == "__main__":
    main()

