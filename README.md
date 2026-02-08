# Semantic Pipeline

Market discovery and semantic matching pipeline for PolyEdge. This is a **separate Python process** that runs independently from the trading engine.

## Overview

The Semantic Pipeline discovers markets from multiple venues (Kalshi, Polymarket, Opinion, Gemini, etc.) and identifies equivalent markets across venues using a two-stage matching approach:

1. **Retrieval Stage**: Embedding model (Qwen 3 0.6B) + Qdrant vector DB for high-recall candidate retrieval
2. **Verification Stage**: Cross-encoder (DeBERTa-v3-large-mnli) + LLM verification for precision classification

## Architecture

```
WebSocket Events → Venue Connectors → Batch Accumulator → Canonical Text + Hashing
    ↓
Embedding → Qdrant → Candidate Retrieval → Cross-Encoder → LLM Verification
    ↓
PostgreSQL (markets, contract_specs, verified_pairs)
```

## Directory Structure

```
semantic_pipeline/
├── discovery/          # Venue connectors (WebSocket-based)
│   ├── kalshi_poller.py
│   ├── polymarket_poller.py
│   └── dedup.py
├── canonicalization/  # Text normalization and hashing
│   ├── text_builder.py
│   ├── hasher.py
│   └── contract_spec.py
├── embedding/          # Embedding generation and vector DB
│   ├── encoder.py
│   ├── cache.py
│   └── index.py
├── matching/          # Two-stage matching pipeline
│   ├── retriever.py
│   ├── cross_encoder.py
│   └── llm_verifier.py
├── extraction/         # LLM-based ContractSpec extraction
│   └── spec_extractor.py
├── persistence/        # Database writes
│   └── writer.py
├── config.py          # Configuration management
├── orchestrator.py    # Main pipeline coordinator
└── requirements.txt   # Python dependencies
```

## Key Design Decisions

- **WebSocket-only discovery**: Real-time market events via WebSocket only (no polling/REST API)
- **Local LLM instances**: No API dependencies, self-hosted models
- **Qdrant vector DB**: Dedicated vector database for scalability
- **Adaptive micro-batching**: 20 markets or 2 seconds (whichever comes first)
- **Two hash strategy**: Identity hash (dedup) + Content hash (change detection)
- **Cross + intra venue**: Supports matching within and across venues

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL (with schema from migrations)
- Qdrant (vector database)
- Redis (optional, for embedding cache)
- GPU (recommended for local LLM and cross-encoder)

### Installation

```bash
cd semantic_pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost/polyedge

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=market_embeddings

# Redis (optional)
REDIS_URL=redis://localhost:6379

# LLM
LLM_MODEL_PATH=/path/to/local/model
LLM_DEVICE=cuda  # or cpu

# Embedding Model
EMBEDDING_MODEL=qwen3-0.6b
EMBEDDING_DEVICE=cuda

# Cross-Encoder
CROSS_ENCODER_MODEL=deberta-v3-large-mnli-fever-anli-ling-wanli
CROSS_ENCODER_DEVICE=cuda
```

## Running

```bash
python -m semantic_pipeline.orchestrator
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .

# Lint
ruff check .
```

## Notes

- This is a **separate process** from the trading engine
- Trading engine NEVER calls LLMs (performance-critical path)
- All LLM operations happen here, results written to database
- Trading engine reads `verified_pairs` table for pair mappings

