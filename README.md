# Cross-Venue Semantic Matcher

A GPU-accelerated semantic matching pipeline for identifying equivalent contracts across prediction market venues (Polymarket, Kalshi). Two-stage retrieve-then-rerank (Qwen3-Embedding-4B + DeBERTa-v3 NLI) over Qdrant, with venue-agnostic discovery, micro-batch GPU scheduling, and esports market latency analysis.

Archived as [PolyEdge](https://polyedge.trade) infrastructure exploration. Built to validate cross-venue arbitrage feasibility — the matching pipeline works, but spread economics don't support profitable execution. Published as a reference iterative implementation.

## Architecture

Venue-agnostic 7-stage pipeline that ingests prediction market contracts from any supported venue and identifies semantically equivalent markets across platforms. Adding a new venue requires only a connector and a text builder — the matching, verification, and persistence layers are fully venue-independent.

```
Venue Connectors (Kalshi WS, Polymarket REST/WS)
    |
    v
[1] Canonicalization ── venue-specific markdown normalization (CPU)
    |
    v
[2] Batch Embedding ── Qwen3-Embedding-4B, 2048-dim, instruction-guided (GPU)
    |
    v
[3] Retrieval ── Qdrant top-K similarity search, cross-venue only (CPU + Vector DB)
    |
    v
[4] Reranking ── DeBERTa-v3 NLI cross-encoder, bidirectional scoring (GPU)
    |
    v
[5] ContractSpec Extraction ── entity/date/threshold parsing (CPU, optional LLM fallback)
    |
    v
[6] Pair Verification ── 5-factor weighted fusion (CPU)
    |
    v
[7] Persistence ── PostgreSQL micro-batch writer + LISTEN/NOTIFY (async)
```

## Key Technical Decisions

**Retrieve-then-rerank.** Recall-heavy embedding retrieval (threshold 0.5, 20 candidates) followed by precision cross-encoder reranking. Bidirectional NLI scoring catches superset/subset false positives.

**Micro-batch GPU scheduling.** Workers accumulate events (1s timeout) into batches before GPU forward pass. GPU concurrency semaphore (default: 1) serializes passes to prevent VRAM fragmentation. CUDA cache defragmentation every 10 batches.

**Dynamic batch sizing.** Post-model-load VRAM probing computes optimal batch size. INT8 quantization for cross-encoder (0.5 GB vs 1.7 GB FP32). Runs dual 4B models on 8 GB GPU without OOM.

**Venue abstraction.** Pluggable discovery strategies (normal, esports, hybrid). Per-venue text builders normalize heterogeneous API schemas to identical markdown. TTL-based deduplication across WebSocket reconnects.

**5-factor verification fusion.** Cross-encoder NLI (50%) + threshold tolerance (20%) + entity fuzzy match (15%) + date tolerance (10%) + data source compatibility (5%). Returns EQUIVALENT / PARTIAL / REJECTED.

## Directory Structure

```
semantic_pipeline/
├── orchestrator/
│   ├── core.py              # 7-stage pipeline coordinator
│   ├── metrics.py           # Thread-safe per-stage metrics
│   └── discovery/           # Discovery strategy pattern
├── discovery/
│   ├── base_connector.py    # Abstract venue connector (reconnection, heartbeats)
│   ├── kalshi_poller.py     # Kalshi REST bootstrap + WebSocket streaming
│   ├── polymarket_poller.py # Polymarket Gamma API + WebSocket
│   ├── dedup.py             # Market deduplication (TTL-based)
│   └── strategies/          # Esports mode, hybrid mode
├── canonicalization/
│   ├── text_builder.py      # Kalshi/Polymarket-specific markdown builders
│   ├── hasher.py            # Identity hash (dedup) + content hash (change detection)
│   └── contract_spec.py     # ContractSpec dataclass
├── embedding/
│   ├── encoder.py           # Qwen3-Embedding-4B with instruction-guided encoding
│   ├── index.py             # Qdrant async client
│   └── cache/               # In-memory LRU embedding cache
├── matching/
│   ├── retriever.py         # Qdrant top-K cross-venue retrieval
│   ├── cross_encoder.py     # DeBERTa-v3-large-mnli bidirectional scorer
│   ├── reranker.py          # Candidate reranking with asymmetry detection
│   └── pair_verifier.py     # Multi-factor weighted verification
├── extraction/
│   └── spec_extractor.py    # Entity/date/threshold extraction (regex + optional LLM)
├── persistence/
│   └── writer.py            # PostgreSQL batch writer + LISTEN/NOTIFY
├── monitoring/
│   ├── compliance/          # Audit logger, circuit breaker
│   ├── core/                # Latency engine, orderbook manager
│   └── feeds/               # WebSocket feeds (Kalshi, Polymarket, Riot API)
├── spread_scanner.py        # Spread arb exploration script
├── spread_scanner_ws.py     # WebSocket-based spread scanner
├── spread_scanner_ws_esports.py  # Esports-specific spread scanner
├── config.py                # 70+ tuneable parameters from env vars
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL
- Qdrant (cloud or self-hosted)
- GPU recommended (MPS/CUDA for local model inference)

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```env
# Qdrant (vector database for similarity search)
QDRANT_URL=https://your-qdrant-instance.cloud.qdrant.io/
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=market_embeddings
QDRANT_VECTOR_SIZE=2048

# Embedding Model
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
EMBEDDING_DEVICE=mps          # mps (Apple Silicon), cuda, or cpu
EMBEDDING_BATCH_SIZE=8         # overridden by AUTO_BATCH_SIZE if enabled
EMBEDDING_DIM=2048
EMBEDDING_QUANTIZATION=true    # INT8 quantization (~50% VRAM reduction)
EMBEDDING_INSTRUCTION=Given a prediction market contract, represent its core event, resolution conditions, and timeframe for matching equivalent contracts across different prediction market platforms

# Cross-Encoder (NLI-based reranking)
CROSS_ENCODER_QUANTIZATION=true

# GPU Scheduling
AUTO_BATCH_SIZE=true           # dynamic batch sizing from free VRAM
EMBEDDING_GPU_CONCURRENCY=1    # serialized GPU access prevents fragmentation
CROSS_ENCODER_GPU_CONCURRENCY=1
ORCHESTRATOR_NUM_WORKERS=2
WORKER_BATCH_TIMEOUT=1.0       # micro-batch accumulation timeout (seconds)

# Cache
EMBEDDING_CACHE_MAX_SIZE=10000
EXTRACTION_CACHE_MAX_SIZE=1000
VERIFICATION_CACHE_MAX_SIZE=10000

# PostgreSQL
DATABASE_URL=postgresql://user:pass@host:port/database

# Kalshi API
KALSHI_API_KEY_ID=your_kalshi_key_id
KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi.pem

# Bootstrap (initial market fetch)
BOOTSTRAP_ENABLED=true
BOOTSTRAP_MAX_MARKETS_PER_VENUE=0  # 0 = unlimited, set to 50 for dev/testing
BOOTSTRAP_FETCH_TIMEOUT=600
POLYMARKET_GAMMA_API_URL=https://gamma-api.polymarket.com

# Discovery mode
DISCOVERY_MODE=esports         # normal | esports | hybrid
ESPORTS_POLYMARKET_CATEGORIES=esports,gaming,video-games
ESPORTS_KALSHI_KEYWORDS=LOL,LEAGUE,DOTA,VALORANT,CSGO

# Riot API (optional, for esports game event latency measurement)
RIOT_API_KEY=your_riot_api_key
GAME_EVENT_POLL_INTERVAL_MS=1000

# LLM fallback (optional)
EXTRACTION_USE_LLM_FALLBACK=false
```

## Running

```bash
# Full pipeline
python -m semantic_pipeline.orchestrator

# Esports spread scanner (exploration)
python spread_scanner_ws_esports.py

# Esports arb monitor (single match)
python monitor_dk_vs_t1.py
```

## Exploration Scripts

These scripts document the research process — investigating whether cross-venue esports prediction markets have exploitable latency or spread divergence.

| Script | Purpose |
|--------|---------|
| `spread_scanner.py` | REST-based cross-venue spread analysis |
| `spread_scanner_ws.py` | WebSocket real-time spread monitoring |
| `spread_scanner_ws_esports.py` | Esports-focused spread scanner with Riot API integration |
| `monitor_dk_vs_t1.py` | Live arb monitor for DK vs T1 LoL match with latency measurement |
| `verify_market_equivalence.py` | Manual market pair verification tool |
| `check_esports_pairs.py` | Inspect matched esports pairs in DB |
| `check_db_for_polymarket_market.py` | Debug tool for Polymarket market lookup |

## Status

**Archived.** This was an iterative exploration into cross-venue arbitrage feasibility and its integration into the PolyEdge platform. The semantic matching pipeline works — cross-venue pairs are correctly identified and verified with high precision. The arb thesis was invalidated: prediction market spreads are too thin and venue-to-venue latency too high for profitable execution after fees. The venue discovery and market resolution infrastructure evolved into [PolyEdge](https://polyedge.trade)'s production data API.
