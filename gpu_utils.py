"""
GPU hardware detection and dynamic batch size auto-tuning.

Probes available VRAM after model weights are loaded, then calculates
the largest safe batch size that fits within the remaining headroom.

Industry patterns (Triton / Ray Serve / BentoML):
- Profile-based: one trial forward pass measures actual per-sample cost
  (no hardcoded constants that drift as models change).
- Safety margin: only 75% of free VRAM is budgeted (leaves headroom for
  PyTorch allocator fragmentation + OS overhead).
- Bounded: batch size is clamped between 1 and the env-configured max
  (EMBEDDING_BATCH_SIZE / CROSS_ENCODER_BATCH_SIZE).
- Fallback: any profiling error -> returns 1 (safe, never OOM).

Called once in ``orchestrator.initialize()`` AFTER both models are loaded.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  VRAM query helpers
# ═══════════════════════════════════════════════════════════════════════


def get_free_vram_mb(device_index: int = 0) -> Optional[float]:
    """
    Query free VRAM on a CUDA device (MB).

    Calls ``torch.cuda.empty_cache()`` first to release PyTorch's
    internal caching-allocator blocks back to the CUDA driver.
    Without this, ``mem_get_info()`` reports near-zero free memory
    because the caching allocator greedily reserves all VRAM even
    though most of it is unused.

    Args:
        device_index: CUDA device ordinal.

    Returns:
        Free VRAM in MB, or None if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return None
    try:
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info(device_index)
        return free_bytes / (1024 ** 2)
    except Exception as exc:
        logger.warning("Failed to query VRAM: %s", exc)
        return None


def get_gpu_summary(device_index: int = 0) -> dict:
    """
    Return a dict of GPU properties for structured logging.

    Keys: name, total_mb, allocated_mb, free_mb, compute_capability.
    """
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    props = torch.cuda.get_device_properties(device_index)
    total_mb = props.total_memory / (1024 ** 2)
    alloc_mb = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
    free = get_free_vram_mb(device_index)
    return {
        "name": props.name,
        "total_mb": round(total_mb, 1),
        "allocated_mb": round(alloc_mb, 1),
        "free_mb": round(free if free is not None else (total_mb - alloc_mb), 1),
        "compute_capability": f"sm_{props.major}{props.minor}",
    }


# ═══════════════════════════════════════════════════════════════════════
#  Per-sample profiling
# ═══════════════════════════════════════════════════════════════════════

_SAMPLE_TEXT = "Will Bitcoin exceed $100,000 by December 31, 2026?"


def profile_embedding_per_sample_mb(
    encoder,
    sample_text: str = _SAMPLE_TEXT,
) -> Optional[float]:
    """
    Run one forward pass and measure peak activation memory (MB).

    Measures the delta between ``memory_allocated`` before and
    ``max_memory_allocated`` after a single-sample encode.  Captures
    actual activation footprint (respects quantization, dtype, arch).

    Args:
        encoder: Initialized ``EmbeddingEncoder`` with ``._model``.
        sample_text: Representative input.

    Returns:
        Per-sample activation memory in MB, or None on failure.
    """
    if not torch.cuda.is_available() or str(encoder.device) != "cuda":
        return None
    try:
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()

        text = sample_text
        if encoder.instruction:
            text = f"{encoder.instruction}\n{text}"
        encoder._model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        peak = torch.cuda.max_memory_allocated()
        per_sample_mb = (peak - before) / (1024 ** 2)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.debug(
            "Profiled embedding per-sample: %.1f MB", per_sample_mb
        )
        return per_sample_mb
    except Exception as exc:
        logger.warning("Embedding profiling failed: %s", exc)
        return None


def profile_cross_encoder_per_sample_mb(
    cross_encoder,
    sample_pair: tuple = (
        _SAMPLE_TEXT,
        "Bitcoin to hit $100k by end of 2026",
    ),
) -> Optional[float]:
    """
    Run one cross-encoder forward pass and measure peak activation memory.

    Args:
        cross_encoder: Initialized ``CrossEncoder`` with ``._pipeline``.
        sample_pair: Representative (query, candidate) text pair.

    Returns:
        Per-sample activation memory in MB, or None on failure.
    """
    if not torch.cuda.is_available() or str(cross_encoder.device) != "cuda":
        return None
    try:
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()

        formatted = f"{sample_pair[0]} [SEP] {sample_pair[1]}"
        cross_encoder._pipeline(
            formatted,
            truncation=True,
            max_length=cross_encoder.max_length,
        )

        peak = torch.cuda.max_memory_allocated()
        per_sample_mb = (peak - before) / (1024 ** 2)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.debug(
            "Profiled cross-encoder per-sample: %.1f MB", per_sample_mb
        )
        return per_sample_mb
    except Exception as exc:
        logger.warning("Cross-encoder profiling failed: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Batch size calculation
# ═══════════════════════════════════════════════════════════════════════

# Use 60% of free VRAM for batch activations.  The remaining 40% is
# reserved for cross-model overlap (embedding + cross-encoder share VRAM
# because their GPU semaphores are separate) and for CUDA allocator
# fragmentation that accumulates over long runs (40 000+ events).
_SAFETY_FACTOR = 0.60

# Fallback per-sample estimates if profiling fails (conservative, MB).
# Measured on Qwen3-4B INT8, DeBERTa INT8, max_length=512.
_FALLBACK_EMBEDDING_MB = 250.0
_FALLBACK_CROSS_ENCODER_MB = 80.0


def compute_batch_size(
    free_vram_mb: float,
    per_sample_mb: float,
    max_batch: int,
    min_batch: int = 1,
    safety_factor: float = _SAFETY_FACTOR,
    label: str = "model",
) -> int:
    """
    Compute the largest safe batch size from available VRAM.

    Formula::

        batch = floor(free_vram * safety_factor / per_sample)
        batch = clamp(batch, min_batch, max_batch)

    Args:
        free_vram_mb: Available VRAM after model weights (MB).
        per_sample_mb: Activation memory per input sample (MB).
        max_batch: Upper bound (from config -- never exceed).
        min_batch: Lower bound (always at least 1).
        safety_factor: Fraction of free VRAM to use.
        label: For logging.

    Returns:
        Safe batch size, clamped to [min_batch, max_batch].
    """
    usable = free_vram_mb * safety_factor
    raw = int(usable / per_sample_mb) if per_sample_mb > 0 else min_batch
    batch = max(min_batch, min(raw, max_batch))

    logger.info(
        "[%s] Auto batch: free=%.0f MB x %.0f%% = %.0f MB usable, "
        "per_sample=%.0f MB -> batch=%d (bounds [%d, %d])",
        label,
        free_vram_mb,
        safety_factor * 100,
        usable,
        per_sample_mb,
        batch,
        min_batch,
        max_batch,
    )
    return batch


# ═══════════════════════════════════════════════════════════════════════
#  Top-level auto-tune entry point
# ═══════════════════════════════════════════════════════════════════════


def auto_tune_batch_sizes(
    embedding_encoder,
    cross_encoder,
    max_embedding_batch: int,
    max_cross_encoder_batch: int,
) -> dict:
    """
    Profile both models and compute optimal batch sizes.

    Called once in ``orchestrator.initialize()`` after both models are
    loaded.  Mutates ``encoder.batch_size`` and
    ``cross_encoder.batch_size`` in place.

    The GPU semaphore (concurrency=1) ensures embedding and cross-encoder
    never run simultaneously, so each model can budget the full free VRAM
    for activations.

    Args:
        embedding_encoder: Initialized EmbeddingEncoder.
        cross_encoder: Initialized CrossEncoder.
        max_embedding_batch: Config ceiling (EMBEDDING_BATCH_SIZE).
        max_cross_encoder_batch: Config ceiling (CROSS_ENCODER_BATCH_SIZE).

    Returns:
        Dict with computed values for logging/diagnostics.
    """
    result: dict = {"gpu": get_gpu_summary()}

    if not torch.cuda.is_available():
        logger.info("No CUDA GPU -- batch sizes unchanged (CPU mode)")
        result["embedding_batch_size"] = embedding_encoder.batch_size
        result["cross_encoder_batch_size"] = cross_encoder.batch_size
        return result

    free_vram = get_free_vram_mb() or 0.0

    # ── Embedding ────────────────────────────────────────────────────
    emb_mb = profile_embedding_per_sample_mb(embedding_encoder)
    if emb_mb is None or emb_mb <= 0:
        emb_mb = _FALLBACK_EMBEDDING_MB
        logger.info(
            "Using fallback embedding per-sample: %.0f MB", emb_mb
        )

    emb_batch = compute_batch_size(
        free_vram,
        emb_mb,
        max_embedding_batch,
        label="embedding",
    )
    embedding_encoder.batch_size = emb_batch
    result["embedding_batch_size"] = emb_batch
    result["embedding_per_sample_mb"] = round(emb_mb, 1)

    # ── Cross-encoder ────────────────────────────────────────────────
    ce_mb = profile_cross_encoder_per_sample_mb(cross_encoder)
    if ce_mb is None or ce_mb <= 0:
        ce_mb = _FALLBACK_CROSS_ENCODER_MB
        logger.info(
            "Using fallback cross-encoder per-sample: %.0f MB", ce_mb
        )

    ce_batch = compute_batch_size(
        free_vram,
        ce_mb,
        max_cross_encoder_batch,
        label="cross_encoder",
    )
    cross_encoder.batch_size = ce_batch
    result["cross_encoder_batch_size"] = ce_batch
    result["cross_encoder_per_sample_mb"] = round(ce_mb, 1)

    logger.info(
        "Auto-tune complete: embedding_batch=%d, cross_encoder_batch=%d",
        emb_batch,
        ce_batch,
    )
    return result

