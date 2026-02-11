"""
Cross-encoder for pairwise equivalence scoring.

Uses DeBERTa-v3-large-mnli-fever-anli-ling-wanli for precision classification.
Runs on local GPU, scores candidate pairs for semantic equivalence.
"""

import asyncio
import logging
import platform
import re
from typing import List, Tuple, Dict, Optional

import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# Use transformers for NLI models (DeBERTa-v3-large-mnli)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers library not available")


class CrossEncoder:
    """
    Async cross-encoder with model reuse and batch processing.
    
    Uses DeBERTa-v3-large-mnli-fever-anli-ling-wanli for semantic equivalence detection.
    Optimized for batch processing and GPU utilization.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 512,
        use_quantization: bool = False,
        use_compilation: bool = True,
        entailment_threshold: float = 0.7,
        neutral_threshold: float = 0.3,
    ):
        """
        Initialize cross-encoder.
        
        Args:
            model_name: Hugging Face model identifier (default: from config.CROSS_ENCODER_MODEL)
            device: 'cuda', 'cpu', or None (auto-detect)
            batch_size: Batch size for encoding (8-16 optimal for DeBERTa-v3-large)
            max_length: Max token length (512 sufficient for market text)
            use_quantization: Use 8-bit quantization (reduces VRAM by ~50%)
            entailment_threshold: Threshold for high confidence (entailment > this)
            neutral_threshold: Threshold for medium confidence (neutral > this)
        """
        # Use config default if model_name not provided
        self.model_name = model_name if model_name is not None else config.CROSS_ENCODER_MODEL
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_quantization = use_quantization
        self.use_compilation = use_compilation
        self.entailment_threshold = entailment_threshold
        self.neutral_threshold = neutral_threshold
        
        # Device selection (supports CUDA, MPS for M4 Mac, or CPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"  # Metal Performance Shaders for M4 Mac
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Model instance (singleton pattern)
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._pipeline: Optional[pipeline] = None
        self._id2label: Optional[Dict[int, str]] = None
        self._model_lock = asyncio.Lock()
        self._initialized = False
        
        # Quantization config
        self.use_quantization = use_quantization and self.device == "cuda"
        
        logger.info(
            f"CrossEncoder initialized: model={model_name}, "
            f"device={self.device}, batch_size={batch_size}, "
            f"quantization={self.use_quantization}, "
            f"compilation={self.use_compilation}"
        )
    
    async def initialize(self) -> None:
        """
        Lazy model loading (async).
        
        Loads model once and reuses for all subsequent calls.
        Thread-safe initialization.
        """
        if self._initialized:
            return
        
        async with self._model_lock:
            if self._initialized:  # Double-check
                return
            
            try:
                logger.info(f"Loading cross-encoder model: {self.model_name}")
                
                # Load model (CPU-bound, run in executor)
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None,
                    self._load_model_sync
                )
                
                self._initialized = True
                # Verify actual device the model is on
                actual_device = next(self._model.parameters()).device
                device_type = str(actual_device.type)
                device_index = actual_device.index if actual_device.index is not None else 0
                
                # Log detailed device information
                logger.info(
                    f"Model loaded successfully - "
                    f"Configured device: {self.device}, "
                    f"Actual device: {device_type}:{device_index}, "
                    f"Device string: {actual_device}"
                )
                
                # Log MPS availability if on Mac
                if hasattr(torch.backends, "mps"):
                    mps_available = torch.backends.mps.is_available()
                    mps_built = torch.backends.mps.is_built()
                    logger.info(
                        f"MPS status - Available: {mps_available}, Built: {mps_built}"
                    )
                
                # Log CUDA availability
                if torch.cuda.is_available():
                    logger.info(
                        f"CUDA available - Device count: {torch.cuda.device_count()}, "
                        f"Current device: {torch.cuda.current_device()}"
                    )
                
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model: {e}")
                raise
    
    def _load_model_sync(self):
        """Synchronous model loading (called from executor)."""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required for cross-encoder")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        model.to(self.device)
        model.eval()
        
        # Capture label mapping before potential torch.compile() wrapping
        # DeBERTa-v3-large-mnli labels: {0: "entailment", 1: "neutral", 2: "contradiction"}
        self._id2label = model.config.id2label
        
        # OPTIMIZATION: Compile model for faster inference
        # torch.compile() is available in PyTorch 2.0+
        # Compiles model to optimized code (one-time overhead, then faster)
        compilation_status = "DISABLED"
        compilation_reason = None
        
        if self.use_compilation:
            # Windows: torch.compile() requires MSVC cl.exe which is typically not in PATH.
            # torch._inductor fails with "Compiler: cl is not found" during codegen.
            # Disable compilation on Windows to avoid runtime errors.
            if platform.system() == "Windows":
                compilation_status = "DISABLED"
                compilation_reason = "Windows platform - torch.compile() requires MSVC cl.exe which is not in PATH"
                logger.warning(
                    "torch.compile() is disabled on Windows due to missing MSVC compiler (cl.exe). "
                    "Model will run uncompiled. Install Visual Studio Build Tools to enable compilation."
                )
                self.use_compilation = False  # Disable compilation for Windows
            # MPS: torch.compile() has known issues with Metal shader generation
            # Disable compilation on MPS until PyTorch fixes support
            elif self.device == "mps":
                compilation_status = "DISABLED"
                compilation_reason = "MPS device - torch.compile() has known Metal shader compilation errors"
                logger.warning(
                    "torch.compile() is disabled on MPS due to Metal shader compilation errors. "
                    "Model will run uncompiled. Consider using CUDA for optimized performance."
                )
                self.use_compilation = False  # Disable compilation for MPS
            elif hasattr(torch, "compile") and callable(torch.compile):
                try:
                    logger.info("Compiling model with torch.compile() for optimization...")
                    
                    # Device-specific compilation modes
                    if self.device == "cuda":
                        # CUDA: Use "reduce-overhead" for best inference performance
                        compile_mode = "reduce-overhead"
                    else:
                        # CPU: Use "reduce-overhead" for inference
                        compile_mode = "reduce-overhead"
                    
                    model = torch.compile(
                        model,
                        mode=compile_mode,
                        fullgraph=True,  # Compile entire model for maximum optimization
                    )
                    compilation_status = "SUCCESS"
                    logger.info(
                        f"Model compilation: {compilation_status} - "
                        f"mode='{compile_mode}', device={self.device}"
                    )
                except Exception as e:
                    compilation_status = "FAILED"
                    compilation_reason = str(e)
                    logger.warning(
                        f"Model compilation failed (will use uncompiled model): {e}"
                    )
                    # Continue with uncompiled model - don't fail on compilation errors
            else:
                compilation_status = "UNAVAILABLE"
                compilation_reason = "torch.compile() not available (requires PyTorch 2.0+)"
                logger.warning(
                    "torch.compile() not available (requires PyTorch 2.0+). "
                    "Skipping model compilation."
                )
        else:
            compilation_reason = "use_compilation=False"
        
        # Log final compilation status
        if compilation_status != "SUCCESS":
            logger.info(
                f"Model compilation: {compilation_status} - "
                f"device={self.device}, reason={compilation_reason}"
            )
        
        # Create pipeline for easier inference
        # Note: transformers pipeline device parameter:
        # - CUDA: device=0 (device index)
        # - MPS: Don't pass device (model already on MPS via model.to())
        # - CPU: device=-1 or don't pass it
        # Note: task must be the first positional argument, not a keyword argument
        if self.device == "cuda":
            pipeline_device = 0
            nli_pipeline = pipeline(
                "text-classification",  # Positional argument
                model=model,
                tokenizer=tokenizer,
                device=pipeline_device,
                return_all_scores=True,
            )
        elif self.device == "mps":
            # MPS: Don't pass device parameter - model is already on MPS
            # The pipeline will use the device the model is already on
            nli_pipeline = pipeline(
                "text-classification",  # Positional argument
                model=model,
                tokenizer=tokenizer,
                return_all_scores=True,
            )
        else:
            # CPU
            nli_pipeline = pipeline(
                "text-classification",  # Positional argument
                model=model,
                tokenizer=tokenizer,
                device=-1,
                return_all_scores=True,
            )
        
        self._tokenizer = tokenizer
        self._pipeline = nli_pipeline
        
        return model
    
    def _parse_nli_results(self, results: list) -> Dict[str, float]:
        """
        Parse NLI pipeline output into normalized {entailment, neutral, contradiction} dict.

        Shared between single-pair and batch scoring paths.
        """
        scores_dict: Dict[str, float] = {}
        for result in results:
            if not isinstance(result, dict):
                continue
            if "label" not in result or "score" not in result:
                continue

            label = result["label"].lower()
            score = result["score"]

            # DeBERTa-v3-large-mnli labels: ENTAILMENT, NEUTRAL, CONTRADICTION
            if "entail" in label:
                scores_dict["entailment"] = score
            elif "neutral" in label:
                scores_dict["neutral"] = score
            elif "contradict" in label:
                scores_dict["contradiction"] = score

        for key in ("entailment", "neutral", "contradiction"):
            scores_dict.setdefault(key, 0.0)

        total = sum(scores_dict.values())
        if total > 0:
            scores_dict = {k: v / total for k, v in scores_dict.items()}

        return scores_dict

    async def score_equivalence_async(
        self,
        query_text: str,
        candidate_text: str
    ) -> Dict[str, float]:
        """
        Score semantic equivalence between two texts (async).

        Args:
            query_text: Query market canonical text
            candidate_text: Candidate market canonical text

        Returns:
            Dictionary with NLI scores: {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}
        """
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._score_pair_sync(query_text, candidate_text)
        )

        return scores

    def _score_pair_sync(self, text1: str, text2: str) -> Dict[str, float]:
        """Synchronous single-pair scoring (called from executor)."""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required")

        formatted_text = f"{text1} [SEP] {text2}"

        results = self._pipeline(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
        )

        if not isinstance(results, list) or not results:
            raise ValueError(f"Pipeline returned unexpected output: {results}")

        # return_all_scores=True wraps single input in an extra list
        if isinstance(results[0], list):
            results = results[0]

        return self._parse_nli_results(results)

    def _score_batch_sync(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """
        Batch scoring via direct tokenizer + model.forward().

        Bypasses HuggingFace pipeline's Dataset/DataLoader overhead which
        causes severe performance regression on Windows CUDA. Tokenizes and
        runs forward passes in micro-batches of self.batch_size for GPU
        memory safety. O(ceil(N/batch_size)) GPU forward passes.

        Args:
            pairs: List of (text1, text2) tuples to score.

        Returns:
            List of NLI score dicts, one per pair.
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required")

        all_scores: List[Dict[str, float]] = []

        # Process in micro-batches to stay within GPU memory
        for start in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[start:start + self.batch_size]
            formatted_texts = [f"{t1} [SEP] {t2}" for t1, t2 in batch_pairs]

            # Direct tokenization — no Dataset/DataLoader overhead
            inputs = self._tokenizer(
                formatted_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # Single forward pass for entire micro-batch
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            # Map label indices → NLI score dicts via stored id2label
            for i in range(probs.shape[0]):
                results_list = [
                    {"label": self._id2label[j], "score": probs[i, j].item()}
                    for j in range(probs.shape[1])
                ]
                all_scores.append(self._parse_nli_results(results_list))

        return all_scores

    async def score_batch_async(
        self,
        pairs: List[Tuple[str, str]],
        max_concurrent: Optional[int] = None
    ) -> List[Dict[str, float]]:
        """
        Score semantic equivalence for a batch of text pairs (async).

        Uses true batch inference — single executor dispatch, HuggingFace
        pipeline handles GPU micro-batching internally. O(ceil(N/batch_size))
        GPU forward passes instead of O(N) separate executor calls.

        Args:
            pairs: List of (query_text, candidate_text) tuples
            max_concurrent: Kept for API compatibility, unused

        Returns:
            List of dictionaries with NLI scores for each pair.
        """
        if not self._initialized:
            await self.initialize()

        if not pairs:
            return []

        default_scores = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}

        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None,
                lambda: self._score_batch_sync(pairs)
            )
            return results
        except Exception as e:
            logger.error(f"Batch scoring failed, returning defaults: {e}", exc_info=True)
            return [dict(default_scores) for _ in pairs]
    
    def map_nli_to_confidence(
        self,
        nli_scores: Dict[str, float]
    ) -> Tuple[float, str]:
        """
        Map NLI scores to confidence score and match type.
        
        Args:
            nli_scores: Dictionary with entailment/neutral/contradiction scores
            
        Returns:
            Tuple of (confidence_score: float, match_type: str)
            match_type: "full_match", "partial_match", or "no_match"
        """
        entailment = nli_scores.get("entailment", 0.0)
        neutral = nli_scores.get("neutral", 0.0)
        contradiction = nli_scores.get("contradiction", 0.0)
        
        # High confidence: Strong entailment (inclusive threshold)
        if entailment >= self.entailment_threshold:
            confidence = entailment
            match_type = "full_match"
        # Medium confidence: Neutral or moderate entailment
        elif neutral > self.neutral_threshold or (0.4 <= entailment <= self.entailment_threshold):
            # Weighted combination favoring entailment
            confidence = 0.6 * entailment + 0.4 * neutral
            match_type = "partial_match"
        # Low confidence: Strong contradiction or weak entailment
        else:
            if contradiction > 0.5:
                confidence = 1.0 - contradiction
            else:
                confidence = entailment
            match_type = "no_match"
        
        # Ensure confidence is in [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence, match_type
    
    def extract_primary_event(self, canonical_text: str) -> str:
        """
        Extract primary event from canonical text.
        
        Args:
            canonical_text: Canonical markdown text
            
        Returns:
            Primary event text (Market Statement section)
        """
        lines = canonical_text.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('Market Statement:'):
                # Get next non-empty line(s) until next section
                event_lines = []
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    if next_line.startswith(('Resolution Criteria:', 'Clarifications:', 'End Date:', 'Outcomes:')):
                        break
                    event_lines.append(next_line)
                if event_lines:
                    return ' '.join(event_lines)
        
        # Fallback to first line
        first_line = canonical_text.split('\n')[0].strip()
        return first_line if first_line else canonical_text[:200]
    
    def extract_secondary_clauses(self, canonical_text: str) -> List[str]:
        """
        Extract secondary clauses from Resolution Criteria section.
        
        Args:
            canonical_text: Canonical markdown text
            
        Returns:
            List of clause strings
        """
        lines = canonical_text.split('\n')
        in_resolution = False
        clause_lines = []
        
        for line in lines:
            if line.startswith('Resolution Criteria:'):
                in_resolution = True
                continue
            elif in_resolution:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith(('Clarifications:', 'End Date:', 'Outcomes:')):
                    break
                clause_lines.append(stripped)
        
        if not clause_lines:
            return []
        
        # Join and split by sentences/clauses
        full_text = ' '.join(clause_lines)
        # Split by periods, semicolons, or numbered lists
        clauses = re.split(r'[.;]\s+|(?=\d+\.\s)', full_text)
        
        # Filter and clean clauses
        cleaned_clauses = []
        for clause in clauses:
            clause = clause.strip()
            # Only include substantial clauses (more than 10 chars)
            if clause and len(clause) > 10:
                cleaned_clauses.append(clause)
        
        return cleaned_clauses
    
    async def score_secondary_clauses_async(
        self,
        query_clauses: List[str],
        candidate_clauses: List[str]
    ) -> float:
        """
        Score equivalence of secondary clauses between two markets.

        Collects all n×m clause pair combinations and scores in a single
        batch GPU call, then computes best-match per query clause.

        Args:
            query_clauses: List of clauses from query market
            candidate_clauses: List of clauses from candidate market

        Returns:
            Confidence score (0-1) for clause equivalence
        """
        if not query_clauses or not candidate_clauses:
            return 0.5

        # O(n×m) pairs scored in one batch call instead of n×m sequential calls
        all_pairs = [
            (qc, cc)
            for qc in query_clauses
            for cc in candidate_clauses
        ]

        all_nli_scores = await self.score_batch_async(all_pairs)

        # Best-match per query clause from the flat results
        n_candidates = len(candidate_clauses)
        clause_scores = []
        for q_idx in range(len(query_clauses)):
            best_match = 0.0
            for c_idx in range(n_candidates):
                flat_idx = q_idx * n_candidates + c_idx
                confidence, _ = self.map_nli_to_confidence(all_nli_scores[flat_idx])
                best_match = max(best_match, confidence)
            clause_scores.append(best_match)

        return sum(clause_scores) / len(clause_scores) if clause_scores else 0.5
