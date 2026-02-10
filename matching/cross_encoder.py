"""
Cross-encoder for pairwise equivalence scoring.

Uses DeBERTa-v3-large-mnli-fever-anli-ling-wanli for precision classification.
Runs on local GPU, scores candidate pairs for semantic equivalence.
"""

import asyncio
import logging
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
        self._model_lock = asyncio.Lock()
        self._initialized = False
        
        # Quantization config
        self.use_quantization = use_quantization and self.device == "cuda"
        
        logger.info(
            f"CrossEncoder initialized: model={model_name}, "
            f"device={self.device}, batch_size={batch_size}, "
            f"quantization={self.use_quantization}"
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
        
        # Run scoring in executor (CPU-bound GPU operations)
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._score_pair_sync(query_text, candidate_text)
        )
        
        return scores
    
    def _score_pair_sync(self, text1: str, text2: str) -> Dict[str, float]:
        """Synchronous pair scoring (called from executor)."""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required")
        
        # Format as NLI task: premise [SEP] hypothesis
        # For DeBERTa-v3-large-mnli, format is typically: text1 [SEP] text2
        formatted_text = f"{text1} [SEP] {text2}"
        
        # Use pipeline for inference
        results = self._pipeline(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
        )
        
        # Validate pipeline output format
        if not isinstance(results, list):
            error_msg = (
                f"Pipeline returned unexpected type: {type(results)}. "
                f"Expected list of dicts, got: {results}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not results:
            error_msg = "Pipeline returned empty list"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate first result is a dict
        if not isinstance(results[0], dict):
            error_msg = (
                f"Pipeline results[0] is not a dict: {type(results[0])}. "
                f"Results: {results}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Extract scores from pipeline output
        scores_dict = {}
        for result in results:
            if not isinstance(result, dict):
                logger.warning(f"Skipping invalid result (not a dict): {result}")
                continue
            
            if "label" not in result or "score" not in result:
                logger.warning(f"Skipping result missing 'label' or 'score': {result}")
                continue
            
            label = result["label"].lower()
            score = result["score"]
            
            # Map labels to standard NLI format
            # DeBERTa-v3-large-mnli typically uses: ENTAILMENT, NEUTRAL, CONTRADICTION
            if "entail" in label:
                scores_dict["entailment"] = score
            elif "neutral" in label:
                scores_dict["neutral"] = score
            elif "contradict" in label:
                scores_dict["contradiction"] = score
        
        # Ensure all three labels exist (normalize if missing)
        if "entailment" not in scores_dict:
            scores_dict["entailment"] = 0.0
        if "neutral" not in scores_dict:
            scores_dict["neutral"] = 0.0
        if "contradiction" not in scores_dict:
            scores_dict["contradiction"] = 0.0
        
        # Normalize to ensure they sum to 1.0 (in case of rounding issues)
        total = sum(scores_dict.values())
        if total > 0:
            scores_dict = {k: v / total for k, v in scores_dict.items()}
        
        return scores_dict
    
    async def score_batch_async(
        self,
        pairs: List[Tuple[str, str]],
        max_concurrent: Optional[int] = None
    ) -> List[Dict[str, float]]:
        """
        Score semantic equivalence for a batch of text pairs (async).
        
        Processes pairs individually in parallel using asyncio.gather().
        This ensures return_all_scores=True works correctly for each pair.
        Uses concurrency control to limit resource usage.
        
        Args:
            pairs: List of (query_text, candidate_text) tuples
            max_concurrent: Maximum number of concurrent tasks (default: batch_size)
            
        Returns:
            List of dictionaries with NLI scores for each pair.
        """
        if not self._initialized:
            await self.initialize()
        
        if not pairs:
            return []
        
        # Use batch_size as default concurrency limit
        if max_concurrent is None:
            max_concurrent = self.batch_size
        
        # Process pairs with concurrency limit using semaphore
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def score_with_limit(text1: str, text2: str) -> Dict[str, float]:
            """Score a single pair with concurrency limit."""
            async with semaphore:
                return await self.score_equivalence_async(text1, text2)
        
        # Create tasks for all pairs
        tasks = [
            score_with_limit(text1, text2)
            for text1, text2 in pairs
        ]
        
        # Execute all tasks concurrently
        # asyncio.gather() maintains order of results and handles errors
        try:
            all_scores = await asyncio.gather(*tasks)
            return list(all_scores)
        except Exception as e:
            logger.error(f"Error during batch scoring: {e}")
            # Return partial results if some succeeded
            # Gather with return_exceptions=True to get partial results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out exceptions and log them
            valid_scores = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error scoring pair {i}: {result}")
                    # Return default scores for failed pairs
                    valid_scores.append({
                        "entailment": 0.0,
                        "neutral": 0.0,
                        "contradiction": 0.0
                    })
                else:
                    valid_scores.append(result)
            return valid_scores
    
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
        
        Args:
            query_clauses: List of clauses from query market
            candidate_clauses: List of clauses from candidate market
            
        Returns:
            Confidence score (0-1) for clause equivalence
        """
        if not query_clauses or not candidate_clauses:
            # If either has no clauses, return neutral score
            return 0.5
        
        # Score each query clause against all candidate clauses
        clause_scores = []
        for query_clause in query_clauses:
            best_match = 0.0
            for candidate_clause in candidate_clauses:
                # Score pair
                nli_scores = await self.score_equivalence_async(query_clause, candidate_clause)
                confidence, _ = self.map_nli_to_confidence(nli_scores)
                best_match = max(best_match, confidence)
            clause_scores.append(best_match)
        
        # Average of best matches
        if clause_scores:
            return sum(clause_scores) / len(clause_scores)
        return 0.5
