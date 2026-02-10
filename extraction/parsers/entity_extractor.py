"""
Entity extractor using spaCy NER with lazy loading and smart skipping.

Optimized for speed: pre-loads model, skips NER for simple cases.
"""

import asyncio
import logging
import re
from typing import List, Optional, Tuple

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

from canonicalization.contract_spec import EntitySpec

logger = logging.getLogger(__name__)

# Common words to exclude from entity extraction
COMMON_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'should', 'could', 'may', 'might', 'must', 'can'
}


class EntityExtractor:
    """
    Extract entities using NER with optimizations.
    
    Features:
    - Lazy model loading (pre-loads in background)
    - Smart skipping (skips NER for simple cases)
    - Fast heuristic pre-filtering
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._nlp = None
        self._loading_task: Optional[asyncio.Task] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Pre-load spaCy model in background."""
        if not HAS_SPACY:
            logger.warning("spaCy not available, entity extraction disabled")
            return
        
        if self._initialized:
            return
        
        self._loading_task = asyncio.create_task(self._load_model_async())
        self._initialized = True
    
    async def _load_model_async(self):
        """Load spaCy model in executor thread."""
        loop = asyncio.get_event_loop()
        try:
            self._nlp = await loop.run_in_executor(
                None,
                lambda: spacy.load(
                    self.model_name,
                    disable=["parser", "lemmatizer"]
                )
            )
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            self._nlp = None
    
    async def extract_entities(
        self,
        statement: str,
        resolution_criteria: Optional[str]
    ) -> List[EntitySpec]:
        """
        Extract entities with smart skipping.
        
        Skips NER if:
        - Text is too short (<50 chars)
        - No potential entities detected (heuristic)
        """
        combined = f"{statement} {resolution_criteria or ''}".strip()
        
        if len(combined) < 50:
            return []
        
        if not self._has_potential_entities(combined):
            return []
        
        if not HAS_SPACY or self._nlp is None:
            if self._loading_task and not self._loading_task.done():
                await self._loading_task
            if self._nlp is None:
                return []
        
        return await self._run_ner(combined)
    
    def _has_potential_entities(self, text: str) -> bool:
        """
        Heuristic check for potential entities.
        
        Returns True if text contains:
        - Capitalized words (proper nouns)
        - Known entity patterns
        """
        words = text.split()
        capitalized = sum(1 for w in words if w and w[0].isupper())
        
        if capitalized < 2:
            return False
        
        return True
    
    async def _run_ner(self, text: str) -> List[EntitySpec]:
        """Run NER on text."""
        loop = asyncio.get_event_loop()
        doc = await loop.run_in_executor(None, self._nlp, text)
        
        entities = []
        seen = set()
        
        for ent in doc.ents:
            if ent.text.lower() in COMMON_WORDS:
                continue
            
            if ent.text in seen:
                continue
            seen.add(ent.text)
            
            entity_type = self._map_spacy_label(ent.label_)
            if entity_type:
                entities.append(EntitySpec(
                    name=ent.text,
                    entity_type=entity_type,
                    aliases=[]
                ))
        
        return entities
    
    def _map_spacy_label(self, label: str) -> Optional[str]:
        """Map spaCy label to EntitySpec entity_type."""
        mapping = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'location',
            'LOC': 'location',
        }
        return mapping.get(label, 'other')

