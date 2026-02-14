"""
Entity extractor using spaCy NER with regex fallback.

Extracts named entities from market statements and resolution criteria.
Uses spaCy NER as the primary method, with a regex-based fallback for
entities that spaCy misses (e.g. multi-word proper nouns in prediction
market text).
"""

import asyncio
import logging
import re
from typing import List, Optional, Set

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

from canonicalization.contract_spec import EntitySpec

logger = logging.getLogger(__name__)

# ── Common words to exclude from entity extraction ────────────────────
COMMON_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'should', 'could', 'may', 'might', 'must', 'can',
    'yes', 'no', 'not', 'this', 'that', 'it', 'its', 'if', 'then',
}

# ── Pre-compiled regex patterns for entity extraction fallback ────────

# Matches 2+ consecutive capitalized words (proper nouns), allowing for
# common name particles (de, von, etc.) and suffixes (Jr, Sr, II-IV).
_PROPER_NOUN_RE = re.compile(
    r'\b([A-Z][a-z]+(?:\s+(?:de|von|van|al|el|bin|of|the|and|del|di|la|le)\s+)?'
    r'(?:[A-Z][a-z]+(?:-[A-Z][a-z]+)*)(?:\s+(?:Jr|Sr|II|III|IV)\.?)?)\b'
)

# Matches "X vs Y", "X v. Y" patterns (common in sports/legal markets).
_VS_PATTERN = re.compile(
    r'([A-Z][A-Za-z\s]+?)\s+(?:vs?\.?|versus)\s+([A-Z][A-Za-z\s]+?)(?:\s|,|\.|$)',
    re.IGNORECASE,
)

# Words/phrases to exclude from regex entity extraction.
_REGEX_STOPWORDS = {
    "will", "the", "yes", "no", "if", "then", "market", "statement",
    "resolution", "criteria", "outcomes", "end", "date", "trading",
    "question", "resolves", "officially", "holds", "position",
    "market statement", "resolution criteria", "end date",
}


class EntityExtractor:
    """
    Extract entities using spaCy NER + regex fallback.

    Features:
    - Direct (awaited) model loading during ``initialize()``
    - Smoke test to verify NER pipeline is active
    - Regex fallback catches multi-word proper nouns spaCy may miss
    - Deduplication via ``_merge_entities``
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize entity extractor.

        Args:
            model_name: spaCy model to use for NER.
        """
        self.model_name = model_name
        self._nlp = None
        self._initialized = False

    # ── Lifecycle ──────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Load spaCy model and verify NER works.

        Called once during pipeline startup.  Uses ``run_in_executor``
        so the event loop is not blocked by model I/O.
        """
        if not HAS_SPACY:
            logger.warning(
                "spaCy not installed — entity extraction will use regex fallback only"
            )
            self._initialized = True
            return

        if self._initialized:
            return

        try:
            loop = asyncio.get_running_loop()
            self._nlp = await loop.run_in_executor(
                None,
                lambda: spacy.load(self.model_name, disable=["parser", "lemmatizer"]),
            )
            logger.info("Loaded spaCy model: %s", self.model_name)

            # Smoke test: verify NER pipeline is active
            doc = self._nlp("Barack Obama was the president of the United States.")
            if doc.ents:
                logger.info(
                    "spaCy NER verified — found %d entities in smoke test",
                    len(doc.ents),
                )
            else:
                logger.warning(
                    "spaCy NER returned 0 entities on smoke test — NER may be disabled"
                )
        except Exception as e:
            logger.error("Failed to load spaCy model '%s': %s", self.model_name, e)
            self._nlp = None

        self._initialized = True

    # ── Public API ────────────────────────────────────────────────

    async def extract_entities(
        self,
        statement: str,
        resolution_criteria: Optional[str],
    ) -> List[EntitySpec]:
        """
        Extract entities with spaCy NER + regex fallback.

        Args:
            statement: Market statement text.
            resolution_criteria: Optional resolution criteria text.

        Returns:
            List of ``EntitySpec`` objects (deduplicated).
        """
        if not self._initialized:
            await self.initialize()

        combined = f"{statement} {resolution_criteria or ''}".strip()

        # Very short text unlikely to contain meaningful entities
        if len(combined) < 10:
            return []

        # ── Try spaCy first ──────────────────────────────────────
        spacy_entities: List[EntitySpec] = []
        if HAS_SPACY and self._nlp is not None:
            spacy_entities = await self._run_ner(combined)

        # ── Regex fallback: catches entities spaCy may miss ──────
        regex_entities = self._extract_entities_regex(combined)

        # ── Merge: spaCy entities are higher quality — add first ─
        merged = self._merge_entities(spacy_entities, regex_entities)

        if not merged:
            logger.debug("No entities found in: %s", combined[:80])

        return merged

    # ── spaCy NER ─────────────────────────────────────────────────

    async def _run_ner(self, text: str) -> List[EntitySpec]:
        """Run spaCy NER on text."""
        loop = asyncio.get_running_loop()
        doc = await loop.run_in_executor(None, self._nlp, text)

        entities: List[EntitySpec] = []
        seen: Set[str] = set()

        for ent in doc.ents:
            if ent.text.lower() in COMMON_WORDS:
                continue
            if ent.text in seen:
                continue
            seen.add(ent.text)

            entity_type = self._map_spacy_label(ent.label_)
            if entity_type is not None:
                entities.append(EntitySpec(
                    name=ent.text,
                    entity_type=entity_type,
                    aliases=[],
                ))

        return entities

    @staticmethod
    def _map_spacy_label(label: str) -> Optional[str]:
        """
        Map spaCy NER label to ``EntitySpec.entity_type``.

        Returns ``None`` for labels we intentionally ignore
        (DATE, CARDINAL, ORDINAL, MONEY, PERCENT, TIME, QUANTITY)
        so they don't pollute the entity list.
        """
        mapping = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'location',
            'LOC': 'location',
            'NORP': 'other',       # nationalities, religious/political groups
            'EVENT': 'other',
            'PRODUCT': 'other',
            'WORK_OF_ART': 'other',
            'FAC': 'location',     # facilities
        }
        return mapping.get(label)  # returns None for DATE, CARDINAL, etc.

    # ── Regex fallback ────────────────────────────────────────────

    def _extract_entities_regex(self, text: str) -> List[EntitySpec]:
        """
        Regex-based entity extraction fallback.

        Catches multi-word proper nouns (e.g. "Alexandria Ocasio-Cortez",
        "Kansas City Chiefs") that spaCy may miss.
        """
        entities: List[EntitySpec] = []
        seen: Set[str] = set()

        # ── Proper noun patterns ─────────────────────────────────
        for m in _PROPER_NOUN_RE.finditer(text):
            name = m.group(1).strip()
            if name.lower() in _REGEX_STOPWORDS or name.lower() in COMMON_WORDS:
                continue
            if len(name) < 3:
                continue
            if name in seen:
                continue
            seen.add(name)

            # Heuristic type detection
            entity_type = "other"
            word_count = len(name.split())
            if 2 <= word_count <= 4:
                entity_type = "person"

            entities.append(EntitySpec(
                name=name,
                entity_type=entity_type,
                aliases=[],
            ))

        # ── "X vs Y" patterns ────────────────────────────────────
        for m in _VS_PATTERN.finditer(text):
            for group_idx in (1, 2):
                name = m.group(group_idx).strip()
                if name.lower() in _REGEX_STOPWORDS or name.lower() in COMMON_WORDS:
                    continue
                if len(name) < 3 or name in seen:
                    continue
                seen.add(name)
                entities.append(EntitySpec(
                    name=name,
                    entity_type="other",
                    aliases=[],
                ))

        return entities

    # ── Merge / dedup ─────────────────────────────────────────────

    @staticmethod
    def _merge_entities(
        spacy_entities: List[EntitySpec],
        regex_entities: List[EntitySpec],
    ) -> List[EntitySpec]:
        """Merge entities from spaCy and regex, deduplicating by name."""
        seen_lower: Set[str] = set()
        merged: List[EntitySpec] = []

        # spaCy entities are higher quality — add first
        for ent in spacy_entities:
            key = ent.name.lower()
            if key not in seen_lower:
                seen_lower.add(key)
                merged.append(ent)

        # Then add unique regex entities
        for ent in regex_entities:
            key = ent.name.lower()
            if key not in seen_lower:
                seen_lower.add(key)
                merged.append(ent)

        return merged
