"""
Comparators for pair verification.

Compares ContractSpec fields to determine equivalence.
"""

from .entity_comparator import EntityComparator
from .threshold_comparator import ThresholdComparator
from .date_comparator import DateComparator
from .outcome_mapper import OutcomeMapper

__all__ = [
    "EntityComparator",
    "ThresholdComparator",
    "DateComparator",
    "OutcomeMapper",
]

