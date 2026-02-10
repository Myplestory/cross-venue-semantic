"""
Section parser for structured markdown sections.

Fast parsing of canonical text sections using direct string operations.
"""

import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class SectionParser:
    """
    Parse structured markdown sections from canonical text.
    
    Leverages fixed section format for fast parsing.
    """
    
    async def parse_statement(
        self,
        canonical_text: str
    ) -> Tuple[str, Optional[Tuple[int, int]]]:
        """
        Extract Market Statement section.
        
        Returns:
            Tuple of (statement, optional_span)
        """
        lines = canonical_text.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('Market Statement:'):
                event_lines = []
                start_pos = canonical_text.find(line) + len(line)
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    if next_line.startswith(('Resolution Criteria:', 'Clarifications:', 'End Date:', 'Outcomes:')):
                        break
                    event_lines.append(next_line)
                
                if event_lines:
                    statement = ' '.join(event_lines)
                    end_pos = start_pos + len(statement)
                    return statement, (start_pos, end_pos)
        
        first_line = canonical_text.split('\n')[0].strip()
        if first_line:
            return first_line, None
        return canonical_text[:200], None
    
    async def parse_resolution_criteria(
        self,
        canonical_text: str
    ) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        """
        Extract Resolution Criteria section.
        
        Returns:
            Tuple of (criteria, optional_span)
        """
        lines = canonical_text.split('\n')
        in_resolution = False
        criteria_lines = []
        start_pos = None
        
        for i, line in enumerate(lines):
            if line.startswith('Resolution Criteria:'):
                in_resolution = True
                start_pos = canonical_text.find(line) + len(line)
                continue
            elif in_resolution:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith(('Clarifications:', 'End Date:', 'Outcomes:')):
                    break
                criteria_lines.append(stripped)
        
        if criteria_lines:
            criteria = ' '.join(criteria_lines)
            if start_pos:
                end_pos = start_pos + len(criteria)
                return criteria, (start_pos, end_pos)
            return criteria, None
        
        return None, None
    
    async def parse_outcomes(
        self,
        canonical_text: str
    ) -> Tuple[List[str], Optional[List[Tuple[int, int]]]]:
        """
        Extract Outcomes section.
        
        Returns:
            Tuple of (outcomes, optional_spans)
        """
        lines = canonical_text.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('Outcomes:'):
                outcomes_str = line.replace('Outcomes:', '').strip()
                if outcomes_str:
                    outcomes = [o.strip() for o in outcomes_str.split(',')]
                    start_pos = canonical_text.find(line) + len('Outcomes:')
                    spans = []
                    current_pos = start_pos
                    for outcome in outcomes:
                        end_pos = current_pos + len(outcome)
                        spans.append((current_pos, end_pos))
                        current_pos = end_pos + 2
                    return outcomes, spans
                break
        
        return ["Yes", "No"], None
    
    async def parse_end_date(
        self,
        canonical_text: str
    ) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        """
        Extract End Date section.
        
        Returns:
            Tuple of (date_string, optional_span)
        """
        lines = canonical_text.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('End Date:'):
                date_str = line.replace('End Date:', '').strip()
                if date_str:
                    start_pos = canonical_text.find(line) + len('End Date:')
                    end_pos = start_pos + len(date_str)
                    return date_str, (start_pos, end_pos)
                break
        
        return None, None

