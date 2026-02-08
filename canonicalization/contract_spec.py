"""
ContractSpec Pydantic model.

Structured representation of market contract specifications.
Used for LLM extraction and pair verification.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime


class DateSpec(BaseModel):
    """Date specification."""
    date: datetime
    timezone: Optional[str] = None
    is_deadline: bool = Field(
        default=False,
        description="True if this is a deadline, False if event date"
    )


class EntitySpec(BaseModel):
    """Entity specification (person, organization, location)."""
    name: str
    entity_type: str = Field(
        ...,
        description="Type of entity: 'person', 'organization', 'location', 'other'"
    )
    aliases: List[str] = Field(default_factory=list)
    
    @field_validator('entity_type')
    @classmethod
    def validate_entity_type(cls, v):
        """Validate entity type."""
        allowed = ['person', 'organization', 'location', 'other']
        if v not in allowed:
            raise ValueError(f"entity_type must be one of {allowed}")
        return v


class ThresholdSpec(BaseModel):
    """Numeric threshold specification."""
    value: float
    unit: Optional[str] = Field(
        None,
        description="Unit of measurement: 'dollars', 'percentage', 'count', etc."
    )
    comparison: str = Field(
        default=">=",
        description="Comparison operator: '>=', '<=', '==', '>', '<'"
    )
    
    @field_validator('comparison')
    @classmethod
    def validate_comparison(cls, v):
        """Validate comparison operator."""
        allowed = ['>=', '<=', '==', '>', '<']
        if v not in allowed:
            raise ValueError(f"comparison must be one of {allowed}")
        return v


class ContractSpec(BaseModel):
    """
    Structured contract specification extracted from market text.
    
    Used for LLM extraction and pair verification.
    """
    
    # Core statement
    statement: str = Field(..., description="Core market statement/question")
    
    # Dates
    resolution_date: Optional[DateSpec] = None
    event_date: Optional[DateSpec] = None
    
    # Entities
    entities: List[EntitySpec] = Field(default_factory=list)
    
    # Thresholds
    thresholds: List[ThresholdSpec] = Field(default_factory=list)
    
    # Resolution criteria
    resolution_criteria: Optional[str] = Field(
        None,
        description="Detailed resolution criteria"
    )
    data_source: Optional[str] = Field(
        None,
        description="Data source for resolution, e.g., 'Coinbase', 'official results'"
    )
    
    # Outcomes
    outcome_labels: List[str] = Field(
        default_factory=list,
        description="List of outcome labels, e.g., ['Yes', 'No']"
    )
    
    # Metadata
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM confidence score for extraction"
    )
    extraction_notes: Optional[str] = Field(
        None,
        description="LLM notes about extraction process"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        validate_assignment = True
        # Pydantic v2 compatibility
        from_attributes = True
    
    @classmethod
    async def from_json_async(cls, json_str: str) -> 'ContractSpec':
        """
        Parse ContractSpec from JSON string (async wrapper).
        
        Args:
            json_str: JSON string representation
            
        Returns:
            ContractSpec instance
        """
        import json
        data = json.loads(json_str)
        return cls(**data)
    
    async def to_json_async(self) -> str:
        """
        Serialize to JSON string (async wrapper).
        
        Returns:
            JSON string representation
        """
        return self.model_dump_json()
