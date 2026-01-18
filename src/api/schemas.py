"""Pydantic models for API request/response schemas.

This module defines Pydantic models for validating API requests
and structuring API responses.
"""

from typing import Optional, List, Union
from pydantic import BaseModel, Field


# GET /classify schemas

class ClassificationResult(BaseModel):
    """Classification result from BrandClassifierAgent.
    
    Attributes:
        brand_name: Extracted brand name
        nature_of_business: Description of business type
        category_description: Matched category description
        category_id: Matched category ID
        confidence: Confidence score (0.0 to 1.0)
    """
    
    brand_name: str = Field(..., description="Extracted brand name")
    nature_of_business: str = Field(..., description="Description of business type")
    category_description: str = Field(..., description="Matched category description")
    category_id: int = Field(..., description="Matched category ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class ValidationResult(BaseModel):
    """Validation result from ValidationAgent.
    
    Attributes:
        is_valid: Whether classification is valid
        confidence: Confidence score (0.0 to 1.0)
        notes: Validation notes/explanation
    """
    
    is_valid: bool = Field(..., description="Whether classification is valid")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    notes: str = Field(..., description="Validation notes/explanation")


class ClassificationWithValidation(BaseModel):
    """Classification result with validation.
    
    Attributes:
        brand_name: Extracted brand name
        nature_of_business: Description of business type
        category_description: Matched category description
        category_id: Matched category ID
        confidence: Confidence score (0.0 to 1.0)
        validation: Validation result
    """
    
    brand_name: str
    nature_of_business: str
    category_description: str
    category_id: int
    confidence: float
    validation: ValidationResult


class ClassifyTextResult(BaseModel):
    """Result for a single text classification.
    
    Attributes:
        text: Input text that was classified
        found_in_db: Whether text was found in database
        brand_id: Brand ID if found in database
        category_ids: List of category IDs if found in database
        classification: Classification result with validation (if not found in DB)
    """
    
    text: str = Field(..., description="Input text that was classified")
    found_in_db: bool = Field(..., description="Whether text was found in database")
    brand_id: Optional[int] = Field(None, description="Brand ID if found in database")
    category_ids: List[int] = Field(default_factory=list, description="List of category IDs")
    classification: Optional[ClassificationWithValidation] = Field(
        None,
        description="Classification result with validation (if not found in DB)"
    )


class ClassifyRequest(BaseModel):
    """Request model for GET /classify endpoint.
    
    Attributes:
        texts: Single text or list of texts to classify
    """
    
    texts: Union[str, List[str]] = Field(
        ...,
        description="Single text or array of texts to classify"
    )


class ClassifyResponse(BaseModel):
    """Response model for GET /classify endpoint.
    
    Attributes:
        results: List of classification results, one per input text
    """
    
    results: List[ClassifyTextResult] = Field(
        ...,
        description="List of classification results"
    )


# POST /approve schemas

class ApprovalItem(BaseModel):
    """Single approval/rejection item.
    
    Attributes:
        text: Transaction text
        approved: Whether approval was granted
        brand_name: Brand name (if approved)
        category_id: Category ID (if approved)
        nature_of_business: Nature of business (if approved)
    """
    
    text: str = Field(..., description="Transaction text")
    approved: bool = Field(..., description="Whether approval was granted")
    brand_name: Optional[str] = Field(None, description="Brand name if approved")
    category_id: Optional[int] = Field(None, description="Category ID if approved")
    nature_of_business: Optional[str] = Field(
        None,
        description="Nature of business if approved"
    )


class ApproveRequest(BaseModel):
    """Request model for POST /approve endpoint.
    
    Attributes:
        approvals: List of approval/rejection items
    """
    
    approvals: List[ApprovalItem] = Field(
        ...,
        description="List of approval/rejection items"
    )


class ApprovalResult(BaseModel):
    """Result for a single approval/rejection.
    
    Attributes:
        text: Transaction text
        status: Status ("approved" or "rejected")
        brand_id: Brand ID if approved
        message: Status message
    """
    
    text: str = Field(..., description="Transaction text")
    status: str = Field(..., description="Status: 'approved' or 'rejected'")
    brand_id: Optional[int] = Field(None, description="Brand ID if approved")
    message: str = Field(..., description="Status message")


class ApproveResponse(BaseModel):
    """Response model for POST /approve endpoint.
    
    Attributes:
        results: List of approval results, one per approval item
    """
    
    results: List[ApprovalResult] = Field(
        ...,
        description="List of approval results"
    )
