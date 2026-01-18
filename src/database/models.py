"""Pydantic models for data validation.

This module defines Pydantic models for validating data structures
used in database operations and API requests/responses.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class BrandModel(BaseModel):
    """Pydantic model for brand data.
    
    Attributes:
        brand_id: Unique identifier for the brand
        brand_name: Name of the brand
        nature_of_business: Description of business nature
        category_id: Associated category ID
        created_at: Timestamp of creation
    """
    
    brand_id: int
    brand_name: str
    nature_of_business: Optional[str] = None
    category_id: int
    created_at: Optional[datetime] = None


class CategoryModel(BaseModel):
    """Pydantic model for category data.
    
    Attributes:
        category_id: Unique identifier for the category
        category_description: Description of the category
        created_at: Timestamp of creation
    """
    
    category_id: int
    category_description: str
    created_at: Optional[datetime] = None


class MerchantMappingModel(BaseModel):
    """Pydantic model for merchant text to brand mapping.
    
    Attributes:
        id: Unique identifier for the mapping
        merchant_text: Text from transaction
        brand_id: Associated brand ID
        created_at: Timestamp of creation
    """
    
    id: int
    merchant_text: str
    brand_id: int
    created_at: Optional[datetime] = None
