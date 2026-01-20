"""FastAPI route handlers for brand classification and approval.

This module implements the POST /classify and POST /approve endpoints
for processing merchant text and handling approvals/rejections.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, status

import sys
from pathlib import Path as PathLib

# Add parent directory to path for imports
sys.path.insert(0, str(PathLib(__file__).resolve().parent.parent.parent))

from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException

logger = CustomLogger().get_logger(__file__)
from src.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ClassifyTextResult,
    ClassificationWithValidation,
    ClassificationResult,
    ValidationResult,
    ApproveRequest,
    ApproveResponse,
    ApprovalResult,
    ApprovalItem
)
from src.database.db_manager import DatabaseManager
from src.vector_store.faiss_manager import FaissManager
from src.agents.brand_classifier_agent import BrandClassifierAgent
from src.agents.validation_agent import ValidationAgent
from utils.helpers import sanitize_input
from utils.config import config

# Initialize router
router = APIRouter(prefix="/api/v1", tags=["classification"])

# Global instances (will be initialized in startup)
db_manager: Optional[DatabaseManager] = None
faiss_manager: Optional[FaissManager] = None
classifier_agent: Optional[BrandClassifierAgent] = None
validation_agent: Optional[ValidationAgent] = None


def get_db_manager() -> DatabaseManager:
    """Get or create database manager instance.
    
    Returns:
        DatabaseManager: Database manager instance
    """
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    elif db_manager.conn is None:
        # Connection was closed, reconnect
        db_manager._reconnect()
    return db_manager


def get_faiss_manager() -> FaissManager:
    """Get or create FAISS manager instance.
    
    Returns:
        FaissManager: FAISS manager instance
    """
    global faiss_manager
    if faiss_manager is None:
        faiss_manager = FaissManager()
    return faiss_manager


def get_classifier_agent() -> BrandClassifierAgent:
    """Get or create classifier agent instance.
    
    Returns:
        BrandClassifierAgent: Classifier agent instance
    """
    global classifier_agent
    if classifier_agent is None:
        classifier_agent = BrandClassifierAgent()
    return classifier_agent


def get_validation_agent() -> ValidationAgent:
    """Get or create validation agent instance.
    
    Returns:
        ValidationAgent: Validation agent instance
    """
    global validation_agent
    if validation_agent is None:
        validation_agent = ValidationAgent()
    return validation_agent


@router.post("/classify", response_model=ClassifyResponse)
async def classify_texts(request: ClassifyRequest):
    """Classify one or more merchant texts.
    
    This endpoint processes merchant text(s) from transaction statements:
    - Searches SQLite database (m2b table) for existing mappings
    - If found: Returns brand_id and category_ids
    - If not found: Uses BrandClassifierAgent + ValidationAgent to classify
    - Uses FAISS to check for similar brand names (fuzzy matching)
    
    Args:
        request: ClassifyRequest containing texts to classify
        
    Returns:
        ClassifyResponse: Classification results for each text
        
    Raises:
        HTTPException: If classification fails
    """
    try:
        # Normalize input: convert single text to list
        texts = [request.texts] if isinstance(request.texts, str) else request.texts
        
        if not texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No texts provided"
            )
        
        # Sanitize inputs
        texts = [sanitize_input(text) for text in texts]
        
        logger.info(
            "Classification request received",
            text_count=len(texts)
        )
        
        # Get instances - handle initialization errors gracefully
        db = get_db_manager()
        faiss_mgr = get_faiss_manager()
        
        # Try to get agents, but don't fail if they're not available
        try:
            classifier = get_classifier_agent()
        except Exception as e:
            logger.warning("Classifier agent not available", error=str(e))
            classifier = None
        
        try:
            validator = get_validation_agent()
        except Exception as e:
            logger.warning("Validation agent not available", error=str(e))
            validator = None
        
        # Get all categories for agent reference
        categories = db.get_all_categories()
        
        results: List[ClassifyTextResult] = []
        
        # Process each text
        for text in texts:
            try:
                result = await _classify_single_text(
                    text=text,
                    db=db,
                    faiss_mgr=faiss_mgr,
                    classifier=classifier,
                    validator=validator,
                    categories=categories
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    "Failed to classify text",
                    text_preview=text[:50],
                    error=str(e),
                    exc_info=True
                )
                # Create error result
                results.append(ClassifyTextResult(
                    text=text,
                    found_in_db=False,
                    brand_id=None,
                    category_ids=[],
                    classification=None
                ))
        
        logger.info(
            "Classification completed",
            total_texts=len(texts),
            found_in_db=sum(1 for r in results if r.found_in_db),
            classified=sum(1 for r in results if r.classification is not None)
        )
        
        return ClassifyResponse(results=results)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Classification request failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )


async def _classify_single_text(
    text: str,
    db: DatabaseManager,
    faiss_mgr: FaissManager,
    classifier: Optional[BrandClassifierAgent],
    validator: Optional[ValidationAgent],
    categories: List[dict]
) -> ClassifyTextResult:
    """Classify a single text.
    
    Args:
        text: Text to classify
        db: Database manager
        faiss_mgr: FAISS manager
        classifier: Classifier agent (may be None if not available)
        validator: Validation agent (may be None if not available)
        categories: Available categories
        
    Returns:
        ClassifyTextResult: Classification result
    """
    # Search database first
    brand_id = db.search_merchant_text(text)
    
    if brand_id:
        # Found in database
        category_ids = db.get_category_ids_by_brand(brand_id)
        
        logger.info(
            "Text found in database",
            text=text[:100],
            brand_id=brand_id,
            category_ids=category_ids
        )
        
        return ClassifyTextResult(
            text=text,
            found_in_db=True,
            brand_id=brand_id,
            category_ids=category_ids,
            classification=None
        )
    
    # Not found in database - use agents if available
    if classifier is None:
        logger.info(
            "Text not found in database, but classifier agent not available",
            text=text[:100]
        )
        # Return result without classification
        return ClassifyTextResult(
            text=text,
            found_in_db=False,
            brand_id=None,
            category_ids=[],
            classification=None
        )
    
    logger.info(
        "Text not found in database, using agents",
        text=text[:100]
    )
    
    try:
        # Classify using BrandClassifierAgent
        classification = classifier.classify_merchant(text, categories)
        
        # Check FAISS for similar brand names (fuzzy matching)
        # If similar brand found, use existing brand_id
        brand_embedding = faiss_mgr.get_embedding(classification["brand_name"])
        similar_brands = faiss_mgr.search_similar(brand_embedding, k=1)
        
        similarity_threshold = config.get_faiss_similarity_threshold()
        if similar_brands and similar_brands[0]["distance"] < similarity_threshold:
            similar_brand_id = similar_brands[0]["brand_id"]
            logger.info(
                "Similar brand found in FAISS",
                brand_name=classification["brand_name"],
                similar_brand_id=similar_brand_id,
                distance=similar_brands[0]["distance"]
            )
            # Update classification with existing brand_id if we find a match
            # (Note: we still return classification, but could adjust brand_id)
        
        # Validate using ValidationAgent if available
        if validator is not None:
            validation = validator.validate_result(
                brand_name=classification["brand_name"],
                nature_of_business=classification["nature_of_business"],
                category_description=classification["category_description"],
                original_text=text
            )
        else:
            # Default validation if validator not available
            validation = {
                "is_valid": True,
                "confidence": classification.get("confidence", 0.7),
                "validation_notes": "Validation skipped (validator not available)"
            }
        
        # Build response
        classification_result = ClassificationResult(
            brand_name=classification["brand_name"],
            nature_of_business=classification["nature_of_business"],
            category_description=classification["category_description"],
            category_id=classification["category_id"],
            confidence=classification["confidence"]
        )
        
        validation_result = ValidationResult(
            is_valid=validation["is_valid"],
            confidence=validation["confidence"],
            notes=validation["validation_notes"]
        )
        
        classification_with_validation = ClassificationWithValidation(
            brand_name=classification_result.brand_name,
            nature_of_business=classification_result.nature_of_business,
            category_description=classification_result.category_description,
            category_id=classification_result.category_id,
            confidence=classification_result.confidence,
            validation=validation_result
        )
        
        logger.info(
            "Text classified successfully",
            text=text[:100],
            brand_name=classification["brand_name"],
            category_id=classification["category_id"],
            is_valid=validation["is_valid"]
        )
        
        return ClassifyTextResult(
            text=text,
            found_in_db=False,
            brand_id=None,
            category_ids=[],
            classification=classification_with_validation
        )
    
    except Exception as e:
        logger.error("Agent classification failed", text_preview=text[:50], error=str(e))
        raise CustomException(f"Failed to classify text: {e}")


@router.post("/approve", response_model=ApproveResponse)
async def approve_classifications(request: ApproveRequest):
    """Handle approval/rejection of classifications.
    
    This endpoint processes approvals/rejections from the frontend:
    - For approved items: Inserts into m2b and b2c tables, updates FAISS
    - For rejected items: Logs but doesn't persist
    
    Args:
        request: ApproveRequest containing approval/rejection items
        
    Returns:
        ApproveResponse: Approval results for each item
        
    Raises:
        HTTPException: If approval processing fails
    """
    try:
        if not request.approvals:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No approvals provided"
            )
        
        logger.info(
            "Approval request received",
            approval_count=len(request.approvals)
        )
        
        # Get instances
        db = get_db_manager()
        faiss_mgr = get_faiss_manager()
        
        results: List[ApprovalResult] = []
        
        # Process each approval
        for approval in request.approvals:
            try:
                result = await _process_approval(
                    approval=approval,
                    db=db,
                    faiss_mgr=faiss_mgr
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    "Failed to process approval",
                    text_preview=approval.text[:50],
                    error=str(e)
                )
                results.append(ApprovalResult(
                    text=approval.text,
                    status="error",
                    brand_id=None,
                    message=f"Failed to process: {str(e)}"
                ))
        
        logger.info(
            "Approval processing completed",
            total_items=len(request.approvals),
            approved=sum(1 for r in results if r.status == "approved"),
            rejected=sum(1 for r in results if r.status == "rejected")
        )
        
        return ApproveResponse(results=results)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Approval request failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Approval processing failed: {str(e)}"
        )


async def _process_approval(
    approval: ApprovalItem,
    db: DatabaseManager,
    faiss_mgr: FaissManager
) -> ApprovalResult:
    """Process a single approval/rejection.
    
    Args:
        approval: Approval item to process
        db: Database manager
        faiss_mgr: FAISS manager
        
    Returns:
        ApprovalResult: Processing result
    """
    if not approval.approved:
        # Rejected - just log
        logger.info(
            "Classification rejected",
            text=approval.text[:100]
        )
        
        return ApprovalResult(
            text=approval.text,
            status="rejected",
            brand_id=None,
            message="Classification rejected by user"
        )
    
    # Approved - persist to database and FAISS
    try:
        # Sanitize inputs
        text = sanitize_input(approval.text)
        brand_name = sanitize_input(approval.brand_name) if approval.brand_name else None
        nature_of_business = sanitize_input(approval.nature_of_business) if approval.nature_of_business else None
        
        if not brand_name or approval.category_id is None:
            raise CustomException("Brand name and category_id are required for approval")
        
        # Check if brand exists (by name or FAISS search)
        brand_id = db.search_brand_by_name(brand_name)
        
        if not brand_id:
            # Check FAISS for similar brand names
            brand_embedding = faiss_mgr.get_embedding(brand_name)
            similar_brands = faiss_mgr.search_similar(brand_embedding, k=1)
            
            similarity_threshold = config.get_faiss_similarity_threshold()
            if similar_brands and similar_brands[0]["distance"] < similarity_threshold:
                # Use existing similar brand
                brand_id = similar_brands[0]["brand_id"]
                logger.info(
                    "Using similar brand from FAISS",
                    brand_name=brand_name,
                    existing_brand_id=brand_id,
                    distance=similar_brands[0]["distance"]
                )
            else:
                # New brand - insert into b2c
                brand_id = db.insert_brand(
                    name=brand_name,
                    nature=nature_of_business,
                    category_id=approval.category_id
                )
                
                # Add embedding to FAISS
                embedding = faiss_mgr.get_embedding(brand_name)
                faiss_mgr.add_embeddings(
                    embedding.reshape(1, -1),
                    [{"brand_id": brand_id, "brand_name": brand_name}]
                )
                faiss_mgr.save_index()
                
                logger.info(
                    "New brand inserted",
                    brand_id=brand_id,
                    brand_name=brand_name,
                    category_id=approval.category_id
                )
        
        # Insert merchant mapping into m2b
        db.insert_merchant_mapping(text=text, brand_id=brand_id)
        
        logger.info(
            "Approval processed successfully",
            text=text[:100],
            brand_id=brand_id,
            brand_name=brand_name
        )
        
        return ApprovalResult(
            text=text,
            status="approved",
            brand_id=brand_id,
            message="Classification approved and persisted"
        )
    
    except Exception as e:
        raise CustomException(f"Failed to process approval: {e}")
