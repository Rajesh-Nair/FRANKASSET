"""Helper functions for text preprocessing and data transformation.

This module provides utility functions for text preprocessing,
embedding utilities, and data transformation helpers.
"""

import re
from typing import List, Optional
from logger.custom_logger import CustomLogger

logger = CustomLogger().get_logger(__file__)


def preprocess_text(text: str) -> str:
    """Preprocess text for better matching and embedding.
    
    Args:
        text: Raw input text
        
    Returns:
        str: Preprocessed text (lowercased, stripped, normalized whitespace)
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Normalize whitespace (multiple spaces to single space)
    text = re.sub(r'\s+', ' ', text)
    
    return text


def extract_merchant_name(text: str) -> Optional[str]:
    """Extract potential merchant name from transaction text.
    
    This is a simple heuristic - actual extraction should be done by the agent.
    
    Args:
        text: Transaction text
        
    Returns:
        Optional[str]: Extracted merchant name or None
    """
    if not text:
        return None
    
    # Common patterns in transaction text
    # "Shopping at AMAZON" -> "AMAZON"
    # "Payment to McDonalds" -> "McDonalds"
    # "TESCO Store" -> "TESCO"
    
    text = preprocess_text(text)
    
    # Remove common prefixes
    prefixes = [
        "shopping at", "payment to", "transaction at",
        "purchase at", "paid to", "charged at"
    ]
    
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    
    # Extract capitalized words (potential merchant name)
    words = text.split()
    if words:
        # Return first word or first few capitalized words
        return " ".join(words[:2]) if len(words) > 1 else words[0]
    
    return None


def normalize_brand_name(name: str) -> str:
    """Normalize brand name for consistent matching.
    
    Args:
        name: Brand name to normalize
        
    Returns:
        str: Normalized brand name
    """
    if not name:
        return ""
    
    # Convert to uppercase for consistency
    name = name.upper()
    
    # Remove common suffixes/prefixes
    name = re.sub(r'\s+(INC|LLC|LTD|CORP|CORPORATION|CO)\s*$', '', name, flags=re.IGNORECASE)
    
    # Normalize whitespace
    name = preprocess_text(name)
    
    return name


def chunk_text(text: str, max_length: int = 500) -> List[str]:
    """Chunk text into smaller pieces if needed.
    
    Args:
        text: Input text to chunk
        max_length: Maximum length per chunk
        
    Returns:
        List[str]: List of text chunks
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        
        if current_length + word_length > max_length:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks.
    
    Args:
        text: User input text
        
    Returns:
        str: Sanitized text
    """
    if not text:
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # Limit length to prevent DoS
    max_length = 10000
    if len(text) > max_length:
        text = text[:max_length]
    
    return text


if __name__ == "__main__":
    """Test helper functions."""
    # Test preprocess_text
    test_text = "  Shopping   at   AMAZON   "
    logger.info("Testing preprocess_text", original=test_text, preprocessed=preprocess_text(test_text))
    
    # Test extract_merchant_name
    test_transaction = "Shopping at AMAZON Marketplace"
    logger.info("Testing extract_merchant_name", transaction=test_transaction, extracted=extract_merchant_name(test_transaction))
    
    # Test normalize_brand_name
    test_brand = "Amazon.com Inc."
    logger.info("Testing normalize_brand_name", original=test_brand, normalized=normalize_brand_name(test_brand))
    
    # Test chunk_text
    long_text = " ".join(["word"] * 200)
    chunks = chunk_text(long_text, max_length=100)
    logger.info("Testing chunk_text", chunks_count=len(chunks), first_chunk_length=len(chunks[0]))
    
    # Test sanitize_input
    unsafe_input = "Normal text\x00with\x01control\x02chars"
    logger.info("Testing sanitize_input", unsafe_input=repr(unsafe_input), sanitized=repr(sanitize_input(unsafe_input)))
