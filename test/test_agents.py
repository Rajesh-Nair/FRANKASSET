"""Test suite for agent modules.

This module contains tests for BrandClassifierAgent and ValidationAgent.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path as PathLib

# Add parent directory to path for imports
sys.path.insert(0, str(PathLib(__file__).resolve().parent.parent))


@pytest.fixture
def mock_categories():
    """Mock categories for testing.
    
    Returns:
        List[dict]: List of mock categories
    """
    return [
        {"category_id": 1, "category_description": "Retail Shopping"},
        {"category_id": 2, "category_description": "Food & Dining"},
        {"category_id": 3, "category_description": "Grocery Store"},
        {"category_id": 4, "category_description": "Online Marketplace"}
    ]


class TestBrandClassifierAgent:
    """Test class for BrandClassifierAgent."""
    
    @pytest.mark.skip(reason="Requires Google ADK configuration")
    def test_init(self):
        """Test agent initialization."""
        from src.agents.brand_classifier_agent import BrandClassifierAgent
        
        # This will fail if Google ADK not configured
        try:
            agent = BrandClassifierAgent()
            assert agent.agent is not None
        except Exception as e:
            if "not installed" in str(e).lower() or "not configured" in str(e).lower():
                pytest.skip("Google ADK not configured")
            raise
    
    @pytest.mark.skip(reason="Requires Google ADK configuration")
    def test_classify_merchant(self, mock_categories):
        """Test merchant classification."""
        from src.agents.brand_classifier_agent import BrandClassifierAgent
        
        try:
            agent = BrandClassifierAgent()
            
            # Test classification
            result = agent.classify_merchant(
                "Shopping at AMAZON",
                mock_categories
            )
            
            assert "brand_name" in result
            assert "nature_of_business" in result
            assert "category_description" in result
            assert "category_id" in result
            assert "confidence" in result
        except Exception as e:
            if "not installed" in str(e).lower() or "not configured" in str(e).lower():
                pytest.skip("Google ADK not configured")
            raise


class TestValidationAgent:
    """Test class for ValidationAgent."""
    
    @pytest.mark.skip(reason="Requires Google ADK configuration")
    def test_init(self):
        """Test agent initialization."""
        from src.agents.validation_agent import ValidationAgent
        
        try:
            agent = ValidationAgent()
            assert agent.agent is not None
        except Exception as e:
            if "not installed" in str(e).lower() or "not configured" in str(e).lower():
                pytest.skip("Google ADK not configured")
            raise
    
    @pytest.mark.skip(reason="Requires Google ADK configuration")
    def test_validate_result(self):
        """Test result validation."""
        from src.agents.validation_agent import ValidationAgent
        
        try:
            agent = ValidationAgent()
            
            # Test validation
            result = agent.validate_result(
                brand_name="AMAZON",
                nature_of_business="E-commerce",
                category_description="Online Marketplace",
                original_text="Shopping at AMAZON"
            )
            
            assert "is_valid" in result
            assert "confidence" in result
            assert "validation_notes" in result
            assert isinstance(result["is_valid"], bool)
            assert 0.0 <= result["confidence"] <= 1.0
        except Exception as e:
            if "not installed" in str(e).lower() or "not configured" in str(e).lower():
                pytest.skip("Google ADK not configured")
            raise


class TestGoogleSearchTool:
    """Test class for Google Search tool."""
    
    @pytest.mark.skip(reason="Requires Google Search API configuration")
    def test_google_search_tool(self):
        """Test Google Search tool."""
        from src.agents.brand_classifier_agent import google_search_tool
        
        try:
            result = google_search_tool("AMAZON e-commerce")
            assert "status" in result
            assert "results" in result
        except Exception as e:
            if "not configured" in str(e).lower():
                pytest.skip("Google Search API not configured")
            raise


if __name__ == "__main__":
    """Run tests."""
    pytest.main([__file__, "-v"])
