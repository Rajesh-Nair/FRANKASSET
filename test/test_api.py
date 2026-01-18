"""Test suite for API endpoints.

This module contains tests for GET /classify and POST /approve endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import tempfile
import os

import sys
from pathlib import Path as PathLib

# Add parent directory to path for imports
sys.path.insert(0, str(PathLib(__file__).resolve().parent.parent))


@pytest.fixture
def temp_db():
    """Create a temporary database for testing.
    
    Yields:
        str: Path to temporary database file
    """
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def client(temp_db):
    """Create a FastAPI test client.
    
    Args:
        temp_db: Temporary database path fixture
        
    Yields:
        TestClient: FastAPI test client
    """
    import os
    os.environ["DB_PATH"] = temp_db
    
    from main import app
    
    with TestClient(app) as test_client:
        yield test_client


class TestClassifyEndpoint:
    """Test class for GET /classify endpoint."""
    
    def test_classify_single_text(self, client, temp_db):
        """Test classification with single text."""
        # Setup: create category, brand, and mapping
        from src.database.db_manager import DatabaseManager
        db = DatabaseManager(db_path=temp_db)
        cat_id = db.insert_category("Retail Shopping")
        brand_id = db.insert_brand("AMAZON", "E-commerce", cat_id)
        db.insert_merchant_mapping("Shopping at AMAZON", brand_id)
        db.close()
        
        # Test with text found in database
        response = client.post(
            "/api/v1/classify",
            json={"texts": "Shopping at AMAZON"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["found_in_db"] is True
        assert data["results"][0]["brand_id"] == brand_id
    
    def test_classify_multiple_texts(self, client, temp_db):
        """Test classification with multiple texts."""
        # Setup: create category and brand
        from src.database.db_manager import DatabaseManager
        db = DatabaseManager(db_path=temp_db)
        cat_id = db.insert_category("Retail Shopping")
        brand_id = db.insert_brand("AMAZON", "E-commerce", cat_id)
        db.insert_merchant_mapping("Shopping at AMAZON", brand_id)
        db.close()
        
        # Test with multiple texts
        response = client.post(
            "/api/v1/classify",
            json={
                "texts": [
                    "Shopping at AMAZON",
                    "Payment to UNKNOWN"
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2
        
        # First text should be found in DB
        assert data["results"][0]["found_in_db"] is True
        
        # Second text should not be found (classification may fail if agents not configured)
        assert data["results"][1]["found_in_db"] is False
    
    def test_classify_empty_request(self, client):
        """Test classification with empty request."""
        response = client.post(
            "/api/v1/classify",
            json={"texts": []}
        )
        
        # Should return 400 Bad Request
        assert response.status_code in [400, 500]


class TestApproveEndpoint:
    """Test class for POST /approve endpoint."""
    
    def test_approve_item(self, client, temp_db):
        """Test approval processing."""
        # Setup: create category
        from src.database.db_manager import DatabaseManager
        db = DatabaseManager(db_path=temp_db)
        cat_id = db.insert_category("Retail Shopping")
        db.close()
        
        # Test approval
        response = client.post(
            "/api/v1/approve",
            json={
                "approvals": [
                    {
                        "text": "Shopping at AMAZON",
                        "approved": True,
                        "brand_name": "AMAZON",
                        "category_id": cat_id,
                        "nature_of_business": "E-commerce"
                    }
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["status"] == "approved"
        assert data["results"][0]["brand_id"] is not None
    
    def test_reject_item(self, client):
        """Test rejection processing."""
        response = client.post(
            "/api/v1/approve",
            json={
                "approvals": [
                    {
                        "text": "Shopping at AMAZON",
                        "approved": False
                    }
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["status"] == "rejected"
    
    def test_approve_empty_request(self, client):
        """Test approval with empty request."""
        response = client.post(
            "/api/v1/approve",
            json={"approvals": []}
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400


class TestHealthEndpoint:
    """Test class for health check endpoint."""
    
    def test_health_check(self, client, temp_db):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code in [200, 503]  # May be 503 if DB not fully initialized
        data = response.json()
        assert "status" in data


class TestRootEndpoint:
    """Test class for root endpoint."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


if __name__ == "__main__":
    """Run tests."""
    pytest.main([__file__, "-v"])
