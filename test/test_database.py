"""Test suite for database module.

This module contains tests for database operations including
CRUD operations for m2b, b2c, and c2desc tables.
"""

import pytest
import os
import tempfile
from pathlib import Path

import sys
from pathlib import Path as PathLib

# Add parent directory to path for imports
sys.path.insert(0, str(PathLib(__file__).resolve().parent.parent))

from src.database.db_manager import DatabaseManager


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
def db_manager(temp_db):
    """Create a database manager instance for testing.
    
    Args:
        temp_db: Temporary database path fixture
        
    Yields:
        DatabaseManager: Database manager instance
    """
    db = DatabaseManager(db_path=temp_db)
    yield db
    db.close()


class TestDatabaseManager:
    """Test class for DatabaseManager."""
    
    def test_init(self, temp_db):
        """Test database manager initialization."""
        db = DatabaseManager(db_path=temp_db)
        assert db.conn is not None
        assert os.path.exists(temp_db)
        db.close()
    
    def test_table_creation(self, db_manager):
        """Test that tables are created correctly."""
        cursor = db_manager.conn.cursor()
        
        # Check if tables exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('m2b', 'b2c', 'c2desc')"
        )
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'm2b' in tables
        assert 'b2c' in tables
        assert 'c2desc' in tables
    
    def test_insert_category(self, db_manager):
        """Test category insertion."""
        category_id = db_manager.insert_category("Retail Shopping")
        assert category_id is not None
        assert isinstance(category_id, int)
        
        # Verify insertion
        description = db_manager.get_category_by_id(category_id)
        assert description == "Retail Shopping"
    
    def test_insert_brand(self, db_manager):
        """Test brand insertion."""
        # First create a category
        category_id = db_manager.insert_category("Retail Shopping")
        
        # Insert brand
        brand_id = db_manager.insert_brand(
            name="AMAZON",
            nature="E-commerce",
            category_id=category_id
        )
        
        assert brand_id is not None
        assert isinstance(brand_id, int)
        
        # Verify insertion
        brand_details = db_manager.get_brand_details(brand_id)
        assert brand_details["brand_name"] == "AMAZON"
        assert brand_details["nature_of_business"] == "E-commerce"
        assert brand_details["category_id"] == category_id
    
    def test_insert_merchant_mapping(self, db_manager):
        """Test merchant mapping insertion."""
        # Setup: create category and brand
        category_id = db_manager.insert_category("Retail Shopping")
        brand_id = db_manager.insert_brand(
            name="AMAZON",
            nature="E-commerce",
            category_id=category_id
        )
        
        # Insert merchant mapping
        db_manager.insert_merchant_mapping("Shopping at AMAZON", brand_id)
        
        # Verify insertion
        found_brand_id = db_manager.search_merchant_text("Shopping at AMAZON")
        assert found_brand_id == brand_id
    
    def test_search_merchant_text(self, db_manager):
        """Test merchant text search."""
        # Setup: create category, brand, and mapping
        category_id = db_manager.insert_category("Retail Shopping")
        brand_id = db_manager.insert_brand(
            name="AMAZON",
            nature="E-commerce",
            category_id=category_id
        )
        db_manager.insert_merchant_mapping("Shopping at AMAZON", brand_id)
        
        # Test search - found
        found_brand_id = db_manager.search_merchant_text("Shopping at AMAZON")
        assert found_brand_id == brand_id
        
        # Test search - not found
        not_found = db_manager.search_merchant_text("Shopping at UNKNOWN")
        assert not_found is None
    
    def test_get_brand_details(self, db_manager):
        """Test getting brand details."""
        # Setup: create category and brand
        category_id = db_manager.insert_category("Retail Shopping")
        brand_id = db_manager.insert_brand(
            name="AMAZON",
            nature="E-commerce",
            category_id=category_id
        )
        
        # Get brand details
        details = db_manager.get_brand_details(brand_id)
        assert details["brand_id"] == brand_id
        assert details["brand_name"] == "AMAZON"
        assert details["nature_of_business"] == "E-commerce"
        assert details["category_id"] == category_id
    
    def test_get_category_ids_by_brand(self, db_manager):
        """Test getting category IDs by brand."""
        # Setup: create category and brand
        category_id = db_manager.insert_category("Retail Shopping")
        brand_id = db_manager.insert_brand(
            name="AMAZON",
            nature="E-commerce",
            category_id=category_id
        )
        
        # Get category IDs
        category_ids = db_manager.get_category_ids_by_brand(brand_id)
        assert category_id in category_ids
    
    def test_search_brand_by_name(self, db_manager):
        """Test brand search by name."""
        # Setup: create category and brand
        category_id = db_manager.insert_category("Retail Shopping")
        brand_id = db_manager.insert_brand(
            name="AMAZON",
            nature="E-commerce",
            category_id=category_id
        )
        
        # Search by name - found
        found_brand_id = db_manager.search_brand_by_name("AMAZON")
        assert found_brand_id == brand_id
        
        # Search by name - not found
        not_found = db_manager.search_brand_by_name("UNKNOWN")
        assert not_found is None
    
    def test_get_all_categories(self, db_manager):
        """Test getting all categories."""
        # Insert multiple categories
        cat1_id = db_manager.insert_category("Retail Shopping")
        cat2_id = db_manager.insert_category("Food & Dining")
        
        # Get all categories
        categories = db_manager.get_all_categories()
        assert len(categories) >= 2
        
        # Verify categories
        category_ids = [cat["category_id"] for cat in categories]
        assert cat1_id in category_ids
        assert cat2_id in category_ids


if __name__ == "__main__":
    """Run tests."""
    pytest.main([__file__, "-v"])
