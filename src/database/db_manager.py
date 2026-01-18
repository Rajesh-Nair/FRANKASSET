"""Database manager for SQLite operations.

This module provides database connection and CRUD operations
for the m2b, b2c, and c2desc tables.
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

import sys
from pathlib import Path as PathLib

# Add parent directory to path for imports
sys.path.insert(0, str(PathLib(__file__).resolve().parent.parent.parent))

from logger import GLOBAL_LOGGER
from exception.custom_exception import CustomException
from utils.config import config


class DatabaseManager:
    """Manages SQLite database operations for merchant, brand, and category data.
    
    This class handles database connection, table creation, and all CRUD operations
    for the three main tables: m2b (merchant-text to brand-id), b2c (brand-id to
    brand details and category-id), and c2desc (category-id to category description).
    
    Attributes:
        db_path: Path to the SQLite database file
        conn: SQLite database connection
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager and create tables if they don't exist.
        
        Args:
            db_path: Path to SQLite database file. If None, uses config.DB_PATH
            
        Raises:
            CustomException: If database connection or table creation fails
        """
        try:
            self.db_path = db_path or config.DB_PATH
            self.conn = None
            self._connect()
            self._create_tables()
            GLOBAL_LOGGER.info(
                "Database manager initialized",
                db_path=self.db_path
            )
        except Exception as e:
            raise CustomException(f"Failed to initialize database manager: {e}")
    
    def _connect(self):
        """Establish database connection."""
        try:
            # Create parent directory if it doesn't exist
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            GLOBAL_LOGGER.info("Database connection established", db_path=self.db_path)
        except Exception as e:
            raise CustomException(f"Failed to connect to database: {e}")
    
    def _reconnect(self):
        """Reconnect to the database if connection is closed."""
        if self.conn is None:
            self._connect()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            cursor = self.conn.cursor()
            
            # Table: c2desc (category-id to category-description)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS c2desc (
                    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category_description TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table: b2c (brand-id to brand-name, nature of business, category-id)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS b2c (
                    brand_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    brand_name TEXT NOT NULL,
                    nature_of_business TEXT,
                    category_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (category_id) REFERENCES c2desc(category_id)
                )
            """)
            
            # Table: m2b (merchant-text to brand-id)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS m2b (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    merchant_text TEXT UNIQUE NOT NULL,
                    brand_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (brand_id) REFERENCES b2c(brand_id)
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_m2b_merchant_text ON m2b(merchant_text)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_m2b_brand_id ON m2b(brand_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_b2c_brand_name ON b2c(brand_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_b2c_category_id ON b2c(category_id)
            """)
            
            self.conn.commit()
            GLOBAL_LOGGER.info("Database tables created/verified")
        except Exception as e:
            self.conn.rollback()
            raise CustomException(f"Failed to create database tables: {e}")
    
    def search_merchant_text(self, text: str) -> Optional[int]:
        """Search for merchant text in m2b table and return brand_id.
        
        Args:
            text: Merchant text to search for
            
        Returns:
            Optional[int]: Brand ID if found, None otherwise
            
        Raises:
            CustomException: If database query fails
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT brand_id FROM m2b WHERE merchant_text = ?",
                (text,)
            )
            row = cursor.fetchone()
            brand_id = row["brand_id"] if row else None
            
            if brand_id:
                GLOBAL_LOGGER.info(
                    "Merchant text found in database",
                    text=text,
                    brand_id=brand_id
                )
            else:
                GLOBAL_LOGGER.debug("Merchant text not found in database", text=text)
            
            return brand_id
        except Exception as e:
            raise CustomException(f"Failed to search merchant text: {e}")
    
    def get_brand_details(self, brand_id: int) -> Dict:
        """Get brand information from b2c table.
        
        Args:
            brand_id: Brand ID to lookup
            
        Returns:
            Dict: Brand details containing brand_name, nature_of_business,
                  category_id, created_at
                  
        Raises:
            CustomException: If brand_id not found or query fails
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT brand_id, brand_name, nature_of_business, category_id, created_at "
                "FROM b2c WHERE brand_id = ?",
                (brand_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                raise CustomException(f"Brand ID {brand_id} not found")
            
            return {
                "brand_id": row["brand_id"],
                "brand_name": row["brand_name"],
                "nature_of_business": row["nature_of_business"],
                "category_id": row["category_id"],
                "created_at": row["created_at"]
            }
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(f"Failed to get brand details: {e}")
    
    def get_category_ids_by_brand(self, brand_id: int) -> List[int]:
        """Get all category IDs associated with a brand.
        
        Since b2c table has one category_id per brand, this returns a list
        with the brand's category_id. This method is provided for consistency
        with the API design that expects a list.
        
        Args:
            brand_id: Brand ID
            
        Returns:
            List[int]: List of category IDs for the brand
            
        Raises:
            CustomException: If query fails
        """
        try:
            brand_details = self.get_brand_details(brand_id)
            return [brand_details["category_id"]]
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(f"Failed to get category IDs by brand: {e}")
    
    def insert_merchant_mapping(self, text: str, brand_id: int) -> None:
        """Insert merchant text to brand_id mapping into m2b table.
        
        Args:
            text: Merchant text
            brand_id: Associated brand ID
            
        Raises:
            CustomException: If insertion fails (e.g., duplicate text)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO m2b (merchant_text, brand_id) VALUES (?, ?)",
                (text, brand_id)
            )
            self.conn.commit()
            GLOBAL_LOGGER.info(
                "Merchant mapping inserted",
                text=text,
                brand_id=brand_id
            )
        except sqlite3.IntegrityError as e:
            self.conn.rollback()
            raise CustomException(f"Merchant text already exists: {e}")
        except Exception as e:
            self.conn.rollback()
            raise CustomException(f"Failed to insert merchant mapping: {e}")
    
    def insert_brand(
        self, name: str, nature: Optional[str], category_id: int
    ) -> int:
        """Insert brand into b2c table and return brand_id.
        
        Args:
            name: Brand name
            nature: Nature of business (optional)
            category_id: Category ID
            
        Returns:
            int: The newly created brand_id
            
        Raises:
            CustomException: If insertion fails
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO b2c (brand_name, nature_of_business, category_id) "
                "VALUES (?, ?, ?)",
                (name, nature, category_id)
            )
            self.conn.commit()
            brand_id = cursor.lastrowid
            
            GLOBAL_LOGGER.info(
                "Brand inserted",
                brand_id=brand_id,
                brand_name=name,
                category_id=category_id
            )
            
            return brand_id
        except Exception as e:
            self.conn.rollback()
            raise CustomException(f"Failed to insert brand: {e}")
    
    def search_brand_by_name(self, name: str) -> Optional[int]:
        """Search for brand by name in b2c table.
        
        Args:
            name: Brand name to search for
            
        Returns:
            Optional[int]: Brand ID if found, None otherwise
            
        Raises:
            CustomException: If query fails
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT brand_id FROM b2c WHERE brand_name = ?",
                (name,)
            )
            row = cursor.fetchone()
            brand_id = row["brand_id"] if row else None
            
            return brand_id
        except Exception as e:
            raise CustomException(f"Failed to search brand by name: {e}")
    
    def get_category_by_id(self, category_id: int) -> Optional[str]:
        """Get category description by category_id.
        
        Args:
            category_id: Category ID to lookup
            
        Returns:
            Optional[str]: Category description if found, None otherwise
            
        Raises:
            CustomException: If query fails
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT category_description FROM c2desc WHERE category_id = ?",
                (category_id,)
            )
            row = cursor.fetchone()
            return row["category_description"] if row else None
        except Exception as e:
            raise CustomException(f"Failed to get category by ID: {e}")
    
    def get_all_categories(self) -> List[Dict]:
        """Get all categories from c2desc table.
        
        Returns:
            List[Dict]: List of dictionaries containing category_id and
                       category_description for all categories
                       
        Raises:
            CustomException: If query fails
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT category_id, category_description FROM c2desc ORDER BY category_id"
            )
            rows = cursor.fetchall()
            
            return [
                {
                    "category_id": row["category_id"],
                    "category_description": row["category_description"]
                }
                for row in rows
            ]
        except Exception as e:
            raise CustomException(f"Failed to get all categories: {e}")
    
    def insert_category(self, description: str) -> int:
        """Insert category into c2desc table and return category_id.
        
        Args:
            description: Category description
            
        Returns:
            int: The newly created category_id
            
        Raises:
            CustomException: If insertion fails (e.g., duplicate description)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO c2desc (category_description) VALUES (?)",
                (description,)
            )
            self.conn.commit()
            category_id = cursor.lastrowid
            
            GLOBAL_LOGGER.info(
                "Category inserted",
                category_id=category_id,
                description=description
            )
            
            return category_id
        except sqlite3.IntegrityError as e:
            self.conn.rollback()
            # If category already exists, return its ID
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT category_id FROM c2desc WHERE category_description = ?",
                (description,)
            )
            row = cursor.fetchone()
            if row:
                return row["category_id"]
            raise CustomException(f"Failed to insert category: {e}")
        except Exception as e:
            self.conn.rollback()
            raise CustomException(f"Failed to insert category: {e}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None  # Set to None after closing
            GLOBAL_LOGGER.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    """Test database operations."""
    import os
    import tempfile
    
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        test_db_path = tmp.name
    
    try:
        # Initialize database
        db = DatabaseManager(db_path=test_db_path)
        
        # Test category insertion
        cat_id = db.insert_category("Retail Shopping")
        print(f"Inserted category ID: {cat_id}")
        
        # Test brand insertion
        brand_id = db.insert_brand("AMAZON", "E-commerce", cat_id)
        print(f"Inserted brand ID: {brand_id}")
        
        # Test merchant mapping
        db.insert_merchant_mapping("Shopping at AMAZON", brand_id)
        print("Inserted merchant mapping")
        
        # Test search
        found_brand_id = db.search_merchant_text("Shopping at AMAZON")
        print(f"Found brand ID for 'Shopping at AMAZON': {found_brand_id}")
        
        # Test get brand details
        brand_details = db.get_brand_details(brand_id)
        print(f"Brand details: {brand_details}")
        
        # Test get categories
        categories = db.get_all_categories()
        print(f"All categories: {categories}")
        
        # Test search brand by name
        found_brand = db.search_brand_by_name("AMAZON")
        print(f"Found brand ID for 'AMAZON': {found_brand}")
        
        db.close()
        print("\nAll database tests passed!")
    except Exception as e:
        print(f"Database test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)
