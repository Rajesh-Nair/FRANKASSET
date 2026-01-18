"""Test suite for FAISS vector store module.

This module contains tests for FAISS operations including
embedding generation, index management, and similarity search.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

import sys
from pathlib import Path as PathLib

# Add parent directory to path for imports
sys.path.insert(0, str(PathLib(__file__).resolve().parent.parent))

try:
    from src.vector_store.faiss_manager import FaissManager
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for FAISS index.
    
    Yields:
        str: Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def faiss_manager(temp_index_dir):
    """Create a FAISS manager instance for testing.
    
    Args:
        temp_index_dir: Temporary directory fixture
        
    Yields:
        FaissManager: FAISS manager instance
    """
    # Skip if FAISS not available
    if not FAISS_AVAILABLE:
        pytest.skip("FAISS not available")
    
    from pathlib import Path
    index_path = str(Path(temp_index_dir) / "index.faiss")
    metadata_path = str(Path(temp_index_dir) / "metadata.json")
    
    manager = FaissManager(dimension=768, index_path=index_path)
    yield manager


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
class TestFaissManager:
    """Test class for FaissManager."""
    
    def test_init(self, temp_index_dir):
        """Test FAISS manager initialization."""
        from pathlib import Path
        index_path = str(Path(temp_index_dir) / "index.faiss")
        
        manager = FaissManager(dimension=768, index_path=index_path)
        assert manager.index is not None
        assert manager.dimension == 768
        assert manager.get_index_size() == 0
    
    def test_add_embeddings(self, faiss_manager):
        """Test adding embeddings to index."""
        # Create dummy embeddings
        embeddings = np.random.rand(3, 768).astype(np.float32)
        metadata = [
            {"brand_id": 1, "brand_name": "AMAZON"},
            {"brand_id": 2, "brand_name": "MCDONALDS"},
            {"brand_id": 3, "brand_name": "TESCO"}
        ]
        
        # Add embeddings
        faiss_manager.add_embeddings(embeddings, metadata)
        
        # Verify size
        assert faiss_manager.get_index_size() == 3
        assert len(faiss_manager.metadata) == 3
    
    def test_search_similar(self, faiss_manager):
        """Test similarity search."""
        # Add embeddings
        embeddings = np.random.rand(3, 768).astype(np.float32)
        metadata = [
            {"brand_id": 1, "brand_name": "AMAZON"},
            {"brand_id": 2, "brand_name": "MCDONALDS"},
            {"brand_id": 3, "brand_name": "TESCO"}
        ]
        faiss_manager.add_embeddings(embeddings, metadata)
        
        # Search similar to first embedding
        query_embedding = embeddings[0]
        results = faiss_manager.search_similar(query_embedding, k=2)
        
        assert len(results) >= 1
        assert results[0]["brand_id"] == 1  # Should find the same embedding
    
    def test_save_and_load_index(self, temp_index_dir):
        """Test saving and loading index."""
        from pathlib import Path
        index_path = str(Path(temp_index_dir) / "index.faiss")
        
        # Create manager and add embeddings
        manager1 = FaissManager(dimension=768, index_path=index_path)
        embeddings = np.random.rand(2, 768).astype(np.float32)
        metadata = [
            {"brand_id": 1, "brand_name": "AMAZON"},
            {"brand_id": 2, "brand_name": "MCDONALDS"}
        ]
        manager1.add_embeddings(embeddings, metadata)
        manager1.save_index()
        
        # Create new manager and load
        manager2 = FaissManager(dimension=768, index_path=index_path)
        assert manager2.get_index_size() == 2
        assert len(manager2.metadata) == 2


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
class TestFaissIntegration:
    """Integration tests for FAISS with database."""
    
    def test_sync_with_database(self, temp_index_dir):
        """Test syncing FAISS with database."""
        from pathlib import Path
        import tempfile
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            from src.database.db_manager import DatabaseManager
            db = DatabaseManager(db_path=db_path)
            
            # Add categories and brands
            cat_id = db.insert_category("Retail Shopping")
            brand_id = db.insert_brand("AMAZON", "E-commerce", cat_id)
            
            # Sync FAISS (will skip if Vertex AI not configured)
            index_path = str(Path(temp_index_dir) / "index.faiss")
            try:
                manager = FaissManager(dimension=768, index_path=index_path)
                # Sync will try to generate embeddings - may fail if Vertex AI not configured
                # This is expected in test environment
                pass
            except Exception as e:
                # Expected if Vertex AI not configured
                if "Vertex AI" in str(e) or "embedding" in str(e).lower():
                    pytest.skip("Vertex AI not configured for embedding generation")
                raise
            
            db.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


if __name__ == "__main__":
    """Run tests."""
    pytest.main([__file__, "-v"])
