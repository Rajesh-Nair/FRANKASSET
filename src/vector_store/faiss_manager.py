"""FAISS vector store manager for embeddings.

This module provides FAISS index management, embedding generation using
Google Vertex AI, and similarity search functionality.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import pickle

try:
    import faiss
except ImportError:
    faiss = None

import sys
from pathlib import Path as PathLib

# Add parent directory to path for imports
sys.path.insert(0, str(PathLib(__file__).resolve().parent.parent.parent))

from logger import GLOBAL_LOGGER
from exception.custom_exception import CustomException
from utils.config import config

try:
    from google.cloud import aiplatform
    from vertexai.preview.language_models import TextEmbeddingModel
except ImportError:
    aiplatform = None
    TextEmbeddingModel = None


class FaissManager:
    """Manages FAISS vector store for brand embeddings.
    
    This class handles FAISS index creation, embedding generation using
    Google Vertex AI, similarity search, and persistence to disk.
    
    Attributes:
        index: FAISS index for vector similarity search
        dimension: Embedding dimension (default: 768 for textembedding-gecko)
        metadata: List of metadata dictionaries, one per embedding
        index_path: Path to saved FAISS index file
        metadata_path: Path to saved metadata file
    """
    
    def __init__(self, dimension: Optional[int] = None, index_path: Optional[str] = None):
        """Initialize FAISS manager and load index if it exists.
        
        Args:
            dimension: Embedding dimension (default from config.app.yaml)
            index_path: Path to saved index file. If None, uses config.FAISS_INDEX_PATH
            
        Raises:
            CustomException: If FAISS is not installed or initialization fails
        """
        # Get dimension from config if not provided
        if dimension is None:
            dimension = config.get_faiss_dimension()
        if faiss is None:
            raise CustomException(
                "FAISS is not installed. Install with: pip install faiss-cpu or faiss-gpu"
            )
        
        try:
            self.dimension = dimension
            self.index = None
            self.metadata: List[Dict] = []
            self.index_path = index_path or str(config.FAISS_INDEX_PATH)
            
            # Derive metadata_path from index_path if index_path is provided
            if index_path:
                index_path_obj = Path(index_path)
                # Replace .faiss extension with .json, or append .json if no extension
                if index_path_obj.suffix == '.faiss':
                    self.metadata_path = str(index_path_obj.with_suffix('.json'))
                else:
                    self.metadata_path = str(index_path_obj.parent / f"{index_path_obj.stem}_metadata.json")
            else:
                self.metadata_path = str(config.FAISS_METADATA_PATH)
            
            # Ensure directory exists
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize Vertex AI if available
            self._init_vertex_ai()
            
            # Load existing index if available
            self._load_index()
            
            GLOBAL_LOGGER.info(
                "FAISS manager initialized",
                dimension=self.dimension,
                index_size=len(self.metadata) if self.index else 0
            )
        except Exception as e:
            raise CustomException(f"Failed to initialize FAISS manager: {e}")
    
    def _init_vertex_ai(self):
        """Initialize Google Vertex AI for embeddings."""
        try:
            if TextEmbeddingModel is None:
                raise CustomException(
                    "Google Cloud Vertex AI is not installed. "
                    "Install with: pip install google-cloud-aiplatform"
                )
            
            if config.GOOGLE_CLOUD_PROJECT:
                aiplatform.init(
                    project=config.GOOGLE_CLOUD_PROJECT,
                    location=config.GOOGLE_CLOUD_LOCATION
                )
                self.embedding_model = TextEmbeddingModel.from_pretrained(
                    config.EMBEDDING_MODEL
                )
                GLOBAL_LOGGER.info(
                    "Vertex AI initialized",
                    project=config.GOOGLE_CLOUD_PROJECT,
                    model=config.EMBEDDING_MODEL
                )
            else:
                self.embedding_model = None
                GLOBAL_LOGGER.warning(
                    "Vertex AI not initialized - GOOGLE_CLOUD_PROJECT not set"
                )
        except Exception as e:
            GLOBAL_LOGGER.warning(f"Vertex AI initialization failed: {e}")
            self.embedding_model = None
    
    def _load_index(self):
        """Load FAISS index and metadata from disk if they exist."""
        try:
            index_file = Path(self.index_path)
            metadata_file = Path(self.metadata_path)
            
            if index_file.exists() and metadata_file.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(index_file))
                
                # Load metadata
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                GLOBAL_LOGGER.info(
                    "FAISS index loaded from disk",
                    index_size=len(self.metadata)
                )
            else:
                # Create new empty index
                self.index = faiss.IndexFlatL2(self.dimension)
                self.metadata = []
                GLOBAL_LOGGER.info("Created new FAISS index")
        except Exception as e:
            GLOBAL_LOGGER.warning(f"Failed to load FAISS index: {e}")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using Google Vertex AI.
        
        Args:
            text: Input text to embed
            
        Returns:
            np.ndarray: Embedding vector
            
        Raises:
            CustomException: If embedding generation fails
        """
        try:
            if self.embedding_model is None:
                raise CustomException(
                    "Embedding model not initialized. "
                    "Set GOOGLE_CLOUD_PROJECT environment variable."
                )
            
            # Generate embedding
            embeddings = self.embedding_model.get_embeddings([text])
            embedding_vector = embeddings[0].values
            
            # Convert to numpy array
            embedding = np.array(embedding_vector, dtype=np.float32)
            
            # Reshape to 2D (1, dimension)
            embedding = embedding.reshape(1, -1)
            
            GLOBAL_LOGGER.debug("Generated embedding", text_length=len(text))
            
            return embedding[0]  # Return 1D array
        except Exception as e:
            raise CustomException(f"Failed to generate embedding: {e}")
    
    def add_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict]) -> None:
        """Add embeddings to FAISS index with metadata.
        
        Args:
            embeddings: Numpy array of embeddings (n_vectors, dimension)
            metadata_list: List of metadata dictionaries, one per embedding
                          Each dict should contain at least 'brand_id'
                          
        Raises:
            CustomException: If addition fails
        """
        try:
            if len(embeddings) != len(metadata_list):
                raise CustomException(
                    "Number of embeddings must match number of metadata entries"
                )
            
            # Ensure embeddings are float32 and 2D
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings, dtype=np.float32)
            
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            embeddings = embeddings.astype(np.float32)
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Add metadata
            self.metadata.extend(metadata_list)
            
            GLOBAL_LOGGER.info(
                "Embeddings added to FAISS index",
                count=len(metadata_list),
                total_size=len(self.metadata)
            )
        except Exception as e:
            raise CustomException(f"Failed to add embeddings: {e}")
    
    def search_similar(self, embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar vectors in the FAISS index.
        
        Args:
            embedding: Query embedding vector (1D array)
            k: Number of similar vectors to return
            
        Returns:
            List[Dict]: List of dictionaries containing 'brand_id', 'distance',
                       and 'metadata' for each similar vector
                       
        Raises:
            CustomException: If search fails
        """
        try:
            if self.index is None or self.index.ntotal == 0:
                return []
            
            # Ensure embedding is float32 and 2D
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            embedding = embedding.astype(np.float32)
            
            # Search
            distances, indices = self.index.search(embedding, min(k, self.index.ntotal))
            
            # Build results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata):
                    result = {
                        "brand_id": self.metadata[idx].get("brand_id"),
                        "distance": float(distance),
                        "metadata": self.metadata[idx]
                    }
                    results.append(result)
            
            GLOBAL_LOGGER.debug(
                "FAISS similarity search completed",
                k=k,
                results_count=len(results)
            )
            
            return results
        except Exception as e:
            raise CustomException(f"Failed to search similar vectors: {e}")
    
    def sync_with_database(self, db_manager) -> None:
        """Sync FAISS index with b2c table in database.
        
        This method reads all brands from the database and ensures their
        embeddings are in the FAISS index.
        
        Args:
            db_manager: DatabaseManager instance
            
        Raises:
            CustomException: If sync fails
        """
        try:
            # Get all brands from database
            cursor = db_manager.conn.cursor()
            cursor.execute(
                "SELECT brand_id, brand_name FROM b2c"
            )
            brands = cursor.fetchall()
            
            # Get existing brand IDs in metadata
            existing_brand_ids = {m.get("brand_id") for m in self.metadata}
            
            # Find brands that need embeddings
            new_brands = [
                {"brand_id": brand["brand_id"], "brand_name": brand["brand_name"]}
                for brand in brands
                if brand["brand_id"] not in existing_brand_ids
            ]
            
            if not new_brands:
                GLOBAL_LOGGER.info("FAISS index is already in sync with database")
                return
            
            # Generate embeddings for new brands
            embeddings = []
            metadata_list = []
            
            for brand in new_brands:
                try:
                    embedding = self.get_embedding(brand["brand_name"])
                    embeddings.append(embedding)
                    metadata_list.append({
                        "brand_id": brand["brand_id"],
                        "brand_name": brand["brand_name"]
                    })
                except Exception as e:
                    GLOBAL_LOGGER.warning(
                        f"Failed to generate embedding for brand {brand['brand_id']}: {e}"
                    )
            
            # Add to index
            if embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                self.add_embeddings(embeddings_array, metadata_list)
                
                # Save index
                self.save_index()
            
            GLOBAL_LOGGER.info(
                "FAISS index synced with database",
                new_brands_added=len(metadata_list)
            )
        except Exception as e:
            raise CustomException(f"Failed to sync FAISS with database: {e}")
    
    def save_index(self) -> None:
        """Persist FAISS index and metadata to disk.
        
        Raises:
            CustomException: If save fails
        """
        try:
            if self.index is None:
                return
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
            
            GLOBAL_LOGGER.info(
                "FAISS index saved to disk",
                index_path=self.index_path,
                metadata_path=self.metadata_path
            )
        except Exception as e:
            raise CustomException(f"Failed to save FAISS index: {e}")
    
    def get_index_size(self) -> int:
        """Get current size of the FAISS index.
        
        Returns:
            int: Number of vectors in the index
        """
        return self.index.ntotal if self.index else 0


if __name__ == "__main__":
    """Test FAISS manager."""
    import os
    import tempfile
    from pathlib import Path as PathLib
    
    # Skip test if FAISS or Vertex AI not available
    if faiss is None:
        print("FAISS not installed - skipping test")
    elif TextEmbeddingModel is None or not config.GOOGLE_CLOUD_PROJECT:
        print("Vertex AI not configured - skipping embedding generation test")
        print("Testing FAISS index operations only...")
        
        # Create temporary index path
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = str(Path(tmpdir) / "test_index.faiss")
            metadata_path = str(Path(tmpdir) / "test_metadata.json")
            
            try:
                # Create FAISS manager (without embeddings)
                manager = FaissManager(dimension=768)
                print(f"FAISS manager initialized: {manager.get_index_size()} vectors")
                
                # Create dummy embeddings
                dummy_embeddings = np.random.rand(3, 768).astype(np.float32)
                dummy_metadata = [
                    {"brand_id": 1, "brand_name": "AMAZON"},
                    {"brand_id": 2, "brand_name": "MCDONALDS"},
                    {"brand_id": 3, "brand_name": "TESCO"}
                ]
                
                # Add embeddings
                manager.add_embeddings(dummy_embeddings, dummy_metadata)
                print(f"Added embeddings: {manager.get_index_size()} vectors")
                
                # Search similar
                query_embedding = dummy_embeddings[0]
                results = manager.search_similar(query_embedding, k=2)
                print(f"Search results: {len(results)} similar vectors found")
                for result in results:
                    print(f"  Brand ID: {result['brand_id']}, Distance: {result['distance']:.4f}")
                
                # Save and reload
                manager.save_index()
                print("Index saved successfully")
                
                print("\nFAISS manager test passed!")
            except Exception as e:
                print(f"FAISS test failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("Running full FAISS manager test with Vertex AI...")
        try:
            manager = FaissManager()
            
            # Test embedding generation
            test_text = "AMAZON"
            embedding = manager.get_embedding(test_text)
            print(f"Generated embedding shape: {embedding.shape}")
            
            # Add embedding
            metadata = [{"brand_id": 1, "brand_name": test_text}]
            manager.add_embeddings(embedding.reshape(1, -1), metadata)
            print(f"Added embedding: {manager.get_index_size()} vectors")
            
            # Search
            results = manager.search_similar(embedding, k=1)
            print(f"Search results: {len(results)}")
            
            print("\nFAISS manager test passed!")
        except Exception as e:
            print(f"FAISS test failed: {e}")
            import traceback
            traceback.print_exc()
