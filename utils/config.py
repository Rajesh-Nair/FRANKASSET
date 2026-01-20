"""Configuration management module.

This module handles loading environment variables and configuration settings
for the FrankAsset backend application.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from logger.custom_logger import CustomLogger

logger = CustomLogger().get_logger(__file__)


class Config:
    """Configuration class for managing application settings.
    
    Attributes:
        BASE_DIR: Base directory of the project
        DATA_DIR: Directory for data storage (SQLite, FAISS)
        DB_PATH: Path to SQLite database file
        FAISS_INDEX_DIR: Directory for FAISS index files
        GOOGLE_CLOUD_PROJECT: Google Cloud project ID
        GOOGLE_CLOUD_LOCATION: Google Cloud location/region
        GOOGLE_SEARCH_API_KEY: API key for Google Search API
        GOOGLE_SEARCH_ENGINE_ID: Custom Search Engine ID
        LOG_DIR: Directory for log files
    """
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # Base directory setup
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.DATA_DIR.mkdir(exist_ok=True)
        
        # Database configuration
        self.DB_PATH = os.getenv(
            "DB_PATH",
            str(self.DATA_DIR / "database.db")
        )
        
        # FAISS index configuration
        self.FAISS_INDEX_DIR = self.DATA_DIR / "faiss_index"
        self.FAISS_INDEX_DIR.mkdir(exist_ok=True)
        self.FAISS_INDEX_PATH = self.FAISS_INDEX_DIR / "index.faiss"
        self.FAISS_METADATA_PATH = self.FAISS_INDEX_DIR / "metadata.json"
        
        # Google Cloud configuration
        self.GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        self.GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        # Google Search API configuration
        self.GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "")
        self.GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "")
        
        # Logging configuration
        self.LOG_DIR = os.getenv("LOG_DIR", str(self.BASE_DIR / "logs"))
        
        # Load YAML configuration files
        self._load_yaml_configs()
        
        # Vertex AI model configuration (env vars override YAML config)
        self.EMBEDDING_MODEL = os.getenv(
            "EMBEDDING_MODEL",
            self._get_from_config("embedding_models.default", "textembedding-gecko@003")
        )
        self.LLM_MODEL = os.getenv(
            "LLM_MODEL",
            self._get_from_config("llm_models.default", "gemini-1.5-pro")
        )
    
    def _load_yaml_configs(self):
        """Load YAML configuration files from config folder."""
        self._models_config = {}
        self._app_config = {}
        
        # Path to config files
        config_dir = self.BASE_DIR / "config"
        models_path = config_dir / "models.yaml"
        app_path = config_dir / "app.yaml"
        
        # Load models.yaml
        if models_path.exists():
            try:
                with open(models_path, 'r', encoding='utf-8') as f:
                    self._models_config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning("Failed to load models.yaml", error=str(e))
                self._models_config = {}
        else:
            logger.warning("models.yaml not found", path=str(models_path))
        
        # Load app.yaml
        if app_path.exists():
            try:
                with open(app_path, 'r', encoding='utf-8') as f:
                    self._app_config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning("Failed to load app.yaml", error=str(e))
                self._app_config = {}
        else:
            logger.warning("app.yaml not found", path=str(app_path))
    
    def _get_from_config(self, key_path: str, default: Any = None) -> Any:
        """Get value from nested config dictionary using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., "faiss.similarity_threshold")
            default: Default value if key not found
            
        Returns:
            Config value or default
        """
        # Try app_config first, then models_config
        for config_dict in [self._app_config, self._models_config]:
            keys = key_path.split('.')
            value = config_dict
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
            if value is not None:
                return value
        
        return default
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration from config/models.yaml.
        
        Returns:
            Dict containing model configurations
        """
        return self._models_config.copy() if self._models_config else {}
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration from config/app.yaml.
        
        Returns:
            Dict containing application settings
        """
        return self._app_config.copy() if self._app_config else {}
    
    def get_model_for_agent(self, agent_name: str) -> str:
        """Get model name for a specific agent.
        
        Args:
            agent_name: Name of the agent ("brand_classifier" or "validation")
            
        Returns:
            Model name for the agent
        """
        model_path = f"agent_assignments.{agent_name}.model"
        default = self.LLM_MODEL
        return self._get_from_config(model_path, default)
    
    def get_faiss_similarity_threshold(self) -> float:
        """Get FAISS similarity threshold from config.
        
        Returns:
            Similarity threshold (default: 0.3)
        """
        return float(self._get_from_config("faiss.similarity_threshold", 0.3))
    
    def get_faiss_default_k(self) -> int:
        """Get FAISS default K value from config.
        
        Returns:
            Default K value (default: 5)
        """
        return int(self._get_from_config("faiss.default_k", 5))
    
    def get_faiss_dimension(self) -> int:
        """Get FAISS embedding dimension from config.
        
        Returns:
            Embedding dimension (default: 768)
        """
        return int(self._get_from_config("faiss.dimension", 768))
    
    def get_api_max_texts(self) -> int:
        """Get maximum texts per API request from config.
        
        Returns:
            Maximum texts per request (default: 100)
        """
        return int(self._get_from_config("api.max_texts_per_request", 100))
    
    def get_agent_confidence_threshold(self) -> float:
        """Get agent confidence threshold from config.
        
        Returns:
            Confidence threshold (default: 0.6)
        """
        return float(self._get_from_config("agents.confidence_threshold", 0.6))
    
    def validate(self) -> bool:
        """Validate that required configuration is present.
        
        Returns:
            bool: True if all required config is present, False otherwise
        """
        required_fields = [
            ("GOOGLE_CLOUD_PROJECT", self.GOOGLE_CLOUD_PROJECT),
            ("GOOGLE_SEARCH_API_KEY", self.GOOGLE_SEARCH_API_KEY),
            ("GOOGLE_SEARCH_ENGINE_ID", self.GOOGLE_SEARCH_ENGINE_ID),
        ]
        
        missing = [name for name, value in required_fields if not value]
        if missing:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing)}. "
                "Please set these environment variables."
            )
        return True


# Global configuration instance
config = Config()


if __name__ == "__main__":
    """Test configuration loading."""
    logger.info("Configuration test", base_dir=str(config.BASE_DIR))
    logger.info("Database path", db_path=config.DB_PATH)
    logger.info("FAISS index directory", faiss_index_dir=str(config.FAISS_INDEX_DIR))
    logger.info("Google Cloud project", project=config.GOOGLE_CLOUD_PROJECT)
    logger.info("Google Cloud location", location=config.GOOGLE_CLOUD_LOCATION)
    
    try:
        config.validate()
        logger.info("Configuration validation: PASSED")
    except ValueError as e:
        logger.error("Configuration validation: FAILED", error=str(e))
