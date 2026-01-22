"""Brand Classifier Agent using Google ADK.

This module implements the BrandClassifierAgent that uses Google ADK
to classify merchant text into brand name, nature of business, and category.
"""

import json
from typing import List, Dict, Optional

import sys
from pathlib import Path as PathLib
import uuid

# Add parent directory to path for imports
sys.path.insert(0, str(PathLib(__file__).resolve().parent.parent.parent))

from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException
from utils.config import config
from src.prompts import get_prompt_loader
from src.database.models import BrandModel

logger = CustomLogger().get_logger(__file__)

from google.adk.agents import Agent
from google.adk.tools import google_search


# Brand Classifier Agent
class BrandClassifierAgent:
    """Agent for classifying merchant text into brand, nature of business, and category.
    
    This agent uses Google ADK with Google Search API tool to:
    - Extract merchant/brand name from transaction text
    - Determine nature of business
    - Match to existing category from c2desc table
    - Use web search when merchant information is unclear
    
    Attributes:
        agent: Google ADK Agent instance
    """
    
    def __init__(self):
        """Initialize BrandClassifierAgent with Google ADK.
        
        Raises:
            CustomException: If Google ADK is not installed or initialization fails
        """
        try:            
            # Load system instruction from prompt file
            prompt_loader = get_prompt_loader()
            system_instruction = prompt_loader.load_prompt("brand_classifier_system.md")
            
            # Get model from config (config file > env var > default)
            model_name = config.get_model_for_agent("brand_classifier")
            
            # Initialize Google ADK agent
            self.agent = Agent(
                model=model_name,
                name="brand_classifier_agent",
                description="Classifies merchant text into brand, nature of business, and category",
                instruction=system_instruction,
                tools=[google_search],
                response_format=BrandModel
            )

            
            logger.info("BrandClassifierAgent initialized", model=model_name)
        except Exception as e:
            raise CustomException(f"Failed to initialize BrandClassifierAgent: {e}")

    async def run_InMemoryRunner(self, text: str, available_categories: List[Dict]) -> str:
        """Run BrandClassifierAgent with InMemoryRunner.
        
        Args:
            text: Transaction text to classify
            
        Returns:
            str: Classification result containing:
                - brand_name: Extracted brand name
                - nature_of_business: Description of business type
                - category_description: Matched category description
                - category_id: Matched category ID
                - confidence: Confidence score (0.0 to 1.0)
                
        Raises:
            CustomException: If Google ADK is not installed or initialization fails
        """
        try:
            from google.adk.runners import InMemoryRunner
            runner = InMemoryRunner(agent=self.agent)
            text = self._classify_merchant_prompt(text, available_categories)
            response = await runner.run_async(
                user_id="batch",
                session_id=str(uuid.uuid4()),
                message=text
            )
            content = ""
            async for event in response:
                if event.content:
                    content += event.content
            return content
        except Exception as e:
            raise CustomException(f"Failed to run BrandClassifierAgent InMemoryRunner: {e}")
    
   def _classify_merchant_prompt(self, text: str, available_categories: List[Dict]) -> str:
        """Classify merchant text into brand name, nature of business, and category.
        
        Args:
            text: Transaction text to classify
            available_categories: List of dictionaries containing category_id and
                                category_description from c2desc table
                                
        Returns:
            Dict: Classification result containing:
                - brand_name: Extracted brand name
                - nature_of_business: Description of business type
                - category_description: Matched category description
                - category_id: Matched category ID
                - confidence: Confidence score (0.0 to 1.0)
                
        Raises:
            CustomException: If classification fails
        """
        try:
            # Build categories string
            categories_str = "\n".join([
                f"- ID {cat['category_id']}: {cat['category_description']}"
                for cat in available_categories
            ])
            
            # Load and format prompt from file
            prompt_loader = get_prompt_loader()
            prompt = prompt_loader.format_prompt(
                "brand_classifier_classify.md",
                text=text,
                categories_str=categories_str
            )
            
            logger.info(
                "Classifying merchant",
                text=text[:100],  # Log first 100 chars
                categories_count=len(available_categories)
            )

            return prompt
        except Exception as e:
            raise CustomException(f"Failed to generateclassify merchant prompt: {e}")


if __name__ == "__main__":
    """Test BrandClassifierAgent."""
    logger.info("Testing BrandClassifierAgent...")

    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    
    # Mock categories for testing
    test_categories = [
        {"category_id": 1, "category_description": "Retail Shopping"},
        {"category_id": 2, "category_description": "Food & Dining"},
        {"category_id": 3, "category_description": "Grocery Store"},
        {"category_id": 4, "category_description": "Online Marketplace"}
    ]
    
    try:
        agent = BrandClassifierAgent()
        logger.info("BrandClassifierAgent initialized successfully")
        
        # Test classification (will fail if ADK not configured)
        test_text = "Shopping at AMAZON"
        logger.info("Classifying merchant text", text=test_text)
        
        result = agent.run_InMemoryRunner(test_text, test_categories)
        logger.info(
            "Classification result",
            brand_name=result.get('brand_name'),
            nature_of_business=result.get('nature_of_business'),
            category=result.get('category_description'),
            category_id=result.get('category_id'),
            confidence=result.get('confidence')
        )
        
        logger.info("BrandClassifierAgent test passed")
    except CustomException as e:
        logger.warning("BrandClassifierAgent test skipped", error=str(e))
    except Exception as e:
        logger.error("BrandClassifierAgent test failed", error=str(e))
        import traceback
        traceback.print_exc()
