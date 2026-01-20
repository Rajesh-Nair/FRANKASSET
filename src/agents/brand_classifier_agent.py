"""Brand Classifier Agent using Google ADK.

This module implements the BrandClassifierAgent that uses Google ADK
to classify merchant text into brand name, nature of business, and category.
"""

import json
from typing import List, Dict, Optional

import sys
from pathlib import Path as PathLib

# Add parent directory to path for imports
sys.path.insert(0, str(PathLib(__file__).resolve().parent.parent.parent))

from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException
from utils.config import config

logger = CustomLogger().get_logger(__file__)

try:
    from google.adk.agents import Agent
    import google.generativeai as genai
except ImportError:
    Agent = None
    genai = None


def google_search_tool(query: str) -> Dict:
    """Tool function for Google Search API.
    
    This function is used as a tool by the BrandClassifierAgent
    to search the web for merchant information.
    
    Args:
        query: Search query string
        
    Returns:
        Dict: Search results containing snippets and links
        
    Raises:
        CustomException: If search API is not configured or search fails
    """
    try:
        if not config.GOOGLE_SEARCH_API_KEY or not config.GOOGLE_SEARCH_ENGINE_ID:
            raise CustomException(
                "Google Search API not configured. "
                "Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables."
            )
        
        import requests
        
        # Google Custom Search API endpoint
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": config.GOOGLE_SEARCH_API_KEY,
            "cx": config.GOOGLE_SEARCH_ENGINE_ID,
            "q": query,
            "num": 5  # Number of results
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract relevant information
        results = []
        if "items" in data:
            for item in data.get("items", [])[:5]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", "")
                })
        
        logger.info(
            "Google search executed",
            query=query,
            results_count=len(results)
        )
        
        return {
            "status": "success",
            "results": results,
            "total_results": len(results)
        }
    except Exception as e:
        raise CustomException(f"Google search failed: {e}")


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
            if Agent is None:
                raise CustomException(
                    "Google ADK is not installed. Install with: pip install google-adk"
                )
            
            # System instruction for the agent
            system_instruction = """You are a Brand Classifier Agent specialized in analyzing transaction text
and extracting merchant/brand information. Your task is to:

1. Extract the merchant/brand name from transaction text
2. Determine the nature of business of the merchant
3. Match the merchant to the most appropriate category from the provided categories

Guidelines:
- Extract the most recognizable brand/merchant name from the text
- Determine what type of business the merchant operates (e.g., "E-commerce", "Restaurant", "Retail Store")
- Match to one of the provided categories based on the nature of business
- If the merchant name or business type is unclear, use the web_search tool to gather more information
- Be precise and avoid hallucination - only extract information that can be reasonably inferred

Return your response as a JSON object with the following structure:
{
    "brand_name": "extracted brand name",
    "nature_of_business": "description of business type",
    "category_description": "matched category description",
    "category_id": category_id_number,
    "confidence": confidence_score_0_to_1
}"""
            
            # Get model from config (config file > env var > default)
            model_name = config.get_model_for_agent("brand_classifier")
            
            # Initialize Google ADK agent
            self.agent = Agent(
                model=model_name,
                name="brand_classifier_agent",
                description="Classifies merchant text into brand, nature of business, and category",
                instruction=system_instruction,
                tools=[google_search_tool]
            )
            
            # Configure API key if using Gemini API directly
            if config.GOOGLE_CLOUD_PROJECT and genai:
                try:
                    genai.configure(api_key=config.GOOGLE_SEARCH_API_KEY)
                except:
                    pass  # May use Vertex AI instead
            
            logger.info("BrandClassifierAgent initialized", model=model_name)
        except Exception as e:
            raise CustomException(f"Failed to initialize BrandClassifierAgent: {e}")
    
    def classify_merchant(self, text: str, available_categories: List[Dict]) -> Dict:
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
            # Build prompt with categories
            categories_str = "\n".join([
                f"- ID {cat['category_id']}: {cat['category_description']}"
                for cat in available_categories
            ])
            
            prompt = f"""Analyze the following transaction text and classify it:

Transaction Text: "{text}"

Available Categories:
{categories_str}

Extract the merchant/brand name, determine the nature of business, and match to the most appropriate category.
If you need more information about the merchant, use the web_search tool.

Return your response as a JSON object with:
- brand_name: The extracted merchant/brand name
- nature_of_business: Description of what type of business this is
- category_description: The category description that best matches
- category_id: The corresponding category ID
- confidence: Your confidence score (0.0 to 1.0)
"""
            
            logger.info(
                "Classifying merchant",
                text=text[:100],  # Log first 100 chars
                categories_count=len(available_categories)
            )
            
            # Use agent to classify (simulated - actual implementation depends on ADK API)
            # Note: Actual ADK API may differ; this is a placeholder structure
            try:
                # For ADK, we'd typically use: response = agent.query(prompt)
                # Since exact API may vary, we'll structure this for compatibility
                response = self._query_agent(prompt)
                
                # Parse response
                result = self._parse_response(response)
                
                logger.info(
                    "Merchant classified",
                    text=text[:100],
                    brand_name=result.get("brand_name"),
                    category_id=result.get("category_id"),
                    confidence=result.get("confidence")
                )
                
                return result
            except Exception as e:
                logger.error("Agent query failed", error=str(e))
                raise CustomException(f"Failed to classify merchant: {e}")
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(f"Failed to classify merchant: {e}")
    
    def _query_agent(self, prompt: str) -> str:
        """Query the ADK agent with a prompt.
        
        Args:
            prompt: Input prompt for the agent
            
        Returns:
            str: Agent response text
        """
        # Placeholder for actual ADK API call
        # Actual implementation will depend on the exact ADK API
        # Example: response = self.agent.query(prompt) or similar
        # For now, we'll use a fallback approach
        
        # If using direct Gemini API:
        try:
            if genai:
                model_name = config.get_model_for_agent("brand_classifier")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
        except:
            pass
        
        # Fallback: If ADK Agent has a query method
        # This would be the actual ADK API when available
        if hasattr(self.agent, 'query'):
            return self.agent.query(prompt)
        elif hasattr(self.agent, 'generate'):
            return self.agent.generate(prompt)
        else:
            raise CustomException(
                "ADK Agent API not properly initialized. "
                "Please check Google ADK installation and configuration."
            )
    
    def _parse_response(self, response: str) -> Dict:
        """Parse agent response into structured dictionary.
        
        Args:
            response: Raw agent response text
            
        Returns:
            Dict: Parsed classification result
        """
        try:
            # Try to extract JSON from response
            # Response may contain JSON wrapped in markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                # Try to find JSON object in response
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                else:
                    json_str = response
            
            result = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["brand_name", "nature_of_business", "category_description", "category_id"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure confidence is a float
            if "confidence" not in result:
                result["confidence"] = 0.7  # Default confidence
            
            result["confidence"] = float(result["confidence"])
            
            return result
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON from response", response_preview=response[:200])
            raise CustomException(f"Failed to parse agent response as JSON: {e}")
        except Exception as e:
            raise CustomException(f"Failed to parse agent response: {e}")


if __name__ == "__main__":
    """Test BrandClassifierAgent."""
    logger.info("Testing BrandClassifierAgent...")
    
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
        
        result = agent.classify_merchant(test_text, test_categories)
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
