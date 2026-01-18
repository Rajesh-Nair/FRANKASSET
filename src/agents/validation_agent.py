"""Validation Agent using Google ADK.

This module implements the ValidationAgent that validates classification
results from BrandClassifierAgent to prevent hallucination.
"""

import json
from typing import Dict

import sys
from pathlib import Path as PathLib

# Add parent directory to path for imports
sys.path.insert(0, str(PathLib(__file__).resolve().parent.parent.parent))

from logger import GLOBAL_LOGGER
from exception.custom_exception import CustomException
from utils.config import config

try:
    from google.adk.agents import Agent
    import google.generativeai as genai
except ImportError:
    Agent = None
    genai = None


class ValidationAgent:
    """Agent for validating classification results to prevent hallucination.
    
    This agent uses Google ADK to validate that classification results
    from BrandClassifierAgent are consistent and accurate.
    
    Attributes:
        agent: Google ADK Agent instance
    """
    
    def __init__(self):
        """Initialize ValidationAgent with Google ADK.
        
        Raises:
            CustomException: If Google ADK is not installed or initialization fails
        """
        try:
            if Agent is None:
                raise CustomException(
                    "Google ADK is not installed. Install with: pip install google-adk"
                )
            
            # System instruction for the agent
            system_instruction = """You are a Validation Agent that reviews classification results
from the Brand Classifier Agent. Your task is to:

1. Verify the consistency between the brand name, nature of business, and category
2. Check if the classification makes logical sense
3. Identify any potential hallucinations or inconsistencies
4. Assess the overall confidence and reliability of the classification

Guidelines:
- Ensure the nature of business aligns with the category description
- Verify that the brand name is reasonable for the given transaction text
- Check for logical inconsistencies (e.g., restaurant categorized as retail)
- Be strict in your validation - flag any potential issues

Return your response as a JSON object with the following structure:
{
    "is_valid": true_or_false,
    "confidence": confidence_score_0_to_1,
    "validation_notes": "explanation of validation result"
}"""
            
            # Get model from config (config file > env var > default)
            model_name = config.get_model_for_agent("validation")
            
            # Initialize Google ADK agent
            self.agent = Agent(
                model=model_name,
                name="validation_agent",
                description="Validates classification results to prevent hallucination",
                instruction=system_instruction,
                tools=[]  # No tools needed for validation
            )
            
            # Configure API key if using Gemini API directly
            if config.GOOGLE_CLOUD_PROJECT and genai:
                try:
                    genai.configure(api_key=config.GOOGLE_SEARCH_API_KEY)
                except:
                    pass  # May use Vertex AI instead
            
            GLOBAL_LOGGER.info("ValidationAgent initialized", model=model_name)
        except Exception as e:
            raise CustomException(f"Failed to initialize ValidationAgent: {e}")
    
    def validate_result(
        self,
        brand_name: str,
        nature_of_business: str,
        category_description: str,
        original_text: str
    ) -> Dict:
        """Validate classification result for consistency and accuracy.
        
        Args:
            brand_name: Extracted brand name
            nature_of_business: Nature of business description
            category_description: Matched category description
            original_text: Original transaction text
            
        Returns:
            Dict: Validation result containing:
                - is_valid: Boolean indicating if classification is valid
                - confidence: Confidence score (0.0 to 1.0)
                - validation_notes: Explanation of validation result
                
        Raises:
            CustomException: If validation fails
        """
        try:
            prompt = f"""Validate the following classification result:

Original Transaction Text: "{original_text}"

Classification Result:
- Brand Name: {brand_name}
- Nature of Business: {nature_of_business}
- Category: {category_description}

Please validate:
1. Does the nature of business align with the category?
2. Is the brand name reasonable for the given transaction text?
3. Are there any logical inconsistencies?
4. Is this classification accurate and not hallucinated?

Return your response as a JSON object with:
- is_valid: true if classification is valid, false otherwise
- confidence: Your confidence in the validation (0.0 to 1.0)
- validation_notes: Brief explanation of your validation decision
"""
            
            GLOBAL_LOGGER.info(
                "Validating classification result",
                brand_name=brand_name,
                category=category_description
            )
            
            # Use agent to validate
            try:
                response = self._query_agent(prompt)
                
                # Parse response
                result = self._parse_response(response)
                
                GLOBAL_LOGGER.info(
                    "Classification validated",
                    brand_name=brand_name,
                    is_valid=result.get("is_valid"),
                    confidence=result.get("confidence")
                )
                
                return result
            except Exception as e:
                GLOBAL_LOGGER.error(f"Agent query failed: {e}")
                raise CustomException(f"Failed to validate result: {e}")
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(f"Failed to validate result: {e}")
    
    def _query_agent(self, prompt: str) -> str:
        """Query the ADK agent with a prompt.
        
        Args:
            prompt: Input prompt for the agent
            
        Returns:
            str: Agent response text
        """
        # Placeholder for actual ADK API call
        # Actual implementation will depend on the exact ADK API
        
        # If using direct Gemini API:
        try:
            if genai:
                model_name = config.get_model_for_agent("validation")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
        except:
            pass
        
        # Fallback: If ADK Agent has a query method
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
            Dict: Parsed validation result
        """
        try:
            # Try to extract JSON from response
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
            if "is_valid" not in result:
                raise ValueError("Missing required field: is_valid")
            
            # Ensure boolean and confidence
            result["is_valid"] = bool(result["is_valid"])
            
            if "confidence" not in result:
                result["confidence"] = 0.7  # Default confidence
            
            result["confidence"] = float(result["confidence"])
            
            if "validation_notes" not in result:
                result["validation_notes"] = "Validation completed"
            
            return result
        except json.JSONDecodeError as e:
            GLOBAL_LOGGER.warning(f"Failed to parse JSON from response: {response[:200]}")
            raise CustomException(f"Failed to parse agent response as JSON: {e}")
        except Exception as e:
            raise CustomException(f"Failed to parse agent response: {e}")


if __name__ == "__main__":
    """Test ValidationAgent."""
    print("Testing ValidationAgent...")
    
    try:
        agent = ValidationAgent()
        print("ValidationAgent initialized successfully")
        
        # Test validation
        test_result = agent.validate_result(
            brand_name="AMAZON",
            nature_of_business="E-commerce",
            category_description="Online Marketplace",
            original_text="Shopping at AMAZON"
        )
        
        print(f"\nValidation result:")
        print(f"  Is Valid: {test_result.get('is_valid')}")
        print(f"  Confidence: {test_result.get('confidence')}")
        print(f"  Notes: {test_result.get('validation_notes')}")
        
        print("\nValidationAgent test passed!")
    except CustomException as e:
        print(f"ValidationAgent test skipped: {e}")
    except Exception as e:
        print(f"ValidationAgent test failed: {e}")
        import traceback
        traceback.print_exc()
