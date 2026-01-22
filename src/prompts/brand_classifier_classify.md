Analyze the following transaction text and classify it:

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
