"""
Competitor Handler Service
Handles queries related to competitor brands and provides appropriate responses.
"""

from typing import Dict, List, Optional, Tuple
import re
import os
import json


class CompetitorHandler:
    """Centralized service for handling competitor brand queries."""

    def __init__(self):
        # Define competitor brands and their variations
        self.competitor_brands = {
            "ikea": ["ikea", "ikea furniture", "ikea products", "ikea home"],
            "pepperfry": ["pepperfry", "pepper fry", "pepperfry furniture"],
            "urban_ladder": ["urban ladder", "urbanladder", "urban ladder furniture"],
            "homecentre": ["homecentre", "home centre", "homecentre furniture"],
            "godrej": ["godrej", "godrej furniture", "godrej home"],
            "hometown": ["hometown", "home town", "hometown furniture"],
            "fabindia": ["fabindia", "fab india", "fabindia furniture"],
            "westside": ["westside", "west side", "westside furniture"],
            "shoppers_stop": [
                "shoppers stop",
                "shoppersstop",
                "shoppers stop furniture",
            ],
            "amazon": ["amazon", "amazon furniture", "amazon home"],
            "flipkart": ["flipkart", "flipkart furniture", "flipkart home"],
            "myntra": ["myntra", "myntra furniture", "myntra home"],
            "ajio": ["ajio", "ajio furniture", "ajio home"],
            "snapdeal": ["snapdeal", "snapdeal furniture", "snapdeal home"],
            "paytmmall": ["paytmmall", "paytm mall", "paytmmall furniture"],
            "bigbasket": ["bigbasket", "big basket", "bigbasket furniture"],
            "grofers": ["grofers", "grofers furniture"],
            "dunzo": ["dunzo", "dunzo furniture"],
            "swiggy": ["swiggy", "swiggy furniture"],
            "zomato": ["zomato", "zomato furniture"],
        }

        # Create a flat list of all competitor terms for easy matching
        self.all_competitor_terms = []
        for brand_terms in self.competitor_brands.values():
            self.all_competitor_terms.extend(brand_terms)

    def get_available_categories(self) -> List[str]:
        """
        Dynamically fetch available categories from the system.

        Returns:
            List of available category names
        """
        try:
            # Try to get categories from config first
            from ..config import PRODUCT_CATEGORIES

            if PRODUCT_CATEGORIES:
                return [cat.title() for cat in PRODUCT_CATEGORIES]
        except ImportError:
            pass

        try:
            # Fallback: try to get from category summary file
            category_summary_path = os.path.join(
                os.path.dirname(__file__), "..", "data_source", "category_summary.json"
            )
            if os.path.exists(category_summary_path):
                with open(category_summary_path, "r") as f:
                    category_summary = json.load(f)
                return list(category_summary.keys())
        except Exception:
            pass

        # Final fallback: return default categories
        return ["Furniture", "Rugs", "Lights", "Furnishing", "Bath"]

    def detect_competitor_query(
        self, user_message: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Detect if the query is related to a competitor brand.

        Returns:
            Tuple of (is_competitor, competitor_brand, detected_term)
        """
        user_lower = user_message.lower().strip()

        # Check for competitor brand mentions
        for brand_name, brand_terms in self.competitor_brands.items():
            for term in brand_terms:
                if term in user_lower:
                    return True, brand_name, term

        return False, None, None

    def generate_competitor_response(
        self, competitor_brand: str, user_message: str
    ) -> Dict:
        """
        Generate an appropriate response for competitor queries.

        Args:
            competitor_brand: The detected competitor brand
            user_message: The original user message

        Returns:
            Dictionary with response type and message
        """

        # Get dynamic categories
        available_categories = self.get_available_categories()
        categories_text = ", ".join(available_categories).lower()

        # Define response templates for different competitor types
        response_templates = {
            "ikea": {
                "type": "competitor_redirect",
                "message": f"I understand you're looking for furniture and home decor products. While I can't show you IKEA products specifically, I'd be happy to help you find beautiful home decor items from our exclusive collection. We offer a wide range of {categories_text} that can transform your space. What type of product are you looking for today?",
                "suggested_categories": available_categories,
            },
            "pepperfry": {
                "type": "competitor_redirect",
                "message": f"I see you're interested in home decor and furniture. While I can't access Pepperfry's catalog, I can help you discover our exclusive collection of beautiful home decor items. We have stunning {categories_text}. What would you like to explore?",
                "suggested_categories": available_categories,
            },
            "urban_ladder": {
                "type": "competitor_redirect",
                "message": f"I understand you're looking for home furniture and decor. While I can't show Urban Ladder products, I'd love to help you find beautiful alternatives from our curated collection. We offer premium {categories_text}. What type of product interests you?",
                "suggested_categories": available_categories,
            },
            "homecentre": {
                "type": "competitor_redirect",
                "message": f"I see you're interested in home decor and furniture. While I can't access HomeCentre's products, I can help you discover our exclusive collection of beautiful home items. We have a wonderful selection of {categories_text}. What would you like to explore?",
                "suggested_categories": available_categories,
            },
            "godrej": {
                "type": "competitor_redirect",
                "message": f"I understand you're looking for home products. While I can't show Godrej products specifically, I'd be happy to help you find beautiful home decor items from our exclusive collection. We offer a wide range of {categories_text}. What type of product are you looking for?",
                "suggested_categories": available_categories,
            },
            "amazon": {
                "type": "competitor_redirect",
                "message": f"I understand you're looking for home decor products. While I can't access Amazon's catalog, I can help you discover our exclusive collection of beautiful home items. We offer curated {categories_text}. What would you like to explore?",
                "suggested_categories": available_categories,
            },
            "flipkart": {
                "type": "competitor_redirect",
                "message": f"I see you're interested in home products. While I can't access Flipkart's catalog, I'd love to help you find beautiful alternatives from our exclusive collection. We offer premium {categories_text}. What type of product interests you?",
                "suggested_categories": available_categories,
            },
        }

        # Get the specific response for the competitor brand
        if competitor_brand in response_templates:
            return response_templates[competitor_brand]
        else:
            # Generic response for other competitors
            return {
                "type": "competitor_redirect",
                "message": f"I understand you're looking for home decor products. While I can't access {competitor_brand.title()}'s catalog, I'd be happy to help you find beautiful alternatives from our exclusive collection. We offer a wide range of {categories_text}. What type of product are you looking for today?",
                "suggested_categories": available_categories,
            }

    def is_competitor_budget_query(self, user_message: str) -> bool:
        """
        Check if the query is asking for budget information about competitor products.
        """
        user_lower = user_message.lower()
        budget_keywords = [
            "price",
            "cost",
            "budget",
            "expensive",
            "cheap",
            "affordable",
            "costly",
        ]

        # Check if both competitor and budget keywords are present
        has_competitor = any(term in user_lower for term in self.all_competitor_terms)
        has_budget = any(keyword in user_lower for keyword in budget_keywords)

        return has_competitor and has_budget

    def generate_budget_redirect_response(self, competitor_brand: str) -> Dict:
        """
        Generate response for budget-related competitor queries.
        """
        # Get dynamic categories
        available_categories = self.get_available_categories()
        categories_text = ", ".join(available_categories).lower()

        return {
            "type": "competitor_budget_redirect",
            "message": f"I understand you're asking about pricing for {competitor_brand.title()} products. While I can't provide specific pricing information for other brands, I'd be happy to help you find beautiful and competitively priced home decor items from our collection. We offer a range of {categories_text} to suit different budgets. What type of product are you looking for?",
            "suggested_categories": available_categories,
        }

    def handle_competitor_query(self, user_message: str) -> Optional[Dict]:
        """
        Main method to handle competitor queries.

        Returns:
            Response dictionary if competitor detected, None otherwise
        """
        is_competitor, competitor_brand, detected_term = self.detect_competitor_query(
            user_message
        )

        if not is_competitor:
            return None

        # Check if it's a budget-related query
        if self.is_competitor_budget_query(user_message):
            return self.generate_budget_redirect_response(competitor_brand)

        # Generate standard competitor redirect response
        return self.generate_competitor_response(competitor_brand, user_message)


# Global instance
competitor_handler = CompetitorHandler()
