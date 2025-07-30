import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv(".env", override=True)

# Company Configuration
COMPANY_NAME = os.getenv("COMPANY_NAME", "Asian Paints Beautiful Homes")
COMPANY_BRAND = os.getenv("COMPANY_BRAND", "Asian Paints Beautiful Homes")
DEFAULT_WARRANTY_PERIOD = os.getenv("DEFAULT_WARRANTY_PERIOD", "1-year")

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# Performance Configuration
MAX_RETRIEVAL_DOCS = int(os.getenv("MAX_RETRIEVAL_DOCS", "20"))
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))

# Response Configuration
DEFAULT_RESPONSE_TIMEOUT = int(os.getenv("DEFAULT_RESPONSE_TIMEOUT", "30"))

# Brand Configuration
DEFAULT_BRANDS = [
    "Asian Paints Beautiful Homes",
    "Nilaya Floor Coverings by Asian Paints with Jaipur Rugs",
    "ROOHE",
    "FRIDA",
    "Remington",
    "Kausar",
    "Mosaique",
]

# Product Categories
PRODUCT_CATEGORIES = ["furnishing", "lights", "bath", "rugs", "furniture"]

# Product Type Mappings are now generated dynamically from the dataset
# No hardcoded mappings needed - they are loaded from product_type_mappings.json

# Warranty Information
WARRANTY_INFO = {
    "standard_period": DEFAULT_WARRANTY_PERIOD,
    "coverage": "defects in materials and workmanship",
    "contact": "customer service team",
    "company": COMPANY_NAME,
}


def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary"""
    return {
        "company_name": COMPANY_NAME,
        "company_brand": COMPANY_BRAND,
        "warranty_period": DEFAULT_WARRANTY_PERIOD,
        "max_retrieval_docs": MAX_RETRIEVAL_DOCS,
        "max_conversation_history": MAX_CONVERSATION_HISTORY,
        "default_brands": DEFAULT_BRANDS,
        "product_categories": PRODUCT_CATEGORIES,
        "product_type_mappings": "dynamic",  # Now loaded from dataset
        "warranty_info": WARRANTY_INFO,
    }


def validate_openai_key() -> bool:
    """Validate that OpenAI API key is properly set"""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        print("❌ OPENAI_API_KEY is not properly configured!")
        print("Please set your OpenAI API key in one of the following ways:")
        print("1. Create a .env file in the backend directory with:")
        print("   OPENAI_API_KEY=your_actual_api_key_here")
        print("2. Set the environment variable:")
        print("   Windows: set OPENAI_API_KEY=your_actual_api_key_here")
        print("   Linux/Mac: export OPENAI_API_KEY=your_actual_api_key_here")
        return False
    return True


def get_openai_key() -> str:
    """Get OpenAI API key with validation"""
    if not validate_openai_key():
        raise ValueError("OpenAI API key is not properly configured")
    return OPENAI_API_KEY
