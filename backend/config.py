import os
from typing import Dict, Any

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

# Product Type Mappings - Keywords to Category mapping
# Order matters - more specific terms should come first
PRODUCT_TYPE_MAPPINGS = {
    "lamp": "lights",
    "light": "lights",
    "lighting": "lights",
    "rug": "rugs",
    "mat": "mats",
    "curtain": "curtains",
    "shower": "bath",
    "bath": "bath",
    "towel": "bath",
    "sofa": "furniture",
    "chair": "furniture",
    "table": "furniture",
    "bed": "furniture",
    "furnishing": "furnishing",
}

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
        "product_type_mappings": PRODUCT_TYPE_MAPPINGS,
        "warranty_info": WARRANTY_INFO,
    }
