"""
Configuration management for the intent classification system.

This module provides centralized configuration for:
- Intent classifier settings
- Model configurations
- Environment variables
- Default values
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(".env")


class IntentClassifierConfig:
    """Configuration for intent classification system"""

    # Fine-tuned model settings
    FINE_TUNED_MODEL_NAME: Optional[str] = os.getenv("FINE_TUNED_MODEL_NAME")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Confidence thresholds
    MIN_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.7")
    )

    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Performance settings
    ENABLE_PERFORMANCE_MONITORING: bool = (
        os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
    )

    @classmethod
    def get_classifier_config(cls) -> dict:
        """Get configuration for intent classifier"""
        return {
            "fine_tuned_model_name": cls.FINE_TUNED_MODEL_NAME,
            "api_key": cls.OPENAI_API_KEY,
            "min_confidence_threshold": cls.MIN_CONFIDENCE_THRESHOLD,
        }

    @classmethod
    def should_use_fine_tuned_model(cls) -> bool:
        """Check if fine-tuned model should be used"""
        return bool(cls.FINE_TUNED_MODEL_NAME and cls.OPENAI_API_KEY)

    @classmethod
    def validate_config(cls) -> list[str]:
        """Validate configuration and return list of issues"""
        issues = []

        if cls.FINE_TUNED_MODEL_NAME and not cls.OPENAI_API_KEY:
            issues.append("FINE_TUNED_MODEL_NAME is set but OPENAI_API_KEY is missing")

        if cls.MIN_CONFIDENCE_THRESHOLD < 0.0 or cls.MIN_CONFIDENCE_THRESHOLD > 1.0:
            issues.append("MIN_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")

        return issues


# Environment variable documentation
ENV_VARS_DOC = """
Environment Variables for Intent Classification:

FINE_TUNED_MODEL_NAME: Name of the fine-tuned OpenAI model (optional)
    Example: "ft:gpt-3.5-turbo-0613:your-org:your-model-name:1234567890"

OPENAI_API_KEY: OpenAI API key for fine-tuned model access (optional)
    Example: "sk-..."

MIN_CONFIDENCE_THRESHOLD: Minimum confidence for fine-tuned model (default: 0.7)
    Range: 0.0 to 1.0

LOG_LEVEL: Logging level (default: INFO)
    Options: DEBUG, INFO, WARNING, ERROR

ENABLE_PERFORMANCE_MONITORING: Enable performance monitoring (default: true)
    Options: true, false
"""
