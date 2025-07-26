"""
Intent Classifier Configuration Management

This module provides configuration management for different intent classification implementations.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(".env")


@dataclass
class IntentClassifierConfig:
    """Configuration for intent classifiers"""

    # General settings
    implementation: str = (
        "rule_based"  # "openai", "huggingface", "rule_based", "hybrid"
    )
    min_confidence_threshold: float = 0.7
    fallback_strategy: str = "best_confidence"  # "best_confidence" or "first_available"

    # OpenAI settings
    openai_model_name: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_temperature: float = 0.0
    openai_max_tokens: int = 10

    # HuggingFace settings
    huggingface_model_name: str = "distilbert-base-uncased"
    huggingface_num_labels: int = 4
    huggingface_max_length: int = 512
    huggingface_device: str = "cpu"

    # Rule-based settings
    rule_based_similarity_threshold: float = 0.5
    rule_based_patterns: Optional[Dict[str, Any]] = None

    # Hybrid settings
    hybrid_implementations: list = None  # List of implementation names to try

    def __post_init__(self):
        """Set default values after initialization"""
        if self.hybrid_implementations is None:
            self.hybrid_implementations = ["openai", "huggingface", "rule_based"]

    @classmethod
    def from_environment(cls) -> "IntentClassifierConfig":
        """Create configuration from environment variables"""
        config = cls()

        # General settings
        config.implementation = os.getenv("INTENT_IMPLEMENTATION", "rule_based")
        config.min_confidence_threshold = float(
            os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.7")
        )
        config.fallback_strategy = os.getenv(
            "INTENT_FALLBACK_STRATEGY", "best_confidence"
        )

        # OpenAI settings
        config.openai_model_name = os.getenv("FINE_TUNED_MODEL_NAME")
        config.openai_api_key = os.getenv("OPENAI_API_KEY")
        config.openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
        config.openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "10"))

        # HuggingFace settings
        config.huggingface_model_name = os.getenv(
            "HF_MODEL_NAME", "distilbert-base-uncased"
        )
        config.huggingface_num_labels = int(os.getenv("HF_NUM_LABELS", "4"))
        config.huggingface_max_length = int(os.getenv("HF_MAX_LENGTH", "512"))
        config.huggingface_device = os.getenv("HF_DEVICE", "cpu")

        # Rule-based settings
        config.rule_based_similarity_threshold = float(
            os.getenv("RULE_SIMILARITY_THRESHOLD", "0.5")
        )

        # Hybrid settings
        hybrid_impls = os.getenv("HYBRID_IMPLEMENTATIONS")
        if hybrid_impls:
            config.hybrid_implementations = [
                impl.strip() for impl in hybrid_impls.split(",")
            ]

        return config

    def get_implementation_config(self, implementation: str) -> Dict[str, Any]:
        """Get configuration for a specific implementation"""
        if implementation == "openai":
            return {
                "model_name": self.openai_model_name,
                "api_key": self.openai_api_key,
                "temperature": self.openai_temperature,
                "max_tokens": self.openai_max_tokens,
            }
        elif implementation == "huggingface":
            return {
                "model_name": self.huggingface_model_name,
                "num_labels": self.huggingface_num_labels,
                "max_length": self.huggingface_max_length,
                "device": self.huggingface_device,
            }
        elif implementation == "rule_based":
            return {
                "similarity_threshold": self.rule_based_similarity_threshold,
                "patterns": self.rule_based_patterns,
            }
        else:
            return {}

    def get_hybrid_config(self) -> Dict[str, Any]:
        """Get configuration for hybrid classifier"""
        return {
            "implementations": self.hybrid_implementations,
            "min_confidence_threshold": self.min_confidence_threshold,
            "fallback_strategy": self.fallback_strategy,
            "implementation_configs": {
                impl: self.get_implementation_config(impl)
                for impl in self.hybrid_implementations
            },
        }

    def validate(self) -> list:
        """Validate configuration and return list of issues"""
        issues = []

        # Check implementation
        valid_implementations = ["openai", "huggingface", "rule_based", "hybrid"]
        if self.implementation not in valid_implementations:
            issues.append(
                f"Invalid implementation: {self.implementation}. Valid: {valid_implementations}"
            )

        # Check confidence threshold
        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            issues.append(
                f"Invalid confidence threshold: {self.min_confidence_threshold}. Must be 0.0-1.0"
            )

        # Check fallback strategy
        valid_strategies = ["best_confidence", "first_available"]
        if self.fallback_strategy not in valid_strategies:
            issues.append(
                f"Invalid fallback strategy: {self.fallback_strategy}. Valid: {valid_strategies}"
            )

        # Check OpenAI configuration
        if self.implementation == "openai" or "openai" in self.hybrid_implementations:
            if not self.openai_model_name:
                issues.append("OpenAI model name not configured")
            if not self.openai_api_key:
                issues.append("OpenAI API key not configured")

        # Check HuggingFace configuration
        if (
            self.implementation == "huggingface"
            or "huggingface" in self.hybrid_implementations
        ):
            if self.huggingface_device not in ["cpu", "cuda"]:
                issues.append(
                    f"Invalid HuggingFace device: {self.huggingface_device}. Valid: cpu, cuda"
                )

        return issues

    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return len(self.validate()) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "implementation": self.implementation,
            "min_confidence_threshold": self.min_confidence_threshold,
            "fallback_strategy": self.fallback_strategy,
            "openai": {
                "model_name": self.openai_model_name,
                "api_key": "***" if self.openai_api_key else None,
                "temperature": self.openai_temperature,
                "max_tokens": self.openai_max_tokens,
            },
            "huggingface": {
                "model_name": self.huggingface_model_name,
                "num_labels": self.huggingface_num_labels,
                "max_length": self.huggingface_max_length,
                "device": self.huggingface_device,
            },
            "rule_based": {
                "similarity_threshold": self.rule_based_similarity_threshold,
                "patterns": self.rule_based_patterns,
            },
            "hybrid": {
                "implementations": self.hybrid_implementations,
            },
        }
