"""
Intent Classification Modules - Modular System

This package provides a unified interface for different intent classification implementations:
- OpenAI Fine-tuned Models
- HuggingFace Transformers
- Rule-based Classifiers
- Custom Implementations

Usage:
    from rag.intent_modules import IntentClassifierFactory

    # Create classifier with preferred implementation
    classifier = IntentClassifierFactory.create(
        implementation="improved_hybrid",  # or "huggingface", "rule_based", "openai"
        config={
            "confidence_threshold": 0.5,
            "primary_classifier": "huggingface",
            "fallback_classifier": "rule_based"
        }
    )

    # Use the classifier
    result = classifier.classify_intent("show me bedside tables")
"""

from .base import BaseIntentClassifier, IntentType, ClassificationResult
from .factory import IntentClassifierFactory
from .config import IntentClassifierConfig
from .improved_hybrid_classifier import ImprovedHybridIntentClassifier

__all__ = [
    "BaseIntentClassifier",
    "IntentClassifierFactory",
    "IntentClassifierConfig",
    "ImprovedHybridIntentClassifier",
]
