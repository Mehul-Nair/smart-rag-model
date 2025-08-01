"""
Base Intent Classifier - Abstract Interface

All intent classification implementations must inherit from this base class
and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class IntentType(str, Enum):
    """Strict intent types for routing"""

    # Core intents
    GREETING = "greeting"
    HELP = "help"
    CATEGORY_LIST = "category_list"
    PRODUCT_SEARCH = "product_search"
    BUDGET_QUERY = "budget_query"
    PRODUCT_DETAIL = "product_detail"
    WARRANTY_QUERY = "warranty_query"
    COMPETITOR_REDIRECT = "competitor_redirect"
    INVALID = "invalid"

    # Legacy/fallback intents
    META = "meta"
    PRODUCT = "product"
    CLARIFY = "clarify"
    BUDGET = "budget"


@dataclass
class ClassificationResult:
    """Structured result for intent classification"""

    intent: IntentType
    confidence: float
    method: str  # Implementation name
    reasoning: str
    scores: Dict[IntentType, float]
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None


class BaseIntentClassifier(ABC):
    """Abstract base class for intent classifiers"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the classifier

        Args:
            name: Name of the implementation
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._is_initialized = False
        self._initialize()

    @abstractmethod
    def _initialize(self) -> bool:
        """
        Initialize the classifier implementation

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def classify_intent(self, user_message: str) -> ClassificationResult:
        """
        Classify user message intent

        Args:
            user_message: The user's input message

        Returns:
            ClassificationResult with intent, confidence, and metadata
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if classifier is available and ready to use

        Returns:
            True if available, False otherwise
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the classifier

        Returns:
            Dictionary with classifier information
        """
        return {
            "name": self.name,
            "available": self.is_available(),
            "initialized": self._is_initialized,
            "config": self.config,
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics (if available)

        Returns:
            Dictionary with performance metrics
        """
        return {
            "name": self.name,
            "avg_processing_time": 0.0,
            "total_queries": 0,
            "success_rate": 0.0,
        }
