"""
Intent Classifier Factory

This module provides a factory pattern for creating different intent classification implementations.
"""

import logging
from typing import Dict, Any, Optional, List
from .base import BaseIntentClassifier, IntentType, ClassificationResult
from .openai_classifier import OpenAIIntentClassifier
from .huggingface_classifier import HuggingFaceIntentClassifier
from .rule_based_classifier import RuleBasedIntentClassifier
from .improved_hybrid_classifier import ImprovedHybridIntentClassifier

logger = logging.getLogger(__name__)


class IntentClassifierFactory:
    """Factory for creating intent classifiers"""

    # Registry of available implementations
    _implementations = {
        "openai": OpenAIIntentClassifier,
        "huggingface": HuggingFaceIntentClassifier,
        "rule_based": RuleBasedIntentClassifier,
        "improved_hybrid": ImprovedHybridIntentClassifier,
    }

    @classmethod
    def register_implementation(cls, name: str, implementation_class: type):
        """Register a new implementation"""
        cls._implementations[name] = implementation_class
        logger.info(f"Registered implementation: {name}")

    @classmethod
    def get_available_implementations(cls) -> List[str]:
        """Get list of available implementations"""
        return list(cls._implementations.keys())

    @classmethod
    def create(
        cls, implementation: str, config: Optional[Dict[str, Any]] = None
    ) -> BaseIntentClassifier:
        """
        Create an intent classifier with the specified implementation

        Args:
            implementation: Name of the implementation ("openai", "huggingface", "rule_based")
            config: Configuration dictionary for the implementation

        Returns:
            Configured intent classifier instance

        Raises:
            ValueError: If implementation is not available
        """
        if implementation not in cls._implementations:
            available = ", ".join(cls._implementations.keys())
            raise ValueError(
                f"Implementation '{implementation}' not available. Available: {available}"
            )

        implementation_class = cls._implementations[implementation]
        return implementation_class(config)

    @classmethod
    def create_hybrid(
        cls, config: Optional[Dict[str, Any]] = None
    ) -> "HybridIntentClassifier":
        """
        Create a hybrid classifier that tries multiple implementations with fallback

        Args:
            config: Configuration dictionary with implementation-specific configs

        Returns:
            Hybrid intent classifier
        """
        return HybridIntentClassifier(config)


class HybridIntentClassifier(BaseIntentClassifier):
    """Hybrid intent classifier with fallback strategy"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize hybrid classifier

        Args:
            config: Configuration dictionary with:
                - implementations: List of implementation names to try in order
                - min_confidence_threshold: Minimum confidence for primary classifier
                - fallback_strategy: "best_confidence" or "first_available"
                - implementation_configs: Dict of configs for each implementation
        """
        # Set attributes before calling parent constructor
        self.implementations = (
            config.get("implementations", ["openai", "huggingface", "rule_based"])
            if config
            else ["openai", "huggingface", "rule_based"]
        )
        self.min_confidence_threshold = (
            config.get("min_confidence_threshold", 0.7) if config else 0.7
        )
        self.fallback_strategy = (
            config.get("fallback_strategy", "best_confidence")
            if config
            else "best_confidence"
        )

        # Get implementation-specific configs
        implementation_configs = (
            config.get("implementation_configs", {}) if config else {}
        )

        # Initialize classifiers
        self.classifiers = []
        for impl_name in self.implementations:
            try:
                impl_config = implementation_configs.get(impl_name, {})
                classifier = IntentClassifierFactory.create(impl_name, impl_config)
                self.classifiers.append(classifier)
                logger.info(f"Initialized {impl_name} classifier")
            except Exception as e:
                logger.warning(f"Failed to initialize {impl_name} classifier: {e}")

        self._total_queries = 0
        self._total_time = 0.0

        # Call parent constructor
        super().__init__("hybrid", config)

    def _initialize(self) -> bool:
        """Initialize the hybrid classifier"""
        # Check if at least one classifier is available
        available_classifiers = [c for c in self.classifiers if c.is_available()]

        if not available_classifiers:
            logger.error("No classifiers available in hybrid setup")
            return False

        self._is_initialized = True
        logger.info(
            f"Hybrid classifier initialized with {len(available_classifiers)} available classifiers"
        )
        return True

    def classify_intent(self, user_message: str) -> ClassificationResult:
        """Classify intent using hybrid approach with fallback strategy"""
        start_time = time.time()

        if not self.is_available():
            return ClassificationResult(
                intent=IntentType.CLARIFY,
                confidence=0.1,
                method=self.name,
                reasoning="No classifiers available",
                scores={IntentType.CLARIFY: 0.1},
                processing_time=0.0,
            )

        # Get available classifiers
        available_classifiers = [c for c in self.classifiers if c.is_available()]

        if not available_classifiers:
            return ClassificationResult(
                intent=IntentType.CLARIFY,
                confidence=0.1,
                method=self.name,
                reasoning="No classifiers available",
                scores={IntentType.CLARIFY: 0.1},
                processing_time=time.time() - start_time,
            )

        # Try each classifier based on strategy
        if self.fallback_strategy == "best_confidence":
            return self._classify_best_confidence(
                user_message, available_classifiers, start_time
            )
        else:  # first_available
            return self._classify_first_available(
                user_message, available_classifiers, start_time
            )

    def _classify_best_confidence(
        self,
        user_message: str,
        classifiers: List[BaseIntentClassifier],
        start_time: float,
    ) -> ClassificationResult:
        """Use the classifier with the best confidence score"""
        best_result = None
        best_confidence = -1

        for classifier in classifiers:
            try:
                result = classifier.classify_intent(user_message)

                # Use this result if confidence is high enough and better than current best
                if (
                    result.confidence >= self.min_confidence_threshold
                    and result.confidence > best_confidence
                ):
                    best_result = result
                    best_confidence = result.confidence

                    # If we have very high confidence, use it immediately
                    if result.confidence >= 0.9:
                        break

            except Exception as e:
                logger.warning(f"Error with {classifier.name} classifier: {e}")
                continue

        # If no high-confidence result, use the best available
        if best_result is None:
            for classifier in classifiers:
                try:
                    result = classifier.classify_intent(user_message)
                    if result.confidence > best_confidence:
                        best_result = result
                        best_confidence = result.confidence
                except Exception as e:
                    logger.warning(f"Error with {classifier.name} classifier: {e}")
                    continue

        # If still no result, use emergency fallback
        if best_result is None:
            processing_time = time.time() - start_time
            return ClassificationResult(
                intent=IntentType.CLARIFY,
                confidence=0.1,
                method=self.name,
                reasoning="All classifiers failed",
                scores={IntentType.CLARIFY: 0.1},
                processing_time=processing_time,
            )

        # Update processing time and statistics
        processing_time = time.time() - start_time
        self._total_queries += 1
        self._total_time += processing_time

        # Update the result with hybrid metadata
        best_result.processing_time = processing_time
        best_result.metadata = best_result.metadata or {}
        best_result.metadata["hybrid_strategy"] = self.fallback_strategy
        best_result.metadata["classifiers_tried"] = len(classifiers)

        return best_result

    def _classify_first_available(
        self,
        user_message: str,
        classifiers: List[BaseIntentClassifier],
        start_time: float,
    ) -> ClassificationResult:
        """Use the first available classifier that meets confidence threshold"""
        for classifier in classifiers:
            try:
                result = classifier.classify_intent(user_message)

                # Use this result if confidence is high enough
                if result.confidence >= self.min_confidence_threshold:
                    processing_time = time.time() - start_time
                    self._total_queries += 1
                    self._total_time += processing_time

                    result.processing_time = processing_time
                    result.metadata = result.metadata or {}
                    result.metadata["hybrid_strategy"] = self.fallback_strategy
                    result.metadata["classifiers_tried"] = (
                        classifiers.index(classifier) + 1
                    )

                    return result

            except Exception as e:
                logger.warning(f"Error with {classifier.name} classifier: {e}")
                continue

        # If no classifier meets threshold, use the first available
        for classifier in classifiers:
            try:
                result = classifier.classify_intent(user_message)
                processing_time = time.time() - start_time
                self._total_queries += 1
                self._total_time += processing_time

                result.processing_time = processing_time
                result.metadata = result.metadata or {}
                result.metadata["hybrid_strategy"] = self.fallback_strategy
                result.metadata["classifiers_tried"] = len(classifiers)

                return result

            except Exception as e:
                logger.warning(f"Error with {classifier.name} classifier: {e}")
                continue

        # Emergency fallback
        processing_time = time.time() - start_time
        return ClassificationResult(
            intent=IntentType.CLARIFY,
            confidence=0.1,
            method=self.name,
            reasoning="All classifiers failed",
            scores={IntentType.CLARIFY: 0.1},
            processing_time=processing_time,
        )

    def is_available(self) -> bool:
        """Check if hybrid classifier is available"""
        return self._is_initialized and any(c.is_available() for c in self.classifiers)

    def get_info(self) -> Dict[str, Any]:
        """Get information about all classifiers"""
        info = super().get_info()
        info["classifiers"] = []

        for classifier in self.classifiers:
            classifier_info = classifier.get_info()
            classifier_info["performance"] = classifier.get_performance_stats()
            info["classifiers"].append(classifier_info)

        return info

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all classifiers"""
        avg_time = (
            self._total_time / self._total_queries if self._total_queries > 0 else 0.0
        )

        return {
            "name": self.name,
            "avg_processing_time": avg_time,
            "total_queries": self._total_queries,
            "success_rate": 1.0,  # Hybrid always has fallback
            "available_classifiers": len(
                [c for c in self.classifiers if c.is_available()]
            ),
            "total_classifiers": len(self.classifiers),
            "strategy": self.fallback_strategy,
            "confidence_threshold": self.min_confidence_threshold,
        }
