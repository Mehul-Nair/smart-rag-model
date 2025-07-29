"""
Improved Hybrid Intent Classifier

This module provides an improved hybrid intent classification system with:
- Confidence-based fallback (only fallback if model confidence < 0.5)
- Intent-specific rules instead of catch-all mapping
- Better edge case handling
- OpenAI as third fallback option
"""

import time
import logging
from typing import Dict, Any, Optional, List
from .base import BaseIntentClassifier, IntentType, ClassificationResult
from .huggingface_classifier import HuggingFaceIntentClassifier
from .rule_based_classifier import RuleBasedIntentClassifier
from .openai_classifier import OpenAIIntentClassifier

logger = logging.getLogger(__name__)


class ImprovedHybridIntentClassifier(BaseIntentClassifier):
    """Improved hybrid intent classifier with confidence-based fallback and OpenAI as third fallback"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize improved hybrid classifier

        Args:
            config: Configuration dictionary with:
                - confidence_threshold: Minimum confidence for primary classifier (default: 0.5)
                - primary_classifier: Primary classifier type (default: "huggingface")
                - fallback_classifier: Fallback classifier type (default: "rule_based")
                - openai_fallback: Enable OpenAI as third fallback (default: True)
                - enable_intent_specific_rules: Enable intent-specific rules (default: True)
        """
        self.confidence_threshold = (
            config.get("confidence_threshold", 0.5) if config else 0.5
        )
        self.primary_classifier_type = (
            config.get("primary_classifier", "huggingface") if config else "huggingface"
        )
        self.fallback_classifier_type = (
            config.get("fallback_classifier", "rule_based") if config else "rule_based"
        )
        self.openai_fallback = config.get("openai_fallback", True) if config else True
        self.enable_intent_specific_rules = (
            config.get("enable_intent_specific_rules", True) if config else True
        )

        # Initialize classifiers
        self.primary_classifier = None
        self.fallback_classifier = None
        self.openai_classifier = None

        # Performance tracking
        self._total_queries = 0
        self._total_time = 0.0
        self._primary_used = 0
        self._fallback_used = 0
        self._openai_fallback_used = 0
        self._confidence_issues = 0

        # Call parent constructor
        super().__init__("improved_hybrid", config)

    def _initialize(self) -> bool:
        """Initialize the improved hybrid classifier"""
        try:
            self._initialize_classifiers()

            # Check if at least one classifier is available
            primary_available = (
                self.primary_classifier is not None
                and self.primary_classifier.is_available()
            )
            fallback_available = (
                self.fallback_classifier is not None
                and self.fallback_classifier.is_available()
            )
            openai_available = (
                self.openai_classifier is not None
                and self.openai_classifier.is_available()
            )

            if (
                not primary_available
                and not fallback_available
                and not openai_available
            ):
                logger.error("No classifiers available in improved hybrid setup")
                return False

            self._is_initialized = True
            logger.info(
                f"Improved hybrid classifier initialized with primary: {primary_available}, fallback: {fallback_available}, openai: {openai_available}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize improved hybrid classifier: {e}")
            return False

    def _initialize_classifiers(self):
        """Initialize primary, fallback, and OpenAI classifiers"""
        # Get implementation-specific configs
        implementation_configs = (
            self.config.get("implementation_configs", {}) if self.config else {}
        )

        try:
            # Initialize primary classifier
            primary_config = implementation_configs.get(
                self.primary_classifier_type, {}
            )
            if self.primary_classifier_type == "huggingface":
                self.primary_classifier = HuggingFaceIntentClassifier(primary_config)
            else:
                from .factory import IntentClassifierFactory

                self.primary_classifier = IntentClassifierFactory.create(
                    self.primary_classifier_type, primary_config
                )
            logger.info(
                f"✅ Primary classifier ({self.primary_classifier_type}) initialized"
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize primary classifier: {e}")
            self.primary_classifier = None

        try:
            # Initialize fallback classifier
            fallback_config = implementation_configs.get(
                self.fallback_classifier_type, {}
            )
            if self.fallback_classifier_type == "rule_based":
                self.fallback_classifier = RuleBasedIntentClassifier(fallback_config)
            else:
                from .factory import IntentClassifierFactory

                self.fallback_classifier = IntentClassifierFactory.create(
                    self.fallback_classifier_type, fallback_config
                )
            logger.info(
                f"✅ Fallback classifier ({self.fallback_classifier_type}) initialized"
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize fallback classifier: {e}")
            self.fallback_classifier = None

        # Initialize OpenAI classifier as third fallback
        if self.openai_fallback:
            try:
                openai_config = implementation_configs.get("openai", {})
                self.openai_classifier = OpenAIIntentClassifier(openai_config)
                logger.info("✅ OpenAI classifier initialized as third fallback")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize OpenAI classifier: {e}")
                self.openai_classifier = None

    def _handle_edge_cases(self, user_input: str) -> Optional[ClassificationResult]:
        """Handle edge cases with safety nets"""
        if not user_input or not user_input.strip():
            return ClassificationResult(
                intent=IntentType.CLARIFY,
                confidence=0.0,
                method=self.name,
                reasoning="Empty or whitespace-only input",
                scores={IntentType.CLARIFY: 0.0},
                processing_time=0.0,
            )

        # Check for very short inputs
        if len(user_input.strip()) < 2:
            return ClassificationResult(
                intent=IntentType.CLARIFY,
                confidence=0.1,
                method=self.name,
                reasoning="Input too short for meaningful classification",
                scores={IntentType.CLARIFY: 0.1},
                processing_time=0.0,
            )

        # Check for only punctuation
        import re

        if re.match(r"^[^\w\s]*$", user_input.strip()):
            return ClassificationResult(
                intent=IntentType.CLARIFY,
                confidence=0.1,
                method=self.name,
                reasoning="Input contains only punctuation",
                scores={IntentType.CLARIFY: 0.1},
                processing_time=0.0,
            )

        return None

    def _apply_intent_specific_rules(
        self, result: ClassificationResult, user_input: str
    ) -> ClassificationResult:
        """Apply intent-specific rules to improve classification"""
        if not self.enable_intent_specific_rules:
            return result

        user_lower = user_input.lower().strip()

        # Intent-specific confidence adjustments
        if result.intent == IntentType.GREETING:
            # Boost confidence for clear greetings
            greeting_keywords = [
                "hello",
                "hi",
                "hey",
                "good morning",
                "good evening",
                "good afternoon",
            ]
            if any(keyword in user_lower for keyword in greeting_keywords):
                result.confidence = min(1.0, result.confidence + 0.1)

        elif result.intent == IntentType.CATEGORY_LIST:
            # Boost confidence for category requests
            category_keywords = [
                "category",
                "categories",
                "what do you have",
                "list",
                "show me",
            ]
            if any(keyword in user_lower for keyword in category_keywords):
                result.confidence = min(1.0, result.confidence + 0.1)

        elif result.intent == IntentType.PRODUCT_SEARCH:
            # Boost confidence for product searches
            product_keywords = ["looking for", "find", "show me", "want", "buy", "need"]
            if any(keyword in user_lower for keyword in product_keywords):
                result.confidence = min(1.0, result.confidence + 0.1)

        elif result.intent == IntentType.BUDGET_QUERY:
            # Boost confidence for budget queries
            budget_keywords = [
                "budget",
                "price",
                "cost",
                "under",
                "less than",
                "rupees",
                "rs",
            ]
            if any(keyword in user_lower for keyword in budget_keywords):
                result.confidence = min(1.0, result.confidence + 0.1)

        elif result.intent == IntentType.WARRANTY_QUERY:
            # Boost confidence for warranty queries
            warranty_keywords = ["warranty", "guarantee", "return", "refund"]
            if any(keyword in user_lower for keyword in warranty_keywords):
                result.confidence = min(1.0, result.confidence + 0.1)

        elif result.intent == IntentType.CLARIFY:
            # Boost confidence for clarification requests
            clarify_keywords = [
                "what do you mean",
                "clarify",
                "explain",
                "don't understand",
                "confused",
            ]
            if any(keyword in user_lower for keyword in clarify_keywords):
                result.confidence = min(1.0, result.confidence + 0.1)

        # Penalize low confidence for clear patterns
        if result.confidence < 0.3:
            # If we have clear patterns but low confidence, it might be a model issue
            if any(keyword in user_lower for keyword in ["hello", "hi", "hey"]):
                result.intent = IntentType.GREETING
                result.confidence = 0.8
                result.reasoning += " (corrected by intent-specific rules)"
            elif any(
                keyword in user_lower
                for keyword in ["category", "categories", "what do you have"]
            ):
                result.intent = IntentType.CATEGORY_LIST
                result.confidence = 0.8
                result.reasoning += " (corrected by intent-specific rules)"
            elif any(
                keyword in user_lower
                for keyword in [
                    "who is",
                    "what is",
                    "when was",
                    "where is",
                    "how old is",
                ]
            ):
                result.intent = IntentType.INVALID
                result.confidence = 0.9
                result.reasoning += (
                    " (corrected by intent-specific rules - out of domain question)"
                )

        # Override any intent for clear out-of-domain questions
        out_of_domain_keywords = [
            "who is",
            "what is",
            "when was",
            "where is",
            "how old is",
            "birthday of",
            "capital of",
            "population of",
            "stock price of",
            "weather in",
            "current time",
            "what time",
            "what date",
            "today is",
            "narendra modi",
            "modi",
            "president",
            "prime minister",
            "minister",
            "celebrity",
            "actor",
            "actress",
            "singer",
            "artist",
            "writer",
            "author",
            "scientist",
            "doctor",
            "teacher",
            "student",
        ]

        if any(keyword in user_lower for keyword in out_of_domain_keywords):
            result.intent = IntentType.INVALID
            result.confidence = 0.95
            result.reasoning += " (overridden by out-of-domain detection)"

        return result

    def classify_intent(self, user_message: str) -> ClassificationResult:
        """
        Classify intent using improved hybrid approach with confidence-based fallback and OpenAI as third fallback

        Args:
            user_message: Input text to classify

        Returns:
            ClassificationResult with intent classification results
        """
        start_time = time.time()
        self._total_queries += 1

        # Log the incoming query
        logger.info(
            f"🔍 INTENT CLASSIFICATION START: '{user_message[:50]}{'...' if len(user_message) > 50 else ''}'"
        )

        # Handle edge cases first
        edge_case_result = self._handle_edge_cases(user_message)
        if edge_case_result:
            logger.info(
                f"✅ EDGE CASE HANDLED: {edge_case_result.intent.value} (method: {edge_case_result.method})"
            )
            return edge_case_result

        # Try primary classifier first
        primary_result = None
        primary_confidence = 0.0
        if self.primary_classifier and self.primary_classifier.is_available():
            try:
                logger.info(
                    f"🎯 TRYING PRIMARY CLASSIFIER: {self.primary_classifier_type}"
                )
                primary_result = self.primary_classifier.classify_intent(user_message)
                primary_confidence = primary_result.confidence
                self._primary_used += 1
                logger.info(
                    f"✅ PRIMARY CLASSIFIER RESULT: {primary_result.intent.value} (confidence: {primary_confidence:.3f})"
                )
            except Exception as e:
                logger.warning(f"❌ PRIMARY CLASSIFIER FAILED: {e}")
                primary_result = None

        # Check if primary result meets confidence threshold
        if primary_result and primary_confidence >= self.confidence_threshold:
            # Apply intent-specific rules
            final_result = self._apply_intent_specific_rules(
                primary_result, user_message
            )
            final_result.method = f"improved_hybrid_{self.primary_classifier_type}"
            final_result.reasoning += (
                f" (primary classifier, confidence: {final_result.confidence:.3f})"
            )
            logger.info(
                f"🎉 FINAL RESULT - PRIMARY SUCCESS: {final_result.intent.value} (confidence: {final_result.confidence:.3f}, method: {final_result.method})"
            )
        else:
            logger.info(
                f"⚠️ PRIMARY CONFIDENCE TOO LOW: {primary_confidence:.3f} < {self.confidence_threshold}"
            )

            # Try fallback classifier
            fallback_result = None
            fallback_confidence = 0.0
            if self.fallback_classifier and self.fallback_classifier.is_available():
                try:
                    logger.info(
                        f"🔄 TRYING FALLBACK CLASSIFIER: {self.fallback_classifier_type}"
                    )
                    fallback_result = self.fallback_classifier.classify_intent(
                        user_message
                    )
                    fallback_confidence = fallback_result.confidence
                    self._fallback_used += 1
                    logger.info(
                        f"✅ FALLBACK CLASSIFIER RESULT: {fallback_result.intent.value} (confidence: {fallback_confidence:.3f})"
                    )

                    # Apply intent-specific rules to fallback result
                    final_result = self._apply_intent_specific_rules(
                        fallback_result, user_message
                    )
                    final_result.method = (
                        f"improved_hybrid_{self.fallback_classifier_type}_fallback"
                    )
                    final_result.reasoning += f" (fallback classifier, primary confidence: {primary_confidence:.3f})"
                    logger.info(
                        f"🎉 FINAL RESULT - FALLBACK SUCCESS: {final_result.intent.value} (confidence: {final_result.confidence:.3f}, method: {final_result.method})"
                    )
                except Exception as e:
                    logger.warning(f"❌ FALLBACK CLASSIFIER FAILED: {e}")
                    fallback_result = None

            # If fallback also failed or not available, try OpenAI as third fallback
            if (
                fallback_result is None
                and self.openai_classifier
                and self.openai_classifier.is_available()
            ):
                try:
                    logger.info("🔄 TRYING OPENAI AS THIRD FALLBACK CLASSIFIER")
                    openai_result = self.openai_classifier.classify_intent(user_message)
                    openai_confidence = openai_result.confidence
                    self._openai_fallback_used += 1
                    logger.info(
                        f"✅ OPENAI FALLBACK RESULT: {openai_result.intent.value} (confidence: {openai_confidence:.3f})"
                    )

                    # Apply intent-specific rules to OpenAI result
                    final_result = self._apply_intent_specific_rules(
                        openai_result, user_message
                    )
                    final_result.method = "improved_hybrid_openai_third_fallback"
                    final_result.reasoning += f" (OpenAI third fallback, primary confidence: {primary_confidence:.3f})"
                    logger.info(
                        f"🎉 FINAL RESULT - OPENAI FALLBACK SUCCESS: {final_result.intent.value} (confidence: {final_result.confidence:.3f}, method: {final_result.method})"
                    )
                except Exception as e:
                    logger.warning(f"❌ OPENAI FALLBACK CLASSIFIER FAILED: {e}")
                    # Last resort: return clarify intent
                    final_result = ClassificationResult(
                        intent=IntentType.CLARIFY,
                        confidence=0.1,
                        method=self.name,
                        reasoning=f"All classifiers failed: {e}",
                        scores={IntentType.CLARIFY: 0.1},
                        processing_time=time.time() - start_time,
                    )
                    logger.error(
                        f"💥 ALL CLASSIFIERS FAILED - USING CLARIFY INTENT: {e}"
                    )
            elif fallback_result is None:
                # No fallback available, use primary result if available
                if primary_result:
                    logger.info(
                        "⚠️ NO FALLBACK AVAILABLE - USING PRIMARY RESULT DESPITE LOW CONFIDENCE"
                    )
                    final_result = self._apply_intent_specific_rules(
                        primary_result, user_message
                    )
                    final_result.method = (
                        f"improved_hybrid_{self.primary_classifier_type}_no_fallback"
                    )
                    final_result.reasoning += " (no fallback available)"
                    logger.info(
                        f"🎉 FINAL RESULT - PRIMARY NO FALLBACK: {final_result.intent.value} (confidence: {final_result.confidence:.3f}, method: {final_result.method})"
                    )
                else:
                    final_result = ClassificationResult(
                        intent=IntentType.CLARIFY,
                        confidence=0.1,
                        method=self.name,
                        reasoning="No classifiers available",
                        scores={IntentType.CLARIFY: 0.1},
                        processing_time=time.time() - start_time,
                    )
                    logger.error("💥 NO CLASSIFIERS AVAILABLE - USING CLARIFY INTENT")

        # Track confidence issues
        if primary_result and primary_confidence < self.confidence_threshold:
            self._confidence_issues += 1

        # Update timing
        final_result.processing_time = time.time() - start_time
        self._total_time += final_result.processing_time

        # Log final summary
        logger.info(
            f"📊 CLASSIFICATION SUMMARY: Query='{user_message[:30]}{'...' if len(user_message) > 30 else ''}' | Intent={final_result.intent.value} | Confidence={final_result.confidence:.3f} | Method={final_result.method} | Time={final_result.processing_time*1000:.2f}ms"
        )

        return final_result

    def get_info(self) -> Dict[str, Any]:
        """Get information about the classifier"""
        base_info = super().get_info()
        base_info.update(
            {
                "confidence_threshold": self.confidence_threshold,
                "primary_classifier": self.primary_classifier_type,
                "fallback_classifier": self.fallback_classifier_type,
                "openai_fallback": self.openai_fallback,
                "enable_intent_specific_rules": self.enable_intent_specific_rules,
                "primary_available": self.primary_classifier is not None
                and self.primary_classifier.is_available(),
                "fallback_available": self.fallback_classifier is not None
                and self.fallback_classifier.is_available(),
                "openai_available": self.openai_classifier is not None
                and self.openai_classifier.is_available(),
            }
        )
        return base_info

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = (
            self._total_time / self._total_queries if self._total_queries > 0 else 0.0
        )
        primary_usage_rate = (
            self._primary_used / self._total_queries if self._total_queries > 0 else 0.0
        )
        fallback_usage_rate = (
            self._fallback_used / self._total_queries
            if self._total_queries > 0
            else 0.0
        )
        openai_usage_rate = (
            self._openai_fallback_used / self._total_queries
            if self._total_queries > 0
            else 0.0
        )
        confidence_issue_rate = (
            self._confidence_issues / self._total_queries
            if self._total_queries > 0
            else 0.0
        )

        return {
            "name": self.name,
            "avg_processing_time": avg_time,
            "total_queries": self._total_queries,
            "primary_usage_rate": primary_usage_rate,
            "fallback_usage_rate": fallback_usage_rate,
            "openai_fallback_usage_rate": openai_usage_rate,
            "confidence_issue_rate": confidence_issue_rate,
            "confidence_threshold": self.confidence_threshold,
        }

    def is_available(self) -> bool:
        """Check if the classifier is available"""
        return (
            (
                self.primary_classifier is not None
                and self.primary_classifier.is_available()
            )
            or (
                self.fallback_classifier is not None
                and self.fallback_classifier.is_available()
            )
            or (
                self.openai_classifier is not None
                and self.openai_classifier.is_available()
            )
        )

    def reset_stats(self):
        """Reset performance statistics"""
        self._total_queries = 0
        self._total_time = 0.0
        self._primary_used = 0
        self._fallback_used = 0
        self._openai_fallback_used = 0
        self._confidence_issues = 0
