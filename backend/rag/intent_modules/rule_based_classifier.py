"""
Rule-based Intent Classifier

This module provides intent classification using rule-based patterns and semantic similarity.
"""

import time
import logging
from typing import Dict, Any, Optional
from difflib import SequenceMatcher
from .base import BaseIntentClassifier, IntentType, ClassificationResult

logger = logging.getLogger(__name__)


class RuleBasedIntentClassifier(BaseIntentClassifier):
    """Rule-based Intent Classifier with Semantic Similarity"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize rule-based classifier

        Args:
            config: Configuration dictionary with:
                - patterns: Custom intent patterns (optional)
                - similarity_threshold: Minimum similarity for matches (default: 0.5)
        """
        super().__init__("rule_based", config)
        self.similarity_threshold = (
            config.get("similarity_threshold", 0.5) if config else 0.5
        )
        self._total_queries = 0
        self._total_time = 0.0

        # Define intent patterns
        self.intent_patterns = (
            config.get("patterns") if config else self._get_default_patterns()
        )

    def _get_default_patterns(self) -> Dict[IntentType, Dict[str, list]]:
        """Get default intent patterns"""
        return {
            IntentType.GREETING: {
                "greetings": [
                    "hi",
                    "hello",
                    "hey",
                    "good morning",
                    "good evening",
                    "good afternoon",
                    "how are you",
                    "what's up",
                ],
            },
            IntentType.HELP: {
                "help_queries": [
                    "help",
                    "how to use",
                    "how does this work",
                    "instructions",
                    "guide",
                    "tutorial",
                    "how to search",
                    "how to find",
                    "usage",
                    "manual",
                    "what can you do",
                    "capabilities",
                ],
            },
            IntentType.CATEGORY_LIST: {
                "category_queries": [
                    "what categories",
                    "what products",
                    "what do you have",
                    "show me categories",
                    "list categories",
                    "available products",
                    "available categories",
                    "what can i buy",
                    "what's available",
                    "list the categories",
                    "categories you have",
                    "what categories do you have",
                    "show me what you have",
                    "show me the list of categories available",
                    "list of categories available",
                    "categories available",
                    "what categories are available",
                    "give all categories",
                    "show all categories",
                    "list all categories",
                    "what are the categories",
                    "tell me the categories",
                    "display categories",
                    "show categories",
                    "get categories",
                    "category list",
                    "product categories",
                    "all categories",
                    "available options",
                    "what types",
                    "what kinds",
                    "what options",
                    "list all",
                    "show all",
                    "give all",
                    "display all",
                ],
            },
            IntentType.PRODUCT_SEARCH: {
                "specific_products": [
                    "show me",
                    "find",
                    "search for",
                    "looking for",
                    "want",
                    "need",
                    "bedside table",
                    "rug",
                    "fabric",
                    "furniture",
                    "decor",
                    "home",
                    "give me",
                    "get me",
                    "find me",
                    "show me all",
                    "list all",
                ],
                "category_product_patterns": [
                    "i want * products",
                    "show me * products",
                    "* products",
                ],
            },
            IntentType.BUDGET_QUERY: {
                "budget_queries": [
                    "under",
                    "less than",
                    "budget",
                    "price",
                    "cost",
                    "expensive",
                    "cheap",
                    "affordable",
                    "within budget",
                    "price range",
                ],
            },
            IntentType.PRODUCT_DETAIL: {
                "detail_queries": [
                    "tell me more",
                    "what are the",
                    "dimensions",
                    "material",
                    "color",
                    "size",
                    "specifications",
                    "details",
                    "information",
                    "describe",
                    "give me details",
                    "details of",
                    "tell me about",
                    "what about",
                    "show me details",
                    "get details",
                ],
                "specific_detail_patterns": [
                    "give me details of",
                    "details of",
                    "tell me about",
                    "what about",
                    "show me details of",
                    "get details of",
                ],
            },
            IntentType.WARRANTY_QUERY: {
                "warranty_queries": [
                    "warranty",
                    "guarantee",
                    "return policy",
                    "refund",
                    "exchange",
                    "repair",
                    "service",
                    "support",
                ],
            },
            IntentType.CLARIFY: {
                "clarification_queries": [
                    "what do you mean",
                    "can you clarify",
                    "i don't understand",
                    "explain",
                    "repeat",
                    "say that again",
                    "what",
                    "huh",
                    "pardon",
                    "sorry",
                    "i'm confused",
                    "not clear",
                    "unclear",
                ],
                "ambiguous_queries": [
                    "maybe",
                    "i think",
                    "possibly",
                    "not sure",
                    "unsure",
                    "dunno",
                    "don't know",
                    "i want something",
                    "show me",
                    "anything",
                    "whatever",
                ],
            },
            IntentType.INVALID: {
                "invalid_topics": [
                    "weather",
                    "news",
                    "sports",
                    "politics",
                    "music",
                    "movie",
                    "game",
                    "cooking",
                    "travel",
                    "health",
                    "fitness",
                    "education",
                    "work",
                    "football",
                    "basketball",
                    "tennis",
                    "baseball",
                    "soccer",
                    "hockey",
                ],
                "general_knowledge_questions": [
                    "who is narendra",
                    "who is modi",
                    "who is president",
                    "who is prime minister",
                    "what is capital",
                    "what is population",
                    "what is weather",
                    "what is stock price",
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
                ],
                "person_questions": [
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
                ],
            },
            IntentType.META: {
                # Legacy fallback patterns for backward compatibility
                "legacy_fallback": [
                    "who are you",
                    "about",
                    "catalog",
                    "capabilities",
                    "introduction",
                    "start",
                    "begin",
                ],
            },
            IntentType.PRODUCT: {
                # Legacy product patterns for backward compatibility
                "legacy_products": [
                    "bedside table",
                    "rug",
                    "fabric",
                    "furniture",
                    "decor",
                    "home",
                ],
            },
        }

    def _initialize(self) -> bool:
        """Initialize the rule-based classifier"""
        self._is_initialized = True
        logger.info("Rule-based classifier initialized successfully")
        return True

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def get_intent_confidence(self, user_message: str, intent: IntentType) -> float:
        """Calculate confidence score for intent classification"""
        user_lower = user_message.lower()
        patterns = self.intent_patterns.get(intent, {})

        max_confidence = 0.0
        for category, phrases in patterns.items():
            for phrase in phrases:
                # Handle wildcard patterns
                if "*" in phrase:
                    import re

                    # Convert wildcard pattern to regex
                    pattern_regex = phrase.replace("*", r"\w+")
                    if re.search(pattern_regex, user_lower):
                        confidence = 1.0
                    else:
                        # Semantic similarity for partial matches
                        confidence = self.calculate_similarity(user_lower, phrase)
                else:
                    # Exact match gets highest confidence
                    if phrase in user_lower:
                        confidence = 1.0
                    else:
                        # Semantic similarity for partial matches
                        confidence = self.calculate_similarity(user_lower, phrase)

                if confidence > max_confidence:
                    max_confidence = confidence

        return max_confidence

    def classify_intent(self, user_message: str) -> ClassificationResult:
        """Classify intent using rule-based approach"""
        start_time = time.time()

        # Safety nets for edge cases
        if not user_message or not user_message.strip():
            return ClassificationResult(
                intent=IntentType.CLARIFY,
                confidence=0.0,
                method=self.name,
                reasoning="Empty or whitespace-only input",
                scores={IntentType.CLARIFY: 0.0},
                processing_time=time.time() - start_time,
            )

        # Convert to lowercase for matching (declare outside condition blocks)
        user_lower = user_message.lower().strip()

        # Check for very short inputs
        if len(user_lower) < 2:
            return ClassificationResult(
                intent=IntentType.CLARIFY,
                confidence=0.1,
                method=self.name,
                reasoning="Input too short for meaningful classification",
                scores={IntentType.CLARIFY: 0.1},
                processing_time=time.time() - start_time,
            )

        # Check for only punctuation
        import re

        if re.match(r"^[^\w\s]*$", user_lower):
            return ClassificationResult(
                intent=IntentType.CLARIFY,
                confidence=0.1,
                method=self.name,
                reasoning="Input contains only punctuation",
                scores={IntentType.CLARIFY: 0.1},
                processing_time=time.time() - start_time,
            )

        # Special handling for "give me details of" pattern - should be PRODUCT_DETAIL
        if "give me details of" in user_lower or "details of" in user_lower:
            intent_scores = {intent: 0.0 for intent in IntentType}
            intent_scores[IntentType.PRODUCT_DETAIL] = 1.0
        else:
            # Calculate confidence for each intent
            intent_scores = {}
            for intent in IntentType:
                confidence = self.get_intent_confidence(user_message, intent)
                intent_scores[intent] = confidence

        # Find the intent with highest confidence
        best_intent = max(intent_scores, key=intent_scores.get)
        best_confidence = intent_scores[best_intent]

        # Priority handling for ties - INVALID should take precedence over META for invalid topics
        if best_confidence >= 0.8:
            user_lower = user_message.lower()
            invalid_topics = [
                "sports",
                "weather",
                "news",
                "politics",
                "music",
                "movie",
                "game",
                "cooking",
                "travel",
                "health",
                "fitness",
                "education",
                "work",
                "football",
                "basketball",
                "tennis",
                "baseball",
                "soccer",
                "hockey",
                "who is narendra",
                "who is modi",
                "who is president",
                "who is prime minister",
                "what is capital",
                "what is population",
                "what is weather",
                "what is stock price",
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
            has_invalid_topic = any(topic in user_lower for topic in invalid_topics)

            if has_invalid_topic and intent_scores.get(IntentType.INVALID, 0) >= 0.8:
                best_intent = IntentType.INVALID
                best_confidence = intent_scores[IntentType.INVALID]

        # Intelligent reasoning for ambiguous cases
        if best_confidence >= 0.8:
            # High confidence case - use the best match
            pass
        elif best_confidence >= 0.5:
            # Medium confidence - apply intelligent reasoning
            product_keywords = [
                "bedside",
                "table",
                "rug",
                "fabric",
                "sofa",
                "curtain",
                "furniture",
            ]
            category_keywords = ["category", "categories", "types", "kinds", "options"]

            user_lower = user_message.lower()
            has_product_keywords = any(
                keyword in user_lower for keyword in product_keywords
            )
            has_category_keywords = any(
                keyword in user_lower for keyword in category_keywords
            )

            # If user mentions specific products, prioritize product intent
            if has_product_keywords and not has_category_keywords:
                best_intent = IntentType.PRODUCT
                best_confidence = 0.9
            # If user asks about categories/types, prioritize meta intent
            elif has_category_keywords and not has_product_keywords:
                best_intent = IntentType.META
                best_confidence = 0.9
        else:
            # Low confidence - apply fallback logic
            # First check for invalid keywords to avoid misclassification
            invalid_keywords = [
                "weather",
                "news",
                "sports",
                "politics",
                "music",
                "movie",
                "game",
                "cooking",
                "travel",
                "health",
                "fitness",
                "education",
                "work",
                "football",
                "basketball",
                "tennis",
                "baseball",
                "soccer",
                "hockey",
                "who is narendra",
                "who is modi",
                "who is president",
                "who is prime minister",
                "what is capital",
                "what is population",
                "what is weather",
                "what is stock price",
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

            if any(keyword in user_lower for keyword in invalid_keywords):
                best_intent = IntentType.INVALID
                best_confidence = 0.8
            else:
                # Then check for category-related keywords
                category_keywords = [
                    "category",
                    "categories",
                    "list",
                    "show",
                    "all",
                    "available",
                    "what",
                    "display",
                    "get",
                    "types",
                    "kinds",
                ]
                if any(keyword in user_lower for keyword in category_keywords):
                    best_intent = IntentType.META
                    best_confidence = 0.8
                else:
                    # If still low confidence, ask for clarification
                    best_intent = IntentType.CLARIFY
                    best_confidence = 0.3

        processing_time = time.time() - start_time

        # Update statistics
        self._total_queries += 1
        self._total_time += processing_time

        return ClassificationResult(
            intent=best_intent,
            confidence=best_confidence,
            method=self.name,
            reasoning=f"Rule-based classification: {best_intent} with confidence {best_confidence:.3f}",
            scores=intent_scores,
            processing_time=processing_time,
            metadata={
                "patterns_used": len(self.intent_patterns),
                "similarity_threshold": self.similarity_threshold,
            },
        )

    def is_available(self) -> bool:
        """Check if rule-based classifier is available"""
        return self._is_initialized

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = (
            self._total_time / self._total_queries if self._total_queries > 0 else 0.0
        )

        patterns_count = 0
        if self.intent_patterns:
            patterns_count = sum(
                len(patterns) for patterns in self.intent_patterns.values()
            )

        return {
            "name": self.name,
            "avg_processing_time": avg_time,
            "total_queries": self._total_queries,
            "success_rate": 1.0,  # Rule-based is always available
            "patterns_count": patterns_count,
        }

    def add_pattern(self, intent: IntentType, category: str, pattern: str):
        """Add a new pattern to the classifier"""
        if intent not in self.intent_patterns:
            self.intent_patterns[intent] = {}

        if category not in self.intent_patterns[intent]:
            self.intent_patterns[intent][category] = []

        self.intent_patterns[intent][category].append(pattern)
        logger.info(f"Added pattern '{pattern}' to {intent}.{category}")

    def remove_pattern(self, intent: IntentType, category: str, pattern: str):
        """Remove a pattern from the classifier"""
        if (
            intent in self.intent_patterns
            and category in self.intent_patterns[intent]
            and pattern in self.intent_patterns[intent][category]
        ):

            self.intent_patterns[intent][category].remove(pattern)
            logger.info(f"Removed pattern '{pattern}' from {intent}.{category}")
