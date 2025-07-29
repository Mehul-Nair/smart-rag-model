"""
OpenAI Fine-tuned Model Intent Classifier

This module provides intent classification using OpenAI's fine-tuned models.
"""

import time
import logging
from typing import Dict, Any, Optional
from .base import BaseIntentClassifier, IntentType, ClassificationResult

logger = logging.getLogger(__name__)


class OpenAIIntentClassifier(BaseIntentClassifier):
    """OpenAI Fine-tuned Model Intent Classifier"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenAI classifier

        Args:
            config: Configuration dictionary with:
                - model_name: Fine-tuned model name
                - api_key: OpenAI API key
                - temperature: Model temperature (default: 0)
                - max_tokens: Max tokens for response (default: 10)
        """
        self.model_name = config.get("model_name") if config else None
        self.api_key = config.get("api_key") if config else None
        self.temperature = config.get("temperature", 0)
        self.max_tokens = config.get("max_tokens", 10)
        self._model = None
        self._total_queries = 0
        self._total_time = 0.0

        super().__init__("openai", config)

    def _initialize(self) -> bool:
        """Initialize the OpenAI model"""
        if not self.model_name or not self.api_key:
            logger.warning(
                "OpenAI model not configured - missing model_name or api_key"
            )
            return False

        try:
            from langchain_openai import ChatOpenAI
            from pydantic import SecretStr

            self._model = ChatOpenAI(
                model=self.model_name,
                api_key=SecretStr(self.api_key),
                temperature=self.temperature,
            )

            self._is_initialized = True
            logger.info(f"OpenAI fine-tuned model initialized: {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI model: {e}")
            return False

    def classify_intent(self, user_message: str) -> ClassificationResult:
        """Classify intent using OpenAI fine-tuned model"""
        start_time = time.time()

        if not self.is_available():
            return ClassificationResult(
                intent=IntentType.CLARIFY,
                confidence=0.1,
                method=self.name,
                reasoning="OpenAI model not available",
                scores={IntentType.CLARIFY: 0.1},
                processing_time=0.0,
            )

        try:
            # Create classification prompt
            prompt = f"""
            Classify the user's intent into exactly one of these categories:
            - META: Categories, greetings, help, about assistant, list categories
            - PRODUCT: Product searches, specific items, find products, show me products
            - INVALID: Unrelated topics (weather, sports, politics, etc.)
            - CLARIFY: Ambiguous or unclear requests
            
            User message: "{user_message}"
            
            Respond with ONLY the intent category and confidence (0.0-1.0):
            INTENT: [category]
            CONFIDENCE: [score]
            """

            # Get model prediction
            response = self._model.predict(prompt)

            # Parse the response
            intent, confidence = self._parse_model_response(response)

            processing_time = time.time() - start_time

            # Update statistics
            self._total_queries += 1
            self._total_time += processing_time

            return ClassificationResult(
                intent=intent,
                confidence=confidence,
                method=self.name,
                reasoning=f"OpenAI fine-tuned model classification: {intent} with confidence {confidence:.3f}",
                scores={intent: confidence},
                processing_time=processing_time,
                metadata={"model_name": self.model_name, "response": response},
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in OpenAI classification: {e}")

            return ClassificationResult(
                intent=IntentType.CLARIFY,
                confidence=0.1,
                method=self.name,
                reasoning=f"OpenAI classification failed: {e}",
                scores={IntentType.CLARIFY: 0.1},
                processing_time=processing_time,
            )

    def _parse_model_response(self, response: str) -> tuple[IntentType, float]:
        """Parse the model response to extract intent and confidence"""
        try:
            lines = response.strip().split("\n")
            intent_line = None
            confidence_line = None

            for line in lines:
                if line.startswith("INTENT:"):
                    intent_line = line
                elif line.startswith("CONFIDENCE:"):
                    confidence_line = line

            if not intent_line or not confidence_line:
                raise ValueError("Invalid response format")

            # Extract intent
            intent_str = intent_line.split(":", 1)[1].strip().upper()
            intent = IntentType(intent_str.lower())

            # Extract confidence
            confidence_str = confidence_line.split(":", 1)[1].strip()
            confidence = float(confidence_str)

            # Validate confidence
            if not 0.0 <= confidence <= 1.0:
                confidence = max(0.0, min(1.0, confidence))

            return intent, confidence

        except Exception as e:
            logger.error(f"Error parsing model response: {e}")
            # Fallback to META with low confidence
            return IntentType.META, 0.3

    def is_available(self) -> bool:
        """Check if OpenAI model is available"""
        return self._is_initialized and self._model is not None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = (
            self._total_time / self._total_queries if self._total_queries > 0 else 0.0
        )

        return {
            "name": self.name,
            "avg_processing_time": avg_time,
            "total_queries": self._total_queries,
            "success_rate": 1.0,  # Assuming all queries succeed if model is available
            "model_name": self.model_name,
        }
