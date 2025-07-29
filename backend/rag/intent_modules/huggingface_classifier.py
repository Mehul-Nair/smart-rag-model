"""
HuggingFace Transformers Intent Classifier

This module provides intent classification using HuggingFace transformers.
"""

import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from typing import Dict, Any, Optional, List
import time
import numpy as np
from .base import BaseIntentClassifier, IntentType, ClassificationResult


class HuggingFaceIntentClassifier(BaseIntentClassifier):
    """HuggingFace-based intent classifier using fine-tuned DeBERTa"""

    def __init__(self, model_path: str = "./trained_deberta_model", **kwargs):
        """
        Initialize the HuggingFace intent classifier

        Args:
            model_path: Path to the trained model directory
            OR
            config: Configuration dictionary with 'model_name' or 'model_path'
        """
        # Support both string and dict for backward compatibility
        if isinstance(model_path, dict):
            config = model_path
            model_path = (
                config.get("model_name")
                or config.get("model_path")
                or "./trained_intent_model"
            )
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        # Updated intent mapping to match the trained model
        self.intent_mapping = {
            0: "BUDGET_QUERY",
            1: "CATEGORY_LIST",
            2: "CLARIFY",
            3: "GREETING",
            4: "HELP",
            5: "INVALID",
            6: "PRODUCT_DETAIL",
            7: "PRODUCT_SEARCH",
            8: "WARRANTY_QUERY",
        }
        self.reverse_mapping = {v: k for k, v in self.intent_mapping.items()}
        self.performance_stats = {
            "total_queries": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "successful_predictions": 0,
        }

        # Call parent constructor
        super().__init__("huggingface", {"model_path": self.model_path})

    def _initialize(self) -> bool:
        """Initialize the tokenizer and model"""
        try:
            print(f"🔧 Loading trained model from: {self.model_path}")
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_path)
            self.model = DebertaV2ForSequenceClassification.from_pretrained(
                self.model_path
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ HuggingFace model loaded successfully on {self.device}")
            self._is_initialized = True
            return True
        except Exception as e:
            print(f"⚠️ Could not load trained model from {self.model_path}: {e}")
            print("🔄 Falling back to base model...")
            try:
                # Fallback to base model if trained model not found
                self.tokenizer = DebertaV2Tokenizer.from_pretrained(
                    "microsoft/deberta-v3-small"
                )
                self.model = DebertaV2ForSequenceClassification.from_pretrained(
                    "microsoft/deberta-v3-small",
                    num_labels=9,  # Now 9 labels including CLARIFY
                )
                self.model.to(self.device)
                self.model.eval()
                print(f"✅ Base model loaded successfully on {self.device}")
                self._is_initialized = True
                return True
            except Exception as e2:
                print(f"❌ Failed to load base model: {e2}")
                self._is_initialized = False
                return False

    def classify_intent(self, user_message: str) -> ClassificationResult:
        """
        Classify the intent of the given text

        Args:
            user_message: Input text to classify

        Returns:
            ClassificationResult with intent classification results
        """
        start_time = time.time()

        if not self.is_available():
            return ClassificationResult(
                intent=IntentType.CLARIFY,
                confidence=0.1,
                method=self.name,
                reasoning="HuggingFace model not available",
                scores={IntentType.CLARIFY: 0.1},
                processing_time=0.0,
            )

        try:
            # Tokenize the input text
            inputs = self.tokenizer(
                user_message,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            # Map to intent
            predicted_intent_name = self.intent_mapping.get(predicted_class, "INVALID")
            intent = IntentType(predicted_intent_name.lower())

            # Update performance stats
            end_time = time.time()
            processing_time = end_time - start_time
            self.performance_stats["total_queries"] += 1
            self.performance_stats["total_time"] += processing_time
            self.performance_stats["avg_time"] = (
                self.performance_stats["total_time"]
                / self.performance_stats["total_queries"]
            )
            self.performance_stats["successful_predictions"] += 1

            # Create scores dictionary
            scores = {}
            for i, intent_name in enumerate(self.intent_mapping.values()):
                scores[IntentType(intent_name.lower())] = probabilities[0][i].item()

            return ClassificationResult(
                intent=intent,
                confidence=confidence,
                method=self.name,
                reasoning=f"HuggingFace model classification: {predicted_intent_name} with confidence {confidence:.3f}",
                scores=scores,
                processing_time=processing_time,
                metadata={
                    "model_path": self.model_path,
                    "device": str(self.device),
                    "predicted_class": predicted_class,
                },
            )

        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            self.performance_stats["total_queries"] += 1
            self.performance_stats["total_time"] += processing_time
            self.performance_stats["avg_time"] = (
                self.performance_stats["total_time"]
                / self.performance_stats["total_queries"]
            )

            print(f"❌ Error in HuggingFace classification: {e}")
            return ClassificationResult(
                intent=IntentType.CLARIFY,
                confidence=0.1,
                method=self.name,
                reasoning=f"HuggingFace classification failed: {e}",
                scores={IntentType.CLARIFY: 0.1},
                processing_time=processing_time,
            )

    def get_info(self) -> Dict[str, Any]:
        """Get information about the classifier"""
        base_info = super().get_info()
        base_info.update(
            {
                "model_path": self.model_path,
                "device": str(self.device),
                "intent_mapping": self.intent_mapping,
                "performance_stats": self.performance_stats,
                "model_type": "DistilBERT (Fine-tuned)",
                "supports_batch": False,
            }
        )
        return base_info

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        base_stats = super().get_performance_stats()
        base_stats.update(
            {
                "avg_processing_time": self.performance_stats["avg_time"],
                "total_queries": self.performance_stats["total_queries"],
                "success_rate": self.performance_stats["successful_predictions"]
                / max(1, self.performance_stats["total_queries"]),
                "model_path": self.model_path,
                "device": str(self.device),
            }
        )
        return base_stats

    def reset_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            "total_queries": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "successful_predictions": 0,
        }

    def is_available(self) -> bool:
        """Check if the classifier is available"""
        return (
            self._is_initialized
            and self.model is not None
            and self.tokenizer is not None
        )
