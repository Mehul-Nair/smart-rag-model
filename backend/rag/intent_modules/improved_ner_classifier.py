#!/usr/bin/env python3
"""
Improved NER Classifier with better product name handling
"""

import os
import time
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import onnxruntime as ort
from transformers import AutoTokenizer
import json


@dataclass
class Entity:
    """Represents a detected entity"""

    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float


@dataclass
class NERResult:
    """Result of NER classification"""

    entities: List[Entity]
    slots: Dict[str, str]  # entity_type -> extracted_value


class ImprovedNERClassifier:
    """Improved NER classifier with better product name handling"""

    def __init__(self, model_path: str = "trained_deberta_ner_model"):
        self.model_path = model_path
        self.tokenizer = None
        self.ort_session = None
        self.label2id = {}
        self.id2label = {}
        self.is_initialized = False
        self.total_queries = 0
        self.total_time = 0.0

        # Known product names from the dataset
        self.known_product_names = [
            "city lights",
            "lighting up lisbon",
            "lights out gold",
            "drop dead gorgeous",
            "time to shine",
            "keeper of my heart",
            "bolt",
            "positive energy",
            "purity",
            "skyfall",
            "captured",
            "mini mammoth",
            "cloudy bay",
            "undivided",
            "party favor",
            "word play",
            "wish you well",
            "oakridge",
            "sunlit",
            "insightful",
        ]

        # Known brands
        self.known_brands = [
            "pure royale",
            "white teak",
            "asian paints",
            "bathsense",
            "ador",
            "royale",
        ]

        self._initialize()

    def _initialize(self) -> bool:
        """Initialize the ONNX model and tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Load ONNX session
            onnx_path = os.path.join(self.model_path, "model.onnx")
            if not os.path.exists(onnx_path):
                print(f"❌ ONNX model not found at {onnx_path}")
                return False

            self.ort_session = ort.InferenceSession(onnx_path)

            # Load label mapping
            label_mapping_path = os.path.join(self.model_path, "label_mapping.json")
            if os.path.exists(label_mapping_path):
                with open(label_mapping_path, "r") as f:
                    self.label2id = json.load(f)
                self.id2label = {v: k for k, v in self.label2id.items()}
            else:
                print(f"❌ Label mapping not found at {label_mapping_path}")
                return False

            self.is_initialized = True
            print(f"✅ Improved NER Classifier initialized successfully")
            return True

        except Exception as e:
            print(f"❌ Error initializing Improved NER Classifier: {e}")
            return False

    def extract_entities(self, text: str) -> NERResult:
        """Extract entities with improved product name handling"""
        start_time = time.time()

        try:
            # First, check for known product names
            product_name = self._extract_known_product_name(text)
            if product_name:
                # If we found a known product name, create a PRODUCT_NAME entity
                entities = [
                    Entity(
                        text=product_name,
                        entity_type="PRODUCT_NAME",
                        start_pos=0,
                        end_pos=len(product_name),
                        confidence=0.95,  # High confidence for known products
                    )
                ]
                slots = {"PRODUCT_NAME": product_name}

                self.total_queries += 1
                self.total_time += time.time() - start_time

                return NERResult(entities=entities, slots=slots)

            # Check for known brands
            brand = self._extract_known_brand(text)
            if brand:
                # If we found a known brand, create a BRAND entity
                entities = [
                    Entity(
                        text=brand,
                        entity_type="BRAND",
                        start_pos=0,
                        end_pos=len(brand),
                        confidence=0.95,  # High confidence for known brands
                    )
                ]
                slots = {"BRAND": brand}

                self.total_queries += 1
                self.total_time += time.time() - start_time

                return NERResult(entities=entities, slots=slots)

            # Fall back to ONNX model extraction
            return self._extract_with_onnx_model(text, start_time)

        except Exception as e:
            print(f"❌ Error in improved NER extraction: {e}")
            return NERResult(entities=[], slots={})

    def _extract_known_product_name(self, text: str) -> Optional[str]:
        """Extract known product names from text"""
        text_lower = text.lower()

        for product_name in self.known_product_names:
            if product_name in text_lower:
                # Find the exact case in the original text
                start_idx = text_lower.find(product_name)
                if start_idx != -1:
                    # Extract the full product name (up to next punctuation)
                    end_idx = text.find("(", start_idx)
                    if end_idx == -1:
                        end_idx = text.find(")", start_idx)
                    if end_idx == -1:
                        end_idx = len(text)

                    return text[start_idx:end_idx].strip()

        return None

    def _extract_known_brand(self, text: str) -> Optional[str]:
        """Extract known brands from text"""
        text_lower = text.lower()

        for brand in self.known_brands:
            if brand in text_lower:
                # Find the exact case in the original text
                start_idx = text_lower.find(brand)
                if start_idx != -1:
                    return text[start_idx : start_idx + len(brand)]

        return None

    def _extract_with_onnx_model(self, text: str, start_time: float) -> NERResult:
        """Extract entities using the ONNX model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )

            # Convert to numpy arrays with correct data types
            input_ids = inputs["input_ids"].numpy().astype(np.int64)
            attention_mask = inputs["attention_mask"].numpy().astype(np.float32)

            # Run ONNX inference
            ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            outputs = self.ort_session.run(None, ort_inputs)
            logits = outputs[0]

            # Get predictions
            predictions = np.argmax(logits, axis=2)
            probabilities = self._softmax(logits, axis=2)

            # Decode entities
            entities = self._decode_entities_simple(
                text, predictions[0], input_ids[0], probabilities[0]
            )

            # Convert to slots
            slots = {}
            for entity in entities:
                entity_type = entity.entity_type
                if entity_type not in slots:
                    slots[entity_type] = entity.text
                else:
                    # If multiple entities of same type, concatenate
                    slots[entity_type] = f"{slots[entity_type]}, {entity.text}"

            # Update performance stats
            self.total_queries += 1
            self.total_time += time.time() - start_time

            return NERResult(entities=entities, slots=slots)

        except Exception as e:
            print(f"❌ Error in ONNX extraction: {e}")
            return NERResult(entities=[], slots={})

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax values for each set of scores in x"""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def _decode_entities_simple(
        self,
        text: str,
        predictions: np.ndarray,
        input_ids: np.ndarray,
        probabilities: np.ndarray,
    ) -> List[Entity]:
        """Simple entity decoding without word_ids"""
        entities = []
        current_entity = None
        current_confidences = []
        current_tokens = []

        # Simple approach: iterate through predictions directly
        for i in range(len(predictions)):
            label_id = predictions[i]
            label = self.id2label.get(label_id, "O")
            confidence = probabilities[i][label_id]

            # Get token text
            token_id = input_ids[i]
            token_text = self.tokenizer.convert_ids_to_tokens([token_id])[0]

            # Skip special tokens
            if token_text in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            # Clean up token text (remove prefixes)
            if token_text.startswith("▁"):
                token_text = token_text[1:]  # Remove SentencePiece prefix
            if token_text.startswith("##"):
                token_text = token_text[2:]  # Remove BERT-style prefix

            # Handle BIO tagging
            if label.startswith("B-"):
                # Start of new entity
                if current_entity:
                    # Save previous entity
                    entity_text = " ".join(current_tokens).strip()
                    if entity_text:
                        entities.append(
                            Entity(
                                text=entity_text,
                                entity_type=current_entity,
                                start_pos=0,  # Simplified position tracking
                                end_pos=len(entity_text),
                                confidence=np.mean(current_confidences),
                            )
                        )

                # Start new entity
                current_entity = label[2:]  # Remove "B-" prefix
                current_tokens = [token_text]
                current_confidences = [confidence]

            elif (
                label.startswith("I-")
                and current_entity
                and label[2:] == current_entity
            ):
                # Continue current entity
                current_tokens.append(token_text)
                current_confidences.append(confidence)

            else:
                # End of entity or O tag
                if current_entity:
                    # Save current entity
                    entity_text = " ".join(current_tokens).strip()
                    if entity_text:
                        entities.append(
                            Entity(
                                text=entity_text,
                                entity_type=current_entity,
                                start_pos=0,  # Simplified position tracking
                                end_pos=len(entity_text),
                                confidence=np.mean(current_confidences),
                            )
                        )

                # Reset
                current_entity = None
                current_tokens = []
                current_confidences = []

        # Handle any remaining entity
        if current_entity:
            entity_text = " ".join(current_tokens).strip()
            if entity_text:
                entities.append(
                    Entity(
                        text=entity_text,
                        entity_type=current_entity,
                        start_pos=0,
                        end_pos=len(entity_text),
                        confidence=np.mean(current_confidences),
                    )
                )

        return entities

    def extract_slots(self, text: str) -> Dict[str, str]:
        """Extract slots from text"""
        result = self.extract_entities(text)
        return result.slots

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = self.total_time / self.total_queries if self.total_queries > 0 else 0
        return {
            "total_queries": self.total_queries,
            "total_time": self.total_time,
            "average_time": avg_time,
            "queries_per_second": 1.0 / avg_time if avg_time > 0 else 0,
        }


# Global instance
_improved_ner_classifier = None


def get_improved_ner_classifier() -> ImprovedNERClassifier:
    """Get or create the global improved NER classifier instance"""
    global _improved_ner_classifier
    if _improved_ner_classifier is None:
        # Get the full path to the model directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels
        model_path = os.path.join(backend_dir, "trained_deberta_ner_model")
        _improved_ner_classifier = ImprovedNERClassifier(model_path)
    return _improved_ner_classifier


def extract_slots_from_text_improved(text: str) -> Dict[str, str]:
    """Extract slots from text using the improved NER classifier"""
    classifier = get_improved_ner_classifier()
    return classifier.extract_slots(text)
