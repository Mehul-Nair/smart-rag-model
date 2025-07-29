#!/usr/bin/env python3
"""
Dynamic NER Classifier - Learns product names from actual dataset
"""

import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import onnxruntime as ort
from transformers import AutoTokenizer
import json
import re
from collections import defaultdict


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
    slots: Dict[str, str]


class DynamicNERClassifier:
    """Dynamic NER classifier that learns from actual dataset"""

    def __init__(self, model_path: str = "trained_deberta_ner_model"):
        self.model_path = model_path
        self.tokenizer = None
        self.ort_session = None
        self.label2id = {}
        self.id2label = {}
        self.is_initialized = False
        self.total_queries = 0
        self.total_time = 0.0

        # Dynamic product database
        self.product_names: Set[str] = set()
        self.brand_names: Set[str] = set()
        self.product_name_patterns: Dict[str, List[str]] = defaultdict(list)

        # Load dynamic data
        self._load_product_database()
        self._initialize()

    def _load_product_database(self):
        """Load product names and brands from the actual dataset"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(os.path.dirname(current_dir))
            excel_path = os.path.join(backend_dir, "data", "BH_PD.xlsx")

            if os.path.exists(excel_path):
                # Load only necessary columns for better performance
                df = pd.read_excel(excel_path, usecols=["title", "brand_name"])

                # Extract product names from titles - optimized
                product_names_set = set()
                for title in df["title"].dropna():
                    product_name = self._extract_main_product_name(str(title))
                    if product_name:
                        product_names_set.add(product_name.lower())

                self.product_names = product_names_set

                # Extract brands - optimized
                brand_names_set = set()
                for brand in df["brand_name"].dropna():
                    brand_str = str(brand).strip()
                    if brand_str and brand_str.lower() != "nan":
                        brand_names_set.add(brand_str.lower())

                self.brand_names = brand_names_set

                print(
                    f"✅ Loaded {len(self.product_names)} product names and {len(self.brand_names)} brands"
                )

            else:
                print(f"⚠️ Excel file not found at {excel_path}")

        except Exception as e:
            print(f"❌ Error loading product database: {e}")

    def _extract_main_product_name(self, title: str) -> Optional[str]:
        """Extract the main product name from a title"""
        # Remove parentheses and everything after
        main_part = re.split(r"[\(\)]", title)[0].strip()

        # Return the full product name as it appears in the database
        # This preserves the complete product name like "ROOHE - Wool Hand Tufted Rug"
        return main_part if main_part else None

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
            print(f"✅ Dynamic NER Classifier initialized successfully")
            return True

        except Exception as e:
            print(f"❌ Error initializing Dynamic NER Classifier: {e}")
            return False

    def extract_entities(self, text: str) -> NERResult:
        """Extract entities with dynamic product name recognition"""
        start_time = time.time()

        try:
            # First, try dynamic product name recognition
            product_name = self._find_product_name_in_text(text)
            if product_name:
                print(f"🎯 Dynamic NER found product name: {product_name}")
                entities = [
                    Entity(
                        text=product_name,
                        entity_type="PRODUCT_NAME",
                        start_pos=0,
                        end_pos=len(product_name),
                        confidence=0.95,
                    )
                ]
                slots = {"PRODUCT_NAME": product_name}

                self.total_queries += 1
                self.total_time += time.time() - start_time

                return NERResult(entities=entities, slots=slots)

            # Try dynamic brand recognition
            brand = self._find_brand_in_text(text)
            if brand:
                entities = [
                    Entity(
                        text=brand,
                        entity_type="BRAND",
                        start_pos=0,
                        end_pos=len(brand),
                        confidence=0.95,
                    )
                ]
                slots = {"BRAND": brand}

                self.total_queries += 1
                self.total_time += time.time() - start_time

                return NERResult(entities=entities, slots=slots)

            # Fall back to ONNX model
            return self._extract_with_onnx_model(text, start_time)

        except Exception as e:
            print(f"❌ Error in dynamic NER extraction: {e}")
            return NERResult(entities=[], slots={})

    def _find_product_name_in_text(self, text: str) -> Optional[str]:
        """Dynamically find product names in text"""
        text_lower = text.lower()

        # Normalize the input text for better matching
        # Remove parentheses and normalize spaces
        normalized_text = re.sub(r"[\(\)]", " ", text_lower)
        normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

        # Try exact matches first
        for product_name in self.product_names:
            # Normalize product name for comparison
            normalized_product = re.sub(r"\s+", " ", product_name).strip()

            if normalized_product in normalized_text:
                # Find the exact case in original text
                start_idx = text_lower.find(product_name)
                if start_idx != -1:
                    # Extract the full product name (up to next punctuation)
                    end_idx = text.find("(", start_idx)
                    if end_idx == -1:
                        end_idx = text.find(")", start_idx)
                    if end_idx == -1:
                        end_idx = len(text)

                    return text[start_idx:end_idx].strip()

        # Try normalized matching
        for product_name in self.product_names:
            normalized_product = re.sub(r"\s+", " ", product_name).strip()

            if normalized_product in normalized_text:
                # Find the closest match in original text
                # Look for the main parts of the product name
                product_words = normalized_product.split()
                if len(product_words) >= 3:  # At least 3 words for a good match
                    # Try to find the main product name part
                    main_part = " ".join(product_words[:3])  # First 3 words
                    if main_part in normalized_text:
                        # Find in original text
                        start_idx = text_lower.find(main_part)
                        if start_idx != -1:
                            # Extract up to the end or next punctuation
                            end_idx = text.find("(", start_idx)
                            if end_idx == -1:
                                end_idx = text.find(")", start_idx)
                            if end_idx == -1:
                                end_idx = len(text)

                            return text[start_idx:end_idx].strip()

        # Try fuzzy matching for partial matches
        words = text_lower.split()
        for i in range(len(words)):
            for j in range(
                i + 1, min(i + 5, len(words) + 1)
            ):  # Try up to 4-word combinations
                phrase = " ".join(words[i:j])
                if phrase in self.product_names:
                    # Find in original text
                    start_idx = text_lower.find(phrase)
                    if start_idx != -1:
                        return text[start_idx : start_idx + len(phrase)]

        return None

    def _find_brand_in_text(self, text: str) -> Optional[str]:
        """Dynamically find brands in text"""
        text_lower = text.lower()

        for brand in self.brand_names:
            if brand in text_lower:
                # Find exact case in original text
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

            # Convert to numpy arrays
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
                    slots[entity_type] = f"{slots[entity_type]}, {entity.text}"

            # Update performance stats
            self.total_queries += 1
            self.total_time += time.time() - start_time

            return NERResult(entities=entities, slots=slots)

        except Exception as e:
            print(f"❌ Error in ONNX extraction: {e}")
            return NERResult(entities=[], slots={})

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax values"""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def _decode_entities_simple(
        self,
        text: str,
        predictions: np.ndarray,
        input_ids: np.ndarray,
        probabilities: np.ndarray,
    ) -> List[Entity]:
        """Decode entities from model predictions"""
        entities = []
        current_entity = None
        current_confidences = []
        current_tokens = []

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

            # Clean up token text
            if token_text.startswith("▁"):
                token_text = token_text[1:]
            if token_text.startswith("##"):
                token_text = token_text[2:]

            # Handle BIO tagging
            if label.startswith("B-"):
                # Save previous entity
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

                # Start new entity
                current_entity = label[2:]
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
            "product_names_loaded": len(self.product_names),
            "brands_loaded": len(self.brand_names),
        }

    def add_product_name(self, product_name: str):
        """Dynamically add a new product name"""
        self.product_names.add(product_name.lower())
        print(f"✅ Added product name: {product_name}")

    def add_brand(self, brand: str):
        """Dynamically add a new brand"""
        self.brand_names.add(brand.lower())
        print(f"✅ Added brand: {brand}")


# Global instance
_dynamic_ner_classifier = None
_initialization_lock = False


def get_dynamic_ner_classifier() -> DynamicNERClassifier:
    """Get or create the global dynamic NER classifier instance"""
    global _dynamic_ner_classifier, _initialization_lock

    if _dynamic_ner_classifier is None and not _initialization_lock:
        _initialization_lock = True
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(os.path.dirname(current_dir))
            model_path = os.path.join(backend_dir, "trained_deberta_ner_model")
            _dynamic_ner_classifier = DynamicNERClassifier(model_path)
        finally:
            _initialization_lock = False

    return _dynamic_ner_classifier


def extract_slots_from_text_dynamic(text: str) -> Dict[str, str]:
    """Extract slots from text using the dynamic NER classifier"""
    classifier = get_dynamic_ner_classifier()
    return classifier.extract_slots(text)
