#!/usr/bin/env python3
"""
ONNX NER Classifier for Slot Extraction

This module provides Named Entity Recognition (NER) functionality using ONNX Runtime
to extract slots like PRODUCT_TYPE, ROOM_TYPE, BUDGET_RANGE, etc.
from user queries for the conversational agent.
"""

import json
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import onnxruntime as ort


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


class ONNXNERClassifier:
    """ONNX NER classifier for slot extraction"""

    def __init__(self, model_path: str = "trained_deberta_ner_model"):
        """Initialize the ONNX NER classifier"""
        self.model_path = model_path
        self.tokenizer = None
        self.ort_session = None
        self.id2label = {}
        self.label2id = {}
        self.is_initialized = False

        # Performance tracking
        self.total_queries = 0
        self.total_time = 0.0

        # Initialize the model
        self.is_initialized = self._initialize()

    def _initialize(self) -> bool:
        """Initialize the ONNX model and tokenizer"""
        try:
            print(f"🔧 Initializing ONNX NER Classifier from {self.model_path}")

            # Load label mapping
            label_mapping_path = os.path.join(self.model_path, "label_mapping.json")
            if os.path.exists(label_mapping_path):
                with open(label_mapping_path, "r") as f:
                    label_mapping = json.load(f)
                    # The label_mapping.json has label2id format, convert to id2label
                    self.label2id = label_mapping
                    self.id2label = {v: k for k, v in label_mapping.items()}
            else:
                print("⚠️ No label mapping found, using default mapping")
                # Fallback to default mapping
                self.id2label = {
                    0: "O",
                    1: "B-PRODUCT_TYPE",
                    2: "I-PRODUCT_TYPE",
                    3: "B-MATERIAL",
                    4: "I-MATERIAL",
                    5: "B-COLOR",
                    6: "I-COLOR",
                    7: "B-SIZE",
                    8: "I-SIZE",
                    9: "B-BRAND",
                    10: "I-BRAND",
                    11: "B-STYLE",
                    12: "I-STYLE",
                    13: "B-ROOM",
                    14: "I-ROOM",
                    15: "B-BUDGET",
                    16: "I-BUDGET",
                    17: "B-PRODUCT_NAME",
                    18: "I-PRODUCT_NAME",
                }
                self.label2id = {v: k for k, v in self.id2label.items()}

            # Load tokenizer
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Load ONNX model
            onnx_path = os.path.join(self.model_path, "model.onnx")
            if os.path.exists(onnx_path):
                self.ort_session = ort.InferenceSession(onnx_path)
                print(f"✅ ONNX model loaded from {onnx_path}")
            else:
                print(f"❌ ONNX model not found at {onnx_path}")
                print("Please run the conversion script first")
                return False

            print(f"✅ ONNX NER Classifier initialized successfully")
            print(f"📊 Available labels: {list(self.label2id.keys())}")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize ONNX NER Classifier: {e}")
            return False

    def extract_entities(self, text: str) -> NERResult:
        """Extract entities from text using ONNX model"""
        start_time = time.time()

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
            logits = outputs[0]  # Shape: [batch_size, sequence_length, num_labels]

            # Get predictions
            predictions = np.argmax(
                logits, axis=2
            )  # Shape: [batch_size, sequence_length]
            probabilities = self._softmax(
                logits, axis=2
            )  # Shape: [batch_size, sequence_length, num_labels]

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
            print(f"❌ Error in ONNX NER extraction: {e}")
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
_ner_classifier = None


def get_ner_classifier() -> ONNXNERClassifier:
    """Get or create the global NER classifier instance"""
    global _ner_classifier
    if _ner_classifier is None:
        # Get the full path to the model directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels
        model_path = os.path.join(backend_dir, "trained_deberta_ner_model")
        _ner_classifier = ONNXNERClassifier(model_path)
    return _ner_classifier


def extract_slots_from_text(text: str) -> Dict[str, str]:
    """Extract slots from text using the global NER classifier"""
    classifier = get_ner_classifier()
    return classifier.extract_slots(text)


# Test function
def test_onnx_ner():
    """Test the ONNX NER classifier"""
    print("🧪 Testing ONNX NER Classifier...")

    classifier = ONNXNERClassifier()
    if not classifier.is_initialized:
        print("❌ Classifier not initialized")
        return

    test_texts = [
        "I want a blue sofa under 50000 rupees",
        "Show me wooden dining table from IKEA",
        "I need bathroom accessories for my master bedroom",
        "Looking for modern lighting fixtures around 20000",
    ]

    for text in test_texts:
        print(f"\n📝 Text: {text}")
        result = classifier.extract_entities(text)
        print(
            f"🏷️ Entities: {[(e.text, e.entity_type, e.confidence) for e in result.entities]}"
        )
        print(f"🎯 Slots: {result.slots}")

    stats = classifier.get_performance_stats()
    print(f"\n📊 Performance: {stats}")


if __name__ == "__main__":
    test_onnx_ner()
