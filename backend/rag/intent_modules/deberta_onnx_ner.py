#!/usr/bin/env python3
"""
ONNX Inference Class for DeBERTa NER Model

This class provides optimized inference using ONNX runtime for the trained DeBERTa NER model.
"""

import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import json
from typing import List, Dict, Any


class DeBERTaONNXNER:
    """ONNX-based NER inference for DeBERTa model"""

    def __init__(self, model_path: str, onnx_path: str):
        self.model_path = model_path
        self.onnx_path = onnx_path

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load ONNX session
        self.session = ort.InferenceSession(onnx_path)

        # Load label mapping
        with open(f"{model_path}/label_mapping.json", "r") as f:
            self.label2id = json.load(f)
        self.id2label = {v: k for k, v in self.label2id.items()}

    def predict(self, text: str, max_length: int = 128) -> List[Dict[str, Any]]:
        """Predict entities in the given text"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding=True,
            max_length=max_length,
        )

        # Run inference - use correct data types as expected by ONNX model
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.float32),
        }
        outputs = self.session.run(None, onnx_inputs)
        logits = outputs[0]

        # Get predictions
        predictions = np.argmax(logits, axis=2)

        # Convert to labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.id2label.get(pred, "O") for pred in predictions[0]]

        # Extract entities
        entities = []
        current_entity = None

        for i, (token, label) in enumerate(zip(tokens, labels)):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity:
                    entities.append(current_entity)

                # Start new entity
                entity_type = label[2:]
                current_entity = {
                    "text": token.replace("▁", ""),
                    "type": entity_type,
                    "start": i,
                    "end": i + 1,
                }
            elif label.startswith("I-") and current_entity:
                # Continue current entity
                entity_type = label[2:]
                if entity_type == current_entity["type"]:
                    current_entity["text"] += " " + token.replace("▁", "")
                    current_entity["end"] = i + 1
                else:
                    # Different entity type, save current and start new
                    entities.append(current_entity)
                    current_entity = {
                        "text": token.replace("▁", ""),
                        "type": entity_type,
                        "start": i,
                        "end": i + 1,
                    }
            else:
                # Save current entity if exists
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Save last entity if exists
        if current_entity:
            entities.append(current_entity)

        return entities

    def predict_batch(
        self, texts: List[str], max_length: int = 128
    ) -> List[List[Dict[str, Any]]]:
        """Predict entities for multiple texts"""
        return [self.predict(text, max_length) for text in texts]


# Example usage
if __name__ == "__main__":
    # Initialize the model
    ner_model = DeBERTaONNXNER(
        model_path="trained_deberta_ner_model",
        onnx_path="trained_deberta_ner_model/model.onnx",
    )

    # Test with sample text
    test_text = "I want a blue leather sofa from IKEA"
    entities = ner_model.predict(test_text)

    print(f"Text: {test_text}")
    print(f"Entities: {entities}")
