#!/usr/bin/env python3
"""
Convert Trained DeBERTa NER Model to ONNX Format

This script converts a trained DeBERTa NER model to ONNX format for optimized inference.
"""

import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
import onnx
import onnxruntime as ort
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeBERTaONNXConverter:
    """Convert DeBERTa NER model to ONNX format"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}

    def load_model(self):
        """Load the trained model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)

        # Load label mapping
        label_mapping_path = os.path.join(self.model_path, "label_mapping.json")
        if os.path.exists(label_mapping_path):
            with open(label_mapping_path, "r") as f:
                self.label2id = json.load(f)
            self.id2label = {v: k for k, v in self.label2id.items()}
        else:
            # Use default label mapping if not found
            self.label2id = self.model.config.label2id
            self.id2label = self.model.config.id2label

        logger.info(f"✅ Model loaded successfully")
        logger.info(f"Label mapping: {self.label2id}")

    def convert_to_onnx(self, output_path: str, max_length: int = 128):
        """Convert model to ONNX format"""
        logger.info("🔄 Converting model to ONNX format...")

        # Set model to evaluation mode
        self.model.eval()

        # Create dummy input
        dummy_input = {
            "input_ids": torch.randint(0, self.tokenizer.vocab_size, (1, max_length)),
            "attention_mask": torch.ones(1, max_length),
        }

        # Export to ONNX
        torch.onnx.export(
            self.model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
        )

        logger.info(f"✅ ONNX model saved to {output_path}")

    def test_onnx_model(
        self, onnx_path: str, test_text: str = "I want a blue leather sofa"
    ):
        """Test the ONNX model with a sample input"""
        logger.info(f"🧪 Testing ONNX model with text: '{test_text}'")

        # Tokenize input
        inputs = self.tokenizer(
            test_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        # Run PyTorch model
        with torch.no_grad():
            pt_outputs = self.model(**inputs)
            pt_logits = pt_outputs.logits
            pt_predictions = torch.argmax(pt_logits, dim=2)

        # Run ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        onnx_inputs = {
            "input_ids": inputs["input_ids"].numpy().astype(np.int64),
            "attention_mask": inputs["attention_mask"].numpy().astype(np.float32),
        }
        onnx_outputs = ort_session.run(None, onnx_inputs)
        onnx_logits = torch.tensor(onnx_outputs[0])
        onnx_predictions = torch.argmax(onnx_logits, dim=2)

        # Compare outputs
        pt_predictions_flat = pt_predictions[0].numpy()
        onnx_predictions_flat = onnx_predictions[0].numpy()

        # Get tokens and predictions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        pt_labels = [self.id2label.get(pred, "O") for pred in pt_predictions_flat]
        onnx_labels = [self.id2label.get(pred, "O") for pred in onnx_predictions_flat]

        logger.info("📊 Comparison Results:")
        logger.info(f"Text: {test_text}")
        logger.info(f"Tokens: {tokens[:10]}...")  # Show first 10 tokens
        logger.info(f"PyTorch predictions: {pt_labels[:10]}...")
        logger.info(f"ONNX predictions: {onnx_labels[:10]}...")

        # Check if predictions match
        predictions_match = np.array_equal(pt_predictions_flat, onnx_predictions_flat)
        logger.info(f"✅ Predictions match: {predictions_match}")

        return predictions_match

    def create_onnx_inference_class(self, onnx_path: str, output_path: str):
        """Create a Python class for ONNX inference"""
        class_code = f'''#!/usr/bin/env python3
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
        with open(f"{{model_path}}/label_mapping.json", "r") as f:
            self.label2id = json.load(f)
        self.id2label = {{v: k for k, v in self.label2id.items()}}

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

        # Run inference
        onnx_inputs = {{
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }}
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
                current_entity = {{
                    "text": token.replace("▁", ""),
                    "type": entity_type,
                    "start": i,
                    "end": i + 1,
                }}
            elif label.startswith("I-") and current_entity:
                # Continue current entity
                entity_type = label[2:]
                if entity_type == current_entity["type"]:
                    current_entity["text"] += " " + token.replace("▁", "")
                    current_entity["end"] = i + 1
                else:
                    # Different entity type, save current and start new
                    entities.append(current_entity)
                    current_entity = {{
                        "text": token.replace("▁", ""),
                        "type": entity_type,
                        "start": i,
                        "end": i + 1,
                    }}
            else:
                # Save current entity if exists
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Save last entity if exists
        if current_entity:
            entities.append(current_entity)

        return entities

    def predict_batch(self, texts: List[str], max_length: int = 128) -> List[List[Dict[str, Any]]]:
        """Predict entities for multiple texts"""
        return [self.predict(text, max_length) for text in texts]


# Example usage
if __name__ == "__main__":
    # Initialize the model
    ner_model = DeBERTaONNXNER(
        model_path="{self.model_path}",
        onnx_path="{onnx_path}"
    )
    
    # Test with sample text
    test_text = "I want a blue leather sofa from IKEA"
    entities = ner_model.predict(test_text)
    
    print(f"Text: {{test_text}}")
    print(f"Entities: {{entities}}")
'''

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(class_code)

        logger.info(f"✅ ONNX inference class saved to {output_path}")


def main():
    """Main conversion function"""
    logger.info("🚀 Starting DeBERTa NER to ONNX Conversion")
    logger.info("=" * 50)

    # Configuration
    model_path = "trained_deberta_ner_model"
    onnx_output_path = "trained_deberta_ner_model/model.onnx"
    inference_class_path = "rag/intent_modules/deberta_onnx_ner.py"

    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found at {model_path}")
        logger.error(
            "Please run the training script first: python backend/scripts/train_deberta_ner.py"
        )
        return

    # Initialize converter
    converter = DeBERTaONNXConverter(model_path)

    # Load model
    converter.load_model()

    # Convert to ONNX
    converter.convert_to_onnx(onnx_output_path)

    # Test ONNX model
    converter.test_onnx_model(onnx_output_path)

    # Create inference class
    converter.create_onnx_inference_class(onnx_output_path, inference_class_path)

    logger.info("🎉 ONNX conversion completed successfully!")
    logger.info(f"📁 ONNX model: {onnx_output_path}")
    logger.info(f"📁 Inference class: {inference_class_path}")


if __name__ == "__main__":
    main()
