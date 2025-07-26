#!/usr/bin/env python3
"""
Test ONNX NER Model
"""

import os
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def test_onnx_ner():
    """Test the ONNX NER model"""

    model_path = "trained_deberta_ner_model"
    onnx_path = os.path.join(model_path, "model.onnx")

    print(f"Testing ONNX model: {onnx_path}")
    print(f"ONNX model exists: {os.path.exists(onnx_path)}")

    if not os.path.exists(onnx_path):
        print("❌ ONNX model not found!")
        return

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer loaded")

        # Load ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        print("✅ ONNX model loaded")

        # Load label mapping
        label_mapping_path = os.path.join(model_path, "label_mapping.json")
        with open(label_mapping_path, "r") as f:
            label2id = json.load(f)
        id2label = {v: k for k, v in label2id.items()}
        print(f"✅ Label mapping loaded: {len(id2label)} labels")

        # Test with sample text
        test_text = "I want a red sofa for my living room"
        print(f"\n🧪 Testing: '{test_text}'")

        # Tokenize
        inputs = tokenizer(
            test_text,
            return_tensors="np",
            truncation=True,
            max_length=128,
            padding=True,
        )

        # Convert to correct data types (ONNX expects int64 for input_ids and float for attention_mask)
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.float32)

        # Run inference
        onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        onnx_outputs = ort_session.run(None, onnx_inputs)
        logits = onnx_outputs[0]

        # Get predictions
        predictions = np.argmax(logits, axis=2)

        # Convert to labels
        predicted_labels = []
        for i in range(predictions.shape[1]):
            label_id = predictions[0, i]
            predicted_labels.append(id2label[label_id])

        # Get word IDs
        word_ids = inputs.word_ids(batch_index=0)

        # Extract entities
        entities = []
        current_entity = None

        for i, (word_id, label) in enumerate(zip(word_ids, predicted_labels)):
            if word_id is None:
                continue

            if label.startswith("B-"):
                # Start new entity
                if current_entity:
                    entities.append(current_entity)

                entity_type = label[2:]
                token_text = tokenizer.convert_ids_to_tokens([input_ids[0][i]])[0]
                current_entity = {
                    "start": word_id,
                    "end": word_id + 1,
                    "label": entity_type,
                    "text": token_text,
                }

            elif label.startswith("I-") and current_entity:
                # Continue entity
                entity_type = label[2:]
                if entity_type == current_entity["label"]:
                    current_entity["end"] = word_id + 1
                    token_text = tokenizer.convert_ids_to_tokens([input_ids[0][i]])[0]
                    current_entity["text"] += " " + token_text

        # Add last entity
        if current_entity:
            entities.append(current_entity)

        # Clean up text
        for entity in entities:
            entity["text"] = entity["text"].replace("▁", "").strip()

        print(f"✅ Found {len(entities)} entities:")
        for i, entity in enumerate(entities, 1):
            print(f"  {i}. {entity['label']}: \"{entity['text']}\"")

        print("\n🎉 ONNX model test successful!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_onnx_ner()
