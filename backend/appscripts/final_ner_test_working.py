#!/usr/bin/env python3
"""
Working NER Model Test Script
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os


def load_model(model_path="trained_deberta_ner_model"):
    """Load the trained model with correct label mapping"""
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # Use the model's actual label mapping
    id2label = model.config.id2label
    label2id = model.config.label2id

    print(f"✅ Model loaded successfully")
    print(f"📊 Number of labels: {len(id2label)}")
    print(f"📋 Label mapping: {id2label}")

    return tokenizer, model, id2label


def predict_entities(text, tokenizer, model, id2label):
    """Predict entities in text"""

    # Tokenize
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128, padding=True
    )

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

    # Convert to labels - fix the tensor iteration issue
    predicted_labels = []
    predictions_list = predictions[0].tolist()  # Convert to list
    for label_id in predictions_list:
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
            # Fix: Convert single tensor element to list for tokenizer
            token_id = inputs["input_ids"][0][i].item()
            token_text = tokenizer.convert_ids_to_tokens([token_id])[0]
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
                # Fix: Convert single tensor element to list for tokenizer
                token_id = inputs["input_ids"][0][i].item()
                token_text = tokenizer.convert_ids_to_tokens([token_id])[0]
                current_entity["text"] += " " + token_text

    # Add last entity
    if current_entity:
        entities.append(current_entity)

    # Clean up text
    for entity in entities:
        entity["text"] = entity["text"].replace("▁", "").strip()

    return entities


def test_query(text, tokenizer, model, id2label):
    """Test a single query"""
    print(f'\n🔍 Testing: "{text}"')
    print("-" * 50)

    try:
        entities = predict_entities(text, tokenizer, model, id2label)

        if entities:
            print(f"✅ Found {len(entities)} entities:")
            for i, entity in enumerate(entities, 1):
                print(f"  {i}. {entity['label']}: \"{entity['text']}\"")
        else:
            print("❌ No entities found")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main testing function"""

    # Check if model exists
    model_path = "trained_deberta_ner_model"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    try:
        # Load model
        tokenizer, model, id2label = load_model(model_path)

        # Test queries
        test_queries = [
            "I want a red sofa for my living room",
            "Show me Pure Royale curtains",
            "Need dining chairs under ₹5000",
            "Looking for cotton bedsheets",
            "I need a king size bed",
            "Modern minimalist table lamp",
            "Bathroom mirror cabinet",
            "White Teak Aurora pendant light",
            "Pure Royale luxury velvet sofa in navy blue for living room under ₹15000",
            "Show me options around ₹3000",
        ]

        print(f"\n🧪 Testing {len(test_queries)} queries...")

        for i, query in enumerate(test_queries, 1):
            print(f"\n{i:2d}. ", end="")
            test_query(query, tokenizer, model, id2label)

        print(f"\n{'='*60}")
        print("✅ Testing completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
