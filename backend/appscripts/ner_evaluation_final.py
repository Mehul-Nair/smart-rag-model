#!/usr/bin/env python3
"""
Final Improved NER Model Evaluation Script
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
from collections import defaultdict
import re


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

    return tokenizer, model, id2label


def normalize_text(text):
    """Improved text normalization for better comparison"""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Normalize currency formatting
    text = re.sub(r"₹\s*(\d+)", r"₹\1", text)  # Remove spaces after ₹
    text = re.sub(
        r"(\d+)\s+(\d{3})", r"\1\2", text
    )  # Remove spaces in numbers like "15 000"

    # Normalize common variations
    text = text.lower()
    text = text.replace("bed sheets", "bedsheets")
    text = text.replace("bed sheet", "bedsheet")

    return text


def post_process_entities(entities):
    """Post-process entities to fix common issues"""
    processed_entities = []

    for entity in entities:
        # Fix budget formatting
        if entity["label"] == "BUDGET":
            # Normalize budget text
            text = entity["text"]
            # Remove extra spaces in currency
            text = re.sub(r"₹\s+(\d+)", r"₹\1", text)
            # Remove spaces in numbers
            text = re.sub(r"(\d+)\s+(\d{3})", r"\1\2", text)
            entity["text"] = text

        # Fix material issues
        elif entity["label"] == "MATERIAL":
            text = entity["text"].lower()
            # Filter out non-material words
            non_materials = ["luxury", "premium", "high", "quality", "good", "best"]
            if text in non_materials:
                continue  # Skip this entity

        processed_entities.append(entity)

    return processed_entities


def predict_entities(text, tokenizer, model, id2label):
    """Predict entities in text with improved post-processing"""

    # Tokenize
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128, padding=True
    )

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

    # Convert to labels
    predicted_labels = []
    predictions_list = predictions[0].tolist()
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
                token_id = inputs["input_ids"][0][i].item()
                token_text = tokenizer.convert_ids_to_tokens([token_id])[0]
                current_entity["text"] += " " + token_text

    # Add last entity
    if current_entity:
        entities.append(current_entity)

    # Clean up text
    for entity in entities:
        entity["text"] = entity["text"].replace("▁", "").strip()

    # Apply post-processing
    entities = post_process_entities(entities)

    return entities


def evaluate_predictions(predicted_entities, expected_entities):
    """Evaluate predicted entities against expected entities with improved normalization"""

    # Normalize for comparison
    pred_normalized = [
        (normalize_text(e["text"]), e["label"]) for e in predicted_entities
    ]
    exp_normalized = [
        (normalize_text(e["text"]), e["label"]) for e in expected_entities
    ]

    # Count matches
    correct = 0
    incorrect = 0
    missed = 0
    extra = 0

    # Check for exact matches
    matched_pred = set()
    matched_exp = set()

    for i, (pred_text, pred_label) in enumerate(pred_normalized):
        found_match = False
        for j, (exp_text, exp_label) in enumerate(exp_normalized):
            if (
                j not in matched_exp
                and pred_text == exp_text
                and pred_label == exp_label
            ):
                correct += 1
                matched_pred.add(i)
                matched_exp.add(j)
                found_match = True
                break

        if not found_match:
            extra += 1

    # Count missed entities
    missed = len(exp_normalized) - len(matched_exp)

    # Count incorrect predictions (wrong label for same text)
    for i, (pred_text, pred_label) in enumerate(pred_normalized):
        if i not in matched_pred:
            for j, (exp_text, exp_label) in enumerate(exp_normalized):
                if (
                    j not in matched_exp
                    and pred_text == exp_text
                    and pred_label != exp_label
                ):
                    incorrect += 1
                    break

    return {
        "correct": correct,
        "incorrect": incorrect,
        "missed": missed,
        "extra": extra,
        "total_expected": len(exp_normalized),
        "total_predicted": len(pred_normalized),
    }


def main():
    """Main evaluation function"""

    # Check if model exists
    model_path = "trained_deberta_ner_model"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    try:
        # Load model
        tokenizer, model, id2label = load_model(model_path)

        # Test cases with expected entities
        test_cases = [
            {
                "text": "I want a red sofa for my living room",
                "expected": [
                    {"text": "red", "label": "COLOR"},
                    {"text": "sofa", "label": "PRODUCT_TYPE"},
                    {"text": "living room", "label": "ROOM"},
                ],
            },
            {
                "text": "Show me Pure Royale curtains",
                "expected": [
                    {"text": "Pure Royale", "label": "BRAND"},
                    {"text": "curtains", "label": "PRODUCT_TYPE"},
                ],
            },
            {
                "text": "Need dining chairs under ₹5000",
                "expected": [
                    {"text": "dining chairs", "label": "PRODUCT_TYPE"},
                    {"text": "under ₹5000", "label": "BUDGET"},
                ],
            },
            {
                "text": "Looking for cotton bedsheets",
                "expected": [
                    {"text": "cotton", "label": "MATERIAL"},
                    {"text": "bedsheets", "label": "PRODUCT_TYPE"},
                ],
            },
            {
                "text": "I need a king size bed",
                "expected": [
                    {"text": "king size", "label": "SIZE"},
                    {"text": "bed", "label": "PRODUCT_TYPE"},
                ],
            },
            {
                "text": "Modern minimalist table lamp",
                "expected": [
                    {"text": "Modern", "label": "STYLE"},
                    {"text": "table lamp", "label": "PRODUCT_TYPE"},
                ],
            },
            {
                "text": "Bathroom mirror cabinet",
                "expected": [
                    {"text": "Bathroom", "label": "ROOM"},
                    {"text": "mirror cabinet", "label": "PRODUCT_TYPE"},
                ],
            },
            {
                "text": "White Teak Aurora pendant light",
                "expected": [
                    {"text": "White Teak Aurora", "label": "BRAND"},
                    {"text": "pendant light", "label": "PRODUCT_TYPE"},
                ],
            },
            {
                "text": "Pure Royale luxury velvet sofa in navy blue for living room under ₹15000",
                "expected": [
                    {"text": "Pure Royale", "label": "BRAND"},
                    {"text": "velvet", "label": "MATERIAL"},
                    {"text": "sofa", "label": "PRODUCT_TYPE"},
                    {"text": "navy blue", "label": "COLOR"},
                    {"text": "living room", "label": "ROOM"},
                    {"text": "under ₹15000", "label": "BUDGET"},
                ],
            },
            {
                "text": "Show me options around ₹3000",
                "expected": [{"text": "around ₹3000", "label": "BUDGET"}],
            },
        ]

        print(
            f"\n🧪 Evaluating {len(test_cases)} test cases with final improvements..."
        )
        print("=" * 80)

        # Overall statistics
        total_correct = 0
        total_incorrect = 0
        total_missed = 0
        total_extra = 0
        total_expected = 0
        total_predicted = 0

        # Per-entity-type statistics
        entity_stats = defaultdict(
            lambda: {"correct": 0, "total_expected": 0, "total_predicted": 0}
        )

        for i, test_case in enumerate(test_cases, 1):
            text = test_case["text"]
            expected = test_case["expected"]

            print(f'\n{i:2d}. Testing: "{text}"')
            print("-" * 60)

            # Get predictions with post-processing
            predicted = predict_entities(text, tokenizer, model, id2label)

            # Evaluate
            results = evaluate_predictions(predicted, expected)

            # Update overall stats
            total_correct += results["correct"]
            total_incorrect += results["incorrect"]
            total_missed += results["missed"]
            total_extra += results["extra"]
            total_expected += results["total_expected"]
            total_predicted += results["total_predicted"]

            # Update per-entity stats
            for entity in expected:
                entity_stats[entity["label"]]["total_expected"] += 1
            for entity in predicted:
                entity_stats[entity["label"]]["total_predicted"] += 1

            # Check correct predictions
            for exp_entity in expected:
                for pred_entity in predicted:
                    if (
                        normalize_text(exp_entity["text"])
                        == normalize_text(pred_entity["text"])
                        and exp_entity["label"] == pred_entity["label"]
                    ):
                        entity_stats[exp_entity["label"]]["correct"] += 1

            # Print results for this test case
            print(f"Expected: {len(expected)} entities")
            for entity in expected:
                print(f"  - {entity['label']}: \"{entity['text']}\"")

            print(f"Predicted: {len(predicted)} entities")
            for entity in predicted:
                print(f"  - {entity['label']}: \"{entity['text']}\"")

            print(
                f"Results: {results['correct']} correct, {results['incorrect']} incorrect, {results['missed']} missed, {results['extra']} extra"
            )

        # Calculate overall metrics
        precision = total_correct / total_predicted if total_predicted > 0 else 0
        recall = total_correct / total_expected if total_expected > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(f"\n{'='*80}")
        print("📊 FINAL IMPROVED EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"Total Expected Entities: {total_expected}")
        print(f"Total Predicted Entities: {total_predicted}")
        print(f"Correct Predictions: {total_correct}")
        print(f"Incorrect Predictions: {total_incorrect}")
        print(f"Missed Entities: {total_missed}")
        print(f"Extra Predictions: {total_extra}")
        print(f"\nPrecision: {precision:.3f} ({total_correct}/{total_predicted})")
        print(f"Recall: {recall:.3f} ({total_correct}/{total_expected})")
        print(f"F1-Score: {f1_score:.3f}")

        # Per-entity-type results
        print(f"\n{'='*80}")
        print("📋 PER-ENTITY-TYPE RESULTS")
        print(f"{'='*80}")

        for entity_type, stats in sorted(entity_stats.items()):
            if stats["total_expected"] > 0:
                entity_precision = (
                    stats["correct"] / stats["total_predicted"]
                    if stats["total_predicted"] > 0
                    else 0
                )
                entity_recall = stats["correct"] / stats["total_expected"]
                entity_f1 = (
                    2
                    * (entity_precision * entity_recall)
                    / (entity_precision + entity_recall)
                    if (entity_precision + entity_recall) > 0
                    else 0
                )

                print(
                    f"{entity_type:15s}: P={entity_precision:.3f} R={entity_recall:.3f} F1={entity_f1:.3f} "
                    f"({stats['correct']}/{stats['total_expected']} expected, {stats['total_predicted']} predicted)"
                )

        print(f"\n{'='*80}")
        print("✅ Final evaluation completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
