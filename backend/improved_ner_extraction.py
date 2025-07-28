#!/usr/bin/env python3
"""
Improved NER extraction for product names
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "rag", "intent_modules"))

from onnx_ner_classifier import ONNXNERClassifier


def test_product_name_extraction():
    """Test NER extraction for product names"""
    print("🔧 Initializing NER model...")

    # Get the full path to the model directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "trained_deberta_ner_model")

    classifier = ONNXNERClassifier(model_path)

    if not classifier.is_initialized:
        print("❌ Classifier not initialized")
        return

    print("✅ NER model loaded successfully")

    # Test cases for product names
    test_cases = [
        "give me details of city lights ceiling light",
        "give me details of crystal chandelier",
        "show me the city lights ceiling light",
        "I want to see the crystal chandelier details",
        "tell me about the lighting up lisbon chandelier",
        "give me details of lights out gold vanity light",
    ]

    print("\n" + "=" * 80)
    print("🧪 TESTING PRODUCT NAME EXTRACTION")
    print("=" * 80)

    for i, text in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: '{text}'")
        print("-" * 60)

        try:
            result = classifier.extract_entities(text)

            print(f"🏷️  Raw entities:")
            for entity in result.entities:
                print(
                    f"   - Text: '{entity.text}' | Type: {entity.entity_type} | Confidence: {entity.confidence:.3f}"
                )

            print(f"🎯 Extracted slots:")
            for slot_type, value in result.slots.items():
                print(f"   - {slot_type}: '{value}'")

            # Try to reconstruct product name
            product_name = reconstruct_product_name(result.entities, text)
            print(f"🔧 Reconstructed product name: '{product_name}'")

        except Exception as e:
            print(f"❌ Error: {e}")


def reconstruct_product_name(entities, original_text):
    """Try to reconstruct a product name from entities"""
    # Look for PRODUCT_NAME first
    product_names = [e.text for e in entities if e.entity_type == "PRODUCT_NAME"]
    if product_names:
        return " ".join(product_names)

    # Look for consecutive PRODUCT_TYPE entities that might form a product name
    product_types = [e.text for e in entities if e.entity_type == "PRODUCT_TYPE"]
    if len(product_types) > 1:
        # Multiple product types might be a product name
        return " ".join(product_types)

    # Look for MATERIAL + PRODUCT_TYPE combinations
    materials = [e.text for e in entities if e.entity_type == "MATERIAL"]
    if materials and product_types:
        return f"{' '.join(materials)} {' '.join(product_types)}"

    # Fallback: look for any entity that might be part of a product name
    all_entities = [
        e.text
        for e in entities
        if e.entity_type in ["PRODUCT_TYPE", "MATERIAL", "COLOR", "BRAND"]
    ]
    if all_entities:
        return " ".join(all_entities)

    return original_text


if __name__ == "__main__":
    test_product_name_extraction()
