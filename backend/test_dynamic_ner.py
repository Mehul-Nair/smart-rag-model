#!/usr/bin/env python3
"""
Test Dynamic NER Classifier
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "rag", "intent_modules"))

from dynamic_ner_classifier import DynamicNERClassifier


def test_dynamic_ner():
    """Test the dynamic NER classifier"""
    print("🔧 Initializing Dynamic NER Classifier...")

    # Get the full path to the model directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "trained_deberta_ner_model")

    classifier = DynamicNERClassifier(model_path)

    if not classifier.is_initialized:
        print("❌ Classifier not initialized")
        return

    print("✅ Dynamic NER Classifier loaded successfully")

    # Show what was loaded
    stats = classifier.get_performance_stats()
    print(
        f"📊 Loaded {stats['product_names_loaded']} product names and {stats['brands_loaded']} brands"
    )

    # Test cases
    test_cases = [
        "give me details of city lights ceiling light",
        "give me details of crystal chandelier ?",
        "give me details of lighting up lisbon chandelier",
        "give me details of lights out gold vanity light",
        "I want Pure Royale curtains",
        "Show me White Teak by Asian Paints lights",
        "tell me about drop dead gorgeous chandelier",
        "what is time to shine chandelier",
        "show me keeper of my heart chandelier",
        "give me details of bolt chandelier",
    ]

    print("\n" + "=" * 80)
    print("🧪 TESTING DYNAMIC NER CLASSIFIER")
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

        except Exception as e:
            print(f"❌ Error: {e}")

    # Test adding new product names dynamically
    print(f"\n🔄 Testing dynamic addition:")
    print("-" * 60)

    classifier.add_product_name("New Amazing Product")
    classifier.add_brand("New Brand")

    # Test with new product
    test_text = "give me details of new amazing product"
    result = classifier.extract_entities(test_text)
    print(f"📝 New product test: '{test_text}'")
    print(f"🎯 Result: {result.slots}")


if __name__ == "__main__":
    test_dynamic_ner()
