#!/usr/bin/env python3
"""
Debug script to test NER model with problematic examples
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "rag", "intent_modules"))

from onnx_ner_classifier import ONNXNERClassifier


def test_ner_model():
    """Test the NER model with problematic examples"""
    print("🔧 Initializing NER model...")

    # Get the full path to the model directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "trained_deberta_ner_model")

    classifier = ONNXNERClassifier(model_path)

    if not classifier.is_initialized:
        print("❌ Classifier not initialized")
        return

    print("✅ NER model loaded successfully")
    print(f"📊 Available labels: {list(classifier.id2label.values())}")

    # Test cases from the conversation
    test_cases = [
        "im looking for lights",
        "give me details of city lights ceiling light",
        "give me details of crystal chandelier ?",
        "I want a blue sofa under 50000 rupees",
        "Show me wooden dining table from IKEA",
    ]

    print("\n" + "=" * 80)
    print("🧪 TESTING NER MODEL")
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


if __name__ == "__main__":
    test_ner_model()
