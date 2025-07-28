#!/usr/bin/env python3
"""
Test conversation flow with problematic examples
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "rag", "intent_modules"))

from dynamic_ner_classifier import DynamicNERClassifier


def test_conversation_flow():
    """Test the conversation flow with problematic examples"""
    print("🔧 Initializing Dynamic NER model...")

    # Get the full path to the model directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "trained_deberta_ner_model")

    classifier = DynamicNERClassifier(model_path)

    if not classifier.is_initialized:
        print("❌ Classifier not initialized")
        return

    print("✅ NER model loaded successfully")

    # Test the exact conversation flow from the logs
    test_cases = [
        "give me details of city lights ceiling light",
        "give me details of crystal chandelier ?",
        "give me details of lighting up lisbon chandelier",
        "give me details of lights out gold vanity light",
        "I want Pure Royale curtains",  # Test brand detection
        "Show me White Teak by Asian Paints lights",  # Test brand detection
    ]

    print("\n" + "=" * 80)
    print("🧪 TESTING CONVERSATION FLOW")
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

            # Simulate the slot reconstruction logic
            from rag.langgraph_agent import reconstruct_product_name_from_entities

            reconstructed_name = reconstruct_product_name_from_entities(
                result.slots, text
            )
            print(f"🔧 Reconstructed product name: '{reconstructed_name}'")

            # Check if this would trigger brand slot requirement
            if "BRAND" in result.slots:
                print(f"✅ Brand detected: {result.slots['BRAND']}")
            else:
                print(f"⚠️ No brand detected - might ask for brand slot")

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_conversation_flow()
