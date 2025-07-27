#!/usr/bin/env python3
"""
Direct test of intent classification without web server
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.intent_modules import IntentClassifierFactory


def test_intent_classification():
    """Test intent classification directly"""

    print("🧪 DIRECT INTENT CLASSIFICATION TEST")
    print("=" * 50)

    # Test cases
    test_cases = [
        ("do you have rugs", "PRODUCT_SEARCH"),
        ("show rugs under 5000", "BUDGET_QUERY"),
        ("help", "HELP"),
        ("hello", "GREETING"),
        ("what categories do you have", "CATEGORY_LIST"),
        ("I need something for my living room", "CLARIFY"),
    ]

    try:
        # Initialize the intent classifier
        print("🔧 Initializing intent classifier...")
        classifier = IntentClassifierFactory.create(
            "improved_hybrid",
            {
                "confidence_threshold": 0.3,
                "primary_classifier": "huggingface",
                "fallback_classifier": "rule_based",
                "enable_intent_specific_rules": True,
                "implementation_configs": {
                    "huggingface": {
                        "model_path": "./trained_deberta_model",
                        "device": "cpu",
                        "intent_mapping": {
                            "BUDGET_QUERY": 0,
                            "CATEGORY_LIST": 1,
                            "CLARIFY": 2,
                            "GREETING": 3,
                            "HELP": 4,
                            "INVALID": 5,
                            "PRODUCT_DETAIL": 6,
                            "PRODUCT_SEARCH": 7,
                            "WARRANTY_QUERY": 8,
                        },
                    },
                    "rule_based": {"similarity_threshold": 0.3},
                },
            },
        )
        print("✅ Intent classifier initialized successfully")

        # Test each case
        passed = 0
        failed = 0

        for query, expected_intent in test_cases:
            print(f"\n📝 Testing: '{query}'")
            print(f"   Expected: {expected_intent}")

            try:
                result = classifier.classify_intent(query)
                predicted_intent = result.intent
                confidence = result.confidence

                print(
                    f"   Predicted: {predicted_intent} (confidence: {confidence:.3f})"
                )

                # Add debugging info
                print(f"   Raw result: {result}")
                print(f"   Result type: {type(result)}")
                print(f"   Result attributes: {dir(result)}")

                # Convert expected_intent string to IntentType enum for comparison
                from rag.intent_modules.base import IntentType

                # Map the expected intent strings to the correct enum values
                intent_mapping = {
                    "PRODUCT_SEARCH": IntentType.PRODUCT_SEARCH,
                    "BUDGET_QUERY": IntentType.BUDGET_QUERY,
                    "HELP": IntentType.HELP,
                    "GREETING": IntentType.GREETING,
                    "CATEGORY_LIST": IntentType.CATEGORY_LIST,
                    "CLARIFY": IntentType.CLARIFY,
                }

                expected_enum = intent_mapping.get(expected_intent)

                if predicted_intent == expected_enum:
                    print(f"   ✅ PASS")
                    passed += 1
                else:
                    print(
                        f"   ❌ FAIL - Expected {expected_enum}, got {predicted_intent}"
                    )
                    failed += 1

            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                failed += 1

        # Summary
        print(f"\n📊 SUMMARY")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")

        if passed == len(test_cases):
            print("🎉 All tests passed! Intent classification is working correctly.")
        else:
            print("⚠️  Some tests failed. There may be issues with the model.")

    except Exception as e:
        print(f"❌ Error initializing classifier: {e}")
        return False

    return True


if __name__ == "__main__":
    test_intent_classification()
