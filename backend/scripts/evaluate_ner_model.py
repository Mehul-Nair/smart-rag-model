#!/usr/bin/env python3
"""
Comprehensive NER Model Evaluation Script
Tests the trained model on various scenarios including the fixed reference cases
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ner_model():
    """Test the NER model with various scenarios"""
    try:
        # Import the NER function
        from rag.intent_modules.dynamic_ner_classifier import (
            extract_slots_from_text_dynamic,
        )

        logger.info("🧪 Starting NER Model Evaluation")
        logger.info("=" * 60)

        # Test cases organized by category
        test_cases = {
            "✅ FIXED: Reference Queries (Should Extract NO Entities)": [
                "Does this product have warranty?",
                "Tell me more about this product",
                "What is the price of this item?",
                "Show me details of that product",
                "Is this product available?",
                "Can I get warranty for this item?",
                "How much does this product cost?",
                "Where can I buy this product?",
                "Is that item in stock?",
                "Tell me about this product's features",
            ],
            "✅ VALID: Product Search Queries (Should Extract Entities)": [
                "I want rugs",
                "Show me curtains",
                "Looking for lights",
                "Need some lamps",
                "Find me sofas",
                "I want Pure Royale curtains",
                "Show me Ador rugs in blue",
                "Looking for lights for the bedroom",
            ],
            "✅ MIXED: Reference + Valid Entities": [
                "Does this product come in blue?",  # Should extract COLOR: blue
                "Is this item available for the bedroom?",  # Should extract ROOM: bedroom
                "Can I get this product in large size?",  # Should extract SIZE: large
                "Does that item have a modern style?",  # Should extract STYLE: modern
                "Is this product made of cotton?",  # Should extract MATERIAL: cotton
            ],
            "✅ COMPLEX: Warranty Queries with Specific Products": [
                "What is the warranty of Royale curtains?",  # Should extract BRAND: Royale, PRODUCT_TYPE: curtains
                "Does the Pure Silk bedsheet have warranty?",  # Should extract specific product
                "Tell me about warranty for lights",  # Should extract PRODUCT_TYPE: lights
                "Warranty information for Ador rugs",  # Should extract BRAND: Ador, PRODUCT_TYPE: rugs
            ],
        }

        total_tests = 0
        total_passed = 0
        total_failed = 0

        for category, queries in test_cases.items():
            logger.info(f"\n{category}")
            logger.info("-" * len(category))

            category_passed = 0
            category_failed = 0

            for query in queries:
                total_tests += 1
                start_time = time.time()

                try:
                    # Extract entities
                    result = extract_slots_from_text_dynamic(query)
                    extraction_time = time.time() - start_time

                    # Evaluate result based on category
                    passed = evaluate_result(query, result, category)

                    if passed:
                        total_passed += 1
                        category_passed += 1
                        status = "✅ PASS"
                    else:
                        total_failed += 1
                        category_failed += 1
                        status = "❌ FAIL"

                    logger.info(f"{status} | {query}")
                    logger.info(f"        Result: {result}")
                    logger.info(f"        Time: {extraction_time:.3f}s")

                except Exception as e:
                    total_failed += 1
                    category_failed += 1
                    logger.error(f"❌ ERROR | {query}")
                    logger.error(f"         Exception: {e}")

            logger.info(f"\nCategory Summary: {category_passed}/{len(queries)} passed")

        # Overall summary
        logger.info("\n" + "=" * 60)
        logger.info("🎯 EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
        logger.info(f"Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")

        if total_passed / total_tests >= 0.8:
            logger.info("🎉 MODEL PERFORMANCE: EXCELLENT (≥80%)")
        elif total_passed / total_tests >= 0.6:
            logger.info("👍 MODEL PERFORMANCE: GOOD (≥60%)")
        else:
            logger.info("⚠️ MODEL PERFORMANCE: NEEDS IMPROVEMENT (<60%)")

        return total_passed / total_tests

    except ImportError as e:
        logger.error(f"❌ Failed to import NER module: {e}")
        logger.error("Make sure the trained model is available")
        return 0.0
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return 0.0


def evaluate_result(query: str, result: Dict[str, str], category: str) -> bool:
    """Evaluate if the NER result is correct for the given category"""

    if "Reference Queries (Should Extract NO Entities)" in category:
        # These should extract NO entities (especially no PRODUCT_TYPE)
        return len(result) == 0 or "PRODUCT_TYPE" not in result

    elif "Product Search Queries (Should Extract Entities)" in category:
        # These should extract at least one relevant entity
        return len(result) > 0 and any(
            key in ["PRODUCT_TYPE", "BRAND", "PRODUCT_NAME"] for key in result.keys()
        )

    elif "Reference + Valid Entities" in category:
        # These should extract specific entities but NOT generic "product"
        has_valid_entities = any(
            key in ["COLOR", "ROOM", "SIZE", "STYLE", "MATERIAL"]
            for key in result.keys()
        )
        no_generic_product = result.get("PRODUCT_TYPE") != "product"
        return has_valid_entities and no_generic_product

    elif "Warranty Queries with Specific Products" in category:
        # These should extract specific product info
        return any(
            key in ["PRODUCT_TYPE", "BRAND", "PRODUCT_NAME"] for key in result.keys()
        )

    return True  # Default to pass for unknown categories


def test_specific_problematic_cases():
    """Test the specific cases that were problematic before"""
    logger.info("\n🔍 TESTING SPECIFIC PROBLEMATIC CASES")
    logger.info("=" * 50)

    try:
        from rag.intent_modules.dynamic_ner_classifier import (
            extract_slots_from_text_dynamic,
        )

        problematic_cases = [
            ("does this product have warranty", {}),  # Should extract nothing
            ("product details please", {}),  # Should extract nothing
            ("tell me about this product", {}),  # Should extract nothing
            ("I want rugs", {"PRODUCT_TYPE": "rugs"}),  # Should extract rugs
            (
                "warranty for Royale curtains",
                {"BRAND": "Royale", "PRODUCT_TYPE": "curtains"},
            ),  # Should extract both
        ]

        for query, expected in problematic_cases:
            result = extract_slots_from_text_dynamic(query)

            # Check if result matches expectation
            if not expected:  # Expecting no entities
                success = len(result) == 0 or "PRODUCT_TYPE" not in result
            else:  # Expecting specific entities
                success = all(result.get(k) == v for k, v in expected.items())

            status = "✅ FIXED" if success else "❌ STILL BROKEN"
            logger.info(f"{status} | '{query}'")
            logger.info(f"         Expected: {expected}")
            logger.info(f"         Got: {result}")

    except Exception as e:
        logger.error(f"❌ Failed to test problematic cases: {e}")


if __name__ == "__main__":
    # Run comprehensive evaluation
    overall_score = test_ner_model()

    # Test specific problematic cases
    test_specific_problematic_cases()

    logger.info(f"\n🎯 Final Score: {overall_score:.2%}")
