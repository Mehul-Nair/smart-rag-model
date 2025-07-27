#!/usr/bin/env python3
"""
Test ONNX NER Model Entity Extraction

This script tests the ONNX NER model to see how it extracts entities
and logs the results for analysis.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.intent_modules.deberta_onnx_ner import DeBERTaONNXNER
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_entity_extraction():
    """Test entity extraction with various examples"""

    # Initialize the ONNX NER model
    try:
        ner_model = DeBERTaONNXNER(
            model_path="trained_deberta_ner_model",
            onnx_path="trained_deberta_ner_model/model.onnx",
        )
        logger.info("✅ ONNX NER model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load ONNX NER model: {e}")
        return

    # Test cases - including the problematic ones
    test_cases = [
        # Original problematic case
        "im looking for lightings",
        # Budget cases
        "i have a budget of 10000 rs",
        "under 5000 rupees",
        "around 8000 rs",
        # Product cases
        "i want some furniture",
        "show me curtains",
        "need lamps for bedroom",
        # Complex cases
        "Pure Royale curtains for bedroom in blue color",
        "White Teak lamps for living room in modern style",
        # Simple cases
        "curtains",
        "furniture",
        "lighting",
        # Mixed cases
        "looking for affordable curtains under 5000",
        "need modern furniture for small apartment",
        # Edge cases
        "budget planning is important",
        "premium range products",
        "affordable options available",
    ]

    logger.info("🧪 Testing Entity Extraction")
    logger.info("=" * 50)

    for i, text in enumerate(test_cases, 1):
        try:
            entities = ner_model.predict(text)

            logger.info(f"\n📝 Test Case {i}: '{text}'")
            logger.info(f"🔍 Extracted Entities: {entities}")

            # Show detailed breakdown
            if entities:
                for entity in entities:
                    logger.info(
                        f"   • {entity['type']}: '{entity['text']}' (pos: {entity['start']}-{entity['end']})"
                    )
            else:
                logger.info("   • No entities extracted")

        except Exception as e:
            logger.error(f"❌ Error processing '{text}': {e}")

    logger.info("\n" + "=" * 50)
    logger.info("🎯 Entity Extraction Test Complete")


def test_specific_problematic_cases():
    """Test specific cases that were problematic before"""

    try:
        ner_model = DeBERTaONNXNER(
            model_path="trained_deberta_ner_model",
            onnx_path="trained_deberta_ner_model/model.onnx",
        )
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return

    logger.info("\n🔍 Testing Specific Problematic Cases")
    logger.info("=" * 50)

    problematic_cases = [
        ("im looking for lightings", "Should extract: lighting"),
        ("i want some furnitures", "Should extract: furniture"),
        ("show me curtains please", "Should extract: curtains"),
        ("find me lamps for bedroom", "Should extract: lamps"),
        ("need rugs for living room", "Should extract: rugs"),
        ("want cushions for sofa", "Should extract: cushions"),
        ("looking for lighting options", "Should extract: lighting"),
        ("get me some furniture", "Should extract: furniture"),
        ("can you show curtains", "Should extract: curtains"),
        ("recommend lamps for study", "Should extract: lamps"),
    ]

    for text, expected in problematic_cases:
        try:
            entities = ner_model.predict(text)

            logger.info(f"\n📝 Text: '{text}'")
            logger.info(f"🎯 Expected: {expected}")
            logger.info(f"🔍 Actual: {entities}")

            # Check if extraction is correct
            product_entities = [e for e in entities if e["type"] == "PRODUCT_TYPE"]
            if product_entities:
                extracted_text = product_entities[0]["text"]
                logger.info(f"✅ Extracted product: '{extracted_text}'")
            else:
                logger.info("❌ No product extracted")

        except Exception as e:
            logger.error(f"❌ Error: {e}")


if __name__ == "__main__":
    logger.info("🚀 Starting ONNX NER Entity Extraction Test")

    # Test general entity extraction
    test_entity_extraction()

    # Test specific problematic cases
    test_specific_problematic_cases()

    logger.info("🎉 Entity extraction test completed!")
