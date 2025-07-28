#!/usr/bin/env python3
"""
Test Combined NER Speed - Two-Layer Approach
"""

import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "rag", "intent_modules"))

from dynamic_ner_classifier import DynamicNERClassifier


def test_combined_ner_speed():
    """Test the combined speed of two-layer NER system"""

    print("🏁 TESTING COMBINED NER SPEED - TWO-LAYER APPROACH")
    print("=" * 70)

    # Initialize the dynamic NER classifier
    print("🔧 Initializing Dynamic NER Classifier...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "trained_deberta_ner_model")

    classifier = DynamicNERClassifier(model_path)

    if not classifier.is_initialized:
        print("❌ Classifier not initialized")
        return

    print("✅ Dynamic NER Classifier loaded successfully")

    # Test cases covering different scenarios
    test_cases = [
        # Layer 1: Known Products (should be 0ms)
        "give me details of city lights ceiling light",
        "show me lighting up lisbon chandelier",
        "tell me about lights out gold vanity light",
        "what is drop dead gorgeous chandelier",
        "show me keeper of my heart chandelier",
        # Layer 1: Known Brands (should be 0ms)
        "I want Pure Royale curtains",
        "Show me White Teak by Asian Paints lights",
        "find me Ador products",
        "give me Bathsense options",
        # Layer 2: ML Model Fallback (should be ~70ms)
        "I need a blue ceiling light for my living room",
        "show me wooden wall lights under 5000",
        "find me a gold chandelier for dining room",
        "what are the best LED pendant lights?",
        "give me options for bathroom vanity lights",
        "show me outdoor wall lights with motion sensor",
        "I want a large crystal chandelier",
        "find me modern ceiling lights",
        "show me vintage wall sconces",
        "give me options for kitchen pendant lights",
        # Mixed: Known Product + Unknown Attributes
        "I want a blue city lights ceiling light",
        "show me large city lights chandelier",
        "find me crystal city lights pendant",
        "give me options for gold city lights wall light",
        # Complex: Multiple Unknown Attributes
        "I need a large wooden chandelier in gold color for dining room",
        "show me modern LED ceiling lights in white for living room",
        "find me vintage crystal pendant lights in antique gold",
        "give me options for outdoor wall lights in black with motion sensor",
    ]

    print(f"\n📊 Testing {len(test_cases)} different scenarios...")
    print("-" * 70)

    # Track performance by category
    known_products_times = []
    known_brands_times = []
    ml_fallback_times = []
    mixed_times = []
    complex_times = []

    # Track results
    all_results = []

    for i, text in enumerate(test_cases, 1):
        print(f"\n📝 Test {i:2d}: '{text[:50]}{'...' if len(text) > 50 else ''}'")

        start_time = time.time()
        result = classifier.extract_entities(text)
        end_time = time.time()

        processing_time = (end_time - start_time) * 1000  # Convert to ms
        all_results.append((text, result, processing_time))

        # Categorize based on results
        has_product_name = "PRODUCT_NAME" in result.slots
        has_brand = "BRAND" in result.slots
        has_ml_attributes = any(
            key in result.slots
            for key in ["COLOR", "MATERIAL", "SIZE", "ROOM", "STYLE"]
        )

        # Determine category
        if has_product_name and not has_ml_attributes:
            known_products_times.append(processing_time)
            category = "Known Product"
        elif has_brand and not has_ml_attributes:
            known_brands_times.append(processing_time)
            category = "Known Brand"
        elif has_product_name and has_ml_attributes:
            mixed_times.append(processing_time)
            category = "Mixed (Product + ML)"
        elif has_ml_attributes and len(result.slots) > 2:
            complex_times.append(processing_time)
            category = "Complex (ML Only)"
        else:
            ml_fallback_times.append(processing_time)
            category = "ML Fallback"

        print(f"   ⏱️  Time: {processing_time:6.2f}ms | Category: {category}")
        print(f"   🎯 Slots: {result.slots}")

    # Calculate statistics
    print(f"\n📊 COMBINED NER SPEED ANALYSIS")
    print("=" * 70)

    # Known Products (Layer 1 only)
    if known_products_times:
        avg_known_products = sum(known_products_times) / len(known_products_times)
        print(f"🔹 Layer 1 - Known Products ({len(known_products_times)} cases):")
        print(f"   Average: {avg_known_products:6.2f}ms")
        print(f"   Min:     {min(known_products_times):6.2f}ms")
        print(f"   Max:     {max(known_products_times):6.2f}ms")
        print(f"   QPS:     {1000/avg_known_products:.1f} queries/second")

    # Known Brands (Layer 1 only)
    if known_brands_times:
        avg_known_brands = sum(known_brands_times) / len(known_brands_times)
        print(f"\n🔹 Layer 1 - Known Brands ({len(known_brands_times)} cases):")
        print(f"   Average: {avg_known_brands:6.2f}ms")
        print(f"   Min:     {min(known_brands_times):6.2f}ms")
        print(f"   Max:     {max(known_brands_times):6.2f}ms")
        print(f"   QPS:     {1000/avg_known_brands:.1f} queries/second")

    # ML Fallback (Layer 2 only)
    if ml_fallback_times:
        avg_ml_fallback = sum(ml_fallback_times) / len(ml_fallback_times)
        print(f"\n🔹 Layer 2 - ML Fallback ({len(ml_fallback_times)} cases):")
        print(f"   Average: {avg_ml_fallback:6.2f}ms")
        print(f"   Min:     {min(ml_fallback_times):6.2f}ms")
        print(f"   Max:     {max(ml_fallback_times):6.2f}ms")
        print(f"   QPS:     {1000/avg_ml_fallback:.1f} queries/second")

    # Mixed (Both layers)
    if mixed_times:
        avg_mixed = sum(mixed_times) / len(mixed_times)
        print(f"\n🔹 Mixed - Product + ML ({len(mixed_times)} cases):")
        print(f"   Average: {avg_mixed:6.2f}ms")
        print(f"   Min:     {min(mixed_times):6.2f}ms")
        print(f"   Max:     {max(mixed_times):6.2f}ms")
        print(f"   QPS:     {1000/avg_mixed:.1f} queries/second")

    # Complex (Multiple ML attributes)
    if complex_times:
        avg_complex = sum(complex_times) / len(complex_times)
        print(f"\n🔹 Complex - Multiple ML ({len(complex_times)} cases):")
        print(f"   Average: {avg_complex:6.2f}ms")
        print(f"   Min:     {min(complex_times):6.2f}ms")
        print(f"   Max:     {max(complex_times):6.2f}ms")
        print(f"   QPS:     {1000/avg_complex:.1f} queries/second")

    # Overall statistics
    all_times = [t for _, _, t in all_results]
    avg_overall = sum(all_times) / len(all_times)
    total_time = sum(all_times)

    print(f"\n🚀 OVERALL COMBINED NER PERFORMANCE:")
    print("-" * 50)
    print(f"   Total Queries: {len(test_cases)}")
    print(f"   Total Time: {total_time:6.2f}ms")
    print(f"   Average Time: {avg_overall:6.2f}ms per query")
    print(f"   Overall QPS: {1000/avg_overall:.1f} queries/second")

    # Performance breakdown
    print(f"\n📈 PERFORMANCE BREAKDOWN:")
    print("-" * 30)
    print(
        f"   Layer 1 (Database): {len(known_products_times) + len(known_brands_times)} queries"
    )
    print(
        f"   Layer 2 (ML Model): {len(ml_fallback_times) + len(complex_times)} queries"
    )
    print(f"   Mixed (Both): {len(mixed_times)} queries")

    # Show some examples
    print(f"\n📝 EXAMPLE RESULTS:")
    print("-" * 30)
    for i, (text, result, time_ms) in enumerate(all_results[:5], 1):
        print(f"{i}. '{text[:40]}{'...' if len(text) > 40 else ''}'")
        print(f"   Time: {time_ms:6.2f}ms | Slots: {result.slots}")


if __name__ == "__main__":
    test_combined_ner_speed()
