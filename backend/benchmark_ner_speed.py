#!/usr/bin/env python3
"""
Benchmark NER Speed Comparison
"""

import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "rag", "intent_modules"))

from dynamic_ner_classifier import DynamicNERClassifier
from onnx_ner_classifier import ONNXNERClassifier


def benchmark_ner_speed():
    """Benchmark the speed of different NER approaches"""

    print("🏁 BENCHMARKING NER SPEED")
    print("=" * 60)

    # Test cases
    test_cases = [
        "give me details of city lights ceiling light",
        "give me details of crystal chandelier ?",
        "give me details of lighting up lisbon chandelier",
        "I want Pure Royale curtains",
        "Show me White Teak by Asian Paints lights",
        "tell me about drop dead gorgeous chandelier",
        "what is time to shine chandelier",
        "show me keeper of my heart chandelier",
        "give me details of bolt chandelier",
        "I need a blue ceiling light for my living room",
        "show me wooden wall lights under 5000",
        "find me a gold chandelier for dining room",
        "what are the best LED pendant lights?",
        "give me options for bathroom vanity lights",
        "show me outdoor wall lights with motion sensor",
    ]

    # Initialize classifiers
    print("🔧 Initializing classifiers...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "trained_deberta_ner_model")

    # Dynamic NER Classifier
    print("📊 Loading Dynamic NER Classifier...")
    dynamic_start = time.time()
    dynamic_classifier = DynamicNERClassifier(model_path)
    dynamic_load_time = time.time() - dynamic_start

    # Original ONNX NER Classifier
    print("📊 Loading Original ONNX NER Classifier...")
    onnx_start = time.time()
    onnx_classifier = ONNXNERClassifier(model_path)
    onnx_load_time = time.time() - onnx_start

    print(f"✅ Dynamic NER load time: {dynamic_load_time:.3f}s")
    print(f"✅ Original ONNX load time: {onnx_load_time:.3f}s")

    # Benchmark Dynamic NER
    print(f"\n🏃‍♂️ Benchmarking Dynamic NER Classifier...")
    print("-" * 50)

    dynamic_times = []
    dynamic_results = []

    for i, text in enumerate(test_cases, 1):
        start_time = time.time()
        result = dynamic_classifier.extract_entities(text)
        end_time = time.time()

        processing_time = end_time - start_time
        dynamic_times.append(processing_time)
        dynamic_results.append(result)

        print(f"Test {i:2d}: {processing_time*1000:6.2f}ms - {text[:40]}...")

    # Benchmark Original ONNX NER
    print(f"\n🏃‍♂️ Benchmarking Original ONNX NER Classifier...")
    print("-" * 50)

    onnx_times = []
    onnx_results = []

    for i, text in enumerate(test_cases, 1):
        start_time = time.time()
        result = onnx_classifier.extract_entities(text)
        end_time = time.time()

        processing_time = end_time - start_time
        onnx_times.append(processing_time)
        onnx_results.append(result)

        print(f"Test {i:2d}: {processing_time*1000:6.2f}ms - {text[:40]}...")

    # Calculate statistics
    print(f"\n📊 SPEED COMPARISON RESULTS")
    print("=" * 60)

    # Dynamic NER Stats
    dynamic_avg = sum(dynamic_times) / len(dynamic_times)
    dynamic_min = min(dynamic_times)
    dynamic_max = max(dynamic_times)
    dynamic_total = sum(dynamic_times)

    # ONNX NER Stats
    onnx_avg = sum(onnx_times) / len(onnx_times)
    onnx_min = min(onnx_times)
    onnx_max = max(onnx_times)
    onnx_total = sum(onnx_times)

    print(f"🔹 Dynamic NER Classifier:")
    print(f"   Average: {dynamic_avg*1000:6.2f}ms per query")
    print(f"   Min:     {dynamic_min*1000:6.2f}ms")
    print(f"   Max:     {dynamic_max*1000:6.2f}ms")
    print(f"   Total:   {dynamic_total*1000:6.2f}ms for {len(test_cases)} queries")
    print(f"   QPS:     {len(test_cases)/dynamic_total:.1f} queries/second")

    print(f"\n🔹 Original ONNX NER Classifier:")
    print(f"   Average: {onnx_avg*1000:6.2f}ms per query")
    print(f"   Min:     {onnx_min*1000:6.2f}ms")
    print(f"   Max:     {onnx_max*1000:6.2f}ms")
    print(f"   Total:   {onnx_total*1000:6.2f}ms for {len(test_cases)} queries")
    print(f"   QPS:     {len(test_cases)/onnx_total:.1f} queries/second")

    # Speed comparison
    speed_improvement = (onnx_avg - dynamic_avg) / onnx_avg * 100
    print(f"\n🚀 SPEED IMPROVEMENT:")
    print(
        f"   Dynamic NER is {speed_improvement:.1f}% {'faster' if speed_improvement > 0 else 'slower'} than Original ONNX"
    )

    if speed_improvement > 0:
        print(f"   Time saved per query: {(onnx_avg - dynamic_avg)*1000:.2f}ms")
        print(
            f"   Time saved for {len(test_cases)} queries: {(onnx_total - dynamic_total)*1000:.2f}ms"
        )

    # Accuracy comparison (simple check)
    print(f"\n🎯 ACCURACY COMPARISON:")
    print("-" * 30)

    dynamic_product_names = sum(1 for r in dynamic_results if "PRODUCT_NAME" in r.slots)
    onnx_product_names = sum(1 for r in onnx_results if "PRODUCT_NAME" in r.slots)

    print(
        f"Dynamic NER found PRODUCT_NAME in {dynamic_product_names}/{len(test_cases)} cases"
    )
    print(
        f"Original ONNX found PRODUCT_NAME in {onnx_product_names}/{len(test_cases)} cases"
    )

    # Show detailed comparison for first few cases
    print(f"\n📝 DETAILED COMPARISON (First 5 cases):")
    print("-" * 50)

    for i in range(min(5, len(test_cases))):
        print(f"\nCase {i+1}: '{test_cases[i]}'")
        print(f"  Dynamic: {dynamic_results[i].slots}")
        print(f"  ONNX:    {onnx_results[i].slots}")

    # Performance stats from classifiers
    print(f"\n📈 CLASSIFIER PERFORMANCE STATS:")
    print("-" * 40)

    dynamic_stats = dynamic_classifier.get_performance_stats()
    onnx_stats = onnx_classifier.get_performance_stats()

    print(f"Dynamic NER Stats: {dynamic_stats}")
    print(f"ONNX NER Stats: {onnx_stats}")


if __name__ == "__main__":
    benchmark_ner_speed()
