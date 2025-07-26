#!/usr/bin/env python3
"""
Comprehensive validation of the NER dataset
"""

import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple


def validate_ner_dataset(file_path: str = "data/training/ner/ner_data.jsonl"):
    """Validate the NER dataset for issues"""

    print("🔍 VALIDATING NER DATASET")
    print("=" * 60)

    issues = []
    stats = {
        "total_samples": 0,
        "total_entities": 0,
        "entity_types": Counter(),
        "text_lengths": [],
        "entity_spans": [],
        "overlapping_entities": 0,
        "invalid_spans": 0,
        "empty_entities": 0,
        "duplicate_ids": set(),
        "missing_ids": 0,
    }

    # Read and validate each line
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                stats["total_samples"] += 1

                # Check for required fields
                if "text" not in data:
                    issues.append(f"Line {line_num}: Missing 'text' field")
                    continue

                if "entities" not in data:
                    issues.append(f"Line {line_num}: Missing 'entities' field")
                    continue

                text = data["text"]
                entities = data["entities"]
                sample_id = data.get("id", f"line_{line_num}")

                # Check for duplicate IDs
                if sample_id in stats["duplicate_ids"]:
                    issues.append(f"Line {line_num}: Duplicate ID '{sample_id}'")
                stats["duplicate_ids"].add(sample_id)

                # Validate text
                if not text or not isinstance(text, str):
                    issues.append(f"Line {line_num}: Invalid or empty text")
                    continue

                text_length = len(text)
                stats["text_lengths"].append(text_length)

                # Validate entities
                if not isinstance(entities, list):
                    issues.append(f"Line {line_num}: Entities must be a list")
                    continue

                # Check for overlapping entities
                entity_spans = []
                for entity_idx, entity in enumerate(entities):
                    # Validate entity structure
                    if not isinstance(entity, dict):
                        issues.append(
                            f"Line {line_num}: Entity {entity_idx} is not a dictionary"
                        )
                        continue

                    required_fields = ["start", "end", "label", "text"]
                    for field in required_fields:
                        if field not in entity:
                            issues.append(
                                f"Line {line_num}: Entity {entity_idx} missing '{field}' field"
                            )
                            continue

                    start = entity["start"]
                    end = entity["end"]
                    label = entity["label"]
                    entity_text = entity["text"]

                    # Validate span indices
                    if not isinstance(start, int) or not isinstance(end, int):
                        issues.append(
                            f"Line {line_num}: Entity {entity_idx} has non-integer start/end"
                        )
                        continue

                    if start < 0 or end > text_length:
                        issues.append(
                            f"Line {line_num}: Entity {entity_idx} span out of bounds ({start}-{end} for text length {text_length})"
                        )
                        stats["invalid_spans"] += 1
                        continue

                    if start >= end:
                        issues.append(
                            f"Line {line_num}: Entity {entity_idx} invalid span (start >= end)"
                        )
                        stats["invalid_spans"] += 1
                        continue

                    # Check if entity text matches span
                    span_text = text[start:end]
                    if span_text != entity_text:
                        issues.append(
                            f"Line {line_num}: Entity {entity_idx} text mismatch: '{entity_text}' vs span '{span_text}'"
                        )
                        continue

                    # Check for empty entities
                    if not entity_text.strip():
                        issues.append(
                            f"Line {line_num}: Entity {entity_idx} has empty text"
                        )
                        stats["empty_entities"] += 1
                        continue

                    # Validate label
                    valid_labels = {
                        "PRODUCT_TYPE",
                        "MATERIAL",
                        "COLOR",
                        "SIZE",
                        "BRAND",
                        "STYLE",
                        "ROOM",
                        "BUDGET",
                        "PRODUCT_NAME",
                    }
                    if label not in valid_labels:
                        issues.append(
                            f"Line {line_num}: Entity {entity_idx} invalid label '{label}'"
                        )
                        continue

                    # Track entity stats
                    stats["total_entities"] += 1
                    stats["entity_types"][label] += 1
                    entity_spans.append((start, end, label))

                # Check for overlapping entities
                entity_spans.sort()
                for i in range(len(entity_spans) - 1):
                    start1, end1, _ = entity_spans[i]
                    start2, end2, _ = entity_spans[i + 1]
                    if end1 > start2:
                        issues.append(
                            f"Line {line_num}: Overlapping entities ({start1}-{end1}) and ({start2}-{end2})"
                        )
                        stats["overlapping_entities"] += 1

                stats["entity_spans"].extend(entity_spans)

            except json.JSONDecodeError as e:
                issues.append(f"Line {line_num}: JSON decode error - {e}")
            except Exception as e:
                issues.append(f"Line {line_num}: Unexpected error - {e}")

    # Generate validation report
    print(f"\n📊 DATASET STATISTICS:")
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Total entities: {stats['total_entities']:,}")
    if stats["total_samples"] > 0:
        print(
            f"Average entities per sample: {stats['total_entities'] / stats['total_samples']:.2f}"
        )

    print(f"\n📏 TEXT LENGTH STATISTICS:")
    if stats["text_lengths"]:
        print(f"Min length: {min(stats['text_lengths'])}")
        print(f"Max length: {max(stats['text_lengths'])}")
        print(
            f"Average length: {sum(stats['text_lengths']) / len(stats['text_lengths']):.2f}"
        )

    print(f"\n🏷️ ENTITY TYPE DISTRIBUTION:")
    for entity_type, count in stats["entity_types"].most_common():
        if stats["total_entities"] > 0:
            percentage = (count / stats["total_entities"]) * 100
            print(f"{entity_type}: {count:,} ({percentage:.1f}%)")
        else:
            print(f"{entity_type}: {count:,}")

    print(f"\n⚠️ VALIDATION ISSUES:")
    if issues:
        print(f"Found {len(issues)} issues:")
        for i, issue in enumerate(issues[:20], 1):  # Show first 20 issues
            print(f"{i:2d}. {issue}")
        if len(issues) > 20:
            print(f"... and {len(issues) - 20} more issues")
    else:
        print("✅ No issues found! Dataset is valid.")

    print(f"\n🔍 QUALITY METRICS:")
    print(f"Invalid spans: {stats['invalid_spans']}")
    print(f"Empty entities: {stats['empty_entities']}")
    print(f"Overlapping entities: {stats['overlapping_entities']}")
    print(f"Duplicate IDs: {len(stats['duplicate_ids'])}")

    # Sample analysis
    print(f"\n📋 SAMPLE ANALYSIS:")
    print("Checking for common patterns and potential issues...")

    # Check for very short or long texts
    if stats["text_lengths"]:
        short_texts = [l for l in stats["text_lengths"] if l < 10]
        long_texts = [l for l in stats["text_lengths"] if l > 200]
        print(f"Very short texts (<10 chars): {len(short_texts)}")
        print(f"Very long texts (>200 chars): {len(long_texts)}")

    # Check entity distribution balance
    entity_counts = list(stats["entity_types"].values())
    if entity_counts:
        min_entities = min(entity_counts)
        max_entities = max(entity_counts)
        balance_ratio = min_entities / max_entities if max_entities > 0 else 1
        print(f"Entity balance ratio (min/max): {balance_ratio:.3f}")

        if balance_ratio < 0.1:
            print("⚠️  Warning: Very imbalanced entity distribution")
        elif balance_ratio < 0.3:
            print("⚠️  Warning: Somewhat imbalanced entity distribution")
        else:
            print("✅ Good entity distribution balance")

    return len(issues) == 0, issues


def analyze_sample_quality(
    file_path: str = "data/training/ner/ner_data.jsonl", num_samples: int = 10
):
    """Analyze quality of random samples"""

    print(f"\n🔍 SAMPLE QUALITY ANALYSIS (showing {num_samples} random samples)")
    print("=" * 60)

    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))

    # Show random samples
    import random

    random.seed(42)  # For reproducible results
    selected_samples = random.sample(samples, min(num_samples, len(samples)))

    for i, sample in enumerate(selected_samples, 1):
        print(f"\n{i}. ID: {sample.get('id', 'N/A')}")
        print(f"   Text: \"{sample['text']}\"")
        print(f"   Entities: {len(sample['entities'])}")
        for j, entity in enumerate(sample["entities"], 1):
            print(
                f"     {j}. {entity['label']}: \"{entity['text']}\" ({entity['start']}-{entity['end']})"
            )


if __name__ == "__main__":
    # Validate the dataset
    is_valid, issues = validate_ner_dataset()

    # Analyze sample quality
    analyze_sample_quality()

    print(f"\n{'='*60}")
    if is_valid:
        print("✅ DATASET VALIDATION PASSED")
    else:
        print("❌ DATASET VALIDATION FAILED")
        print(f"Found {len(issues)} issues that need to be addressed")
    print(f"{'='*60}")
