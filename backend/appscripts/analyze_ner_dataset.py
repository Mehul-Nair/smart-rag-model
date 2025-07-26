#!/usr/bin/env python3
"""
Comprehensive analysis of the NER training dataset
"""

import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple


def load_dataset(file_path: str) -> List[Dict]:
    """Load the JSONL dataset"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def analyze_dataset(data: List[Dict]) -> Dict:
    """Comprehensive dataset analysis"""

    # Basic statistics
    total_samples = len(data)
    total_tokens = sum(len(sample["tokens"]) for sample in data)
    avg_tokens_per_sample = total_tokens / total_samples

    # Label analysis
    all_labels = []
    label_counts = Counter()
    entity_counts = Counter()

    # Sample categories
    sample_categories = defaultdict(int)

    # Token length distribution
    token_lengths = []

    # Entity span analysis
    entity_spans = []

    for sample in data:
        # Category analysis
        sample_id = sample["id"]
        if sample_id.startswith("balanced_"):
            sample_categories["balanced"] += 1
        elif sample_id.startswith("discovery_"):
            sample_categories["discovery"] += 1
        elif sample_id.startswith("indian_augment_"):
            sample_categories["indian_augment"] += 1
        elif sample_id.startswith("gen_"):
            sample_categories["generated"] += 1
        elif sample_id.startswith("ex_"):
            sample_categories["extended"] += 1
        else:
            sample_categories["other"] += 1

        # Token length
        token_lengths.append(len(sample["tokens"]))

        # Label analysis
        labels = sample["labels"]
        all_labels.extend(labels)
        label_counts.update(labels)

        # Entity analysis
        current_entity = None
        entity_start = None

        for i, label in enumerate(labels):
            if label.startswith("B-"):
                # End previous entity if exists
                if current_entity:
                    entity_spans.append((current_entity, entity_start, i - 1))

                # Start new entity
                current_entity = label[2:]  # Remove 'B-' prefix
                entity_start = i
                entity_counts[current_entity] += 1

            elif label.startswith("I-"):
                # Continue current entity
                if current_entity and label[2:] == current_entity:
                    continue
                else:
                    # Inconsistent labeling
                    if current_entity:
                        entity_spans.append((current_entity, entity_start, i - 1))
                    current_entity = None
                    entity_start = None

            else:  # 'O' label
                # End current entity if exists
                if current_entity:
                    entity_spans.append((current_entity, entity_start, i - 1))
                    current_entity = None
                    entity_start = None

        # Handle entity at end of sequence
        if current_entity:
            entity_spans.append((current_entity, entity_start, len(labels) - 1))

    # Entity span length analysis
    entity_lengths = [end - start + 1 for _, start, end in entity_spans]

    # Label distribution analysis
    label_distribution = dict(label_counts.most_common())

    # Entity type distribution
    entity_distribution = dict(entity_counts.most_common())

    # Quality metrics
    quality_metrics = {
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": avg_tokens_per_sample,
        "unique_labels": len(set(all_labels)),
        "unique_entities": len(entity_distribution),
        "total_entities": sum(entity_distribution.values()),
        "avg_entities_per_sample": sum(entity_distribution.values()) / total_samples,
        "avg_entity_length": (
            sum(entity_lengths) / len(entity_lengths) if entity_lengths else 0
        ),
        "max_entity_length": max(entity_lengths) if entity_lengths else 0,
        "min_entity_length": min(entity_lengths) if entity_lengths else 0,
        "label_distribution": label_distribution,
        "entity_distribution": entity_distribution,
        "sample_categories": dict(sample_categories),
        "token_length_stats": {
            "min": min(token_lengths),
            "max": max(token_lengths),
            "avg": sum(token_lengths) / len(token_lengths),
            "median": sorted(token_lengths)[len(token_lengths) // 2],
        },
    }

    return quality_metrics


def analyze_labeling_consistency(data: List[Dict]) -> Dict:
    """Analyze labeling consistency and potential issues"""

    issues = {
        "inconsistent_entities": [],
        "orphaned_i_labels": [],
        "missing_b_labels": [],
        "overlapping_entities": [],
    }

    for i, sample in enumerate(data):
        labels = sample["labels"]
        tokens = sample["tokens"]

        # Check for orphaned I-labels (I- without preceding B-)
        for j, label in enumerate(labels):
            if label.startswith("I-"):
                if (
                    j == 0
                    or not labels[j - 1].startswith(("B-", "I-"))
                    or labels[j - 1][2:] != label[2:]
                ):
                    issues["orphaned_i_labels"].append(
                        {
                            "sample_id": sample["id"],
                            "position": j,
                            "token": tokens[j],
                            "label": label,
                        }
                    )

        # Check for missing B-labels in entity sequences
        current_entity = None
        for j, label in enumerate(labels):
            if label.startswith("I-"):
                if current_entity is None:
                    issues["missing_b_labels"].append(
                        {
                            "sample_id": sample["id"],
                            "position": j,
                            "token": tokens[j],
                            "label": label,
                        }
                    )
                elif label[2:] != current_entity:
                    issues["inconsistent_entities"].append(
                        {
                            "sample_id": sample["id"],
                            "position": j,
                            "token": tokens[j],
                            "expected": current_entity,
                            "found": label[2:],
                        }
                    )
            elif label.startswith("B-"):
                current_entity = label[2:]
            else:
                current_entity = None

    return issues


def generate_report(metrics: Dict, issues: Dict) -> str:
    """Generate a comprehensive report"""

    report = []
    report.append("=" * 80)
    report.append("NER TRAINING DATASET ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # Basic Statistics
    report.append("📊 BASIC STATISTICS")
    report.append("-" * 40)
    report.append(f"Total samples: {metrics['total_samples']:,}")
    report.append(f"Total tokens: {metrics['total_tokens']:,}")
    report.append(f"Average tokens per sample: {metrics['avg_tokens_per_sample']:.2f}")
    report.append(f"Unique labels: {metrics['unique_labels']}")
    report.append(f"Unique entity types: {metrics['unique_entities']}")
    report.append(f"Total entities: {metrics['total_entities']:,}")
    report.append(
        f"Average entities per sample: {metrics['avg_entities_per_sample']:.2f}"
    )
    report.append("")

    # Sample Categories
    report.append("📁 SAMPLE CATEGORIES")
    report.append("-" * 40)
    for category, count in metrics["sample_categories"].items():
        percentage = (count / metrics["total_samples"]) * 100
        report.append(
            f"{category.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)"
        )
    report.append("")

    # Token Length Statistics
    report.append("📏 TOKEN LENGTH STATISTICS")
    report.append("-" * 40)
    stats = metrics["token_length_stats"]
    report.append(f"Minimum: {stats['min']}")
    report.append(f"Maximum: {stats['max']}")
    report.append(f"Average: {stats['avg']:.2f}")
    report.append(f"Median: {stats['median']}")
    report.append("")

    # Entity Distribution
    report.append("🏷️  ENTITY TYPE DISTRIBUTION")
    report.append("-" * 40)
    for entity, count in list(metrics["entity_distribution"].items())[:15]:
        percentage = (count / metrics["total_entities"]) * 100
        report.append(f"{entity}: {count:,} ({percentage:.1f}%)")
    if len(metrics["entity_distribution"]) > 15:
        report.append(
            f"... and {len(metrics['entity_distribution']) - 15} more entity types"
        )
    report.append("")

    # Label Distribution
    report.append("🏷️  LABEL DISTRIBUTION")
    report.append("-" * 40)
    for label, count in list(metrics["label_distribution"].items())[:10]:
        percentage = (count / metrics["total_tokens"]) * 100
        report.append(f"{label}: {count:,} ({percentage:.1f}%)")
    if len(metrics["label_distribution"]) > 10:
        report.append(f"... and {len(metrics['label_distribution']) - 10} more labels")
    report.append("")

    # Quality Issues
    report.append("⚠️  QUALITY ISSUES")
    report.append("-" * 40)
    report.append(f"Orphaned I-labels: {len(issues['orphaned_i_labels'])}")
    report.append(f"Missing B-labels: {len(issues['missing_b_labels'])}")
    report.append(f"Inconsistent entities: {len(issues['inconsistent_entities'])}")
    report.append("")

    # Sample Issues
    if issues["orphaned_i_labels"]:
        report.append("🔍 SAMPLE ISSUES (First 5)")
        report.append("-" * 40)
        for issue in issues["orphaned_i_labels"][:5]:
            report.append(
                f"Sample {issue['sample_id']}: Token '{issue['token']}' has {issue['label']} without B-"
            )
        report.append("")

    # Recommendations
    report.append("💡 RECOMMENDATIONS")
    report.append("-" * 40)

    if len(issues["orphaned_i_labels"]) > 0:
        report.append("• Fix orphaned I-labels by adding appropriate B-labels")

    if len(issues["missing_b_labels"]) > 0:
        report.append("• Add missing B-labels for entity sequences")

    if len(issues["inconsistent_entities"]) > 0:
        report.append("• Ensure consistent entity type labeling within sequences")

    # Check for class imbalance
    entity_counts = list(metrics["entity_distribution"].values())
    if entity_counts:
        max_count = max(entity_counts)
        min_count = min(entity_counts)
        if max_count / min_count > 10:
            report.append("• Consider balancing entity type distribution")

    # Check for short sequences
    if metrics["token_length_stats"]["avg"] < 5:
        report.append("• Consider longer training sequences for better context")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    """Main analysis function"""

    # Load dataset
    print("Loading dataset...")
    data = load_dataset("data/training/ner_training_data.jsonl")

    # Analyze dataset
    print("Analyzing dataset...")
    metrics = analyze_dataset(data)

    # Analyze labeling consistency
    print("Checking labeling consistency...")
    issues = analyze_labeling_consistency(data)

    # Generate report
    print("Generating report...")
    report = generate_report(metrics, issues)

    # Save report
    with open("backend/ner_dataset_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # Print report
    print(report)

    # Save detailed metrics as JSON
    with open("backend/ner_dataset_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": metrics, "issues": issues}, f, indent=2, ensure_ascii=False
        )

    print(f"\nDetailed metrics saved to: backend/ner_dataset_metrics.json")
    print(f"Report saved to: backend/ner_dataset_analysis_report.txt")


if __name__ == "__main__":
    main()
