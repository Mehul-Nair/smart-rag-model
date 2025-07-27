#!/usr/bin/env python3
"""
Check intent collisions using sentence transformers
"""

import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict


def load_training_data():
    """Load all training data from JSONL files"""
    training_dir = "data/training/intent"
    intent_data = defaultdict(list)

    for filename in os.listdir(training_dir):
        if filename.endswith(".jsonl"):
            intent_name = filename.replace(".jsonl", "").upper()
            filepath = os.path.join(training_dir, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        intent_data[intent_name].append(data["text"])
                    except json.JSONDecodeError as e:
                        print(
                            f"   ⚠️  Skipping invalid JSON in {filename} line {line_num}: {e}"
                        )
                        continue

    return intent_data


def analyze_intent_collisions():
    """Analyze intent collisions using sentence transformers"""

    print("🔍 ANALYZING INTENT COLLISIONS")
    print("=" * 50)

    # Load training data
    print("📂 Loading training data...")
    intent_data = load_training_data()

    # Initialize sentence transformer
    print("🤖 Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Calculate embeddings for each intent
    print("🧮 Calculating embeddings...")
    intent_embeddings = {}
    intent_centroids = {}

    for intent_name, texts in intent_data.items():
        if texts:
            # Get embeddings for all texts in this intent
            embeddings = model.encode(texts)
            intent_embeddings[intent_name] = embeddings

            # Calculate centroid (mean embedding)
            centroid = np.mean(embeddings, axis=0)
            intent_centroids[intent_name] = centroid

            print(f"   {intent_name}: {len(texts)} examples")

    # Calculate pairwise similarities between intent centroids
    print(f"\n📊 INTENT CENTROID SIMILARITIES")
    print("=" * 50)

    intent_names = list(intent_centroids.keys())
    similarity_matrix = np.zeros((len(intent_names), len(intent_names)))

    for i, intent1 in enumerate(intent_names):
        for j, intent2 in enumerate(intent_names):
            if i <= j:  # Only calculate upper triangle
                centroid1 = intent_centroids[intent1]
                centroid2 = intent_centroids[intent2]

                similarity = cosine_similarity([centroid1], [centroid2])[0][0]
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity

                if i != j:  # Don't print self-similarity
                    print(f"{intent1:15} ↔ {intent2:15}: {similarity:.3f}")

    # Find high similarity pairs (potential collisions)
    print(f"\n⚠️  POTENTIAL INTENT COLLISIONS (similarity > 0.7)")
    print("=" * 50)

    high_similarity_pairs = []
    for i, intent1 in enumerate(intent_names):
        for j, intent2 in enumerate(intent_names):
            if i < j:  # Only check each pair once
                similarity = similarity_matrix[i][j]
                if similarity > 0.7:
                    high_similarity_pairs.append((intent1, intent2, similarity))
                    print(f"🚨 {intent1:15} ↔ {intent2:15}: {similarity:.3f}")

    if not high_similarity_pairs:
        print("✅ No high similarity pairs found!")

    # Analyze individual examples that might be misclassified
    print(f"\n🔍 ANALYZING INDIVIDUAL EXAMPLES")
    print("=" * 50)

    # Check examples that might be close to other intent centroids
    for intent_name, texts in intent_data.items():
        if intent_name in intent_embeddings:
            embeddings = intent_embeddings[intent_name]

            print(f"\n📝 Checking {intent_name} examples:")

            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                # Calculate similarity to all intent centroids
                similarities = {}
                for other_intent, centroid in intent_centroids.items():
                    if other_intent != intent_name:
                        similarity = cosine_similarity([embedding], [centroid])[0][0]
                        similarities[other_intent] = similarity

                # Find the most similar other intent
                if similarities:
                    most_similar_intent = max(similarities.items(), key=lambda x: x[1])
                    if most_similar_intent[1] > 0.8:  # High similarity threshold
                        print(
                            f"   ⚠️  '{text[:50]}...' → {most_similar_intent[0]} ({most_similar_intent[1]:.3f})"
                        )

    # Calculate intra-intent vs inter-intent similarities
    print(f"\n📈 INTRA vs INTER INTENT SIMILARITIES")
    print("=" * 50)

    for intent_name in intent_names:
        if intent_name in intent_embeddings:
            embeddings = intent_embeddings[intent_name]

            # Intra-intent similarity (within same intent)
            if len(embeddings) > 1:
                intra_similarities = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                        intra_similarities.append(sim)

                avg_intra = np.mean(intra_similarities)

                # Inter-intent similarity (with other intents)
                inter_similarities = []
                for other_intent, other_embeddings in intent_embeddings.items():
                    if other_intent != intent_name:
                        for emb1 in embeddings:
                            for emb2 in other_embeddings:
                                sim = cosine_similarity([emb1], [emb2])[0][0]
                                inter_similarities.append(sim)

                avg_inter = np.mean(inter_similarities)

                print(
                    f"{intent_name:15}: Intra={avg_intra:.3f}, Inter={avg_inter:.3f}, Gap={avg_intra-avg_inter:.3f}"
                )

    return intent_data, intent_centroids, similarity_matrix


if __name__ == "__main__":
    analyze_intent_collisions()
