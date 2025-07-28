#!/usr/bin/env python3
"""
Create better NER training data with proper product name annotations
"""

import json
import os


def create_better_ner_training_data():
    """Create improved NER training data with proper product name annotations"""

    # Get product titles from the Excel file
    import pandas as pd

    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(current_dir, "data", "BH_PD.xlsx")

    df = pd.read_excel(excel_path)

    # Extract product titles and create proper NER annotations
    training_examples = []

    # Sample product titles that need proper annotation
    sample_titles = [
        "City Lights (Dimmable LED With Remote Control) Ceiling Light",
        "Lighting Up Lisbon (Small, Dimmable LED With Remote Control) Crystal Chandelier",
        "Lights Out Gold (Built-In LED) Vanity Light",
        "Drop Dead Gorgeous (Gold) Crystal Chandelier",
        "Time To Shine (Medium, Antique Gold Foil Gilded) Chandelier",
        "Keeper Of My Heart Chandelier",
        "Bolt (Dimmable LED with Remote Control) Chandelier",
        "Positive Energy Wooden Wall Light",
        "Purity Pendant Light",
        "Skyfall Floor Lamp",
        "Captured Pendant Light",
        "Mini Mammoth (Kid's Room, Built-In LED) Wall Light",
        "Cloudy Bay (Kid's Room, Dimmable LED with Remote Control ) Ceiling Light",
        "Undivided (3 Colour, Dimmable LED with Remote Control) Ceiling Light",
        "Party Favor Ceiling Light",
        "Word Play (Dimmable LED With Remote Control) Ceiling Light",
        "Wish You Well Ceiling Light",
        "Oakridge (Built-In LED ) Outdoor/ Indoor Wall Light (IP65 Rated)",
        "Sunlit (Cream) Gold Foil Finish Wall Light",
        "Insightful (White, Built-In LED) Wall Light",
    ]

    # Known brands
    known_brands = [
        "Pure Royale",
        "White Teak",
        "Asian Paints",
        "Bathsense",
        "Ador",
        "Royale",
    ]

    print("🔧 Creating better NER training data...")

    # Create training examples for product names
    for title in sample_titles:
        # Create user queries that would ask for these products
        queries = [
            f"give me details of {title.lower()}",
            f"show me {title.lower()}",
            f"tell me about {title.lower()}",
            f"I want to see {title.lower()}",
            f"what is {title.lower()}",
            f"give me information about {title.lower()}",
        ]

        for query in queries:
            # Tokenize the query
            tokens = query.split()
            labels = ["O"] * len(tokens)  # Start with all O labels

            # Find the product name in the query
            title_lower = title.lower()
            query_lower = query.lower()

            # Find the start and end of the product name in the query
            start_idx = query_lower.find(title_lower)
            if start_idx != -1:
                # Calculate token positions
                start_token = len(query[:start_idx].split())
                end_token = start_token + len(title_lower.split())

                # Mark the product name tokens
                for i in range(start_token, min(end_token, len(tokens))):
                    if i == start_token:
                        labels[i] = "B-PRODUCT_NAME"
                    else:
                        labels[i] = "I-PRODUCT_NAME"

            # Create training example
            example = {
                "id": f"product_name_{len(training_examples):03d}",
                "tokens": tokens,
                "labels": labels,
            }
            training_examples.append(example)

    # Create training examples for brand detection
    for brand in known_brands:
        queries = [
            f"I want {brand} curtains",
            f"Show me {brand} lights",
            f"Find {brand} products",
            f"Give me {brand} options",
            f"Looking for {brand} furniture",
        ]

        for query in queries:
            tokens = query.split()
            labels = ["O"] * len(tokens)

            # Find brand in query
            brand_lower = brand.lower()
            query_lower = query.lower()

            start_idx = query_lower.find(brand_lower)
            if start_idx != -1:
                start_token = len(query[:start_idx].split())
                end_token = start_token + len(brand_lower.split())

                for i in range(start_token, min(end_token, len(tokens))):
                    if i == start_token:
                        labels[i] = "B-BRAND"
                    else:
                        labels[i] = "I-BRAND"

            example = {
                "id": f"brand_{len(training_examples):03d}",
                "tokens": tokens,
                "labels": labels,
            }
            training_examples.append(example)

    # Save the improved training data
    output_file = os.path.join(
        current_dir, "data", "training", "improved_ner_training_data.jsonl"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        for example in training_examples:
            f.write(json.dumps(example) + "\n")

    print(f"✅ Created {len(training_examples)} improved training examples")
    print(f"📁 Saved to: {output_file}")

    # Show some examples
    print(f"\n📝 Sample improved training examples:")
    print("-" * 60)
    for i, example in enumerate(training_examples[:5], 1):
        print(f"{i}. Query: {' '.join(example['tokens'])}")
        print(f"   Labels: {example['labels']}")
        print()


if __name__ == "__main__":
    create_better_ner_training_data()
