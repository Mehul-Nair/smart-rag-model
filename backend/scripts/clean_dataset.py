#!/usr/bin/env python3
"""
Dataset Cleaning Script

This script cleans and prepares the intent classification dataset.
"""

import pandas as pd
import json
import re
from typing import List, Dict
import argparse


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^\w\s.,!?]", "", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def validate_intent(intent: str, valid_intents: List[str]) -> str:
    """Validate and normalize intent labels"""
    if not isinstance(intent, str):
        return "INVALID"

    # Convert to uppercase for consistency
    intent = intent.upper()

    # Check if intent is valid
    if intent in valid_intents:
        return intent
    else:
        # Try to find closest match
        for valid_intent in valid_intents:
            if valid_intent in intent or intent in valid_intent:
                return valid_intent
        return "INVALID"


def clean_dataset(input_path: str, output_path: str, valid_intents: List[str] = None):
    """Clean the dataset"""
    print(f"Loading dataset from {input_path}")

    # Load dataset
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".json"):
        with open(input_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON.")

    print(f"Original dataset shape: {df.shape}")

    # Define valid intents if not provided
    if valid_intents is None:
        valid_intents = [
            "GREETING",
            "HELP",
            "CATEGORY_LIST",
            "PRODUCT_SEARCH",
            "BUDGET_QUERY",
            "PRODUCT_DETAIL",
            "WARRANTY_QUERY",
            "INVALID",
            "CLARIFY",
        ]

    # Clean text column
    if "text" in df.columns:
        df["text"] = df["text"].apply(clean_text)

    # Clean intent column
    if "intent" in df.columns:
        df["intent"] = df["intent"].apply(lambda x: validate_intent(x, valid_intents))

    # Remove rows with empty text
    df = df[df["text"].str.len() > 0]

    # Remove rows with invalid intents
    df = df[df["intent"] != "INVALID"]

    # Remove duplicates
    df = df.drop_duplicates(subset=["text"])

    print(f"Cleaned dataset shape: {df.shape}")

    # Save cleaned dataset
    if output_path.endswith(".csv"):
        df.to_csv(output_path, index=False)
    elif output_path.endswith(".json"):
        df.to_json(output_path, orient="records", indent=2)

    print(f"Cleaned dataset saved to {output_path}")

    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Unique intents: {df['intent'].nunique()}")
    print("\nIntent distribution:")
    print(df["intent"].value_counts())


def main():
    parser = argparse.ArgumentParser(description="Clean intent classification dataset")
    parser.add_argument("--input", type=str, required=True, help="Input dataset path")
    parser.add_argument("--output", type=str, required=True, help="Output dataset path")
    parser.add_argument("--intents", type=str, nargs="+", help="Valid intent labels")

    args = parser.parse_args()

    clean_dataset(args.input, args.output, args.intents)


if __name__ == "__main__":
    main()
