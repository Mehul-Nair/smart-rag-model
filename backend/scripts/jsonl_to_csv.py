#!/usr/bin/env python3
"""
JSONL to CSV Converter

This script converts JSONL files to CSV format for easier processing.
"""

import json
import pandas as pd
import argparse
from typing import List, Dict


def jsonl_to_csv(input_path: str, output_path: str):
    """Convert JSONL file to CSV"""
    print(f"Converting {input_path} to {output_path}")

    # Read JSONL file
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save as CSV
    df.to_csv(output_path, index=False)

    print(f"Converted {len(data)} records to CSV")
    print(f"Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to CSV")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")

    args = parser.parse_args()

    jsonl_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()
