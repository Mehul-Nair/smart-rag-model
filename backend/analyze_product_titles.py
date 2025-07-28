#!/usr/bin/env python3
"""
Analyze product titles to understand patterns
"""

import pandas as pd
import os


def analyze_product_titles():
    """Analyze product titles for patterns"""
    # Get the full path to the Excel file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(current_dir, "data", "BH_PD.xlsx")

    try:
        # Read the Excel file
        df = pd.read_excel(excel_path)

        print(f"📊 Analyzing product titles...")

        # Look for titles that might match our test cases
        test_patterns = [
            "city lights",
            "crystal chandelier",
            "lighting up lisbon",
            "lights out gold",
            "ceiling light",
            "wall light",
            "chandelier",
        ]

        print(f"\n🔍 Searching for titles matching our test patterns:")
        print("-" * 80)

        for pattern in test_patterns:
            matching_titles = []
            for title in df["title"]:
                if pd.notna(title) and pattern.lower() in str(title).lower():
                    matching_titles.append(title)
                    if len(matching_titles) >= 5:  # Limit to 5 examples per pattern
                        break

            if matching_titles:
                print(f"\n📝 Titles containing '{pattern}':")
                for i, title in enumerate(matching_titles, 1):
                    print(f"  {i}. {title}")
            else:
                print(f"\n❌ No titles found containing '{pattern}'")

        # Look for titles with similar structure to "City Lights Ceiling Light"
        print(f"\n🎯 Titles with similar structure to 'City Lights Ceiling Light':")
        print("-" * 80)

        # Look for titles that have multiple words that could be misclassified
        complex_titles = []
        for title in df["title"]:
            if pd.notna(title):
                title_str = str(title)
                # Look for titles with multiple words that might be split by NER
                words = title_str.split()
                if len(words) >= 3 and any(
                    word.lower() in ["light", "lights", "ceiling", "wall", "chandelier"]
                    for word in words
                ):
                    complex_titles.append(title_str)
                    if len(complex_titles) >= 15:  # Limit to 15 examples
                        break

        for i, title in enumerate(complex_titles, 1):
            print(f"{i:2d}. {title}")

        # Check brand names
        print(f"\n🏷️ Sample brand names:")
        print("-" * 80)
        brands = df["brand_name"].dropna().unique()
        for i, brand in enumerate(brands[:10], 1):
            print(f"{i:2d}. {brand}")

    except Exception as e:
        print(f"❌ Error analyzing Excel file: {e}")


if __name__ == "__main__":
    analyze_product_titles()
