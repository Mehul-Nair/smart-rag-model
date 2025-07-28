#!/usr/bin/env python3
"""
Check product titles from BH_PD.xlsx
"""

import pandas as pd
import os


def check_product_titles():
    """Check product titles from the Excel file"""
    # Get the full path to the Excel file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(current_dir, "data", "BH_PD.xlsx")

    try:
        # Read the Excel file
        df = pd.read_excel(excel_path)

        print(f"📊 Excel file loaded successfully")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📈 Total rows: {len(df)}")

        # Look for title-related columns
        title_columns = [
            col for col in df.columns if "title" in col.lower() or "name" in col.lower()
        ]
        print(f"🎯 Title-related columns: {title_columns}")

        if title_columns:
            title_col = title_columns[0]
            print(f"\n📝 Sample product titles from '{title_col}':")
            print("-" * 80)

            # Show first 20 product titles
            for i, title in enumerate(df[title_col].head(20), 1):
                if pd.notna(title):  # Check if not NaN
                    print(f"{i:2d}. {title}")

            # Show some examples that might be similar to "City Lights Ceiling Light"
            print(f"\n🔍 Looking for lighting-related products:")
            print("-" * 80)

            lighting_keywords = [
                "light",
                "lamp",
                "chandelier",
                "ceiling",
                "wall",
                "led",
            ]
            lighting_products = []

            for title in df[title_col]:
                if pd.notna(title) and any(
                    keyword in str(title).lower() for keyword in lighting_keywords
                ):
                    lighting_products.append(title)
                    if len(lighting_products) >= 10:  # Limit to 10 examples
                        break

            for i, title in enumerate(lighting_products, 1):
                print(f"{i:2d}. {title}")

        # Check for any other relevant columns
        print(f"\n📋 All columns in the dataset:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")

    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")


if __name__ == "__main__":
    check_product_titles()
