#!/usr/bin/env python3
"""
Fixed FAISS Index Builder

This script builds a FAISS index from Excel files with improved category handling
and better error handling for the smart AI agent.
"""

import os
import pandas as pd
from glob import glob
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv(".env", override=True)  # override=True ensures .env takes precedence

# Import config after loading environment variables
from config import get_openai_key, validate_openai_key

# Verify API key is loaded
if not validate_openai_key():
    exit(1)
else:
    print("✅ OpenAI API key loaded successfully")
    OPENAI_API_KEY = get_openai_key()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "data_source", "faiss_index")
os.makedirs(INDEX_DIR, exist_ok=True)

# Enhanced category normalization dictionary
# Now completely dynamic - no hardcoded mappings


def get_all_excel_files(data_dir):
    """Get all Excel files, excluding temporary lock files"""
    all_files = glob(os.path.join(data_dir, "**", "*.xlsx"), recursive=True)
    # Filter out temporary Excel lock files (files starting with ~$)
    valid_files = [f for f in all_files if not os.path.basename(f).startswith("~$")]
    return valid_files


def generate_product_type_mappings(df):
    """
    Generate product type mappings from sub_category to main_category.
    """
    print("🔍 Generating sub_category → main_category mappings...")
    mappings = {}
    if "sub_category" in df.columns and "main_category" in df.columns:
        pairs = df[["sub_category", "main_category"]].dropna().drop_duplicates()
        for _, row in pairs.iterrows():
            sub = str(row["sub_category"]).strip().lower()
            main = str(row["main_category"]).strip().lower()
            mappings[sub] = main
            print(f"  📝 {sub} → {main}")
    print(f"✅ Generated {len(mappings)} sub_category → main_category mappings")
    return mappings


def build_faiss_index():
    """Build FAISS index with improved category handling"""
    print("🔧 Building FAISS index with fixed category handling...")

    all_docs = []
    category_summary = {}

    # Load data from Excel file
    excel_file = os.path.join(DATA_DIR, "BH_PD.xlsx")
    print(f"📂 Loading data from: {excel_file}")

    try:
        df = pd.read_excel(excel_file)
        print(f"✅ Loaded {len(df)} rows from Excel file")
        # print(f"📋 Available columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return

    # Generate dynamic product type mappings
    product_type_mappings = generate_product_type_mappings(df)

    # Save product type mappings to JSON file
    mappings_file = os.path.join(INDEX_DIR, "product_type_mappings.json")
    with open(mappings_file, "w") as f:
        json.dump(product_type_mappings, f, indent=2)
    print(f"💾 Product type mappings saved to: {mappings_file}")

    # Analyze categories in the data
    if "main_category" in df.columns:
        unique_categories = df["main_category"].unique()
        print(f"🔍 Found categories: {unique_categories}")

        # Process each category
        for category in unique_categories:
            if pd.isna(category):
                continue

            # Normalize category name (use original from dataset)
            normalized_category = str(category).lower()

            # Filter data for this category
            category_df = df[df["main_category"] == category].copy()

            print(
                f"📊 Processing category '{category}' -> '{normalized_category}': {len(category_df)} products"
            )

            # Clean and convert mrp column for price range
            if "mrp" in category_df.columns:
                category_df["mrp_clean"] = pd.to_numeric(
                    category_df["mrp"].astype(str).str.replace(",", ""), errors="coerce"
                )
                min_price = (
                    float(category_df["mrp_clean"].min())
                    if not category_df["mrp_clean"].isnull().all()
                    else 0
                )
                max_price = (
                    float(category_df["mrp_clean"].max())
                    if not category_df["mrp_clean"].isnull().all()
                    else 0
                )
            else:
                min_price = 0
                max_price = 0

            # Store category summary
            category_summary[normalized_category] = {
                "file": os.path.basename(excel_file),
                "count": len(category_df),
                "price_range": {
                    "min": min_price,
                    "max": max_price,
                },
            }

            # Create documents for this category
            for i, row in category_df.iterrows():
                # Create document content from relevant fields based on NER mapping
                content_parts = []

                # Basic product information
                if "title" in row and pd.notna(row["title"]):
                    content_parts.append(f"Title: {row['title']}")
                if "pdp_header" in row and pd.notna(row["pdp_header"]):
                    content_parts.append(f"Description: {row['pdp_header']}")
                if "detailed_product_description" in row and pd.notna(
                    row["detailed_product_description"]
                ):
                    content_parts.append(
                        f"Detailed Description: {row['detailed_product_description']}"
                    )
                
                # Product image
                if "featuredImg" in row and pd.notna(row["featuredImg"]):
                    content_parts.append(f"Product Image: {row['featuredImg']}")

                # Category information
                if "main_category" in row and pd.notna(row["main_category"]):
                    content_parts.append(f"Main Category: {row['main_category']}")
                if "sub_category" in row and pd.notna(row["sub_category"]):
                    content_parts.append(f"Sub Category: {row['sub_category']}")

                # Product specifications
                if "primary_material" in row and pd.notna(row["primary_material"]):
                    content_parts.append(f"Primary Material: {row['primary_material']}")
                if "secondary_material" in row and pd.notna(row["secondary_material"]):
                    content_parts.append(
                        f"Secondary Material: {row['secondary_material']}"
                    )
                if "colour" in row and pd.notna(row["colour"]):
                    content_parts.append(f"Color: {row['colour']}")
                if "size" in row and pd.notna(row["size"]):
                    content_parts.append(f"Size: {row['size']}")

                # Brand and style information
                if "brand_name" in row and pd.notna(row["brand_name"]):
                    content_parts.append(f"Brand: {row['brand_name']}")
                if "style" in row and pd.notna(row["style"]):
                    content_parts.append(f"Style: {row['style']}")

                # Price and budget information
                if "mrp" in row and pd.notna(row["mrp"]):
                    content_parts.append(f"Price: {row['mrp']}")
                if "discounted_price" in row and pd.notna(row["discounted_price"]):
                    content_parts.append(f"Discounted Price: {row['discounted_price']}")
                
                # Calculate and add discount percentage to content
                if (pd.notna(row.get("mrp")) and pd.notna(row.get("discounted_price")) and 
                    float(str(row.get("mrp", 0) or 0).replace(",", "")) > 0 and 
                    float(str(row.get("discounted_price", 0) or 0).replace(",", "")) > 0):
                    discount_pct = round(
                        ((float(str(row.get("mrp", 0) or 0).replace(",", "")) - float(str(row.get("discounted_price", 0) or 0).replace(",", ""))) / float(str(row.get("mrp", 1) or 1).replace(",", ""))) * 100
                    )
                    content_parts.append(f"Discount Percentage: {discount_pct}%")
                
                # Add URL to content
                if "url" in row and pd.notna(row["url"]):
                    content_parts.append(f"Product URL: {row['url']}")

                # Warranty and care information
                if "warranty" in row and pd.notna(row["warranty"]):
                    content_parts.append(f"Warranty: {row['warranty']}")
                if "careInstructions" in row and pd.notna(row["careInstructions"]):
                    content_parts.append(
                        f"Care Instructions: {row['careInstructions']}"
                    )
                if "instructions_for_care" in row and pd.notna(
                    row["instructions_for_care"]
                ):
                    content_parts.append(
                        f"Care Instructions: {row['instructions_for_care']}"
                    )

                # Additional specifications
                if "pattern" in row and pd.notna(row["pattern"]):
                    content_parts.append(f"Pattern: {row['pattern']}")
                if "collection" in row and pd.notna(row["collection"]):
                    content_parts.append(f"Collection: {row['collection']}")
                if "end_use" in row and pd.notna(row["end_use"]):
                    content_parts.append(f"End Use: {row['end_use']}")

                doc_content = " | ".join(content_parts)

                doc = Document(
                    page_content=doc_content,
                    metadata={
                        "row": i,
                        "file": excel_file,
                        "category": normalized_category,
                        "main_category": str(row.get("main_category", "")),
                        "sub_category": str(row.get("sub_category", "")),
                        "filename": os.path.basename(excel_file),
                        "title": str(row.get("title", "")),
                        "price": str(row.get("mrp", "")),
                        "url": str(row.get("url", "")),
                        "product_image": str(row.get("featuredImg", "")),
                        "warranty": str(row.get("warranty", "")),
                        "brand_name": str(row.get("brand_name", "")),
                        "primary_material": str(row.get("primary_material", "")),
                        "secondary_material": str(row.get("secondary_material", "")),
                        "colour": str(row.get("colour", "")),
                        "size": str(row.get("size", "")),
                        "style": str(row.get("style", "")),
                        "pattern": str(row.get("pattern", "")),
                        "collection": str(row.get("collection", "")),
                        "end_use": str(row.get("end_use", "")),
                        "careInstructions": str(row.get("careInstructions", "")),
                        "instructions_for_care": str(
                            row.get("instructions_for_care", "")
                        ),
                        "discounted_price": str(row.get("discounted_price", "")),
                        "discount_percentage": str(
                            round(
                                ((float(str(row.get("mrp", 0) or 0).replace(",", "")) - float(str(row.get("discounted_price", 0) or 0).replace(",", ""))) / float(str(row.get("mrp", 1) or 1).replace(",", ""))) * 100
                            ) if (pd.notna(row.get("mrp")) and pd.notna(row.get("discounted_price")) and 
                                  float(str(row.get("mrp", 0) or 0).replace(",", "")) > 0 and float(str(row.get("discounted_price", 0) or 0).replace(",", "")) > 0) else "0"
                        ),
                        "detailed_product_description": str(
                            row.get("detailed_product_description", "")
                        ),
                    },
                )
                all_docs.append(doc)
    else:
        print(f"❌ No main_category column found in {excel_file}")
        return

    print(f"📚 Total documents created: {len(all_docs)}")

    # Show final unique categories
    unique_categories = set()
    for doc in all_docs:
        if "category" in doc.metadata:
            unique_categories.add(doc.metadata["category"])
    print(f"🎯 Final unique categories: {sorted(list(unique_categories))}")

    # Save category summary for quick access
    category_summary_path = os.path.join(INDEX_DIR, "category_summary.json")
    with open(category_summary_path, "w") as f:
        json.dump(category_summary, f, indent=2)
    print(f"💾 Category summary saved to: {category_summary_path}")

    # Create smaller chunks to preserve more context
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)
    print(f"✂️ Split into {len(split_docs)} chunks for embedding.")

    if OPENAI_API_KEY is None:
        raise ValueError(
            "Missing OpenAI API Key. Please set OPENAI_API_KEY in your environment."
        )

    print("🔍 Creating embeddings...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"✅ FAISS index saved to {INDEX_DIR}")

    # Print final summary
    print("\n🎉 Indexing completed successfully!")
    print(f"📊 Categories indexed: {len(category_summary)}")
    for category, info in category_summary.items():
        price_range = info["price_range"]
        print(
            f"  - {category}: {info['count']} products (₹{price_range['min']} - ₹{price_range['max']})"
        )


if __name__ == "__main__":
    build_faiss_index()
