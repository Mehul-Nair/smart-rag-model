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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Verify API key is loaded
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not found in .env file")
    print("Please create a .env file in the backend directory with:")
    print("OPENAI_API_KEY=your_actual_api_key_here")
    exit(1)
else:
    print("✅ OpenAI API key loaded successfully")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "data_source", "faiss_index")
os.makedirs(INDEX_DIR, exist_ok=True)

# Enhanced category normalization dictionary
CATEGORY_MAP = {
    "bedside-table": "bedside tables",
    "handmade_rugs": "handmade rugs",
    "handmade-rugs": "handmade rugs",
    "fabrics": "fabrics",
    "Furnishing": "furnishing",
    "Lights": "lights",
    "Bath": "bath",
    "Rugs": "rugs",
    "Furniture": "furniture",
}


def get_all_excel_files(data_dir):
    """Get all Excel files, excluding temporary lock files"""
    all_files = glob(os.path.join(data_dir, "**", "*.xlsx"), recursive=True)
    # Filter out temporary Excel lock files (files starting with ~$)
    valid_files = [f for f in all_files if not os.path.basename(f).startswith("~$")]
    return valid_files


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
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return

    # Analyze categories in the data
    if "main_category" in df.columns:
        unique_categories = df["main_category"].unique()
        print(f"🔍 Found categories: {unique_categories}")

        # Process each category
        for category in unique_categories:
            if pd.isna(category):
                continue

            # Normalize category name
            normalized_category = CATEGORY_MAP.get(str(category), str(category).lower())

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
                # Create document content from relevant fields
                content_parts = []
                if "title" in row and pd.notna(row["title"]):
                    content_parts.append(f"Title: {row['title']}")
                if "pdp_header" in row and pd.notna(row["pdp_header"]):
                    content_parts.append(f"Description: {row['pdp_header']}")
                if "mrp" in row and pd.notna(row["mrp"]):
                    content_parts.append(f"Price: {row['mrp']}")
                if "main_category" in row and pd.notna(row["main_category"]):
                    content_parts.append(f"Main Category: {row['main_category']}")
                if "sub_category" in row and pd.notna(row["sub_category"]):
                    content_parts.append(f"Sub Category: {row['sub_category']}")

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
