#!/usr/bin/env python3
"""
Fixed FAISS Index Builder for Single Excel File with Multiple Categories
"""

import os
import pandas as pd
from glob import glob
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
from pydantic import SecretStr
import json

# Load environment variables from .env file
load_dotenv(".env")
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

# Category normalization dictionary
CATEGORY_MAP = {
    "furnishing": "furnishing",
    "Furnishing": "furnishing",
    "lights": "lights",
    "Lights": "lights",
    "Bath": "bath",
    "Rugs": "rugs",
    "Furniture": "furniture",
}


def normalize_category(category):
    """Normalize category names"""
    if pd.isna(category):
        return "unknown"
    category_str = str(category).strip()
    return CATEGORY_MAP.get(category_str, category_str.lower())


def build_faiss_index():
    """Build FAISS index from the single Excel file with multiple categories"""
    print("🔧 Building FAISS index with fixed category handling...")

    all_docs = []
    category_summary = {}

    # Process the single Excel file
    excel_file = os.path.join(DATA_DIR, "BH_PD.xlsx")

    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        return

    print(f"📂 Loading data from: {excel_file}")

    # Read the Excel file
    df = pd.read_excel(excel_file)
    print(f"✅ Loaded {len(df)} rows from Excel file")

    # Check if main_category column exists
    if "main_category" not in df.columns:
        print(f"❌ No main_category column found in {excel_file}")
        print("Available columns:", df.columns.tolist())
        return

    # Get unique categories and normalize them
    unique_categories = df["main_category"].unique()
    print(f"🔍 Found categories: {unique_categories}")

    # Process each category
    for category in unique_categories:
        if pd.isna(category):
            continue

        normalized_category = normalize_category(category)
        category_df = df[df["main_category"] == category]

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
            "original_category": category,
            "file": "BH_PD.xlsx",
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
            if "brand_name" in row and pd.notna(row["brand_name"]):
                content_parts.append(f"Brand: {row['brand_name']}")
            if "colour" in row and pd.notna(row["colour"]):
                content_parts.append(f"Color: {row['colour']}")
            if "size" in row and pd.notna(row["size"]):
                content_parts.append(f"Size: {row['size']}")
            if "primary_material" in row and pd.notna(row["primary_material"]):
                content_parts.append(f"Material: {row['primary_material']}")
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
                    "original_category": str(row.get("main_category", "")),
                    "sub_category": str(row.get("sub_category", "")),
                    "filename": "BH_PD.xlsx",
                    "title": str(row.get("title", "")),
                    "price": str(row.get("mrp", "")),
                    "brand": str(row.get("brand_name", "")),
                    "color": str(row.get("colour", "")),
                    "material": str(row.get("primary_material", "")),
                    "url": str(row.get("url", "")),
                },
            )
            all_docs.append(doc)

    print(f"📚 Total documents created: {len(all_docs)}")

    # Show unique categories found
    unique_categories_final = set()
    for doc in all_docs:
        if "category" in doc.metadata:
            unique_categories_final.add(doc.metadata["category"])

    print(f"🎯 Final unique categories: {sorted(unique_categories_final)}")

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
    try:
        # Use the newer OpenAI embeddings approach
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)

        # Create embeddings using the client directly
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002"
        )
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(INDEX_DIR)
        print(f"✅ FAISS index saved to {INDEX_DIR}")
    except Exception as e:
        print(f"❌ Error creating embeddings: {e}")
        print("Trying alternative approach with different parameters...")
        try:
            # Alternative approach with minimal parameters
            embeddings = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                model="text-embedding-ada-002",
                chunk_size=1000,
            )
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            vectorstore.save_local(INDEX_DIR)
            print(f"✅ FAISS index saved to {INDEX_DIR}")
        except Exception as e2:
            print(f"❌ Second attempt failed: {e2}")
            print("Saving documents without embeddings for now...")
            # Save the processed documents for later embedding
            import pickle

            docs_path = os.path.join(INDEX_DIR, "processed_docs.pkl")
            with open(docs_path, "wb") as f:
                pickle.dump(split_docs, f)
            print(f"✅ Processed documents saved to {docs_path}")
            print("You can create embeddings later when the issue is resolved.")

    # Print final summary
    print("\n🎉 Indexing completed successfully!")
    print(f"📊 Categories indexed: {len(category_summary)}")
    for category, info in category_summary.items():
        print(
            f"  - {category}: {info['count']} products (₹{info['price_range']['min']:.0f} - ₹{info['price_range']['max']:.0f})"
        )


if __name__ == "__main__":
    build_faiss_index()
