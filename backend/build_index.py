import os
import pandas as pd
from glob import glob

# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
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
    "bedside-table": "bedside tables",
    "handmade_rugs": "handmade rugs",
    "handmade-rugs": "handmade rugs",
    "fabrics": "fabrics",
}


def get_all_excel_files(data_dir):
    return glob(os.path.join(data_dir, "**", "*.xlsx"), recursive=True)


def build_faiss_index():
    all_docs = []
    category_summary = {}

    for file_path in get_all_excel_files(DATA_DIR):
        filename = os.path.basename(file_path)
        print(f"Processing file: {filename}")

        df = pd.read_excel(file_path)

        # Check if main_category column exists
        if "main_category" not in df.columns:
            print(f"Warning: No main_category column found in {filename}")
            print("Available columns:", df.columns)
            continue

        # Extract main_category and sub_category from the actual data
        main_category = str(df.iloc[0].get("main_category", "")).strip()
        sub_category = (
            str(df.iloc[0].get("sub_category", "")).strip()
            if "sub_category" in df.columns
            else ""
        )
        # You can combine them if you want more granularity
        category = main_category  # or f"{main_category} > {sub_category}" for both

        # Normalize category using dictionary mapping
        category = CATEGORY_MAP.get(category, category)

        # Clean and convert mrp column for price range
        if "mrp" in df.columns:
            df["mrp_clean"] = pd.to_numeric(
                df["mrp"].astype(str).str.replace(",", ""), errors="coerce"
            )
            min_price = (
                float(df["mrp_clean"].min())
                if not df["mrp_clean"].isnull().all()
                else 0
            )
            max_price = (
                float(df["mrp_clean"].max())
                if not df["mrp_clean"].isnull().all()
                else 0
            )
        else:
            min_price = 0
            max_price = 0

        # Store category summary
        category_summary[category] = {
            "file": filename,
            "count": len(df),
            "price_range": {
                "min": min_price,
                "max": max_price,
            },
        }

        print(f"  - Added {len(df)} documents with category: {category}")

        for i, row in df.iterrows():
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
                    "file": file_path,
                    "category": category,
                    "main_category": str(row.get("main_category", "")),
                    "sub_category": str(row.get("sub_category", "")),
                    "filename": filename,
                    "title": str(row.get("title", "")),
                    "price": str(row.get("mrp", "")),
                    "url": str(row.get("url", "")),
                },
            )
            all_docs.append(doc)

    print(f"Total documents loaded: {len(all_docs)}")

    # Show unique categories found
    unique_categories = set()
    for doc in all_docs:
        if "category" in doc.metadata:
            unique_categories.add(doc.metadata["category"])
    print(f"Unique categories found: {sorted(unique_categories)}")

    # Save category summary for quick access
    category_summary_path = os.path.join(INDEX_DIR, "category_summary.json")
    with open(category_summary_path, "w") as f:
        json.dump(category_summary, f, indent=2)
    print(f"Category summary saved to: {category_summary_path}")

    # Create smaller chunks to preserve more context
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)
    print(f"Split into {len(split_docs)} chunks for embedding.")

    if OPENAI_API_KEY is None:
        raise ValueError(
            "Missing OpenAI API Key. Please set OPENAI_API_KEY in your environment."
        )

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"FAISS index saved to {INDEX_DIR}")


if __name__ == "__main__":
    build_faiss_index()
