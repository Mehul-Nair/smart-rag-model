import pandas as pd

# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import os
from glob import glob
from pydantic import SecretStr
import json


class ExcelRetriever:
    def __init__(self, data_dir: str, openai_api_key: str):
        self.data_dir = data_dir
        self.openai_api_key = openai_api_key
        self.index_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data_source", "faiss_index"
        )
        self.index_path = os.path.join(self.index_dir, "index.faiss")
        self.store_path = os.path.join(self.index_dir, "index.pkl")
        self.category_summary_path = os.path.join(
            self.index_dir, "category_summary.json"
        )

        print("Looking for FAISS index at:", self.index_path)
        self.retriever = self._load_retriever()
        self.categories = self._load_categories()
        self.product_type_mappings = self._load_product_type_mappings()

    def _load_retriever(self):
        embeddings = OpenAIEmbeddings(api_key=SecretStr(self.openai_api_key))
        if os.path.exists(self.index_path) and os.path.exists(self.store_path):
            print("Loading FAISS index from disk...")
            vectorstore = FAISS.load_local(
                self.index_dir, embeddings, allow_dangerous_deserialization=True
            )
            return vectorstore.as_retriever()
        else:
            raise FileNotFoundError(
                f"FAISS index not found in {self.index_dir}. Please run 'python backend/build_index.py' to build the index before starting the server."
            )

    def _load_categories(self):
        """Load category information directly from the saved summary."""
        if os.path.exists(self.category_summary_path):
            try:
                with open(self.category_summary_path, "r") as f:
                    category_summary = json.load(f)
                categories = list(category_summary.keys())
                print(f"Loaded categories: {categories}")
                return categories
            except Exception as e:
                print(f"Error loading category summary: {e}")
                return []
        else:
            print("Category summary not found, will extract from documents")
            return []

    def get_categories(self):
        """Get all available categories directly."""
        if self.categories:
            return self.categories

        # Fallback: extract from documents if summary not available
        try:
            sample_docs = self.retriever.get_relevant_documents("product", k=500)
            categories = set()
            for doc in sample_docs:
                if hasattr(doc, "metadata") and doc.metadata:
                    if "category" in doc.metadata and doc.metadata["category"]:
                        category = str(doc.metadata["category"]).strip()
                        if category:
                            categories.add(category)
            return sorted(list(categories))
        except Exception as e:
            print(f"Error extracting categories: {e}")
            return []

    def _load_product_type_mappings(self):
        """Load dynamic product type mappings from JSON file."""
        mappings_file = os.path.join(self.index_dir, "product_type_mappings.json")
        if os.path.exists(mappings_file):
            try:
                with open(mappings_file, "r") as f:
                    mappings = json.load(f)
                print(f"Loaded {len(mappings)} dynamic product type mappings")
                return mappings
            except Exception as e:
                print(f"Error loading product type mappings: {e}")
                return {}
        else:
            print("Product type mappings not found, using empty mappings")
            return {}

    def get_product_type_mappings(self):
        """Get the loaded product type mappings."""
        return self.product_type_mappings

    def get_product_data(self, category: str, limit: int = 10):
        """
        Get sample product data for a specific category for slot relevance analysis.

        Args:
            category: The category to get products for
            limit: Maximum number of products to return

        Returns:
            List of product documents/metadata
        """
        try:
            # Search for products in the specified category
            query = f"category: {category}"
            docs = self.retriever.get_relevant_documents(query, k=limit)

            # Extract product information from documents
            products = []
            for doc in docs:
                if hasattr(doc, "metadata") and doc.metadata:
                    product_info = {
                        "title": doc.metadata.get("title", ""),
                        "description": doc.page_content,
                        "category": doc.metadata.get("category", ""),
                        "sub_category": doc.metadata.get("sub_category", ""),
                        "brand": doc.metadata.get("brand_name", ""),
                        "material": doc.metadata.get("primary_material", ""),
                        "color": doc.metadata.get("colour", ""),
                        "size": doc.metadata.get("size", ""),
                        "style": doc.metadata.get("style", ""),
                        "room_type": doc.metadata.get("end_use", ""),
                        "price": doc.metadata.get("price", ""),
                        "full_text": f"{doc.page_content} {doc.metadata.get('title', '')} {doc.metadata.get('brand_name', '')} {doc.metadata.get('primary_material', '')}",
                    }
                    products.append(product_info)

            return products

        except Exception as e:
            print(f"Error getting product data for category {category}: {e}")
            return []

    def retrieve(self, query: str, k: int = 3):
        return self.retriever.get_relevant_documents(query, k=k)
