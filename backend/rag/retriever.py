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

    def retrieve(self, query: str, k: int = 3):
        return self.retriever.get_relevant_documents(query, k=k)
