#!/usr/bin/env python3
"""
Test Integration Script

This script tests the complete integration of:
1. Intent Model (Improved Hybrid)
2. ONNX NER Model
3. LangGraph Agent
4. FAISS Index
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

from rag.retriever import ExcelRetriever
from rag.langgraph_agent import build_langgraph_agent
from rag.intent_modules.onnx_ner_classifier import extract_slots_from_text


def test_integration():
    """Test the complete integration"""
    print("🧪 Testing Complete Integration...")

    # Check environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        print("Please create a .env file in the backend directory with:")
        print("OPENAI_API_KEY=your_actual_api_key_here")
        return False

    print("✅ Environment variables loaded successfully")

    # Test NER extraction
    print("\n🔍 Testing ONNX NER Extraction...")
    test_texts = [
        "I want a blue sofa under 50000 rupees",
        "Show me wooden dining table from IKEA",
        "I need bathroom accessories for my master bedroom",
    ]

    for text in test_texts:
        slots = extract_slots_from_text(text)
        print(f"📝 '{text}' -> {slots}")

    # Test retriever
    print("\n📚 Testing FAISS Retriever...")
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        retriever = ExcelRetriever(data_dir, openai_api_key)
        print("✅ Retriever initialized successfully")

        # Test retrieval
        test_query = "blue sofa"
        docs = retriever.retrieve(test_query, k=3)
        print(f"✅ Retrieved {len(docs)} documents for '{test_query}'")

    except Exception as e:
        print(f"❌ Retriever error: {e}")
        return False

    # Test LangGraph agent
    print("\n🤖 Testing LangGraph Agent...")
    try:
        langgraph_agent = build_langgraph_agent(retriever, openai_api_key)
        print("✅ LangGraph agent built successfully")

        # Test agent with a simple query
        test_query = "Hello, can you help me find furniture?"
        print(f"📝 Testing agent with: '{test_query}'")

        response = langgraph_agent(test_query)
        print(f"✅ Agent response: {response}")

    except Exception as e:
        print(f"❌ LangGraph agent error: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n🎉 All integration tests passed!")
    return True


if __name__ == "__main__":
    success = test_integration()
    if success:
        print("\n✅ Integration is working correctly!")
    else:
        print("\n❌ Integration has issues that need to be fixed.")
