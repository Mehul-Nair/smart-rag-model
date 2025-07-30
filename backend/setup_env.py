#!/usr/bin/env python3
"""
Environment Setup Script for Smart AI Agent

This script helps you set up your environment variables properly.
It will create a .env file and provide instructions for setting up your OpenAI API key.
"""

import os
import sys
from pathlib import Path


def create_env_file():
    """Create a .env file with proper configuration"""
    env_content = """# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Intent Classification Configuration
INTENT_IMPLEMENTATION=huggingface
INTENT_CONFIDENCE_THRESHOLD=0.7

# HuggingFace Model Settings
HF_MODEL_NAME=distilbert-base-uncased
HF_NUM_LABELS=4
HF_MAX_LENGTH=512
HF_DEVICE=cpu

# Performance Monitoring
ENABLE_PERFORMANCE_MONITORING=true
LOG_LEVEL=INFO

# Company Configuration
COMPANY_NAME=Asian Paints Beautiful Homes
COMPANY_BRAND=Asian Paints Beautiful Homes
DEFAULT_WARRANTY_PERIOD=1-year

# API Configuration
DEFAULT_MODEL=gpt-3.5-turbo
TEMPERATURE=0

# Performance Configuration
MAX_RETRIEVAL_DOCS=20
MAX_CONVERSATION_HISTORY=10
DEFAULT_RESPONSE_TIMEOUT=30
"""

    env_file = Path(".env")
    if env_file.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != "y":
            print("Keeping existing .env file")
            return

    with open(env_file, "w") as f:
        f.write(env_content)

    print("✅ Created .env file")
    print(
        "📝 Please edit the .env file and replace 'your_openai_api_key_here' with your actual OpenAI API key"
    )


def check_current_env():
    """Check current environment variable status"""
    print("🔍 Checking current environment setup...")

    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("❌ .env file not found")

    # Check system environment variable
    system_key = os.getenv("OPENAI_API_KEY")
    if system_key:
        print(f"⚠️  System OPENAI_API_KEY found: {system_key[:10]}...")
        print("   This might be your old key. Consider using .env file instead.")
    else:
        print("✅ No system OPENAI_API_KEY found (good for .env usage)")

    # Check if we can load from .env
    try:
        from dotenv import load_dotenv

        load_dotenv(".env", override=True)
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key and env_key != "your_openai_api_key_here":
            print("✅ Valid OpenAI API key found in .env file")
        elif env_key == "your_openai_api_key_here":
            print("❌ OpenAI API key in .env file needs to be updated")
        else:
            print("❌ No OpenAI API key found in .env file")
    except ImportError:
        print("❌ python-dotenv not installed")


def clear_system_env():
    """Clear system environment variable"""
    print("🧹 Clearing system OPENAI_API_KEY environment variable...")

    if os.name == "nt":  # Windows
        os.system("set OPENAI_API_KEY=")
        print("✅ Cleared system environment variable")
        print(
            "📝 Note: You may need to restart your terminal for changes to take effect"
        )
    else:  # Unix/Linux/Mac
        print("📝 To clear system environment variable on Unix/Linux/Mac:")
        print("   unset OPENAI_API_KEY")
        print("   Or restart your terminal")


def main():
    """Main setup function"""
    print("🚀 Smart AI Agent Environment Setup")
    print("=" * 50)

    while True:
        print("\nOptions:")
        print("1. Create/Update .env file")
        print("2. Check current environment status")
        print("3. Clear system environment variable")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ")

        if choice == "1":
            create_env_file()
        elif choice == "2":
            check_current_env()
        elif choice == "3":
            clear_system_env()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
