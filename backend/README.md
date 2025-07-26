# Backend - Smart AI Agent

This directory contains the backend implementation of the Smart AI Agent with modular intent classification and RAG capabilities.

## 📁 Directory Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── build_index.py         # Data indexing script for FAISS
├── train_huggingface.py   # HuggingFace model training script
├── data/                  # Product data and training files
│   ├── *.xlsx            # Product data files
│   └── training/         # Training data and stats
├── data_source/          # FAISS index storage
├── rag/                  # RAG implementation
│   ├── langgraph_agent.py    # Main LangGraph agent
│   ├── retriever.py          # Data retrieval
│   ├── config.py             # Core configuration
│   └── intent_modules/       # Modular intent classification
│       ├── base.py           # Abstract base class
│       ├── factory.py        # Factory pattern implementation
│       ├── config.py         # Intent classifier configuration
│       ├── huggingface_classifier.py  # HuggingFace implementation
│       ├── rule_based_classifier.py   # Rule-based implementation
│       └── openai_classifier.py       # OpenAI implementation
└── venv/                 # Virtual environment (not in git)
```

## 🚀 Quick Start

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt

   # For training (if not already included):
   pip install -r requirements-training.txt
   ```

2. **Set up environment variables:**

   ```bash
   # Create .env file with:
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Build FAISS index:**

   ```bash
   python build_index.py
   ```

4. **Train HuggingFace model (optional):**

   ```bash
   python train_huggingface.py
   ```

5. **Run the application:**
   ```bash
   python main.py
   ```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key for LLM access
- `FINE_TUNED_MODEL_NAME`: Fine-tuned model name (optional)
- `INTENT_IMPLEMENTATION`: Intent classifier to use (`huggingface`, `rule_based`, `openai`, `hybrid`)
- `INTENT_CONFIDENCE_THRESHOLD`: Minimum confidence threshold (default: 0.7)

### Intent Classification

The system supports multiple intent classification implementations:

- **HuggingFace**: Fine-tuned DistilBERT model (recommended)
- **Rule-based**: Pattern matching and semantic similarity
- **OpenAI**: Fine-tuned OpenAI model
- **Hybrid**: Combines multiple classifiers with fallback

## 📊 Training

The HuggingFace model can be trained on custom data:

1. Place training data in `data/training/training_data.jsonl`
2. Run `python train_huggingface.py`
3. Trained model is saved to `./trained_intent_model/`

## 🔄 API Endpoints

- `POST /chat`: Main chat endpoint for user interactions

## 🧹 Maintenance

- Training data and stats are stored in `data/training/`
- FAISS index is stored in `data_source/faiss_index/`
- Product data is in `data/*.xlsx` files

## start server

cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
