# Smart AI Agent

A modular AI agent with intent classification and RAG (Retrieval-Augmented Generation) capabilities for product recommendations and customer support.

## 🚀 Features

- **Modular Intent Classification**: Support for HuggingFace, OpenAI, and rule-based classifiers
- **RAG System**: Retrieval-augmented generation for accurate product recommendations
- **LangGraph Workflow**: Structured conversation flow with proper routing
- **FastAPI Backend**: RESTful API with real-time chat capabilities
- **React Frontend**: Modern, responsive UI with dark mode support
- **Trained Models**: Fine-tuned HuggingFace model for intent classification

## 📁 Project Structure

```
smart-ai-agent/
├── backend/              # FastAPI backend with RAG and intent classification
│   ├── main.py          # FastAPI application
│   ├── rag/             # RAG implementation with LangGraph
│   ├── data/            # Product data and training files
│   └── trained_intent_model/  # Fine-tuned HuggingFace model
├── frontend/            # React TypeScript frontend
│   ├── src/             # React components and hooks
│   └── public/          # Static assets
└── docs/                # Documentation
```

## 🛠️ Quick Start

### Backend Setup

1. **Navigate to backend directory:**

   ```bash
   cd backend
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**

   ```bash
   # Create .env file with:
   OPENAI_API_KEY=your_openai_api_key
   ```

4. **Build FAISS index:**

   ```bash
   python build_index.py
   ```

5. **Start the backend server:**
   ```bash
   python main.py
   ```

### Frontend Setup

1. **Navigate to frontend directory:**

   ```bash
   cd frontend
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```

## 🔧 Configuration

### Intent Classification

The system supports multiple intent classification implementations:

- **HuggingFace** (Recommended): Fine-tuned DistilBERT model with 65% accuracy
- **Rule-based**: Pattern matching and semantic similarity
- **OpenAI**: Fine-tuned OpenAI model
- **Hybrid**: Combines multiple classifiers with fallback

Set the implementation via environment variable:

```bash
INTENT_IMPLEMENTATION=huggingface  # or rule_based, openai, hybrid
```

### Model Training

To train the HuggingFace model on custom data:

1. Place training data in `backend/data/training/training_data.jsonl`
2. Run: `python backend/train_huggingface.py`
3. Trained model is saved to `backend/trained_intent_model/`

## 📊 Performance

- **Intent Classification Accuracy**: 65% (HuggingFace model)
- **Processing Speed**: ~26ms per query
- **Supported Intents**: META, PRODUCT, INVALID, CLARIFY

## 🔄 API Endpoints

- `POST /chat`: Main chat endpoint for AI interactions
- `GET /`: Health check endpoint

## 📚 Documentation

- [Complete Setup Guide](docs/SETUP.md) - **Start here for fresh installation**
- [Backend Setup Guide](docs/BACKEND_SETUP.md)
- [Robust Agent Improvements](docs/ROBUST_AGENT_IMPROVEMENTS.md)
- [Backend README](backend/README.md)

## 🧹 Maintenance

- Training data and stats: `backend/data/training/`
- FAISS index: `backend/data_source/faiss_index/`
- Product data: `backend/data/*.xlsx`
- Trained models: `backend/trained_intent_model/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
