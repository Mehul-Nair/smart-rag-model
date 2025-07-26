# Complete Backend Setup Guide

This guide will walk you through setting up the Smart AI Agent backend from scratch on a fresh system, including data indexing, model training, and all necessary configurations.

## 🎯 Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended for model training)
- **Storage**: At least 2GB free space
- **OS**: Windows, macOS, or Linux

### Required Accounts

- **OpenAI API Key**: For LLM functionality
- **Git**: For cloning the repository

## 📋 Step-by-Step Setup

### Step 1: Clone and Navigate to Project

```bash
# Clone the repository
git clone <your-repository-url>
cd smart-ai-agent

# Navigate to backend directory
cd backend
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Verify activation (should see (venv) in prompt)
```

### Step 3: Install Dependencies

```bash
# Install all required packages (includes training dependencies)
pip install -r requirements.txt

# Alternative: Install training dependencies separately
pip install -r requirements-training.txt
```

**Expected output**: All packages should install successfully without errors.

### Step 4: Set Up Environment Variables

Create a `.env` file in the backend directory:

```bash
# Create .env file
touch .env  # On Windows: echo. > .env
```

Add the following content to `.env`:

```env
# Required: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Intent Classification Configuration
INTENT_IMPLEMENTATION=huggingface
INTENT_CONFIDENCE_THRESHOLD=0.7

# Optional: HuggingFace Model Settings
HF_MODEL_NAME=distilbert-base-uncased
HF_NUM_LABELS=4
HF_MAX_LENGTH=512
HF_DEVICE=cpu

# Optional: Performance Monitoring
ENABLE_PERFORMANCE_MONITORING=true
LOG_LEVEL=INFO
```

**Important**: Replace `your_openai_api_key_here` with your actual OpenAI API key.

### Step 5: Verify Data Structure

Ensure your data directory structure is correct:

```bash
# Check data directory
ls data/

# Expected files:
# - handmade_rugs.xlsx
# - bedside-table.xlsx
# - fabrics.xlsx
```

If files are missing, you'll need to add your product data files in Excel format.

### Step 6: Build FAISS Index

Create the search index for product data:

```bash
# Build FAISS index
python build_index.py
```

**Expected output**:

```
🔧 Building FAISS index...
📂 Loading data from: backend/data
✅ Loaded 3 Excel files
🔍 Creating embeddings...
✅ FAISS index created successfully
💾 Index saved to: backend/data_source/faiss_index/
```

### Step 7: Prepare Training Data

Create training data for the HuggingFace model:

```bash
# Create training directory
mkdir -p data/training

# Create training data file (if not exists)
touch data/training/training_data.jsonl
```

Add training examples to `data/training/training_data.jsonl`:

```json
{"messages": [{"role": "user", "content": "list all categories"}, {"role": "assistant", "content": "META: Here are the available categories..."}]}
{"messages": [{"role": "user", "content": "show me bedside tables"}, {"role": "assistant", "content": "PRODUCT: I found some bedside tables..."}]}
{"messages": [{"role": "user", "content": "what's the weather like"}, {"role": "assistant", "content": "INVALID: I can only help with product-related questions."}]}
{"messages": [{"role": "user", "content": "help"}, {"role": "assistant", "content": "CLARIFY: Could you please be more specific about what you need help with?"}]}
```

**Note**: The training script will also add synthetic data automatically.

### Step 8: Train HuggingFace Model

Train the intent classification model:

```bash
# Train the model
python train_huggingface.py
```

**Expected output**:

```
🚀 HuggingFace Intent Classification Training
============================================================
🔧 Using device: cpu
📂 Loading training data...
✅ Loaded 74 training examples
📊 Training data distribution:
   META: 15
   PRODUCT: 19
   INVALID: 20
   CLARIFY: 20
🔧 Initializing model...
🔧 Preparing data loaders...
✅ Created data loaders:
   Training: 59 examples
   Validation: 15 examples
🚀 Starting training for 5 epochs...

📈 Epoch 1/5
----------------------------------------
   Batch 0/4 - Loss: 1.3862
   Training Loss: 1.3772
   Validation Loss: 1.3619
   Validation Accuracy: 0.3333
   Training Time: 82.01s
   🏆 New best accuracy! Saving model...

[... continues for 5 epochs ...]

🎉 Training completed!
🏆 Best validation accuracy: 0.7333
✅ Training completed successfully!
💾 Model saved to: ./trained_intent_model/
```

**Training time**: 5-10 minutes depending on your system.

### Step 9: Test the Backend

Verify everything is working:

```bash
# Start the backend server
python main.py
```

**Expected output**:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 10: Test API Endpoints

Open a new terminal and test the API:

```bash
# Test health check
curl http://localhost:8000/

# Test chat endpoint
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "list all categories", "session_id": "test"}'
```

**Expected response**:

```json
{
  "response": ["Bedside Tables", "Handmade Rugs", "Fabrics"],
  "type": "category_list"
}
```

## 🔧 Configuration Options

### Intent Classification Implementation

You can switch between different intent classifiers by setting the environment variable:

```bash
# In .env file:
INTENT_IMPLEMENTATION=huggingface    # Fine-tuned model (recommended)
INTENT_IMPLEMENTATION=rule_based     # Pattern matching
INTENT_IMPLEMENTATION=openai         # OpenAI fine-tuned model
INTENT_IMPLEMENTATION=hybrid         # Multiple classifiers
```

### Model Performance Tuning

Adjust confidence thresholds:

```bash
# In .env file:
INTENT_CONFIDENCE_THRESHOLD=0.7      # Default: 0.7
```

### HuggingFace Model Settings

```bash
# In .env file:
HF_MODEL_NAME=distilbert-base-uncased  # Model architecture
HF_NUM_LABELS=4                        # Number of intent classes
HF_MAX_LENGTH=512                      # Max sequence length
HF_DEVICE=cpu                          # cpu or cuda
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. **OpenAI API Key Error**

```
Error: Missing OpenAI API Key
```

**Solution**:

- Verify your `.env` file exists in the backend directory
- Check that `OPENAI_API_KEY` is set correctly
- Ensure no extra spaces or quotes around the API key

#### 2. **Missing Dependencies**

```
ModuleNotFoundError: No module named 'torch'
```

**Solution**:

```bash
# Make sure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

#### 3. **FAISS Index Build Failure**

```
Error: No Excel files found in data directory
```

**Solution**:

- Ensure Excel files exist in `backend/data/`
- Check file permissions
- Verify file format is `.xlsx`

#### 4. **Model Training Issues**

```
Error: Not enough training data
```

**Solution**:

- Add more training examples to `data/training/training_data.jsonl`
- Ensure at least 50 examples total
- Check JSON format is valid

#### 5. **Port Already in Use**

```
Error: Address already in use
```

**Solution**:

```bash
# Use different port
python main.py --port 8001
```

#### 6. **Memory Issues During Training**

```
CUDA out of memory
```

**Solution**:

- Set `HF_DEVICE=cpu` in `.env`
- Reduce batch size in training script
- Close other applications

### Performance Optimization

#### For Better Training Performance:

```bash
# Install CUDA version (if you have GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set device to GPU
HF_DEVICE=cuda
```

#### For Production Deployment:

```bash
# Use production server
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 📊 Verification Checklist

After setup, verify these components are working:

- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Environment variables set
- [ ] FAISS index built successfully
- [ ] Training data prepared
- [ ] HuggingFace model trained
- [ ] Backend server starts without errors
- [ ] API endpoints respond correctly
- [ ] Intent classification working
- [ ] Product retrieval working

## 🔄 Maintenance

### Regular Tasks

1. **Update Dependencies**:

   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Retrain Model** (if needed):

   ```bash
   python train_huggingface.py
   ```

3. **Rebuild Index** (if data changes):

   ```bash
   python build_index.py
   ```

4. **Monitor Logs**:
   ```bash
   # Check for errors in server logs
   tail -f logs/app.log
   ```

### Backup Important Files

- `backend/trained_intent_model/` - Trained HuggingFace model
- `backend/data_source/faiss_index/` - Search index
- `backend/data/training/` - Training data and stats
- `.env` - Environment configuration

## 🎉 Success!

Your backend is now fully set up and ready for production use! The system includes:

- ✅ **FastAPI backend** with RESTful API
- ✅ **FAISS search index** for product retrieval
- ✅ **Trained HuggingFace model** for intent classification
- ✅ **LangGraph workflow** for conversation management
- ✅ **Modular architecture** for easy maintenance

You can now integrate with the frontend or use the API directly for your AI agent application.
