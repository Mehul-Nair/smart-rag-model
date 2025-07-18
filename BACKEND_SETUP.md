# Backend Setup Guide

This guide will help you set up the virtual environment and start the backend server for the Decor Intelligence AI project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (to clone the repository)

## Step 1: Navigate to Backend Directory

```bash
cd backend
```

## Step 2: Create Virtual Environment

### On Windows:

```bash
python -m venv venv
```

### On macOS/Linux:

```bash
python3 -m venv venv
```

## Step 3: Activate Virtual Environment

### On Windows:

```bash
venv\Scripts\activate
```

### On macOS/Linux:

```bash
source venv/bin/activate
```

**Note:** You should see `(venv)` at the beginning of your command prompt when the virtual environment is activated.

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:

- FastAPI (web framework)
- Uvicorn (ASGI server)
- OpenAI (AI API client)
- LangChain (AI framework)
- LangGraph (workflow framework)
- Pandas (data manipulation)
- Python-dotenv (environment variables)
- And other dependencies...

## Step 5: Set Up Environment Variables

Create a `.env` file in the backend directory:

```bash
# On Windows
echo OPENAI_API_KEY=your_openai_api_key_here > .env

# On macOS/Linux
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

**Important:** Replace `your_openai_api_key_here` with your actual OpenAI API key.

### Alternative: Manual .env Creation

Create a file named `.env` in the backend directory and add:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Step 6: Verify Data Directory

Ensure the `data` directory exists in the backend folder:

```bash
# Check if data directory exists
ls data

# If it doesn't exist, create it
mkdir data
```

## Step 7: Start the Backend Server

### Development Mode (with auto-reload):

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Step 8: Verify Server is Running

The server should start and display output similar to:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Step 9: Test the API

Open your browser or use curl to test the API:

```bash
# Test if server is running
curl http://localhost:8000

# Test the chat endpoint
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello", "session_id": "test"}'
```

## API Endpoints

- **POST /chat**: Main chat endpoint for AI interactions
- **GET /**: Health check endpoint

## Troubleshooting

### Common Issues:

1. **Port already in use:**

   ```bash
   # Use a different port
   uvicorn main:app --reload --host 0.0.0.0 --port 8001
   ```

2. **Virtual environment not activated:**

   - Make sure you see `(venv)` in your prompt
   - Re-run the activation command

3. **Missing dependencies:**

   ```bash
   pip install -r requirements.txt --upgrade
   ```

4. **OpenAI API Key error:**

   - Verify your `.env` file exists and contains the correct API key
   - Ensure the API key is valid and has sufficient credits

5. **Permission errors (Linux/macOS):**
   ```bash
   chmod +x venv/bin/activate
   ```

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.

## Deactivating Virtual Environment

When you're done working:

```bash
deactivate
```

## Next Steps

Once the backend is running, you can:

1. Start the frontend development server
2. Test the full application
3. Begin development work

## Environment Variables Reference

| Variable         | Description                              | Required |
| ---------------- | ---------------------------------------- | -------- |
| `OPENAI_API_KEY` | Your OpenAI API key for AI functionality | Yes      |

## File Structure

```
backend/
├── venv/                 # Virtual environment
├── data/                 # Data files
├── rag/                  # RAG (Retrieval-Augmented Generation) modules
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
└── BACKEND_SETUP.md    # This file
```
