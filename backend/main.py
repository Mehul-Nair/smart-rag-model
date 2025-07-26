from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from rag.retriever import ExcelRetriever
from rag.langgraph_agent import build_langgraph_agent
from typing import Optional
import time
import datetime

# Load environment variables
load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

if OPENAI_API_KEY is None:
    raise ValueError(
        "Missing OpenAI API Key. Please set OPENAI_API_KEY in your environment."
    )

app = FastAPI()

# Allow CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


# Initialize retriever and LangGraph agent at startup
retriever = ExcelRetriever(DATA_DIR, OPENAI_API_KEY)
langgraph_agent = build_langgraph_agent(retriever, OPENAI_API_KEY)


import json


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{timestamp}] Received chat request: {request.message}")
    user_message = request.message

    try:
        print(f"[{timestamp}] Calling langgraph_agent...")
        response = langgraph_agent(user_message)
        print(f"[{timestamp}] langgraph_agent completed successfully")
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        print(f"[{timestamp}] Error in langgraph_agent: {e}")
        print(f"[{timestamp}] Response time: {response_time:.2f} seconds")
        import traceback

        traceback.print_exc()
        result = {
            "response": "Sorry, I encountered an error processing your request.",
            "type": "text",
        }
        print(f"[{timestamp}] Returning error result: {result}")
        return result

    end_time = time.time()
    response_time = end_time - start_time

    print(f"[{timestamp}] Backend response type: {type(response)}")
    print(f"[{timestamp}] Backend response: {response}")
    print(f"[{timestamp}] Response time: {response_time:.2f} seconds")

    # Ensure response is JSON-serializable
    if isinstance(response, BaseModel):
        response = response.dict()

    # Check if response is already a dict (product suggestion) or string
    if isinstance(response, dict):
        response_type = response.get("type")
        if response_type in [
            "product_suggestion",
            "category_not_found",
            "budget_constraint",
            "clarification",
            "error",
            "greeting",  # Added to handle greeting responses
        ]:
            print(f"[{timestamp}] Returning {response_type}")
            result = {"response": response, "type": response_type}
            print(f"[{timestamp}] Returning to frontend: {result}")
            return result
    elif isinstance(response, list):
        print(f"[{timestamp}] Returning category list")
        result = {"response": response, "type": "category_list"}
        print(f"[{timestamp}] Returning to frontend: {result}")
        return result
    else:
        print(f"[{timestamp}] Returning text response")
        result = {"response": response, "type": "text"}
        print(f"[{timestamp}] Returning to frontend: {result}")
        return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
