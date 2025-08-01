from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from rag.retriever import ExcelRetriever
from rag.langgraph_agent import build_langgraph_agent
from typing import Optional
import time
import datetime

# Setup logging for intent classification
from logging_config import setup_intent_classification_logging

# Load environment variables
load_dotenv(".env", override=True)  # override=True ensures .env takes precedence

# Import config after loading environment variables
from config import get_openai_key, validate_openai_key

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Validate OpenAI API key
if not validate_openai_key():
    raise ValueError(
        "Missing or invalid OpenAI API Key. Please set OPENAI_API_KEY in your environment or .env file."
    )

OPENAI_API_KEY = get_openai_key()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    print("🚀 Starting backend initialization...")

    # Setup intent classification logging
    print("📝 Setting up intent classification logging...")
    try:
        intent_logger = setup_intent_classification_logging()
        print("✅ Intent classification logging setup complete")
    except Exception as e:
        print(f"⚠️ Intent classification logging setup failed: {e}")

    # Pre-initialize NER classifier for faster response times
    print("🧠 Pre-initializing NER classifier...")
    try:
        from rag.intent_modules.dynamic_ner_classifier import get_dynamic_ner_classifier

        ner_classifier = get_dynamic_ner_classifier()
        print(f"✅ NER classifier pre-initialized successfully")
        print(f"   - Loaded {len(ner_classifier.product_names)} product names")
        print(f"   - Loaded {len(ner_classifier.brand_names)} brands")
    except Exception as e:
        print(f"⚠️ NER classifier pre-initialization failed: {e}")

    print("🎯 Backend ready to serve requests!")

    yield

    # Shutdown
    print("🛑 Shutting down backend...")


app = FastAPI(lifespan=lifespan)

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
        session_id = request.session_id or "default"
        response = langgraph_agent(user_message, session_id)
        print(f"[{timestamp}] langgraph_agent completed successfully")

        # Debug: Print session state after processing
        from rag.langgraph_agent import GLOBAL_SESSION_STATES

        if session_id in GLOBAL_SESSION_STATES:
            session_state = GLOBAL_SESSION_STATES[session_id]
            print(
                f"[{timestamp}] Session {session_id} slots: {session_state.get('slots', {})}"
            )
            print(
                f"[{timestamp}] Session {session_id} history length: {len(session_state.get('conversation_history', []))}"
            )

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
        response = response.model_dump()

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
                "category_list",  # Added to handle category list responses
                "product_detail",  # Added to handle product detail responses
                "competitor_redirect",  # Added to handle competitor redirect responses
                "competitor_budget_redirect",  # Added to handle competitor budget redirect responses
                "text",  # Added to handle text responses (like warranty info)
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


@app.get("/health")
async def health_check():
    """Health check endpoint to verify system status"""
    try:
        from rag.intent_modules.dynamic_ner_classifier import get_dynamic_ner_classifier

        ner_classifier = get_dynamic_ner_classifier()

        return {
            "status": "healthy",
            "ner_classifier": {
                "initialized": ner_classifier.is_initialized,
                "product_names_count": len(ner_classifier.product_names),
                "brand_names_count": len(ner_classifier.brand_names),
                "total_queries": ner_classifier.total_queries,
                "average_time": (
                    ner_classifier.total_time / ner_classifier.total_queries
                    if ner_classifier.total_queries > 0
                    else 0
                ),
            },
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/debug/session/{session_id}")
async def debug_session(session_id: str):
    """Debug endpoint to check session state"""
    from rag.langgraph_agent import GLOBAL_SESSION_STATES

    if session_id in GLOBAL_SESSION_STATES:
        session_state = GLOBAL_SESSION_STATES[session_id]
        return {
            "session_id": session_id,
            "slots": session_state.get("slots", {}),
            "conversation_history": session_state.get("conversation_history", []),
            "last_prompted_slot": session_state.get("last_prompted_slot"),
            "pending_intent": session_state.get("pending_intent"),
        }
    else:
        return {"error": "Session not found"}


@app.get("/analytics/intent-classification")
async def get_intent_analytics(time_window_minutes: Optional[int] = None):
    """Get intent classification analytics"""
    from rag.intent_modules.intent_analytics import analytics, get_analytics_summary
    from datetime import timedelta

    if time_window_minutes:
        time_window = timedelta(minutes=time_window_minutes)
    else:
        time_window = None

    stats = analytics.get_statistics(time_window)
    summary = get_analytics_summary(time_window)

    return {
        "summary": summary,
        "statistics": stats,
        "rule_based_percentage": analytics.get_rule_based_percentage(time_window),
        "performance_comparison": analytics.get_classifier_performance_comparison(
            time_window
        ),
    }


@app.get("/analytics/export")
async def export_analytics(time_window_minutes: Optional[int] = None):
    """Export analytics data to JSON file"""
    from rag.intent_modules.intent_analytics import analytics
    from datetime import timedelta
    import os

    if time_window_minutes:
        time_window = timedelta(minutes=time_window_minutes)
    else:
        time_window = None

    # Create analytics directory if it doesn't exist
    analytics_dir = os.path.join(os.path.dirname(__file__), "analytics")
    os.makedirs(analytics_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"intent_analytics_{timestamp}.json"
    filepath = os.path.join(analytics_dir, filename)

    analytics.export_data(filepath, time_window)

    return {
        "message": "Analytics exported successfully",
        "filepath": filepath,
        "filename": filename,
    }


@app.get("/logs/intent-classification")
async def get_intent_classification_logs(
    session_id: Optional[str] = None,
    intent: Optional[str] = None,
    limit: Optional[int] = 100,
):
    """Get intent classification logs with optional filtering"""
    try:
        from logging_config import get_intent_classification_logs
        from datetime import datetime, timedelta

        # Get logs from the last 24 hours by default
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)

        logs = get_intent_classification_logs(
            start_time=start_time,
            end_time=end_time,
            session_id=session_id,
            intent=intent,
        )

        # Limit the number of logs returned
        if limit:
            logs = logs[-limit:]

        return {
            "logs": logs,
            "total_count": len(logs),
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "filters": {"session_id": session_id, "intent": intent, "limit": limit},
        }
    except Exception as e:
        return {"error": f"Failed to get logs: {str(e)}"}


@app.get("/logs/classifier-performance")
async def get_classifier_performance_logs(limit: Optional[int] = 50):
    """Get classifier performance logs"""
    try:
        from logging_config import get_intent_classification_logs
        from datetime import datetime, timedelta

        # Get logs from the last 7 days
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)

        logs = get_intent_classification_logs(start_time=start_time, end_time=end_time)

        # Filter for performance-related logs
        performance_logs = [
            log for log in logs if "PERFORMANCE_STATS" in log["message"]
        ]

        if limit:
            performance_logs = performance_logs[-limit:]

        return {
            "performance_logs": performance_logs,
            "total_count": len(performance_logs),
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
        }
    except Exception as e:
        return {"error": f"Failed to get performance logs: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
