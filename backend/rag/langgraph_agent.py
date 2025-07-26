import os
from langgraph.graph import StateGraph, END
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import Dict, Any, Optional, List, Union
from typing_extensions import TypedDict
from pydantic import SecretStr, BaseModel, Field, validator
import time
import datetime
import re
import json

# Import the modular intent classifier
from .intent_modules import (
    IntentClassifierFactory,
    IntentType,
    ClassificationResult,
)
from .intent_modules.onnx_ner_classifier import extract_slots_from_text


# --- Advanced Product Detail Handler ---
def resolve_product(slots, history=None):
    product = slots.get("PRODUCT_TYPE")
    if not product and history:
        for turn in reversed(history or []):
            if "PRODUCT_TYPE" in turn.get("slots", {}):
                product = turn["slots"]["PRODUCT_TYPE"]
                break
    return product


def resolve_details(slots, user_query):
    detail = slots.get("PRODUCT_DETAIL")
    if not detail:
        keywords = ["pattern", "material", "dimensions", "color", "type", "description"]
        found = [k for k in keywords if k in user_query.lower()]
        return found if found else ["all"]
    if isinstance(detail, list):
        return detail
    return [detail]


def get_available_fields(doc):
    return list(doc.metadata.keys())


def map_detail_to_field(detail, available_fields):
    field_map = {
        "pattern": "pattern",
        "material": "primary_material",
        "dimensions": "dimension",
        "color": "dominant_color",
        "type": "sub_category",
        "description": "sku_description",
    }
    for k, v in field_map.items():
        if detail.lower() in [k, v] and v in available_fields:
            return v
    if detail in available_fields:
        return detail
    return None


def build_contextual_prompt(product, details, user_query, history, fallback_msg):
    details_text = "\n".join(f"{d.capitalize()}: {v}" for d, v in details.items() if v)
    history_text = ""
    if history:
        history_text = "\n".join(
            f"{turn['role']}: {turn['message']}" for turn in history[-5:]
        )
    prompt = (
        f"User asked: '{user_query}'\n"
        f"Product: {product}\n"
        f"Details found:\n{details_text}\n"
        f"{fallback_msg}\n"
        f"Recent conversation:\n{history_text}\n"
        "Please generate a friendly, informative, and concise reply using only the details found. "
        "If some details are missing, mention only what is available."
    )
    return prompt


def advanced_product_detail_handler(slots, retriever, llm, user_query, history=None):
    product = resolve_product(slots, history)
    if not product:
        return "Which product do you mean? Please specify."
    details_requested = resolve_details(slots, user_query)
    docs = retriever.retrieve(product, k=1)
    if not docs:
        return f"Sorry, I couldn't find details for {product}."
    doc = docs[0]
    available_fields = get_available_fields(doc)
    details = {}
    for detail in details_requested:
        field = map_detail_to_field(detail, available_fields)
        if field and field in doc.metadata:
            details[detail] = doc.metadata[field]
        else:
            details[detail] = None
    missing = [d for d, v in details.items() if not v]
    fallback_msg = ""
    if missing:
        fallback_msg = f"Sorry, I couldn't find: {', '.join(missing)} for {product}."
    prompt = build_contextual_prompt(
        product, details, user_query, history, fallback_msg
    )
    return llm.predict(prompt)


# --- Intent Classification System ---

# Global intent classifier instance - using Improved Hybrid
intent_classifier = IntentClassifierFactory.create(
    "improved_hybrid",
    {
        "confidence_threshold": 0.5,
        "primary_classifier": "huggingface",
        "fallback_classifier": "rule_based",
        "enable_intent_specific_rules": True,
        "implementation_configs": {
            "huggingface": {"model_path": "trained_deberta_model"},
            "rule_based": {"similarity_threshold": 0.5},
        },
    },
)


def set_intent_classifier(classifier):
    """Set the global intent classifier instance"""
    global intent_classifier
    intent_classifier = classifier


def switch_to_huggingface():
    """Switch to HuggingFace implementation"""
    global intent_classifier
    intent_classifier = IntentClassifierFactory.create(
        "huggingface", {"model_path": "trained_deberta_model", "device": "cpu"}
    )
    print("✅ Switched to HuggingFace implementation")


def switch_to_rule_based():
    """Switch to rule-based implementation"""
    global intent_classifier
    intent_classifier = IntentClassifierFactory.create("rule_based")
    print("✅ Switched to rule-based implementation")


def switch_to_improved_hybrid():
    """Switch to improved hybrid implementation"""
    global intent_classifier
    intent_classifier = IntentClassifierFactory.create(
        "improved_hybrid",
        {
            "confidence_threshold": 0.5,
            "primary_classifier": "huggingface",
            "fallback_classifier": "rule_based",
            "enable_intent_specific_rules": True,
            "implementation_configs": {
                "huggingface": {"model_path": "trained_deberta_model"},
                "rule_based": {"similarity_threshold": 0.5},
            },
        },
    )
    print("✅ Switched to improved hybrid implementation")


def switch_to_hybrid():
    """Switch to legacy hybrid implementation"""
    global intent_classifier
    intent_classifier = IntentClassifierFactory.create_hybrid(
        {
            "implementations": ["huggingface", "rule_based"],
            "min_confidence_threshold": 0.7,
            "fallback_strategy": "best_confidence",
            "implementation_configs": {
                "huggingface": {
                    "model_path": "trained_deberta_model",
                    "device": "cpu",
                },
                "rule_based": {"similarity_threshold": 0.5},
            },
        }
    )
    print("✅ Switched to legacy hybrid implementation")


# --- Response Schema Validation ---


class ProductSuggestion(BaseModel):
    """Schema for product suggestion responses"""

    type: str = Field(default="product_suggestion")
    summary: str
    products: List[Dict[str, str]] = Field(
        ..., description="List of products with name, price, url"
    )


class CategoryNotFound(BaseModel):
    """Schema for category not found responses"""

    type: str = Field(default="category_not_found")
    requested_category: str
    available_categories: List[str]
    message: str


class BudgetConstraint(BaseModel):
    """Schema for budget constraint responses"""

    type: str = Field(default="budget_constraint")
    category: str
    requested_budget: str
    message: str


class CategoryList(BaseModel):
    """Schema for category listing responses"""

    type: str = Field(default="category_list")
    categories: List[str]
    message: str


class GreetingResponse(BaseModel):
    """Schema for greeting responses"""

    type: str = Field(default="greeting")
    message: str


class ClarificationRequest(BaseModel):
    """Schema for clarification requests"""

    type: str = Field(default="clarification")
    message: str


class ErrorResponse(BaseModel):
    """Schema for error responses"""

    type: str = Field(default="error")
    message: str


# --- State Management ---


class AgentState(TypedDict, total=False):
    user_message: str
    retriever: Any
    llm: Any
    retrieved_docs: Optional[List[Document]]
    llm_response: Optional[Union[Dict, str]]
    intent: Optional[IntentType]
    intent_confidence: Optional[float]
    intent_scores: Optional[Dict[str, float]]
    response_schema: Optional[str]
    last_prompted_slot: Optional[str]
    slots: Dict[str, str]
    conversation_history: Optional[list]
    pending_intent: Optional[IntentType]
    required_slots: Optional[list]
    slot_prompt_turn: Optional[int]
    corrections: Optional[list]
    conversation_summary: Optional[str]


# --- Slot Templates ---
SLOT_TEMPLATES = {
    # Core product-related intents
    IntentType.PRODUCT_SEARCH: [
        "room_type",
        "product_type",
        "brand",
        "color",
        "material",
        "style",
    ],
    IntentType.PRODUCT_DETAIL: ["product_type", "brand"],
    IntentType.BUDGET_QUERY: ["room_type", "product_type", "budget"],
    IntentType.WARRANTY_QUERY: ["product_type", "brand"],
    # Legacy intents (for backward compatibility)
    IntentType.PRODUCT: ["room_type", "product_type"],
    # IntentType.BUDGET: ["room_type", "product_type", "budget"],  # Legacy - use BUDGET_QUERY
    # No slots needed for these intents
    IntentType.GREETING: [],
    IntentType.HELP: [],
    IntentType.CATEGORY_LIST: [],
    IntentType.INVALID: [],
    IntentType.CLARIFY: [],
    IntentType.META: [],
}


def get_required_slots_for_intent(intent):
    return SLOT_TEMPLATES.get(intent, [])


# --- Centralized Slot Prompting ---
def prompt_for_slot(state: AgentState, slot: str) -> AgentState:
    clarification_msg = f"What type of {slot.replace('_', ' ')} are you looking for?"
    state["last_prompted_slot"] = slot
    state["llm_response"] = {"type": "clarification", "message": clarification_msg}
    state["intent"] = IntentType.CLARIFY
    state["slot_prompt_turn"] = len(state.get("conversation_history", []))
    update_conversation_history(state, "system", clarification_msg)
    print(f"Prompting for slot: {slot}")
    return state


# --- Correction Utility (slightly enhanced) ---
def detect_and_apply_correction(state: AgentState, user_message: str) -> bool:
    # Look for 'actually' and known slot names, try to extract new value
    if "actually" in user_message.lower():
        for slot in state.get("slots", {}):
            if slot in user_message.lower():
                # Try to extract value after slot name
                import re

                match = re.search(
                    rf"{slot}[^a-zA-Z0-9]*([\w\s]+)", user_message, re.IGNORECASE
                )
                if match:
                    new_value = match.group(1).strip().split()[0]
                else:
                    new_value = user_message.split()[-1]
                old_value = state["slots"][slot]
                state["slots"][slot] = new_value
                state.setdefault("corrections", []).append(
                    {
                        "slot": slot,
                        "old": old_value,
                        "new": new_value,
                        "turn": len(state.get("conversation_history", [])),
                    }
                )
                state["llm_response"] = {
                    "type": "clarification",
                    "message": f"Updated {slot} to {new_value}.",
                }
                return True
    return False


# --- Conversation Summarization Utility ---
def maybe_summarize_conversation(state: AgentState, llm):
    history = state.get("conversation_history", [])
    if len(history) > 10:
        summary_prompt = (
            "Summarize the following conversation between a user and an assistant:\n"
            + "\n".join([f"{turn['role']}: {turn['message']}" for turn in history])
        )
        summary = llm.predict(summary_prompt)
        state["conversation_summary"] = summary
        return summary
    return state.get("conversation_summary")


# --- Node Definitions ---


def classify_node(state: AgentState) -> AgentState:
    """Robust intent classification with slot-filling support."""
    node_start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{timestamp}] Starting hybrid classify_node...")
    user_message = state.get("user_message", "")
    print(f"[{timestamp}] User message: {user_message}")
    update_conversation_history(state, "user", user_message)

    # Rule-based greeting detection override (stricter)
    user_lower = user_message.lower().strip()
    # Only match if the message is exactly a greeting/help phrase or is very short
    if user_lower in greeting_keywords or (
        len(user_lower.split()) <= 4
        and any(user_lower.startswith(k) for k in greeting_keywords)
    ):
        print(f"[{timestamp}] Rule-based override: classified as GREETING")
        state["intent"] = IntentType.GREETING
        return state

    # Slot correction
    if detect_and_apply_correction(state, user_message):
        return state

    # Slot timeout/reset
    if state.get("last_prompted_slot") and state.get("slot_prompt_turn") is not None:
        turns_since_prompt = (
            len(state.get("conversation_history", [])) - state["slot_prompt_turn"]
        )
        if turns_since_prompt > 3:
            state["last_prompted_slot"] = None
            state["slot_prompt_turn"] = None
            state["slots"] = {}
            state["llm_response"] = {
                "type": "clarification",
                "message": "Let's start over. What are you looking for?",
            }
            return state

    # Slot-filling: If waiting for a slot, treat input as slot value
    if state.get("last_prompted_slot"):
        slot = state["last_prompted_slot"]
        value = user_message.strip()
        # Normalize slot name to match required_slots
        slot = slot.replace(" ", "_").lower()
        state.setdefault("slots", {})[slot] = value
        print(f"[{timestamp}] Filled slot: {slot} = {value}")
        print(f"[{timestamp}] Current slots: {state['slots']}")
        state["last_prompted_slot"] = None
        state["slot_prompt_turn"] = None
        # Restore pending intent if present
        if "pending_intent" in state:
            state["intent"] = state.pop("pending_intent")
        else:
            state["intent"] = IntentType.PRODUCT
        return state

    # Use hybrid intent classifier
    classification_result = intent_classifier.classify_intent(user_message)
    intent = classification_result.intent
    confidence = classification_result.confidence
    method = classification_result.method
    reasoning = classification_result.reasoning
    scores = classification_result.scores
    processing_time = classification_result.processing_time

    # Store classification details in state
    state["intent"] = intent
    state["intent_confidence"] = confidence
    state["intent_scores"] = scores

    print(f"[{timestamp}] Hybrid classification results:")
    print(f"[{timestamp}] - Intent: {intent}")
    print(f"[{timestamp}] - Confidence: {confidence:.3f}")
    print(f"[{timestamp}] - Method: {method}")
    print(f"[{timestamp}] - Reasoning: {reasoning}")
    print(f"[{timestamp}] - Processing time: {processing_time:.3f}s")

    # Log decision reasoning
    if confidence >= 0.8:
        print(
            f"[{timestamp}] High confidence classification ({confidence:.3f}) using {method}"
        )
    elif confidence >= 0.5:
        print(
            f"[{timestamp}] Medium confidence classification ({confidence:.3f}) using {method}"
        )
    else:
        print(
            f"[{timestamp}] Low confidence classification ({confidence:.3f}) using {method}"
        )

    # Set dynamic required slots
    state["required_slots"] = get_required_slots_for_intent(intent)

    # NER-based slot extraction for product-related intents
    if intent in [
        IntentType.PRODUCT_SEARCH,
        IntentType.PRODUCT_DETAIL,
        IntentType.BUDGET_QUERY,
        IntentType.WARRANTY_QUERY,
        IntentType.PRODUCT,  # Legacy
        # IntentType.BUDGET,  # Legacy - use BUDGET_QUERY
    ]:
        try:
            extracted_slots = extract_slots_from_text(user_message)
            if extracted_slots:
                print(f"[{timestamp}] NER extracted slots: {extracted_slots}")
                # Map NER entity types to slot names
                slot_mapping = {
                    "PRODUCT_TYPE": "product_type",
                    "ROOM_TYPE": "room_type",
                    "BRAND": "brand",
                    "COLOR": "color",
                    "MATERIAL": "material",
                    "BUDGET_RANGE": "budget",
                    "SIZE": "size",
                    "STYLE": "style",
                    "OCCASION": "occasion",
                    "PATTERN": "pattern",
                    "WARRANTY": "warranty",
                    "CATEGORY": "category",
                }

                # Update slots with extracted entities
                for entity_type, value in extracted_slots.items():
                    if entity_type in slot_mapping:
                        slot_name = slot_mapping[entity_type]
                        state.setdefault("slots", {})[slot_name] = value
                        print(f"[{timestamp}] Auto-filled slot: {slot_name} = {value}")
        except Exception as e:
            print(f"[{timestamp}] NER extraction failed: {e}")

    # If a slot is needed, store the intent for after slot-filling
    if intent == IntentType.CLARIFY:
        state["pending_intent"] = intent

    node_end_time = time.time()
    node_response_time = node_end_time - node_start_time
    print(
        f"[{timestamp}] classify_node total response time: {node_response_time:.2f} seconds"
    )

    return state


def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve relevant chunks from the vector store."""
    node_start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{timestamp}] Starting retrieve_node...")
    retriever = state.get("retriever")
    user_message = state.get("user_message", "")

    # Get more documents for the final response
    retrieve_start = time.time()
    docs = retriever.retrieve(user_message, k=20) if retriever else []
    retrieve_end = time.time()
    retrieve_time = retrieve_end - retrieve_start

    state["retrieved_docs"] = docs
    print(
        f"[{timestamp}] Retrieved {len(docs)} documents for response generation in {retrieve_time:.2f} seconds"
    )

    node_end_time = time.time()
    node_response_time = node_end_time - node_start_time
    print(
        f"[{timestamp}] retrieve_node total response time: {node_response_time:.2f} seconds"
    )
    return state


# --- Reason Node ---
def is_out_of_domain(user_message: str) -> bool:
    # Simple heuristic: math/general knowledge patterns
    math_patterns = [
        "+",
        "-",
        "*",
        "/",
        "what is",
        "calculate",
        "sum",
        "difference",
        "multiply",
        "divide",
    ]
    general_knowledge_patterns = [
        "president",
        "prime minister",
        "capital of",
        "weather",
        "joke",
        "news",
        "movie",
        "song",
        "who is",
        "what is",
        "when is",
        "where is",
        "how do I",
        "stock price",
        "population",
        "ceo",
        "formula",
        "history",
        "science",
        "math",
        "number",
        "equation",
        "poem",
        "riddle",
        "trivia",
    ]
    msg = user_message.lower()
    if any(p in msg for p in math_patterns + general_knowledge_patterns):
        return True
    # Add more patterns as needed
    return False


def reason_node(state: AgentState) -> AgentState:
    node_start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Starting reason_node...")
    llm = state.get("llm")
    docs = state.get("retrieved_docs") or []
    user_message = state.get("user_message", "")
    slots = state.get("slots", {})
    required_slots = state.get("required_slots", [])
    # Out-of-domain filter
    if is_out_of_domain(user_message):
        clarification_msg = (
            "Sorry, I can only help with home decor and product-related queries."
        )
        state["llm_response"] = {"type": "clarification", "message": clarification_msg}
        update_conversation_history(state, "system", clarification_msg)
        return state
    # Check for missing slots
    for slot in required_slots:
        if slot not in slots:
            return prompt_for_slot(state, slot)
    # Use conversation summary if available
    summary = maybe_summarize_conversation(state, llm)
    context = "\n".join([doc.page_content for doc in docs])
    # Include recent message history (last 5 turns)
    history = state.get("conversation_history", [])
    recent_history = "\n".join(
        [f"{turn['role']}: {turn['message']}" for turn in history[-5:]]
    )
    # Build prompt using slots, summary, and recent history
    slot_str = ", ".join([f"{k}: {v}" for k, v in slots.items()])
    prompt = (
        (f"Conversation summary: {summary}\n" if summary else "")
        + (f"Recent conversation:\n{recent_history}\n" if recent_history else "")
        + f"User is looking for: {slot_str}.\n"
        "Use ONLY the provided context to answer. If the context does NOT contain relevant information, do NOT attempt to answer.\n"
        "If the context lacks sufficient detail, ask a clear follow-up question instead of assuming.\n"
        "\n"
        "IMPORTANT: If the user asks about anything outside home decor, furniture, or products in the catalog, reply ONLY with:\n"
        '{{\n  "type": "clarification", "message": "Sorry, I can only help with home decor and product-related queries."}}\n'
        "Never answer general knowledge, math, or unrelated questions.\n"
        "\n"
        "Context:\n{context}\n\n"
        "User Message:\n{user_message}\n\n"
        "Assistant:\n"
        "IMPORTANT: Reply ONLY in valid JSON as per the above schema. Do not reply in free text."
    )
    try:
        full_prompt = prompt.format(context=context, user_message=user_message)
        print(f"[{timestamp}] Using LLM to generate response...")
        llm_start = time.time()
        response = llm.predict(full_prompt) if llm else ""
        llm_end = time.time()
        llm_time = llm_end - llm_start
        print(f"[{timestamp}] LLM response generation time: {llm_time:.2f} seconds")
        print(f"[{timestamp}] Raw LLM response: {response}")
        parsed_response = parse_and_validate_response(response, timestamp)
        # Fallback: if parsed_response is an error or invalid, wrap as clarification
        if isinstance(parsed_response, dict) and parsed_response.get("type") == "error":
            state["llm_response"] = {
                "type": "clarification",
                "message": response.strip()
                or "Sorry, I couldn't process your request.",
            }
        else:
            state["llm_response"] = parsed_response
        update_conversation_history(state, "system", str(state["llm_response"]))
    except Exception as e:
        print(f"[{timestamp}] Error in reason_node: {e}")
        import traceback

        traceback.print_exc()
        error_msg = "Sorry, I couldn't process your request."
        state["llm_response"] = {"type": "clarification", "message": error_msg}
        update_conversation_history(state, "system", error_msg)
    node_end_time = time.time()
    node_response_time = node_end_time - node_start_time
    print(
        f"[{timestamp}] reason_node total response time: {node_response_time:.2f} seconds"
    )
    return state


def parse_and_validate_response(response: str, timestamp: str) -> Union[Dict, str]:
    """Parse and validate LLM response with strict schema validation"""
    try:
        # Clean up the response to extract JSON
        response_text = response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Try to parse as JSON first
        try:
            parsed_response = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to convert Python dict to JSON
            json_text = response_text.replace("'", '"')
            json_text = json_text.replace('"None"', "null")
            json_text = json_text.replace('"True"', "true")
            json_text = json_text.replace('"False"', "false")
            parsed_response = json.loads(json_text)

        if isinstance(parsed_response, dict):
            response_type = parsed_response.get("type")

            # Validate against schemas
            try:
                if response_type == "product_suggestion":
                    validated = ProductSuggestion(**parsed_response)
                    return validated.model_dump()
                elif response_type == "category_not_found":
                    validated = CategoryNotFound(**parsed_response)
                    return validated.model_dump()
                elif response_type == "budget_constraint":
                    validated = BudgetConstraint(**parsed_response)
                    return validated.model_dump()
                elif response_type == "clarification":
                    validated = ClarificationRequest(**parsed_response)
                    return validated.model_dump()
                elif response_type == "error":
                    validated = ErrorResponse(**parsed_response)
                    return validated.model_dump()
                else:
                    print(f"[{timestamp}] Unknown response type: {response_type}")
                    return ErrorResponse(message="Invalid response type").model_dump()
            except Exception as validation_error:
                print(f"[{timestamp}] Schema validation error: {validation_error}")
                return ErrorResponse(
                    message="Response failed schema validation"
                ).model_dump()
        else:
            return ErrorResponse(message="Invalid response format").model_dump()

    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"[{timestamp}] JSON parsing error: {e}")
        print(f"[{timestamp}] Response text: {response_text}")
        return ErrorResponse(message="Failed to parse response").model_dump()


def clarify_node(state: AgentState) -> AgentState:
    """Ask for missing info (e.g., room type, budget)."""
    llm = state.get("llm")
    user_message = state.get("user_message", "")

    prompt = (
        "You are a helpful assistant specialized in home decor products.\n"
        "If the user's request is missing important details (like room type, budget, preferred style, or quantity), ask a clear, concise follow-up question to gather that missing information.\n"
        "Do NOT attempt to answer unless required details are available.\n"
        "\n"
        "Example Follow-up:\n"
        "What type of room is this product for? Or do you have a preferred budget range?\n"
        "\n"
        "User Request:\n{user_message}\n\n"
        "Your follow-up question:"
    )
    response = llm.predict(prompt.format(user_message=user_message)) if llm else ""

    # Create structured clarification response
    clarification = ClarificationRequest(message=response)
    state["llm_response"] = clarification.model_dump()
    return state


def reject_node(state: AgentState) -> AgentState:
    """Respond with fallback for irrelevant queries."""
    user_message = state.get("user_message", "")
    retriever = state.get("retriever")
    llm = state.get("llm")

    # Use LLM to intelligently discover available categories from the vector store
    try:
        # Get a sample of documents to understand what's available
        sample_docs = retriever.retrieve("product", k=10) if retriever else []

        if sample_docs and llm:
            # Create context from sample documents
            context = "\n".join([doc.page_content for doc in sample_docs])

            # Use LLM to extract categories intelligently
            category_prompt = f"""
            Based on the following product catalog data, extract 3-5 main product categories or types that customers might ask about.
            
            Catalog Data:
            {context[:1000]}...
            
            Extract only the main product categories (e.g., "bedside tables", "sofas", "curtains", "rugs", "fabrics").
            Return ONLY a comma-separated list of categories, nothing else.
            """

            category_response = llm.predict(category_prompt)
            categories = [
                cat.strip() for cat in category_response.split(",") if cat.strip()
            ]

            if categories:
                category_list = ", ".join(categories[:5])  # Show up to 5 categories
                message = (
                    f"I'm a home decor assistant. I can help you find products like {category_list}, and more. "
                    f"Try asking about specific products or categories you're interested in!"
                )
            else:
                message = (
                    "I'm a home decor assistant. I can help you find various home decor products. "
                    "Try asking about specific products you're looking for!"
                )
        else:
            # Fallback if no documents or LLM
            message = (
                "I'm a home decor assistant. I can help you find various home decor products. "
                "Try asking about specific products you're looking for!"
            )

    except Exception as e:
        print(f"Error in reject_node: {e}")
        message = (
            "I'm a home decor assistant. I can help you find various home decor products. "
            "Try asking about specific products you're looking for!"
        )

    # Create structured rejection response
    rejection = ErrorResponse(message=message)
    state["llm_response"] = rejection.model_dump()
    return state


def meta_node(state: AgentState) -> AgentState:
    """Handle meta queries (categories, greetings, help) with structured responses."""
    node_start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{timestamp}] Starting meta_node...")
    user_message = state.get("user_message", "")
    intent = state.get("intent")
    retriever = state.get("retriever")

    # Handle different meta intents
    if intent == IntentType.GREETING:
        print(f"[{timestamp}] Processing greeting...")
        greeting = GreetingResponse(
            message=(
                "Hello! I'm your home decor assistant. I can help you find furniture, fabrics, rugs, and other home decor products. "
                "You can ask me about specific products, prices, or request items within your budget. "
                "What would you like to find today?"
            )
        )
        state["llm_response"] = greeting.model_dump()
        update_conversation_history(state, "system", greeting.message)

    elif intent == IntentType.HELP:
        print(f"[{timestamp}] Processing help request...")
        help_response = GreetingResponse(
            message=(
                "I'm here to help you find home decor products! Here's what I can do:\n"
                "• Search for specific products (curtains, furniture, lighting, etc.)\n"
                "• List available categories\n"
                "• Find products within your budget\n"
                "• Get product details and warranty information\n"
                "• Recommend products for specific rooms\n\n"
                "What would you like to do?"
            )
        )
        state["llm_response"] = help_response.model_dump()
        update_conversation_history(state, "system", help_response.message)

    elif intent == IntentType.CATEGORY_LIST:
        print(f"[{timestamp}] Processing category list request...")
        categories = get_categories(retriever, timestamp)
        if categories:
            category_response = CategoryList(
                categories=categories,
                message="Here are the available product categories:",
            )
            state["llm_response"] = category_response.model_dump()
            update_conversation_history(state, "system", category_response.message)
        else:
            error_response = ErrorResponse(
                message="Sorry, I couldn't retrieve the category list at the moment. Please try again."
            )
            state["llm_response"] = error_response.model_dump()
            update_conversation_history(state, "system", error_response.message)

    else:
        # Fallback for legacy META intent
        print(f"[{timestamp}] Processing legacy meta intent...")
        # Check for greeting keywords
        user_lower = user_message.lower()
        is_greeting = any(keyword in user_lower for keyword in greeting_keywords)

        if is_greeting:
            greeting = GreetingResponse(
                message=(
                    "Hello! I'm your home decor assistant. I can help you find furniture, fabrics, rugs, and other home decor products. "
                    "You can ask me about specific products, prices, or request items within your budget. "
                    "What would you like to find today?"
                )
            )
            state["llm_response"] = greeting.model_dump()
            update_conversation_history(state, "system", greeting.message)
        else:
            help_response = GreetingResponse(
                message=(
                    "I'm here to help you find home decor products! You can ask me to list categories, "
                    "search for specific products, or get recommendations within your budget. "
                    "What would you like to do?"
                )
            )
            state["llm_response"] = help_response.model_dump()
            update_conversation_history(state, "system", help_response.message)

    node_end_time = time.time()
    node_response_time = node_end_time - node_start_time
    print(
        f"[{timestamp}] meta_node total response time: {node_response_time:.2f} seconds"
    )
    return state


def get_categories(retriever, timestamp: str) -> List[str]:
    """Extract categories from retriever with multiple fallback methods"""
    # Method 1: Direct category access (most efficient)
    if hasattr(retriever, "get_categories"):
        try:
            categories = retriever.get_categories()
            if categories and len(categories) > 0:
                print(f"[{timestamp}] ✅ Categories from direct access: {categories}")
                return categories
            else:
                print(f"[{timestamp}] ⚠️ Direct access returned empty categories")
        except Exception as e:
            print(f"[{timestamp}] ❌ Error in direct category access: {e}")

    # Method 2: Fallback to document analysis
    print(f"[{timestamp}] Using document analysis fallback...")
    try:
        retrieve_start = time.time()
        sample_docs = retriever.retrieve("product category", k=500) if retriever else []
        retrieve_end = time.time()
        retrieve_time = retrieve_end - retrieve_start
        print(
            f"[{timestamp}] Retrieved {len(sample_docs)} documents in {retrieve_time:.2f} seconds"
        )

        categories = set()

        # Extract categories from document metadata
        print(f"[{timestamp}] Extracting categories from document metadata...")
        for doc in sample_docs:
            if hasattr(doc, "metadata") and doc.metadata:
                if "category" in doc.metadata and doc.metadata["category"]:
                    category = str(doc.metadata["category"]).strip()
                    if category:  # Only add non-empty categories
                        categories.add(category)

        print(f"[{timestamp}] Categories found in metadata: {sorted(list(categories))}")

        if categories:
            category_list = sorted(list(categories))
            print(
                f"[{timestamp}] ✅ Returning categories from metadata: {category_list}"
            )
            return category_list
        else:
            print(f"[{timestamp}] ⚠️ No categories found in document metadata")

    except Exception as e:
        print(f"[{timestamp}] ❌ Error in document analysis: {e}")
        import traceback

        traceback.print_exc()

    # Method 3: Hardcoded fallback (last resort)
    print(f"[{timestamp}] Using hardcoded fallback...")
    fallback_categories = ["fabrics", "handmade rugs", "bedside tables"]
    print(f"[{timestamp}] ✅ Returning hardcoded categories: {fallback_categories}")
    return fallback_categories


# --- Slot Processor Node ---
def slot_processor_node(state: AgentState) -> AgentState:
    required_slots = state.get("required_slots", ["room_type"])
    filled_slots = state.get("slots", {})
    missing_slots = [slot for slot in required_slots if slot not in filled_slots]
    if missing_slots:
        return prompt_for_slot(state, missing_slots[0])
    print("All required slots filled. Proceeding to reason node.")
    return state


# --- LangGraph Workflow ---


def build_langgraph_agent(retriever, openai_api_key: str):
    llm = ChatOpenAI(api_key=SecretStr(openai_api_key), temperature=0)

    state_schema = AgentState
    graph = StateGraph(state_schema)

    # Register nodes
    graph.add_node("classify", classify_node)
    graph.add_node("slot_processor", slot_processor_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("reject", reject_node)
    graph.add_node("meta", meta_node)

    # Routing logic
    graph.add_conditional_edges(
        "classify",
        lambda s: s["intent"],
        {
            # Clarification intents (direct to clarify node)
            IntentType.CLARIFY: "clarify",
            # Product-related intents (need retrieval)
            IntentType.PRODUCT_SEARCH: "retrieve",
            IntentType.PRODUCT_DETAIL: "retrieve",
            IntentType.BUDGET_QUERY: "retrieve",
            IntentType.WARRANTY_QUERY: "retrieve",
            IntentType.PRODUCT: "retrieve",  # Legacy
            # Meta intents (direct responses)
            IntentType.GREETING: "meta",
            IntentType.HELP: "meta",
            IntentType.CATEGORY_LIST: "meta",
            IntentType.META: "meta",  # Legacy
            # Error intents
            IntentType.INVALID: "reject",
        },
    )
    graph.add_edge("slot_processor", "retrieve")  # After slot filling, go to retrieval
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", END)
    graph.add_edge("meta", END)
    graph.add_edge("clarify", END)
    graph.add_edge("reject", END)

    # Set start node
    graph.set_entry_point("classify")
    app = graph.compile()

    def run_agent(user_message: str):
        state: AgentState = {
            "user_message": user_message,
            "retriever": retriever,
            "llm": llm,
        }

        # Run the graph with strict routing
        result = app.invoke(state)

        # Return the structured response
        return result.get(
            "llm_response",
            ErrorResponse(
                message="Sorry, I couldn't process your request."
            ).model_dump(),
        )

    return run_agent


# Utility to update conversation history


def update_conversation_history(state: AgentState, role: str, message: str):
    state.setdefault("conversation_history", []).append(
        {"role": role, "message": message}
    )


# Global greeting keywords for rule-based detection

greeting_keywords = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "good evening",
    "good afternoon",
    "how are you",
    "what's up",
    "who are you",
    "help",
    "about",
    "catalog",
    "what can you do",
    "capabilities",
    "introduction",
    "start",
    "begin",
]

# The intent classifier is now initialized at the top of the file using the modular system
