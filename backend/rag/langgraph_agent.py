import os
from langgraph.graph import StateGraph, END
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import Dict, Any, Optional, List, Union

try:
    from typing import TypedDict
except ImportError:
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

# Dynamic NER is imported locally where needed for better performance

# Import configuration
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import (
        COMPANY_NAME,
        COMPANY_BRAND,
        DEFAULT_WARRANTY_PERIOD,
    )
except ImportError:
    # Fallback to environment variables if config import fails
    COMPANY_NAME = os.getenv("COMPANY_NAME", "Asian Paints Beautiful Homes")
    COMPANY_BRAND = os.getenv("COMPANY_BRAND", "Asian Paints Beautiful Homes")
    DEFAULT_WARRANTY_PERIOD = os.getenv("DEFAULT_WARRANTY_PERIOD", "1-year")
    # Dynamic product type mappings are now loaded from the retriever
# No hardcoded mappings needed - they are generated from the dataset


# --- Advanced Product Detail Handler ---
def resolve_product(slots, history=None):
    # First try to get product_name, then fall back to product_type
    product = slots.get("product_name") or slots.get("PRODUCT_TYPE")

    # If we have a product_name, use it directly
    if slots.get("product_name"):
        return slots["product_name"]

    # If we have a PRODUCT_TYPE, try to construct a more specific product name
    if slots.get("PRODUCT_TYPE"):
        product_type = slots["PRODUCT_TYPE"]
        # Try to combine with brand if available
        if slots.get("brand"):
            return f"{slots['brand']} {product_type}"
        # Try to combine with size if available
        elif slots.get("size"):
            return f"{slots['size']} {product_type}"
        else:
            return product_type

    # Fallback to history
    if not product and history:
        for turn in reversed(history or []):
            if "product_name" in turn.get("slots", {}):
                product = turn["slots"]["product_name"]
                break
            elif "PRODUCT_TYPE" in turn.get("slots", {}):
                product = turn["slots"]["PRODUCT_TYPE"]
                break

    return product


def extract_product_name_from_query(user_query):
    """Extract product name from queries like 'give me details of [Product Name]'"""
    import re

    # Pattern to match "details of [Product Name]" or similar
    patterns = [
        r"details of (.+)",
        r"give me details of (.+)",
        r"tell me about (.+)",
        r"what about (.+)",
        r"show me details of (.+)",
        r"get details of (.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, user_query, re.IGNORECASE)
        if match:
            product_name = match.group(1).strip()
            # Remove trailing punctuation
            product_name = re.sub(r"[^\w\s-]+$", "", product_name)
            return product_name

    return None


def resolve_details(slots, user_query):
    detail = slots.get("PRODUCT_DETAIL")
    if not detail:
        # Look for specific detail keywords in the query
        keywords = [
            "pattern",
            "material",
            "dimensions",
            "color",
            "type",
            "description",
            "brand",
            "price",
            "size",
        ]
        found = [k for k in keywords if k in user_query.lower()]
        # If no specific details requested, return comprehensive details
        return (
            found
            if found
            else ["title", "brand", "price", "material", "color", "size", "category"]
        )
    if isinstance(detail, list):
        return detail
    return [detail]


def get_available_fields(doc):
    return list(doc.metadata.keys())


def map_detail_to_field(detail, available_fields):
    field_map = {
        "pattern": "pattern",
        "material": "material",
        "dimensions": "size",
        "color": "color",
        "type": "sub_category",
        "description": "title",
        "brand": "brand",
        "price": "price",
        "category": "category",
        "size": "size",
    }
    for k, v in field_map.items():
        if detail.lower() in [k, v] and v in available_fields:
            return v
    if detail in available_fields:
        return detail
    return None


def build_contextual_prompt(product, details, user_query, history, fallback_msg):
    # Format details in a more structured way
    details_text = ""
    for detail_type, value in details.items():
        if value and str(value).lower() != "nan":
            if detail_type == "title":
                details_text += f"Product Name: {value}\n"
            elif detail_type == "price":
                details_text += f"Price: ₹{value}\n"
            elif detail_type == "brand":
                details_text += f"Brand: {value}\n"
            elif detail_type == "material":
                details_text += f"Material: {value}\n"
            elif detail_type == "color":
                details_text += f"Color: {value}\n"
            elif detail_type == "size":
                details_text += f"Size: {value}\n"
            elif detail_type == "category":
                details_text += f"Category: {value}\n"
            else:
                details_text += f"{detail_type.capitalize()}: {value}\n"

    # Smart history selection: preserve important context while optimizing performance
    history_text = ""
    if history:
        # Always include the last 2 turns for immediate context
        recent_turns = history[-2:]

        # Look for product-related context in the last 10 turns
        product_context_turns = []
        for turn in history[-10:]:
            if any(
                keyword in turn.get("message", "").lower()
                for keyword in [
                    "product",
                    "bed",
                    "sofa",
                    "chair",
                    "table",
                    "rug",
                    "light",
                    "curtain",
                ]
            ):
                product_context_turns.append(turn)

        # Combine recent turns with product context turns (avoid duplicates)
        all_relevant_turns = recent_turns + product_context_turns
        unique_turns = []
        seen_messages = set()

        for turn in all_relevant_turns:
            message = turn.get("message", "")
            if message not in seen_messages:
                unique_turns.append(turn)
                seen_messages.add(message)

        # Limit to reasonable size (max 5 turns to balance context and performance)
        selected_turns = unique_turns[-5:]

        history_text = "\n".join(
            f"{turn['role']}: {turn['message']}" for turn in selected_turns
        )

    prompt = (
        f"User asked: '{user_query}'\n"
        f"Product: {product}\n"
        f"Available Details:\n{details_text}\n"
        f"{fallback_msg}\n"
        f"Recent conversation:\n{history_text}\n"
        "Please provide a concise, friendly, and informative response about this product. "
        "Include all available details in a natural, conversational way. "
        "If some information is missing, only mention what is available. "
        "Keep the response focused and helpful."
    )
    return prompt


def advanced_product_detail_handler(slots, retriever, llm, user_query, history=None):
    # First try to extract product name directly from the query
    product = extract_product_name_from_query(user_query)
    print(f"Extracted product from query: {product}")

    # If not found in query, try to resolve from slots
    if not product:
        product = resolve_product(slots, history)
        print(f"Resolved product from slots: {product}")

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

    # Generate natural language response with timeout handling
    try:
        llm_response = llm.predict(prompt)
    except Exception as e:
        print(f"LLM prediction failed: {e}")
        # Fallback response if LLM fails
        details_text = ""
        for detail_type, value in details.items():
            if value and str(value).lower() != "nan":
                if detail_type == "title":
                    details_text += f"Product Name: {value}. "
                elif detail_type == "price":
                    details_text += f"Price: ₹{value}. "
                elif detail_type == "brand":
                    details_text += f"Brand: {value}. "
                elif detail_type == "material":
                    details_text += f"Material: {value}. "
                elif detail_type == "color":
                    details_text += f"Color: {value}. "
                elif detail_type == "size":
                    details_text += f"Size: {value}. "
                elif detail_type == "category":
                    details_text += f"Category: {value}. "
                else:
                    details_text += f"{detail_type.capitalize()}: {value}. "

        llm_response = f"Here are the details for {product}: {details_text}"

    # Return structured response for UI
    return ProductDetailResponse(
        product_name=product,
        details={k: v for k, v in details.items() if v and str(v).lower() != "nan"},
        message=llm_response,
    )


# --- Intent Classification System ---

# Global intent classifier instance - using hybrid approach with trained model + rule-based fallback
import os

# Import config for centralized API key management
from config import get_openai_key

current_dir = os.path.dirname(os.path.abspath(__file__))
trained_model_path = os.path.join(current_dir, "..", "trained_deberta_model")

intent_classifier = IntentClassifierFactory.create(
    "improved_hybrid",
    {
        "confidence_threshold": 0.7,
        "primary_classifier": "huggingface",
        "fallback_classifier": "rule_based",
        "openai_fallback": True,  # Enable OpenAI as third fallback
        "enable_intent_specific_rules": True,
        "implementation_configs": {
            "huggingface": {
                "model_path": trained_model_path,
                "device": "cpu",
            },
            "rule_based": {"similarity_threshold": 0.3},
            "openai": {
                "model_name": "gpt-3.5-turbo",  # Use regular model for fallback
                "api_key": get_openai_key(),
                "temperature": 0.0,
                "max_tokens": 10,
            },
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
        "huggingface",
        {
            "model_path": trained_model_path,
            "device": "cpu",
            "intent_mapping": {
                "BUDGET_QUERY": 0,
                "CATEGORY_LIST": 1,
                "CLARIFY": 2,
                "GREETING": 3,
                "HELP": 4,
                "INVALID": 5,
                "PRODUCT_DETAIL": 6,
                "PRODUCT_SEARCH": 7,
                "WARRANTY_QUERY": 8,
            },
        },
    )
    print("✅ Switched to HuggingFace implementation")


def switch_to_rule_based():
    """Switch to rule-based implementation"""
    global intent_classifier
    intent_classifier = IntentClassifierFactory.create("rule_based")
    print("✅ Switched to rule-based implementation")


def switch_to_improved_hybrid():
    """Switch to improved hybrid implementation with OpenAI fallback"""
    global intent_classifier
    intent_classifier = IntentClassifierFactory.create(
        "improved_hybrid",
        {
            "confidence_threshold": 0.7,
            "primary_classifier": "huggingface",
            "fallback_classifier": "rule_based",
            "openai_fallback": True,  # Enable OpenAI as third fallback
            "enable_intent_specific_rules": True,
            "implementation_configs": {
                "huggingface": {
                    "model_path": trained_model_path,
                    "device": "cpu",
                },
                "rule_based": {"similarity_threshold": 0.3},
                "openai": {
                    "model_name": "gpt-3.5-turbo",  # Use regular model for fallback
                    "api_key": get_openai_key(),
                    "temperature": 0.0,
                    "max_tokens": 10,
                },
            },
        },
    )
    print("✅ Switched to improved hybrid implementation with OpenAI fallback")


def switch_to_hybrid():
    """Switch to legacy hybrid implementation"""
    global intent_classifier
    intent_classifier = IntentClassifierFactory.create_hybrid(
        {
            "implementations": ["huggingface", "rule_based"],
            "min_confidence_threshold": 0.3,
            "fallback_strategy": "best_confidence",
            "implementation_configs": {
                "huggingface": {
                    "model_path": trained_model_path,
                    "device": "cpu",
                    "intent_mapping": {
                        "BUDGET_QUERY": 0,
                        "CATEGORY_LIST": 1,
                        "CLARIFY": 2,
                        "GREETING": 3,
                        "HELP": 4,
                        "INVALID": 5,
                        "PRODUCT_DETAIL": 6,
                        "PRODUCT_SEARCH": 7,
                        "WARRANTY_QUERY": 8,
                    },
                },
                "rule_based": {"similarity_threshold": 0.3},
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


class ProductDetailResponse(BaseModel):
    """Schema for product detail responses"""

    type: str = Field(default="product_detail")
    product_name: str
    details: Dict[str, str]
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
    session_id: Optional[str]
    product_type_mappings: Optional[Dict[str, str]]


# --- Slot Templates ---
SLOT_TEMPLATES = {
    # Core product-related intents - Only require essential slots
    IntentType.PRODUCT_SEARCH: [
        "product_type",  # Essential - what they want to buy
    ],
    IntentType.PRODUCT_DETAIL: [
        "product_type"
    ],  # Only require product_type, brand is optional
    IntentType.BUDGET_QUERY: [
        "product_type",  # Essential - what they want to buy
        "budget",  # Essential - their budget constraint
    ],
    IntentType.WARRANTY_QUERY: ["product_type", "brand"],
    # Legacy intents (for backward compatibility)
    IntentType.PRODUCT: ["product_type"],  # Simplified - only product_type required
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


def reconstruct_product_name_from_entities(extracted_slots, original_text):
    """Reconstruct a product name from extracted entities"""
    # Priority order for product name reconstruction
    if "PRODUCT_NAME" in extracted_slots:
        return extracted_slots["PRODUCT_NAME"]

    # Special handling for known product name patterns
    # Look for patterns like "City Lights", "Lighting Up Lisbon", "Lights Out Gold"
    known_product_patterns = [
        "city lights",
        "lighting up lisbon",
        "lights out gold",
        "drop dead gorgeous",
        "time to shine",
        "keeper of my heart",
        "bolt",
        "positive energy",
        "purity",
        "skyfall",
        "captured",
    ]

    # Known brand names that should not be treated as product names
    known_brands = [
        "pure royale",
        "white teak",
        "asian paints",
        "bathsense",
        "ador",
        "royale",
    ]

    # Check if the original text contains any known product name patterns
    original_lower = original_text.lower()
    for pattern in known_product_patterns:
        if pattern in original_lower:
            # Extract the full product name from the original text
            # Look for the pattern and extend to get the full product name
            start_idx = original_lower.find(pattern)
            if start_idx != -1:
                # Try to extract the full product name (up to the next punctuation or end)
                end_idx = original_text.find("(", start_idx)
                if end_idx == -1:
                    end_idx = original_text.find(")", start_idx)
                if end_idx == -1:
                    end_idx = len(original_text)

                product_name = original_text[start_idx:end_idx].strip()
                return product_name

    # Check if the text contains a known brand (should be treated as BRAND, not PRODUCT_NAME)
    for brand in known_brands:
        if brand in original_lower:
            # If it's a known brand, don't treat it as a product name
            # Let the NER model handle it as BRAND
            pass

    # Combine multiple product types
    if "PRODUCT_TYPE" in extracted_slots:
        product_types = extracted_slots["PRODUCT_TYPE"].split(", ")
        if len(product_types) > 1:
            return " ".join(product_types)
        else:
            base_product = product_types[0]

    # Combine material + product type
    if "MATERIAL" in extracted_slots and "PRODUCT_TYPE" in extracted_slots:
        material = extracted_slots["MATERIAL"]
        product_type = extracted_slots["PRODUCT_TYPE"].split(", ")[
            0
        ]  # Take first if multiple
        return f"{material} {product_type}"

    # Combine color + product type (if color is not a location word)
    if "COLOR" in extracted_slots and "PRODUCT_TYPE" in extracted_slots:
        color = extracted_slots["COLOR"]
        # Filter out location words that might be misclassified as colors
        location_words = [
            "city",
            "town",
            "village",
            "country",
            "state",
            "place",
            "location",
        ]
        if color.lower() not in location_words:
            product_type = extracted_slots["PRODUCT_TYPE"].split(", ")[0]
            return f"{color} {product_type}"

    # Fallback to product type only
    if "PRODUCT_TYPE" in extracted_slots:
        return extracted_slots["PRODUCT_TYPE"]

    # Last resort: return original text
    return original_text


# --- Enhanced Budget Extraction ---
def extract_budget_from_text(text: str) -> Optional[str]:
    """Extract budget information from text using regex patterns"""
    import re

    # Common budget patterns
    patterns = [
        r"budget\s+of\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees?|₹)?",
        r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees?|₹)\s+budget",
        r"under\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees?|₹)?",
        r"less\s+than\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees?|₹)?",
        r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees?|₹)",
        r"budget\s+(\d+(?:,\d+)*(?:\.\d+)?)",
    ]

    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            budget_value = match.group(1)
            # Clean up the budget value
            budget_value = budget_value.replace(",", "")
            return budget_value

    return None


# --- Centralized Slot Prompting ---
def prompt_for_slot(state: AgentState, slot: str) -> AgentState:
    # Create more specific and helpful clarification messages

    # Get dynamic brand suggestions from the NER classifier
    brand_suggestions = COMPANY_NAME
    try:
        from rag.intent_modules.dynamic_ner_classifier import get_dynamic_ner_classifier

        ner_classifier = get_dynamic_ner_classifier()
        if ner_classifier and ner_classifier.brand_names:
            # Get first few brands as examples
            sample_brands = list(ner_classifier.brand_names)[:5]
            brand_suggestions = ", ".join(sample_brands) + f", {COMPANY_NAME}"
    except Exception as e:
        print(f"Error getting brand suggestions: {e}")

    # Get dynamic product type suggestions from retriever
    product_type_suggestions = (
        "furniture, lights, bath, rugs, furnishing"  # Default fallback
    )
    retriever = state.get("retriever")
    if retriever and hasattr(retriever, "get_product_type_mappings"):
        mappings = retriever.get_product_type_mappings()
        if mappings:
            product_type_suggestions = ", ".join(
                set(mappings.values())
            )  # Get unique categories

    slot_prompts = {
        "product_type": f"What specific type of product are you looking for? (e.g., {product_type_suggestions})",
        "room_type": "Which room or space are you looking to decorate? (e.g., bathroom, bedroom, living room, kitchen, dining room, balcony, garden, office, study, hall, patio, terrace)",
        "budget": "What's your budget range? (e.g., under 1000, 1000-5000, above 10000)",
        "brand": f"Do you have a preferred brand? (e.g., {brand_suggestions})",
        "color": "What color scheme are you looking for? (e.g., white, blue, neutral, colorful)",
        "material": "Any specific material preference? (e.g., wood, metal, plastic, fabric)",
        "style": "What style are you going for? (e.g., modern, traditional, minimalist, rustic)",
    }

    clarification_msg = slot_prompts.get(
        slot, f"What type of {slot.replace('_', ' ')} are you looking for?"
    )
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

    # Rule-based intent detection overrides
    user_lower = user_message.lower().strip()

    # Handle short responses that might be slot values
    if len(user_lower.split()) <= 2 and state.get("last_prompted_slot"):
        # This is likely a slot value, not a new conversation
        pass
    elif len(user_lower.split()) <= 2 and not state.get("last_prompted_slot"):
        # Check if this looks like a product type or category
        product_keywords = [
            "furniture",
            "furnishings",
            "sofa",
            "chair",
            "table",
            "bed",
            "desk",
            "wardrobe",
            "cabinet",
            "curtain",
            "drape",
            "blind",
            "shade",
            "rug",
            "carpet",
            "mat",
            "light",
            "lamp",
            "chandelier",
            "bath",
            "shower",
            "toilet",
            "sink",
            "kitchen",
            "dining",
            "bedroom",
            "living",
            "study",
            "office",
        ]

        if any(keyword in user_lower for keyword in product_keywords):
            # This is likely a product type, treat as PRODUCT_SEARCH
            state["intent"] = IntentType.PRODUCT_SEARCH
            state["intent_confidence"] = 0.8
            state["intent_scores"] = {IntentType.PRODUCT_SEARCH.value: 0.8}
            print(f"[{timestamp}] Short message detected as product type: {user_lower}")
            # Continue with slot extraction below
        else:
            # Let the classifier handle it
            pass

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

    # Slot-filling: If waiting for a slot, try NER first, then fall back to full message
    if state.get("last_prompted_slot"):
        slot = state["last_prompted_slot"]
        # Normalize slot name to match required_slots
        slot = slot.replace(" ", "_").lower()

        # First, try to extract the requested entity type using NER
        try:
            # Use dynamic NER extraction
            from rag.intent_modules.dynamic_ner_classifier import (
                extract_slots_from_text_dynamic,
            )

            extracted_slots = extract_slots_from_text_dynamic(user_message)
            print(
                f"[{timestamp}] Dynamic NER extraction for slot '{slot}': {extracted_slots}"
            )

            # Map slot names to entity types
            slot_to_entity_mapping = {
                "product_type": "PRODUCT_TYPE",
                "room_type": "ROOM",
                "brand": "BRAND",
                "color": "COLOR",
                "material": "MATERIAL",
                "size": "SIZE",
                "style": "STYLE",
                "budget": "BUDGET",
                "product_name": "PRODUCT_NAME",
            }

            target_entity = slot_to_entity_mapping.get(slot)

            # For product_type slot, prioritize PRODUCT_NAME if available
            if slot == "product_type" and "PRODUCT_NAME" in extracted_slots:
                value = extracted_slots["PRODUCT_NAME"]
                print(
                    f"[{timestamp}] ✅ Using PRODUCT_NAME '{value}' for product_type slot"
                )
            elif slot == "product_type" and (
                "PRODUCT_TYPE" in extracted_slots
                or "PRODUCT_NAME" in extracted_slots
                or "MATERIAL" in extracted_slots
            ):
                # Try to reconstruct a product name from multiple entities
                value = reconstruct_product_name_from_entities(
                    extracted_slots, user_message
                )
                print(f"[{timestamp}] ✅ Reconstructed product name: '{value}'")
            elif slot == "brand" and "BRAND" in extracted_slots:
                # Handle brand slot specifically
                value = extracted_slots["BRAND"]
                print(f"[{timestamp}] ✅ Using BRAND '{value}' for brand slot")
            elif target_entity and target_entity in extracted_slots:
                # NER found the requested entity type
                value = extracted_slots[target_entity]

                # Post-process to filter out obviously incorrect classifications
                if target_entity == "COLOR":
                    # Filter out non-color words
                    non_color_words = [
                        "city",
                        "town",
                        "village",
                        "country",
                        "state",
                        "place",
                        "location",
                    ]
                    if value.lower() in non_color_words:
                        print(
                            f"[{timestamp}] ⚠️ Filtered out non-color word '{value}' from COLOR slot"
                        )
                        value = user_message.strip()
                    else:
                        print(f"[{timestamp}] ✅ NER found {slot} = '{value}'")
                else:
                    print(f"[{timestamp}] ✅ NER found {slot} = '{value}'")
            else:
                # NER didn't find the requested entity, use the full message
                value = user_message.strip()
                print(
                    f"[{timestamp}] ⚠️ NER didn't find {slot}, using full message: '{value}'"
                )

        except Exception as e:
            print(f"[{timestamp}] ❌ NER extraction failed: {e}, using full message")
            value = user_message.strip()

        state.setdefault("slots", {})[slot] = value
        print(f"[{timestamp}] Filled slot: {slot} = {value}")
        print(f"[{timestamp}] Current slots: {state['slots']}")
        state["last_prompted_slot"] = None
        state["slot_prompt_turn"] = None

        # Check if we now have all required slots for the pending intent
        if "pending_intent" in state:
            pending_intent = state["pending_intent"]
            required_slots = get_required_slots_for_intent(pending_intent)
            current_slots = state.get("slots", {})

            # If all required slots are filled, restore the pending intent
            if all(slot in current_slots for slot in required_slots):
                state["intent"] = state.pop("pending_intent")
                print(f"[{timestamp}] Restored pending intent: {state['intent']}")
            else:
                # Still missing slots, continue with clarification
                missing_slots = [s for s in required_slots if s not in current_slots]
                if missing_slots:
                    return prompt_for_slot(state, missing_slots[0])
        else:
            # Let the intent classifier determine the intent based on the filled slots
            pass
        return state

    # Use hybrid intent classifier with error handling and analytics
    try:
        # Import analytics module
        from rag.intent_modules.intent_analytics import record_classification_event

        print(
            f"[{timestamp}] 🎯 STARTING INTENT CLASSIFICATION for: '{user_message[:50]}{'...' if len(user_message) > 50 else ''}'"
        )

        classification_result = intent_classifier.classify_intent(user_message)
        if classification_result is None:
            print(f"[{timestamp}] ❌ Intent classifier returned None, using fallback")
            # Fallback to CLARIFY intent
            intent = IntentType.CLARIFY
            confidence = 0.1
            method = "fallback"
            reasoning = "Intent classifier failed, using fallback"
            scores = {}
            processing_time = 0.0

            # Record analytics for fallback
            record_classification_event(
                user_message=user_message,
                classifier_method=method,
                intent=intent.value,
                confidence=confidence,
                processing_time=processing_time,
                session_id=state.get("session_id", "default"),
                success=False,
                error_message="Intent classifier returned None",
            )
        else:
            intent = classification_result.intent
            confidence = classification_result.confidence
            method = classification_result.method
            reasoning = classification_result.reasoning
            scores = classification_result.scores
            processing_time = classification_result.processing_time

            # Enhanced logging for intent classification results
            print(f"[{timestamp}] 🎉 INTENT CLASSIFICATION COMPLETE:")
            print(
                f"[{timestamp}]   📝 Query: '{user_message[:40]}{'...' if len(user_message) > 40 else ''}'"
            )
            print(f"[{timestamp}]   🎯 Intent: {intent.value}")
            print(f"[{timestamp}]   📊 Confidence: {confidence:.3f}")
            print(f"[{timestamp}]   🔧 Method: {method}")
            print(f"[{timestamp}]   ⏱️  Processing Time: {processing_time*1000:.2f}ms")
            print(f"[{timestamp}]   💭 Reasoning: {reasoning}")

            # Record analytics for successful classification
            record_classification_event(
                user_message=user_message,
                classifier_method=method,
                intent=intent.value,
                confidence=confidence,
                processing_time=processing_time,
                session_id=state.get("session_id", "default"),
                success=True,
            )
    except Exception as e:
        print(f"[{timestamp}] ❌ Intent classifier failed with error: {e}")
        # Fallback to CLARIFY intent
        intent = IntentType.CLARIFY
        confidence = 0.1
        method = "error_fallback"
        reasoning = f"Intent classifier error: {e}"
        scores = {}
        processing_time = 0.0

        # Record analytics for error
        try:
            from rag.intent_modules.intent_analytics import record_classification_event

            record_classification_event(
                user_message=user_message,
                classifier_method=method,
                intent=intent.value,
                confidence=confidence,
                processing_time=processing_time,
                session_id=state.get("session_id", "default"),
                success=False,
                error_message=str(e),
            )
        except:
            pass  # Don't let analytics recording break the main flow

    # Trust the intent classifier completely - no overrides
    print(
        f"[{timestamp}] ✅ FINAL INTENT DECISION: {intent.value} (confidence: {confidence:.3f}, method: {method})"
    )

    # Ensure intent is properly set in state
    state["intent"] = intent
    state["intent_confidence"] = confidence
    state["intent_scores"] = scores

    # Log the final intent decision
    print(f"[{timestamp}] FINAL INTENT DECISION: {intent.value}")

    # Set dynamic required slots based on final intent (after overrides)
    final_intent = state[
        "intent"
    ]  # Use the state intent which may have been overridden
    state["required_slots"] = get_required_slots_for_intent(final_intent)
    print(
        f"[{timestamp}] Final intent: {final_intent}, Required slots: {state['required_slots']}"
    )

    # Ensure proper slot initialization for product searches
    if final_intent in [
        IntentType.PRODUCT_SEARCH,
        IntentType.PRODUCT_DETAIL,
        IntentType.BUDGET_QUERY,
    ]:
        if "slots" not in state:
            state["slots"] = {}
        print(f"[{timestamp}] Initialized slots for product intent: {state['slots']}")

    # Enhanced slot extraction for product-related intents
    if intent in [
        IntentType.PRODUCT_SEARCH,
        IntentType.PRODUCT_DETAIL,
        IntentType.BUDGET_QUERY,
        IntentType.WARRANTY_QUERY,
        IntentType.PRODUCT,  # Legacy
        # IntentType.BUDGET,  # Legacy - use BUDGET_QUERY
    ]:
        try:
            # Fast NER-based extraction
            from rag.intent_modules.dynamic_ner_classifier import (
                extract_slots_from_text_dynamic,
            )

            extracted_slots = extract_slots_from_text_dynamic(user_message)
            if extracted_slots:
                print(f"[{timestamp}] NER extracted slots: {extracted_slots}")
                # Map NER entity types to slot names
                slot_mapping = {
                    "PRODUCT_TYPE": "product_type",
                    "PRODUCT_NAME": "product_name",  # Add missing PRODUCT_NAME mapping
                    "ROOM_TYPE": "room_type",
                    "ROOM": "room_type",  # Add this mapping
                    "BRAND": "brand",
                    "COLOR": "color",
                    "MATERIAL": "material",
                    "BUDGET_RANGE": "budget",
                    "BUDGET": "budget",  # Add this mapping
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

            # Fast budget extraction using regex patterns
            if "budget" not in state.get("slots", {}):
                budget_value = extract_budget_from_text(user_message)
                if budget_value:
                    state.setdefault("slots", {})["budget"] = budget_value
                    print(
                        f"[{timestamp}] Enhanced budget extraction: budget = {budget_value}"
                    )

            # Fast product type extraction for common patterns
            if "product_type" not in state.get("slots", {}):
                # Look for common product type patterns
                product_patterns = [
                    r"(furniture|furnishings?|sofa|chair|table|bed|desk|wardrobe|cabinet)",
                    r"(curtain|drape|blind|shade)",
                    r"(rug|carpet|mat)",
                    r"(light|lamp|chandelier|bulb)",
                    r"(bath|shower|toilet|sink)",
                    r"(kitchen|cook|dining)",
                    r"(bedroom|living|dining|study|office)",
                ]

                for pattern in product_patterns:
                    match = re.search(pattern, user_message.lower())
                    if match:
                        product_type = match.group(1)
                        state.setdefault("slots", {})["product_type"] = product_type
                        print(
                            f"[{timestamp}] Enhanced product type extraction: product_type = {product_type}"
                        )
                        break

                # Smart post-processing to extract room information from product types
                # Only extract room_type if product_type actually contains room information
                if "product_type" in state.get("slots", {}):
                    product_type = state["slots"]["product_type"].lower()

                    # Check if product_type already contains room/space information
                    room_keywords = [
                        # Indoor rooms
                        "bathroom",
                        "bedroom",
                        "living",
                        "dining",
                        "kitchen",
                        "office",
                        "study",
                        "garage",
                        "basement",
                        "attic",
                        "hall",
                        "corridor",
                        "passage",
                        "entryway",
                        "foyer",
                        "pantry",
                        "laundry",
                        "mudroom",
                        "closet",
                        "wardrobe",
                        "nursery",
                        "kids",
                        "children",
                        "guest",
                        "master",
                        # Outdoor spaces
                        "balcony",
                        "terrace",
                        "patio",
                        "deck",
                        "porch",
                        "garden",
                        "yard",
                        "backyard",
                        "frontyard",
                        "outdoor",
                        "veranda",
                        "lanai",
                        "sunroom",
                        "conservatory",
                        # Commercial spaces
                        "lobby",
                        "reception",
                        "conference",
                        "meeting",
                        "waiting",
                        "break",
                        "cafeteria",
                        "canteen",
                        # Common variations
                        "room",
                        "area",
                        "space",
                        "zone",
                        "section",
                    ]

                    # Only extract room_type if:
                    # 1. room_type is not already present
                    # 2. product_type actually contains a room keyword
                    # 3. The room keyword is not just a generic term like "accessories"
                    if "room_type" not in state.get("slots", {}):
                        extracted_room = None
                        for room_keyword in room_keywords:
                            if room_keyword in product_type:
                                extracted_room = room_keyword
                                break

                        if extracted_room:
                            state["slots"]["room_type"] = extracted_room
                            print(
                                f"[{timestamp}] Auto-extracted room_type: {extracted_room} from product_type"
                            )
                        else:
                            print(
                                f"[{timestamp}] No room information found in product_type: '{product_type}'"
                            )

                # Debug: Print current slots after extraction
                print(
                    f"[{timestamp}] Current slots after NER: {state.get('slots', {})}"
                )
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
    slots = state.get("slots", {})

    # Build a better search query using extracted product type
    search_query = user_message
    if slots.get("product_type"):
        product_type = slots["product_type"]
        # Use the product type for more targeted search
        search_query = f"{product_type} products"
        print(
            f"[{timestamp}] Using enhanced search query: '{search_query}' (from product_type: {product_type})"
        )

    # Get more documents for the final response
    retrieve_start = time.time()

    # Try multiple search strategies for better results
    all_docs = []

    # Strategy 1: Original search query
    docs1 = retriever.retrieve(search_query, k=10) if retriever else []
    all_docs.extend(docs1)

    # Strategy 2: If we have product type mappings, try searching by sub-category
    product_type_mappings = None
    if retriever and hasattr(retriever, "get_product_type_mappings"):
        product_type_mappings = retriever.get_product_type_mappings()

    if slots.get("product_type") and product_type_mappings:
        product_type = slots["product_type"]
        mapped_category = product_type_mappings.get(product_type.lower())
        if mapped_category:
            # Search for the product type specifically
            sub_category_search = f"Sub Category: {product_type}"
            docs2 = retriever.retrieve(sub_category_search, k=10) if retriever else []
            all_docs.extend(docs2)
            print(
                f"[{timestamp}] Added {len(docs2)} docs from sub-category search: '{sub_category_search}'"
            )

    # Strategy 3: Search for the product type directly
    if slots.get("product_type"):
        product_type = slots["product_type"]
        direct_search = f"{product_type}"
        docs3 = retriever.retrieve(direct_search, k=10) if retriever else []
        all_docs.extend(docs3)
        print(
            f"[{timestamp}] Added {len(docs3)} docs from direct search: '{direct_search}'"
        )

    # Remove duplicates based on content
    unique_docs = []
    seen_content = set()
    for doc in all_docs:
        content_hash = hash(doc.page_content)
        if content_hash not in seen_content:
            unique_docs.append(doc)
            seen_content.add(content_hash)

    docs = unique_docs[:20]  # Limit to 20 docs
    retrieve_end = time.time()
    retrieve_time = retrieve_end - retrieve_start

    state["retrieved_docs"] = docs
    print(
        f"[{timestamp}] Retrieved {len(docs)} documents for response generation in {retrieve_time:.2f} seconds"
    )

    # Debug: Print some sample documents to see what's being retrieved
    if docs:
        print(f"[{timestamp}] Sample retrieved documents:")
        for i, doc in enumerate(docs[:3]):  # Show first 3 docs
            print(f"[{timestamp}] Doc {i+1}: {doc.page_content[:200]}...")
            if hasattr(doc, "metadata") and doc.metadata:
                print(f"[{timestamp}] Doc {i+1} metadata: {doc.metadata}")

    node_end_time = time.time()
    node_response_time = node_end_time - node_start_time
    print(
        f"[{timestamp}] retrieve_node total response time: {node_response_time:.2f} seconds"
    )
    return state


# --- Reason Node ---
def is_out_of_domain(user_message: str, intent_confidence: float = None) -> bool:
    """Check if the message is clearly outside the home decor domain"""
    user_lower = user_message.lower().strip()

    # Debug: Print the message being checked
    print(f"🔍 Checking domain for: '{user_message}' (confidence: {intent_confidence})")

    # If we have high confidence from the ML model, trust it
    if intent_confidence is not None and intent_confidence >= 0.5:
        print(f"✅ High confidence ({intent_confidence:.3f}) - trusting ML model")
        return False

    # Clear out-of-domain patterns (comprehensive)
    out_of_domain_patterns = [
        # Math/calculations
        r"\d+\s*[+\-*/]\s*\d+",  # Mathematical operations
        r"calculate\s+",  # Calculate requests
        r"solve\s+",  # Solve requests
        r"what\s+is\s+\d+\s*[+\-*/]\s*\d+",  # Math questions
        # Only flag clearly out-of-domain questions, not general "what is" questions
        r"who\s+is\s+(narendra|modi|president|prime minister|celebrity|actor|actress|singer|artist|writer|author|scientist|doctor|teacher|student)",  # Specific person questions
        r"what\s+is\s+(capital|population|weather|stock price|birthday|age|time|date|today)",  # Specific non-product questions
        r"capital\s+of\s+",  # Geography questions
        r"weather\s+in\s+",  # Weather questions
        r"weather\s+",  # General weather
        r"stock\s+price\s+of\s+",  # Stock market
        r"population\s+of\s+",  # Population questions
        r"when\s+was\s+",  # Historical questions
        r"where\s+is\s+",  # Location questions
        r"how\s+old\s+is\s+",  # Age questions
        r"birthday\s+of\s+",  # Birthday questions
        # Entertainment
        r"tell\s+me\s+a\s+joke",  # Jokes
        r"play\s+music",  # Music
        r"recommend\s+a\s+movie",  # Movies
        r"song\s+",  # Songs
        r"movie\s+",  # Movies
        r"game\s+",  # Games
        # Technical/Programming
        r"how\s+to\s+code",  # Programming
        r"programming\s+help",  # Programming help
        r"debug\s+",  # Debugging
        r"error\s+",  # Errors
        # Time/Date specific
        r"what\s+time\s+is\s+it",  # Time
        r"what\s+date\s+is\s+it",  # Date
        r"today\s+is\s+",  # Today
        r"current\s+time",  # Current time
        # Sports
        r"football\s+",  # Football
        r"basketball\s+",  # Basketball
        r"tennis\s+",  # Tennis
        r"soccer\s+",  # Soccer
        r"baseball\s+",  # Baseball
        r"hockey\s+",  # Hockey
        r"sport\s+",  # General sports
        # Politics/News
        r"politics\s+",  # Politics
        r"news\s+",  # News
        r"election\s+",  # Elections
        r"president\s+",  # President
        r"prime\s+minister",  # Prime Minister
        r"minister\s+",  # Minister
        # Health/Fitness
        r"health\s+",  # Health
        r"fitness\s+",  # Fitness
        r"exercise\s+",  # Exercise
        r"diet\s+",  # Diet
        r"medicine\s+",  # Medicine
        # Education
        r"education\s+",  # Education
        r"school\s+",  # School
        r"university\s+",  # University
        r"college\s+",  # College
        r"study\s+",  # Study
        # Travel
        r"travel\s+",  # Travel
        r"vacation\s+",  # Vacation
        r"trip\s+",  # Trip
        r"hotel\s+",  # Hotel
        r"flight\s+",  # Flight
        # Cooking/Food
        r"cooking\s+",  # Cooking
        r"recipe\s+",  # Recipe
        r"food\s+",  # Food
        r"restaurant\s+",  # Restaurant
        # Work/Business
        r"work\s+",  # Work
        r"job\s+",  # Job
        r"business\s+",  # Business
        r"company\s+",  # Company
        r"office\s+",  # Office
    ]

    import re

    # Check for clear out-of-domain patterns
    for pattern in out_of_domain_patterns:
        if re.search(pattern, user_lower):
            print(f"❌ Out-of-domain pattern detected: {pattern}")
            return True

    # Check for very short messages that might be unclear
    # Only reject if it's a single word AND doesn't contain any recognizable patterns
    if len(user_lower.split()) <= 1:
        # Allow common greetings and help words
        if any(word in user_lower for word in ["help", "hi", "hello", "hey"]):
            print(f"✅ Short message allowed (greeting/help): '{user_message}'")
            return False

        # Allow if it contains any letters (not just numbers/symbols)
        if not re.search(r"[a-zA-Z]", user_lower):
            print(f"❌ Very short unclear message (no letters): '{user_message}'")
            return True

        # For single words, let the intent classifier decide
        # If it's a single word, it might be a room type, product type, etc.
        # We'll let the ML model handle the classification
        print(f"✅ Short message allowed (let ML model decide): '{user_message}'")
        return False

    print(f"✅ Message is in-domain")
    return False


def validate_products_against_requirements(products, user_requirements):
    """Validate if products actually match user requirements"""
    if not user_requirements:
        return products, []

    matching_products = []
    missing_requirements = []

    # Track which requirements are missing across all products
    all_missing = set()

    for product in products:
        product_text = (
            f"{product.get('name', '')} {product.get('description', '')}".lower()
        )
        matches_all = True

        for req_type, req_value in user_requirements.items():
            if req_value and req_value.lower() != "nan":
                # Normalize the requirement value for better matching
                normalized_req = req_value.lower().strip()

                if req_type == "color" and normalized_req not in product_text:
                    matches_all = False
                    all_missing.add(f"{req_type}:{req_value}")
                elif req_type == "material" and normalized_req not in product_text:
                    matches_all = False
                    all_missing.add(f"{req_type}:{req_value}")
                elif req_type == "size":
                    # Handle size matching more intelligently
                    size_matched = False

                    # Check for exact match
                    if normalized_req in product_text:
                        size_matched = True
                    else:
                        # Handle common size variations
                        size_variations = {
                            "king sized": ["king", "king size", "king-sized"],
                            "queen sized": ["queen", "queen size", "queen-sized"],
                            "double": ["double", "full", "full size"],
                            "single": ["single", "twin", "twin size"],
                        }

                        if normalized_req in size_variations:
                            for variation in size_variations[normalized_req]:
                                if variation in product_text:
                                    size_matched = True
                                    break

                    if not size_matched:
                        matches_all = False
                        all_missing.add(f"{req_type}:{req_value}")

        if matches_all:
            matching_products.append(product)

    # Only report missing requirements if NO products match
    if not matching_products:
        for missing in all_missing:
            req_type, req_value = missing.split(":", 1)
            missing_requirements.append({"type": req_type, "value": req_value})

    return matching_products, missing_requirements


def reason_node(state: AgentState) -> AgentState:
    node_start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Starting reason_node...")
    llm = state.get("llm")
    docs = state.get("retrieved_docs") or []
    user_message = state.get("user_message", "")
    slots = state.get("slots", {})
    required_slots = state.get("required_slots", [])
    intent = state.get("intent")

    # Out-of-domain filter with confidence consideration
    intent_confidence = state.get("intent_confidence")
    if is_out_of_domain(user_message, intent_confidence):
        clarification_msg = (
            "Sorry, I can only help with home decor and product-related queries."
        )
        state["llm_response"] = {"type": "clarification", "message": clarification_msg}
        update_conversation_history(state, "system", clarification_msg)
        return state

    # Handle PRODUCT_DETAIL intent with advanced product detail handler
    if intent == IntentType.PRODUCT_DETAIL:
        print(
            f"[{timestamp}] Using advanced product detail handler for PRODUCT_DETAIL intent"
        )
        try:
            retriever = state.get("retriever")
            history = state.get("conversation_history", [])
            response = advanced_product_detail_handler(
                slots, retriever, llm, user_message, history
            )
            state["llm_response"] = response
            return state
        except Exception as e:
            print(f"[{timestamp}] Error in advanced product detail handler: {e}")
            # Fall back to generic handling

        # Handle WARRANTY_QUERY intent with dynamic warranty handler
    if intent == IntentType.WARRANTY_QUERY:
        print(f"[{timestamp}] Using dynamic warranty handler for WARRANTY_QUERY intent")
        try:
            product_type = slots.get("product_type", "")
            brand = slots.get("brand", "")
            product_name = slots.get("product_name", "")
            retriever = state.get("retriever")

            # Retrieve relevant documents to find warranty information
            search_queries = []

            # Try different search strategies
            if product_name:
                search_queries.append(f"warranty {product_name}")
            if product_type:
                search_queries.append(f"warranty {product_type}")
            if brand:
                search_queries.append(f"warranty {brand}")

            # Also search for the product directly to get all its information
            if product_name:
                search_queries.append(product_name)
            elif product_type:
                search_queries.append(product_type)

            print(
                f"[{timestamp}] Searching for warranty info with queries: {search_queries}"
            )

            # Collect documents from all search queries
            all_warranty_docs = []
            for query in search_queries:
                docs = retriever.retrieve(query, k=3)
                all_warranty_docs.extend(docs)

            # Remove duplicates based on content
            unique_docs = []
            seen_content = set()
            for doc in all_warranty_docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(content_hash)

            # Extract warranty information from retrieved documents
            warranty_info = []
            product_details = []

            for doc in unique_docs:
                # Extract warranty information
                if "warranty" in doc.page_content.lower():
                    warranty_info.append(doc.page_content)
                if hasattr(doc, "metadata") and doc.metadata.get("warranty"):
                    warranty_info.append(f"Warranty: {doc.metadata['warranty']}")

                # Extract product details for context
                product_detail_parts = []
                if hasattr(doc, "metadata"):
                    metadata = doc.metadata
                    if metadata.get("title"):
                        product_detail_parts.append(f"Product: {metadata['title']}")
                    if metadata.get("brand_name"):
                        product_detail_parts.append(f"Brand: {metadata['brand_name']}")
                    if metadata.get("primary_material"):
                        product_detail_parts.append(
                            f"Material: {metadata['primary_material']}"
                        )
                    if metadata.get("colour"):
                        product_detail_parts.append(f"Color: {metadata['colour']}")
                    if metadata.get("size"):
                        product_detail_parts.append(f"Size: {metadata['size']}")

                if product_detail_parts:
                    product_details.append(" | ".join(product_detail_parts))

            if warranty_info:
                # Use actual warranty data from the database
                warranty_context = "\n".join(
                    warranty_info[:3]
                )  # Use first 3 relevant pieces
                product_context = (
                    "\n".join(product_details[:2]) if product_details else ""
                )

                warranty_prompt = f"""You are a helpful customer service representative for {COMPANY_NAME}. 

The customer is asking about warranty for: {product_name or product_type} from {brand or COMPANY_NAME}.

Here is the actual warranty information from our product database:
{warranty_context}

{f"Product Details: {product_context}" if product_context else ""}

Please provide a professional, helpful warranty response that:
1. Acknowledges the specific product they're asking about
2. Uses the actual warranty information provided above
3. Explains the warranty terms clearly
4. Maintains focus on {COMPANY_NAME} brand

IMPORTANT: Respond with a simple, friendly text message. Do NOT use JSON format or product suggestion format. Just provide a natural warranty explanation.

Response:"""

                warranty_message = (
                    llm.predict(warranty_prompt)
                    if llm
                    else f"Based on our product database, here's the warranty information for {product_name or product_type}: {warranty_context}"
                )

                # Ensure the response is a simple text message, not a product suggestion
                if (
                    warranty_message.strip().startswith("{")
                    and '"type"' in warranty_message
                ):
                    # If LLM returned JSON, extract just the message part
                    try:
                        import json

                        parsed = json.loads(warranty_message)
                        if "message" in parsed:
                            warranty_message = parsed["message"]
                        elif "summary" in parsed:
                            warranty_message = parsed["summary"]
                    except:
                        # If JSON parsing fails, use a fallback message
                        warranty_message = f"Based on our product database, here's the warranty information for {product_name or product_type}: {warranty_context}"
            else:
                # Fallback to default warranty if no specific info found
                warranty_prompt = f"""You are a helpful customer service representative for {COMPANY_NAME}. 

The customer is asking about warranty for: {product_name or product_type} from {brand or COMPANY_NAME}.

Please provide a professional, helpful warranty response that:
1. Acknowledges the specific product they're asking about
2. Explains our standard warranty terms
3. Offers to help with extended warranty options
4. Maintains focus on {COMPANY_NAME} brand

IMPORTANT: Respond with a simple, friendly text message. Do NOT use JSON format or product suggestion format. Just provide a natural warranty explanation.

Response:"""

                warranty_message = (
                    llm.predict(warranty_prompt)
                    if llm
                    else f"Regarding warranty for {product_name or product_type} from {brand or COMPANY_NAME}: All our products come with a standard {DEFAULT_WARRANTY_PERIOD} manufacturer warranty covering defects in materials and workmanship. For extended warranty options or specific warranty terms, please contact our customer service team."
                )

            warranty_response = {"type": "text", "message": warranty_message}

            state["llm_response"] = warranty_response
            return state
        except Exception as e:
            print(f"[{timestamp}] Error in warranty handler: {e}")
            # Fall back to generic handling

    # Check for missing slots
    print(f"[{timestamp}] Required slots: {required_slots}")
    print(f"[{timestamp}] Current slots: {slots}")
    for slot in required_slots:
        if slot not in slots:
            print(f"[{timestamp}] Missing slot: {slot}")
            return prompt_for_slot(state, slot)
    print(f"[{timestamp}] All required slots filled, proceeding to LLM")

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
        "You are a helpful home decor assistant. Provide product suggestions based on the user's requirements.\n"
        "Available response types:\n"
        "1. product_suggestion: When you have relevant products to suggest\n"
        "2. budget_constraint: When the user's budget is too low for available products\n"
        "3. category_not_found: When the requested category doesn't exist\n"
        "4. clarification: When you need more information\n"
        "\n"
        "Response schemas:\n"
        "For product suggestions:\n"
        '{{\n  "type": "product_suggestion",\n  "summary": "Brief summary of what you found",\n  "products": [\n    {{\n      "name": "Product Name",\n      "category": "Product Category",\n      "price": "Price",\n      "description": "Brief description"\n    }}\n  ]\n}}\n'
        "\n"
        "For budget constraints:\n"
        '{{\n  "type": "budget_constraint",\n  "category": "Product category",\n  "requested_budget": "User\'s budget",\n  "message": "Explanation of budget constraint"\n}}\n'
        "\n"
        "For clarifications:\n"
        '{{\n  "type": "clarification",\n  "message": "Your clarification question"\n}}\n'
        "\n"
        + (f"Conversation summary: {summary}\n" if summary else "")
        + (f"Recent conversation:\n{recent_history}\n" if recent_history else "")
        + f"User is looking for: {slot_str}.\n"
        "Use ONLY the provided context to answer. If the context does NOT contain relevant information, do NOT attempt to answer.\n"
        "If the context lacks sufficient detail, ask a clear follow-up question instead of assuming.\n"
        "\n"
        "IMPORTANT: The user is asking about home decor products. Focus on providing helpful product suggestions based on the context.\n"
        "If you find relevant products in the context, suggest them. If the budget is too low, explain the budget constraint.\n"
        "Only ask for clarification if you genuinely need more information to provide a good suggestion.\n"
        "\n"
        "Context:\n{context}\n\n"
        "User Message:\n{user_message}\n\n"
        "Assistant:\n"
        "IMPORTANT: Reply ONLY in valid JSON as per the above schema. Do not reply in free text."
    )
    try:
        # Extract user requirements from slots for validation
        user_requirements = {}
        if "color" in slots:
            user_requirements["color"] = slots["color"]
        if "material" in slots:
            user_requirements["material"] = slots["material"]
        if "size" in slots:
            user_requirements["size"] = slots["size"]

        # Get available categories for category_not_found responses
        retriever = state.get("retriever")
        available_categories = []
        if retriever:
            try:
                available_categories = get_categories(retriever, timestamp)
            except Exception as e:
                print(f"[{timestamp}] Error getting categories: {e}")
                available_categories = [
                    "furnishing",
                    "lights",
                    "bath",
                    "rugs",
                    "furniture",
                ]  # fallback

        # Simplified prompt to avoid syntax issues
        summary_text = f"Conversation summary: {summary}\n" if summary else ""
        history_text = (
            f"Recent conversation:\n{recent_history}\n" if recent_history else ""
        )
        categories_text = (
            f"Available categories: {available_categories}\n"
            if available_categories
            else ""
        )

        # Get product type mappings for better category understanding
        product_type_mappings_text = ""

        # Get dynamic mappings from retriever
        retriever = state.get("retriever")
        if retriever and hasattr(retriever, "get_product_type_mappings"):
            mappings = retriever.get_product_type_mappings()
            if mappings:
                mappings_list = [f"'{k}' → '{v}'" for k, v in mappings.items()]
                product_type_mappings_text = (
                    f"\nProduct Type Mappings: {', '.join(mappings_list)}\n"
                )
                print(f"[{timestamp}] Dynamic product type mappings: {mappings_list}")
            else:
                print(f"[{timestamp}] No dynamic product type mappings found")
        else:
            print(f"[{timestamp}] Product type mappings not available")

        full_prompt = f"""You are a helpful home decor assistant. Provide product suggestions based on the user's requirements.

IMPORTANT RULES:
1. ONLY suggest products that are explicitly mentioned in the context
2. If the context doesn't contain products matching the user's requirements, be honest about it
3. Do NOT claim to have products in specific colors/materials if they're not in the context
4. If you can't find exact matches, suggest alternatives but clearly state what's missing
5. ALWAYS check the context first before deciding if a category is not found
6. If the context contains products of the requested type, suggest them regardless of category names
7. CRITICAL: Product types are mapped to categories. When a user asks for a specific product type (like "bed"), you should look for products in the mapped category (like "furniture"). Do NOT look for the product type as a direct category.
8. If the user asks for "bed", look in the "furniture" category. If they ask for "lamp", look in the "lights" category. If they ask for "rug", look in the "rugs" category.

Available response types:
1. product_suggestion: When you have relevant products to suggest
2. budget_constraint: When the user's budget is too low for available products
3. category_not_found: When the requested category doesn't exist AND no relevant products are found in context
4. clarification: When you need more information

Response schemas:
For product suggestions:
{{
  "type": "product_suggestion",
  "summary": "Brief summary of what you found",
  "products": [
    {{
      "name": "Product Name",
      "category": "Product Category",
      "price": "Price",
      "description": "Brief description"
    }}
  ]
}}

For budget constraints:
{{
  "type": "budget_constraint",
  "category": "Product category",
  "requested_budget": "User's budget",
  "message": "Explanation of budget constraint"
}}

For clarifications:
{{
  "type": "clarification",
  "message": "Your clarification question"
}}

For category not found:
{{
  "type": "category_not_found",
  "requested_category": "The category the user requested",
  "available_categories": ["list", "of", "available", "categories"],
  "message": "Explanation of what categories are available"
}}

{summary_text}{history_text}{categories_text}{product_type_mappings_text}User is looking for: {slot_str}.

IMPORTANT BUDGET HANDLING:
- If the user has specified a budget, ONLY suggest products within that budget
- If no products are available within the budget, use budget_constraint response type
- Always consider the budget when making suggestions

CRITICAL: Check the context first! If the context contains products that match what the user is looking for, suggest them as product_suggestion, even if the category names don't exactly match. Only use category_not_found if the context is completely empty or contains no relevant products.

IMPORTANT: When the user asks for a specific product type (like "bed"), the context should contain products from the mapped category (like "furniture"). Look for products in the context that match the user's request, regardless of the exact category names.

SPECIFIC INSTRUCTIONS FOR BED PRODUCTS:
- When the user asks for "bed", look for products with "Sub Category: bed" in the context
- These bed products are typically found in the "Furniture" category
- If you see products with "Sub Category: bed" in the context, suggest them as product_suggestion
- Do NOT say "bed" category is not found if you see bed products in the context

Use ONLY the provided context to answer. If the context does NOT contain relevant information, do NOT attempt to answer.
If the context lacks sufficient detail, ask a clear follow-up question instead of assuming.

IMPORTANT: The user is asking about home decor products. Focus on providing helpful product suggestions based on the context.
If you find relevant products in the context, suggest them. If the budget is too low, explain the budget constraint.
Only ask for clarification if you genuinely need more information to provide a good suggestion.

Context:
{context}

User Message:
{user_message}

Assistant:
IMPORTANT: Reply ONLY in valid JSON as per the above schema. Do not reply in free text.

DEBUG: Before responding, check if the context contains any products with "Sub Category: bed" or mentions of bed products. If it does, you MUST suggest them as product_suggestion, not category_not_found."""
        print(f"[{timestamp}] Using LLM to generate response...")
        llm_start = time.time()
        response = llm.predict(full_prompt) if llm else ""
        llm_end = time.time()
        llm_time = llm_end - llm_start
        print(f"[{timestamp}] LLM response generation time: {llm_time:.2f} seconds")
        print(f"[{timestamp}] Raw LLM response: {response}")

        # Validate the response against user requirements
        try:
            import json

            response_data = json.loads(response)
            if (
                response_data.get("type") == "product_suggestion"
                and "products" in response_data
            ):
                products = response_data["products"]
                matching_products, missing_requirements = (
                    validate_products_against_requirements(products, user_requirements)
                )

                if missing_requirements:
                    # Update the response to be honest about missing requirements
                    missing_text = ", ".join(
                        [
                            f"{req['type']} ({req['value']})"
                            for req in missing_requirements
                        ]
                    )
                    response_data["summary"] = (
                        f"I couldn't find products with the exact {missing_text} you requested, but here are some alternatives:"
                    )
                    response_data["products"] = matching_products
                    response = json.dumps(response_data)
                    print(
                        f"[{timestamp}] Updated response to reflect missing requirements: {missing_text}"
                    )
        except Exception as e:
            print(f"[{timestamp}] Error validating response: {e}")

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

        # Provide more specific error handling based on intent
        intent = state.get("intent")
        if intent == IntentType.PRODUCT_SEARCH:
            error_msg = "I'm having trouble finding products right now. Could you please try rephrasing your request or ask about a specific category?"
        elif intent == IntentType.CATEGORY_LIST:
            error_msg = "I'm having trouble retrieving the category list. Please try again in a moment."
        else:
            error_msg = "Sorry, I couldn't process your request. Please try rephrasing or ask for help."

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
    """Ask for missing info or handle unclear requests gracefully."""
    llm = state.get("llm")
    user_message = state.get("user_message", "")
    intent = state.get("intent")
    intent_confidence = state.get("intent_confidence", 0.0)

    # Handle unknown or unclear intents gracefully
    if intent is None or intent_confidence < 0.3:
        # This is an unclear request - ask for clarification in a friendly way
        clarification_msg = (
            "I'd love to help you find the perfect home decor items! Could you tell me a bit more about what you're looking for? "
            "For example:\n"
            "• What type of product are you interested in? (furniture, curtains, rugs, lighting, etc.)\n"
            "• Which room are you decorating?\n"
            "• Do you have a budget in mind?\n"
            "• Any specific style preferences?"
        )
        state["llm_response"] = {"type": "clarification", "message": clarification_msg}
        update_conversation_history(state, "system", clarification_msg)
        return state

    # Handle known intents that need more information
    prompt = (
        "You are a helpful assistant specialized in home decor products.\n"
        "If the user's request is missing important details (like room type, budget, preferred style, or quantity), ask a clear, concise follow-up question to gather that missing information.\n"
        "Do NOT attempt to answer unless required details are available.\n"
        "\n"
        "Example Follow-up:\n"
        "What type of room is this product for? Or do you have a preferred budget range?\n"
        "\n"
        f"User Request:\n{user_message}\n\n"
        "Your follow-up question:"
    )
    response = llm.predict(prompt) if llm else ""

    # Create structured clarification response
    clarification = ClarificationRequest(message=response)
    state["llm_response"] = clarification.model_dump()
    update_conversation_history(state, "system", response)
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
        try:
            categories = get_categories(retriever, timestamp)
            print(f"[{timestamp}] Retrieved categories: {categories}")

            if categories and len(categories) > 0:
                category_response = CategoryList(
                    categories=categories,
                    message="Here are the available product categories:",
                )
                state["llm_response"] = category_response.model_dump()
                update_conversation_history(state, "system", category_response.message)
                print(f"[{timestamp}] ✅ Category list response generated successfully")
            else:
                error_response = ErrorResponse(
                    message="Sorry, I couldn't retrieve the category list at the moment. Please try again."
                )
                state["llm_response"] = error_response.model_dump()
                update_conversation_history(state, "system", error_response.message)
                print(f"[{timestamp}] ❌ No categories found")
        except Exception as e:
            print(f"[{timestamp}] ❌ Error in category list processing: {e}")
            error_response = ErrorResponse(
                message="Sorry, I encountered an error while retrieving the category list. Please try again."
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

                # Normalize categories to remove duplicates and standardize case
                normalized_categories = []
                seen_lowercase = set()

                for category in categories:
                    category_lower = category.lower()
                    if category_lower not in seen_lowercase:
                        # Prefer the version with capital F for "furnishing"
                        if category_lower == "furnishing":
                            normalized_categories.append("Furnishing")
                        else:
                            # For other categories, use the first occurrence
                            normalized_categories.append(category)
                        seen_lowercase.add(category_lower)

                print(
                    f"[{timestamp}] ✅ Normalized categories: {normalized_categories}"
                )
                return normalized_categories
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
    user_message = state.get("user_message", "").lower()

    # Check if user is referring to a previous product with "this", "it", "that"
    if any(
        word in user_message
        for word in ["this", "it", "that", "the product", "the item"]
    ):
        print(f"[SLOT_PROCESSOR] Processing follow-up question: {user_message}")
        print(f"[SLOT_PROCESSOR] Current slots: {filled_slots}")

        # Try to infer missing slots from context
        if "product_type" not in filled_slots and "product_name" in filled_slots:
            # If we have a product name but no type, try to infer from the name dynamically
            product_name = filled_slots.get("product_name", "")
            product_name_lower = product_name.lower()

            # Use dynamic product type mappings from retriever
            retriever = state.get("retriever")
            if retriever and hasattr(retriever, "get_product_type_mappings"):
                mappings = retriever.get_product_type_mappings()
                if mappings:
                    for keyword, category in mappings.items():
                        if keyword in product_name_lower:
                            filled_slots["product_type"] = category
                            print(
                                f"[SLOT_PROCESSOR] Auto-filled product_type '{category}' from keyword '{keyword}' in product name"
                            )
                            break
                    else:
                        # If no keyword match found, try to extract from product name structure
                        if " - " in product_name:
                            # Extract the part after the dash which might contain product type
                            after_dash = product_name.split(" - ")[1].lower()
                            for keyword, category in mappings.items():
                                if keyword in after_dash:
                                    filled_slots["product_type"] = category
                                    print(
                                        f"[SLOT_PROCESSOR] Auto-filled product_type '{category}' from keyword '{keyword}' in product description"
                                    )
                                    break

        if "brand" not in filled_slots and "product_name" in filled_slots:
            # Try to extract brand from product name dynamically
            product_name = filled_slots.get("product_name", "")

            # Extract the first part of the product name (usually the brand)
            if " - " in product_name:
                brand_from_name = product_name.split(" - ")[0].strip()
                filled_slots["brand"] = brand_from_name
                print(
                    f"[SLOT_PROCESSOR] Auto-filled brand from product name: {brand_from_name}"
                )
            else:
                # If no clear brand separator, use the first word as brand
                first_word = product_name.split()[0] if product_name else ""
                if first_word:
                    filled_slots["brand"] = first_word
                    print(
                        f"[SLOT_PROCESSOR] Auto-filled brand from first word: {first_word}"
                    )
                else:
                    filled_slots["brand"] = COMPANY_NAME
                    print(f"[SLOT_PROCESSOR] Using default brand: {COMPANY_NAME}")

        # Update the state with the filled slots
        state["slots"] = filled_slots
        print(f"[SLOT_PROCESSOR] Updated slots: {filled_slots}")

    missing_slots = [slot for slot in required_slots if slot not in filled_slots]
    if missing_slots:
        print(f"[SLOT_PROCESSOR] Missing slots: {missing_slots}")
        return prompt_for_slot(state, missing_slots[0])
    print("[SLOT_PROCESSOR] All required slots filled. Proceeding to reason node.")
    return state


# --- LangGraph Workflow ---


def build_langgraph_agent(retriever, openai_api_key: str):
    llm = ChatOpenAI(
        api_key=SecretStr(openai_api_key),
        temperature=0,
        request_timeout=30,  # 30 second timeout to prevent very slow responses
    )

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

    # Routing logic with enhanced debugging
    def route_by_intent(state):
        intent = state.get("intent")
        print(f"[ROUTING] Intent: {intent}")

        if intent == IntentType.CATEGORY_LIST:
            print(f"[ROUTING] 🎯 Routing CATEGORY_LIST to meta_node")
            return "meta"
        elif intent == IntentType.GREETING:
            print(f"[ROUTING] 🎯 Routing GREETING to meta_node")
            return "meta"
        elif intent == IntentType.HELP:
            print(f"[ROUTING] 🎯 Routing HELP to meta_node")
            return "meta"
        elif intent == IntentType.CLARIFY:
            print(f"[ROUTING] 🎯 Routing CLARIFY to clarify_node")
            return "clarify"
        elif intent in [
            IntentType.PRODUCT_SEARCH,
            IntentType.PRODUCT_DETAIL,
            IntentType.BUDGET_QUERY,
            IntentType.PRODUCT,
        ]:
            print(f"[ROUTING] 🎯 Routing {intent.value} to retrieve_node")
            return "retrieve"
        elif intent == IntentType.WARRANTY_QUERY:
            print(
                f"[ROUTING] 🎯 Routing WARRANTY_QUERY to slot_processor (for context handling)"
            )
            return "slot_processor"
        elif intent == IntentType.INVALID:
            print(f"[ROUTING] 🎯 Routing INVALID to reject_node")
            return "reject"
        elif intent is None:
            # Handle unknown/None intents gracefully
            print(
                f"[ROUTING] ⚠️ Unknown/None intent, routing to clarify_node for graceful handling"
            )
            return "clarify"
        else:
            print(
                f"[ROUTING] ⚠️ Unknown intent {intent}, routing to clarify_node for graceful handling"
            )
            return "clarify"

    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "meta": "meta",
            "clarify": "clarify",
            "retrieve": "retrieve",
            "slot_processor": "slot_processor",
            "reject": "reject",
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

    def run_agent(user_message: str, session_id: str = "default"):
        # Get or create session state
        if session_id not in GLOBAL_SESSION_STATES:
            GLOBAL_SESSION_STATES[session_id] = {
                "conversation_history": [],
                "slots": {},
                "last_prompted_slot": None,
                "slot_prompt_turn": None,
                "pending_intent": None,
                "corrections": [],
                "conversation_summary": None,
            }

        # Get existing session state
        session_state = GLOBAL_SESSION_STATES[session_id]

        # Smart slot management: Clear slots if this appears to be a new conversation
        # Check if the user message seems like a new query vs continuation
        user_lower = user_message.lower().strip()

        # Define greeting keywords
        greeting_keywords = [
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "greetings",
            "howdy",
            "what's up",
        ]

        # More intelligent new conversation detection - preserve context better
        is_new_conversation = (
            # Only clear for explicit greetings or meta questions
            any(greeting in user_lower for greeting in greeting_keywords)
            or
            # Category listing questions (always new)
            "categories" in user_lower
            and len(user_lower.split()) <= 3
            or
            # Meta questions (always new)
            any(
                meta in user_lower
                for meta in [
                    "help",
                    "what can you do",
                    "how does this work",
                    "what are you",
                ]
            )
            # Preserve context for follow-up questions, clarifications, and product requests
        )

        # Enhanced context preservation: Don't clear if user is asking about "this" or "it"
        context_preservation_keywords = [
            "this",
            "it",
            "that",
            "the product",
            "the item",
            "the bed",
            "the sofa",
            "the chair",
            "the table",
            "the rug",
            "the light",
            "the curtain",
            "details",
            "warranty",
            "price",
            "color",
            "size",
            "material",
            "specifications",
        ]

        if any(word in user_lower for word in context_preservation_keywords):
            is_new_conversation = False
            print(
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Preserving context for follow-up question: {user_message}"
            )

        # Smart slot preservation: Keep important slots even for new conversations
        slots_to_use = session_state["slots"].copy() if not is_new_conversation else {}

        # Even for new conversations, preserve critical product context if it exists
        if is_new_conversation and session_state["slots"]:
            # Preserve product_name if it exists (most important for context)
            if "product_name" in session_state["slots"]:
                slots_to_use["product_name"] = session_state["slots"]["product_name"]
                print(
                    f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Preserving product_name: {session_state['slots']['product_name']}"
                )

            # Preserve brand if it exists
            if "brand" in session_state["slots"]:
                slots_to_use["brand"] = session_state["slots"]["brand"]
                print(
                    f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Preserving brand: {session_state['slots']['brand']}"
                )

        if is_new_conversation and session_state["slots"]:
            print(
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Clearing old slots for new conversation: {session_state['slots']}"
            )

        # Create new state with smart session persistence
        state: AgentState = {
            "user_message": user_message,
            "retriever": retriever,
            "llm": llm,
            "conversation_history": session_state["conversation_history"],
            "slots": slots_to_use,
            "last_prompted_slot": session_state["last_prompted_slot"],
            "slot_prompt_turn": session_state["slot_prompt_turn"],
            "pending_intent": session_state["pending_intent"],
            "corrections": session_state["corrections"].copy(),
            "conversation_summary": session_state["conversation_summary"],
            "session_id": session_id,  # Add session_id for analytics tracking
            # Remove hardcoded mappings - let the system use dynamic mappings from retriever
        }

        # Run the graph with strict routing
        result = app.invoke(state)

        # Update session state with new information
        GLOBAL_SESSION_STATES[session_id].update(
            {
                "conversation_history": result.get(
                    "conversation_history", session_state["conversation_history"]
                ),
                "slots": result.get("slots", session_state["slots"]),
                "last_prompted_slot": result.get("last_prompted_slot"),
                "slot_prompt_turn": result.get("slot_prompt_turn"),
                "pending_intent": result.get("pending_intent"),
                "corrections": result.get("corrections", session_state["corrections"]),
                "conversation_summary": result.get("conversation_summary"),
            }
        )

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


# Global session storage for conversation state
GLOBAL_SESSION_STATES = {}

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
