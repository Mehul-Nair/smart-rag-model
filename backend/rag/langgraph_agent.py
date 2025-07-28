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

# Global intent classifier instance - using hybrid approach with trained model + rule-based fallback
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
trained_model_path = os.path.join(current_dir, "..", "trained_deberta_model")

intent_classifier = IntentClassifierFactory.create(
    "huggingface",
    {
        "model_path": trained_model_path,
        "device": "cpu",
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
    """Switch to improved hybrid implementation"""
    global intent_classifier
    intent_classifier = IntentClassifierFactory.create(
        "improved_hybrid",
        {
            "confidence_threshold": 0.3,
            "primary_classifier": "huggingface",
            "fallback_classifier": "rule_based",
            "enable_intent_specific_rules": True,
            "implementation_configs": {
                "huggingface": {
                    "model_path": trained_model_path,
                    "device": "cpu",
                },
                "rule_based": {"similarity_threshold": 0.3},
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
    slot_prompts = {
        "product_type": "What specific type of product are you looking for? (e.g., shower curtains, bath mats, storage solutions, lighting, furniture)",
        "room_type": "Which room or space are you looking to decorate? (e.g., bathroom, bedroom, living room, kitchen, dining room, balcony, garden, office, study, hall, patio, terrace)",
        "budget": "What's your budget range? (e.g., under 1000, 1000-5000, above 10000)",
        "brand": "Do you have a preferred brand? (e.g., IKEA, Home Depot, local brands)",
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

    # Use hybrid intent classifier with error handling
    try:
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
        else:
            intent = classification_result.intent
            confidence = classification_result.confidence
            method = classification_result.method
            reasoning = classification_result.reasoning
            scores = classification_result.scores
            processing_time = classification_result.processing_time
    except Exception as e:
        print(f"[{timestamp}] ❌ Intent classifier failed with error: {e}")
        # Fallback to CLARIFY intent
        intent = IntentType.CLARIFY
        confidence = 0.1
        method = "error_fallback"
        reasoning = f"Intent classifier error: {e}"
        scores = {}
        processing_time = 0.0

    # Trust the intent classifier completely - no overrides
    print(f"[{timestamp}] Intent classifier result - trusting completely")
    print(
        f"[{timestamp}] Intent: {intent}, Confidence: {confidence:.3f}, Method: {method}"
    )

    # Ensure intent is properly set in state
    state["intent"] = intent
    state["intent_confidence"] = confidence
    state["intent_scores"] = scores

    # Log the final intent decision
    print(f"[{timestamp}] FINAL INTENT DECISION: {intent.value}")

    # For debugging: log what node this intent should route to
    if intent == IntentType.CATEGORY_LIST:
        print(f"[{timestamp}] DEBUG: CATEGORY_LIST intent should route to meta_node")
    elif intent == IntentType.PRODUCT_SEARCH:
        print(
            f"[{timestamp}] DEBUG: PRODUCT_SEARCH intent should route to retrieve_node"
        )
    elif intent == IntentType.GREETING:
        print(f"[{timestamp}] DEBUG: GREETING intent should route to meta_node")
    elif intent == IntentType.HELP:
        print(f"[{timestamp}] DEBUG: HELP intent should route to meta_node")

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
            # First, try Dynamic NER-based extraction
            from rag.intent_modules.dynamic_ner_classifier import (
                extract_slots_from_text_dynamic,
            )

            extracted_slots = extract_slots_from_text_dynamic(user_message)
            if extracted_slots:
                print(f"[{timestamp}] NER extracted slots: {extracted_slots}")
                # Map NER entity types to slot names
                slot_mapping = {
                    "PRODUCT_TYPE": "product_type",
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

            # Enhanced budget extraction using regex patterns
            if "budget" not in state.get("slots", {}):
                budget_value = extract_budget_from_text(user_message)
                if budget_value:
                    state.setdefault("slots", {})["budget"] = budget_value
                    print(
                        f"[{timestamp}] Enhanced budget extraction: budget = {budget_value}"
                    )
                else:
                    # Check conversation history for budget information
                    conversation_history = state.get("conversation_history", [])
                    for turn in reversed(
                        conversation_history[-5:]
                    ):  # Check last 5 turns
                        if turn.get("role") == "user":
                            history_budget = extract_budget_from_text(
                                turn.get("message", "")
                            )
                            if history_budget:
                                state.setdefault("slots", {})["budget"] = history_budget
                                print(
                                    f"[{timestamp}] Budget found in conversation history: budget = {history_budget}"
                                )
                                break

            # Enhanced product type extraction for common patterns
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
    """Check if the message is clearly outside the home decor domain"""
    user_lower = user_message.lower().strip()

    # Debug: Print the message being checked
    print(f"🔍 Checking domain for: '{user_message}'")

    # Clear out-of-domain patterns (comprehensive)
    out_of_domain_patterns = [
        # Math/calculations
        r"\d+\s*[+\-*/]\s*\d+",  # Mathematical operations
        r"calculate\s+",  # Calculate requests
        r"solve\s+",  # Solve requests
        r"what\s+is\s+\d+\s*[+\-*/]\s*\d+",  # Math questions
        # General knowledge questions (comprehensive)
        r"who\s+is\s+",  # Any "who is" question (person, politician, celebrity, etc.)
        r"what\s+is\s+",  # General "what is" questions
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
    if len(user_lower.split()) <= 1 and not any(
        word in user_lower
        for word in [
            "help",
            "hi",
            "hello",
            "hey",
            "furniture",
            "curtain",
            "rug",
            "light",
            "bath",
        ]
    ):
        print(f"❌ Very short unclear message: '{user_message}'")
        return True

    print(f"✅ Message is in-domain")
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
        # Simplified prompt to avoid syntax issues
        summary_text = f"Conversation summary: {summary}\n" if summary else ""
        history_text = (
            f"Recent conversation:\n{recent_history}\n" if recent_history else ""
        )

        full_prompt = f"""You are a helpful home decor assistant. Provide product suggestions based on the user's requirements.

Available response types:
1. product_suggestion: When you have relevant products to suggest
2. budget_constraint: When the user's budget is too low for available products
3. category_not_found: When the requested category doesn't exist
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

{summary_text}{history_text}User is looking for: {slot_str}.

IMPORTANT BUDGET HANDLING:
- If the user has specified a budget, ONLY suggest products within that budget
- If no products are available within the budget, use budget_constraint response type
- Always consider the budget when making suggestions

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
IMPORTANT: Reply ONLY in valid JSON as per the above schema. Do not reply in free text."""
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
            IntentType.WARRANTY_QUERY,
            IntentType.PRODUCT,
        ]:
            print(f"[ROUTING] 🎯 Routing {intent.value} to retrieve_node")
            return "retrieve"
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

        # More intelligent new conversation detection - less aggressive
        is_new_conversation = (
            # Greeting or help request
            any(greeting in user_lower for greeting in greeting_keywords)
            or
            # Category listing questions (always new)
            "categories" in user_lower
            or
            # Meta questions (always new)
            any(
                meta in user_lower
                for meta in ["help", "what can you do", "how does this work"]
            )
            # Removed the aggressive short message detection that was clearing slots
        )

        # Preserve slots unless it's clearly a new conversation
        slots_to_use = session_state["slots"].copy() if not is_new_conversation else {}

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
