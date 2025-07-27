# Conversation History and Slot Persistence Fix

## Problem Identified

The AI agent was losing conversation context and slot information across multiple user messages. Specifically:

1. **Budget information was being lost** when users mentioned budget in one message and product type in subsequent messages
2. **Short messages like "furnitures"** were being treated as new conversations, clearing all slots
3. **NER extraction wasn't working properly** for budget extraction
4. **New conversation detection was too aggressive** and clearing slots unnecessarily

## Root Causes

1. **Aggressive New Conversation Detection**: The condition `len(user_lower.split()) <= 2` was clearing slots for any short message
2. **Poor Budget Extraction**: NER model wasn't reliably extracting budget information from text
3. **No Conversation History Check**: The system wasn't looking back at conversation history for previously mentioned information
4. **Incomplete Slot Persistence**: Slots were being cleared when they shouldn't have been

## Solutions Implemented

### 1. Enhanced Budget Extraction

Added a robust regex-based budget extraction function:

```python
def extract_budget_from_text(text: str) -> Optional[str]:
    """Extract budget information from text using regex patterns"""
    patterns = [
        r'budget\s+of\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees?|₹)?',
        r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees?|₹)\s+budget',
        r'under\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees?|₹)?',
        r'less\s+than\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees?|₹)?',
        r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees?|₹)',
        r'budget\s+(\d+(?:,\d+)*(?:\.\d+)?)',
    ]
```

### 2. Improved New Conversation Detection

Removed the aggressive short message detection:

```python
# Before (problematic):
is_new_conversation = (
    any(greeting in user_lower for greeting in greeting_keywords)
    or "categories" in user_lower
    or any(meta in user_lower for meta in ["help", "what can you do", "how does this work"])
    or len(user_lower.split()) <= 2  # This was the problem!
)

# After (fixed):
is_new_conversation = (
    any(greeting in user_lower for greeting in greeting_keywords)
    or "categories" in user_lower
    or any(meta in user_lower for meta in ["help", "what can you do", "how does this work"])
    # Removed the aggressive short message detection
)
```

### 3. Conversation History Check for Budget

Added logic to check conversation history for budget information:

```python
# Check conversation history for budget information
conversation_history = state.get("conversation_history", [])
for turn in reversed(conversation_history[-5:]):  # Check last 5 turns
    if turn.get("role") == "user":
        history_budget = extract_budget_from_text(turn.get("message", ""))
        if history_budget:
            state.setdefault("slots", {})["budget"] = history_budget
            print(f"Budget found in conversation history: budget = {history_budget}")
            break
```

### 4. Enhanced Product Type Detection

Added pattern matching for common product types in short messages:

```python
# Handle short responses that might be slot values
if len(user_lower.split()) <= 2 and not state.get("last_prompted_slot"):
    product_keywords = [
        "furniture", "furnishings", "sofa", "chair", "table", "bed", "desk",
        "wardrobe", "cabinet", "curtain", "drape", "blind", "shade", "rug",
        "carpet", "mat", "light", "lamp", "chandelier", "bath", "shower",
        "toilet", "sink", "kitchen", "dining", "bedroom", "living", "study", "office"
    ]

    if any(keyword in user_lower for keyword in product_keywords):
        # This is likely a product type, treat as PRODUCT_SEARCH
        state["intent"] = IntentType.PRODUCT_SEARCH
        state["intent_confidence"] = 0.8
```

### 5. Improved Slot Filling Logic

Enhanced the slot filling to handle pending intents better:

```python
# Check if we now have all required slots for the pending intent
if "pending_intent" in state:
    pending_intent = state["pending_intent"]
    required_slots = get_required_slots_for_intent(pending_intent)
    current_slots = state.get("slots", {})

    # If all required slots are filled, restore the pending intent
    if all(slot in current_slots for slot in required_slots):
        state["intent"] = state.pop("pending_intent")
        print(f"Restored pending intent: {state['intent']}")
    else:
        # Still missing slots, continue with clarification
        missing_slots = [s for s in required_slots if s not in current_slots]
        if missing_slots:
            return prompt_for_slot(state, missing_slots[0])
```

### 6. Enhanced LLM Prompt for Budget Handling

Improved the LLM prompt to better handle budget constraints:

```python
IMPORTANT BUDGET HANDLING:
- If the user has specified a budget, ONLY suggest products within that budget
- If no products are available within the budget, use budget_constraint response type
- Always consider the budget when making suggestions
```

## Testing

Created test scripts to verify the fixes:

1. **`test_budget_extraction.py`**: Tests budget extraction patterns
2. **`test_conversation_flow.py`**: Tests the complete conversation flow
3. **Debug endpoint**: `/debug/session/{session_id}` to inspect session state

## Expected Behavior Now

1. **"i have a budget of 10000 rs"** → Extracts budget: 10000, stores in slots
2. **"furnitures"** → Recognizes as product type, preserves budget from previous message
3. **"i want some furnitures"** → Uses both budget and product type for search

The conversation history and slot persistence should now work correctly, maintaining context across multiple user messages.
