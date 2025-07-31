# Smart Slot Management System

## Overview

The Smart Slot Management System is a dynamic, context-aware slot management solution that replaces the hardcoded slot management with intelligent, adaptive slot handling. It understands context switching between different categories and manages slot relevance dynamically.

## Key Features

### 🧠 Dynamic Context Understanding

- **Context Switch Detection**: Automatically detects when users switch between different product categories
- **Continuity Preservation**: Maintains relevant context when users ask follow-up questions
- **Semantic Relevance**: Determines slot relevance based on current intent and conversation context

### 🔄 Intelligent Slot Management

- **Smart Slot Clearing**: Only clears slots that are no longer relevant to the current context
- **Priority-Based Preservation**: Preserves high-priority slots (like product_name) across context switches
- **Dynamic Slot Relationships**: Understands relationships between different slot types

### 🎯 Intent-Aware Processing

- **Intent-Specific Slot Requirements**: Different intents require different slots
- **Dynamic Slot Validation**: Validates slots based on current intent requirements
- **Context-Sensitive Slot Handling**: Adjusts slot behavior based on conversation context

## Architecture

### Core Components

#### 1. Slot Manager (`slot_manager.py`)

The core engine that handles:

- Slot registry with relationships and semantic information
- Context switch detection
- Slot relevance calculation
- Dynamic slot clearing and preservation

#### 2. Smart Slot Integration (`smart_slot_integration.py`)

Integration layer that:

- Connects the slot manager with the LangGraph agent
- Provides dynamic slot prompts
- Handles slot corrections
- Tracks slot analytics

#### 3. Slot Categories

Slots are organized into categories for better management:

- **PRODUCT**: `product_type`, `product_name`, `brand`
- **ROOM**: `room_type`, `room_size`
- **STYLE**: `color`, `material`, `style`, `pattern`
- **BUDGET**: `budget`, `price_range`
- **QUANTITY**: `quantity`, `count`
- **WARRANTY**: `warranty_period`, `warranty_type`
- **META**: `session_id`, `user_id`

## How It Works

### 1. Context Switch Detection

The system detects context switches using multiple indicators:

```python
# Context switch indicators
context_switch_indicators = [
    r"show me (?:some|different|other) (?:products|items|things)",
    r"what about (?:.*?) (?:products|items)",
    r"instead of (?:.*?) (?:show|find) (?:.*?)",
    r"change to (?:.*?)",
    r"switch to (?:.*?)",
    r"now (?:show|find) (?:.*?)",
    r"how about (?:.*?)",
]

# Continuity indicators (preserve context)
continuity_indicators = [
    r"this (?:product|item|one)",
    r"that (?:product|item|one)",
    r"it",
    r"the (?:product|item)",
    r"details (?:about|of)",
    r"more (?:about|details)",
]
```

### 2. Slot Relevance Calculation

Each slot has a relevance score calculated based on:

- **Intent Requirements**: Is the slot required for the current intent?
- **Semantic Relevance**: Does the user message contain relevant keywords?
- **Related Slot Presence**: Are related slots already filled?
- **Priority Level**: Higher priority slots are preserved longer
- **Context Sensitivity**: Some slots are cleared on context switches

### 3. Dynamic Slot Management

```python
def manage_slots(self, current_slots, user_message, current_intent,
                previous_intent, extracted_entities, conversation_history):
    # 1. Detect context switch
    context_switch_detected, continuity_score = self.detect_context_switch(
        user_message, current_intent, previous_intent
    )

    # 2. Determine which slots to clear
    slots_to_clear = self.get_slots_to_clear(current_slots, context)

    # 3. Determine which slots to preserve
    slots_to_preserve = self.get_slots_to_preserve(current_slots, context)

    # 4. Add new extracted entities
    # 5. Return updated slots
```

## Usage Examples

### Example 1: Context Switching

**User**: "I'm looking for a bed for my bedroom"

- **Slots**: `{"product_type": "bed", "room_type": "bedroom"}`

**User**: "What's the warranty for this bed?"

- **Slots**: `{"product_type": "bed", "room_type": "bedroom"}` (preserved)

**User**: "Now show me some curtains for the living room"

- **Slots**: `{"product_type": "curtains", "room_type": "living room"}` (context switch detected, old slots cleared)

### Example 2: Continuity Preservation

**User**: "Show me wooden tables"

- **Slots**: `{"product_type": "table", "material": "wooden"}`

**User**: "What's the price of this table?"

- **Slots**: `{"product_type": "table", "material": "wooden"}` (preserved due to "this" reference)

### Example 3: New Conversation

**User**: "Hello, I need help with lighting"

- **Slots**: `{"product_type": "lighting"}` (previous context cleared for new conversation)

## Configuration

### Slot Registry Configuration

Each slot in the registry can be configured with:

```python
SlotInfo(
    name="product_type",
    category=SlotCategory.PRODUCT,
    required_for_intents={IntentType.PRODUCT_SEARCH, IntentType.PRODUCT_DETAIL},
    related_slots={"product_name", "brand", "category"},
    semantic_keywords={"furniture", "sofa", "chair", "table", "bed"},
    priority=3,  # Higher priority = preserved longer
    context_sensitive=True  # Cleared on context switch
)
```

### Dynamic Updates

The system can dynamically update slot information:

```python
# Update brand keywords from retriever
if hasattr(retriever, 'get_brand_names'):
    brand_names = retriever.get_brand_names()
    slot_registry["brand"].semantic_keywords.update(brand_names)

# Update product type mappings
if hasattr(retriever, 'get_product_type_mappings'):
    product_mappings = retriever.get_product_type_mappings()
    slot_registry["product_type"].semantic_keywords.update(product_mappings.keys())
```

## Integration with LangGraph Agent

### 1. Classification Node Integration

```python
def classify_node(state: AgentState) -> AgentState:
    # Initialize smart slot integration
    slot_integration = get_smart_slot_integration()

    # Process slots using smart slot management
    state = slot_integration.process_slots_for_classification(state)

    # Validate slots for the current intent
    state = slot_integration.validate_slots_for_intent(state)

    return state
```

### 2. Slot Processor Node Integration

```python
def slot_processor_node(state: AgentState) -> AgentState:
    slot_integration = get_smart_slot_integration()

    # Process slots using smart slot management
    state = slot_integration.process_slots_for_classification(state)

    # Validate slots for the current intent
    state = slot_integration.validate_slots_for_intent(state)

    # Check for missing slots
    missing_slots = state.get("missing_slots", [])
    if missing_slots:
        return prompt_for_slot(state, missing_slots[0])

    return state
```

## Benefits

### 1. **Improved User Experience**

- Maintains context appropriately across conversation turns
- Clears irrelevant information when switching topics
- Provides more natural conversation flow

### 2. **Dynamic and Flexible**

- No hardcoded rules or patterns
- Adapts to different conversation styles
- Learns from available data sources

### 3. **Robust Context Management**

- Handles complex multi-turn conversations
- Preserves important information across context switches
- Clears irrelevant information intelligently

### 4. **Analytics and Insights**

- Tracks slot usage patterns
- Provides insights into conversation flow
- Helps optimize slot management strategies

## Testing

Run the test script to verify the system:

```bash
python test_smart_slot_management.py
```

The test script covers:

- Basic slot management scenarios
- Context switch detection
- Slot relevance calculation
- Continuity preservation
- New conversation handling

## Troubleshooting

### Common Issues

1. **Slots not clearing on context switch**

   - Check if context switch detection is working
   - Verify slot context_sensitive settings
   - Review continuity indicators

2. **Slots clearing when they shouldn't**

   - Check slot priority settings
   - Verify continuity detection patterns
   - Review slot relationships

3. **Missing slot validation**
   - Ensure intent requirements are properly configured
   - Check slot registry initialization
   - Verify integration with intent classifier

### Debug Mode

Enable detailed logging by setting environment variables:

```bash
export SLOT_MANAGER_DEBUG=1
export SMART_SLOT_DEBUG=1
```

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**

   - Learn slot relevance patterns from conversation data
   - Adaptive context switch detection
   - Personalized slot management

2. **Advanced Analytics**

   - Slot usage heatmaps
   - Conversation flow analysis
   - Performance optimization recommendations

3. **Multi-Modal Support**

   - Image-based slot extraction
   - Voice-based slot management
   - Gesture-based context switching

4. **Real-time Adaptation**
   - Dynamic slot registry updates
   - Real-time context learning
   - Adaptive slot relationships

## Conclusion

The Smart Slot Management System provides a robust, dynamic solution for managing conversation context in AI agents. It replaces hardcoded slot management with intelligent, context-aware slot handling that adapts to user behavior and conversation patterns.

The system is designed to be:

- **Flexible**: Easy to configure and extend
- **Robust**: Handles edge cases gracefully
- **Scalable**: Can handle complex conversation scenarios
- **Maintainable**: Clear separation of concerns and modular design

This system significantly improves the user experience by maintaining appropriate context while clearing irrelevant information, leading to more natural and effective conversations.
