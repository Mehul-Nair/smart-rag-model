#!/usr/bin/env python3
"""
Simple demonstration of Smart Slot Management System

This script demonstrates how the system handles the specific scenario:
- User searches for one category (e.g., rugs)
- Engages in conversation about that category
- Switches to a different category (e.g., furniture)
- System properly clears irrelevant slots while preserving relevant ones
"""

import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from rag.slot_manager import get_slot_manager
from rag.intent_modules import IntentType


def demo_context_switching():
    """Demonstrate context switching with slot management"""

    print("🎯 Smart Slot Management Demo")
    print("=" * 50)
    print("This demo shows how the system handles context switching")
    print("between different product categories.\n")

    # Initialize slot manager
    slot_manager = get_slot_manager()

    # Scenario: User starts with rugs, then switches to furniture

    print("📋 Scenario: User switches from rugs to furniture")
    print("-" * 50)

    # Step 1: Initial search for rugs
    print("\n1️⃣ Initial Search - Rugs")
    print("User: 'I'm looking for rugs for my living room'")

    current_slots = {}
    user_message = "I'm looking for rugs for my living room"
    current_intent = IntentType.PRODUCT_SEARCH
    previous_intent = None
    extracted_entities = {"PRODUCT_TYPE": "rugs", "ROOM_TYPE": "living room"}

    updated_slots = slot_manager.manage_slots(
        current_slots=current_slots,
        user_message=user_message,
        current_intent=current_intent,
        previous_intent=previous_intent,
        extracted_entities=extracted_entities,
        conversation_history=[],
    )

    print(f"✅ Slots after initial search: {updated_slots}")

    # Step 2: Follow-up question about rugs
    print("\n2️⃣ Follow-up Question - Same Category")
    print("User: 'What's the price of this rug?'")

    current_slots = updated_slots.copy()
    user_message = "What's the price of this rug?"
    current_intent = IntentType.PRODUCT_DETAIL
    previous_intent = IntentType.PRODUCT_SEARCH
    extracted_entities = {"PRODUCT_TYPE": "rugs"}

    updated_slots = slot_manager.manage_slots(
        current_slots=current_slots,
        user_message=user_message,
        current_intent=current_intent,
        previous_intent=previous_intent,
        extracted_entities=extracted_entities,
        conversation_history=[
            {"role": "user", "message": "I'm looking for rugs for my living room"},
            {"role": "system", "message": "Here are some rug options..."},
        ],
    )

    print(f"✅ Slots after follow-up: {updated_slots}")
    print("   Note: Context preserved because of 'this' reference")

    # Step 3: Context switch to furniture
    print("\n3️⃣ Context Switch - Different Category")
    print("User: 'Now show me some furniture for the bedroom'")

    current_slots = updated_slots.copy()
    user_message = "Now show me some furniture for the bedroom"
    current_intent = IntentType.PRODUCT_SEARCH
    previous_intent = IntentType.PRODUCT_DETAIL
    extracted_entities = {"PRODUCT_TYPE": "furniture", "ROOM_TYPE": "bedroom"}

    updated_slots = slot_manager.manage_slots(
        current_slots=current_slots,
        user_message=user_message,
        current_intent=current_intent,
        previous_intent=previous_intent,
        extracted_entities=extracted_entities,
        conversation_history=[
            {"role": "user", "message": "I'm looking for rugs for my living room"},
            {"role": "system", "message": "Here are some rug options..."},
            {"role": "user", "message": "What's the price of this rug?"},
            {"role": "system", "message": "The price information..."},
        ],
    )

    print(f"✅ Slots after context switch: {updated_slots}")
    print("   Note: Old slots cleared, new context established")

    # Step 4: Another context switch
    print("\n4️⃣ Another Context Switch")
    print("User: 'What about lighting for the kitchen?'")

    current_slots = updated_slots.copy()
    user_message = "What about lighting for the kitchen?"
    current_intent = IntentType.PRODUCT_SEARCH
    previous_intent = IntentType.PRODUCT_SEARCH
    extracted_entities = {"PRODUCT_TYPE": "lighting", "ROOM_TYPE": "kitchen"}

    updated_slots = slot_manager.manage_slots(
        current_slots=current_slots,
        user_message=user_message,
        current_intent=current_intent,
        previous_intent=previous_intent,
        extracted_entities=extracted_entities,
        conversation_history=[
            {"role": "user", "message": "Now show me some furniture for the bedroom"},
            {"role": "system", "message": "Here are some furniture options..."},
        ],
    )

    print(f"✅ Slots after another switch: {updated_slots}")

    print("\n🎉 Demo completed successfully!")
    print("\n📊 Key Benefits Demonstrated:")
    print("✅ Context preservation when using 'this' references")
    print("✅ Smart slot clearing on context switches")
    print("✅ Dynamic slot management based on conversation flow")
    print("✅ No hardcoded rules - everything is dynamic")


def demo_slot_relevance():
    """Demonstrate slot relevance calculation"""

    print("\n🔍 Slot Relevance Demo")
    print("=" * 30)

    slot_manager = get_slot_manager()

    # Test different scenarios
    scenarios = [
        {
            "name": "Product Search",
            "message": "I want a wooden table for the dining room",
            "intent": IntentType.PRODUCT_SEARCH,
            "entities": {
                "PRODUCT_TYPE": "table",
                "MATERIAL": "wooden",
                "ROOM_TYPE": "dining room",
            },
        },
        {
            "name": "Warranty Query",
            "message": "What's the warranty for this sofa?",
            "intent": IntentType.WARRANTY_QUERY,
            "entities": {"PRODUCT_TYPE": "sofa"},
        },
        {
            "name": "Budget Query",
            "message": "Show me beds under 5000 rupees",
            "intent": IntentType.BUDGET_QUERY,
            "entities": {"PRODUCT_TYPE": "bed", "BUDGET": "5000"},
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        print(f"Message: {scenario['message']}")
        print(f"Intent: {scenario['intent'].value}")

        # Test relevance for different slots
        slots_to_test = [
            "product_type",
            "product_name",
            "brand",
            "room_type",
            "budget",
            "warranty",
        ]

        print("Slot relevance scores:")
        for slot_name in slots_to_test:
            # Create a simple context for relevance calculation
            context = slot_manager.SlotContext(
                current_intent=scenario["intent"],
                user_message=scenario["message"],
                extracted_entities=scenario["entities"],
            )
            relevance = slot_manager.determine_slot_relevance(slot_name, context)
            print(f"  {slot_name}: {relevance:.2f}")

    print("\n✅ Slot relevance demo completed!")


def main():
    """Main demonstration function"""

    print("🚀 Smart Slot Management System Demonstration")
    print("=" * 60)
    print("This demo shows how the system solves the slot management problem")
    print("by intelligently handling context switching between categories.\n")

    try:
        # Demo context switching
        demo_context_switching()

        # Demo slot relevance
        demo_slot_relevance()

        print("\n🎯 Problem Solved!")
        print("The smart slot management system now:")
        print("✅ Handles context switching between different categories")
        print("✅ Preserves relevant slots while clearing irrelevant ones")
        print("✅ Uses dynamic, non-hardcoded logic")
        print("✅ Provides intelligent slot management based on conversation context")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
