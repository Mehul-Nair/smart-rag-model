"""
Smart Slot Management Integration

This module integrates the smart slot management system with the existing LangGraph agent,
replacing the hardcoded slot management with dynamic, context-aware slot handling.
"""

import time
import datetime
from typing import Dict, List, Optional, Any
from .slot_manager import get_slot_manager, initialize_slot_manager
from .intent_modules import IntentType


class SmartSlotIntegration:
    """
    Integration layer between the smart slot manager and the LangGraph agent
    """

    def __init__(self):
        self.slot_manager = get_slot_manager()
        self.session_slot_history = {}  # Track slot changes per session

    def _correct_room_type_spelling(
        self, extracted_entities: Dict[str, str], user_message: str
    ) -> Dict[str, str]:
        """Correct common misspellings of room types using pattern matching"""
        import re

        # Dynamic room type patterns - can be extended without hardcoding
        room_patterns = {
            r"\bkitched\b": "kitchen",
            r"\bkitchen\b": "kitchen",
            r"\bbedroom\b": "bedroom",
            r"\bliving\s+room\b": "living room",
            r"\bdining\s+room\b": "dining room",
            r"\bbathroom\b": "bathroom",
            r"\boffice\b": "office",
            r"\bstudy\b": "study",
            r"\bbalcony\b": "balcony",
            r"\bgarden\b": "garden",
            r"\bhall\b": "hall",
            r"\bpatio\b": "patio",
            r"\bterrace\b": "terrace",
            r"\bgarage\b": "garage",
            r"\bcloset\b": "closet",
            r"\bwardrobe\b": "wardrobe",
            r"\bden\b": "den",
            r"\blibrary\b": "library",
            r"\bplayroom\b": "playroom",
            r"\bguest\s+room\b": "guest room",
            r"\bhome\s+office\b": "home office",
            r"\bworkout\s+room\b": "workout room",
            r"\bgame\s+room\b": "game room",
            r"\bmedia\s+room\b": "media room",
            r"\bconservatory\b": "conservatory",
            r"\bsunroom\b": "sunroom",
            r"\bporch\b": "porch",
            r"\bdeck\b": "deck",
            r"\bveranda\b": "veranda",
        }

        corrected_entities = extracted_entities.copy()

        # Check each entity value for room type patterns
        for entity_type, entity_value in extracted_entities.items():
            if entity_type == "ROOM" or "room" in entity_type.lower():
                # Apply pattern-based corrections
                corrected_value = entity_value
                for pattern, correction in room_patterns.items():
                    if re.search(pattern, entity_value.lower()):
                        corrected_value = correction
                        break

                if corrected_value != entity_value:
                    corrected_entities[entity_type] = corrected_value
                    print(
                        f"Applied spelling correction: '{entity_value}' -> '{corrected_value}'"
                    )

        return corrected_entities

    def process_slots_for_classification(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process slots during the classification phase
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_message = state.get("user_message", "")
        current_intent = state.get("intent")
        previous_intent = state.get("previous_intent")
        current_slots = state.get("slots", {})
        session_id = state.get("session_id", "default")

        print(
            f"[{timestamp}] [SMART_SLOT_INTEGRATION] Processing slots for classification..."
        )

        # Get extracted entities from NER (if available)
        extracted_entities = {}
        try:
            from .intent_modules.dynamic_ner_classifier import (
                extract_slots_from_text_dynamic,
            )

            extracted_entities = extract_slots_from_text_dynamic(user_message)

            # Apply spelling correction for room types
            corrected_entities = self._correct_room_type_spelling(
                extracted_entities, user_message
            )
            if corrected_entities != extracted_entities:
                print(
                    f"[{timestamp}] [SMART_SLOT_INTEGRATION] Applied spelling correction: {extracted_entities} -> {corrected_entities}"
                )
                extracted_entities = corrected_entities

            print(
                f"[{timestamp}] [SMART_SLOT_INTEGRATION] Extracted entities: {extracted_entities}"
            )
        except Exception as e:
            print(f"[{timestamp}] [SMART_SLOT_INTEGRATION] NER extraction failed: {e}")

        # Use smart slot manager to update slots
        updated_slots = self.slot_manager.manage_slots(
            current_slots=current_slots,
            user_message=user_message,
            current_intent=current_intent,
            previous_intent=previous_intent,
            extracted_entities=extracted_entities,
            conversation_history=state.get("conversation_history", []),
        )

        # Update state with new slots
        state["slots"] = updated_slots

        # Track slot changes for this session
        if session_id not in self.session_slot_history:
            self.session_slot_history[session_id] = []

        self.session_slot_history[session_id].append(
            {
                "timestamp": timestamp,
                "user_message": user_message,
                "intent": current_intent.value if current_intent else None,
                "slots_before": current_slots,
                "slots_after": updated_slots,
                "extracted_entities": extracted_entities,
            }
        )

        print(f"[{timestamp}] [SMART_SLOT_INTEGRATION] Updated slots: {updated_slots}")
        return state

    def validate_slots_for_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate if all required slots for the current intent are filled
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        intent = state.get("intent")
        slots = state.get("slots", {})

        if not intent:
            return state

        print(
            f"[{timestamp}] [SMART_SLOT_INTEGRATION] Validating slots for intent: {intent}"
        )

        all_required_filled, missing_slots = (
            self.slot_manager.validate_slots_for_intent(slots, intent)
        )

        if not all_required_filled:
            print(
                f"[{timestamp}] [SMART_SLOT_INTEGRATION] Missing required slots: {missing_slots}"
            )
            state["required_slots"] = missing_slots
            state["missing_slots"] = missing_slots
        else:
            print(f"[{timestamp}] [SMART_SLOT_INTEGRATION] All required slots filled")
            state["required_slots"] = []
            state["missing_slots"] = []

        return state

    def get_slot_prompt_message(self, slot_name: str, state: Dict[str, Any]) -> str:
        """
        Generate dynamic slot prompt messages based on context
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get dynamic suggestions from the system
        suggestions = self._get_dynamic_suggestions_for_slot(slot_name, state)

        slot_prompts = {
            "product_type": f"What specific type of product are you looking for? (e.g., {suggestions})",
            "room_type": "Which room or space are you looking to decorate? (e.g., bathroom, bedroom, living room, kitchen, dining room, balcony, garden, office, study, hall, patio, terrace)",
            "budget": "What's your budget range? (e.g., under 1000, 1000-5000, above 10000)",
            "brand": f"Do you have a preferred brand? (e.g., {suggestions})",
            "color": "What color scheme are you looking for? (e.g., white, blue, neutral, colorful)",
            "material": "Any specific material preference? (e.g., wood, metal, plastic, fabric)",
            "style": "What style are you going for? (e.g., modern, traditional, minimalist, rustic)",
            "product_name": "Which specific product are you interested in?",
            "warranty": "What warranty information do you need?",
        }

        clarification_msg = slot_prompts.get(
            slot_name,
            f"What type of {slot_name.replace('_', ' ')} are you looking for?",
        )

        print(
            f"[{timestamp}] [SMART_SLOT_INTEGRATION] Generated prompt for slot '{slot_name}': {clarification_msg}"
        )
        return clarification_msg

    def _get_dynamic_suggestions_for_slot(
        self, slot_name: str, state: Dict[str, Any]
    ) -> str:
        """
        Get dynamic suggestions for a slot based on available data
        """
        retriever = state.get("retriever")

        if slot_name == "product_type":
            # Get dynamic product type suggestions from retriever
            if retriever and hasattr(retriever, "get_product_type_mappings"):
                try:
                    mappings = retriever.get_product_type_mappings()
                    if mappings:
                        unique_categories = set(mappings.values())
                        return ", ".join(
                            list(unique_categories)[:8]
                        )  # Limit to 8 suggestions
                except Exception as e:
                    print(f"Error getting product type mappings: {e}")
            return "furniture, lights, bath, rugs, furnishing"

        elif slot_name == "brand":
            # Get dynamic brand suggestions
            if retriever and hasattr(retriever, "get_brand_names"):
                try:
                    brand_names = retriever.get_brand_names()
                    if brand_names:
                        return ", ".join(
                            list(brand_names)[:5]
                        )  # Limit to 5 suggestions
                except Exception as e:
                    print(f"Error getting brand names: {e}")
            return "Asian Paints, Pure Royale, White Teak"

        elif slot_name == "color":
            return "white, blue, neutral, colorful, brown, gray, pink, purple"

        elif slot_name == "material":
            return "wood, metal, plastic, fabric, leather, cotton, silk, wool"

        else:
            return "various options"

    def handle_slot_correction(self, state: Dict[str, Any], user_message: str) -> bool:
        """
        Handle slot corrections using smart detection
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_lower = user_message.lower()

        # Look for correction indicators
        correction_indicators = [
            "actually",
            "correction",
            "i meant",
            "not that",
            "instead",
            "change",
            "update",
        ]

        if not any(indicator in user_lower for indicator in correction_indicators):
            return False

        current_slots = state.get("slots", {})
        corrected = False

        # Check each slot for potential corrections
        for slot_name in current_slots:
            if slot_name in user_lower:
                # Try to extract new value after slot name
                import re

                match = re.search(
                    rf"{slot_name}[^a-zA-Z0-9]*([\w\s]+)", user_message, re.IGNORECASE
                )
                if match:
                    new_value = match.group(1).strip().split()[0]
                    old_value = current_slots[slot_name]
                    current_slots[slot_name] = new_value

                    # Track correction
                    if "corrections" not in state:
                        state["corrections"] = []

                    state["corrections"].append(
                        {
                            "slot": slot_name,
                            "old": old_value,
                            "new": new_value,
                            "turn": len(state.get("conversation_history", [])),
                        }
                    )

                    print(
                        f"[{timestamp}] [SMART_SLOT_INTEGRATION] Corrected slot '{slot_name}': '{old_value}' -> '{new_value}'"
                    )
                    corrected = True

        if corrected:
            state["slots"] = current_slots

        return corrected

    def get_slot_analytics(self, session_id: str = "default") -> Dict[str, Any]:
        """
        Get analytics about slot usage for a session
        """
        if session_id not in self.session_slot_history:
            return {"error": "No session data found"}

        history = self.session_slot_history[session_id]

        analytics = {
            "total_interactions": len(history),
            "slot_changes": {},
            "intent_distribution": {},
            "most_active_slots": {},
            "corrections": [],
        }

        # Analyze slot changes
        for entry in history:
            intent = entry.get("intent")
            if intent:
                analytics["intent_distribution"][intent] = (
                    analytics["intent_distribution"].get(intent, 0) + 1
                )

            # Track slot changes
            before = entry.get("slots_before", {})
            after = entry.get("slots_after", {})

            for slot_name in set(before.keys()) | set(after.keys()):
                if slot_name not in analytics["slot_changes"]:
                    analytics["slot_changes"][slot_name] = {
                        "added": 0,
                        "removed": 0,
                        "changed": 0,
                    }

                if slot_name in after and slot_name not in before:
                    analytics["slot_changes"][slot_name]["added"] += 1
                elif slot_name in before and slot_name not in after:
                    analytics["slot_changes"][slot_name]["removed"] += 1
                elif (
                    slot_name in before
                    and slot_name in after
                    and before[slot_name] != after[slot_name]
                ):
                    analytics["slot_changes"][slot_name]["changed"] += 1

        # Find most active slots
        slot_activity = {}
        for entry in history:
            for slot_name in entry.get("slots_after", {}):
                slot_activity[slot_name] = slot_activity.get(slot_name, 0) + 1

        analytics["most_active_slots"] = dict(
            sorted(slot_activity.items(), key=lambda x: x[1], reverse=True)
        )

        return analytics


# Global integration instance
_smart_slot_integration = None


def get_smart_slot_integration() -> SmartSlotIntegration:
    """Get the global smart slot integration instance"""
    global _smart_slot_integration
    if _smart_slot_integration is None:
        _smart_slot_integration = SmartSlotIntegration()
    return _smart_slot_integration


def initialize_smart_slot_integration(retriever=None, llm=None):
    """Initialize the smart slot integration with dynamic updates"""
    # Initialize slot manager
    initialize_slot_manager(retriever, llm)

    # Get integration instance
    integration = get_smart_slot_integration()
    return integration
