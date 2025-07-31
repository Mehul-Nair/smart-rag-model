"""
Smart Dynamic Slot Management System

This module provides a robust, dynamic slot management system that:
1. Understands context switching between different categories
2. Dynamically determines slot relevance based on intent and content
3. Manages slot relationships and dependencies
4. Provides intelligent slot clearing and preservation
5. Uses semantic understanding rather than hardcoded rules
"""

import re
import time
import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json

# Import the modular intent classifier
from .intent_modules import IntentType, ClassificationResult


class SlotCategory(Enum):
    """Categories of slots for better organization and management"""

    PRODUCT = "product"  # product_type, product_name, brand
    ROOM = "room"  # room_type, room_size
    STYLE = "style"  # color, material, style, pattern
    BUDGET = "budget"  # budget, price_range
    QUANTITY = "quantity"  # quantity, count
    WARRANTY = "warranty"  # warranty_period, warranty_type
    META = "meta"  # session_id, user_id


@dataclass
class SlotInfo:
    """Information about a slot including its relationships and relevance"""

    name: str
    category: SlotCategory
    required_for_intents: Set[IntentType] = field(default_factory=set)
    related_slots: Set[str] = field(default_factory=set)
    semantic_keywords: Set[str] = field(default_factory=set)
    priority: int = 1  # Higher priority slots are preserved longer
    context_sensitive: bool = (
        True  # Whether this slot should be cleared on context switch
    )


@dataclass
class SlotContext:
    """Context information for slot management"""

    current_intent: Optional[IntentType] = None
    previous_intent: Optional[IntentType] = None
    user_message: str = ""
    conversation_history: List[Dict] = field(default_factory=list)
    extracted_entities: Dict[str, str] = field(default_factory=dict)
    category_switch_detected: bool = False
    context_continuity_score: float = 1.0


class SmartSlotManager:
    """
    Dynamic slot management system that understands context and relationships
    """

    def __init__(self):
        self.slot_registry: Dict[str, SlotInfo] = {}
        self.context_history: List[SlotContext] = []
        self._initialize_slot_registry()

    def _initialize_slot_registry(self):
        """Initialize the dynamic slot registry with relationships and semantic information"""

        # Product-related slots
        self.slot_registry["product_type"] = SlotInfo(
            name="product_type",
            category=SlotCategory.PRODUCT,
            required_for_intents={
                IntentType.PRODUCT_SEARCH,
                IntentType.PRODUCT_DETAIL,
                IntentType.BUDGET_QUERY,
                IntentType.WARRANTY_QUERY,
            },
            related_slots={"product_name", "brand", "category"},
            semantic_keywords={
                "furniture",
                "furnishings",
                "sofa",
                "chair",
                "table",
                "bed",
                "desk",
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
            },
            priority=3,
            context_sensitive=True,
        )

        self.slot_registry["product_name"] = SlotInfo(
            name="product_name",
            category=SlotCategory.PRODUCT,
            required_for_intents={IntentType.PRODUCT_DETAIL, IntentType.WARRANTY_QUERY},
            related_slots={"product_type", "brand"},
            semantic_keywords=set(),  # Will be populated dynamically
            priority=4,  # Highest priority - most important for context
            context_sensitive=False,  # Preserve across context switches
        )

        self.slot_registry["brand"] = SlotInfo(
            name="brand",
            category=SlotCategory.PRODUCT,
            required_for_intents={IntentType.WARRANTY_QUERY},
            related_slots={"product_name", "product_type"},
            semantic_keywords=set(),  # Will be populated dynamically
            priority=2,
            context_sensitive=True,
        )

        # Room-related slots
        self.slot_registry["room_type"] = SlotInfo(
            name="room_type",
            category=SlotCategory.ROOM,
            required_for_intents={IntentType.PRODUCT_SEARCH, IntentType.BUDGET_QUERY},
            related_slots={"room_size"},
            semantic_keywords={
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
                "balcony",
                "terrace",
                "patio",
                "deck",
                "porch",
                "garden",
            },
            priority=2,
            context_sensitive=True,
        )

        # Style-related slots
        self.slot_registry["color"] = SlotInfo(
            name="color",
            category=SlotCategory.STYLE,
            required_for_intents=set(),
            related_slots={"material", "style"},
            semantic_keywords={
                "white",
                "black",
                "blue",
                "red",
                "green",
                "yellow",
                "brown",
                "gray",
                "pink",
                "purple",
                "orange",
                "neutral",
                "colorful",
            },
            priority=1,
            context_sensitive=True,
        )

        self.slot_registry["material"] = SlotInfo(
            name="material",
            category=SlotCategory.STYLE,
            required_for_intents=set(),
            related_slots={"color", "style"},
            semantic_keywords={
                "wood",
                "metal",
                "plastic",
                "fabric",
                "leather",
                "cotton",
                "silk",
                "wool",
                "polyester",
                "glass",
                "ceramic",
                "stone",
            },
            priority=1,
            context_sensitive=True,
        )

        # Budget-related slots
        self.slot_registry["budget"] = SlotInfo(
            name="budget",
            category=SlotCategory.BUDGET,
            required_for_intents={IntentType.BUDGET_QUERY},
            related_slots={"price_range"},
            semantic_keywords={
                "budget",
                "price",
                "cost",
                "under",
                "less than",
                "above",
                "expensive",
                "cheap",
                "affordable",
                "luxury",
            },
            priority=2,
            context_sensitive=True,
        )

        # Warranty-related slots
        self.slot_registry["warranty"] = SlotInfo(
            name="warranty",
            category=SlotCategory.WARRANTY,
            required_for_intents={IntentType.WARRANTY_QUERY},
            related_slots={"warranty_period", "warranty_type"},
            semantic_keywords={
                "warranty",
                "guarantee",
                "coverage",
                "protection",
                "period",
            },
            priority=1,
            context_sensitive=True,
        )

    def detect_context_switch(
        self,
        user_message: str,
        current_intent: IntentType,
        previous_intent: Optional[IntentType] = None,
    ) -> Tuple[bool, float]:
        """
        Detect if the user is switching context to a different category or topic

        Returns:
            Tuple[bool, float]: (is_context_switch, continuity_score)
        """
        user_lower = user_message.lower().strip()

        # Context switch indicators
        context_switch_indicators = [
            # Explicit category changes
            r"show me (?:some|different|other) (?:products|items|things)",
            r"what about (?:.*?) (?:products|items)",
            r"instead of (?:.*?) (?:show|find) (?:.*?)",
            r"change to (?:.*?)",
            r"switch to (?:.*?)",
            r"now (?:show|find) (?:.*?)",
            r"how about (?:.*?)",
            # New search patterns
            r"find (?:me )?(?:some )?(?:.*?) (?:for|in) (?:.*?)",
            r"search for (?:.*?)",
            r"look for (?:.*?)",
            # Category-specific requests
            r"(?:beds?|sofas?|chairs?|tables?|curtains?|rugs?|lights?) (?:for|in)",
            r"(?:furniture|furnishings|lighting|bath|kitchen) (?:for|in)",
        ]

        # Continuity indicators (preserve context)
        continuity_indicators = [
            r"this (?:product|item|one)",
            r"that (?:product|item|one)",
            r"it",
            r"the (?:product|item)",
            r"details (?:about|of)",
            r"more (?:about|details)",
            r"tell me (?:more|about)",
            r"what (?:about|is) (?:this|that|it)",
            r"price (?:of|for)",
            r"warranty (?:of|for)",
            r"color (?:of|for)",
            r"size (?:of|for)",
            r"material (?:of|for)",
        ]

        # Check for context switch patterns
        context_switch_detected = False
        for pattern in context_switch_indicators:
            if re.search(pattern, user_lower):
                context_switch_detected = True
                break

        # Check for continuity patterns
        continuity_detected = False
        for pattern in continuity_indicators:
            if re.search(pattern, user_lower):
                continuity_detected = True
                break

        # Calculate continuity score
        continuity_score = 1.0
        if context_switch_detected:
            continuity_score = 0.2
        elif continuity_detected:
            continuity_score = 0.9
        elif previous_intent and current_intent != previous_intent:
            # Intent change might indicate context switch
            continuity_score = 0.5

        # Additional heuristics
        if len(user_lower.split()) <= 2 and not continuity_detected:
            # Short messages without continuity indicators might be new context
            continuity_score = min(continuity_score, 0.7)

        return context_switch_detected, continuity_score

    def determine_slot_relevance(self, slot_name: str, context: SlotContext) -> float:
        """
        Determine how relevant a slot is in the current context

        Returns:
            float: Relevance score (0.0 to 1.0)
        """
        if slot_name not in self.slot_registry:
            return 0.0

        slot_info = self.slot_registry[slot_name]
        relevance_score = 0.0

        # Check if slot is required for current intent
        if context.current_intent in slot_info.required_for_intents:
            relevance_score += 0.4

        # Check semantic relevance in user message
        user_lower = context.user_message.lower()
        for keyword in slot_info.semantic_keywords:
            if keyword in user_lower:
                relevance_score += 0.3
                break

        # Check if related slots are present (indicates context relevance)
        for related_slot in slot_info.related_slots:
            if related_slot in context.extracted_entities:
                relevance_score += 0.2
                break

        # Priority bonus
        relevance_score += slot_info.priority * 0.1

        # Context sensitivity adjustment
        if slot_info.context_sensitive and context.category_switch_detected:
            relevance_score *= 0.5

        return min(relevance_score, 1.0)

    def get_slots_to_clear(
        self, current_slots: Dict[str, str], context: SlotContext
    ) -> List[str]:
        """
        Determine which slots should be cleared based on context

        Returns:
            List[str]: List of slot names to clear
        """
        slots_to_clear = []

        for slot_name, slot_value in current_slots.items():
            if not slot_value:  # Skip empty slots
                continue

            relevance = self.determine_slot_relevance(slot_name, context)

            # Clear slots with low relevance
            if relevance < 0.3:
                slots_to_clear.append(slot_name)
                print(
                    f"[SLOT_MANAGER] Clearing slot '{slot_name}' (relevance: {relevance:.2f})"
                )

            # Clear context-sensitive slots on category switch
            elif (
                context.category_switch_detected
                and slot_name in self.slot_registry
                and self.slot_registry[slot_name].context_sensitive
            ):
                slots_to_clear.append(slot_name)
                print(
                    f"[SLOT_MANAGER] Clearing context-sensitive slot '{slot_name}' on category switch"
                )

        return slots_to_clear

    def get_slots_to_preserve(
        self, current_slots: Dict[str, str], context: SlotContext
    ) -> List[str]:
        """
        Determine which slots should be preserved based on context

        Returns:
            List[str]: List of slot names to preserve
        """
        slots_to_preserve = []

        for slot_name, slot_value in current_slots.items():
            if not slot_value:  # Skip empty slots
                continue

            relevance = self.determine_slot_relevance(slot_name, context)

            # Preserve high-relevance slots
            if relevance >= 0.6:
                slots_to_preserve.append(slot_name)
                print(
                    f"[SLOT_MANAGER] Preserving slot '{slot_name}' (relevance: {relevance:.2f})"
                )

            # Always preserve high-priority slots unless context switch is very strong
            elif (
                slot_name in self.slot_registry
                and self.slot_registry[slot_name].priority >= 3
                and context.context_continuity_score > 0.3
            ):
                slots_to_preserve.append(slot_name)
                print(f"[SLOT_MANAGER] Preserving high-priority slot '{slot_name}'")

        return slots_to_preserve

    def update_slot_registry_dynamically(self, retriever, llm):
        """
        Dynamically update slot registry with information from retriever and LLM
        """
        try:
            # Update brand keywords from retriever
            if hasattr(retriever, "get_brand_names"):
                brand_names = retriever.get_brand_names()
                if brand_names:
                    self.slot_registry["brand"].semantic_keywords.update(brand_names)
                    print(
                        f"[SLOT_MANAGER] Updated brand keywords: {len(brand_names)} brands"
                    )

            # Update product type keywords from retriever
            if hasattr(retriever, "get_product_type_mappings"):
                product_mappings = retriever.get_product_type_mappings()
                if product_mappings:
                    product_keywords = set(product_mappings.keys())
                    self.slot_registry["product_type"].semantic_keywords.update(
                        product_keywords
                    )
                    print(
                        f"[SLOT_MANAGER] Updated product type keywords: {len(product_keywords)} types"
                    )

            # Update product name keywords dynamically
            if hasattr(retriever, "get_sample_product_names"):
                product_names = retriever.get_sample_product_names()
                if product_names:
                    # Extract common words from product names
                    common_words = set()
                    for name in product_names[:100]:  # Limit to first 100
                        words = name.lower().split()
                        common_words.update(words)

                    # Filter out common words that aren't product-specific
                    common_words = {
                        word
                        for word in common_words
                        if len(word) > 2
                        and word
                        not in {"the", "and", "for", "with", "from", "this", "that"}
                    }

                    self.slot_registry["product_name"].semantic_keywords.update(
                        common_words
                    )
                    print(
                        f"[SLOT_MANAGER] Updated product name keywords: {len(common_words)} words"
                    )

        except Exception as e:
            print(f"[SLOT_MANAGER] Error updating slot registry: {e}")

    def manage_slots(
        self,
        current_slots: Dict[str, str],
        user_message: str,
        current_intent: IntentType,
        previous_intent: Optional[IntentType] = None,
        extracted_entities: Dict[str, str] = None,
        conversation_history: List[Dict] = None,
    ) -> Dict[str, str]:
        """
        Main slot management function that intelligently manages slots based on context

        Returns:
            Dict[str, str]: Updated slots
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [SLOT_MANAGER] Starting slot management...")
        print(f"[{timestamp}] [SLOT_MANAGER] Current slots: {current_slots}")
        print(f"[{timestamp}] [SLOT_MANAGER] User message: {user_message}")
        print(f"[{timestamp}] [SLOT_MANAGER] Current intent: {current_intent}")

        # Create context
        context_switch_detected, continuity_score = self.detect_context_switch(
            user_message, current_intent, previous_intent
        )

        context = SlotContext(
            current_intent=current_intent,
            previous_intent=previous_intent,
            user_message=user_message,
            conversation_history=conversation_history or [],
            extracted_entities=extracted_entities or {},
            category_switch_detected=context_switch_detected,
            context_continuity_score=continuity_score,
        )

        print(
            f"[{timestamp}] [SLOT_MANAGER] Context switch detected: {context_switch_detected}"
        )
        print(f"[{timestamp}] [SLOT_MANAGER] Continuity score: {continuity_score:.2f}")

        # Determine which slots to clear and preserve
        slots_to_clear = self.get_slots_to_clear(current_slots, context)
        slots_to_preserve = self.get_slots_to_preserve(current_slots, context)

        # Create updated slots
        updated_slots = {}

        # Preserve relevant slots
        for slot_name in slots_to_preserve:
            if slot_name in current_slots:
                updated_slots[slot_name] = current_slots[slot_name]
                print(
                    f"[{timestamp}] [SLOT_MANAGER] Preserved slot: {slot_name} = {current_slots[slot_name]}"
                )

        # Add new extracted entities
        if extracted_entities:
            for entity_type, value in extracted_entities.items():
                # Map entity types to slot names
                entity_to_slot_mapping = {
                    "PRODUCT_TYPE": "product_type",
                    "PRODUCT_NAME": "product_name",
                    "BRAND": "brand",
                    "ROOM_TYPE": "room_type",
                    "ROOM": "room_type",
                    "COLOR": "color",
                    "MATERIAL": "material",
                    "BUDGET": "budget",
                    "BUDGET_RANGE": "budget",
                    "SIZE": "size",
                    "STYLE": "style",
                    "WARRANTY": "warranty",
                }

                slot_name = entity_to_slot_mapping.get(entity_type)
                if slot_name and value:
                    updated_slots[slot_name] = value
                    print(
                        f"[{timestamp}] [SLOT_MANAGER] Added new slot: {slot_name} = {value}"
                    )

        # Log slot management summary
        cleared_count = len(slots_to_clear)
        preserved_count = len(slots_to_preserve)
        added_count = len(extracted_entities) if extracted_entities else 0

        print(f"[{timestamp}] [SLOT_MANAGER] Slot management summary:")
        print(f"[{timestamp}] [SLOT_MANAGER]   - Cleared: {cleared_count} slots")
        print(f"[{timestamp}] [SLOT_MANAGER]   - Preserved: {preserved_count} slots")
        print(f"[{timestamp}] [SLOT_MANAGER]   - Added: {added_count} new slots")
        print(f"[{timestamp}] [SLOT_MANAGER] Final slots: {updated_slots}")

        return updated_slots

    def get_required_slots_for_intent(self, intent: IntentType) -> List[str]:
        """
        Get required slots for an intent based on the dynamic slot registry
        """
        required_slots = []

        for slot_name, slot_info in self.slot_registry.items():
            if intent in slot_info.required_for_intents:
                required_slots.append(slot_name)

        return required_slots

    def validate_slots_for_intent(
        self, slots: Dict[str, str], intent: IntentType
    ) -> Tuple[bool, List[str]]:
        """
        Validate if all required slots for an intent are filled

        Returns:
            Tuple[bool, List[str]]: (all_required_filled, missing_slots)
        """
        required_slots = self.get_required_slots_for_intent(intent)
        missing_slots = [
            slot for slot in required_slots if slot not in slots or not slots[slot]
        ]

        return len(missing_slots) == 0, missing_slots


# Global slot manager instance
_slot_manager = None


def get_slot_manager() -> SmartSlotManager:
    """Get the global slot manager instance"""
    global _slot_manager
    if _slot_manager is None:
        _slot_manager = SmartSlotManager()
    return _slot_manager


def initialize_slot_manager(retriever=None, llm=None):
    """Initialize the slot manager with dynamic updates"""
    manager = get_slot_manager()
    if retriever and llm:
        manager.update_slot_registry_dynamically(retriever, llm)
    return manager
