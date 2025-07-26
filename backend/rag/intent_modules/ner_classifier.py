#!/usr/bin/env python3
"""
NER Classifier for Slot Extraction

This module provides Named Entity Recognition (NER) functionality
to extract slots like PRODUCT_TYPE, ROOM_TYPE, BUDGET_RANGE, etc.
from user queries for the conversational agent.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import torch.nn.functional as F


@dataclass
class Entity:
    """Represents a detected entity"""

    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float


@dataclass
class NERResult:
    """Result of NER classification"""

    entities: List[Entity]
    slots: Dict[str, str]  # entity_type -> extracted_value


class NERClassifier:
    """NER classifier for slot extraction"""

    def __init__(self, model_path: str = "trained_ner_model"):
        """Initialize the NER classifier"""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load label mapping
        with open(os.path.join(model_path, "label_mapping.json"), "r") as f:
            label_mapping = json.load(f)
            self.id2label = {int(k): v for k, v in label_mapping["id2label"].items()}
            self.label2id = label_mapping["label2id"]

        print(f"✅ NER Classifier loaded from {model_path}")
        print(f"🔧 Using device: {self.device}")

    def extract_entities(self, text: str) -> NERResult:
        """Extract entities from text"""
        # Use proper tokenization to avoid duplicate tokens
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        # Move to device
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        # Get predictions and probabilities
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
            probs = F.softmax(logits, dim=2)  # [batch, seq, num_labels]

        word_ids = tokenized.word_ids()
        entities = []
        current_entity = None
        current_confidences = []
        current_token_ids = []

        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue  # Skip special tokens

            label_id = predictions[0][i].item()
            label = self.id2label.get(label_id, "O")
            confidence = probs[0][i][label_id].item()
            token_id = input_ids[0][i].item()
            token_text = self.tokenizer.convert_ids_to_tokens(token_id)
            # Clean up the token text (remove special characters)
            if token_text.startswith("##"):
                token_text = token_text[2:]

            if label.startswith("B-"):
                # Start of new entity
                if current_entity:
                    # Compute mean confidence for the entity
                    mean_conf = sum(current_confidences) / len(current_confidences)
                    current_entity.confidence = mean_conf
                    entities.append(current_entity)
                entity_type = label[2:]
                current_entity = Entity(
                    text=token_text,
                    entity_type=entity_type,
                    start_pos=word_id,
                    end_pos=word_id,
                    confidence=confidence,
                )
                current_confidences = [confidence]
                current_token_ids = [token_id]
            elif label.startswith("I-") and current_entity:
                entity_type = label[2:]
                if entity_type == current_entity.entity_type:
                    # Smarter subword merging: add space only if not a subword
                    if not self.tokenizer.convert_ids_to_tokens(token_id).startswith(
                        "##"
                    ) and not current_entity.text.endswith(" "):
                        current_entity.text += " "
                    current_entity.text += token_text
                    current_entity.end_pos = word_id
                    current_confidences.append(confidence)
                    current_token_ids.append(token_id)
                else:
                    # Different entity type, end current and start new
                    mean_conf = sum(current_confidences) / len(current_confidences)
                    current_entity.confidence = mean_conf
                    entities.append(current_entity)
                    current_entity = Entity(
                        text=token_text,
                        entity_type=entity_type,
                        start_pos=word_id,
                        end_pos=word_id,
                        confidence=confidence,
                    )
                    current_confidences = [confidence]
                    current_token_ids = [token_id]
            else:
                # O label or no current entity
                if current_entity:
                    mean_conf = sum(current_confidences) / len(current_confidences)
                    current_entity.confidence = mean_conf
                    entities.append(current_entity)
                    current_entity = None
                    current_confidences = []
                    current_token_ids = []

        # Add the last entity
        if current_entity:
            mean_conf = sum(current_confidences) / len(current_confidences)
            current_entity.confidence = mean_conf
            entities.append(current_entity)

        # Post-process entities to fix common issues
        processed_entities = []
        for entity in entities:
            # Clean up entity text
            entity.text = entity.text.strip()
            # Remove duplicate words
            words = entity.text.split()
            unique_words = []
            for word in words:
                if word not in unique_words:
                    unique_words.append(word)
            entity.text = " ".join(unique_words)
            # Fix common tokenization issues
            entity.text = entity.text.replace("ru pee s", "rupees")
            entity.text = entity.text.replace("ru pee", "rupees")
            entity.text = entity.text.replace("pee s", "pees")
            entity.text = entity.text.replace("1000 0", "10000")
            entity.text = entity.text.replace("500 0", "5000")
            # Remove stray punctuation at start/end
            entity.text = entity.text.strip(".,;:!?-\"'")
            # Add to processed entities if not empty
            if entity.text:
                processed_entities.append(entity)

        # Convert to slots dictionary
        slots = {}
        for entity in processed_entities:
            if entity.entity_type not in slots:
                slots[entity.entity_type] = entity.text

        return NERResult(entities=processed_entities, slots=slots)

    def extract_slots(self, text: str) -> Dict[str, str]:
        """Extract slots from text (simplified interface)"""
        result = self.extract_entities(text)
        return result.slots


# Global NER classifier instance
_ner_classifier: Optional[NERClassifier] = None


def get_ner_classifier() -> NERClassifier:
    """Get the global NER classifier instance"""
    global _ner_classifier
    if _ner_classifier is None:
        _ner_classifier = NERClassifier()
    return _ner_classifier


def extract_slots_from_text(text: str) -> Dict[str, str]:
    """Extract slots from text using the global NER classifier"""
    classifier = get_ner_classifier()
    return classifier.extract_slots(text)


def test_multi_word_entity():
    ner = get_ner_classifier()
    text = "Show me Pure Royale curtains for my bedroom"
    result = ner.extract_entities(text)
    assert any(
        e.text == "Pure Royale" and e.entity_type == "BRAND" for e in result.entities
    ), "Multi-word BRAND entity extraction failed!"
    print("✅ Multi-word entity extraction test passed.")


if __name__ == "__main__":
    classifier = NERClassifier()
    test_texts = [
        "I need Pure Royale curtains for my bedroom",
        "Show me White Teak by Asian Paints ceiling lights for the dining area",
        "I want a budget-friendly sofa for the living room",
        "Looking for red cotton cushions under 5000 rupees",
    ]
    for text in test_texts:
        print(f"\n🔍 Text: {text}")
        result = classifier.extract_entities(text)
        print(f"📋 Slots: {result.slots}")
        print(
            f"🏷️  Entities: {[(e.text, e.entity_type, e.confidence) for e in result.entities]}"
        )
    # Run the multi-word entity test
    test_multi_word_entity()
