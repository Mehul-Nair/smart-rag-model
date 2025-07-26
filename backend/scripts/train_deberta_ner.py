#!/usr/bin/env python3
"""
Train Custom DeBERTa v3 Small NER Model

This script trains a custom Named Entity Recognition model using DeBERTa v3 Small
for the home decor domain with ONNX optimization support.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeBERTaNERTrainer:
    """Trainer for DeBERTa v3 Small NER model"""

    def __init__(self, model_name: str = "microsoft/deberta-v3-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}
        self.num_labels = 0

    def load_training_data(self, data_file: str):
        """Load training data and create label mapping"""
        logger.info(f"Loading training data from {data_file}")

        # Load training data (supports both JSON and JSONL)
        training_data = []
        with open(data_file, "r", encoding="utf-8") as f:
            if data_file.endswith(".jsonl"):
                # JSONL format - one JSON object per line
                for line in f:
                    if line.strip():
                        training_data.append(json.loads(line.strip()))
            else:
                # JSON format - single JSON array
                training_data = json.load(f)

        # Create label mapping from entity types
        entity_types = [
            "PRODUCT_TYPE",
            "MATERIAL",
            "COLOR",
            "SIZE",
            "BRAND",
            "STYLE",
            "ROOM",
            "BUDGET",
            "PRODUCT_NAME",
        ]

        self.label2id = {"O": 0}
        for entity_type in entity_types:
            self.label2id[f"B-{entity_type}"] = len(self.label2id)
            self.label2id[f"I-{entity_type}"] = len(self.label2id)

        # Create reverse mapping
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_labels = len(self.label2id)

        logger.info(f"Loaded {len(training_data)} training examples")
        logger.info(f"Label mapping: {self.label2id}")
        logger.info(f"Number of labels: {self.num_labels}")

        return training_data

    def tokenize_and_align_labels(self, examples):
        """Tokenize inputs and align labels"""
        tokenized_inputs = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )

        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id.get(label[word_idx], -100))
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def convert_custom_format_to_huggingface(
        self, training_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert custom dataset format to HuggingFace format"""
        logger.info("Converting custom dataset format to HuggingFace format...")

        huggingface_data = []

        for example in training_data:
            text = example["text"]
            entities = example["entities"]

            # Tokenize the text
            tokens = text.split()
            labels = ["O"] * len(tokens)

            # Map entities to tokens
            for entity in entities:
                entity_text = entity["text"]
                entity_label = entity["label"]

                # Find the tokens that correspond to this entity
                entity_tokens = entity_text.split()
                start_idx, end_idx = self._find_tokens_for_entity(
                    text, entity_text, tokens
                )

                if start_idx is not None and end_idx is not None:
                    # Mark beginning token
                    labels[start_idx] = f"B-{entity_label}"
                    # Mark continuation tokens
                    for i in range(start_idx + 1, end_idx + 1):
                        labels[i] = f"I-{entity_label}"

            huggingface_data.append(
                {
                    "text": text,
                    "tokens": tokens,
                    "labels": labels,
                }
            )

        return huggingface_data

    def _find_tokens_for_entity(
        self, text: str, entity_text: str, tokens: List[str]
    ) -> tuple:
        """Find token indices for an entity"""
        entity_text_lower = entity_text.lower()
        text_lower = text.lower()

        # Find entity position in text
        start_char = text_lower.find(entity_text_lower)
        if start_char == -1:
            return None, None

        end_char = start_char + len(entity_text)

        # Map character positions to token positions
        current_pos = 0
        start_token = None
        end_token = None

        for i, token in enumerate(tokens):
            token_start = text_lower.find(token.lower(), current_pos)
            token_end = token_start + len(token)

            if token_start <= start_char < token_end:
                start_token = i

            if token_start < end_char <= token_end:
                end_token = i
                break

            current_pos = token_end

        if start_token is not None and end_token is not None:
            return start_token, end_token

        return None, None

    def prepare_dataset(self, training_data: List[Dict[str, Any]]) -> Dataset:
        """Prepare dataset for training"""
        logger.info("Preparing dataset for training...")

        # Convert custom format to HuggingFace format
        huggingface_data = self.convert_custom_format_to_huggingface(training_data)

        # Convert to HuggingFace dataset format
        dataset_dict = {
            "text": [item["text"] for item in huggingface_data],
            "tokens": [item["tokens"] for item in huggingface_data],
            "labels": [item["labels"] for item in huggingface_data],
        }

        dataset = Dataset.from_dict(dataset_dict)

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset.column_names,
        )

        logger.info(f"Dataset prepared: {len(tokenized_dataset)} examples")
        return tokenized_dataset

    def initialize_model(self):
        """Initialize the DeBERTa model"""
        logger.info(f"Initializing DeBERTa v3 Small model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        logger.info("✅ Model initialized successfully")

    def train_model(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        output_dir: str = "./trained_deberta_ner_model",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
    ):
        """Train the DeBERTa NER model"""
        logger.info("🚀 Starting model training...")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=learning_rate,
            save_total_limit=2,
            report_to=None,  # Disable wandb/tensorboard
            logging_first_step=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )

        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, return_tensors="pt"
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train the model
        logger.info("📈 Training started...")
        start_time = time.time()

        trainer.train()

        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time/60:.1f} minutes")

        # Save the model and tokenizer
        logger.info(f"💾 Saving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save label mapping
        mapping_path = os.path.join(output_dir, "label_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(self.label2id, f, indent=2)

        logger.info("✅ Model saved successfully")

        return trainer

    def evaluate_model(self, trainer: Trainer, test_dataset: Dataset) -> Dict[str, Any]:
        """Evaluate the trained model"""
        logger.info("🔍 Evaluating model...")

        # Get predictions
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=2)

        # Convert predictions to labels
        pred_labels = []
        true_labels = []

        for i, pred in enumerate(preds):
            word_ids = predictions.label_ids[i]
            pred_label = []
            true_label = []

            for j, word_id in enumerate(word_ids):
                if word_id != -100:
                    pred_label.append(self.id2label.get(pred[j], "O"))
                    true_label.append(self.id2label.get(word_id, "O"))

            pred_labels.extend(pred_label)
            true_labels.extend(true_label)

        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels, output_dict=True)

        results = {
            "accuracy": accuracy,
            "classification_report": report,
            "predictions": pred_labels[:100],  # First 100 for inspection
            "true_labels": true_labels[:100],
        }

        logger.info(f"📊 Test Accuracy: {accuracy:.4f}")
        logger.info(
            f"📊 Classification Report:\n{classification_report(true_labels, pred_labels)}"
        )

        return results

    def save_training_results(self, results: Dict[str, Any], output_dir: str):
        """Save training results"""
        results_path = os.path.join(output_dir, "training_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"📊 Training results saved to {results_path}")


def main():
    """Main training function"""
    logger.info("🚀 Starting DeBERTa v3 Small NER Training Pipeline")
    logger.info("=" * 60)

    # Configuration
    model_name = "microsoft/deberta-v3-small"
    data_file = "data/training/ner/ner_data.jsonl"  # Your JSONL dataset
    output_dir = "trained_deberta_ner_model"

    # Training parameters for better performance
    num_epochs = 5  # Increased from 3
    batch_size = 8  # Reduced for better stability
    learning_rate = 3e-5  # Slightly higher learning rate

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/training/ner", exist_ok=True)

    # Check if custom dataset exists
    if not os.path.exists(data_file):
        logger.error(f"❌ Custom dataset not found at {data_file}")
        logger.error("Please create your dataset file before training.")
        logger.error(
            "Expected format: JSONL file with 'text' and 'entities' fields (one JSON object per line)"
        )
        return

    # Initialize trainer
    trainer = DeBERTaNERTrainer(model_name)

    # Load training data
    training_data = trainer.load_training_data(data_file)

    # Initialize model
    trainer.initialize_model()

    # Prepare dataset
    dataset = trainer.prepare_dataset(training_data)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size

    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )

    logger.info(
        f"📊 Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval"
    )

    # Train model
    trainer_instance = trainer.train_model(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # Evaluate model
    results = trainer.evaluate_model(trainer_instance, eval_dataset)

    # Save results
    trainer.save_training_results(results, output_dir)

    logger.info("🎉 Training pipeline completed successfully!")
    logger.info(f"📁 Model saved to: {output_dir}")
    logger.info(f"📊 Final accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
