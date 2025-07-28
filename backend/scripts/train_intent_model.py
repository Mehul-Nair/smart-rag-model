#!/usr/bin/env python3
"""
DeBERTa Intent Classification Training Script

This script trains a DeBERTa model for intent classification on your dataset.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
import logging
from typing import Dict, List, Tuple
import argparse
from transformers import TrainerCallback
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingProgressCallback(TrainerCallback):
    """Custom callback for detailed training progress logging"""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.current_epoch = 0

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        logger.info(f"🚀 Starting training with {self.total_steps} total steps")
        logger.info(f"📊 Training configuration:")
        logger.info(f"   - Epochs: {args.num_train_epochs}")
        logger.info(f"   - Batch size: {args.per_device_train_batch_size}")
        logger.info(f"   - Learning rate: {args.learning_rate}")
        logger.info(f"   - Total steps: {self.total_steps}")

    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each step"""
        self.current_step = state.global_step
        self.current_epoch = state.epoch
        # Removed verbose step logging - let the progress bar handle it

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation"""
        if metrics:
            eval_loss = metrics.get("eval_loss", "N/A")
            eval_accuracy = metrics.get("eval_accuracy", "N/A")
            current_epoch = int(state.epoch)
            progress_percent = (current_epoch / args.num_train_epochs) * 100

            logger.info(
                f"📊 Epoch {current_epoch} Results ({progress_percent:.1f}% complete):"
            )

            # Handle loss formatting
            if isinstance(eval_loss, (int, float)):
                logger.info(f"   - Validation Loss: {eval_loss:.4f}")
            else:
                logger.info(f"   - Validation Loss: {eval_loss}")

            # Handle accuracy formatting
            if isinstance(eval_accuracy, (int, float)):
                logger.info(f"   - Validation Accuracy: {eval_accuracy:.4f}")
            else:
                logger.info(f"   - Validation Accuracy: {eval_accuracy}")

            logger.info("   " + "=" * 50)

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        epoch_progress = (self.current_epoch / args.num_train_epochs) * 100

        # Get the latest training loss if available
        latest_loss = "N/A"
        if state.log_history:
            for log in reversed(state.log_history):
                if "loss" in log:
                    latest_loss = log["loss"]
                    break

        logger.info(
            f"✅ Epoch {self.current_epoch:.0f} completed! ({epoch_progress:.1f}% of training)"
        )
        if isinstance(latest_loss, (int, float)):
            logger.info(f"   - Training Loss: {latest_loss:.4f}")
        logger.info("   " + "=" * 50)

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        logger.info(f"✅ Training completed! Total steps: {self.current_step}")
        logger.info(f"📊 Final training statistics:")
        logger.info(f"   - Total epochs: {self.current_epoch:.2f}")
        logger.info(f"   - Total steps: {self.current_step}")
        logger.info(
            f"   - Final loss: {state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'}"
        )


class DeBERTaIntentTrainer:
    """DeBERTa model trainer for intent classification"""

    def __init__(self, model_name: str = "microsoft/deberta-v3-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.intent_mapping = {}
        self.reverse_mapping = {}

    def load_data(self, data_path: str) -> Tuple[List[str], List[int]]:
        """Load and prepare training data"""
        logger.info(f"Loading data from {data_path}")

        # Load your dataset (adjust based on your data format)
        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        elif data_path.endswith(".json"):
            with open(data_path, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported data format. Use CSV or JSON.")

        # Extract text and intent columns (adjust column names as needed)
        texts = df["text"].tolist()  # Adjust column name
        intents = df["intent"].tolist()  # Adjust column name

        # Create intent mapping
        unique_intents = sorted(list(set(intents)))
        self.intent_mapping = {intent: idx for idx, intent in enumerate(unique_intents)}
        self.reverse_mapping = {
            idx: intent for intent, idx in self.intent_mapping.items()
        }

        # Convert intents to numeric labels
        labels = [self.intent_mapping[intent] for intent in intents]

        logger.info(f"Loaded {len(texts)} samples with {len(unique_intents)} intents")
        logger.info(f"Intent mapping: {self.intent_mapping}")

        return texts, labels

    def prepare_dataset(self, texts: List[str], labels: List[int]) -> Dataset:
        """Prepare dataset for training"""
        logger.info("Preparing dataset...")

        # Tokenize texts
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)

        # Tokenize all texts
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
        )

        # Create dataset
        dataset_dict = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels),
        }

        dataset = Dataset.from_dict(dataset_dict)
        return dataset

    def train_model(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        output_dir: str = "./trained_deberta_model",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
    ):
        """Train the DeBERTa model"""
        logger.info("Initializing model...")

        # Initialize model
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.intent_mapping)
        )

        # Training arguments with clean progress bar
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,  # Log every 100 steps instead of every step
            eval_strategy="epoch",  # Evaluate at end of each epoch
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            learning_rate=learning_rate,
            save_total_limit=2,
            # Clean logging
            report_to=None,  # Disable wandb/tensorboard for simplicity
            logging_first_step=False,  # Don't log first step
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            # Progress bar settings
            disable_tqdm=False,  # Keep progress bar
        )

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Calculate total steps for progress tracking
        total_steps = len(train_dataset) // batch_size * num_epochs

        # Custom evaluation function to compute accuracy
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(labels, predictions)
            return {"eval_accuracy": accuracy}

        # Initialize trainer with custom callback and metrics
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[TrainingProgressCallback(total_steps)],
        )

        # Train the model with progress tracking
        logger.info("Starting training...")
        logger.info(f"📊 Dataset info:")
        logger.info(f"   - Training samples: {len(train_dataset)}")
        logger.info(f"   - Validation samples: {len(eval_dataset)}")
        logger.info(f"   - Total steps: {total_steps}")
        logger.info(f"   - Steps per epoch: {len(train_dataset) // batch_size}")

        # Start training with progress tracking
        trainer.train()

        # Save the model and tokenizer
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save intent mapping
        mapping_path = os.path.join(output_dir, "intent_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(self.intent_mapping, f, indent=2)

        logger.info("Training completed!")

        # Display final training statistics
        self._display_training_stats(trainer, total_steps)

        return trainer

    def _display_training_stats(self, trainer, total_steps):
        """Display comprehensive training statistics"""
        logger.info("📈 Final Training Statistics:")
        logger.info("=" * 50)

        # Get training history
        if hasattr(trainer, "state") and trainer.state.log_history:
            history = trainer.state.log_history

            # Find best metrics
            best_loss = min(
                [log.get("loss", float("inf")) for log in history if "loss" in log],
                default="N/A",
            )
            best_eval_loss = min(
                [
                    log.get("eval_loss", float("inf"))
                    for log in history
                    if "eval_loss" in log
                ],
                default="N/A",
            )
            best_accuracy = max(
                [
                    log.get("eval_accuracy", 0)
                    for log in history
                    if "eval_accuracy" in log
                ],
                default="N/A",
            )

            logger.info(f"🏆 Best Metrics:")
            logger.info(
                f"   - Best Training Loss: {best_loss:.4f}"
                if best_loss != "N/A"
                else f"   - Best Training Loss: {best_loss}"
            )
            logger.info(
                f"   - Best Validation Loss: {best_eval_loss:.4f}"
                if best_eval_loss != "N/A"
                else f"   - Best Validation Loss: {best_eval_loss}"
            )
            logger.info(
                f"   - Best Validation Accuracy: {best_accuracy:.4f}"
                if best_accuracy != "N/A"
                else f"   - Best Validation Accuracy: {best_accuracy}"
            )

            # Training progress summary
            logger.info(f"📊 Training Progress:")
            logger.info(f"   - Total Steps Completed: {len(history)}")
            logger.info(f"   - Total Steps Planned: {total_steps}")
            logger.info(f"   - Progress: {(len(history) / total_steps * 100):.1f}%")

        logger.info("=" * 50)

    def evaluate_model(self, test_texts: List[str], test_labels: List[int]) -> Dict:
        """Evaluate the trained model"""
        logger.info("Evaluating model...")

        # Tokenize test texts
        test_encodings = self.tokenizer(
            test_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**test_encodings)
            predictions = torch.argmax(outputs.logits, dim=1)

        # Convert predictions to intent names
        pred_intents = [self.reverse_mapping[pred.item()] for pred in predictions]
        true_intents = [self.reverse_mapping[label] for label in test_labels]

        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(true_intents, pred_intents, output_dict=True)

        results = {
            "accuracy": accuracy,
            "classification_report": report,
            "predictions": pred_intents,
            "true_labels": true_intents,
        }

        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(
            f"Classification Report:\n{classification_report(true_intents, pred_intents)}"
        )

        return results


def combine_training_data():
    """Combine all JSONL files from data/training/intent folder"""
    training_dir = "data/training/intent"
    all_data = []
    seen_entries = set()

    # Check if directory exists
    if not os.path.exists(training_dir):
        raise FileNotFoundError(f"Training directory not found: {training_dir}")

    # Check if directory has JSONL files
    jsonl_files = [f for f in os.listdir(training_dir) if f.endswith(".jsonl")]
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {training_dir}")

    print("📖 Reading training data from JSONL files...")

    for file in jsonl_files:
        file_path = os.path.join(training_dir, file)
        print(f"   Reading {file}...")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        # Validate required fields
                        if "text" not in data or "intent" not in data:
                            print(
                                f"   ⚠️  Skipping invalid data in {file} line {line_num}: missing text or intent"
                            )
                            continue

                        # Create unique key to avoid duplicates
                        entry_key = f"{data['text'].lower().strip()}_{data['intent']}"

                        if entry_key not in seen_entries:
                            all_data.append(
                                {"text": data["text"].strip(), "intent": data["intent"]}
                            )
                            seen_entries.add(entry_key)
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️  Skipping invalid JSON in {file} line {line_num}")
                        continue
        except Exception as e:
            print(f"   ❌ Error reading {file}: {e}")
            continue

    # Check if we have any data
    if not all_data:
        raise ValueError("No valid training examples found in JSONL files")

    # Save combined data
    output_file = "data/training/combined_intent_data.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)

    print(f"✅ Combined {len(df)} unique training examples")
    print(f"📊 Intent distribution:")
    print(df["intent"].value_counts())
    print(f"💾 Saved to: {output_file}")

    return output_file


def main():
    """Main training function - automatically reads from data/training/intent"""

    import argparse

    # Parse only epochs argument
    parser = argparse.ArgumentParser(
        description="Train DeBERTa for intent classification"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    args = parser.parse_args()

    try:
        print("🔄 DeBERTa Intent Classification Training")
        print("=" * 50)
        print(f"📊 Training for {args.epochs} epochs")

        # Step 1: Combine training data
        print("\n1️⃣ Combining training data...")
        data_file = combine_training_data()

        # Step 2: Initialize trainer
        print("\n2️⃣ Initializing trainer...")
        trainer = DeBERTaIntentTrainer()

        # Step 3: Load and split data
        print("\n3️⃣ Loading and splitting data...")
        texts, labels = trainer.load_data(data_file)

        # Validate we have enough data for splitting
        if len(texts) < 10:
            raise ValueError(
                f"Not enough training examples: {len(texts)}. Need at least 10."
            )

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Step 4: Prepare datasets
        print("\n4️⃣ Preparing datasets...")
        train_dataset = trainer.prepare_dataset(train_texts, train_labels)
        test_dataset = trainer.prepare_dataset(test_texts, test_labels)

        # Step 5: Train model
        print("\n5️⃣ Training model...")
        trainer.train_model(
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            output_dir="./trained_deberta_model",
            num_epochs=args.epochs,
            batch_size=8,
            learning_rate=2e-5,
        )

        # Step 6: Evaluate model
        print("\n6️⃣ Evaluating model...")
        results = trainer.evaluate_model(test_texts, test_labels)

        # Step 7: Save results
        results_path = os.path.join("./trained_deberta_model", "training_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Training completed!")
        print(f"📈 Final accuracy: {results['accuracy']:.2%}")
        print(f"💾 Results saved to: {results_path}")

    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        print("💡 Make sure the data/training/intent directory exists with JSONL files")
        return 1
    except ValueError as e:
        print(f"❌ Data error: {e}")
        print("💡 Check your training data format and content")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Check the error details above")
        return 1

    return 0


if __name__ == "__main__":
    main()
