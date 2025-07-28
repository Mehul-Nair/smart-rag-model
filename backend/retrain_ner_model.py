#!/usr/bin/env python3
"""
Retrain NER model with better data
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

from train_deberta_ner import DeBERTaNERTrainer


def retrain_ner_model():
    """Retrain the NER model with improved data"""

    print("🔧 Starting NER model retraining...")

    # Initialize trainer
    trainer = DeBERTaNERTrainer()

    # Load improved training data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    training_data_file = os.path.join(
        current_dir, "data", "training", "improved_ner_training_data.jsonl"
    )

    if not os.path.exists(training_data_file):
        print(
            "❌ Improved training data not found. Please run create_better_ner_training_data.py first."
        )
        return

    # Load training data
    training_data = trainer.load_training_data(training_data_file)

    # Prepare dataset
    dataset = trainer.prepare_dataset(training_data)

    # Split into train/eval
    train_dataset = dataset.select(range(0, int(0.8 * len(dataset))))
    eval_dataset = dataset.select(range(int(0.8 * len(dataset)), len(dataset)))

    print(f"📊 Training dataset size: {len(train_dataset)}")
    print(f"📊 Evaluation dataset size: {len(eval_dataset)}")

    # Initialize model
    trainer.initialize_model()

    # Train model
    output_dir = os.path.join(current_dir, "trained_deberta_ner_model_improved")
    trainer.train_model(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        num_epochs=5,  # More epochs for better learning
        batch_size=8,  # Smaller batch size for better generalization
        learning_rate=1e-5,  # Lower learning rate for fine-tuning
        warmup_steps=100,
        weight_decay=0.01,
    )

    # Evaluate model
    results = trainer.evaluate_model(trainer.trainer, eval_dataset)
    print(f"📊 Evaluation results: {results}")

    # Save training results
    trainer.save_training_results(results, output_dir)

    print(f"✅ Model retrained and saved to: {output_dir}")
    print(
        "🔄 Please update the model path in your configuration to use the improved model."
    )


if __name__ == "__main__":
    retrain_ner_model()
