# DeBERTa NER Training Guide with ONNX Support

This guide walks you through training a custom Named Entity Recognition (NER) model using Microsoft's DeBERTa v3 Small model with ONNX runtime optimization for your home decor domain.

## Overview

The training pipeline consists of three main steps:

1. **Data Preparation**: Create training data with entity annotations
2. **Model Training**: Train the DeBERTa model on your custom dataset
3. **ONNX Conversion**: Convert the trained model to ONNX format for optimized inference

## Prerequisites

### 1. Install Dependencies

```bash
# Install training dependencies
pip install -r backend/requirements-training.txt

# Install accelerate (required for training)
pip install accelerate
```

### 2. Verify Installation

```bash
python -c "import torch, transformers, datasets, onnxruntime; print('All dependencies installed successfully!')"
```

## Step 1: Data Preparation

The training data preparation script creates synthetic training examples for the home decor domain with the following entity types:

- **PRODUCT_TYPE**: sofa, chair, table, rug, etc.
- **BRAND**: IKEA, West Elm, Pottery Barn, etc.
- **COLOR**: blue, navy, cream, beige, etc.
- **MATERIAL**: leather, cotton, wood, metal, etc.
- **ROOM**: living room, bedroom, kitchen, etc.
- **BUDGET**: under $500, cheap, luxury, etc.
- **STYLE**: modern, traditional, bohemian, etc.
- **SIZE**: large, small, queen, king, etc.

### Run Data Preparation

```bash
python backend/scripts/prepare_ner_training_data.py
```

**Expected Output:**

```
INFO:__main__:🚀 Preparing NER training data for home decor domain...
INFO:__main__:Created 27 training examples
INFO:__main__:Label mapping: {'O': 0, 'B-PRODUCT_TYPE': 1, 'I-PRODUCT_TYPE': 2, ...}
INFO:__main__:✅ Training data preparation completed!
```

**Generated Files:**

- `backend/data/training/ner/ner_training_data.json` - Training examples
- `backend/data/training/ner/ner_label_mapping.json` - Label to ID mapping

## Step 2: Model Training

### Training Configuration

The training script uses the following configuration:

- **Model**: `microsoft/deberta-v3-small`
- **Epochs**: 3
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Dataset Split**: 80% train, 20% validation
- **Max Sequence Length**: 128 tokens

### Run Training

```bash
python backend/scripts/train_deberta_ner.py
```

**Expected Output:**

```
INFO:__main__:🚀 Starting DeBERTa v3 Small NER Training Pipeline
INFO:__main__:Loading training data from backend/data/training/ner/ner_training_data.json
INFO:__main__:Loaded 27 training examples
INFO:__main__:Label mapping: {'O': 0, 'B-PRODUCT_TYPE': 1, ...}
INFO:__main__:✅ Model initialized successfully
INFO:__main__:📊 Dataset split: 21 train, 6 eval
INFO:__main__:🚀 Starting model training...
INFO:__main__:📈 Training started...
INFO:__main__:✅ Training completed in X.X minutes
INFO:__main__:💾 Saving model to backend/trained_deberta_ner_model
INFO:__main__:✅ Model saved successfully
INFO:__main__:🔍 Evaluating model...
INFO:__main__:📊 Test Accuracy: 0.XXXX
INFO:__main__:🎉 Training pipeline completed successfully!
```

**Generated Files:**

- `backend/trained_deberta_ner_model/` - Trained model directory
  - `pytorch_model.bin` - Model weights
  - `config.json` - Model configuration
  - `tokenizer.json` - Tokenizer files
  - `label_mapping.json` - Label mapping
  - `training_results.json` - Training metrics

## Step 3: ONNX Conversion

### Convert to ONNX Format

```bash
python backend/scripts/convert_deberta_to_onnx.py
```

**Expected Output:**

```
INFO:__main__:🚀 Starting DeBERTa NER to ONNX Conversion
INFO:__main__:Loading model from backend/trained_deberta_ner_model
INFO:__main__:✅ Model loaded successfully
INFO:__main__:🔄 Converting model to ONNX format...
INFO:__main__:✅ ONNX model saved to backend/trained_deberta_ner_model/model.onnx
INFO:__main__:🧪 Testing ONNX model with text: 'I want a blue leather sofa'
INFO:__main__:✅ Predictions match: True
INFO:__main__:✅ ONNX inference class saved to backend/rag/intent_modules/deberta_onnx_ner.py
INFO:__main__:🎉 ONNX conversion completed successfully!
```

**Generated Files:**

- `backend/trained_deberta_ner_model/model.onnx` - ONNX model file
- `backend/rag/intent_modules/deberta_onnx_ner.py` - ONNX inference class

## Step 4: Testing the Model

### Test ONNX Model

```bash
python backend/rag/intent_modules/deberta_onnx_ner.py
```

**Expected Output:**

```
Text: I want a blue leather sofa from IKEA
Entities: [
    {'text': 'blue', 'type': 'COLOR', 'start': 3, 'end': 4},
    {'text': 'leather', 'type': 'MATERIAL', 'start': 5, 'end': 6},
    {'text': 'sofa', 'type': 'PRODUCT_TYPE', 'start': 7, 'end': 8},
    {'text': 'IKEA', 'type': 'BRAND', 'start': 13, 'end': 14}
]
```

## Integration with Your RAG System

### 1. Update Intent Module Factory

Add the ONNX NER module to your intent classification system:

```python
# In backend/rag/intent_modules/factory.py
from .deberta_onnx_ner import DeBERTaONNXNER

def create_ner_classifier():
    return DeBERTaONNXNER(
        model_path="backend/trained_deberta_ner_model",
        onnx_path="backend/trained_deberta_ner_model/model.onnx"
    )
```

### 2. Use in Your Agent

```python
# In your main agent code
ner_model = create_ner_classifier()

# Extract entities from user query
entities = ner_model.predict("I want a blue leather sofa under $500")
print(f"Extracted entities: {entities}")
```

## Customizing Training Data

### Adding More Training Examples

Edit `backend/scripts/prepare_ner_training_data.py` and add more examples to the `create_training_examples()` method:

```python
def create_training_examples(self) -> List[TrainingExample]:
    training_data = [
        # Add your custom examples here
        (
            "Your custom text here",
            [
                Entity("entity_text", start_char, end_char, "ENTITY_TYPE"),
                # ... more entities
            ],
        ),
        # ... more examples
    ]
    return training_data
```

### Adding New Entity Types

1. Add new entity types to the `entity_types` list in the `NERTrainingDataPreparer` class
2. Add training examples with the new entity types
3. Re-run the data preparation and training scripts

## Performance Optimization

### Training Optimization

- **GPU Training**: Set `CUDA_VISIBLE_DEVICES=0` for GPU training
- **Mixed Precision**: Add `fp16=True` to TrainingArguments for faster training
- **Gradient Accumulation**: Increase effective batch size with `gradient_accumulation_steps`

### Inference Optimization

- **ONNX Runtime**: Use ONNX runtime for 2-3x faster inference
- **Batch Processing**: Use `predict_batch()` for multiple texts
- **Model Quantization**: Consider INT8 quantization for further speedup

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or sequence length
2. **Training Not Converging**: Increase learning rate or epochs
3. **Poor Entity Recognition**: Add more diverse training examples
4. **ONNX Conversion Errors**: Check model compatibility and ONNX opset version

### Debugging

- Check training logs in `backend/trained_deberta_ner_model/logs/`
- Verify training data format in `backend/data/training/ner/ner_training_data.json`
- Test model predictions with sample inputs

## Next Steps

1. **Expand Training Data**: Add more diverse examples from your domain
2. **Fine-tune Hyperparameters**: Experiment with learning rates, batch sizes, etc.
3. **Evaluate Performance**: Test on real user queries
4. **Deploy**: Integrate with your production system

## Files Overview

```
backend/
├── scripts/
│   ├── prepare_ner_training_data.py    # Data preparation
│   ├── train_deberta_ner.py           # Model training
│   └── convert_deberta_to_onnx.py     # ONNX conversion
├── data/training/ner/
│   ├── ner_training_data.json         # Training examples
│   └── ner_label_mapping.json         # Label mapping
├── trained_deberta_ner_model/         # Trained model
│   ├── model.onnx                     # ONNX model
│   └── label_mapping.json             # Label mapping
└── rag/intent_modules/
    └── deberta_onnx_ner.py            # ONNX inference class
```

This completes the training pipeline for your custom DeBERTa NER model with ONNX optimization!
