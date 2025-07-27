# NER Model Improvements Guide

## Problem Statement

The original NER model was extracting entire phrases like "im looking for lightings" as the product type instead of just the core product "lighting". This was happening because:

1. **Training Data Issues**: The training data had examples where entire phrases were labeled as entities
2. **Entity Boundary Problems**: The model wasn't learning precise entity boundaries
3. **Lack of Specific Examples**: Missing examples for problematic cases with prefixes/suffixes

## Solution Overview

We've implemented a comprehensive solution to improve NER model training:

### 1. Enhanced Training Data Generation

**File**: `backend/scripts/improve_ner_training.py`

This script creates improved training data with:

- **Precise Entity Boundaries**: Examples where only the core entity is labeled
- **Problematic Case Examples**: Specific examples for phrases like "im looking for lightings"
- **Prefix/Suffix Variations**: Multiple examples with different prefixes and suffixes
- **Budget Extraction**: Dedicated examples for budget entity extraction
- **Complex Multi-Entity Examples**: Examples with multiple entities in one sentence

### 2. Improved Training Script

**File**: `backend/scripts/train_improved_ner.py`

Enhanced training with:

- **Better Parameters**: Optimized learning rate, batch size, and training epochs
- **Early Stopping**: Prevents overfitting
- **Detailed Metrics**: Per-entity precision, recall, and F1 scores
- **Sample Predictions**: Manual review of model outputs
- **Improved Tokenization**: Better handling of subword tokens

### 3. Comprehensive Testing

**File**: `backend/test_improved_ner.py`

Test script to verify improvements:

- **Problematic Cases**: Tests the specific cases that were failing
- **Complex Cases**: Tests multi-entity extraction
- **Budget Extraction**: Tests budget entity recognition
- **Comparison**: Compares with expected behavior

## How to Use

### Step 1: Generate Improved Training Data

```bash
cd backend
python scripts/improve_ner_training.py
```

This will:

- Create `backend/data/training/improved_ner_training_data.jsonl` (combined data)
- Create `backend/data/training/new_ner_examples.jsonl` (new examples only)

### Step 2: Train the Improved Model

```bash
cd backend
python scripts/train_improved_ner.py
```

This will:

- Train the model using the improved data
- Save the model to `./trained_improved_ner_model/`
- Generate detailed training results and metrics

### Step 3: Test the Improved Model

```bash
cd backend
python test_improved_ner.py
```

This will test the model on various cases and show:

- How it handles problematic phrases
- Entity extraction accuracy
- Comparison with expected behavior

## Key Improvements

### 1. Precise Entity Boundaries

**Before**: `"im looking for lightings"` → `B-PRODUCT_TYPE` for entire phrase
**After**: `"im looking for lightings"` → `B-PRODUCT_TYPE` for "lighting" only

### 2. Training Data Examples

The improved training data includes:

```json
{
  "id": "problematic_001",
  "tokens": ["im", "looking", "for", "lightings"],
  "labels": ["O", "O", "O", "B-PRODUCT_TYPE"]
}
```

### 3. Enhanced Training Parameters

- **Learning Rate**: 3e-5 (optimized for NER)
- **Batch Size**: 16 (balanced for memory and performance)
- **Epochs**: 5 (with early stopping)
- **Gradient Accumulation**: 2 steps
- **FP16**: Enabled for faster training

### 4. Better Evaluation

- **Per-Entity Metrics**: Individual precision/recall for each entity type
- **Sample Predictions**: Manual review of model outputs
- **Error Analysis**: Insights into common prediction errors

## Expected Results

After training the improved model, you should see:

1. **Precise Extraction**: Only core product types extracted, not entire phrases
2. **Better Accuracy**: Higher F1 scores across all entity types
3. **Consistent Behavior**: Reliable extraction across different input variations
4. **Budget Recognition**: Improved budget entity extraction

## Integration with Agent

To use the improved model in your agent:

1. **Update Model Path**: Point to the new trained model
2. **Remove Rule-Based Cleaning**: The improved model should eliminate the need for post-processing
3. **Test Integration**: Verify the agent works correctly with the new model

## Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure you've trained the improved model first
2. **Memory Issues**: Reduce batch size if training fails
3. **Poor Results**: Check training data quality and try more epochs

### Performance Optimization

- **GPU Training**: Use CUDA for faster training
- **Data Augmentation**: Add more variations to training data
- **Hyperparameter Tuning**: Experiment with learning rates and batch sizes

## Future Improvements

1. **Active Learning**: Continuously improve with user feedback
2. **Domain Adaptation**: Adapt to specific user language patterns
3. **Multi-Language Support**: Extend to other languages
4. **Real-time Learning**: Update model based on conversation patterns

## Files Summary

| File                                                     | Purpose                         |
| -------------------------------------------------------- | ------------------------------- |
| `backend/scripts/improve_ner_training.py`                | Generate improved training data |
| `backend/scripts/train_improved_ner.py`                  | Train improved NER model        |
| `backend/test_improved_ner.py`                           | Test model improvements         |
| `backend/data/training/improved_ner_training_data.jsonl` | Enhanced training data          |
| `./trained_improved_ner_model/`                          | Trained model directory         |

## Conclusion

This comprehensive approach addresses the core issue of imprecise entity extraction by:

1. **Creating better training data** with precise entity boundaries
2. **Using improved training techniques** with optimized parameters
3. **Providing comprehensive testing** to verify improvements
4. **Enabling easy integration** with the existing agent system

The result should be a much more accurate NER model that extracts only the core entities rather than entire phrases.
