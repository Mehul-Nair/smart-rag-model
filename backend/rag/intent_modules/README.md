# Modular Intent Classification System

A flexible, production-ready intent classification system that supports multiple implementations with easy switching and hybrid fallback strategies.

## 🚀 Features

- **Multiple Implementations**: OpenAI Fine-tuned Models, HuggingFace Transformers, Rule-based Classifiers
- **Three-Tier Fallback**: Primary → Rule-based → OpenAI with intelligent confidence-based routing
- **Hybrid Fallback**: Combine multiple classifiers with intelligent fallback strategies
- **Easy Configuration**: Environment variables and configuration management
- **Performance Monitoring**: Built-in performance statistics and monitoring
- **Production Ready**: Modular design with proper error handling and logging
- **Easy Integration**: Simple factory pattern for creating classifiers

## 📁 Structure

```
rag/intent_modules/
├── __init__.py              # Main module interface
├── base.py                  # Abstract base classes
├── factory.py               # Factory pattern for creating classifiers
├── config.py                # Configuration management
├── openai_classifier.py     # OpenAI fine-tuned model implementation
├── huggingface_classifier.py # HuggingFace transformers implementation
├── rule_based_classifier.py # Rule-based pattern matching implementation
└── README.md               # This file
```

## 🎯 Intent Types

The system classifies user messages into these intent types:

- **GREETING**: Hello, hi, greetings, introductions
- **HELP**: Help requests, how to use, instructions
- **CATEGORY_LIST**: What categories, list products, available options
- **PRODUCT_SEARCH**: Show me, find, looking for specific products
- **BUDGET_QUERY**: Price queries, budget constraints, cost questions
- **PRODUCT_DETAIL**: Tell me more, specifications, details about products
- **WARRANTY_QUERY**: Warranty, guarantee, return policy questions
- **CLARIFY**: Ambiguous or unclear requests, confusion
- **INVALID**: Unrelated topics (weather, sports, politics, etc.)
- **META**: Legacy fallback for backward compatibility
- **PRODUCT**: Legacy product intent for backward compatibility

## 🔄 Three-Tier Fallback System

The improved hybrid classifier implements a robust three-tier fallback strategy:

### **Tier 1: Primary Classifier (HuggingFace)**

- **Purpose**: Fast, accurate classification for most queries
- **When Used**: When confidence ≥ threshold (default: 0.3)
- **Performance**: ~60-80ms, high accuracy
- **Cost**: Free

### **Tier 2: Rule-based Fallback**

- **Purpose**: Reliable pattern matching for edge cases
- **When Used**: When primary confidence < threshold
- **Performance**: ~2-5ms, medium accuracy
- **Cost**: Free

### **Tier 3: OpenAI Fallback**

- **Purpose**: Handle complex, ambiguous, or novel queries
- **When Used**: When both primary and rule-based fail
- **Performance**: ~500-2000ms, highest accuracy
- **Cost**: Paid (API calls)

### **Fallback Logic**

```python
# Simplified flow
if primary_confidence >= threshold:
    return primary_result
elif rule_based_available:
    return rule_based_result
elif openai_available:
    return openai_result
else:
    return clarify_intent  # Last resort
```

## 🛠️ Quick Start

### Basic Usage

```python
from rag.intent_modules import IntentClassifierFactory

# Create a rule-based classifier (fastest, no dependencies)
classifier = IntentClassifierFactory.create("rule_based")

# Classify a user message
result = classifier.classify_intent("show me bedside tables")
print(f"Intent: {result.intent}")
print(f"Confidence: {result.confidence}")
print(f"Method: {result.method}")
```

### Using OpenAI Fine-tuned Model

```python
# Set environment variables
import os
os.environ["FINE_TUNED_MODEL_NAME"] = "ft:gpt-3.5-turbo-0613:your-org:your-model:1234567890"
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Create OpenAI classifier
config = {
    "model_name": os.getenv("FINE_TUNED_MODEL_NAME"),
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0.0
}
classifier = IntentClassifierFactory.create("openai", config)
```

### Using HuggingFace Transformers (DeBERTa)

```python
# Install dependencies first: pip install transformers torch
config = {
    "model_path": "./trained_deberta_model",  # Path to trained DeBERTa model
    "device": "cpu"  # or "cuda" for GPU
}
classifier = IntentClassifierFactory.create("huggingface", config)
```

### Using Improved Hybrid Classifier with OpenAI Fallback (Recommended)

```python
# Create improved hybrid classifier with three-tier fallback
hybrid_config = {
    "confidence_threshold": 0.5,
    "primary_classifier": "huggingface",
    "fallback_classifier": "rule_based",
    "openai_fallback": True,  # Enable OpenAI as third fallback
    "enable_intent_specific_rules": True,
    "implementation_configs": {
        "huggingface": {
            "model_path": "./trained_deberta_model",
            "device": "cpu"
        },
        "rule_based": {
            "similarity_threshold": 0.5
        },
        "openai": {
            "model_name": os.getenv("FINE_TUNED_MODEL_NAME"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.0,
            "max_tokens": 10
        }
    }
}
classifier = IntentClassifierFactory.create("improved_hybrid", hybrid_config)
```

**Three-Tier Fallback Strategy:**

1. **Primary**: HuggingFace (fast, accurate, free)
2. **Fallback**: Rule-based (very fast, reliable patterns)
3. **Third Fallback**: OpenAI (high accuracy, handles complex cases)

## ⚙️ Configuration

### Environment Variables

```bash
# General settings
INTENT_IMPLEMENTATION=hybrid
INTENT_CONFIDENCE_THRESHOLD=0.7
INTENT_FALLBACK_STRATEGY=best_confidence

# OpenAI settings
FINE_TUNED_MODEL_NAME=ft:gpt-3.5-turbo-0613:your-org:your-model:1234567890
OPENAI_API_KEY=your-api-key
OPENAI_TEMPERATURE=0.0
OPENAI_MAX_TOKENS=10

# HuggingFace settings
HF_MODEL_NAME=distilbert-base-uncased
HF_NUM_LABELS=4
HF_MAX_LENGTH=512
HF_DEVICE=cpu

# Rule-based settings
RULE_SIMILARITY_THRESHOLD=0.5

# Hybrid settings
HYBRID_IMPLEMENTATIONS=openai,huggingface,rule_based
```

### Configuration Management

```python
from rag.intent_modules import IntentClassifierConfig

# Load from environment
config = IntentClassifierConfig.from_environment()

# Validate configuration
issues = config.validate()
if issues:
    print("Configuration issues:", issues)

# Get implementation-specific config
openai_config = config.get_implementation_config("openai")
hybrid_config = config.get_hybrid_config()
```

## 📊 Performance Comparison

| Implementation | Speed      | Accuracy   | Cost     | Dependencies        |
| -------------- | ---------- | ---------- | -------- | ------------------- |
| Rule-based     | ⚡⚡⚡⚡⚡ | ⭐⭐⭐     | Free     | None                |
| HuggingFace    | ⚡⚡⚡⚡   | ⭐⭐⭐⭐⭐ | Free     | transformers, torch |
| OpenAI         | ⚡⚡       | ⭐⭐⭐⭐⭐ | Paid     | openai              |
| Hybrid         | ⚡⚡⚡⚡   | ⭐⭐⭐⭐⭐ | Variable | All above           |
| **Three-Tier** | ⚡⚡⚡⚡   | ⭐⭐⭐⭐⭐ | Variable | All above           |

### Speed Benchmarks (per query)

- **Rule-based**: ~2-5ms
- **HuggingFace**: ~60-80ms (PyTorch CPU)
- **OpenAI**: ~500-2000ms
- **Hybrid**: ~60-100ms (depends on fallback strategy)
- **Three-Tier**: ~60-200ms (depends on which fallback is used)

## 🔧 Advanced Usage

### Custom Rule-based Patterns

```python
from rag.intent_modules import IntentClassifierFactory
from rag.intent_modules.base import IntentType

# Create custom patterns
custom_patterns = {
    IntentType.PRODUCT: {
        "custom_products": ["my special product", "unique item"],
        "brands": ["nike", "adidas", "apple"]
    }
}

config = {
    "patterns": custom_patterns,
    "similarity_threshold": 0.6
}

classifier = IntentClassifierFactory.create("rule_based", config)

# Add patterns dynamically
classifier.add_pattern(IntentType.PRODUCT, "new_category", "new pattern")
```

### Performance Monitoring

```python
# Get performance statistics
stats = classifier.get_performance_stats()
print(f"Average time: {stats['avg_processing_time']*1000:.2f}ms")
print(f"Total queries: {stats['total_queries']}")
print(f"Success rate: {stats['success_rate']:.2f}")

# Get detailed info
info = classifier.get_info()
print(f"Available: {info['available']}")
print(f"Initialized: {info['initialized']}")
```

### Custom Implementations

```python
from rag.intent_modules.base import BaseIntentClassifier, IntentType, ClassificationResult

class CustomClassifier(BaseIntentClassifier):
    def _initialize(self) -> bool:
        # Initialize your custom classifier
        return True

    def classify_intent(self, user_message: str) -> ClassificationResult:
        # Implement your classification logic
        return ClassificationResult(
            intent=IntentType.PRODUCT,
            confidence=0.8,
            method=self.name,
            reasoning="Custom classification",
            scores={IntentType.PRODUCT: 0.8},
            processing_time=0.001
        )

    def is_available(self) -> bool:
        return self._is_initialized

# Register custom implementation
IntentClassifierFactory.register_implementation("custom", CustomClassifier)

# Use custom classifier
classifier = IntentClassifierFactory.create("custom")
```

## 🧪 Testing

Run the comprehensive demo:

```bash
cd backend
python modular_demo.py
```

This will test:

- Individual implementations
- Hybrid classifier
- Configuration management
- Speed comparison
- Performance monitoring

## 🔄 Integration with LangGraph

Update your LangGraph agent to use the modular system:

```python
from rag.intent_modules import IntentClassifierFactory

# Create classifier
classifier = IntentClassifierFactory.create_hybrid({
    "implementations": ["openai", "rule_based"],
    "min_confidence_threshold": 0.7,
    "implementation_configs": {
        "openai": {
            "model_name": os.getenv("FINE_TUNED_MODEL_NAME"),
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    }
})

# Use in your agent
def classify_node(state):
    result = classifier.classify_intent(state["user_message"])
    state["intent"] = result.intent
    state["intent_confidence"] = result.confidence
    return state
```

## 🚨 Error Handling

The system includes robust error handling:

- **Graceful Fallbacks**: If one classifier fails, others are tried
- **Configuration Validation**: Validates settings before initialization
- **Performance Monitoring**: Tracks success rates and processing times
- **Detailed Logging**: Comprehensive logging for debugging

## 📈 Best Practices

1. **Start with Rule-based**: Use rule-based for development and testing
2. **Add OpenAI for Production**: Fine-tune OpenAI model for best accuracy
3. **Use Hybrid for Reliability**: Combine multiple classifiers for robustness
4. **Monitor Performance**: Track processing times and success rates
5. **Validate Configuration**: Always validate config before deployment

## 🔮 Future Enhancements

- [ ] Custom training for HuggingFace models
- [ ] Ensemble voting strategies
- [ ] Real-time model switching
- [ ] A/B testing framework
- [ ] Model performance analytics
- [ ] Auto-scaling configurations

## 🤝 Contributing

To add a new implementation:

1. Create a new class inheriting from `BaseIntentClassifier`
2. Implement required methods
3. Register with `IntentClassifierFactory`
4. Add tests and documentation
5. Update this README

## 📄 License

This module is part of the smart-ai-agent project and follows the same license terms.
