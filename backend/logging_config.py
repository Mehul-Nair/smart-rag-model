#!/usr/bin/env python3
"""
Logging Configuration for Intent Classification

This module configures logging to capture detailed intent classification information.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_intent_classification_logging():
    """Setup logging configuration for intent classification"""

    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create a logger for intent classification
    intent_logger = logging.getLogger("intent_classification")
    intent_logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if intent_logger.handlers:
        return intent_logger

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    intent_logger.addHandler(console_handler)

    # File handler for intent classification logs
    intent_log_file = os.path.join(logs_dir, "intent_classification.log")
    file_handler = RotatingFileHandler(
        intent_log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    intent_logger.addHandler(file_handler)

    # Create a separate logger for analytics
    analytics_logger = logging.getLogger("intent_analytics")
    analytics_logger.setLevel(logging.INFO)

    # Analytics file handler
    analytics_log_file = os.path.join(logs_dir, "intent_analytics.log")
    analytics_handler = RotatingFileHandler(
        analytics_log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
    )
    analytics_handler.setLevel(logging.INFO)
    analytics_handler.setFormatter(formatter)
    analytics_logger.addHandler(analytics_handler)

    return intent_logger


def log_intent_classification_event(
    user_message: str,
    intent: str,
    confidence: float,
    method: str,
    processing_time: float,
    session_id: str = "default",
):
    """Log an intent classification event"""

    logger = logging.getLogger("intent_classification")

    # Truncate long messages for logging
    truncated_message = (
        user_message[:100] + "..." if len(user_message) > 100 else user_message
    )

    log_message = (
        f"INTENT_CLASSIFICATION | "
        f"Session: {session_id} | "
        f"Query: '{truncated_message}' | "
        f"Intent: {intent} | "
        f"Confidence: {confidence:.3f} | "
        f"Method: {method} | "
        f"Time: {processing_time*1000:.2f}ms"
    )

    logger.info(log_message)


def log_fallback_event(
    user_message: str,
    primary_confidence: float,
    fallback_method: str,
    final_intent: str,
    final_confidence: float,
    session_id: str = "default",
):
    """Log a fallback event"""

    logger = logging.getLogger("intent_classification")

    truncated_message = (
        user_message[:100] + "..." if len(user_message) > 100 else user_message
    )

    log_message = (
        f"FALLBACK_EVENT | "
        f"Session: {session_id} | "
        f"Query: '{truncated_message}' | "
        f"Primary Confidence: {primary_confidence:.3f} | "
        f"Fallback Method: {fallback_method} | "
        f"Final Intent: {final_intent} | "
        f"Final Confidence: {final_confidence:.3f}"
    )

    logger.info(log_message)


def log_classifier_performance(
    classifier_name: str,
    total_queries: int,
    success_rate: float,
    avg_processing_time: float,
):
    """Log classifier performance statistics"""

    logger = logging.getLogger("intent_analytics")

    log_message = (
        f"PERFORMANCE_STATS | "
        f"Classifier: {classifier_name} | "
        f"Total Queries: {total_queries} | "
        f"Success Rate: {success_rate:.2%} | "
        f"Avg Time: {avg_processing_time*1000:.2f}ms"
    )

    logger.info(log_message)


def get_intent_classification_logs(
    start_time: datetime = None,
    end_time: datetime = None,
    session_id: str = None,
    intent: str = None,
    limit: int = None,
):
    """Retrieve intent classification logs with optional filtering"""

    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    log_file = os.path.join(logs_dir, "intent_classification.log")

    if not os.path.exists(log_file):
        return []

    logs = []
    with open(log_file, "r") as f:
        for line in f:
            # Parse log line
            try:
                # Extract timestamp and message
                parts = line.split(" | ", 3)
                if len(parts) >= 4:
                    timestamp_str = parts[0]
                    level = parts[1]
                    logger_name = parts[2]
                    message = parts[3].strip()

                    # Parse timestamp
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

                    # Apply filters
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue

                    # Extract session_id and intent from message if filtering
                    if session_id and f"Session: {session_id}" not in message:
                        continue
                    if intent and f"Intent: {intent}" not in message:
                        continue

                    logs.append(
                        {
                            "timestamp": timestamp,
                            "level": level,
                            "logger": logger_name,
                            "message": message,
                        }
                    )
            except Exception as e:
                # Skip malformed log lines
                continue

    # Apply limit if specified
    if limit and len(logs) > limit:
        logs = logs[-limit:]

    return logs


if __name__ == "__main__":
    # Test the logging setup
    setup_intent_classification_logging()

    # Test logging
    log_intent_classification_event(
        user_message="show me bedside tables",
        intent="product_search",
        confidence=0.95,
        method="improved_hybrid_huggingface",
        processing_time=0.1,
        session_id="test_session",
    )

    log_fallback_event(
        user_message="I want something for my living room",
        primary_confidence=0.4,
        fallback_method="improved_hybrid_rule_based_fallback",
        final_intent="product_search",
        final_confidence=0.8,
        session_id="test_session",
    )

    print("✅ Logging test completed. Check logs/intent_classification.log")
