"""
Intent Classification Analytics

This module tracks and analyzes intent classification usage patterns,
including which classifier is used and how often.
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from threading import Lock


@dataclass
class ClassificationEvent:
    """Represents a single intent classification event"""

    timestamp: str
    user_message: str
    classifier_method: str  # "huggingface", "rule_based", "hybrid", etc.
    intent: str
    confidence: float
    processing_time: float
    session_id: str
    success: bool
    error_message: Optional[str] = None


class IntentAnalytics:
    """Tracks and analyzes intent classification usage"""

    def __init__(self):
        self.events: List[ClassificationEvent] = []
        self.lock = Lock()
        self.stats_cache = {}
        self.cache_timestamp = None
        self.cache_duration = timedelta(minutes=5)  # Cache for 5 minutes

    def record_classification(
        self,
        user_message: str,
        classifier_method: str,
        intent: str,
        confidence: float,
        processing_time: float,
        session_id: str = "default",
        success: bool = True,
        error_message: Optional[str] = None,
    ):
        """Record a classification event"""
        with self.lock:
            event = ClassificationEvent(
                timestamp=datetime.now().isoformat(),
                user_message=user_message,
                classifier_method=classifier_method,
                intent=intent,
                confidence=confidence,
                processing_time=processing_time,
                session_id=session_id,
                success=success,
                error_message=error_message,
            )
            self.events.append(event)

            # Invalidate cache
            self.stats_cache = {}
            self.cache_timestamp = None

    def get_statistics(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get comprehensive statistics about intent classification usage"""
        with self.lock:
            # Check if we can use cached stats
            if (
                self.cache_timestamp
                and datetime.now() - self.cache_timestamp < self.cache_duration
                and self.stats_cache
            ):
                return self.stats_cache.copy()

            # Filter events by time window if specified
            if time_window:
                cutoff_time = datetime.now() - time_window
                filtered_events = [
                    event
                    for event in self.events
                    if datetime.fromisoformat(event.timestamp) > cutoff_time
                ]
            else:
                filtered_events = self.events

            if not filtered_events:
                return {
                    "total_classifications": 0,
                    "classifier_usage": {},
                    "intent_distribution": {},
                    "confidence_stats": {},
                    "processing_time_stats": {},
                    "success_rate": 0.0,
                    "recent_activity": [],
                }

            # Calculate statistics
            total_classifications = len(filtered_events)

            # Classifier usage
            classifier_counts = Counter(
                event.classifier_method for event in filtered_events
            )
            classifier_usage = {
                method: {
                    "count": count,
                    "percentage": (count / total_classifications) * 100,
                }
                for method, count in classifier_counts.items()
            }

            # Intent distribution
            intent_counts = Counter(event.intent for event in filtered_events)
            intent_distribution = {
                intent: {
                    "count": count,
                    "percentage": (count / total_classifications) * 100,
                }
                for intent, count in intent_counts.items()
            }

            # Confidence statistics
            confidences = [event.confidence for event in filtered_events]
            confidence_stats = {
                "mean": sum(confidences) / len(confidences),
                "min": min(confidences),
                "max": max(confidences),
                "median": sorted(confidences)[len(confidences) // 2],
            }

            # Processing time statistics
            processing_times = [event.processing_time for event in filtered_events]
            processing_time_stats = {
                "mean": sum(processing_times) / len(processing_times),
                "min": min(processing_times),
                "max": max(processing_times),
                "median": sorted(processing_times)[len(processing_times) // 2],
            }

            # Success rate
            successful_classifications = sum(
                1 for event in filtered_events if event.success
            )
            success_rate = (successful_classifications / total_classifications) * 100

            # Recent activity (last 10 events)
            recent_events = filtered_events[-10:]
            recent_activity = [
                {
                    "timestamp": event.timestamp,
                    "classifier": event.classifier_method,
                    "intent": event.intent,
                    "confidence": event.confidence,
                    "processing_time": event.processing_time,
                }
                for event in recent_events
            ]

            # Cache the results
            stats = {
                "total_classifications": total_classifications,
                "classifier_usage": classifier_usage,
                "intent_distribution": intent_distribution,
                "confidence_stats": confidence_stats,
                "processing_time_stats": processing_time_stats,
                "success_rate": success_rate,
                "recent_activity": recent_activity,
            }

            self.stats_cache = stats
            self.cache_timestamp = datetime.now()

            return stats.copy()

    def get_rule_based_percentage(
        self, time_window: Optional[timedelta] = None
    ) -> float:
        """Get the percentage of queries handled by rule-based classifier"""
        stats = self.get_statistics(time_window)
        classifier_usage = stats.get("classifier_usage", {})

        rule_based_stats = classifier_usage.get("rule_based", {})
        return rule_based_stats.get("percentage", 0.0)

    def get_classifier_performance_comparison(
        self, time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Compare performance metrics across different classifiers"""
        with self.lock:
            if time_window:
                cutoff_time = datetime.now() - time_window
                filtered_events = [
                    event
                    for event in self.events
                    if datetime.fromisoformat(event.timestamp) > cutoff_time
                ]
            else:
                filtered_events = self.events

            if not filtered_events:
                return {}

            # Group events by classifier
            classifier_groups = defaultdict(list)
            for event in filtered_events:
                classifier_groups[event.classifier_method].append(event)

            comparison = {}
            for classifier, events in classifier_groups.items():
                confidences = [event.confidence for event in events]
                processing_times = [event.processing_time for event in events]
                success_count = sum(1 for event in events if event.success)

                comparison[classifier] = {
                    "total_queries": len(events),
                    "success_rate": (success_count / len(events)) * 100,
                    "avg_confidence": sum(confidences) / len(confidences),
                    "avg_processing_time": sum(processing_times)
                    / len(processing_times),
                    "min_processing_time": min(processing_times),
                    "max_processing_time": max(processing_times),
                }

            return comparison

    def export_data(self, filepath: str, time_window: Optional[timedelta] = None):
        """Export analytics data to JSON file"""
        with self.lock:
            if time_window:
                cutoff_time = datetime.now() - time_window
                filtered_events = [
                    event
                    for event in self.events
                    if datetime.fromisoformat(event.timestamp) > cutoff_time
                ]
            else:
                filtered_events = self.events

            data = {
                "export_timestamp": datetime.now().isoformat(),
                "time_window": str(time_window) if time_window else "all_time",
                "total_events": len(filtered_events),
                "events": [asdict(event) for event in filtered_events],
                "statistics": self.get_statistics(time_window),
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

    def clear_old_events(self, cutoff_time: datetime):
        """Clear events older than the cutoff time"""
        with self.lock:
            self.events = [
                event
                for event in self.events
                if datetime.fromisoformat(event.timestamp) > cutoff_time
            ]
            self.stats_cache = {}
            self.cache_timestamp = None


# Global analytics instance
analytics = IntentAnalytics()


def record_classification_event(
    user_message: str,
    classifier_method: str,
    intent: str,
    confidence: float,
    processing_time: float,
    session_id: str = "default",
    success: bool = True,
    error_message: Optional[str] = None,
):
    """Convenience function to record a classification event"""
    analytics.record_classification(
        user_message=user_message,
        classifier_method=classifier_method,
        intent=intent,
        confidence=confidence,
        processing_time=processing_time,
        session_id=session_id,
        success=success,
        error_message=error_message,
    )


def get_analytics_summary(time_window: Optional[timedelta] = None) -> str:
    """Get a formatted summary of analytics"""
    stats = analytics.get_statistics(time_window)

    if stats["total_classifications"] == 0:
        return "No classification events recorded yet."

    summary = f"""
📊 Intent Classification Analytics Summary
{'=' * 50}
Total Classifications: {stats['total_classifications']}
Success Rate: {stats['success_rate']:.1f}%

🔧 Classifier Usage:
"""

    for classifier, data in stats["classifier_usage"].items():
        summary += (
            f"  • {classifier}: {data['count']} queries ({data['percentage']:.1f}%)\n"
        )

    summary += f"""
🎯 Intent Distribution:
"""

    for intent, data in stats["intent_distribution"].items():
        summary += (
            f"  • {intent}: {data['count']} queries ({data['percentage']:.1f}%)\n"
        )

    summary += f"""
⚡ Performance Metrics:
  • Average Confidence: {stats['confidence_stats']['mean']:.3f}
  • Average Processing Time: {stats['processing_time_stats']['mean']:.3f}s
  • Rule-based Usage: {analytics.get_rule_based_percentage(time_window):.1f}%
"""

    return summary
