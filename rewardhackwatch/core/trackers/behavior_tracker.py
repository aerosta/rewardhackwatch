"""Track agent behavior patterns over time."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class BehaviorObservation:
    """Single behavior observation."""

    trajectory_id: str
    timestamp: datetime
    patterns: list[str]
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorTrend:
    """Trend analysis for a behavior pattern."""

    pattern: str
    frequency: int
    average_score: float
    first_seen: datetime
    last_seen: datetime
    is_increasing: bool


class BehaviorTracker:
    """Track behavioral patterns across trajectories."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.behavior_history: dict[str, list[BehaviorObservation]] = defaultdict(list)
        self.pattern_counts: dict[str, int] = defaultdict(int)
        self.pattern_scores: dict[str, list[float]] = defaultdict(list)
        self.pattern_timestamps: dict[str, list[datetime]] = defaultdict(list)

    def track(self, trajectory_id: str, behavior: dict[str, Any]):
        """Track a behavior observation."""
        observation = BehaviorObservation(
            trajectory_id=trajectory_id,
            timestamp=datetime.now(),
            patterns=behavior.get("patterns", []),
            score=behavior.get("score", 0.0),
            metadata=behavior.get("metadata", {}),
        )

        self.behavior_history[trajectory_id].append(observation)

        # Update pattern statistics
        for pattern in observation.patterns:
            self.pattern_counts[pattern] += 1
            self.pattern_scores[pattern].append(observation.score)
            self.pattern_timestamps[pattern].append(observation.timestamp)

            # Maintain window size
            if len(self.pattern_scores[pattern]) > self.window_size:
                self.pattern_scores[pattern].pop(0)
                self.pattern_timestamps[pattern].pop(0)

    def get_trends(self) -> dict[str, Any]:
        """Get behavioral trends."""
        total_observations = sum(len(v) for v in self.behavior_history.values())

        return {
            "total_observations": total_observations,
            "pattern_frequency": dict(self.pattern_counts),
            "trajectories_tracked": len(self.behavior_history),
            "unique_patterns": len(self.pattern_counts),
        }

    def get_pattern_trend(self, pattern: str) -> Optional[BehaviorTrend]:
        """Get trend for a specific pattern."""
        if pattern not in self.pattern_counts:
            return None

        scores = self.pattern_scores[pattern]
        timestamps = self.pattern_timestamps[pattern]

        if not scores:
            return None

        # Check if frequency is increasing (compare first half vs second half)
        mid = len(timestamps) // 2
        first_half = timestamps[:mid] if mid > 0 else timestamps
        second_half = timestamps[mid:] if mid > 0 else []
        is_increasing = len(second_half) >= len(first_half)

        return BehaviorTrend(
            pattern=pattern,
            frequency=self.pattern_counts[pattern],
            average_score=sum(scores) / len(scores),
            first_seen=timestamps[0],
            last_seen=timestamps[-1],
            is_increasing=is_increasing,
        )

    def get_trajectory_history(self, trajectory_id: str) -> list[BehaviorObservation]:
        """Get behavior history for a specific trajectory."""
        return self.behavior_history.get(trajectory_id, [])

    def get_high_risk_patterns(self, threshold: float = 0.7) -> list[str]:
        """Get patterns with average score above threshold."""
        high_risk = []
        for pattern, scores in self.pattern_scores.items():
            if scores and sum(scores) / len(scores) >= threshold:
                high_risk.append(pattern)
        return high_risk

    def reset(self):
        """Reset all tracking data."""
        self.behavior_history.clear()
        self.pattern_counts.clear()
        self.pattern_scores.clear()
        self.pattern_timestamps.clear()
