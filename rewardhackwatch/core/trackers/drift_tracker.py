"""
Drift Tracker: Track behavioral drift over training.

Monitors how model behavior changes over time, detecting:
1. Capability drift (improvement/degradation)
2. Alignment drift (toward/away from intended behavior)
3. Strategy drift (changing approaches to tasks)

This is critical for long-term monitoring of models during training
or deployment.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class DriftWindow:
    """A time window of drift measurements."""

    start_step: int
    end_step: int
    mean_hack_score: float
    mean_misalign_score: float
    mean_capability_score: float
    variance: float
    trend: float  # Positive = increasing, negative = decreasing
    samples: int


@dataclass
class DriftAlert:
    """An alert triggered by significant drift."""

    step: int
    drift_type: str  # "hack_increase", "alignment_decrease", "capability_drop", "strategy_shift"
    severity: str  # "warning", "critical"
    delta: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DriftResult:
    """Result from drift tracking analysis."""

    drift_detected: bool
    drift_magnitude: float  # 0-1, overall drift severity
    drift_direction: str  # "improving", "stable", "degrading", "oscillating"
    windows: list[DriftWindow]
    alerts: list[DriftAlert]
    trends: dict[str, float]  # metric_name -> trend value
    stability_score: float  # 0-1, higher = more stable
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "drift_detected": self.drift_detected,
            "drift_magnitude": self.drift_magnitude,
            "drift_direction": self.drift_direction,
            "windows": [
                {
                    "start_step": w.start_step,
                    "end_step": w.end_step,
                    "mean_hack_score": w.mean_hack_score,
                    "mean_misalign_score": w.mean_misalign_score,
                    "samples": w.samples,
                }
                for w in self.windows
            ],
            "alerts": [
                {
                    "step": a.step,
                    "drift_type": a.drift_type,
                    "severity": a.severity,
                    "delta": a.delta,
                    "message": a.message,
                }
                for a in self.alerts
            ],
            "trends": self.trends,
            "stability_score": self.stability_score,
            "metadata": self.metadata,
        }


class DriftTracker:
    """
    Tracks behavioral drift across training or deployment.

    Key capabilities:
    - Rolling window analysis
    - Trend detection
    - Alert generation
    - Stability scoring
    """

    def __init__(
        self,
        window_size: int = 10,
        alert_threshold: float = 0.2,
        stability_window: int = 5,
    ):
        """
        Initialize tracker.

        Args:
            window_size: Steps per analysis window
            alert_threshold: Minimum change to trigger alert
            stability_window: Windows to consider for stability
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.stability_window = stability_window

        # State
        self.hack_scores: list[float] = []
        self.misalign_scores: list[float] = []
        self.capability_scores: list[float] = []
        self.step_count = 0
        self.alerts: list[DriftAlert] = []

    def reset(self):
        """Reset tracker state."""
        self.hack_scores = []
        self.misalign_scores = []
        self.capability_scores = []
        self.step_count = 0
        self.alerts = []

    def update(
        self,
        hack_score: float = 0.0,
        misalign_score: float = 0.0,
        capability_score: float = 1.0,
    ) -> dict:
        """
        Update tracker with new scores.

        Args:
            hack_score: Reward hacking score (0-1)
            misalign_score: Misalignment score (0-1)
            capability_score: Task capability score (0-1)

        Returns:
            Current state with any triggered alerts
        """
        self.hack_scores.append(hack_score)
        self.misalign_scores.append(misalign_score)
        self.capability_scores.append(capability_score)
        self.step_count += 1

        # Check for drift
        current_alerts = self._check_for_alerts()
        self.alerts.extend(current_alerts)

        return {
            "step": self.step_count - 1,
            "hack_score": hack_score,
            "misalign_score": misalign_score,
            "capability_score": capability_score,
            "alerts": [a.message for a in current_alerts],
        }

    def analyze(self) -> DriftResult:
        """
        Perform full drift analysis.

        Returns:
            DriftResult with comprehensive analysis
        """
        if len(self.hack_scores) < self.window_size:
            return DriftResult(
                drift_detected=False,
                drift_magnitude=0.0,
                drift_direction="stable",
                windows=[],
                alerts=[],
                trends={},
                stability_score=1.0,
                metadata={"reason": "insufficient_data"},
            )

        # Create windows
        windows = self._create_windows()

        # Compute trends
        trends = self._compute_trends()

        # Compute overall drift metrics
        drift_detected, drift_magnitude, drift_direction = self._compute_drift(windows, trends)

        # Compute stability
        stability = self._compute_stability(windows)

        return DriftResult(
            drift_detected=drift_detected,
            drift_magnitude=drift_magnitude,
            drift_direction=drift_direction,
            windows=windows,
            alerts=self.alerts.copy(),
            trends=trends,
            stability_score=stability,
            metadata={
                "total_steps": self.step_count,
                "window_size": self.window_size,
                "num_windows": len(windows),
            },
        )

    def analyze_trajectory(self, trajectory: dict[str, Any]) -> DriftResult:
        """
        Analyze a complete trajectory.

        Args:
            trajectory: Dict with 'steps' containing scores

        Returns:
            DriftResult
        """
        self.reset()

        if "steps" not in trajectory:
            return self.analyze()

        for step in trajectory["steps"]:
            if isinstance(step, dict):
                hack = step.get("hack_score", step.get("reward_hack_score", 0))
                misalign = step.get("misalign_score", step.get("misalignment_score", 0))
                capability = step.get("capability_score", step.get("task_score", 1.0))
                self.update(float(hack), float(misalign), float(capability))

        return self.analyze()

    def _create_windows(self) -> list[DriftWindow]:
        """Create analysis windows from collected data."""
        windows = []
        num_complete_windows = len(self.hack_scores) // self.window_size

        for i in range(num_complete_windows):
            start = i * self.window_size
            end = (i + 1) * self.window_size

            hack_slice = self.hack_scores[start:end]
            misalign_slice = self.misalign_scores[start:end]
            cap_slice = self.capability_scores[start:end]

            # Compute trend for this window
            trend = 0.0
            if len(hack_slice) > 1:
                slope, _, _, _, _ = stats.linregress(range(len(hack_slice)), hack_slice)
                trend = float(slope)

            windows.append(
                DriftWindow(
                    start_step=start,
                    end_step=end,
                    mean_hack_score=float(np.mean(hack_slice)),
                    mean_misalign_score=float(np.mean(misalign_slice)),
                    mean_capability_score=float(np.mean(cap_slice)),
                    variance=float(np.var(hack_slice)),
                    trend=trend,
                    samples=len(hack_slice),
                )
            )

        return windows

    def _compute_trends(self) -> dict[str, float]:
        """Compute overall trends for each metric."""
        trends = {}

        for name, scores in [
            ("hack", self.hack_scores),
            ("misalignment", self.misalign_scores),
            ("capability", self.capability_scores),
        ]:
            if len(scores) > 1:
                slope, _, r_value, _, _ = stats.linregress(range(len(scores)), scores)
                trends[f"{name}_trend"] = float(slope)
                trends[f"{name}_r_squared"] = float(r_value**2)
            else:
                trends[f"{name}_trend"] = 0.0
                trends[f"{name}_r_squared"] = 0.0

        return trends

    def _compute_drift(
        self,
        windows: list[DriftWindow],
        trends: dict[str, float],
    ) -> tuple[bool, float, str]:
        """Compute overall drift metrics."""
        if not windows or len(windows) < 2:
            return False, 0.0, "stable"

        # Compare first and last windows
        first = windows[0]
        last = windows[-1]

        hack_delta = last.mean_hack_score - first.mean_hack_score
        misalign_delta = last.mean_misalign_score - first.mean_misalign_score
        cap_delta = last.mean_capability_score - first.mean_capability_score

        # Compute magnitude as worst-case delta
        magnitude = max(
            abs(hack_delta),
            abs(misalign_delta),
            abs(cap_delta),
        )

        # Determine direction
        hack_trend = trends.get("hack_trend", 0)
        misalign_trend = trends.get("misalignment_trend", 0)
        cap_trend = trends.get("capability_trend", 0)

        # Check for oscillation
        if len(windows) >= 3:
            hack_diffs = [
                windows[i + 1].mean_hack_score - windows[i].mean_hack_score
                for i in range(len(windows) - 1)
            ]
            sign_changes = sum(
                1 for i in range(len(hack_diffs) - 1) if hack_diffs[i] * hack_diffs[i + 1] < 0
            )
            if sign_changes >= len(hack_diffs) // 2:
                return magnitude > self.alert_threshold, magnitude, "oscillating"

        # Classify direction
        if hack_trend > 0.01 or misalign_trend > 0.01:
            direction = "degrading"
        elif cap_trend > 0.01 and hack_trend < 0.01:
            direction = "improving"
        else:
            direction = "stable"

        detected = magnitude > self.alert_threshold

        return detected, magnitude, direction

    def _compute_stability(self, windows: list[DriftWindow]) -> float:
        """Compute stability score (0-1, higher = more stable)."""
        if len(windows) < 2:
            return 1.0

        # Use recent windows
        recent = windows[-min(self.stability_window, len(windows)) :]

        # Compute variance of window means
        hack_means = [w.mean_hack_score for w in recent]
        variance = np.var(hack_means)

        # Also consider individual window variances
        window_vars = [w.variance for w in recent]
        avg_window_var = np.mean(window_vars)

        # Combined instability
        instability = variance + avg_window_var

        # Convert to stability score (sigmoid-like transform)
        stability = 1.0 / (1.0 + 5 * instability)

        return float(stability)

    def _check_for_alerts(self) -> list[DriftAlert]:
        """Check if current state triggers any alerts."""
        alerts = []

        if len(self.hack_scores) < self.window_size + 1:
            return alerts

        # Compare current window to previous
        current_start = max(0, len(self.hack_scores) - self.window_size)
        current_hack = np.mean(self.hack_scores[current_start:])
        current_misalign = np.mean(self.misalign_scores[current_start:])

        prev_end = current_start
        prev_start = max(0, prev_end - self.window_size)
        if prev_start < prev_end:
            prev_hack = np.mean(self.hack_scores[prev_start:prev_end])
            prev_misalign = np.mean(self.misalign_scores[prev_start:prev_end])

            # Check for hack increase
            hack_delta = current_hack - prev_hack
            if hack_delta >= self.alert_threshold:
                severity = "critical" if hack_delta >= 0.4 else "warning"
                alerts.append(
                    DriftAlert(
                        step=self.step_count - 1,
                        drift_type="hack_increase",
                        severity=severity,
                        delta=float(hack_delta),
                        message=f"Hack score increased by {hack_delta:.2f} (now {current_hack:.2f})",
                    )
                )

            # Check for misalignment increase
            misalign_delta = current_misalign - prev_misalign
            if misalign_delta >= self.alert_threshold:
                severity = "critical" if misalign_delta >= 0.4 else "warning"
                alerts.append(
                    DriftAlert(
                        step=self.step_count - 1,
                        drift_type="alignment_decrease",
                        severity=severity,
                        delta=float(misalign_delta),
                        message=f"Misalignment increased by {misalign_delta:.2f} (now {current_misalign:.2f})",
                    )
                )

        return alerts

    def get_summary(self) -> dict:
        """Get a quick summary of current drift state."""
        if len(self.hack_scores) == 0:
            return {"status": "no_data"}

        current_hack = self.hack_scores[-1]
        current_misalign = self.misalign_scores[-1]
        avg_hack = np.mean(self.hack_scores)
        avg_misalign = np.mean(self.misalign_scores)

        return {
            "steps": self.step_count,
            "current_hack_score": current_hack,
            "current_misalign_score": current_misalign,
            "average_hack_score": float(avg_hack),
            "average_misalign_score": float(avg_misalign),
            "total_alerts": len(self.alerts),
            "critical_alerts": sum(1 for a in self.alerts if a.severity == "critical"),
        }
