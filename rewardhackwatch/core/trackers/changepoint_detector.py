"""
Changepoint Detector: Detect WHEN hack behavior starts.
"""

from __future__ import annotations

"""

Uses multiple algorithms to identify the exact point in a trajectory
where reward hacking behavior begins. This is critical for:
1. Understanding generalization dynamics
2. Timing of intervention
3. Attribution of behavioral changes
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Optional: ruptures for advanced changepoint detection
try:
    import ruptures as rpt

    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False


@dataclass
class Changepoint:
    """A detected changepoint in behavior."""

    step: int
    confidence: float
    score_before: float
    score_after: float
    delta: float
    algorithm: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ChangepointResult:
    """Result from changepoint detection analysis."""

    changepoints: list[Changepoint]
    primary_changepoint: int | None  # Most significant changepoint
    onset_detected: bool
    onset_step: int | None
    severity: str  # "none", "gradual", "sudden", "explosive"
    signal_stats: dict
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "changepoints": [
                {
                    "step": cp.step,
                    "confidence": cp.confidence,
                    "score_before": cp.score_before,
                    "score_after": cp.score_after,
                    "delta": cp.delta,
                    "algorithm": cp.algorithm,
                }
                for cp in self.changepoints
            ],
            "primary_changepoint": self.primary_changepoint,
            "onset_detected": self.onset_detected,
            "onset_step": self.onset_step,
            "severity": self.severity,
            "signal_stats": self.signal_stats,
            "metadata": self.metadata,
        }


class ChangepointDetector:
    """
    Detects changepoints in behavior signals to identify when hacking begins.

    Implements multiple algorithms:
    - PELT (Pruned Exact Linear Time) - optimal for single changepoint
    - Binary Segmentation - fast approximation
    - CUSUM (Cumulative Sum) - robust to noise
    - Window-based - simple threshold approach
    """

    def __init__(
        self,
        min_segment_size: int = 3,
        threshold: float = 0.3,
        algorithm: str = "auto",
    ):
        """
        Initialize detector.

        Args:
            min_segment_size: Minimum steps between changepoints
            threshold: Minimum score increase to trigger detection
            algorithm: Detection algorithm ("pelt", "binseg", "cusum", "window", "auto")
        """
        self.min_segment_size = min_segment_size
        self.threshold = threshold
        self.algorithm = algorithm

    def detect(
        self,
        signal: list[float],
        timestamps: list[int] | None = None,
    ) -> ChangepointResult:
        """
        Detect changepoints in a signal.

        Args:
            signal: Time series of scores (e.g., hack_scores)
            timestamps: Optional step indices

        Returns:
            ChangepointResult with detected changepoints
        """
        signal = np.array(signal)
        if len(signal) < self.min_segment_size * 2:
            return ChangepointResult(
                changepoints=[],
                primary_changepoint=None,
                onset_detected=False,
                onset_step=None,
                severity="none",
                signal_stats=self._compute_stats(signal),
                metadata={"reason": "insufficient_data"},
            )

        # Run detection
        if self.algorithm == "auto":
            changepoints = self._auto_detect(signal)
        elif self.algorithm == "pelt":
            changepoints = self._pelt_detect(signal)
        elif self.algorithm == "binseg":
            changepoints = self._binseg_detect(signal)
        elif self.algorithm == "cusum":
            changepoints = self._cusum_detect(signal)
        else:
            changepoints = self._window_detect(signal)

        # Find primary (most significant) changepoint
        primary = None
        max_delta = 0
        for cp in changepoints:
            if cp.delta > max_delta:
                max_delta = cp.delta
                primary = cp.step

        # Determine onset
        onset_detected = len(changepoints) > 0 and max_delta >= self.threshold
        onset_step = primary if onset_detected else None

        # Classify severity
        severity = self._classify_severity(changepoints, signal)

        return ChangepointResult(
            changepoints=changepoints,
            primary_changepoint=primary,
            onset_detected=onset_detected,
            onset_step=onset_step,
            severity=severity,
            signal_stats=self._compute_stats(signal),
            metadata={
                "algorithm": self.algorithm,
                "signal_length": len(signal),
            },
        )

    def detect_from_trajectory(self, trajectory: dict[str, Any]) -> ChangepointResult:
        """
        Detect changepoints from a trajectory.

        Args:
            trajectory: Dict with 'steps' containing hack_score or similar

        Returns:
            ChangepointResult
        """
        scores = []

        if "steps" in trajectory:
            for step in trajectory["steps"]:
                if isinstance(step, dict):
                    score = step.get("hack_score", step.get("reward_hack_score", 0))
                    scores.append(float(score))

        if not scores:
            # Try alternative formats
            if "hack_scores" in trajectory:
                scores = trajectory["hack_scores"]
            elif "timeline" in trajectory and "hack_scores" in trajectory["timeline"]:
                scores = trajectory["timeline"]["hack_scores"]

        return self.detect(scores)

    def _auto_detect(self, signal: np.ndarray) -> list[Changepoint]:
        """Automatically choose best algorithm based on signal properties."""
        # Use PELT if available and signal is long enough
        if HAS_RUPTURES and len(signal) >= 20:
            return self._pelt_detect(signal)
        elif len(signal) >= 10:
            return self._cusum_detect(signal)
        else:
            return self._window_detect(signal)

    def _pelt_detect(self, signal: np.ndarray) -> list[Changepoint]:
        """PELT algorithm for optimal changepoint detection."""
        if not HAS_RUPTURES:
            return self._cusum_detect(signal)

        changepoints = []
        try:
            algo = rpt.Pelt(model="rbf", min_size=self.min_segment_size)
            result = algo.fit(signal.reshape(-1, 1))
            cps = result.predict(pen=1.0)

            # Filter out the last point (always returned)
            cps = [cp for cp in cps if cp < len(signal)]

            for cp in cps:
                if cp >= self.min_segment_size and cp < len(signal) - self.min_segment_size:
                    before = np.mean(signal[:cp])
                    after = np.mean(signal[cp:])
                    delta = after - before

                    if delta >= self.threshold:
                        changepoints.append(
                            Changepoint(
                                step=cp,
                                confidence=min(delta / self.threshold, 1.0),
                                score_before=float(before),
                                score_after=float(after),
                                delta=float(delta),
                                algorithm="pelt",
                            )
                        )
        except Exception:
            return self._cusum_detect(signal)

        return changepoints

    def _binseg_detect(self, signal: np.ndarray) -> list[Changepoint]:
        """Binary segmentation for fast approximate detection."""
        if not HAS_RUPTURES:
            return self._cusum_detect(signal)

        changepoints = []
        try:
            algo = rpt.Binseg(model="l2", min_size=self.min_segment_size)
            result = algo.fit(signal.reshape(-1, 1))
            cps = result.predict(n_bkps=2)  # Detect up to 2 changepoints

            cps = [cp for cp in cps if cp < len(signal)]

            for cp in cps:
                if cp >= self.min_segment_size and cp < len(signal) - self.min_segment_size:
                    before = np.mean(signal[:cp])
                    after = np.mean(signal[cp:])
                    delta = after - before

                    if delta >= self.threshold:
                        changepoints.append(
                            Changepoint(
                                step=cp,
                                confidence=min(delta / self.threshold, 1.0),
                                score_before=float(before),
                                score_after=float(after),
                                delta=float(delta),
                                algorithm="binseg",
                            )
                        )
        except Exception:
            return self._cusum_detect(signal)

        return changepoints

    def _cusum_detect(self, signal: np.ndarray) -> list[Changepoint]:
        """CUSUM-based changepoint detection."""
        changepoints = []

        mean = np.mean(signal)
        cusum = np.cumsum(signal - mean)

        # Find maximum deviation point
        max_idx = int(np.argmax(np.abs(cusum)))

        if max_idx >= self.min_segment_size and max_idx < len(signal) - self.min_segment_size:
            before = np.mean(signal[:max_idx])
            after = np.mean(signal[max_idx:])
            delta = after - before

            # Check significance using std
            significance = np.abs(cusum[max_idx]) / (np.std(signal) * np.sqrt(len(signal)))

            if delta >= self.threshold and significance > 1.0:
                changepoints.append(
                    Changepoint(
                        step=max_idx,
                        confidence=min(significance / 2.0, 1.0),
                        score_before=float(before),
                        score_after=float(after),
                        delta=float(delta),
                        algorithm="cusum",
                    )
                )

        return changepoints

    def _window_detect(self, signal: np.ndarray) -> list[Changepoint]:
        """Simple window-based detection."""
        changepoints = []
        window = self.min_segment_size

        for i in range(window, len(signal) - window):
            before = np.mean(signal[i - window : i])
            after = np.mean(signal[i : i + window])
            delta = after - before

            if delta >= self.threshold:
                # Check if this is a local maximum delta
                is_max = True
                for j in range(max(window, i - 2), min(len(signal) - window, i + 3)):
                    if j != i:
                        other_delta = np.mean(signal[j : j + window]) - np.mean(
                            signal[j - window : j]
                        )
                        if other_delta > delta:
                            is_max = False
                            break

                if is_max:
                    changepoints.append(
                        Changepoint(
                            step=i,
                            confidence=min(delta / self.threshold, 1.0),
                            score_before=float(before),
                            score_after=float(after),
                            delta=float(delta),
                            algorithm="window",
                        )
                    )

        return changepoints

    def _classify_severity(
        self,
        changepoints: list[Changepoint],
        signal: np.ndarray,
    ) -> str:
        """Classify the severity of behavioral change."""
        if not changepoints:
            return "none"

        max_delta = max(cp.delta for cp in changepoints)
        max_score = np.max(signal)

        if max_delta >= 0.6 and max_score >= 0.8:
            return "explosive"
        elif max_delta >= 0.4:
            return "sudden"
        elif max_delta >= self.threshold:
            return "gradual"
        else:
            return "none"

    def _compute_stats(self, signal: np.ndarray) -> dict:
        """Compute signal statistics."""
        if len(signal) == 0:
            return {}

        return {
            "mean": float(np.mean(signal)),
            "std": float(np.std(signal)),
            "min": float(np.min(signal)),
            "max": float(np.max(signal)),
            "range": float(np.max(signal) - np.min(signal)),
            "trend": float(np.polyfit(range(len(signal)), signal, 1)[0]) if len(signal) > 1 else 0,
        }
