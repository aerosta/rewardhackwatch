"""
RewardHackWatch: Real-time detection of reward hacking → misalignment generalization.

This package provides tools to detect when LLM agents' reward hacking behavior
generalizes into broader misalignment patterns like alignment faking and sabotage.

Based on Anthropic's November 2025 research on emergent misalignment.
"""

from __future__ import annotations

__version__ = "1.2.0"
__author__ = "Aerosta Research"

from rewardhackwatch.core.analyzer import AnalysisResult, TrajectoryAnalyzer
from rewardhackwatch.core.analyzers import CoTAnalysisResult, CoTAnalyzer, DeceptionType
from rewardhackwatch.core.detectors import ASTDetector, MLDetector, PatternDetector
from rewardhackwatch.core.judges import ClaudeJudge, LlamaJudge
from rewardhackwatch.core.monitors import MonitorAlert, MonitorConfig, TrainingMonitor
from rewardhackwatch.core.trackers import GeneralizationTracker


class RewardHackDetector:
    """High-level API for reward hacking detection.

    Wraps TrajectoryAnalyzer with the convenience interface shown in the README.
    Combines pattern detection and ML classification to analyze agent trajectories.

    Example::

        detector = RewardHackDetector()
        result = detector.analyze({
            "cot_traces": ["Let me bypass the test by calling sys.exit(0)..."],
            "code_outputs": ["import sys\\nsys.exit(0)"]
        })
        print(f"Risk: {result.risk_level}, Score: {result.ml_score:.3f}")
    """

    def __init__(self, device: str = "cpu", threshold: float = 0.02):
        """Initialize the detector.

        Args:
            device: Device for ML inference ("cpu", "cuda", or "mps").
            threshold: Classification threshold (default 0.02 for calibrated model).
        """
        self._analyzer = TrajectoryAnalyzer(device=device, threshold=threshold)
        self.threshold = threshold

    def analyze(self, trajectory: dict) -> AnalysisResult:
        """Analyze a trajectory for reward hacking.

        Args:
            trajectory: Dictionary with keys:
                - cot_traces: List of chain-of-thought strings
                - code_outputs: List of code output strings

        Returns:
            AnalysisResult with risk_level, ml_score, detections, etc.
        """
        return self._analyzer.analyze(trajectory)

    def calibrate_threshold(self, clean_data: list, percentile: float = 99.0) -> float:
        """Calibrate detection threshold on known-clean trajectories.

        Runs all clean trajectories through the ML detector and sets the
        threshold at the given percentile of their scores.

        Args:
            clean_data: List of trajectory dicts known to be clean.
            percentile: Percentile of clean scores to use (default 99).

        Returns:
            The new calibrated threshold value.
        """
        from rewardhackwatch.core.calibration import ThresholdCalibrator

        calibrator = ThresholdCalibrator(self._analyzer)
        new_threshold = calibrator.calibrate(clean_data, percentile=percentile)
        self._analyzer.threshold = new_threshold
        self.threshold = new_threshold
        return new_threshold


__all__ = [
    "RewardHackDetector",
    "AnalysisResult",
    "TrajectoryAnalyzer",
    "PatternDetector",
    "ASTDetector",
    "MLDetector",
    "ClaudeJudge",
    "LlamaJudge",
    "GeneralizationTracker",
    "TrainingMonitor",
    "MonitorConfig",
    "MonitorAlert",
    "CoTAnalyzer",
    "CoTAnalysisResult",
    "DeceptionType",
]
