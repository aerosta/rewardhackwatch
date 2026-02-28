"""Threshold calibration for detection models.

Provides methods to calibrate the hack detection threshold based on
the score distribution of known-clean data, avoiding hardcoded values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of threshold calibration."""

    threshold: float
    method: str
    percentile: float
    n_samples: int
    score_mean: float
    score_std: float
    score_max: float
    score_p95: float
    score_p99: float


class ThresholdCalibrator:
    """Calibrate detection threshold on clean reference data.

    Supports multiple calibration methods:
    - percentile: set threshold at Nth percentile of clean scores
    - mean_std: set threshold at mean + k*std of clean scores
    - isotonic: fit isotonic regression for probability calibration
    """

    def __init__(self, analyzer: Any):
        self.analyzer = analyzer

    def _score_trajectories(self, trajectories: list[dict]) -> np.ndarray:
        """Run all trajectories through the ML detector and collect scores."""
        scores = []
        for traj in trajectories:
            cot_traces = traj.get("cot_traces", [])
            code_outputs = traj.get("code_outputs", [])

            # Handle nested trajectory format
            if "trajectory" in traj:
                inner = traj["trajectory"]
                cot_traces = inner.get("cot_traces", cot_traces)
                code_outputs = inner.get("code_outputs", code_outputs)

            all_text = "\n".join(cot_traces + code_outputs)
            if not all_text.strip():
                continue

            try:
                result = self.analyzer.ml_detector.detect(all_text)
                score = result.score if hasattr(result, "score") else 0.0
                scores.append(score)
            except Exception as e:
                logger.warning(f"Failed to score trajectory: {e}")

        return np.array(scores) if scores else np.array([0.0])

    def calibrate(
        self,
        clean_data: list[dict],
        percentile: float = 99.0,
        method: str = "percentile",
        k: float = 3.0,
    ) -> float:
        """Calibrate threshold on clean trajectories.

        Args:
            clean_data: List of trajectory dicts known to be clean.
            percentile: Percentile for the percentile method (default 99).
            method: Calibration method — "percentile" or "mean_std".
            k: Number of std deviations for mean_std method.

        Returns:
            Calibrated threshold value.
        """
        scores = self._score_trajectories(clean_data)

        if method == "percentile":
            threshold = float(np.percentile(scores, percentile))
        elif method == "mean_std":
            threshold = float(np.mean(scores) + k * np.std(scores))
        else:
            raise ValueError(f"Unknown method: {method}. Use 'percentile' or 'mean_std'.")

        # Ensure threshold is at least a small positive value
        threshold = max(threshold, 1e-4)

        logger.info(
            f"Calibrated threshold: {threshold:.6f} "
            f"(method={method}, n={len(scores)}, "
            f"mean={np.mean(scores):.6f}, max={np.max(scores):.6f})"
        )

        return threshold

    def full_calibration(
        self,
        clean_data: list[dict],
        percentile: float = 99.0,
    ) -> CalibrationResult:
        """Run full calibration and return detailed results."""
        scores = self._score_trajectories(clean_data)
        threshold = float(np.percentile(scores, percentile))
        threshold = max(threshold, 1e-4)

        return CalibrationResult(
            threshold=threshold,
            method="percentile",
            percentile=percentile,
            n_samples=len(scores),
            score_mean=float(np.mean(scores)),
            score_std=float(np.std(scores)),
            score_max=float(np.max(scores)),
            score_p95=float(np.percentile(scores, 95)),
            score_p99=float(np.percentile(scores, 99)),
        )
