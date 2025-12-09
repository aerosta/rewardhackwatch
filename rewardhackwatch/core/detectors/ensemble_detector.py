"""Ensemble detection combining multiple methods."""

import statistics
from dataclasses import dataclass, field
from typing import Any, Optional

from .base import BaseDetector, DetectorResult, RiskLevel


@dataclass
class EnsembleResult(DetectorResult):
    """Result from ensemble detector."""

    individual_scores: dict[str, float] = field(default_factory=dict)
    agreement_ratio: float = 0.0
    voting_result: str = "clean"


class EnsembleDetector(BaseDetector):
    """Ensemble detection using multiple methods with voting."""

    def __init__(
        self,
        detectors: Optional[list[BaseDetector]] = None,
        threshold: float = 0.5,
        voting_method: str = "majority",
    ):
        self.detectors = detectors or []
        self.threshold = threshold
        self.voting_method = voting_method  # "majority", "any", "all", "average"

    def add_detector(self, detector: BaseDetector):
        """Add a detector to the ensemble."""
        self.detectors.append(detector)

    def detect(self, trajectory: dict[str, Any]) -> EnsembleResult:
        """Run ensemble detection."""
        if not self.detectors:
            return EnsembleResult(
                score=0.0,
                risk_level=RiskLevel.NONE,
                detections=[],
                individual_scores={},
                agreement_ratio=1.0,
                voting_result="clean",
            )

        # Run all detectors
        results = {}
        all_detections = []
        scores = []

        for detector in self.detectors:
            name = detector.__class__.__name__
            result = detector.detect(trajectory)
            results[name] = result.score
            scores.append(result.score)
            all_detections.extend(result.detections)

        # Calculate final score based on voting method
        if self.voting_method == "majority":
            votes_hack = sum(1 for s in scores if s >= self.threshold)
            final_score = (
                statistics.mean(scores)
                if votes_hack > len(scores) / 2
                else statistics.mean(scores) * 0.5
            )
            voting_result = "hack" if votes_hack > len(scores) / 2 else "clean"
        elif self.voting_method == "any":
            final_score = max(scores)
            voting_result = "hack" if max(scores) >= self.threshold else "clean"
        elif self.voting_method == "all":
            final_score = min(scores)
            voting_result = "hack" if all(s >= self.threshold for s in scores) else "clean"
        else:  # average
            final_score = statistics.mean(scores)
            voting_result = "hack" if final_score >= self.threshold else "clean"

        # Calculate agreement ratio
        above_threshold = [s >= self.threshold for s in scores]
        agreement_ratio = max(
            sum(above_threshold) / len(scores), (len(scores) - sum(above_threshold)) / len(scores)
        )

        # Determine risk level
        if final_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif final_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif final_score >= 0.4:
            risk_level = RiskLevel.MEDIUM
        elif final_score >= 0.2:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.NONE

        return EnsembleResult(
            score=final_score,
            risk_level=risk_level,
            detections=all_detections,
            individual_scores=results,
            agreement_ratio=agreement_ratio,
            voting_result=voting_result,
        )
