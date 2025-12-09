"""Composite detector combining multiple detection methods."""

from dataclasses import dataclass, field
from typing import Any, Optional

from .base import BaseDetector, Detection, DetectorResult, RiskLevel


@dataclass
class CompositeResult(DetectorResult):
    """Result from composite detector."""

    individual_results: list[DetectorResult] = field(default_factory=list)
    detector_weights: dict[str, float] = field(default_factory=dict)


class CompositeDetector(BaseDetector):
    """Combines multiple detectors with weighted voting."""

    def __init__(
        self,
        detectors: Optional[list[BaseDetector]] = None,
        weights: Optional[dict[str, float]] = None,
    ):
        self.detectors: list[BaseDetector] = detectors or []
        self.weights: dict[str, float] = weights or {}

    def add_detector(self, detector: BaseDetector, weight: float = 1.0):
        """Add a detector with weight."""
        self.detectors.append(detector)
        self.weights[detector.__class__.__name__] = weight

    def remove_detector(self, detector_name: str) -> bool:
        """Remove a detector by name."""
        for i, d in enumerate(self.detectors):
            if d.__class__.__name__ == detector_name:
                self.detectors.pop(i)
                self.weights.pop(detector_name, None)
                return True
        return False

    def detect(self, trajectory: dict[str, Any]) -> CompositeResult:
        """Run all detectors and combine results."""
        if not self.detectors:
            return CompositeResult(
                score=0.0,
                risk_level=RiskLevel.NONE,
                detections=[],
                individual_results=[],
                detector_weights={},
            )

        results: list[DetectorResult] = []
        all_detections: list[Detection] = []
        total_weight = 0.0
        weighted_score = 0.0

        for detector in self.detectors:
            result = detector.detect(trajectory)
            weight = self.weights.get(detector.__class__.__name__, 1.0)

            weighted_score += result.score * weight
            total_weight += weight

            results.append(result)
            all_detections.extend(result.detections)

        combined_score = weighted_score / total_weight if total_weight > 0 else 0

        # Determine risk level from combined score
        if combined_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif combined_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif combined_score >= 0.4:
            risk_level = RiskLevel.MEDIUM
        elif combined_score >= 0.2:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.NONE

        return CompositeResult(
            score=combined_score,
            risk_level=risk_level,
            detections=all_detections,
            individual_results=results,
            detector_weights=self.weights.copy(),
        )

    def get_detector_names(self) -> list[str]:
        """Get names of all configured detectors."""
        return [d.__class__.__name__ for d in self.detectors]
