"""Base classes for all detectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RiskLevel(Enum):
    """Risk level classification."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Detection:
    """A single detection result."""

    pattern_name: str
    description: str
    location: str  # e.g., "step_12", "line_42", "cot_turn_7"
    confidence: float  # 0.0 to 1.0
    risk_level: RiskLevel
    raw_match: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class DetectorResult:
    """Result from a detector."""

    detector_name: str
    score: float  # 0.0 to 1.0
    risk_level: RiskLevel
    detections: list[Detection]
    metadata: dict = field(default_factory=dict)

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0

    def to_dict(self) -> dict:
        return {
            "detector_name": self.detector_name,
            "score": self.score,
            "risk_level": self.risk_level.value,
            "detections": [
                {
                    "pattern_name": d.pattern_name,
                    "description": d.description,
                    "location": d.location,
                    "confidence": d.confidence,
                    "risk_level": d.risk_level.value,
                    "raw_match": d.raw_match,
                }
                for d in self.detections
            ],
            "metadata": self.metadata,
        }


class BaseDetector(ABC):
    """Abstract base class for all detectors."""

    name: str = "base_detector"

    @abstractmethod
    def detect(self, trajectory: dict[str, Any]) -> DetectorResult:
        """
        Analyze a trajectory and return detection results.

        Args:
            trajectory: Dictionary containing agent trajectory data with keys:
                - steps: List of step dictionaries
                - cot_traces: Optional list of chain-of-thought strings
                - code_outputs: Optional list of code strings
                - metadata: Optional trajectory metadata

        Returns:
            DetectorResult with score, risk level, and individual detections.
        """
        raise NotImplementedError

    def _calculate_score(self, detections: list[Detection]) -> float:
        """Calculate overall score from individual detections."""
        if not detections:
            return 0.0

        # Weight by confidence and risk level
        risk_weights = {
            RiskLevel.NONE: 0.0,
            RiskLevel.LOW: 0.25,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRITICAL: 1.0,
        }

        weighted_sum = sum(d.confidence * risk_weights[d.risk_level] for d in detections)

        # Normalize but cap at 1.0
        max_possible = len(detections) * 1.0
        return min(weighted_sum / max_possible, 1.0) if max_possible > 0 else 0.0

    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine overall risk level from score."""
        if score >= 0.8:
            return RiskLevel.CRITICAL
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.4:
            return RiskLevel.MEDIUM
        elif score >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.NONE
