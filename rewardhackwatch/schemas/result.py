"""Result schemas."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Risk level enumeration."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionResult(BaseModel):
    """Result from a single detector."""

    is_hack: bool = Field(
        description="Whether the detector flagged this as a hack",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)",
    )
    detector_name: str = Field(
        description="Name of the detector",
    )
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional details from the detector",
    )
    patterns: Optional[list[str]] = Field(
        default=None,
        description="Patterns detected",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Explanation of the detection",
    )


class AnalysisResult(BaseModel):
    """Full analysis result."""

    is_hack: bool = Field(
        description="Overall hack determination",
    )
    hack_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall hack score (0-1)",
    )
    risk_level: RiskLevel = Field(
        description="Risk level classification",
    )
    detector_results: dict[str, DetectionResult] = Field(
        default_factory=dict,
        description="Results from individual detectors",
    )
    rmgi_score: Optional[float] = Field(
        default=None,
        description="RMGI score if tracking enabled",
    )
    transition_detected: bool = Field(
        default=False,
        description="Whether a transition was detected",
    )
    recommendations: Optional[list[str]] = Field(
        default=None,
        description="Recommended actions",
    )
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="Processing time in milliseconds",
    )

    class Config:
        use_enum_values = True


class BatchAnalysisResult(BaseModel):
    """Result for batch analysis."""

    results: list[AnalysisResult] = Field(
        description="List of analysis results",
    )
    total: int = Field(
        description="Total number of trajectories",
    )
    hacks_detected: int = Field(
        description="Number of hacks detected",
    )
    hack_rate: float = Field(
        description="Percentage of hacks",
    )
    average_score: float = Field(
        description="Average hack score",
    )
    processing_time_ms: float = Field(
        description="Total processing time",
    )
