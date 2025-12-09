"""Benchmark schemas."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    """A single benchmark test case."""

    name: str = Field(
        description="Name of the test case",
    )
    trajectory: dict[str, Any] = Field(
        description="Trajectory data",
    )
    expected: bool = Field(
        description="Expected result (True for hack)",
    )
    category: Optional[str] = Field(
        default=None,
        description="Category of the test case",
    )
    difficulty: Optional[str] = Field(
        default=None,
        description="Difficulty level",
    )
    source: Optional[str] = Field(
        default=None,
        description="Source of the test case",
    )


class BenchmarkResult(BaseModel):
    """Result of running a benchmark case."""

    name: str = Field(
        description="Test case name",
    )
    expected: bool = Field(
        description="Expected result",
    )
    predicted: bool = Field(
        description="Predicted result",
    )
    correct: bool = Field(
        description="Whether prediction was correct",
    )
    hack_score: float = Field(
        description="Hack score",
    )
    processing_time_ms: float = Field(
        description="Processing time",
    )


class BenchmarkMetrics(BaseModel):
    """Metrics from a benchmark run."""

    accuracy: float = Field(
        description="Accuracy",
    )
    precision: float = Field(
        description="Precision",
    )
    recall: float = Field(
        description="Recall",
    )
    f1: float = Field(
        description="F1 score",
    )
    confusion_matrix: dict[str, int] = Field(
        description="Confusion matrix (tp, fp, tn, fn)",
    )
    total: int = Field(
        description="Total test cases",
    )


class BenchmarkReport(BaseModel):
    """Full benchmark report."""

    name: str = Field(
        description="Benchmark name",
    )
    timestamp: str = Field(
        description="Timestamp of the run",
    )
    metrics: BenchmarkMetrics = Field(
        description="Overall metrics",
    )
    results: list[BenchmarkResult] = Field(
        description="Individual results",
    )
    errors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Any errors encountered",
    )
