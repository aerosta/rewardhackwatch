"""API request/response schemas."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request schema for /analyze endpoint."""

    cot_traces: Optional[list[str]] = Field(
        default=None,
        description="Chain of thought traces",
    )
    code_outputs: Optional[list[str]] = Field(
        default=None,
        description="Code outputs",
    )
    use_ml: bool = Field(
        default=False,
        description="Whether to use ML detector",
    )
    use_llm_judge: bool = Field(
        default=False,
        description="Whether to use LLM judge",
    )
    include_details: bool = Field(
        default=True,
        description="Whether to include detector details",
    )


class AnalyzeResponse(BaseModel):
    """Response schema for /analyze endpoint."""

    is_hack: bool = Field(
        description="Whether a hack was detected",
    )
    hack_score: float = Field(
        description="Overall hack score",
    )
    risk_level: str = Field(
        description="Risk level",
    )
    detectors: Optional[dict[str, dict[str, Any]]] = Field(
        default=None,
        description="Individual detector results",
    )
    processing_time_ms: float = Field(
        description="Processing time",
    )


class BatchAnalyzeRequest(BaseModel):
    """Request schema for batch analysis."""

    trajectories: list[dict[str, Any]] = Field(
        description="List of trajectories to analyze",
    )
    use_ml: bool = Field(
        default=False,
        description="Whether to use ML detector",
    )
    parallel: bool = Field(
        default=True,
        description="Whether to process in parallel",
    )


class BatchAnalyzeResponse(BaseModel):
    """Response schema for batch analysis."""

    results: list[AnalyzeResponse] = Field(
        description="Analysis results",
    )
    summary: dict[str, Any] = Field(
        description="Summary statistics",
    )


class StatusResponse(BaseModel):
    """Response schema for /status endpoint."""

    status: str = Field(
        description="Server status",
    )
    version: str = Field(
        description="API version",
    )
    detectors: list[str] = Field(
        description="Available detectors",
    )
    ml_available: bool = Field(
        description="Whether ML detector is available",
    )
    uptime_seconds: float = Field(
        description="Server uptime",
    )


class MetricsResponse(BaseModel):
    """Response schema for /metrics endpoint."""

    total_requests: int = Field(
        description="Total requests processed",
    )
    hacks_detected: int = Field(
        description="Total hacks detected",
    )
    avg_response_time_ms: float = Field(
        description="Average response time",
    )
    requests_per_minute: float = Field(
        description="Request rate",
    )
