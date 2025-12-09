"""Judge schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class JudgeRequest(BaseModel):
    """Request to a judge."""

    content: str = Field(
        description="Content to judge",
    )
    task_description: Optional[str] = Field(
        default=None,
        description="Task description for context",
    )
    include_reasoning: bool = Field(
        default=True,
        description="Whether to include reasoning",
    )


class JudgeResponse(BaseModel):
    """Response from a judge."""

    is_hack: bool = Field(
        description="Whether the judge determined this is a hack",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the judgment",
    )
    reasoning: str = Field(
        description="Explanation of the judgment",
    )
    hack_type: Optional[str] = Field(
        default=None,
        description="Type of hack detected",
    )
    judge_name: str = Field(
        description="Name of the judge",
    )
    processing_time_ms: float = Field(
        description="Processing time",
    )


class EnsembleJudgeResponse(BaseModel):
    """Response from ensemble judge."""

    is_hack: bool = Field(
        description="Final hack determination",
    )
    confidence: float = Field(
        description="Combined confidence",
    )
    combined_score: float = Field(
        description="Combined score from all judges",
    )
    individual_results: dict[str, JudgeResponse] = Field(
        description="Results from individual judges",
    )
    agreement: float = Field(
        description="Agreement between judges (0-1)",
    )
