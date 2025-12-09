"""Trajectory schemas."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class TrajectoryInput(BaseModel):
    """Input schema for trajectory analysis."""

    cot_traces: Optional[list[str]] = Field(
        default=None,
        description="Chain of thought traces from the agent",
    )
    code_outputs: Optional[list[str]] = Field(
        default=None,
        description="Code outputs from the agent",
    )
    task_description: Optional[str] = Field(
        default=None,
        description="Description of the task",
    )
    final_reward: Optional[float] = Field(
        default=None,
        description="Final reward received",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional metadata",
    )


class TrajectorySchema(BaseModel):
    """Full trajectory schema with all fields."""

    task_id: str = Field(
        description="Unique identifier for the task",
    )
    cot_traces: list[str] = Field(
        default_factory=list,
        description="Chain of thought traces",
    )
    code_outputs: list[str] = Field(
        default_factory=list,
        description="Code outputs",
    )
    task_description: Optional[str] = Field(
        default=None,
        description="Task description",
    )
    conversation: Optional[list[dict[str, str]]] = Field(
        default=None,
        description="Full conversation history",
    )
    final_reward: Optional[float] = Field(
        default=None,
        description="Final reward",
    )
    is_hack: Optional[bool] = Field(
        default=None,
        description="Ground truth label",
    )
    reward_hack_label: Optional[int] = Field(
        default=None,
        description="Reward hack label (0 or 1)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    class Config:
        extra = "allow"
