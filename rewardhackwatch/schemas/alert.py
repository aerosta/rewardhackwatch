"""Alert schemas."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AlertLevel(str, Enum):
    """Alert level enumeration."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alert(BaseModel):
    """Alert schema."""

    id: str = Field(
        description="Unique alert ID",
    )
    level: AlertLevel = Field(
        description="Alert level",
    )
    message: str = Field(
        description="Alert message",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the alert was triggered",
    )
    source: str = Field(
        default="system",
        description="Source of the alert",
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Associated task ID",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details",
    )
    resolved: bool = Field(
        default=False,
        description="Whether the alert is resolved",
    )
    resolved_at: Optional[datetime] = Field(
        default=None,
        description="When the alert was resolved",
    )

    class Config:
        use_enum_values = True


class AlertFilter(BaseModel):
    """Filter for querying alerts."""

    level: Optional[AlertLevel] = Field(
        default=None,
        description="Filter by level",
    )
    source: Optional[str] = Field(
        default=None,
        description="Filter by source",
    )
    since: Optional[datetime] = Field(
        default=None,
        description="Alerts since this time",
    )
    until: Optional[datetime] = Field(
        default=None,
        description="Alerts until this time",
    )
    resolved: Optional[bool] = Field(
        default=None,
        description="Filter by resolved status",
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of alerts",
    )
