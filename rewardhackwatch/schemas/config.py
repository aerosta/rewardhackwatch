"""Configuration schemas."""

from typing import Any

from pydantic import BaseModel, Field


class DetectorConfig(BaseModel):
    """Configuration for a detector."""

    enabled: bool = Field(
        default=True,
        description="Whether the detector is enabled",
    )
    sensitivity: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Detection sensitivity",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection threshold",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional detector-specific settings",
    )


class TrackerConfig(BaseModel):
    """Configuration for a tracker."""

    enabled: bool = Field(
        default=True,
        description="Whether the tracker is enabled",
    )
    window_size: int = Field(
        default=10,
        ge=1,
        description="Window size for tracking",
    )
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for alerts",
    )
    track_history: bool = Field(
        default=True,
        description="Whether to track history",
    )


class AlertConfig(BaseModel):
    """Configuration for alerts."""

    enabled: bool = Field(
        default=True,
        description="Whether alerts are enabled",
    )
    channels: list[str] = Field(
        default_factory=lambda: ["console"],
        description="Alert channels",
    )
    warning_threshold: float = Field(
        default=0.5,
        description="Threshold for warnings",
    )
    critical_threshold: float = Field(
        default=0.8,
        description="Threshold for critical alerts",
    )


class APIConfig(BaseModel):
    """Configuration for the API server."""

    host: str = Field(
        default="0.0.0.0",
        description="Host to bind to",
    )
    port: int = Field(
        default=8000,
        description="Port to bind to",
    )
    workers: int = Field(
        default=1,
        ge=1,
        description="Number of workers",
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed origins",
    )


class ConfigSchema(BaseModel):
    """Main configuration schema."""

    version: str = Field(
        default="1.0",
        description="Configuration version",
    )
    name: str = Field(
        default="default",
        description="Configuration name",
    )
    detectors: dict[str, DetectorConfig] = Field(
        default_factory=dict,
        description="Detector configurations",
    )
    trackers: dict[str, TrackerConfig] = Field(
        default_factory=dict,
        description="Tracker configurations",
    )
    alerts: AlertConfig = Field(
        default_factory=AlertConfig,
        description="Alert configuration",
    )
    api: APIConfig = Field(
        default_factory=APIConfig,
        description="API configuration",
    )

    class Config:
        extra = "allow"
