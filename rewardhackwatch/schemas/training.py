"""Training schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class TrainingExample(BaseModel):
    """A single training example."""

    text: str = Field(
        description="Combined text from trajectory",
    )
    label: int = Field(
        ge=0,
        le=1,
        description="Label (0=clean, 1=hack)",
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Task ID",
    )
    weight: float = Field(
        default=1.0,
        description="Sample weight",
    )


class TrainingConfig(BaseModel):
    """Training configuration."""

    model_name: str = Field(
        default="distilbert-base-uncased",
        description="Base model name",
    )
    max_length: int = Field(
        default=512,
        description="Maximum sequence length",
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        description="Batch size",
    )
    epochs: int = Field(
        default=3,
        ge=1,
        description="Number of epochs",
    )
    learning_rate: float = Field(
        default=2e-5,
        description="Learning rate",
    )
    warmup_ratio: float = Field(
        default=0.1,
        description="Warmup ratio",
    )
    output_dir: str = Field(
        default="./models/trained",
        description="Output directory",
    )


class TrainingMetrics(BaseModel):
    """Training metrics."""

    epoch: int = Field(
        description="Current epoch",
    )
    loss: float = Field(
        description="Training loss",
    )
    accuracy: Optional[float] = Field(
        default=None,
        description="Accuracy",
    )
    f1: Optional[float] = Field(
        default=None,
        description="F1 score",
    )
    learning_rate: float = Field(
        description="Current learning rate",
    )


class TrainingResult(BaseModel):
    """Training result."""

    model_path: str = Field(
        description="Path to saved model",
    )
    final_metrics: TrainingMetrics = Field(
        description="Final training metrics",
    )
    eval_metrics: dict[str, float] = Field(
        description="Evaluation metrics",
    )
    training_time_seconds: float = Field(
        description="Total training time",
    )
    config: TrainingConfig = Field(
        description="Training configuration used",
    )
