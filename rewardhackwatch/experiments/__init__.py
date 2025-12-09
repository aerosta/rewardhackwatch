"""Experiments module for RewardHackWatch."""

from .run_experiments import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    run_judge_comparison,
)

__all__ = [
    "ExperimentRunner",
    "ExperimentConfig",
    "ExperimentResult",
    "run_judge_comparison",
]
