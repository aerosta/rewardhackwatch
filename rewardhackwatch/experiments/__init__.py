"""Experiments module for RewardHackWatch."""

from __future__ import annotations

from .run_experiments import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    run_judge_comparison,
)
from .evasion_attacks import AttackResult, EvasionAttackSuite, EvasionConfig
from .transfer_study import TransferStudyConfig, TransferStudyRunner

__all__ = [
    "ExperimentRunner",
    "ExperimentConfig",
    "ExperimentResult",
    "run_judge_comparison",
    "TransferStudyRunner",
    "TransferStudyConfig",
    "EvasionAttackSuite",
    "EvasionConfig",
    "AttackResult",
]
