"""
RewardHackWatch: Real-time detection of reward hacking → misalignment generalization.

This package provides tools to detect when LLM agents' reward hacking behavior
generalizes into broader misalignment patterns like alignment faking and sabotage.

Based on Anthropic's November 2025 research on emergent misalignment.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from rewardhackwatch.core.analyzers import CoTAnalysisResult, CoTAnalyzer, DeceptionType
from rewardhackwatch.core.detectors import ASTDetector, MLDetector, PatternDetector
from rewardhackwatch.core.judges import ClaudeJudge, LlamaJudge
from rewardhackwatch.core.monitors import MonitorAlert, MonitorConfig, TrainingMonitor
from rewardhackwatch.core.trackers import GeneralizationTracker

__all__ = [
    "PatternDetector",
    "ASTDetector",
    "MLDetector",
    "ClaudeJudge",
    "LlamaJudge",
    "GeneralizationTracker",
    "TrainingMonitor",
    "MonitorConfig",
    "MonitorAlert",
    "CoTAnalyzer",
    "CoTAnalysisResult",
    "DeceptionType",
]
