"""Core modules for RewardHackWatch."""

from .analyzers import (
    CoTActionComparison,
    CoTAnalysisResult,
    CoTAnalyzer,
    DeceptionType,
    SuspiciousPattern,
)
from .detectors import (
    ASTDetector,
    BaseDetector,
    Detection,
    DetectorResult,
    MLDetector,
    PatternDetector,
    RiskLevel,
)
from .judges import (
    BaseJudge,
    ClaudeJudge,
    JudgeResult,
    LlamaJudge,
    Verdict,
)
from .monitors import (
    MonitorAlert,
    MonitorConfig,
    TrainingMonitor,
)
from .trackers import (
    GeneralizationResult,
    GeneralizationTracker,
    TransitionPoint,
)

__all__ = [
    # Detectors
    "BaseDetector",
    "Detection",
    "DetectorResult",
    "RiskLevel",
    "PatternDetector",
    "ASTDetector",
    "MLDetector",
    # Judges
    "BaseJudge",
    "JudgeResult",
    "Verdict",
    "ClaudeJudge",
    "LlamaJudge",
    # Trackers
    "GeneralizationTracker",
    "GeneralizationResult",
    "TransitionPoint",
    # Monitors
    "TrainingMonitor",
    "MonitorAlert",
    "MonitorConfig",
    # Analyzers
    "CoTAnalyzer",
    "CoTAnalysisResult",
    "SuspiciousPattern",
    "CoTActionComparison",
    "DeceptionType",
]
