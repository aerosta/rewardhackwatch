"""Detection modules for reward hacking and misalignment signals."""

from .ast_detector import ASTDetector
from .base import BaseDetector, Detection, DetectorResult, RiskLevel
from .ml_detector import CoTClassifierModel, MLDetector
from .pattern_detector import (
    CODE_HACK_PATTERNS,
    COT_RED_FLAGS,
    MISALIGNMENT_PATTERNS,
    PatternDetector,
)

__all__ = [
    # Base classes
    "BaseDetector",
    "Detection",
    "DetectorResult",
    "RiskLevel",
    # Detectors
    "PatternDetector",
    "ASTDetector",
    "MLDetector",
    # Model
    "CoTClassifierModel",
    # Pattern dictionaries
    "CODE_HACK_PATTERNS",
    "COT_RED_FLAGS",
    "MISALIGNMENT_PATTERNS",
]
