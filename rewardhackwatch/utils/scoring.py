"""Scoring utilities for detection results."""

import math
from typing import Optional


def combine_scores(scores: list[float], weights: Optional[list[float]] = None) -> float:
    """Combine multiple scores with optional weights."""
    if not scores:
        return 0.0
    if weights is None:
        weights = [1.0] * len(scores)

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return weighted_sum / total_weight


def normalize_score(score: float, min_val: float = 0, max_val: float = 1) -> float:
    """Normalize score to [0, 1] range."""
    return max(min_val, min(max_val, score))


def score_to_risk_level(score: float) -> str:
    """Convert score to risk level."""
    if score >= 0.8:
        return "CRITICAL"
    elif score >= 0.6:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    elif score >= 0.2:
        return "LOW"
    return "NONE"


def aggregate_detector_scores(results: dict[str, float]) -> dict[str, float]:
    """Aggregate scores from multiple detectors."""
    if not results:
        return {"combined": 0.0, "max": 0.0, "mean": 0.0}

    scores = list(results.values())
    return {
        "combined": combine_scores(scores),
        "max": max(scores),
        "mean": sum(scores) / len(scores),
        "min": min(scores),
    }


def confidence_weighted_score(score: float, confidence: float) -> float:
    """Apply confidence weighting to a score."""
    return score * confidence


def sigmoid_normalize(x: float, steepness: float = 1.0) -> float:
    """Apply sigmoid normalization to a value."""
    return 1 / (1 + math.exp(-steepness * x))
