"""Validation utilities for trajectories and results."""

import re
from typing import Any, Optional


def validate_trajectory(trajectory: dict[str, Any]) -> list[str]:
    """Validate trajectory structure, return list of errors."""
    errors = []

    if not isinstance(trajectory, dict):
        errors.append("Trajectory must be a dictionary")
        return errors

    # Check for required fields
    if (
        "steps" not in trajectory
        and "cot_traces" not in trajectory
        and "code_outputs" not in trajectory
    ):
        errors.append("Trajectory must have 'steps', 'cot_traces', or 'code_outputs'")

    # Validate steps if present
    if "steps" in trajectory:
        if not isinstance(trajectory["steps"], list):
            errors.append("'steps' must be a list")
        else:
            for i, step in enumerate(trajectory["steps"]):
                if not isinstance(step, dict):
                    errors.append(f"Step {i} must be a dictionary")

    # Validate cot_traces if present
    if "cot_traces" in trajectory and trajectory["cot_traces"] is not None:
        if not isinstance(trajectory["cot_traces"], list):
            errors.append("'cot_traces' must be a list")

    # Validate code_outputs if present
    if "code_outputs" in trajectory and trajectory["code_outputs"] is not None:
        if not isinstance(trajectory["code_outputs"], list):
            errors.append("'code_outputs' must be a list")

    return errors


def validate_score(score: Any) -> bool:
    """Validate that score is a valid float in [0, 1]."""
    try:
        s = float(score)
        return 0 <= s <= 1
    except (TypeError, ValueError):
        return False


def validate_risk_level(level: str) -> bool:
    """Validate risk level string."""
    valid_levels = {"none", "low", "medium", "high", "critical"}
    return level.lower() in valid_levels


def validate_code_syntax(code: str) -> Optional[str]:
    """Validate Python syntax, return error message or None."""
    try:
        compile(code, "<string>", "exec")
        return None
    except SyntaxError as e:
        return f"Syntax error at line {e.lineno}: {e.msg}"


def validate_detector_result(result: dict[str, Any]) -> list[str]:
    """Validate detector result structure."""
    errors = []

    if "score" not in result:
        errors.append("Result must have 'score' field")
    elif not validate_score(result["score"]):
        errors.append("Score must be float in [0, 1]")

    if "detections" in result and not isinstance(result["detections"], list):
        errors.append("'detections' must be a list")

    return errors


def is_valid_json_path(path: str) -> bool:
    """Check if path is a valid JSON file path."""
    return path.endswith(".json") and not re.search(r'[<>:"|?*]', path)
