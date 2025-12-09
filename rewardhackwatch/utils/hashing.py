"""Hashing utilities for trajectory deduplication."""

import hashlib
import json


def hash_trajectory(trajectory: dict) -> str:
    """Create hash of trajectory for deduplication."""
    content = json.dumps(trajectory, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def hash_code(code: str) -> str:
    """Hash code content."""
    return hashlib.md5(code.encode()).hexdigest()[:12]


def hash_steps(steps: list) -> str:
    """Hash trajectory steps."""
    content = json.dumps(steps, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def content_fingerprint(content: str) -> str:
    """Create fingerprint of content (normalized)."""
    normalized = " ".join(content.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()[:10]


def trajectory_similarity_hash(trajectory: dict, granularity: str = "coarse") -> str:
    """Create similarity hash for trajectory grouping.

    Args:
        trajectory: The trajectory dict
        granularity: "coarse" (by structure) or "fine" (by content)
    """
    if granularity == "coarse":
        # Hash based on structure only
        structure = {
            "step_count": len(trajectory.get("steps", [])),
            "has_code": any("code" in str(s) for s in trajectory.get("steps", [])),
            "has_thinking": any("thinking" in str(s) for s in trajectory.get("steps", [])),
        }
        return hashlib.md5(json.dumps(structure, sort_keys=True).encode()).hexdigest()[:8]
    else:
        return hash_trajectory(trajectory)
