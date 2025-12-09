#!/usr/bin/env python3
"""
Trajectory Analysis Example

Shows how to load and analyze trajectory files from disk.
"""

import json
import sys
from pathlib import Path
from typing import Any

from rewardhackwatch.core.analyzer import TrajectoryAnalyzer
from rewardhackwatch.core.detectors.ml_detector import MLDetector


def load_trajectory(file_path: str) -> dict[str, Any]:
    """Load a trajectory from a JSON file."""
    with open(file_path) as f:
        return json.load(f)


def analyze_single(trajectory: dict[str, Any], use_ml: bool = False) -> dict[str, Any]:
    """
    Analyze a single trajectory.

    Args:
        trajectory: The trajectory data to analyze
        use_ml: Whether to use the ML detector

    Returns:
        Analysis results dictionary
    """
    analyzer = TrajectoryAnalyzer()
    result = analyzer.analyze(trajectory)

    results = {
        "task_id": trajectory.get("task_id", "unknown"),
        "hack_score": result.hack_score,
        "risk_level": result.risk_level,
        "detectors": result.detector_results,
    }

    if use_ml:
        try:
            ml_detector = MLDetector()
            ml_result = ml_detector.detect(trajectory)
            results["ml_prediction"] = {
                "is_hack": ml_result.is_hack,
                "confidence": ml_result.confidence,
            }
        except Exception as e:
            results["ml_error"] = str(e)

    return results


def analyze_batch(trajectories: list[dict[str, Any]], use_ml: bool = False) -> list[dict[str, Any]]:
    """
    Analyze multiple trajectories.

    Args:
        trajectories: List of trajectory data
        use_ml: Whether to use the ML detector

    Returns:
        List of analysis results
    """
    return [analyze_single(t, use_ml) for t in trajectories]


def main():
    """Run trajectory analysis from command line."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_trajectory.py <trajectory.json> [--ml]")
        print("\nExample:")
        print("  python analyze_trajectory.py sample.json")
        print("  python analyze_trajectory.py sample.json --ml")
        sys.exit(1)

    file_path = sys.argv[1]
    use_ml = "--ml" in sys.argv

    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Load and analyze
    print(f"Loading trajectory from: {file_path}")
    trajectory = load_trajectory(file_path)

    print(f"Analyzing (ML={use_ml})...")
    result = analyze_single(trajectory, use_ml)

    # Print results
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Task ID: {result['task_id']}")
    print(f"Hack Score: {result['hack_score']:.2f}")
    print(f"Risk Level: {result['risk_level']}")

    if "ml_prediction" in result:
        print("\nML Prediction:")
        print(f"  Is Hack: {result['ml_prediction']['is_hack']}")
        print(f"  Confidence: {result['ml_prediction']['confidence']:.2f}")

    print("\nDetector Results:")
    for detector, res in result["detectors"].items():
        print(f"  {detector}: {res}")


if __name__ == "__main__":
    main()
