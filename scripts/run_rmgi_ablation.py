#!/usr/bin/env python3
"""
RMGI Ablation Study Script

Tests different window sizes and thresholds for the RMGI tracker
to determine optimal parameters for transition detection.

Window sizes tested: [5, 10, 20]
Thresholds tested: [0.5, 0.6, 0.7, 0.8]
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rewardhackwatch.core.trackers.generalization_tracker import GeneralizationTracker


def generate_synthetic_trajectories(
    n_trajectories: int = 100, seed: int = 42
) -> list[dict[str, Any]]:
    """
    Generate synthetic trajectories for ablation study.

    Creates trajectories with known transition points for validation.
    """
    np.random.seed(seed)
    trajectories = []

    for i in range(n_trajectories):
        # Determine if this trajectory has a transition
        has_transition = np.random.random() < 0.3  # 30% have transitions

        n_steps = np.random.randint(15, 50)
        hack_scores = []
        misalign_scores = []
        true_transition_step = None

        if has_transition:
            # Transition occurs at random point after step 5
            true_transition_step = np.random.randint(5, n_steps - 5)

            for step in range(n_steps):
                if step < true_transition_step:
                    # Before transition: independent noise
                    hack_scores.append(np.random.uniform(0, 0.3))
                    misalign_scores.append(np.random.uniform(0, 0.2))
                else:
                    # After transition: correlated behavior
                    base = np.random.uniform(0.5, 0.9)
                    hack_scores.append(base + np.random.uniform(-0.1, 0.1))
                    misalign_scores.append(base * 0.9 + np.random.uniform(-0.1, 0.1))
        else:
            # No transition: independent noise throughout
            for step in range(n_steps):
                hack_scores.append(np.random.uniform(0, 0.4))
                misalign_scores.append(np.random.uniform(0, 0.3))

        trajectories.append(
            {
                "id": f"traj_{i}",
                "hack_scores": hack_scores,
                "misalign_scores": misalign_scores,
                "has_transition": has_transition,
                "true_transition_step": true_transition_step,
                "n_steps": n_steps,
            }
        )

    return trajectories


def evaluate_configuration(
    trajectories: list[dict[str, Any]], window_size: int, threshold: float
) -> dict[str, Any]:
    """
    Evaluate a specific window size and threshold configuration.

    Returns metrics:
    - detection_precision: TP / (TP + FP)
    - detection_recall: TP / (TP + FN)
    - detection_f1: harmonic mean
    - mean_delay: average steps between true and detected transition
    - false_positive_rate: FP / (FP + TN)
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    detection_delays = []

    for traj in trajectories:
        tracker = GeneralizationTracker(window_size=window_size)

        detected_transition = None
        rmgi_values = []

        # Process trajectory
        for step, (hack, misalign) in enumerate(zip(traj["hack_scores"], traj["misalign_scores"])):
            tracker.update(hack, misalign)
            summary = tracker.get_summary()
            # Use correlation as RMGI proxy
            rmgi_values.append(summary.get("correlation", 0.0))

            # Check for transition detection
            current_rmgi = summary.get("correlation", 0)
            if detected_transition is None and current_rmgi >= threshold:
                # Need 3 consecutive above threshold
                if len(rmgi_values) >= 3:
                    if all(r >= threshold for r in rmgi_values[-3:]):
                        detected_transition = step

        # Evaluate detection
        if traj["has_transition"]:
            if detected_transition is not None:
                true_positives += 1
                delay = detected_transition - traj["true_transition_step"]
                detection_delays.append(max(0, delay))
            else:
                false_negatives += 1
        else:
            if detected_transition is not None:
                false_positives += 1
            else:
                true_negatives += 1

    # Compute metrics
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2 * precision * recall / max(0.001, precision + recall)
    fpr = false_positives / max(1, false_positives + true_negatives)

    mean_delay = np.mean(detection_delays) if detection_delays else float("nan")
    median_delay = np.median(detection_delays) if detection_delays else float("nan")

    return {
        "window_size": window_size,
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "mean_delay": round(mean_delay, 2) if not np.isnan(mean_delay) else None,
        "median_delay": round(median_delay, 2) if not np.isnan(median_delay) else None,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }


def run_ablation_study() -> dict[str, Any]:
    """
    Run full RMGI ablation study across all parameter combinations.
    """
    print("=" * 60)
    print("RMGI ABLATION STUDY")
    print("=" * 60)

    # Configuration
    window_sizes = [5, 10, 20]
    thresholds = [0.5, 0.6, 0.7, 0.8]

    print(f"\nWindow sizes: {window_sizes}")
    print(f"Thresholds: {thresholds}")
    print(f"Total configurations: {len(window_sizes) * len(thresholds)}")

    # Generate synthetic data
    print("\nGenerating synthetic trajectories...")
    trajectories = generate_synthetic_trajectories(n_trajectories=200, seed=42)

    n_with_transition = sum(1 for t in trajectories if t["has_transition"])
    print(f"  Total trajectories: {len(trajectories)}")
    print(f"  With transitions: {n_with_transition}")
    print(f"  Without transitions: {len(trajectories) - n_with_transition}")

    # Run ablation
    print("\nRunning ablation study...")
    results = []

    for window_size in window_sizes:
        for threshold in thresholds:
            result = evaluate_configuration(trajectories, window_size, threshold)
            results.append(result)
            print(
                f"  Window={window_size:2d}, Threshold={threshold:.1f}: "
                f"F1={result['f1']:.3f}, Precision={result['precision']:.3f}, "
                f"Recall={result['recall']:.3f}, FPR={result['false_positive_rate']:.3f}"
            )

    # Find best configuration
    best_result = max(results, key=lambda x: x["f1"])

    # Generate markdown table
    markdown_table = generate_markdown_table(results, window_sizes, thresholds)

    study_results = {
        "study": "RMGI Ablation",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "window_sizes": window_sizes,
            "thresholds": thresholds,
            "n_trajectories": len(trajectories),
            "n_with_transitions": n_with_transition,
        },
        "results": results,
        "best_configuration": best_result,
        "markdown_table": markdown_table,
    }

    return study_results


def generate_markdown_table(
    results: list[dict[str, Any]], window_sizes: list[int], thresholds: list[float]
) -> str:
    """Generate markdown table for the ablation results."""
    lines = []
    lines.append("### RMGI Ablation Results")
    lines.append("")
    lines.append("| Window | Threshold | F1 | Precision | Recall | FPR | Mean Delay |")
    lines.append("|--------|-----------|-----|-----------|--------|-----|------------|")

    for result in results:
        delay_str = f"{result['mean_delay']:.1f}" if result["mean_delay"] else "N/A"
        lines.append(
            f"| {result['window_size']:6d} | {result['threshold']:.1f} | "
            f"{result['f1']:.3f} | {result['precision']:.3f} | "
            f"{result['recall']:.3f} | {result['false_positive_rate']:.3f} | "
            f"{delay_str:>10} |"
        )

    return "\n".join(lines)


def main():
    """Main entry point."""
    # Create results directory
    results_dir = Path(__file__).parent.parent / "results" / "ablation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run study
    study_results = run_ablation_study()

    # Save results
    output_path = results_dir / "rmgi_ablation.json"
    with open(output_path, "w") as f:
        json.dump(study_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print("=" * 60)

    best = study_results["best_configuration"]
    print("\nBest Configuration:")
    print(f"  Window Size: {best['window_size']}")
    print(f"  Threshold: {best['threshold']}")
    print(f"  F1 Score: {best['f1']:.3f}")
    print(f"  Precision: {best['precision']:.3f}")
    print(f"  Recall: {best['recall']:.3f}")
    print(f"  False Positive Rate: {best['false_positive_rate']:.3f}")

    print(f"\n{study_results['markdown_table']}")

    print(f"\nResults saved to: {output_path}")

    return study_results


if __name__ == "__main__":
    main()
