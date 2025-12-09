#!/usr/bin/env python3
"""
Evaluate Model

Evaluates a trained model on test data and generates metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not installed")

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_test_data(data_path: str) -> list[dict[str, Any]]:
    """Load test data."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with open(path) as f:
        return json.load(f)


def evaluate_model(
    model_path: str,
    test_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Evaluate model on test data.

    Args:
        model_path: Path to trained model
        test_data: Test dataset

    Returns:
        Evaluation metrics
    """
    from rewardhackwatch.core.detectors.ml_detector import MLDetector

    print(f"Loading model from {model_path}...")
    detector = MLDetector(model_path=model_path)

    print(f"Evaluating on {len(test_data)} samples...")

    predictions = []
    labels = []
    scores = []

    for i, sample in enumerate(test_data):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(test_data)}")

        trajectory = {
            "cot_traces": sample.get("cot_traces", []),
            "code_outputs": sample.get("code_outputs", []),
        }

        result = detector.detect(trajectory)

        predictions.append(1 if result.is_hack else 0)
        labels.append(sample.get("reward_hack_label", sample.get("is_hack", 0)))
        scores.append(result.confidence)

    # Calculate metrics
    if SKLEARN_AVAILABLE:
        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
            "f1": f1_score(labels, predictions, zero_division=0),
            "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        }

        try:
            metrics["auc_roc"] = roc_auc_score(labels, scores)
        except ValueError:
            metrics["auc_roc"] = None

        print("\nClassification Report:")
        print(classification_report(labels, predictions, target_names=["Clean", "Hack"]))
    else:
        # Manual calculation
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        metrics = {
            "accuracy": correct / len(predictions),
            "total": len(predictions),
            "correct": correct,
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/evaluation.json",
        help="Output path for metrics",
    )

    args = parser.parse_args()

    test_data = load_test_data(args.test_data)
    metrics = evaluate_model(args.model, test_data)

    # Save metrics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {output_path}")
    print("\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    if "f1" in metrics:
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall: {metrics['recall']:.2%}")
        print(f"  F1: {metrics['f1']:.2%}")


if __name__ == "__main__":
    main()
