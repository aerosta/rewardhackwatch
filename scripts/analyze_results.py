#!/usr/bin/env python3
"""
Analyze Results

Analyzes benchmark or evaluation results and generates visualizations.
"""

import argparse
import json
from typing import Any


def load_results(results_path: str) -> dict[str, Any]:
    """Load results from file."""
    with open(results_path) as f:
        return json.load(f)


def print_summary(results: dict[str, Any]) -> None:
    """Print summary of results."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if "accuracy" in results:
        print("\nOverall Metrics:")
        print(f"  Accuracy:  {results.get('accuracy', 0):.2%}")
        print(f"  Precision: {results.get('precision', 0):.2%}")
        print(f"  Recall:    {results.get('recall', 0):.2%}")
        print(f"  F1 Score:  {results.get('f1', 0):.2%}")

    if "confusion_matrix" in results:
        cm = results["confusion_matrix"]
        if isinstance(cm, dict):
            print("\nConfusion Matrix:")
            print(f"  TP: {cm.get('tp', 0):4d}  FP: {cm.get('fp', 0):4d}")
            print(f"  FN: {cm.get('fn', 0):4d}  TN: {cm.get('tn', 0):4d}")
        elif isinstance(cm, list) and len(cm) == 2:
            print("\nConfusion Matrix:")
            print(f"  [[{cm[0][0]:4d}, {cm[0][1]:4d}]")
            print(f"   [{cm[1][0]:4d}, {cm[1][1]:4d}]]")

    if "results" in results:
        # Detailed results
        details = results["results"]
        errors = [r for r in details if not r.get("correct", True)]

        print("\nDetailed Results:")
        print(f"  Total:   {len(details)}")
        print(f"  Errors:  {len(errors)}")

        if errors:
            print("\nSample Errors (first 5):")
            for e in errors[:5]:
                print(
                    f"  - {e.get('case_id', 'unknown')}: "
                    f"expected={e.get('expected')}, "
                    f"predicted={e.get('predicted')}"
                )


def analyze_by_category(results: dict[str, Any]) -> None:
    """Analyze results by category."""
    if "results" not in results:
        return

    details = results["results"]

    # Group by expected label
    by_expected = {}
    for r in details:
        exp = "hack" if r.get("expected") else "clean"
        if exp not in by_expected:
            by_expected[exp] = []
        by_expected[exp].append(r)

    print("\n" + "-" * 60)
    print("Analysis by Category")
    print("-" * 60)

    for category, items in by_expected.items():
        correct = sum(1 for i in items if i.get("correct", False))
        accuracy = correct / len(items) if items else 0

        print(f"\n{category.upper()} samples:")
        print(f"  Total:    {len(items)}")
        print(f"  Correct:  {correct}")
        print(f"  Accuracy: {accuracy:.2%}")


def analyze_score_distribution(results: dict[str, Any]) -> None:
    """Analyze hack score distribution."""
    if "results" not in results:
        return

    details = results["results"]
    scores = [r.get("hack_score", 0) for r in details]

    if not scores:
        return

    print("\n" + "-" * 60)
    print("Score Distribution")
    print("-" * 60)

    # Basic statistics
    import statistics

    print(f"\n  Min:    {min(scores):.3f}")
    print(f"  Max:    {max(scores):.3f}")
    print(f"  Mean:   {statistics.mean(scores):.3f}")
    print(f"  Median: {statistics.median(scores):.3f}")
    print(f"  StdDev: {statistics.stdev(scores):.3f}")

    # Distribution buckets
    buckets = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print("\n  Distribution:")

    for i in range(len(buckets) - 1):
        low, high = buckets[i], buckets[i + 1]
        count = sum(1 for s in scores if low <= s < high)
        bar = "█" * (count * 40 // len(scores))
        print(f"    {low:.1f}-{high:.1f}: {count:4d} {bar}")


def main():
    parser = argparse.ArgumentParser(description="Analyze results")
    parser.add_argument(
        "results",
        type=str,
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed analysis",
    )

    args = parser.parse_args()

    results = load_results(args.results)

    print_summary(results)

    if args.detailed:
        analyze_by_category(results)
        analyze_score_distribution(results)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
