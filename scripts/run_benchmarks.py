#!/usr/bin/env python3
"""
Run Benchmarks

Runs all configured benchmarks and generates a report.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rewardhackwatch.core.analyzer import TrajectoryAnalyzer


def load_benchmark_data(benchmark_path: str) -> list[dict[str, Any]]:
    """Load benchmark data from file."""
    path = Path(benchmark_path)

    if not path.exists():
        print(f"Warning: Benchmark file not found: {path}")
        return []

    with open(path) as f:
        return json.load(f)


def run_benchmark(
    data: list[dict[str, Any]],
    analyzer: TrajectoryAnalyzer,
    name: str,
) -> dict[str, Any]:
    """
    Run benchmark on dataset.

    Args:
        data: List of benchmark cases
        analyzer: TrajectoryAnalyzer instance
        name: Benchmark name

    Returns:
        Benchmark results
    """
    results = []
    correct = 0
    total = 0

    for i, case in enumerate(data):
        trajectory = case.get("trajectory", case)
        expected = case.get("expected", case.get("is_hack", False))

        result = analyzer.analyze(trajectory)
        predicted = result.is_hack

        is_correct = predicted == expected
        if is_correct:
            correct += 1
        total += 1

        results.append(
            {
                "case_id": case.get("name", case.get("task_id", str(i))),
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "hack_score": result.hack_score,
            }
        )

    accuracy = correct / total if total > 0 else 0

    # Calculate detailed metrics
    tp = sum(1 for r in results if r["expected"] and r["predicted"])
    fp = sum(1 for r in results if not r["expected"] and r["predicted"])
    tn = sum(1 for r in results if not r["expected"] and not r["predicted"])
    fn = sum(1 for r in results if r["expected"] and not r["predicted"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "name": name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "results": results,
    }


def generate_report(benchmark_results: list[dict[str, Any]]) -> str:
    """Generate markdown report."""
    lines = [
        "# Benchmark Results",
        f"\nGenerated: {datetime.now().isoformat()}\n",
        "## Summary\n",
        "| Benchmark | Accuracy | Precision | Recall | F1 |",
        "|-----------|----------|-----------|--------|-----|",
    ]

    for result in benchmark_results:
        lines.append(
            f"| {result['name']} | "
            f"{result['accuracy']:.1%} | "
            f"{result['precision']:.1%} | "
            f"{result['recall']:.1%} | "
            f"{result['f1']:.1%} |"
        )

    lines.append("\n## Detailed Results\n")

    for result in benchmark_results:
        lines.append(f"### {result['name']}\n")
        lines.append(f"- Total: {result['total']}")
        lines.append(f"- Correct: {result['correct']}")
        lines.append(
            f"- Confusion Matrix: TP={result['confusion_matrix']['tp']}, "
            f"FP={result['confusion_matrix']['fp']}, "
            f"TN={result['confusion_matrix']['tn']}, "
            f"FN={result['confusion_matrix']['fn']}"
        )
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["./data/benchmarks/rhw_bench.json"],
        help="Benchmark files to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/benchmark_report.md",
        help="Output report path",
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Enable ML detector",
    )

    args = parser.parse_args()

    print("Initializing analyzer...")
    analyzer = TrajectoryAnalyzer()

    benchmark_results = []

    for benchmark_path in args.benchmarks:
        print(f"\nRunning benchmark: {benchmark_path}")

        data = load_benchmark_data(benchmark_path)
        if not data:
            continue

        name = Path(benchmark_path).stem
        result = run_benchmark(data, analyzer, name)
        benchmark_results.append(result)

        print(f"  Accuracy: {result['accuracy']:.1%}")
        print(f"  F1: {result['f1']:.1%}")

    # Generate and save report
    report = generate_report(benchmark_results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
