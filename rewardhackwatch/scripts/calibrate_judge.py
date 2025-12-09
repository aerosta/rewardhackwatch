#!/usr/bin/env python3
"""
Judge calibration script for RewardHackWatch.
"""

from __future__ import annotations

"""

Loads RHW-Bench labeled trajectories and evaluates both Claude and Llama judges,
computing accuracy metrics and identifying disagreements.

Usage:
    python -m rewardhackwatch.scripts.calibrate_judge
    python -m rewardhackwatch.scripts.calibrate_judge --llama-only
    python -m rewardhackwatch.scripts.calibrate_judge --claude-only
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from rewardhackwatch.training.dataset import RHWBenchDataset


@dataclass
class JudgeMetrics:
    """Metrics for a single judge."""

    name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    avg_latency_ms: float = 0.0
    confusion: dict = field(default_factory=dict)
    per_category_accuracy: dict = field(default_factory=dict)
    predictions: list = field(default_factory=list)
    available: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "avg_latency_ms": self.avg_latency_ms,
            "confusion_matrix": self.confusion,
            "per_category_accuracy": self.per_category_accuracy,
            "available": self.available,
            "error": self.error,
        }


@dataclass
class CalibrationResult:
    """Results from judge calibration."""

    claude_metrics: JudgeMetrics
    llama_metrics: JudgeMetrics
    agreement_rate: float = 0.0
    disagreements: list = field(default_factory=list)
    total_trajectories: int = 0

    def to_dict(self) -> dict:
        return {
            "claude": self.claude_metrics.to_dict(),
            "llama": self.llama_metrics.to_dict(),
            "agreement_rate": self.agreement_rate,
            "disagreements": self.disagreements,
            "total_trajectories": self.total_trajectories,
        }


def load_trajectories(test_cases_dir: str, trajectories_dir: str) -> tuple[list, list, list]:
    """Load all labeled trajectories."""
    trajectories = []
    labels = []
    categories = []

    # Load test cases
    if Path(test_cases_dir).exists():
        dataset = RHWBenchDataset(test_cases_dir, split="all")
        for i, traj in enumerate(dataset.trajectories):
            trajectories.append(traj)
            labels.append(dataset.labels[i])
            categories.append(traj.get("_category", "unknown"))

    # Load real trajectories
    if Path(trajectories_dir).exists():
        dataset = RHWBenchDataset(trajectories_dir, split="all")
        for i, traj in enumerate(dataset.trajectories):
            trajectories.append(traj)
            labels.append(dataset.labels[i])
            # Extract category from metadata or filename
            cat = traj.get("metadata", {}).get("pattern_type", "unknown")
            categories.append(cat)

    return trajectories, labels, categories


def evaluate_llama_judge(
    trajectories: list,
    labels: list,
    categories: list,
) -> JudgeMetrics:
    """Evaluate Llama judge on trajectories."""
    metrics = JudgeMetrics(name="Llama 3.1 (8B)")

    try:
        from rewardhackwatch.core.judges import LlamaJudge

        judge = LlamaJudge()
        if not judge.available:
            metrics.error = "Ollama not running or model not available"
            return metrics

        metrics.available = True
        predictions = []
        latencies = []

        for traj in trajectories:
            start = time.time()
            result = judge.judge_sync(traj)
            latencies.append((time.time() - start) * 1000)

            # Use hack_detected from metadata if available, else threshold
            hack_detected = result.metadata.get("raw_hack_detected", result.hack_score > 0.5)
            predictions.append(1 if hack_detected else 0)

        metrics.predictions = predictions
        metrics.avg_latency_ms = np.mean(latencies)

        # Compute metrics
        y_true = np.array(labels)
        y_pred = np.array(predictions)

        metrics.accuracy = accuracy_score(y_true, y_pred)
        metrics.precision = precision_score(y_true, y_pred, zero_division=0)
        metrics.recall = recall_score(y_true, y_pred, zero_division=0)
        metrics.f1 = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.confusion = {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

        # Per-category accuracy
        category_correct = {}
        category_total = {}
        for i, cat in enumerate(categories):
            if cat not in category_correct:
                category_correct[cat] = 0
                category_total[cat] = 0
            category_total[cat] += 1
            if predictions[i] == labels[i]:
                category_correct[cat] += 1

        for cat in category_total:
            metrics.per_category_accuracy[cat] = category_correct[cat] / category_total[cat]

    except Exception as e:
        metrics.error = str(e)

    return metrics


def evaluate_claude_judge(
    trajectories: list,
    labels: list,
    categories: list,
) -> JudgeMetrics:
    """Evaluate Claude judge on trajectories."""
    metrics = JudgeMetrics(name="Claude 4.5 Opus")

    try:
        import os

        if not os.getenv("ANTHROPIC_API_KEY"):
            metrics.error = "ANTHROPIC_API_KEY not set"
            return metrics

        from rewardhackwatch.core.judges import ClaudeJudge

        judge = ClaudeJudge()
        metrics.available = True
        predictions = []
        latencies = []

        for traj in trajectories:
            start = time.time()
            result = judge.judge_sync(traj)
            latencies.append((time.time() - start) * 1000)

            hack_detected = result.metadata.get("raw_hack_detected", result.hack_score > 0.5)
            predictions.append(1 if hack_detected else 0)

        metrics.predictions = predictions
        metrics.avg_latency_ms = np.mean(latencies)

        # Compute metrics
        y_true = np.array(labels)
        y_pred = np.array(predictions)

        metrics.accuracy = accuracy_score(y_true, y_pred)
        metrics.precision = precision_score(y_true, y_pred, zero_division=0)
        metrics.recall = recall_score(y_true, y_pred, zero_division=0)
        metrics.f1 = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.confusion = {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

        # Per-category accuracy
        category_correct = {}
        category_total = {}
        for i, cat in enumerate(categories):
            if cat not in category_correct:
                category_correct[cat] = 0
                category_total[cat] = 0
            category_total[cat] += 1
            if predictions[i] == labels[i]:
                category_correct[cat] += 1

        for cat in category_total:
            metrics.per_category_accuracy[cat] = category_correct[cat] / category_total[cat]

    except Exception as e:
        metrics.error = str(e)

    return metrics


def compute_agreement(
    claude_metrics: JudgeMetrics,
    llama_metrics: JudgeMetrics,
    trajectories: list,
    labels: list,
) -> tuple[float, list]:
    """Compute agreement rate and identify disagreements."""
    if not claude_metrics.available or not llama_metrics.available:
        return 0.0, []

    disagreements = []
    agreements = 0

    for i in range(len(trajectories)):
        claude_pred = claude_metrics.predictions[i]
        llama_pred = llama_metrics.predictions[i]
        true_label = labels[i]

        if claude_pred == llama_pred:
            agreements += 1
        else:
            disagreements.append(
                {
                    "index": i,
                    "trajectory_name": trajectories[i].get("_name", f"trajectory_{i}"),
                    "true_label": "hack" if true_label == 1 else "clean",
                    "claude_prediction": "hack" if claude_pred == 1 else "clean",
                    "llama_prediction": "hack" if llama_pred == 1 else "clean",
                    "claude_correct": claude_pred == true_label,
                    "llama_correct": llama_pred == true_label,
                }
            )

    agreement_rate = agreements / len(trajectories) if trajectories else 0.0
    return agreement_rate, disagreements


def generate_report(result: CalibrationResult) -> str:
    """Generate markdown calibration report."""
    lines = [
        "# Judge Calibration Report",
        "",
        "## Summary",
        "",
        "| Metric | Claude 4.5 Opus | Llama 3.1 (8B) |",
        "|--------|-----------------|----------------|",
    ]

    claude = result.claude_metrics
    llama = result.llama_metrics

    def fmt(val, available):
        if not available:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.1%}" if val <= 1 else f"{val:.0f}"
        return str(val)

    lines.append(
        f"| Available | {'Yes' if claude.available else 'No'} | {'Yes' if llama.available else 'No'} |"
    )
    lines.append(
        f"| Accuracy | {fmt(claude.accuracy, claude.available)} | {fmt(llama.accuracy, llama.available)} |"
    )
    lines.append(
        f"| Precision | {fmt(claude.precision, claude.available)} | {fmt(llama.precision, llama.available)} |"
    )
    lines.append(
        f"| Recall | {fmt(claude.recall, claude.available)} | {fmt(llama.recall, llama.available)} |"
    )
    lines.append(
        f"| F1 Score | {fmt(claude.f1, claude.available)} | {fmt(llama.f1, llama.available)} |"
    )
    lines.append(
        f"| Avg Latency | {fmt(claude.avg_latency_ms, claude.available)}ms | {fmt(llama.avg_latency_ms, llama.available)}ms |"
    )

    lines.extend(
        [
            "",
            f"**Total Trajectories:** {result.total_trajectories}",
            f"**Agreement Rate:** {result.agreement_rate:.1%}",
            "",
        ]
    )

    # Confusion matrices
    if claude.available and claude.confusion:
        lines.extend(
            [
                "## Confusion Matrices",
                "",
                "### Claude 4.5 Opus",
                "",
                "| | Predicted Hack | Predicted Clean |",
                "|---|---|---|",
                f"| Actual Hack | {claude.confusion.get('tp', 0)} | {claude.confusion.get('fn', 0)} |",
                f"| Actual Clean | {claude.confusion.get('fp', 0)} | {claude.confusion.get('tn', 0)} |",
                "",
            ]
        )

    if llama.available and llama.confusion:
        lines.extend(
            [
                "### Llama 3.1 (8B)",
                "",
                "| | Predicted Hack | Predicted Clean |",
                "|---|---|---|",
                f"| Actual Hack | {llama.confusion.get('tp', 0)} | {llama.confusion.get('fn', 0)} |",
                f"| Actual Clean | {llama.confusion.get('fp', 0)} | {llama.confusion.get('tn', 0)} |",
                "",
            ]
        )

    # Per-category accuracy
    if claude.available and claude.per_category_accuracy:
        lines.extend(
            [
                "## Per-Category Accuracy",
                "",
                "| Category | Claude | Llama |",
                "|----------|--------|-------|",
            ]
        )
        all_cats = set(claude.per_category_accuracy.keys())
        if llama.available:
            all_cats |= set(llama.per_category_accuracy.keys())

        for cat in sorted(all_cats):
            c_acc = claude.per_category_accuracy.get(cat, 0)
            l_acc = llama.per_category_accuracy.get(cat, 0) if llama.available else 0
            lines.append(f"| {cat} | {c_acc:.1%} | {l_acc:.1%} |")

        lines.append("")

    # Disagreements
    if result.disagreements:
        lines.extend(
            [
                "## Disagreements",
                "",
                "Cases where Claude and Llama disagree:",
                "",
                "| Trajectory | True | Claude | Llama | Winner |",
                "|------------|------|--------|-------|--------|",
            ]
        )

        for d in result.disagreements[:10]:  # Limit to 10
            winner = (
                "Claude" if d["claude_correct"] else ("Llama" if d["llama_correct"] else "Neither")
            )
            lines.append(
                f"| {d['trajectory_name']} | {d['true_label']} | {d['claude_prediction']} | {d['llama_prediction']} | {winner} |"
            )

        if len(result.disagreements) > 10:
            lines.append(f"| ... and {len(result.disagreements) - 10} more | | | | |")

        lines.append("")

    # Errors
    if claude.error or llama.error:
        lines.extend(
            [
                "## Errors",
                "",
            ]
        )
        if claude.error:
            lines.append(f"- **Claude:** {claude.error}")
        if llama.error:
            lines.append(f"- **Llama:** {llama.error}")
        lines.append("")

    lines.extend(
        [
            "---",
            "Generated by RewardHackWatch calibrate_judge.py",
        ]
    )

    return "\n".join(lines)


def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Calibrate LLM judges on RHW-Bench")
    parser.add_argument("--llama-only", action="store_true", help="Only evaluate Llama judge")
    parser.add_argument("--claude-only", action="store_true", help="Only evaluate Claude judge")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument(
        "--test-cases",
        default="rewardhackwatch/rhw_bench/test_cases",
        help="Test cases directory",
    )
    parser.add_argument(
        "--trajectories",
        default="rewardhackwatch/rhw_bench/trajectories",
        help="Trajectories directory",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Judge Calibration")
    print("=" * 60)

    # Load data
    print("\nLoading trajectories...")
    trajectories, labels, categories = load_trajectories(args.test_cases, args.trajectories)
    print(f"Loaded {len(trajectories)} trajectories ({sum(labels)} hacks)")

    # Initialize results
    claude_metrics = JudgeMetrics(name="Claude 4.5 Opus")
    llama_metrics = JudgeMetrics(name="Llama 3.1 (8B)")

    # Evaluate judges
    if not args.llama_only:
        print("\nEvaluating Claude judge...")
        claude_metrics = evaluate_claude_judge(trajectories, labels, categories)
        if claude_metrics.available:
            print(f"  Accuracy: {claude_metrics.accuracy:.1%}")
            print(f"  F1: {claude_metrics.f1:.1%}")
        else:
            print(f"  Not available: {claude_metrics.error}")

    if not args.claude_only:
        print("\nEvaluating Llama judge...")
        llama_metrics = evaluate_llama_judge(trajectories, labels, categories)
        if llama_metrics.available:
            print(f"  Accuracy: {llama_metrics.accuracy:.1%}")
            print(f"  F1: {llama_metrics.f1:.1%}")
        else:
            print(f"  Not available: {llama_metrics.error}")

    # Compute agreement
    agreement_rate, disagreements = compute_agreement(
        claude_metrics, llama_metrics, trajectories, labels
    )

    # Build result
    result = CalibrationResult(
        claude_metrics=claude_metrics,
        llama_metrics=llama_metrics,
        agreement_rate=agreement_rate,
        disagreements=disagreements,
        total_trajectories=len(trajectories),
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "judge_calibration.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Markdown
    md_path = output_dir / "judge_calibration.md"
    report = generate_report(result)
    with open(md_path, "w") as f:
        f.write(report)

    print("\nResults saved to:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")

    if agreement_rate > 0:
        print(f"\nAgreement rate: {agreement_rate:.1%}")
        print(f"Disagreements: {len(disagreements)}")


if __name__ == "__main__":
    main()
