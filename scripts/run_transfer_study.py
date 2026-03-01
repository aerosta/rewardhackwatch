#!/usr/bin/env python3
"""Run cross-model transfer study.

Trains detectors on trajectories from one model family and evaluates
on all other families, producing an NxN transfer matrix of F1 scores.

Usage:
    python scripts/run_transfer_study.py
    python scripts/run_transfer_study.py --data-dir data/hackbench
    python scripts/run_transfer_study.py --output-dir results/transfer
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rewardhackwatch.experiments.transfer_study import (
    TransferStudyConfig,
    TransferStudyRunner,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-model transfer study for reward hacking detection"
    )
    parser.add_argument(
        "--data-dir",
        default="data/hackbench",
        help="Directory containing trajectory JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default="results/transfer_study",
        help="Directory to save results",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="Minimum samples per model family",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["anthropic", "openai", "meta", "deepseek"],
        help="Model families to include",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = TransferStudyConfig(
        data_dir=args.data_dir,
        model_families=args.families,
        min_samples_per_family=args.min_samples,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    runner = TransferStudyRunner(config)

    print("=" * 60)
    print("Cross-Model Transfer Study")
    print("=" * 60)
    print(f"Data directory: {config.data_dir}")
    print(f"Model families: {config.model_families}")
    print(f"Min samples/family: {config.min_samples_per_family}")
    print()

    results = runner.run_transfer_matrix()
    path = runner.export_results(results)

    # Print transfer matrix
    families = results["families"]
    matrix = results["transfer_matrix"]

    print("\nTransfer Matrix (F1 scores):")
    print("-" * 60)

    # Header
    header = f"{'Train↓ / Test→':>16}"
    for fam in families:
        header += f"  {fam:>10}"
    print(header)
    print("-" * 60)

    # Rows
    for i, train_fam in enumerate(families):
        row = f"{train_fam:>16}"
        for j, _test_fam in enumerate(families):
            val = matrix[i][j]
            marker = " *" if i == j else "  "
            row += f"  {val:>8.4f}{marker}"
        print(row)

    print("-" * 60)
    print(f"Diagonal mean F1 (self-transfer):  {results['diagonal_mean_f1']:.4f}")
    print(f"Off-diagonal mean F1 (cross-transfer): {results['off_diagonal_mean_f1']:.4f}")

    if "note" in results:
        print(f"\nNote: {results['note']}")

    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    main()
