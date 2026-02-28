#!/usr/bin/env python3
"""Build and export the HackBench benchmark dataset.

Usage:
    python scripts/build_hackbench.py
    python scripts/build_hackbench.py --output-dir data/hackbench --generate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rewardhackwatch.rhw_bench.hackbench import HackBenchConfig, HackBenchDataset


def main():
    parser = argparse.ArgumentParser(description="Build HackBench benchmark dataset")
    parser.add_argument(
        "--output-dir", type=str, default="data/hackbench", help="Output directory"
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate synthetic data first (runs generate_synthetic_data.py)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Optionally generate synthetic data first
    if args.generate:
        print("Generating synthetic data first...")
        import subprocess
        subprocess.run(
            [sys.executable, "scripts/generate_synthetic_data.py"],
            check=True,
        )
        print()

    # Build HackBench
    print("Building HackBench dataset...")
    config = HackBenchConfig(
        output_dir=args.output_dir,
        seed=args.seed,
    )
    hb = HackBenchDataset(config)
    stats = hb.curate()

    print(f"\nHackBench Statistics:")
    print(f"  Total trajectories: {stats.total}")
    print(f"  Hack count: {stats.hack_count}")
    print(f"  Clean count: {stats.clean_count}")
    print(f"  Hack rate: {stats.hack_rate:.1%}")
    print(f"\n  By category:")
    for cat, count in sorted(stats.by_category.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")
    print(f"\n  By source:")
    for src, count in sorted(stats.by_source.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count}")
    print(f"\n  Splits:")
    for split, count in stats.splits.items():
        print(f"    {split}: {count}")

    # Export
    out = hb.export(args.output_dir)
    print(f"\nExported to: {out}")


if __name__ == "__main__":
    main()
