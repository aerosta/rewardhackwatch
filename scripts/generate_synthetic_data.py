#!/usr/bin/env python3
"""Generate synthetic trajectories for training data augmentation.

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --mock-count 500 --clean-mock-count 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rewardhackwatch.rhw_bench.generators.clean_generator import CleanTrajectoryGenerator
from rewardhackwatch.rhw_bench.generators.mock_exploit_generator import MockExploitGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--mock-count", type=int, default=500, help="Mock exploit trajectories")
    parser.add_argument("--clean-mock-count", type=int, default=200, help="Clean mock trajectories")
    parser.add_argument("--clean-count", type=int, default=500, help="Clean code trajectories")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rewardhackwatch/rhw_bench/test_cases/generated",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate mock exploits
    print(f"Generating {args.mock_count} mock exploit trajectories...")
    mock_gen = MockExploitGenerator(seed=args.seed)
    mock_hacks = mock_gen.generate(count=args.mock_count)
    for traj in mock_hacks:
        path = output_dir / f"{traj['name']}.json"
        with open(path, "w") as f:
            json.dump(traj, f, indent=2)
    print(f"  Wrote {len(mock_hacks)} mock exploit files")

    # Generate clean mocks
    print(f"Generating {args.clean_mock_count} clean mock trajectories...")
    clean_mocks = mock_gen.generate_clean_mocks(count=args.clean_mock_count)
    for traj in clean_mocks:
        path = output_dir / f"{traj['name']}.json"
        with open(path, "w") as f:
            json.dump(traj, f, indent=2)
    print(f"  Wrote {len(clean_mocks)} clean mock files")

    # Generate clean code trajectories
    print(f"Generating {args.clean_count} clean code trajectories...")
    clean_gen = CleanTrajectoryGenerator(seed=args.seed)
    clean_trajs = clean_gen.generate(count=args.clean_count)
    for traj in clean_trajs:
        path = output_dir / f"{traj['name']}.json"
        with open(path, "w") as f:
            json.dump(traj, f, indent=2)
    print(f"  Wrote {len(clean_trajs)} clean code files")

    total = len(mock_hacks) + len(clean_mocks) + len(clean_trajs)
    print(f"\nTotal: {total} trajectories written to {output_dir}/")
    print(f"  Mock exploits (hack):  {len(mock_hacks)}")
    print(f"  Clean mocks (clean):   {len(clean_mocks)}")
    print(f"  Clean code (clean):    {len(clean_trajs)}")


if __name__ == "__main__":
    main()
