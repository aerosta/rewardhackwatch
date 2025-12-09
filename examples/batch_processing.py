#!/usr/bin/env python3
"""
Batch Processing Example

Demonstrates how to process multiple trajectory files in parallel
and aggregate results.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rewardhackwatch.core.analyzer import TrajectoryAnalyzer


def process_file(file_path: Path) -> dict[str, Any]:
    """
    Process a single trajectory file.

    Args:
        file_path: Path to the trajectory JSON file

    Returns:
        Dictionary with file path and analysis results
    """
    try:
        with open(file_path) as f:
            trajectory = json.load(f)

        analyzer = TrajectoryAnalyzer()
        result = analyzer.analyze(trajectory)

        return {
            "file": str(file_path),
            "task_id": trajectory.get("task_id", "unknown"),
            "hack_score": result.hack_score,
            "risk_level": result.risk_level,
            "is_hack": result.is_hack,
            "error": None,
        }
    except Exception as e:
        return {
            "file": str(file_path),
            "task_id": "unknown",
            "hack_score": 0.0,
            "risk_level": "unknown",
            "is_hack": False,
            "error": str(e),
        }


def batch_process(
    directory: str,
    pattern: str = "*.json",
    max_workers: int = 4,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Process all trajectory files in a directory.

    Args:
        directory: Directory containing trajectory files
        pattern: Glob pattern for matching files
        max_workers: Number of parallel workers
        verbose: Whether to print progress

    Returns:
        List of results for all processed files
    """
    dir_path = Path(directory)
    files = list(dir_path.glob(pattern))

    if verbose:
        print(f"Found {len(files)} files matching '{pattern}'")

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in files}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)

            if verbose:
                status = "✓" if result["error"] is None else "✗"
                hack_indicator = "🚨" if result["is_hack"] else "✅"
                print(
                    f"[{i}/{len(files)}] {status} {result['file']}: {hack_indicator} (score: {result['hack_score']:.2f})"
                )

    elapsed = time.time() - start_time

    if verbose:
        print(f"\nProcessed {len(files)} files in {elapsed:.2f}s")
        print(f"Average: {elapsed / len(files):.3f}s per file")

    return results


def generate_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Generate a summary report from batch results.

    Args:
        results: List of individual analysis results

    Returns:
        Summary report dictionary
    """
    total = len(results)
    errors = sum(1 for r in results if r["error"] is not None)
    hacks = sum(1 for r in results if r["is_hack"])

    # Risk level distribution
    risk_counts = {}
    for r in results:
        level = r["risk_level"]
        risk_counts[level] = risk_counts.get(level, 0) + 1

    # Average hack score
    valid_scores = [r["hack_score"] for r in results if r["error"] is None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    return {
        "total_files": total,
        "successful": total - errors,
        "errors": errors,
        "hacks_detected": hacks,
        "hack_rate": hacks / (total - errors) if (total - errors) > 0 else 0,
        "average_hack_score": avg_score,
        "risk_distribution": risk_counts,
    }


def main():
    """Run batch processing from command line."""
    if len(sys.argv) < 2:
        print("Usage: python batch_processing.py <directory> [--pattern=*.json] [--workers=4]")
        print("\nExample:")
        print("  python batch_processing.py ./trajectories/")
        print("  python batch_processing.py ./data/ --pattern='*.jsonl' --workers=8")
        sys.exit(1)

    directory = sys.argv[1]
    pattern = "*.json"
    workers = 4

    for arg in sys.argv[2:]:
        if arg.startswith("--pattern="):
            pattern = arg.split("=", 1)[1]
        elif arg.startswith("--workers="):
            workers = int(arg.split("=", 1)[1])

    # Check directory exists
    if not Path(directory).is_dir():
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)

    # Process files
    results = batch_process(directory, pattern, workers)

    # Generate and print report
    report = generate_report(results)

    print("\n" + "=" * 50)
    print("BATCH PROCESSING REPORT")
    print("=" * 50)
    print(f"Total files: {report['total_files']}")
    print(f"Successful: {report['successful']}")
    print(f"Errors: {report['errors']}")
    print(f"\nHacks detected: {report['hacks_detected']}")
    print(f"Hack rate: {report['hack_rate']:.1%}")
    print(f"Average hack score: {report['average_hack_score']:.2f}")

    print("\nRisk Distribution:")
    for level, count in sorted(report["risk_distribution"].items()):
        print(f"  {level}: {count}")


if __name__ == "__main__":
    main()
