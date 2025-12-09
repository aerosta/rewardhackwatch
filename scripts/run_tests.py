#!/usr/bin/env python3
"""
Run Tests

Runs the test suite with coverage reporting.
"""

import argparse
import subprocess
import sys


def run_tests(
    test_path: str = "tests/",
    verbose: bool = False,
    coverage: bool = False,
    markers: str = None,
    pattern: str = None,
) -> int:
    """
    Run tests with pytest.

    Args:
        test_path: Path to tests
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        markers: Pytest markers to filter
        pattern: Pattern to match test names

    Returns:
        Exit code
    """
    cmd = [sys.executable, "-m", "pytest", test_path]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(
            [
                "--cov=rewardhackwatch",
                "--cov-report=term-missing",
                "--cov-report=html",
            ]
        )

    if markers:
        cmd.extend(["-m", markers])

    if pattern:
        cmd.extend(["-k", pattern])

    # Add common options
    cmd.extend(
        [
            "--tb=short",
            "-x",  # Stop on first failure
        ]
    )

    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd)

    return result.returncode


def run_quick_tests() -> int:
    """Run quick smoke tests."""
    return run_tests(
        test_path="tests/",
        pattern="not slow and not integration",
        verbose=True,
    )


def run_all_tests() -> int:
    """Run all tests with coverage."""
    return run_tests(
        test_path="tests/",
        verbose=True,
        coverage=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Run tests")
    parser.add_argument(
        "test_path",
        type=str,
        nargs="?",
        default="tests/",
        help="Path to tests",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable coverage",
    )
    parser.add_argument(
        "-m",
        "--markers",
        type=str,
        help="Pytest markers",
    )
    parser.add_argument(
        "-k",
        "--pattern",
        type=str,
        help="Test name pattern",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests with coverage",
    )

    args = parser.parse_args()

    if args.quick:
        exit_code = run_quick_tests()
    elif args.all:
        exit_code = run_all_tests()
    else:
        exit_code = run_tests(
            args.test_path,
            args.verbose,
            args.coverage,
            args.markers,
            args.pattern,
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
