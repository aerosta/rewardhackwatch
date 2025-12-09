#!/usr/bin/env python3
"""
Format Code

Formats code using Black and isort.
"""

import argparse
import subprocess
import sys


def run_black(paths: list, check: bool = False) -> int:
    """Run Black formatter."""
    cmd = [sys.executable, "-m", "black"]

    if check:
        cmd.append("--check")

    cmd.extend(paths)

    print("Running Black...")
    result = subprocess.run(cmd)
    return result.returncode


def run_isort(paths: list, check: bool = False) -> int:
    """Run isort."""
    cmd = [sys.executable, "-m", "isort"]

    if check:
        cmd.append("--check-only")

    cmd.extend(paths)

    print("Running isort...")
    result = subprocess.run(cmd)
    return result.returncode


def run_ruff(paths: list, fix: bool = False) -> int:
    """Run Ruff linter."""
    cmd = [sys.executable, "-m", "ruff", "check"]

    if fix:
        cmd.append("--fix")

    cmd.extend(paths)

    print("Running Ruff...")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Format code")
    parser.add_argument(
        "paths",
        type=str,
        nargs="*",
        default=["rewardhackwatch/", "tests/", "scripts/"],
        help="Paths to format",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check only, don't modify",
    )
    parser.add_argument(
        "--skip-black",
        action="store_true",
        help="Skip Black",
    )
    parser.add_argument(
        "--skip-isort",
        action="store_true",
        help="Skip isort",
    )
    parser.add_argument(
        "--skip-ruff",
        action="store_true",
        help="Skip Ruff",
    )

    args = parser.parse_args()

    exit_code = 0

    if not args.skip_black:
        code = run_black(args.paths, args.check)
        if code != 0:
            exit_code = code

    if not args.skip_isort:
        code = run_isort(args.paths, args.check)
        if code != 0:
            exit_code = code

    if not args.skip_ruff:
        code = run_ruff(args.paths, not args.check)
        if code != 0:
            exit_code = code

    if exit_code == 0:
        print("\n✅ All checks passed!")
    else:
        print("\n❌ Some checks failed")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
