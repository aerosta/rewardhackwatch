#!/usr/bin/env python3
"""
Generate Documentation

Generates API documentation from docstrings.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def generate_sphinx_docs(output_dir: str = "./docs/api") -> int:
    """Generate Sphinx documentation."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if Sphinx is installed
    try:
        import sphinx
    except ImportError:
        print("Sphinx not installed. Run: pip install sphinx sphinx-rtd-theme")
        return 1

    # Generate API docs
    cmd = [
        sys.executable,
        "-m",
        "sphinx.ext.apidoc",
        "-o",
        str(output_path),
        "-f",  # Force overwrite
        "rewardhackwatch/",
    ]

    print("Generating API documentation...")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        return result.returncode

    # Build HTML
    cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
        str(output_path),
        str(output_path / "_build" / "html"),
    ]

    print("Building HTML documentation...")
    result = subprocess.run(cmd)

    return result.returncode


def generate_markdown_docs(output_dir: str = "./docs/api") -> int:
    """Generate markdown documentation."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Walk through modules
    rewardhackwatch_path = Path("rewardhackwatch")

    for py_file in rewardhackwatch_path.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue

        # Create markdown file
        relative = py_file.relative_to(rewardhackwatch_path)
        md_path = output_path / relative.with_suffix(".md")
        md_path.parent.mkdir(parents=True, exist_ok=True)

        module_name = str(relative.with_suffix("")).replace("/", ".")

        with open(md_path, "w") as f:
            f.write(f"# {module_name}\n\n")
            f.write(f"Module: `rewardhackwatch.{module_name}`\n\n")
            f.write("---\n\n")
            f.write(f"Source: `{py_file}`\n")

    print(f"Generated markdown docs in {output_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Generate documentation")
    parser.add_argument(
        "--output",
        type=str,
        default="./docs/api",
        help="Output directory",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["sphinx", "markdown"],
        default="markdown",
        help="Documentation format",
    )

    args = parser.parse_args()

    if args.format == "sphinx":
        exit_code = generate_sphinx_docs(args.output)
    else:
        exit_code = generate_markdown_docs(args.output)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
