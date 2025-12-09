"""Text processing utilities for trajectory analysis."""

import re


def extract_code_blocks(text: str) -> list[str]:
    """Extract code blocks from markdown/text."""
    pattern = r"```(?:python)?\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)


def extract_thinking_blocks(text: str) -> list[str]:
    """Extract <thinking> blocks from CoT."""
    pattern = r"<thinking>(.*?)</thinking>"
    return re.findall(pattern, text, re.DOTALL)


def normalize_code(code: str) -> str:
    """Normalize code for comparison."""
    return re.sub(r"\s+", " ", code.strip())


def count_tokens_approx(text: str) -> int:
    """Approximate token count (words * 1.3)."""
    return int(len(text.split()) * 1.3)


def strip_comments(code: str) -> str:
    """Remove comments from Python code."""
    lines = []
    for line in code.split("\n"):
        # Remove inline comments
        if "#" in line:
            line = line.split("#")[0].rstrip()
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


def extract_function_names(code: str) -> list[str]:
    """Extract function names from Python code."""
    pattern = r"def\s+(\w+)\s*\("
    return re.findall(pattern, code)


def extract_class_names(code: str) -> list[str]:
    """Extract class names from Python code."""
    pattern = r"class\s+(\w+)\s*[:\(]"
    return re.findall(pattern, code)
