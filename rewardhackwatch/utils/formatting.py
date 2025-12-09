"""Output formatting utilities."""

from typing import Any


def format_score_bar(score: float, width: int = 10) -> str:
    """Create visual score bar."""
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def format_detection_result(result: dict[str, Any]) -> str:
    """Format detection result for display."""
    score = result.get("score", 0)
    bar = format_score_bar(score)
    level = "HIGH" if score >= 0.7 else "MEDIUM" if score >= 0.4 else "LOW"
    return f"{score:.2f} [{bar}] {level}"


def format_table(headers: list, rows: list) -> str:
    """Format data as ASCII table."""
    if not rows:
        return "No data"

    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    separator = "-+-".join("-" * w for w in widths)
    row_lines = [" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))) for r in rows]
    return "\n".join([header_line, separator] + row_lines)


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_risk_badge(level: str) -> str:
    """Format risk level as colored badge (ANSI)."""
    colors = {
        "CRITICAL": "\033[91m",  # Red
        "HIGH": "\033[93m",  # Yellow
        "MEDIUM": "\033[33m",  # Orange
        "LOW": "\033[92m",  # Green
        "NONE": "\033[90m",  # Gray
    }
    reset = "\033[0m"
    color = colors.get(level.upper(), "")
    return f"{color}[{level}]{reset}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"
