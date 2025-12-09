"""Summarize trajectory for analysis."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TrajectorySummary:
    """Summary of a trajectory."""

    trajectory_id: str
    total_steps: int
    has_code: bool
    has_thinking: bool
    estimated_tokens: int
    code_blocks: int
    thinking_blocks: int
    unique_patterns: list[str]


class TrajectorySummarizer:
    """Create summaries of agent trajectories."""

    def __init__(self, max_preview_length: int = 200):
        self.max_preview_length = max_preview_length

    def summarize(self, trajectory: dict[str, Any]) -> TrajectorySummary:
        """Create trajectory summary."""
        steps = trajectory.get("steps", [])
        cot_traces = trajectory.get("cot_traces", [])
        code_outputs = trajectory.get("code_outputs", [])

        # Count content types
        all_content = str(steps) + str(cot_traces) + str(code_outputs)
        has_code = any(
            [
                "code" in all_content.lower(),
                "def " in all_content,
                "class " in all_content,
                "import " in all_content,
            ]
        )
        has_thinking = any(
            [
                "thinking" in all_content.lower(),
                "<thinking>" in all_content,
            ]
        )

        # Estimate tokens
        total_text = " ".join(
            [str(s) for s in steps] + list(cot_traces or []) + list(code_outputs or [])
        )
        estimated_tokens = int(len(total_text.split()) * 1.3)

        return TrajectorySummary(
            trajectory_id=trajectory.get("id", "unknown"),
            total_steps=len(steps),
            has_code=has_code,
            has_thinking=has_thinking,
            estimated_tokens=estimated_tokens,
            code_blocks=len(code_outputs or []),
            thinking_blocks=len(cot_traces or []),
            unique_patterns=self._extract_patterns(trajectory),
        )

    def _extract_patterns(self, trajectory: dict[str, Any]) -> list[str]:
        """Extract unique patterns from trajectory."""
        patterns = set()
        code_outputs = trajectory.get("code_outputs", []) or []

        for code in code_outputs:
            if not isinstance(code, str):
                continue
            if "sys.exit" in code:
                patterns.add("sys_exit")
            if "__eq__" in code:
                patterns.add("eq_override")
            if "os.system" in code:
                patterns.add("os_system")
            if "subprocess" in code:
                patterns.add("subprocess")
            if "eval(" in code:
                patterns.add("eval")
            if "exec(" in code:
                patterns.add("exec")

        return list(patterns)

    def extract_key_moments(self, trajectory: dict[str, Any]) -> list[dict]:
        """Extract key moments from trajectory."""
        moments = []
        steps = trajectory.get("steps", [])

        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                continue

            content = str(step.get("content", ""))

            # Check for significant events
            if any(kw in content.lower() for kw in ["error", "exception", "failed"]):
                moments.append(
                    {"step": i, "type": "error", "preview": content[: self.max_preview_length]}
                )
            elif any(kw in content for kw in ["sys.exit", "os._exit"]):
                moments.append(
                    {"step": i, "type": "suspicious", "preview": content[: self.max_preview_length]}
                )

        return moments

    def get_code_preview(self, trajectory: dict[str, Any]) -> Optional[str]:
        """Get a preview of the code in the trajectory."""
        code_outputs = trajectory.get("code_outputs", []) or []
        if code_outputs:
            first_code = str(code_outputs[0])
            return (
                first_code[: self.max_preview_length] + "..."
                if len(first_code) > self.max_preview_length
                else first_code
            )
        return None
