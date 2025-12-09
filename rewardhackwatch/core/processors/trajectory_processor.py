"""Process individual trajectories."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ProcessedTrajectory:
    """A processed trajectory ready for analysis."""

    id: str
    cot_traces: list[str]
    code_outputs: list[str]
    steps: list[dict[str, Any]]
    metadata: dict[str, Any]
    normalized: bool = True


class TrajectoryProcessor:
    """Process and normalize trajectories for analysis."""

    def __init__(self, normalize: bool = True, extract_code: bool = True):
        self.normalize = normalize
        self.extract_code = extract_code

    def process(self, trajectory: dict[str, Any]) -> ProcessedTrajectory:
        """Process a single trajectory."""
        traj_id = trajectory.get("id", "unknown")

        # Extract CoT traces
        cot_traces = self._extract_cot(trajectory)

        # Extract code outputs
        code_outputs = self._extract_code(trajectory)

        # Extract steps
        steps = trajectory.get("steps", [])

        # Build metadata
        metadata = {
            "original_keys": list(trajectory.keys()),
            "cot_count": len(cot_traces),
            "code_count": len(code_outputs),
            "step_count": len(steps),
        }

        return ProcessedTrajectory(
            id=traj_id,
            cot_traces=cot_traces,
            code_outputs=code_outputs,
            steps=steps,
            metadata=metadata,
            normalized=self.normalize,
        )

    def _extract_cot(self, trajectory: dict[str, Any]) -> list[str]:
        """Extract chain-of-thought traces."""
        cot_traces = []

        # Direct field
        if "cot_traces" in trajectory:
            traces = trajectory["cot_traces"]
            if isinstance(traces, list):
                cot_traces.extend(traces)
            elif isinstance(traces, str):
                cot_traces.append(traces)

        # From steps
        for step in trajectory.get("steps", []):
            if step.get("type") == "thinking":
                content = step.get("content", "")
                if content:
                    cot_traces.append(content)

        if self.normalize:
            cot_traces = [self._normalize_text(t) for t in cot_traces]

        return cot_traces

    def _extract_code(self, trajectory: dict[str, Any]) -> list[str]:
        """Extract code outputs."""
        code_outputs = []

        # Direct field
        if "code_outputs" in trajectory:
            outputs = trajectory["code_outputs"]
            if isinstance(outputs, list):
                code_outputs.extend(outputs)
            elif isinstance(outputs, str):
                code_outputs.append(outputs)

        # From steps
        for step in trajectory.get("steps", []):
            if step.get("type") == "code":
                content = step.get("content", "")
                if content:
                    code_outputs.append(content)

        return code_outputs

    def _normalize_text(self, text: str) -> str:
        """Normalize text for analysis."""
        # Remove extra whitespace
        text = " ".join(text.split())
        # Convert to lowercase for comparison
        return text.strip()

    def batch_process(self, trajectories: list[dict[str, Any]]) -> list[ProcessedTrajectory]:
        """Process multiple trajectories."""
        return [self.process(t) for t in trajectories]
