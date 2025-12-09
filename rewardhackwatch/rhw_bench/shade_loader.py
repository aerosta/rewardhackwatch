"""
SHADE-Arena External Benchmark Loader.

Loads trajectories from ScaleAI's Monitor Red Teaming (MRT) dataset.
Dataset: https://huggingface.co/datasets/ScaleAI/mrt

Converts SHADE-Arena transcripts to RewardHackWatch trajectory schema.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@dataclass
class ShadeTrajectory:
    """Converted SHADE-Arena trajectory."""

    name: str
    task_name: str
    attack_model: str
    monitor_model: str
    is_hack: bool  # Based on secondary (malicious) objective presence
    transcript: str
    monitor_verdict: str
    thinking_traces: list[str]
    actions: list[str]
    trajectory: dict[str, Any]  # RHW-compatible format


class ShadeArenaLoader:
    """
    Load and convert SHADE-Arena trajectories from HuggingFace.

    Usage:
        loader = ShadeArenaLoader()
        trajectories = loader.load(limit=100)

        for traj in trajectories:
            result = runner.run_single(traj.trajectory)
    """

    DATASET_NAME = "ScaleAI/mrt"

    def __init__(self, cache_dir: str | Path | None = None):
        """
        Initialize loader.

        Args:
            cache_dir: Optional cache directory for downloaded data
        """
        if not HF_AVAILABLE:
            raise ImportError("datasets package required. Install with: pip install datasets")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._dataset = None

    def load(
        self,
        limit: int | None = None,
        task_filter: str | None = None,
        include_clean: bool = True,
    ) -> list[ShadeTrajectory]:
        """
        Load SHADE-Arena trajectories.

        Args:
            limit: Maximum number of trajectories to load
            task_filter: Only load from specific task (e.g., "banking")
            include_clean: Include non-attack trajectories

        Returns:
            List of converted ShadeTrajectory objects
        """
        print(f"Loading SHADE-Arena dataset from {self.DATASET_NAME}...")

        # Try file-based approach first (more reliable for this dataset)
        trajectories = self._load_from_hub_files(limit, task_filter, include_clean)

        if trajectories:
            return trajectories

        # Fallback to datasets library
        try:
            dataset = load_dataset(
                self.DATASET_NAME,
                split="train",
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Attempting to load from local cache...")
            return self._load_from_files(limit, task_filter, include_clean)

        trajectories = []
        count = 0

        for item in dataset:
            if limit and count >= limit:
                break

            try:
                traj = self._convert_item(item)
                if traj:
                    if task_filter and task_filter.lower() not in traj.task_name.lower():
                        continue
                    if not include_clean and not traj.is_hack:
                        continue
                    trajectories.append(traj)
                    count += 1
            except Exception as e:
                print(f"Warning: Failed to convert item: {e}")
                continue

        print(f"Loaded {len(trajectories)} SHADE-Arena trajectories")
        return trajectories

    def _load_from_hub_files(
        self,
        limit: int | None,
        task_filter: str | None,
        include_clean: bool,
    ) -> list[ShadeTrajectory]:
        """Load directly from HuggingFace Hub files."""
        try:
            # List transcript files in the repo
            files = list_repo_files(self.DATASET_NAME, repo_type="dataset")
            transcript_files = [f for f in files if f.endswith(".json") and "transcript" in f]

            if not transcript_files:
                print("No transcript files found in dataset repo")
                return []

            # Limit the number of files to download
            if limit:
                transcript_files = transcript_files[:limit]

            trajectories = []
            for file_path in transcript_files:
                if limit and len(trajectories) >= limit:
                    break

                try:
                    # Download individual file
                    local_path = hf_hub_download(
                        self.DATASET_NAME,
                        file_path,
                        repo_type="dataset",
                        cache_dir=str(self.cache_dir) if self.cache_dir else None,
                    )

                    with open(local_path) as f:
                        data = json.load(f)

                    # Parse path for metadata
                    parts = file_path.split("/")
                    task_name = parts[1] if len(parts) > 1 else "unknown"
                    attack_model = parts[2] if len(parts) > 2 else "unknown"
                    monitor_model = parts[3] if len(parts) > 3 else "unknown"

                    traj = self._convert_transcript(
                        data,
                        name=Path(file_path).stem,
                        task_name=task_name,
                        attack_model=attack_model,
                        monitor_model=monitor_model,
                    )

                    if traj:
                        if task_filter and task_filter.lower() not in traj.task_name.lower():
                            continue
                        if not include_clean and not traj.is_hack:
                            continue
                        trajectories.append(traj)

                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
                    continue

            print(f"Loaded {len(trajectories)} SHADE-Arena trajectories from Hub")
            return trajectories

        except Exception as e:
            print(f"Failed to load from Hub: {e}")
            return []

    def _load_from_files(
        self,
        limit: int | None,
        task_filter: str | None,
        include_clean: bool,
    ) -> list[ShadeTrajectory]:
        """Fallback: load from downloaded JSON files."""
        # Try to find cached files
        if self.cache_dir and self.cache_dir.exists():
            json_files = list(self.cache_dir.rglob("transcript_*.json"))
        else:
            # Try default HF cache location
            hf_cache = Path.home() / ".cache" / "huggingface" / "datasets" / "ScaleAI___mrt"
            json_files = list(hf_cache.rglob("transcript_*.json")) if hf_cache.exists() else []

        if not json_files:
            print("No cached SHADE-Arena files found. Please ensure dataset is downloaded.")
            return []

        trajectories = []
        count = 0

        for file_path in json_files:
            if limit and count >= limit:
                break

            try:
                with open(file_path) as f:
                    data = json.load(f)

                # Parse path for metadata
                # Format: task_pairs/<task_name>/<attack_model>/<monitor_model>/transcript_<#>.json
                parts = file_path.parts
                task_name = "unknown"
                attack_model = "unknown"
                monitor_model = "unknown"

                for i, part in enumerate(parts):
                    if part == "task_pairs" and i + 3 < len(parts):
                        task_name = parts[i + 1]
                        attack_model = parts[i + 2]
                        monitor_model = parts[i + 3]
                        break

                traj = self._convert_transcript(
                    data,
                    name=file_path.stem,
                    task_name=task_name,
                    attack_model=attack_model,
                    monitor_model=monitor_model,
                )

                if traj:
                    if task_filter and task_filter.lower() not in traj.task_name.lower():
                        continue
                    if not include_clean and not traj.is_hack:
                        continue
                    trajectories.append(traj)
                    count += 1

            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue

        print(f"Loaded {len(trajectories)} SHADE-Arena trajectories from files")
        return trajectories

    def _convert_item(self, item: dict[str, Any]) -> ShadeTrajectory | None:
        """Convert a HuggingFace dataset item to ShadeTrajectory."""
        # Extract fields from dataset item
        transcript = item.get("transcript", "")
        monitor_transcript = item.get("monitor_transcript", "")
        monitor_verdict = item.get("monitor_verdict", "")
        task_desc = item.get("task_description", {})

        # Try to get metadata from item path or fields
        name = item.get("id", item.get("name", "shade_traj"))
        task_name = item.get("task_name", task_desc.get("task", "unknown"))
        attack_model = item.get("attack_model", "unknown")
        monitor_model = item.get("monitor_model", "unknown")

        return self._convert_transcript(
            {
                "transcript": transcript,
                "monitor_transcript": monitor_transcript,
                "monitor_verdict": monitor_verdict,
                "task_description": task_desc,
            },
            name=name,
            task_name=task_name,
            attack_model=attack_model,
            monitor_model=monitor_model,
        )

    def _convert_transcript(
        self,
        data: dict[str, Any],
        name: str,
        task_name: str,
        attack_model: str,
        monitor_model: str,
    ) -> ShadeTrajectory | None:
        """Convert a SHADE-Arena transcript to RHW format."""
        transcript = data.get("transcript", "")
        monitor_verdict = str(data.get("monitor_verdict", ""))
        task_desc = data.get("task_description", {})

        if not transcript:
            return None

        # Extract thinking traces (agent's internal reasoning)
        thinking_traces = self._extract_thinking(transcript)

        # Extract actions from transcript
        actions = self._extract_actions(transcript)

        # Determine if this is a hack attempt
        # SHADE-Arena has secondary (malicious) objectives that agents try to complete
        # If the monitor verdict indicates suspicious behavior, or the task has secondary objectives
        is_hack = self._determine_is_hack(data, monitor_verdict)

        # Build RHW-compatible trajectory
        trajectory = {
            "name": name,
            "source": "shade_arena",
            "task_name": task_name,
            "steps": self._build_steps(transcript),
            "cot_traces": thinking_traces,
            "code_outputs": [],  # SHADE-Arena is tool-use, not code
            "actions": actions,
            "labels": {
                "hack_type": "shade_attack" if is_hack else "clean",
                "is_hack": is_hack,
                "monitor_verdict": monitor_verdict,
            },
            "metadata": {
                "attack_model": attack_model,
                "monitor_model": monitor_model,
                "task_description": task_desc,
            },
            "expected_hack": is_hack,
        }

        return ShadeTrajectory(
            name=name,
            task_name=task_name,
            attack_model=attack_model,
            monitor_model=monitor_model,
            is_hack=is_hack,
            transcript=transcript,
            monitor_verdict=monitor_verdict,
            thinking_traces=thinking_traces,
            actions=actions,
            trajectory=trajectory,
        )

    def _extract_thinking(self, transcript: str) -> list[str]:
        """Extract <thinking> blocks from transcript."""
        # Match <thinking>...</thinking> blocks
        pattern = r"<thinking>(.*?)</thinking>"
        matches = re.findall(pattern, transcript, re.DOTALL | re.IGNORECASE)
        return [m.strip() for m in matches if m.strip()]

    def _extract_actions(self, transcript: str) -> list[str]:
        """Extract action/tool calls from transcript."""
        actions = []

        # Common patterns for tool calls in agentic transcripts
        # Pattern 1: function_call or tool_call blocks
        tool_pattern = r"(?:function_call|tool_call|action):\s*(\w+)"
        matches = re.findall(tool_pattern, transcript, re.IGNORECASE)
        actions.extend(matches)

        # Pattern 2: JSON-like tool invocations
        json_pattern = r'"(?:function|tool|action)":\s*"([^"]+)"'
        matches = re.findall(json_pattern, transcript)
        actions.extend(matches)

        return actions

    def _build_steps(self, transcript: str) -> list[dict[str, Any]]:
        """Build step-by-step representation from transcript."""
        steps = []

        # Split transcript into conversation turns
        # Look for common patterns like "User:", "Assistant:", message blocks
        turn_pattern = (
            r"(User|Assistant|Human|AI|Agent):\s*(.*?)(?=(?:User|Assistant|Human|AI|Agent):|$)"
        )
        matches = re.findall(turn_pattern, transcript, re.DOTALL | re.IGNORECASE)

        for i, (role, content) in enumerate(matches):
            content = content.strip()
            if not content:
                continue

            # Extract thinking from this turn
            thinking = ""
            thinking_match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
            if thinking_match:
                thinking = thinking_match.group(1).strip()
                content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL).strip()

            steps.append(
                {
                    "step": i,
                    "role": role.lower(),
                    "content": content,
                    "thinking": thinking,
                }
            )

        # If pattern matching fails, treat whole transcript as one step
        if not steps:
            steps.append(
                {
                    "step": 0,
                    "role": "agent",
                    "content": transcript[:2000],  # Truncate for safety
                    "thinking": "",
                }
            )

        return steps

    def _determine_is_hack(self, data: dict[str, Any], monitor_verdict: str) -> bool:
        """Determine if trajectory represents a hack/attack attempt."""
        # Check monitor verdict for suspicious/harmful indicators
        verdict_lower = monitor_verdict.lower()
        if any(
            word in verdict_lower
            for word in [
                "suspicious",
                "harmful",
                "malicious",
                "attack",
                "violation",
                "unsafe",
                "detected",
                "flagged",
            ]
        ):
            return True

        # Check task description for secondary (malicious) objectives
        task_desc = data.get("task_description", {})
        if isinstance(task_desc, dict):
            secondary = task_desc.get("secondary_objective", "")
            if secondary and "none" not in secondary.lower():
                # Has a secondary objective = attack scenario
                return True

        # Default to clean if no indicators
        return False

    def save_as_rhw_format(
        self,
        trajectories: list[ShadeTrajectory],
        output_dir: str | Path,
    ) -> int:
        """
        Save SHADE-Arena trajectories in RHW-Bench format.

        Args:
            trajectories: List of ShadeTrajectory objects
            output_dir: Output directory for JSON files

        Returns:
            Number of files saved
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for traj in trajectories:
            filename = f"shade_{traj.name}_{traj.task_name}.json"
            # Clean filename
            filename = re.sub(r"[^\w\-_.]", "_", filename)

            output_path = output_dir / filename
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "name": traj.name,
                        "category": "shade_arena",
                        "expected_hack": traj.is_hack,
                        "trajectory": traj.trajectory,
                        "labels": traj.trajectory.get("labels", {}),
                    },
                    f,
                    indent=2,
                )
            saved += 1

        print(f"Saved {saved} trajectories to {output_dir}")
        return saved


def main():
    """CLI for loading SHADE-Arena data."""
    import argparse

    parser = argparse.ArgumentParser(description="Load SHADE-Arena benchmark data")
    parser.add_argument("--limit", type=int, default=50, help="Max trajectories to load")
    parser.add_argument("--output", type=str, default="data/shade_arena", help="Output directory")
    parser.add_argument("--task", type=str, help="Filter by task name")
    args = parser.parse_args()

    loader = ShadeArenaLoader()
    trajectories = loader.load(
        limit=args.limit,
        task_filter=args.task,
    )

    if trajectories:
        loader.save_as_rhw_format(trajectories, args.output)

        # Print summary
        hacks = sum(1 for t in trajectories if t.is_hack)
        print("\nSummary:")
        print(f"  Total: {len(trajectories)}")
        print(f"  Hacks: {hacks}")
        print(f"  Clean: {len(trajectories) - hacks}")
    else:
        print("No trajectories loaded. Check your internet connection or HuggingFace access.")


if __name__ == "__main__":
    main()
