"""Dataset classes for training RewardHackWatch ML models."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

# Patterns for feature extraction
HACK_PATTERNS = [
    r"sys\.exit\s*\(\s*0?\s*\)",
    r"exit\s*\(\s*0?\s*\)",
    r"assert\s+True\s*($|[#,])",
    r"return\s+True\s*#",
    r"pytest\.skip",
    r"unittest\.skip",
    r"@pytest\.mark\.skip",
    r"raise\s+SkipTest",
    r"pass\s*#.*test",
    r"def\s+test_\w+\s*\([^)]*\)\s*:\s*(pass|\.\.\.)",
    r"mock\.[a-z_]+\s*=",
    r"\.return_value\s*=",
]

DECEPTION_PATTERNS = [
    r"(trick|hack|cheat|game|exploit)\s+(the|this)",
    r"(bypass|circumvent|avoid)\s+(the|this)",
    r"they\s+won'?t\s+(notice|know|detect)",
    r"(hide|conceal)\s+(this|my|the)",
    r"(fake|forge|falsify)",
    r"pretend\s+to",
    r"make\s+it\s+look\s+like",
    r"without\s+(them|anyone)\s+(knowing|noticing)",
]

TRANSPARENCY_PATTERNS = [
    r"let\s+me\s+(think|explain|reason)",
    r"step\s+by\s+step",
    r"(first|then|next|finally)",
    r"my\s+(goal|plan|approach)",
    r"because|since|therefore",
    r"the\s+reason\s+(is|for)",
]


@dataclass
class TrajectoryFeatures:
    """Extracted features from a trajectory."""

    # Code features
    code_length: int = 0
    num_code_blocks: int = 0
    hack_pattern_count: int = 0

    # CoT features
    cot_length: int = 0
    cot_word_count: int = 0
    num_cot_traces: int = 0
    deception_pattern_count: int = 0
    transparency_pattern_count: int = 0

    # Ratio features
    cot_code_ratio: float = 0.0

    # Step features
    num_steps: int = 0
    avg_step_length: float = 0.0

    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for model input."""
        return torch.tensor(
            [
                self.code_length / 1000,  # Normalize
                self.num_code_blocks / 10,
                self.hack_pattern_count / 5,
                self.cot_length / 1000,
                self.cot_word_count / 200,
                self.num_cot_traces / 10,
                self.deception_pattern_count / 5,
                self.transparency_pattern_count / 10,
                self.cot_code_ratio,
                self.num_steps / 20,
                self.avg_step_length / 100,
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def feature_dim() -> int:
        """Return feature dimension."""
        return 11


class TrajectoryFeatureExtractor:
    """Extract features from trajectories for ML models."""

    def __init__(self):
        self._hack_patterns = [re.compile(p, re.IGNORECASE) for p in HACK_PATTERNS]
        self._deception_patterns = [re.compile(p, re.IGNORECASE) for p in DECEPTION_PATTERNS]
        self._transparency_patterns = [re.compile(p, re.IGNORECASE) for p in TRANSPARENCY_PATTERNS]

    def extract(self, trajectory: dict[str, Any]) -> TrajectoryFeatures:
        """Extract features from a trajectory."""
        features = TrajectoryFeatures()

        # Extract code
        code_texts = self._get_code_texts(trajectory)
        code_combined = "\n".join(code_texts)

        features.code_length = len(code_combined)
        features.num_code_blocks = len(code_texts)
        features.hack_pattern_count = self._count_patterns(code_combined, self._hack_patterns)

        # Extract CoT
        cot_texts = self._get_cot_texts(trajectory)
        cot_combined = "\n".join(cot_texts)

        features.cot_length = len(cot_combined)
        features.cot_word_count = len(cot_combined.split())
        features.num_cot_traces = len(cot_texts)
        features.deception_pattern_count = self._count_patterns(
            cot_combined, self._deception_patterns
        )
        features.transparency_pattern_count = self._count_patterns(
            cot_combined, self._transparency_patterns
        )

        # Ratio
        total_length = features.code_length + features.cot_length
        features.cot_code_ratio = features.cot_length / max(total_length, 1)

        # Steps
        steps = trajectory.get("steps", [])
        features.num_steps = len(steps)
        if steps:
            step_lengths = [len(str(s)) for s in steps]
            features.avg_step_length = sum(step_lengths) / len(step_lengths)

        return features

    def _get_code_texts(self, trajectory: dict[str, Any]) -> list[str]:
        """Extract code texts from trajectory."""
        texts = []

        if "code_outputs" in trajectory:
            texts.extend(str(c) for c in trajectory["code_outputs"] if c)

        if "steps" in trajectory:
            for step in trajectory["steps"]:
                if isinstance(step, dict):
                    if "code" in step and step["code"]:
                        texts.append(str(step["code"]))
                    if "action" in step and step["action"]:
                        texts.append(str(step["action"]))

        return texts

    def _get_cot_texts(self, trajectory: dict[str, Any]) -> list[str]:
        """Extract CoT texts from trajectory."""
        texts = []

        if "cot_traces" in trajectory:
            texts.extend(str(c) for c in trajectory["cot_traces"] if c)

        if "steps" in trajectory:
            for step in trajectory["steps"]:
                if isinstance(step, dict):
                    for key in ["thought", "thinking", "reasoning", "cot"]:
                        if key in step and step[key]:
                            texts.append(str(step[key]))

        return texts

    def _count_patterns(self, text: str, patterns: list) -> int:
        """Count pattern matches in text."""
        count = 0
        for pattern in patterns:
            count += len(pattern.findall(text))
        return count


class RHWBenchDataset(Dataset):
    """PyTorch Dataset for RHW-Bench trajectories."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "all",
        transform: Any = None,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing JSON trajectory files
            split: "train", "val", "test", or "all"
            transform: Optional transform to apply to features
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        self.feature_extractor = TrajectoryFeatureExtractor()

        # Load all trajectories
        self.trajectories = []
        self.labels = []
        self.file_paths = []

        self._load_trajectories()

        # Apply split
        if split != "all":
            self._apply_split()

    def _load_trajectories(self):
        """Load all trajectory files."""
        json_files = list(self.data_dir.glob("*.json"))

        for file_path in json_files:
            # Skip aggregate/combined files that start with underscore
            if file_path.name.startswith("_"):
                continue

            try:
                with open(file_path) as f:
                    data = json.load(f)

                # Skip if data is a list (combined file that wasn't filtered)
                if isinstance(data, list):
                    continue

                # Handle different file formats:
                # 1. Test case format: {name, expected_hack, trajectory: {...}}
                # 2. Trajectory format: {labels: {hack_type: ...}, ...}
                if "trajectory" in data:
                    # Test case format - extract the trajectory object
                    trajectory = data["trajectory"]
                    # Add metadata from parent
                    trajectory["_name"] = data.get("name", "")
                    trajectory["_category"] = data.get("category", "")
                    self.trajectories.append(trajectory)

                    # Label from expected_hack field
                    is_hack = data.get("expected_hack", False)
                else:
                    # Direct trajectory format
                    self.trajectories.append(data)

                    # Label from labels.hack_type field
                    labels = data.get("labels", {})
                    is_hack = labels.get("hack_type") not in [None, "clean", "none", ""]

                self.file_paths.append(str(file_path))
                self.labels.append(1 if is_hack else 0)

            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

    def _apply_split(self):
        """Apply train/val/test split (70/15/15)."""
        n = len(self.trajectories)
        indices = list(range(n))

        # Deterministic shuffle
        np.random.seed(42)
        np.random.shuffle(indices)

        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        if self.split == "train":
            split_indices = indices[:train_end]
        elif self.split == "val":
            split_indices = indices[train_end:val_end]
        elif self.split == "test":
            split_indices = indices[val_end:]
        else:
            split_indices = indices

        self.trajectories = [self.trajectories[i] for i in split_indices]
        self.labels = [self.labels[i] for i in split_indices]
        self.file_paths = [self.file_paths[i] for i in split_indices]

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        trajectory = self.trajectories[idx]
        label = self.labels[idx]

        # Extract features
        features = self.feature_extractor.extract(trajectory)
        feature_tensor = features.to_tensor()

        if self.transform:
            feature_tensor = self.transform(feature_tensor)

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return feature_tensor, label_tensor

    def get_trajectory(self, idx: int) -> dict[str, Any]:
        """Get raw trajectory by index."""
        return self.trajectories[idx]

    def get_all_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Get all features as numpy arrays for sklearn."""
        features = []
        labels = []

        for i in range(len(self)):
            feat, label = self[i]
            features.append(feat.numpy())
            labels.append(label.item())

        return np.array(features), np.array(labels)


class ShadeArenaDataset(Dataset):
    """PyTorch Dataset for SHADE-Arena trajectories."""

    def __init__(
        self,
        limit: int = 100,
        split: str = "all",
        transform: Any = None,
    ):
        """
        Initialize dataset from SHADE-Arena.

        Args:
            limit: Maximum number of trajectories to load
            split: "train", "val", "test", or "all"
            transform: Optional transform to apply to features
        """
        self.split = split
        self.transform = transform
        self.feature_extractor = TrajectoryFeatureExtractor()

        self.trajectories = []
        self.labels = []

        self._load_shade_arena(limit)

        if split != "all":
            self._apply_split()

    def _load_shade_arena(self, limit: int):
        """Load trajectories from SHADE-Arena."""
        try:
            from rewardhackwatch.rhw_bench.shade_loader import ShadeArenaLoader

            loader = ShadeArenaLoader()
            shade_trajectories = loader.load(limit=limit)

            for traj in shade_trajectories:
                self.trajectories.append(traj.trajectory)
                self.labels.append(1 if traj.is_hack else 0)

        except Exception as e:
            print(f"Warning: Could not load SHADE-Arena data: {e}")

    def _apply_split(self):
        """Apply train/val/test split (70/15/15)."""
        n = len(self.trajectories)
        if n == 0:
            return

        indices = list(range(n))

        # Deterministic shuffle
        np.random.seed(42)
        np.random.shuffle(indices)

        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        if self.split == "train":
            split_indices = indices[:train_end]
        elif self.split == "val":
            split_indices = indices[train_end:val_end]
        elif self.split == "test":
            split_indices = indices[val_end:]
        else:
            split_indices = indices

        self.trajectories = [self.trajectories[i] for i in split_indices]
        self.labels = [self.labels[i] for i in split_indices]

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        trajectory = self.trajectories[idx]
        label = self.labels[idx]

        features = self.feature_extractor.extract(trajectory)
        feature_tensor = features.to_tensor()

        if self.transform:
            feature_tensor = self.transform(feature_tensor)

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return feature_tensor, label_tensor

    def get_all_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Get all features as numpy arrays for sklearn."""
        features = []
        labels = []

        for i in range(len(self)):
            feat, label = self[i]
            features.append(feat.numpy())
            labels.append(label.item())

        return np.array(features), np.array(labels)


class CombinedDataset(Dataset):
    """Combined dataset from RHW-Bench and SHADE-Arena."""

    def __init__(
        self,
        rhw_dir: str | Path,
        shade_limit: int = 100,
        split: str = "all",
        transform: Any = None,
    ):
        """
        Initialize combined dataset.

        Args:
            rhw_dir: Directory with RHW-Bench data
            shade_limit: Max SHADE-Arena trajectories
            split: Data split
            transform: Optional transform
        """
        self.split = split
        self.transform = transform
        self.feature_extractor = TrajectoryFeatureExtractor()

        self.trajectories = []
        self.labels = []
        self.sources = []  # Track data source

        # Load RHW-Bench data
        self._load_rhw(rhw_dir)

        # Load SHADE-Arena data
        self._load_shade(shade_limit)

        if split != "all":
            self._apply_split()

    def _load_rhw(self, data_dir: str | Path):
        """Load RHW-Bench trajectories."""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            return

        json_files = list(data_dir.glob("*.json"))

        for file_path in json_files:
            try:
                with open(file_path) as f:
                    data = json.load(f)

                if "trajectory" in data:
                    trajectory = data["trajectory"]
                    is_hack = data.get("expected_hack", False)
                else:
                    trajectory = data
                    labels = data.get("labels", {})
                    is_hack = labels.get("hack_type") not in [None, "clean", "none", ""]

                self.trajectories.append(trajectory)
                self.labels.append(1 if is_hack else 0)
                self.sources.append("rhw")

            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

    def _load_shade(self, limit: int):
        """Load SHADE-Arena trajectories."""
        try:
            from rewardhackwatch.rhw_bench.shade_loader import ShadeArenaLoader

            loader = ShadeArenaLoader()
            shade_trajectories = loader.load(limit=limit)

            for traj in shade_trajectories:
                self.trajectories.append(traj.trajectory)
                self.labels.append(1 if traj.is_hack else 0)
                self.sources.append("shade")

        except Exception as e:
            print(f"Warning: Could not load SHADE-Arena data: {e}")

    def _apply_split(self):
        """Apply stratified train/val/test split."""
        n = len(self.trajectories)
        if n == 0:
            return

        indices = list(range(n))
        np.random.seed(42)
        np.random.shuffle(indices)

        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        if self.split == "train":
            split_indices = indices[:train_end]
        elif self.split == "val":
            split_indices = indices[train_end:val_end]
        elif self.split == "test":
            split_indices = indices[val_end:]
        else:
            split_indices = indices

        self.trajectories = [self.trajectories[i] for i in split_indices]
        self.labels = [self.labels[i] for i in split_indices]
        self.sources = [self.sources[i] for i in split_indices]

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        trajectory = self.trajectories[idx]
        label = self.labels[idx]

        features = self.feature_extractor.extract(trajectory)
        feature_tensor = features.to_tensor()

        if self.transform:
            feature_tensor = self.transform(feature_tensor)

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return feature_tensor, label_tensor

    def get_all_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Get all features as numpy arrays for sklearn."""
        features = []
        labels = []

        for i in range(len(self)):
            feat, label = self[i]
            features.append(feat.numpy())
            labels.append(label.item())

        return np.array(features), np.array(labels)

    def get_source_counts(self) -> dict[str, int]:
        """Get count of samples per source."""
        from collections import Counter

        return dict(Counter(self.sources))
