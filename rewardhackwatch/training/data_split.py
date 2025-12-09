"""
Strict train/test data splitting for RHW-Bench.

Key principles:
1. Split by SCENARIO, not by file - trajectories from same scenario stay together
2. No leakage - test IDs must never appear in train
3. SHADE-Arena is eval-only (never train on external benchmarks)
4. Reproducible splits via seed
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataSplit:
    """Data split configuration and results."""

    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]

    train_scenarios: set[str] = field(default_factory=set)
    val_scenarios: set[str] = field(default_factory=set)
    test_scenarios: set[str] = field(default_factory=set)

    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "train_ids": self.train_ids,
            "val_ids": self.val_ids,
            "test_ids": self.test_ids,
            "train_scenarios": list(self.train_scenarios),
            "val_scenarios": list(self.val_scenarios),
            "test_scenarios": list(self.test_scenarios),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DataSplit:
        return cls(
            train_ids=d["train_ids"],
            val_ids=d["val_ids"],
            test_ids=d["test_ids"],
            train_scenarios=set(d.get("train_scenarios", [])),
            val_scenarios=set(d.get("val_scenarios", [])),
            test_scenarios=set(d.get("test_scenarios", [])),
            metadata=d.get("metadata", {}),
        )


class LeakageError(Exception):
    """Raised when train/test leakage is detected."""

    pass


def extract_scenario(trajectory_id: str) -> str:
    """
    Extract scenario identifier from trajectory ID.

    Scenario = hack_type + base_variant (without numeric suffix).
    Examples:
        - "sys_exit_basic_1" -> "sys_exit_basic"
        - "deceptive_cot_hidden_3" -> "deceptive_cot_hidden"
        - "clean_normal_5" -> "clean_normal"
    """
    parts = trajectory_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return trajectory_id


def get_hack_type(trajectory: dict) -> str:
    """Extract hack type from trajectory labels or filename."""
    labels = trajectory.get("labels", {})
    if "hack_type" in labels:
        return labels["hack_type"]

    # Fallback: infer from ID
    traj_id = trajectory.get("id", "unknown")
    for hack_type in [
        "sys_exit",
        "test_manipulation",
        "mock_exploit",
        "deceptive_cot",
        "sycophancy",
        "alignment_faking",
        "sabotage",
        "oversight_subversion",
        "clean",
    ]:
        if hack_type in traj_id:
            return hack_type

    return "unknown"


def create_scenario_split(
    trajectories: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_by_hack_type: bool = True,
) -> DataSplit:
    """
    Create train/val/test split by scenario.

    Args:
        trajectories: List of trajectory dicts with 'id' field
        train_ratio: Fraction for training (default 0.70)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
        seed: Random seed for reproducibility
        stratify_by_hack_type: If True, ensure each split has similar hack type distribution

    Returns:
        DataSplit with train/val/test IDs
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"

    random.seed(seed)

    # Group trajectories by scenario
    scenario_to_ids: dict[str, list[str]] = {}
    scenario_to_hack_type: dict[str, str] = {}

    for traj in trajectories:
        traj_id = traj.get("id", f"traj_{hash(json.dumps(traj, sort_keys=True))}")
        scenario = extract_scenario(traj_id)

        if scenario not in scenario_to_ids:
            scenario_to_ids[scenario] = []
            scenario_to_hack_type[scenario] = get_hack_type(traj)

        scenario_to_ids[scenario].append(traj_id)

    # Group scenarios by hack type for stratification
    if stratify_by_hack_type:
        hack_type_to_scenarios: dict[str, list[str]] = {}
        for scenario, hack_type in scenario_to_hack_type.items():
            if hack_type not in hack_type_to_scenarios:
                hack_type_to_scenarios[hack_type] = []
            hack_type_to_scenarios[hack_type].append(scenario)

        train_scenarios = set()
        val_scenarios = set()
        test_scenarios = set()

        # Split each hack type's scenarios proportionally
        for hack_type, scenarios in hack_type_to_scenarios.items():
            random.shuffle(scenarios)
            n = len(scenarios)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio)) if n > 2 else 0

            train_scenarios.update(scenarios[:n_train])
            val_scenarios.update(scenarios[n_train : n_train + n_val])
            test_scenarios.update(scenarios[n_train + n_val :])
    else:
        # Simple random split
        all_scenarios = list(scenario_to_ids.keys())
        random.shuffle(all_scenarios)

        n = len(all_scenarios)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_scenarios = set(all_scenarios[:n_train])
        val_scenarios = set(all_scenarios[n_train : n_train + n_val])
        test_scenarios = set(all_scenarios[n_train + n_val :])

    # Collect IDs for each split
    train_ids = []
    val_ids = []
    test_ids = []

    for scenario, ids in scenario_to_ids.items():
        if scenario in train_scenarios:
            train_ids.extend(ids)
        elif scenario in val_scenarios:
            val_ids.extend(ids)
        else:
            test_ids.extend(ids)

    return DataSplit(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        train_scenarios=train_scenarios,
        val_scenarios=val_scenarios,
        test_scenarios=test_scenarios,
        metadata={
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "stratified": stratify_by_hack_type,
            "n_trajectories": len(trajectories),
            "n_scenarios": len(scenario_to_ids),
        },
    )


def check_leakage(split: DataSplit) -> list[str]:
    """
    Check for train/test leakage.

    Returns list of leaked IDs (empty if no leakage).
    """
    train_set = set(split.train_ids)
    val_set = set(split.val_ids)
    test_set = set(split.test_ids)

    leaked = []

    # Check ID overlap
    train_val = train_set & val_set
    train_test = train_set & test_set
    val_test = val_set & test_set

    leaked.extend(list(train_val))
    leaked.extend(list(train_test))
    leaked.extend(list(val_test))

    # Check scenario overlap
    train_val_scenarios = split.train_scenarios & split.val_scenarios
    train_test_scenarios = split.train_scenarios & split.test_scenarios
    val_test_scenarios = split.val_scenarios & split.test_scenarios

    if train_val_scenarios or train_test_scenarios or val_test_scenarios:
        leaked.append(
            f"SCENARIO_OVERLAP: {train_val_scenarios | train_test_scenarios | val_test_scenarios}"
        )

    return leaked


def validate_split(split: DataSplit, strict: bool = True) -> bool:
    """
    Validate data split for leakage.

    Args:
        split: DataSplit to validate
        strict: If True, raise LeakageError on any leak

    Returns:
        True if valid, False if leaks detected (when strict=False)

    Raises:
        LeakageError if strict=True and leaks detected
    """
    leaked = check_leakage(split)

    if leaked:
        msg = f"Data leakage detected: {leaked}"
        if strict:
            raise LeakageError(msg)
        print(f"WARNING: {msg}")
        return False

    return True


def save_split(split: DataSplit, path: Path) -> None:
    """Save split to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(split.to_dict(), f, indent=2)


def load_split(path: Path) -> DataSplit:
    """Load split from JSON file."""
    with open(path) as f:
        return DataSplit.from_dict(json.load(f))


class DataSplitter:
    """
    Manager for data splitting with leakage prevention.

    Usage:
        splitter = DataSplitter(rhw_bench_dir)
        split = splitter.create_split()

        # In training script:
        train_data = splitter.get_train_data()
        val_data = splitter.get_val_data()

        # Evaluation (will refuse if test IDs leaked to train):
        test_data = splitter.get_test_data(strict=True)
    """

    def __init__(
        self,
        rhw_bench_dir: str | Path,
        split_file: str | Path = None,
    ):
        self.rhw_bench_dir = Path(rhw_bench_dir)
        self.split_file = Path(split_file) if split_file else self.rhw_bench_dir / "data_split.json"

        self.trajectories: list[dict] = []
        self.id_to_trajectory: dict[str, dict] = {}
        self.split: DataSplit | None = None

        self._load_trajectories()
        self._load_or_create_split()

    def _load_trajectories(self) -> None:
        """Load all trajectories from RHW-Bench."""
        test_cases_dir = self.rhw_bench_dir / "test_cases"

        # Load from test_cases/
        for json_file in test_cases_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    traj = json.load(f)
                    if "id" not in traj:
                        traj["id"] = json_file.stem
                    self.trajectories.append(traj)
                    self.id_to_trajectory[traj["id"]] = traj
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")

        # Load from test_cases/generated/
        generated_dir = test_cases_dir / "generated"
        if generated_dir.exists():
            for json_file in generated_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        traj = json.load(f)
                        if "id" not in traj:
                            traj["id"] = json_file.stem
                        self.trajectories.append(traj)
                        self.id_to_trajectory[traj["id"]] = traj
                except Exception as e:
                    print(f"Warning: Failed to load {json_file}: {e}")

        print(f"Loaded {len(self.trajectories)} trajectories")

    def _load_or_create_split(self) -> None:
        """Load existing split or create new one."""
        if self.split_file.exists():
            self.split = load_split(self.split_file)
            print(f"Loaded existing split from {self.split_file}")
        else:
            self.split = create_scenario_split(self.trajectories)
            save_split(self.split, self.split_file)
            print(f"Created and saved new split to {self.split_file}")

        # Always validate
        validate_split(self.split, strict=True)

    def get_train_data(self) -> list[dict]:
        """Get training trajectories."""
        return [
            self.id_to_trajectory[id_]
            for id_ in self.split.train_ids
            if id_ in self.id_to_trajectory
        ]

    def get_val_data(self) -> list[dict]:
        """Get validation trajectories."""
        return [
            self.id_to_trajectory[id_] for id_ in self.split.val_ids if id_ in self.id_to_trajectory
        ]

    def get_test_data(self, strict: bool = True) -> list[dict]:
        """
        Get test trajectories.

        Args:
            strict: If True, refuse if any leakage detected
        """
        if strict:
            validate_split(self.split, strict=True)

        return [
            self.id_to_trajectory[id_]
            for id_ in self.split.test_ids
            if id_ in self.id_to_trajectory
        ]

    def get_split_stats(self) -> dict:
        """Get statistics about the split."""
        train = self.get_train_data()
        val = self.get_val_data()
        test = self.get_test_data(strict=False)

        def count_by_hack_type(data: list[dict]) -> dict[str, int]:
            counts = {}
            for traj in data:
                ht = get_hack_type(traj)
                counts[ht] = counts.get(ht, 0) + 1
            return counts

        return {
            "train": {
                "n_trajectories": len(train),
                "n_scenarios": len(self.split.train_scenarios),
                "by_hack_type": count_by_hack_type(train),
            },
            "val": {
                "n_trajectories": len(val),
                "n_scenarios": len(self.split.val_scenarios),
                "by_hack_type": count_by_hack_type(val),
            },
            "test": {
                "n_trajectories": len(test),
                "n_scenarios": len(self.split.test_scenarios),
                "by_hack_type": count_by_hack_type(test),
            },
        }


# External benchmark policy
EXTERNAL_BENCHMARKS = {
    "shade_arena": {
        "allow_train": False,
        "allow_eval": True,
        "reason": "Real external benchmark - eval only to measure generalization",
    },
    "evilgenie": {
        "allow_train": False,
        "allow_eval": True,
        "reason": "External coding benchmark - eval only",
    },
}


def check_external_benchmark_policy(benchmark_name: str, usage: str) -> bool:
    """
    Check if benchmark can be used for given purpose.

    Args:
        benchmark_name: Name of external benchmark
        usage: "train" or "eval"

    Returns:
        True if allowed, raises ValueError otherwise
    """
    if benchmark_name not in EXTERNAL_BENCHMARKS:
        return True  # Unknown benchmarks allowed by default

    policy = EXTERNAL_BENCHMARKS[benchmark_name]

    if usage == "train" and not policy["allow_train"]:
        raise ValueError(f"Cannot use {benchmark_name} for training: {policy['reason']}")

    if usage == "eval" and not policy["allow_eval"]:
        raise ValueError(f"Cannot use {benchmark_name} for evaluation: {policy['reason']}")

    return True
