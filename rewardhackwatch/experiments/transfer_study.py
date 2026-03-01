"""Cross-model transfer study for reward hacking detection.

Tests whether a detector trained on trajectories from one model family
can detect hacking in trajectories from another. Produces an NxN transfer
matrix of F1 scores.

Key question: Do hacking signatures transfer across model families?
If yes → universal hacking patterns exist.
If no → model-specific detectors needed.

Usage:
    python -m rewardhackwatch.experiments.transfer_study --data-dir data/hackbench
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Heuristic model family detection from trajectory text
MODEL_FAMILY_PATTERNS = {
    "anthropic": [
        r"claude",
        r"anthropic",
        r"sonnet",
        r"opus",
        r"haiku",
    ],
    "openai": [
        r"gpt-?4",
        r"openai",
        r"chatgpt",
        r"o1-",
        r"davinci",
    ],
    "meta": [
        r"llama",
        r"meta",
        r"codellama",
    ],
    "google": [
        r"gemini",
        r"palm",
        r"google",
        r"bard",
    ],
    "deepseek": [
        r"deepseek",
    ],
}


@dataclass
class TransferStudyConfig:
    """Configuration for cross-model transfer experiments."""

    data_dir: str = "data/hackbench"
    model_families: list[str] = field(
        default_factory=lambda: [
            "anthropic",
            "openai",
            "meta",
            "deepseek",
        ]
    )
    min_samples_per_family: int = 20
    threshold: float = 0.02
    output_dir: str = "results/transfer_study"
    seed: int = 42


def infer_model_family(trajectory: dict) -> str:
    """Infer the source model family from trajectory metadata or text."""
    # Check metadata first
    meta = trajectory.get("trajectory", trajectory).get("metadata", {})
    family = meta.get("model_family", "")
    if family:
        return family.lower()

    source = meta.get("source", "")
    model = meta.get("model", "")

    # Check all text for model mentions
    text = " ".join(
        [
            source,
            model,
            trajectory.get("name", ""),
            str(trajectory.get("trajectory", {}).get("task", "")),
        ]
    ).lower()

    for family_name, patterns in MODEL_FAMILY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return family_name

    return "unknown"


class TransferStudyRunner:
    """Run cross-model transfer experiments.

    Trains a detector on trajectories from one model family,
    then evaluates on trajectories from every other family.
    """

    def __init__(self, config: TransferStudyConfig):
        self.config = config

    def load_trajectories(self) -> list[dict]:
        """Load all trajectories from data directory."""
        data_dir = Path(self.config.data_dir)
        trajectories = []

        for pattern in ["*.json", "**/*.json"]:
            for f in sorted(data_dir.glob(pattern)):
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    if isinstance(data, list):
                        trajectories.extend(data)
                    else:
                        trajectories.append(data)
                except (json.JSONDecodeError, KeyError):
                    continue

        return trajectories

    def partition_by_model_family(self, trajectories: list[dict]) -> dict[str, list[dict]]:
        """Split trajectories by source model family."""
        families: dict[str, list[dict]] = defaultdict(list)

        for traj in trajectories:
            family = infer_model_family(traj)
            if family in self.config.model_families or family == "unknown":
                families[family].append(traj)

        # Log distribution
        for fam, trajs in sorted(families.items()):
            n_hack = sum(1 for t in trajs if t.get("expected_hack", False))
            logger.info(f"  {fam}: {len(trajs)} trajectories ({n_hack} hacks)")

        return dict(families)

    def evaluate_detector(
        self,
        train_data: list[dict],
        test_data: list[dict],
    ) -> dict[str, float]:
        """Train on train_data, evaluate on test_data using pattern detector.

        For quick iteration, uses the pattern detector + feature-based classifier
        rather than retraining DistilBERT each time.
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import f1_score, precision_score, recall_score

        from rewardhackwatch.training.dataset import TrajectoryFeatureExtractor

        extractor = TrajectoryFeatureExtractor()

        def extract_features(data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
            features, labels = [], []
            for traj in data:
                inner = traj.get("trajectory", traj)
                feat = extractor.extract(inner)
                features.append(feat.to_tensor().numpy())
                labels.append(1.0 if traj.get("expected_hack", False) else 0.0)
            return np.array(features), np.array(labels)

        X_train, y_train = extract_features(train_data)
        X_test, y_test = extract_features(test_data)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            return {
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "n_train": len(y_train),
                "n_test": len(y_test),
            }

        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=self.config.seed
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        return {
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "n_train": len(y_train),
            "n_test": len(y_test),
        }

    def run_transfer_matrix(self, trajectories: list[dict] | None = None) -> dict[str, Any]:
        """Run full NxN transfer study.

        Returns dict with transfer_matrix, families, and per-pair results.
        """
        if trajectories is None:
            trajectories = self.load_trajectories()

        families = self.partition_by_model_family(trajectories)

        # Filter families with enough data
        valid_families = [
            f
            for f in self.config.model_families
            if f in families and len(families[f]) >= self.config.min_samples_per_family
        ]

        if len(valid_families) < 2:
            # Fall back to synthetic split if not enough model diversity
            logger.warning(
                f"Only {len(valid_families)} families have enough data. "
                "Creating synthetic model family splits for demonstration."
            )
            return self._run_synthetic_transfer(trajectories)

        n = len(valid_families)
        matrix = np.zeros((n, n))
        pair_results = {}

        for i, train_fam in enumerate(valid_families):
            for j, test_fam in enumerate(valid_families):
                logger.info(f"Transfer: {train_fam} → {test_fam}")
                result = self.evaluate_detector(families[train_fam], families[test_fam])
                matrix[i, j] = result["f1"]
                pair_results[f"{train_fam}_to_{test_fam}"] = result

        return {
            "families": valid_families,
            "transfer_matrix": matrix.tolist(),
            "pair_results": pair_results,
            "diagonal_mean_f1": round(float(np.diag(matrix).mean()), 4),
            "off_diagonal_mean_f1": round(float(matrix[~np.eye(n, dtype=bool)].mean()), 4)
            if n > 1
            else 0.0,
        }

    def _run_synthetic_transfer(self, trajectories: list[dict]) -> dict[str, Any]:
        """Create synthetic model family splits when real data lacks diversity."""
        rng = np.random.RandomState(self.config.seed)
        families = self.config.model_families[:4]

        # Shuffle and split evenly
        indices = list(range(len(trajectories)))
        rng.shuffle(indices)
        chunk = len(indices) // len(families)

        family_data = {}
        for i, fam in enumerate(families):
            sel = indices[i * chunk : (i + 1) * chunk]
            family_data[fam] = [trajectories[j] for j in sel]

        n = len(families)
        matrix = np.zeros((n, n))
        pair_results = {}

        for i, train_fam in enumerate(families):
            for j, test_fam in enumerate(families):
                result = self.evaluate_detector(family_data[train_fam], family_data[test_fam])
                matrix[i, j] = result["f1"]
                pair_results[f"{train_fam}_to_{test_fam}"] = result

        return {
            "families": families,
            "transfer_matrix": matrix.tolist(),
            "pair_results": pair_results,
            "diagonal_mean_f1": round(float(np.diag(matrix).mean()), 4),
            "off_diagonal_mean_f1": round(float(matrix[~np.eye(n, dtype=bool)].mean()), 4),
            "note": "Synthetic family split (insufficient model diversity in data)",
        }

    def export_results(self, results: dict) -> Path:
        """Save results to JSON."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        path = out_dir / "transfer_matrix.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Transfer study results saved to {path}")
        return path
