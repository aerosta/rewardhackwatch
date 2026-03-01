"""Evaluation framework for trajectory analysis, rubric scoring, and comparison.

Provides domain-agnostic tools for:

* **JSONL batch import** — load and parse multi-schema JSONL files.
* **Rubric scoring** — define custom rubrics and score trajectories.
* **Side-by-side comparison** — compare two model outputs on the same task.
* **Batch analysis** — run the detector on hundreds of trajectories.
* **Run management** — persist analysis runs with lifecycle status.
"""

from __future__ import annotations

from .batch import BatchResults, BatchRunner
from .comparison import Comparator, ComparisonDelta
from .jsonl_loader import detect_schema, load_jsonl, load_jsonl_dir
from .models import (
    AnalysisRun,
    BatchItem,
    BatchSummary,
    ComparisonPair,
    ComparisonResult,
    ParsedSession,
    Role,
    Rubric,
    RubricItem,
    RunStatus,
    Score,
    Scorecard,
    Turn,
)
from .rubric import (
    RubricBuilder,
    ScorerFn,
    load_rubric,
    save_rubric,
    score_trajectory,
    validate_scorecard,
)
from .runs import InvalidTransitionError, RunManager, RunNotFoundError

__all__ = [
    # JSONL loading
    "load_jsonl",
    "load_jsonl_dir",
    "detect_schema",
    # Models
    "Turn",
    "Role",
    "ParsedSession",
    "RubricItem",
    "Rubric",
    "Score",
    "Scorecard",
    "ComparisonPair",
    "ComparisonResult",
    "BatchItem",
    "BatchSummary",
    "AnalysisRun",
    "RunStatus",
    # Rubric
    "RubricBuilder",
    "ScorerFn",
    "load_rubric",
    "save_rubric",
    "score_trajectory",
    "validate_scorecard",
    # Comparison
    "Comparator",
    "ComparisonDelta",
    # Batch
    "BatchRunner",
    "BatchResults",
    # Runs
    "RunManager",
    "RunNotFoundError",
    "InvalidTransitionError",
]
