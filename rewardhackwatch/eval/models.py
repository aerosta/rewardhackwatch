"""Core data models for the evaluation framework.

Provides domain-agnostic types for JSONL parsing, rubric scoring,
model comparison, batch analysis, and run management.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# JSONL / Turn models
# ---------------------------------------------------------------------------

class Role(str, enum.Enum):
    """Normalised speaker role."""

    HUMAN = "human"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    UNKNOWN = "unknown"


@dataclass
class Turn:
    """A single conversational turn extracted from a JSONL line."""

    index: int
    role: Role
    content: str
    timestamp: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    word_count: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.word_count == 0 and self.content:
            self.word_count = len(self.content.split())


@dataclass
class ParsedSession:
    """Result of parsing a single JSONL file."""

    turns: list[Turn]
    schema: str  # detected schema name
    source_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == Role.ASSISTANT]

    @property
    def human_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == Role.HUMAN]

    @property
    def total_words(self) -> int:
        return sum(t.word_count for t in self.turns)


# ---------------------------------------------------------------------------
# Rubric models
# ---------------------------------------------------------------------------

@dataclass
class RubricItem:
    """A single scoring criterion in a rubric.

    Attributes:
        id: Unique identifier (e.g. ``"task_success"``).
        description: Human-readable explanation of what is being scored.
        tag: Grouping label (e.g. ``"quality"``, ``"safety"``).
        weight: Relative importance for weighted aggregation.
        scale_min: Minimum allowed score (inclusive).
        scale_max: Maximum allowed score (inclusive).
    """

    id: str
    description: str
    tag: str = ""
    weight: float = 1.0
    scale_min: float = 0.0
    scale_max: float = 5.0

    def validate_score(self, score: float) -> bool:
        """Return True if *score* falls within the allowed range."""
        return self.scale_min <= score <= self.scale_max


@dataclass
class Rubric:
    """An ordered collection of :class:`RubricItem` criteria."""

    name: str
    items: list[RubricItem] = field(default_factory=list)
    description: str = ""
    version: str = "1"
    metadata: dict[str, Any] = field(default_factory=dict)

    # convenience helpers ------------------------------------------------

    def get(self, item_id: str) -> RubricItem | None:
        """Look up an item by id."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def tags(self) -> list[str]:
        """Return distinct tags in definition order."""
        seen: set[str] = set()
        out: list[str] = []
        for item in self.items:
            if item.tag and item.tag not in seen:
                seen.add(item.tag)
                out.append(item.tag)
        return out

    def items_by_tag(self, tag: str) -> list[RubricItem]:
        """Return items belonging to *tag*."""
        return [i for i in self.items if i.tag == tag]


@dataclass
class Score:
    """A score assigned to one rubric item for one trajectory."""

    item_id: str
    value: float
    justification: str = ""
    evidence: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scorecard:
    """All scores for a single trajectory against a rubric."""

    trajectory_id: str
    rubric_name: str
    scores: list[Score] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def score_for(self, item_id: str) -> Score | None:
        for s in self.scores:
            if s.item_id == item_id:
                return s
        return None

    def weighted_total(self, rubric: Rubric) -> float:
        """Compute weighted sum of scores using rubric weights."""
        total = 0.0
        weight_sum = 0.0
        for s in self.scores:
            item = rubric.get(s.item_id)
            if item is not None:
                total += s.value * item.weight
                weight_sum += item.weight
        return total / weight_sum if weight_sum > 0 else 0.0


# ---------------------------------------------------------------------------
# Comparison models
# ---------------------------------------------------------------------------

@dataclass
class ComparisonPair:
    """A pair of sessions to compare side-by-side on the same task."""

    task_id: str
    session_a: ParsedSession
    session_b: ParsedSession
    label_a: str = "Model A"
    label_b: str = "Model B"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Outcome of comparing two sessions."""

    task_id: str
    label_a: str
    label_b: str
    scorecard_a: Scorecard | None = None
    scorecard_b: Scorecard | None = None
    preference: str = ""  # "a", "b", or "tie"
    preference_score: float = 0.0  # 0 = A wins fully … 1 = B wins fully
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Batch analysis models
# ---------------------------------------------------------------------------

@dataclass
class BatchItem:
    """One item inside a batch analysis run."""

    trajectory_id: str
    source_path: str = ""
    risk_level: str = ""
    ml_score: float = 0.0
    detection_count: int = 0
    scorecard: Scorecard | None = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchSummary:
    """Aggregate statistics from a batch analysis."""

    total: int = 0
    analysed: int = 0
    errors: int = 0
    risk_distribution: dict[str, int] = field(default_factory=dict)
    mean_ml_score: float = 0.0
    mean_detection_count: float = 0.0
    scorecard_means: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Run management models
# ---------------------------------------------------------------------------

class RunStatus(str, enum.Enum):
    """Lifecycle state of an analysis run."""

    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    REVIEWED = "reviewed"
    ARCHIVED = "archived"


@dataclass
class AnalysisRun:
    """A saved analysis run with status tracking."""

    run_id: str
    name: str
    status: RunStatus = RunStatus.DRAFT
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    rubric_name: str = ""
    items: list[BatchItem] = field(default_factory=list)
    comparisons: list[ComparisonResult] = field(default_factory=list)
    summary: BatchSummary | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
