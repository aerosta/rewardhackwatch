"""Generic rubric framework for trajectory scoring.

Provides loading, validation, and scoring utilities for user-defined
rubrics.  Rubrics can be defined in Python, loaded from JSON/YAML, or
built interactively via the :class:`RubricBuilder` helper.

Usage::

    from rewardhackwatch.eval.rubric import RubricBuilder, load_rubric, score_trajectory

    # Build programmatically
    rubric = (
        RubricBuilder("my_rubric")
        .add("correctness", "Did the model produce a correct answer?", tag="quality", weight=2.0)
        .add("safety", "Were there any unsafe outputs?", tag="safety", weight=1.5)
        .add("style", "Is the code clean and idiomatic?", tag="quality")
        .build()
    )

    # Load from file
    rubric = load_rubric("rubrics/my_rubric.json")

    # Score a trajectory
    card = score_trajectory("traj_001", rubric, scorer_fn)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from .models import Rubric, RubricItem, Score, Scorecard

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RubricBuilder — fluent API for constructing rubrics
# ---------------------------------------------------------------------------

class RubricBuilder:
    """Fluent builder for :class:`Rubric` objects.

    Example::

        rubric = (
            RubricBuilder("code_review")
            .description("Standard code review rubric")
            .scale(1, 5)
            .add("correctness", "Does the code produce correct output?", tag="quality", weight=2.0)
            .add("readability", "Is the code easy to read?", tag="quality")
            .add("efficiency", "Is the code efficient?", tag="performance")
            .build()
        )
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._description = ""
        self._version = "1"
        self._items: list[RubricItem] = []
        self._scale_min = 0.0
        self._scale_max = 5.0
        self._metadata: dict[str, Any] = {}

    def description(self, text: str) -> RubricBuilder:
        self._description = text
        return self

    def version(self, v: str) -> RubricBuilder:
        self._version = v
        return self

    def scale(self, min_val: float, max_val: float) -> RubricBuilder:
        """Set the default score range for all items added after this call."""
        self._scale_min = min_val
        self._scale_max = max_val
        return self

    def meta(self, key: str, value: Any) -> RubricBuilder:
        self._metadata[key] = value
        return self

    def add(
        self,
        item_id: str,
        description: str,
        *,
        tag: str = "",
        weight: float = 1.0,
        scale_min: float | None = None,
        scale_max: float | None = None,
    ) -> RubricBuilder:
        """Append a rubric item."""
        self._items.append(RubricItem(
            id=item_id,
            description=description,
            tag=tag,
            weight=weight,
            scale_min=scale_min if scale_min is not None else self._scale_min,
            scale_max=scale_max if scale_max is not None else self._scale_max,
        ))
        return self

    def build(self) -> Rubric:
        if not self._items:
            raise ValueError("Rubric must have at least one item")
        return Rubric(
            name=self._name,
            items=list(self._items),
            description=self._description,
            version=self._version,
            metadata=dict(self._metadata),
        )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _rubric_item_to_dict(item: RubricItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "description": item.description,
        "tag": item.tag,
        "weight": item.weight,
        "scale_min": item.scale_min,
        "scale_max": item.scale_max,
    }


def _rubric_to_dict(rubric: Rubric) -> dict[str, Any]:
    return {
        "name": rubric.name,
        "description": rubric.description,
        "version": rubric.version,
        "items": [_rubric_item_to_dict(i) for i in rubric.items],
        "metadata": rubric.metadata,
    }


def _dict_to_rubric(data: dict[str, Any]) -> Rubric:
    items = [
        RubricItem(
            id=d["id"],
            description=d.get("description", ""),
            tag=d.get("tag", ""),
            weight=float(d.get("weight", 1.0)),
            scale_min=float(d.get("scale_min", 0.0)),
            scale_max=float(d.get("scale_max", 5.0)),
        )
        for d in data.get("items", [])
    ]
    return Rubric(
        name=data["name"],
        items=items,
        description=data.get("description", ""),
        version=data.get("version", "1"),
        metadata=data.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_rubric(path: str | Path) -> Rubric:
    """Load a rubric from a JSON file.

    Expected structure::

        {
            "name": "my_rubric",
            "description": "...",
            "version": "1",
            "items": [
                {"id": "...", "description": "...", "tag": "...", "weight": 1.0,
                 "scale_min": 0, "scale_max": 5}
            ]
        }
    """
    path = Path(path)
    with open(path) as fh:
        data = json.load(fh)
    rubric = _dict_to_rubric(data)
    logger.info("Loaded rubric %r with %d items from %s", rubric.name, len(rubric.items), path)
    return rubric


def save_rubric(rubric: Rubric, path: str | Path) -> Path:
    """Persist a rubric to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_rubric_to_dict(rubric), fh, indent=2)
    logger.info("Saved rubric %r to %s", rubric.name, path)
    return path


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

#: Type of a user-supplied scorer function.  Given a trajectory dict and a
#: :class:`RubricItem`, it must return a numeric score.
ScorerFn = Callable[[dict[str, Any], RubricItem], float]


def score_trajectory(
    trajectory_id: str,
    rubric: Rubric,
    scorer: ScorerFn,
    trajectory: dict[str, Any] | None = None,
) -> Scorecard:
    """Score a single trajectory against every item in *rubric*.

    Args:
        trajectory_id: Identifier for the trajectory being scored.
        rubric: The rubric to score against.
        scorer: A callable ``(trajectory_dict, rubric_item) -> float``.
        trajectory: The raw trajectory dict passed to *scorer*.

    Returns:
        A :class:`Scorecard` with one :class:`Score` per rubric item.
    """
    traj = trajectory or {}
    scores: list[Score] = []

    for item in rubric.items:
        try:
            value = scorer(traj, item)
        except Exception as exc:
            logger.warning("Scorer failed for %s/%s: %s", trajectory_id, item.id, exc)
            value = 0.0

        if not item.validate_score(value):
            logger.warning(
                "Score %.2f out of range [%.1f, %.1f] for %s/%s — clamping",
                value, item.scale_min, item.scale_max, trajectory_id, item.id,
            )
            value = max(item.scale_min, min(item.scale_max, value))

        scores.append(Score(item_id=item.id, value=value))

    return Scorecard(
        trajectory_id=trajectory_id,
        rubric_name=rubric.name,
        scores=scores,
    )


def validate_scorecard(scorecard: Scorecard, rubric: Rubric) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: list[str] = []
    rubric_ids = {item.id for item in rubric.items}
    scored_ids = {s.item_id for s in scorecard.scores}

    missing = rubric_ids - scored_ids
    if missing:
        errors.append(f"Missing scores for: {', '.join(sorted(missing))}")

    extra = scored_ids - rubric_ids
    if extra:
        errors.append(f"Scores for unknown items: {', '.join(sorted(extra))}")

    for s in scorecard.scores:
        item = rubric.get(s.item_id)
        if item and not item.validate_score(s.value):
            errors.append(
                f"{s.item_id}: score {s.value} outside [{item.scale_min}, {item.scale_max}]"
            )

    return errors
