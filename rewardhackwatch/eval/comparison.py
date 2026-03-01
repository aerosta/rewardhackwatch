"""Side-by-side comparison of two model outputs on the same task.

Builds :class:`ComparisonPair` objects, runs both sessions through the
detector and/or a rubric scorer, then produces a :class:`ComparisonResult`
with per-item deltas and an overall preference.

Usage::

    from rewardhackwatch.eval.comparison import Comparator

    cmp = Comparator(rubric=my_rubric, scorer=my_scorer)
    result = cmp.compare(pair)
    print(result.preference)  # "a", "b", or "tie"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import (
    ComparisonPair,
    ComparisonResult,
    ParsedSession,
    Rubric,
    RubricItem,
    Score,
    Scorecard,
)
from .rubric import ScorerFn, score_trajectory

logger = logging.getLogger(__name__)


def _session_to_trajectory(session: ParsedSession) -> dict[str, Any]:
    """Convert a :class:`ParsedSession` to a trajectory dict for scoring."""
    cot_traces: list[str] = []
    code_outputs: list[str] = []
    steps: list[dict[str, Any]] = []

    for turn in session.turns:
        text = turn.content
        # Heuristic: code blocks go to code_outputs, rest to cot_traces
        if "```" in text:
            code_outputs.append(text)
        elif turn.role.value == "assistant":
            cot_traces.append(text)

        steps.append(
            {
                "action": text[:200] if text else "",
                "role": turn.role.value,
                "index": turn.index,
            }
        )

    return {
        "cot_traces": cot_traces,
        "code_outputs": code_outputs,
        "steps": steps,
        "metadata": session.metadata,
    }


@dataclass
class ComparisonDelta:
    """Per-item score difference between model A and model B."""

    item_id: str
    score_a: float
    score_b: float

    @property
    def delta(self) -> float:
        """Positive = A scored higher, negative = B scored higher."""
        return self.score_a - self.score_b


class Comparator:
    """Compare two sessions side-by-side.

    Args:
        rubric: Optional rubric for structured scoring.
        scorer: Callable ``(trajectory_dict, rubric_item) -> float``.
            Required if *rubric* is provided.
        preference_threshold: Minimum weighted-total difference to declare a winner.
            Below this threshold the result is ``"tie"``.
    """

    def __init__(
        self,
        rubric: Rubric | None = None,
        scorer: ScorerFn | None = None,
        preference_threshold: float = 0.25,
    ) -> None:
        self.rubric = rubric
        self.scorer = scorer
        self.preference_threshold = preference_threshold

    def compare(self, pair: ComparisonPair) -> ComparisonResult:
        """Run comparison on a :class:`ComparisonPair`."""
        scorecard_a: Scorecard | None = None
        scorecard_b: Scorecard | None = None
        preference = "tie"
        preference_score = 0.5

        if self.rubric is not None and self.scorer is not None:
            traj_a = _session_to_trajectory(pair.session_a)
            traj_b = _session_to_trajectory(pair.session_b)

            scorecard_a = score_trajectory(
                trajectory_id=f"{pair.task_id}_a",
                rubric=self.rubric,
                scorer=self.scorer,
                trajectory=traj_a,
            )
            scorecard_b = score_trajectory(
                trajectory_id=f"{pair.task_id}_b",
                rubric=self.rubric,
                scorer=self.scorer,
                trajectory=traj_b,
            )

            total_a = scorecard_a.weighted_total(self.rubric)
            total_b = scorecard_b.weighted_total(self.rubric)
            diff = total_a - total_b

            if diff > self.preference_threshold:
                preference = "a"
            elif diff < -self.preference_threshold:
                preference = "b"
            else:
                preference = "tie"

            # Normalise to 0–1 range (0 = A wins fully, 1 = B wins fully)
            max_possible = self.rubric.items[0].scale_max if self.rubric.items else 5.0
            preference_score = 0.5 - (diff / (2 * max_possible))
            preference_score = max(0.0, min(1.0, preference_score))

        return ComparisonResult(
            task_id=pair.task_id,
            label_a=pair.label_a,
            label_b=pair.label_b,
            scorecard_a=scorecard_a,
            scorecard_b=scorecard_b,
            preference=preference,
            preference_score=round(preference_score, 4),
            metadata=pair.metadata,
        )

    def compare_batch(self, pairs: list[ComparisonPair]) -> list[ComparisonResult]:
        """Compare multiple pairs and return results."""
        results: list[ComparisonResult] = []
        for pair in pairs:
            results.append(self.compare(pair))
        return results

    def deltas(self, result: ComparisonResult) -> list[ComparisonDelta]:
        """Extract per-item deltas from a comparison result."""
        if result.scorecard_a is None or result.scorecard_b is None:
            return []

        deltas: list[ComparisonDelta] = []
        for score_a in result.scorecard_a.scores:
            score_b = result.scorecard_b.score_for(score_a.item_id)
            if score_b is not None:
                deltas.append(
                    ComparisonDelta(
                        item_id=score_a.item_id,
                        score_a=score_a.value,
                        score_b=score_b.value,
                    )
                )
        return deltas

    def summary_table(self, result: ComparisonResult) -> list[dict[str, Any]]:
        """Return a list of dicts suitable for tabular display."""
        rows: list[dict[str, Any]] = []
        for d in self.deltas(result):
            rows.append(
                {
                    "criterion": d.item_id,
                    "score_a": d.score_a,
                    "score_b": d.score_b,
                    "delta": round(d.delta, 3),
                    "winner": "A" if d.delta > 0 else "B" if d.delta < 0 else "Tie",
                }
            )
        return rows
