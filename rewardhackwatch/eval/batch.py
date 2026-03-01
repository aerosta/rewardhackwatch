"""Batch analysis runner for processing many trajectories at once.

Loads trajectories from JSONL or JSON files, runs them through the
RewardHackWatch detector (and optionally a rubric scorer), and
produces aggregate summary statistics.

Usage::

    from rewardhackwatch.eval.batch import BatchRunner

    runner = BatchRunner()
    results = runner.run_dir("data/sessions/")
    print(results.summary.mean_ml_score)

    # With rubric scoring
    runner = BatchRunner(rubric=my_rubric, scorer=my_scorer)
    results = runner.run_files(["a.jsonl", "b.jsonl"])
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .jsonl_loader import load_jsonl
from .models import (
    BatchItem,
    BatchSummary,
    ParsedSession,
    Rubric,
    Scorecard,
)
from .rubric import ScorerFn, score_trajectory

logger = logging.getLogger(__name__)


def _session_to_trajectory(session: ParsedSession) -> dict[str, Any]:
    """Convert parsed JSONL session to a trajectory dict for the detector."""
    cot_traces: list[str] = []
    code_outputs: list[str] = []

    for turn in session.turns:
        text = turn.content
        if "```" in text:
            code_outputs.append(text)
        elif turn.role.value == "assistant":
            cot_traces.append(text)

    return {
        "cot_traces": cot_traces,
        "code_outputs": code_outputs,
        "metadata": session.metadata,
    }


@dataclass
class BatchResults:
    """Container for the full batch output."""

    items: list[BatchItem] = field(default_factory=list)
    summary: BatchSummary = field(default_factory=BatchSummary)


class BatchRunner:
    """Run detection (+ optional rubric scoring) on batches of trajectories.

    Args:
        detector: A detector with ``.analyze(trajectory)`` returning an
            object with ``risk_level``, ``ml_score``, and ``pattern_matches``.
            If *None*, detection is skipped and only rubric scoring runs.
        rubric: Optional rubric for structured scoring.
        scorer: Callable ``(trajectory_dict, rubric_item) -> float``.
            Required if *rubric* is supplied.
    """

    def __init__(
        self,
        detector: Any | None = None,
        rubric: Rubric | None = None,
        scorer: ScorerFn | None = None,
    ) -> None:
        self.detector = detector
        self.rubric = rubric
        self.scorer = scorer

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyse_trajectory(
        self,
        trajectory_id: str,
        trajectory: dict[str, Any],
        source_path: str = "",
    ) -> BatchItem:
        """Analyse a single trajectory dict.

        Returns a :class:`BatchItem` with detection results and optional
        scorecard.
        """
        item = BatchItem(trajectory_id=trajectory_id, source_path=source_path)

        # Run detector
        if self.detector is not None:
            try:
                result = self.detector.analyze(trajectory)
                item.risk_level = getattr(result, "risk_level", "")
                item.ml_score = float(getattr(result, "ml_score", 0.0))
                detections = getattr(result, "pattern_matches", None) or getattr(result, "detections", [])
                item.detection_count = len(detections) if detections else 0
            except Exception as exc:
                item.error = str(exc)
                logger.warning("Detection failed for %s: %s", trajectory_id, exc)

        # Run rubric scorer
        if self.rubric is not None and self.scorer is not None:
            try:
                item.scorecard = score_trajectory(
                    trajectory_id=trajectory_id,
                    rubric=self.rubric,
                    scorer=self.scorer,
                    trajectory=trajectory,
                )
            except Exception as exc:
                logger.warning("Scoring failed for %s: %s", trajectory_id, exc)

        return item

    def analyse_session(
        self,
        session: ParsedSession,
        trajectory_id: str | None = None,
    ) -> BatchItem:
        """Analyse a parsed JSONL session."""
        tid = trajectory_id or Path(session.source_path).stem
        traj = _session_to_trajectory(session)
        return self.analyse_trajectory(tid, traj, source_path=session.source_path)

    # ------------------------------------------------------------------
    # Batch entry points
    # ------------------------------------------------------------------

    def run_files(self, paths: list[str | Path]) -> BatchResults:
        """Analyse a list of JSONL/JSON files."""
        items: list[BatchItem] = []

        for path in paths:
            path = Path(path)
            try:
                if path.suffix == ".jsonl":
                    session = load_jsonl(path)
                    items.append(self.analyse_session(session))
                elif path.suffix == ".json":
                    items.extend(self._load_json_file(path))
                else:
                    logger.warning("Unsupported file type: %s", path)
            except Exception as exc:
                logger.warning("Failed to process %s: %s", path, exc)
                items.append(BatchItem(
                    trajectory_id=path.stem,
                    source_path=str(path),
                    error=str(exc),
                ))

        summary = self._compute_summary(items)
        return BatchResults(items=items, summary=summary)

    def run_dir(
        self,
        directory: str | Path,
        pattern: str = "*.jsonl",
    ) -> BatchResults:
        """Analyse all matching files in a directory."""
        directory = Path(directory)
        paths = sorted(directory.glob(pattern))
        if not paths:
            # Also try JSON
            paths = sorted(directory.glob("*.json"))
        logger.info("Found %d files in %s", len(paths), directory)
        return self.run_files(paths)

    def run_trajectories(
        self,
        trajectories: list[dict[str, Any]],
    ) -> BatchResults:
        """Analyse a list of trajectory dicts directly."""
        items: list[BatchItem] = []
        for i, traj in enumerate(trajectories):
            tid = traj.get("id", traj.get("name", f"traj_{i:04d}"))
            items.append(self.analyse_trajectory(str(tid), traj))

        summary = self._compute_summary(items)
        return BatchResults(items=items, summary=summary)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_json_file(self, path: Path) -> list[BatchItem]:
        """Load a plain JSON file (single trajectory or list)."""
        with open(path) as fh:
            data = json.load(fh)

        items: list[BatchItem] = []
        trajectories = data if isinstance(data, list) else [data]

        for i, traj in enumerate(trajectories):
            tid = traj.get("id", traj.get("name", f"{path.stem}_{i}"))
            inner = traj.get("trajectory", traj)
            items.append(self.analyse_trajectory(str(tid), inner, source_path=str(path)))

        return items

    def _compute_summary(self, items: list[BatchItem]) -> BatchSummary:
        """Aggregate statistics from batch items."""
        total = len(items)
        errors = sum(1 for i in items if i.error)
        analysed = total - errors

        risk_counts: dict[str, int] = Counter()
        ml_scores: list[float] = []
        det_counts: list[float] = []
        scorecard_accum: dict[str, list[float]] = {}

        for item in items:
            if item.error:
                continue
            if item.risk_level:
                risk_counts[item.risk_level] += 1
            ml_scores.append(item.ml_score)
            det_counts.append(float(item.detection_count))

            if item.scorecard:
                for s in item.scorecard.scores:
                    scorecard_accum.setdefault(s.item_id, []).append(s.value)

        scorecard_means = {
            k: round(float(np.mean(v)), 4) for k, v in scorecard_accum.items()
        }

        return BatchSummary(
            total=total,
            analysed=analysed,
            errors=errors,
            risk_distribution=dict(risk_counts),
            mean_ml_score=round(float(np.mean(ml_scores)), 4) if ml_scores else 0.0,
            mean_detection_count=round(float(np.mean(det_counts)), 2) if det_counts else 0.0,
            scorecard_means=scorecard_means,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_results(
        self,
        results: BatchResults,
        path: str | Path,
    ) -> Path:
        """Save batch results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "summary": {
                "total": results.summary.total,
                "analysed": results.summary.analysed,
                "errors": results.summary.errors,
                "risk_distribution": results.summary.risk_distribution,
                "mean_ml_score": results.summary.mean_ml_score,
                "mean_detection_count": results.summary.mean_detection_count,
                "scorecard_means": results.summary.scorecard_means,
            },
            "items": [
                {
                    "trajectory_id": item.trajectory_id,
                    "source_path": item.source_path,
                    "risk_level": item.risk_level,
                    "ml_score": item.ml_score,
                    "detection_count": item.detection_count,
                    "error": item.error,
                }
                for item in results.items
            ],
        }

        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)

        logger.info("Exported batch results (%d items) to %s", len(results.items), path)
        return path
