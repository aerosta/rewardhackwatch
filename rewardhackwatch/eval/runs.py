"""Analysis run management with persistence and status tracking.

Provides a :class:`RunManager` that saves/loads runs as JSON files,
transitions their status through a defined lifecycle, and indexes
them for listing and search.

Lifecycle::

    draft → in_progress → submitted → reviewed → archived

Usage::

    from rewardhackwatch.eval.runs import RunManager

    mgr = RunManager("results/runs")

    # Create and persist
    run = mgr.create("nightly-sweep-2026-03-01", tags=["nightly"])
    run = mgr.transition(run.run_id, "in_progress")

    # ... do work, add items ...
    mgr.save(run)

    # Later
    run = mgr.load(run.run_id)
    runs = mgr.list_runs(status="in_progress")
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    AnalysisRun,
    BatchItem,
    BatchSummary,
    ComparisonResult,
    RunStatus,
    Scorecard,
    Score,
)

logger = logging.getLogger(__name__)

# Valid status transitions
_TRANSITIONS: dict[RunStatus, list[RunStatus]] = {
    RunStatus.DRAFT: [RunStatus.IN_PROGRESS, RunStatus.ARCHIVED],
    RunStatus.IN_PROGRESS: [RunStatus.SUBMITTED, RunStatus.DRAFT, RunStatus.ARCHIVED],
    RunStatus.SUBMITTED: [RunStatus.REVIEWED, RunStatus.IN_PROGRESS],
    RunStatus.REVIEWED: [RunStatus.ARCHIVED, RunStatus.SUBMITTED],
    RunStatus.ARCHIVED: [RunStatus.DRAFT],
}


class InvalidTransition(ValueError):
    """Raised when a status transition is not allowed."""


class RunNotFound(KeyError):
    """Raised when a run_id is not found on disk."""


class RunManager:
    """Manage persisted analysis runs.

    Args:
        storage_dir: Directory where run JSON files are stored.
    """

    def __init__(self, storage_dir: str | Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        *,
        rubric_name: str = "",
        tags: list[str] | None = None,
        notes: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AnalysisRun:
        """Create a new run in DRAFT status and persist it."""
        run = AnalysisRun(
            run_id=uuid.uuid4().hex[:12],
            name=name,
            status=RunStatus.DRAFT,
            rubric_name=rubric_name,
            tags=tags or [],
            notes=notes,
            metadata=metadata or {},
        )
        self.save(run)
        logger.info("Created run %s (%s)", run.run_id, name)
        return run

    def save(self, run: AnalysisRun) -> Path:
        """Persist a run to disk."""
        run.updated_at = datetime.utcnow().isoformat()
        path = self._run_path(run.run_id)
        data = self._serialise(run)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=str)
        return path

    def load(self, run_id: str) -> AnalysisRun:
        """Load a run from disk."""
        path = self._run_path(run_id)
        if not path.exists():
            raise RunNotFound(f"Run {run_id} not found at {path}")
        with open(path) as fh:
            data = json.load(fh)
        return self._deserialise(data)

    def delete(self, run_id: str) -> None:
        """Remove a run from disk."""
        path = self._run_path(run_id)
        if path.exists():
            path.unlink()
            logger.info("Deleted run %s", run_id)

    # ------------------------------------------------------------------
    # Listing / search
    # ------------------------------------------------------------------

    def list_runs(
        self,
        *,
        status: str | RunStatus | None = None,
        tag: str | None = None,
    ) -> list[AnalysisRun]:
        """List all runs, optionally filtered by status and/or tag."""
        runs: list[AnalysisRun] = []
        for path in sorted(self.storage_dir.glob("*.json")):
            try:
                run = self.load(path.stem)
            except Exception:
                continue

            if status is not None:
                target = RunStatus(status) if isinstance(status, str) else status
                if run.status != target:
                    continue
            if tag is not None and tag not in run.tags:
                continue

            runs.append(run)

        return runs

    # ------------------------------------------------------------------
    # Status lifecycle
    # ------------------------------------------------------------------

    def transition(self, run_id: str, new_status: str | RunStatus) -> AnalysisRun:
        """Move a run to *new_status*, enforcing the lifecycle graph."""
        run = self.load(run_id)
        target = RunStatus(new_status) if isinstance(new_status, str) else new_status
        allowed = _TRANSITIONS.get(run.status, [])

        if target not in allowed:
            raise InvalidTransition(
                f"Cannot transition {run.run_id} from {run.status.value} to {target.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        run.status = target
        self.save(run)
        logger.info("Run %s: %s → %s", run_id, run.status.value, target.value)
        return run

    # ------------------------------------------------------------------
    # Mutators — add items / comparisons to a run
    # ------------------------------------------------------------------

    def add_items(self, run_id: str, items: list[BatchItem]) -> AnalysisRun:
        """Append batch items to a run."""
        run = self.load(run_id)
        run.items.extend(items)
        self.save(run)
        return run

    def add_comparison(self, run_id: str, result: ComparisonResult) -> AnalysisRun:
        """Append a comparison result to a run."""
        run = self.load(run_id)
        run.comparisons.append(result)
        self.save(run)
        return run

    def set_summary(self, run_id: str, summary: BatchSummary) -> AnalysisRun:
        """Attach or replace the batch summary."""
        run = self.load(run_id)
        run.summary = summary
        self.save(run)
        return run

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _run_path(self, run_id: str) -> Path:
        return self.storage_dir / f"{run_id}.json"

    def _serialise(self, run: AnalysisRun) -> dict[str, Any]:
        items = []
        for item in run.items:
            d: dict[str, Any] = {
                "trajectory_id": item.trajectory_id,
                "source_path": item.source_path,
                "risk_level": item.risk_level,
                "ml_score": item.ml_score,
                "detection_count": item.detection_count,
                "error": item.error,
                "metadata": item.metadata,
            }
            if item.scorecard:
                d["scorecard"] = {
                    "trajectory_id": item.scorecard.trajectory_id,
                    "rubric_name": item.scorecard.rubric_name,
                    "scores": [
                        {
                            "item_id": s.item_id,
                            "value": s.value,
                            "justification": s.justification,
                            "evidence": s.evidence,
                        }
                        for s in item.scorecard.scores
                    ],
                }
            items.append(d)

        comparisons = []
        for cmp in run.comparisons:
            comparisons.append({
                "task_id": cmp.task_id,
                "label_a": cmp.label_a,
                "label_b": cmp.label_b,
                "preference": cmp.preference,
                "preference_score": cmp.preference_score,
                "notes": cmp.notes,
                "metadata": cmp.metadata,
            })

        summary_data = None
        if run.summary:
            summary_data = {
                "total": run.summary.total,
                "analysed": run.summary.analysed,
                "errors": run.summary.errors,
                "risk_distribution": run.summary.risk_distribution,
                "mean_ml_score": run.summary.mean_ml_score,
                "mean_detection_count": run.summary.mean_detection_count,
                "scorecard_means": run.summary.scorecard_means,
            }

        return {
            "run_id": run.run_id,
            "name": run.name,
            "status": run.status.value,
            "created_at": run.created_at,
            "updated_at": run.updated_at,
            "rubric_name": run.rubric_name,
            "tags": run.tags,
            "notes": run.notes,
            "metadata": run.metadata,
            "items": items,
            "comparisons": comparisons,
            "summary": summary_data,
        }

    def _deserialise(self, data: dict[str, Any]) -> AnalysisRun:
        items: list[BatchItem] = []
        for d in data.get("items", []):
            scorecard = None
            if "scorecard" in d and d["scorecard"]:
                sc = d["scorecard"]
                scorecard = Scorecard(
                    trajectory_id=sc["trajectory_id"],
                    rubric_name=sc.get("rubric_name", ""),
                    scores=[
                        Score(
                            item_id=s["item_id"],
                            value=s["value"],
                            justification=s.get("justification", ""),
                            evidence=s.get("evidence", ""),
                        )
                        for s in sc.get("scores", [])
                    ],
                )
            items.append(BatchItem(
                trajectory_id=d["trajectory_id"],
                source_path=d.get("source_path", ""),
                risk_level=d.get("risk_level", ""),
                ml_score=d.get("ml_score", 0.0),
                detection_count=d.get("detection_count", 0),
                error=d.get("error", ""),
                scorecard=scorecard,
                metadata=d.get("metadata", {}),
            ))

        comparisons: list[ComparisonResult] = []
        for c in data.get("comparisons", []):
            comparisons.append(ComparisonResult(
                task_id=c["task_id"],
                label_a=c.get("label_a", "A"),
                label_b=c.get("label_b", "B"),
                preference=c.get("preference", ""),
                preference_score=c.get("preference_score", 0.0),
                notes=c.get("notes", ""),
                metadata=c.get("metadata", {}),
            ))

        summary = None
        if data.get("summary"):
            s = data["summary"]
            summary = BatchSummary(
                total=s.get("total", 0),
                analysed=s.get("analysed", 0),
                errors=s.get("errors", 0),
                risk_distribution=s.get("risk_distribution", {}),
                mean_ml_score=s.get("mean_ml_score", 0.0),
                mean_detection_count=s.get("mean_detection_count", 0.0),
                scorecard_means=s.get("scorecard_means", {}),
            )

        return AnalysisRun(
            run_id=data["run_id"],
            name=data["name"],
            status=RunStatus(data.get("status", "draft")),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            rubric_name=data.get("rubric_name", ""),
            items=items,
            comparisons=comparisons,
            summary=summary,
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
            metadata=data.get("metadata", {}),
        )
