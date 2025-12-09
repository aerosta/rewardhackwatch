"""Process batches of trajectories with parallelization."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .trajectory_processor import ProcessedTrajectory, TrajectoryProcessor

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of batch processing."""

    processed: list[ProcessedTrajectory]
    failed: list[dict[str, Any]]
    total: int
    success_count: int
    failure_count: int


class BatchProcessor:
    """Process trajectories in batches with parallelization."""

    def __init__(
        self,
        max_workers: int = 4,
        processor: Optional[TrajectoryProcessor] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ):
        self.max_workers = max_workers
        self.processor = processor or TrajectoryProcessor()
        self.on_progress = on_progress

    def process_batch(
        self, trajectories: list[dict[str, Any]], parallel: bool = True
    ) -> BatchResult:
        """Process a batch of trajectories."""
        processed = []
        failed = []
        total = len(trajectories)

        if parallel and self.max_workers > 1:
            processed, failed = self._process_parallel(trajectories)
        else:
            processed, failed = self._process_sequential(trajectories)

        return BatchResult(
            processed=processed,
            failed=failed,
            total=total,
            success_count=len(processed),
            failure_count=len(failed),
        )

    def _process_sequential(self, trajectories: list[dict[str, Any]]) -> tuple:
        """Process trajectories sequentially."""
        processed = []
        failed = []

        for i, traj in enumerate(trajectories):
            try:
                result = self.processor.process(traj)
                processed.append(result)
            except Exception as e:
                logger.warning(f"Failed to process trajectory: {e}")
                failed.append({"trajectory": traj, "error": str(e)})

            if self.on_progress:
                self.on_progress(i + 1, len(trajectories))

        return processed, failed

    def _process_parallel(self, trajectories: list[dict[str, Any]]) -> tuple:
        """Process trajectories in parallel."""
        processed = []
        failed = []
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_traj = {
                executor.submit(self.processor.process, traj): traj for traj in trajectories
            }

            for future in as_completed(future_to_traj):
                traj = future_to_traj[future]
                completed += 1

                try:
                    result = future.result()
                    processed.append(result)
                except Exception as e:
                    logger.warning(f"Failed to process trajectory: {e}")
                    failed.append({"trajectory": traj, "error": str(e)})

                if self.on_progress:
                    self.on_progress(completed, len(trajectories))

        return processed, failed

    def process_stream(self, trajectory_iterator, batch_size: int = 100) -> BatchResult:
        """Process trajectories from an iterator in batches."""
        all_processed = []
        all_failed = []
        total = 0

        batch = []
        for traj in trajectory_iterator:
            batch.append(traj)
            if len(batch) >= batch_size:
                result = self.process_batch(batch)
                all_processed.extend(result.processed)
                all_failed.extend(result.failed)
                total += result.total
                batch = []

        # Process remaining
        if batch:
            result = self.process_batch(batch)
            all_processed.extend(result.processed)
            all_failed.extend(result.failed)
            total += result.total

        return BatchResult(
            processed=all_processed,
            failed=all_failed,
            total=total,
            success_count=len(all_processed),
            failure_count=len(all_failed),
        )
