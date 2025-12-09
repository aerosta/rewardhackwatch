"""Monitor batch analysis operations."""

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional


@dataclass
class BatchProgress:
    """Progress information for batch analysis."""

    total: int
    completed: int
    failed: int
    current_item: Optional[str] = None
    started_at: Optional[datetime] = None
    elapsed_seconds: float = 0.0

    @property
    def progress_percent(self) -> float:
        if self.total == 0:
            return 100.0
        return (self.completed + self.failed) / self.total * 100

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.completed - self.failed)


class BatchMonitor:
    """Monitor and report on batch analysis progress."""

    def __init__(
        self,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
        update_interval: float = 1.0,
    ):
        self.progress_callback = progress_callback
        self.update_interval = update_interval
        self._progress: Optional[BatchProgress] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start_batch(self, total_items: int) -> BatchProgress:
        """Start monitoring a batch."""
        with self._lock:
            self._progress = BatchProgress(
                total=total_items, completed=0, failed=0, started_at=datetime.now()
            )
            self._running = True

        if self.progress_callback:
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()

        return self._progress

    def item_completed(self, item_id: Optional[str] = None, success: bool = True):
        """Mark an item as completed."""
        with self._lock:
            if self._progress:
                if success:
                    self._progress.completed += 1
                else:
                    self._progress.failed += 1
                self._progress.current_item = item_id

    def end_batch(self) -> Optional[BatchProgress]:
        """End batch monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            if self._progress and self._progress.started_at:
                self._progress.elapsed_seconds = (
                    datetime.now() - self._progress.started_at
                ).total_seconds()
            return self._progress

    def get_progress(self) -> Optional[BatchProgress]:
        """Get current progress."""
        with self._lock:
            if self._progress and self._progress.started_at:
                self._progress.elapsed_seconds = (
                    datetime.now() - self._progress.started_at
                ).total_seconds()
            return self._progress

    def _update_loop(self):
        """Background loop for progress updates."""
        while self._running:
            if self.progress_callback:
                progress = self.get_progress()
                if progress:
                    self.progress_callback(progress)
            time.sleep(self.update_interval)


def print_progress(progress: BatchProgress):
    """Default progress printer."""
    percent = progress.progress_percent
    bar_length = 30
    filled = int(bar_length * percent / 100)
    bar = "█" * filled + "░" * (bar_length - filled)

    print(
        f"\r[{bar}] {percent:.1f}% "
        f"({progress.completed}/{progress.total}) "
        f"Elapsed: {progress.elapsed_seconds:.1f}s",
        end="",
        flush=True,
    )
