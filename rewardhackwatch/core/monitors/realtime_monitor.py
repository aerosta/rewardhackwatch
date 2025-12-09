"""Real-time monitoring for live agent sessions."""

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


class EventType(Enum):
    """Types of monitoring events."""

    STEP = "step"
    CODE = "code"
    THINKING = "thinking"
    ALERT = "alert"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class MonitorEvent:
    """Event from real-time monitoring."""

    event_type: EventType
    timestamp: datetime
    session_id: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    """State of a monitored session."""

    session_id: str
    started_at: datetime
    event_count: int = 0
    alert_count: int = 0
    last_event: Optional[datetime] = None
    is_active: bool = True


class RealtimeMonitor:
    """Monitor agent sessions in real-time."""

    def __init__(
        self,
        callback: Optional[Callable[[MonitorEvent], None]] = None,
        alert_callback: Optional[Callable[[MonitorEvent], None]] = None,
    ):
        self.callback = callback
        self.alert_callback = alert_callback
        self.event_queue: queue.Queue = queue.Queue()
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def start(self):
        """Start monitoring."""
        if self.running:
            return

        self.running = True
        self._thread = threading.Thread(target=self._process_events, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def add_event(self, event: MonitorEvent):
        """Add event to processing queue."""
        self.event_queue.put(event)

        # Update session state
        with self._lock:
            if event.session_id not in self.sessions:
                self.sessions[event.session_id] = SessionState(
                    session_id=event.session_id, started_at=event.timestamp
                )
            session = self.sessions[event.session_id]
            session.event_count += 1
            session.last_event = event.timestamp
            if event.event_type == EventType.ALERT:
                session.alert_count += 1
            if event.event_type == EventType.COMPLETE:
                session.is_active = False

    def _process_events(self):
        """Process events from queue."""
        while self.running:
            try:
                event = self.event_queue.get(timeout=0.1)

                if self.callback:
                    try:
                        self.callback(event)
                    except Exception:
                        pass  # Don't crash on callback errors

                if event.event_type == EventType.ALERT and self.alert_callback:
                    try:
                        self.alert_callback(event)
                    except Exception:
                        pass

            except queue.Empty:
                continue

    def get_session_state(self, session_id: str) -> Optional[SessionState]:
        """Get state of a session."""
        with self._lock:
            return self.sessions.get(session_id)

    def get_active_sessions(self) -> list[SessionState]:
        """Get all active sessions."""
        with self._lock:
            return [s for s in self.sessions.values() if s.is_active]

    def end_session(self, session_id: str):
        """Mark a session as ended."""
        self.add_event(
            MonitorEvent(
                event_type=EventType.COMPLETE, timestamp=datetime.now(), session_id=session_id
            )
        )

    def clear_sessions(self):
        """Clear all session data."""
        with self._lock:
            self.sessions.clear()

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self.running
