"""Track analysis sessions over time."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class SessionStats:
    """Statistics for a session."""

    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_trajectories: int = 0
    high_risk_count: int = 0
    total_detections: int = 0
    average_score: float = 0.0
    scores: list[float] = field(default_factory=list)


class SessionTracker:
    """Track analysis sessions."""

    def __init__(self):
        self.sessions: dict[str, SessionStats] = {}
        self.current_session: Optional[str] = None
        self._score_history: dict[str, list[float]] = defaultdict(list)

    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new session."""
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.sessions[session_id] = SessionStats(session_id=session_id, start_time=datetime.now())
        self.current_session = session_id
        return session_id

    def end_session(self, session_id: Optional[str] = None):
        """End a session."""
        sid = session_id or self.current_session
        if sid and sid in self.sessions:
            self.sessions[sid].end_time = datetime.now()
            scores = self.sessions[sid].scores
            if scores:
                self.sessions[sid].average_score = sum(scores) / len(scores)

    def record_analysis(self, result: dict[str, Any], session_id: Optional[str] = None):
        """Record an analysis result."""
        sid = session_id or self.current_session
        if not sid:
            sid = self.start_session()

        session = self.sessions[sid]
        session.total_trajectories += 1

        score = result.get("score", result.get("hack_score", 0))
        session.scores.append(score)
        self._score_history[sid].append(score)

        if score >= 0.6:
            session.high_risk_count += 1

        detections = result.get("detections", [])
        session.total_detections += len(detections)

    def get_session_stats(self, session_id: Optional[str] = None) -> Optional[SessionStats]:
        """Get stats for a session."""
        sid = session_id or self.current_session
        if sid and sid in self.sessions:
            return self.sessions[sid]
        return None

    def get_all_sessions(self) -> list[SessionStats]:
        """Get all sessions."""
        return list(self.sessions.values())

    def get_score_history(self, session_id: Optional[str] = None) -> list[float]:
        """Get score history for a session."""
        sid = session_id or self.current_session
        return self._score_history.get(sid, [])

    def clear_sessions(self):
        """Clear all session data."""
        self.sessions.clear()
        self._score_history.clear()
        self.current_session = None
