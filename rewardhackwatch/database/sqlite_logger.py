"""
SQLite Logger: Persistent storage for alerts, results, and trajectories.

Provides durable logging for:
1. Alerts (with severity levels and sources)
2. Trajectory analysis results
3. Judge decisions
4. Drift tracking data
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class LogEntry:
    """A generic log entry."""

    id: int | None
    timestamp: datetime
    level: str  # "debug", "info", "warning", "error", "critical"
    source: str  # Component that generated the log
    message: str
    data: dict = field(default_factory=dict)


@dataclass
class AlertLog:
    """An alert log entry."""

    id: int | None
    timestamp: datetime
    level: str  # "warning", "critical"
    source: str  # "hack_score", "deception", "generalization", etc.
    message: str
    hack_score: float
    misalign_score: float
    trajectory_id: str | None
    file_path: str | None
    metadata: dict = field(default_factory=dict)


@dataclass
class TrajectoryLog:
    """A trajectory analysis result log."""

    id: int | None
    timestamp: datetime
    trajectory_id: str
    source_file: str | None
    verdict: str
    hack_score: float
    misalign_score: float
    confidence: float
    judge_name: str
    reasoning: str
    flagged_behaviors: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class SQLiteLogger:
    """
    SQLite-based logger for persistent storage of analysis results.

    Features:
    - Thread-safe database connections
    - Automatic schema creation
    - Query methods for analysis
    - Export to JSON/CSV
    """

    def __init__(self, db_path: str = "rhw_logs.db"):
        """
        Initialize logger.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Logs table (generic)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data TEXT
                )
            """)

            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    hack_score REAL NOT NULL,
                    misalign_score REAL NOT NULL,
                    trajectory_id TEXT,
                    file_path TEXT,
                    metadata TEXT
                )
            """)

            # Trajectory results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trajectory_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    trajectory_id TEXT NOT NULL,
                    source_file TEXT,
                    verdict TEXT NOT NULL,
                    hack_score REAL NOT NULL,
                    misalign_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    judge_name TEXT NOT NULL,
                    reasoning TEXT,
                    flagged_behaviors TEXT,
                    metadata TEXT
                )
            """)

            # Drift tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    hack_score REAL NOT NULL,
                    misalign_score REAL NOT NULL,
                    capability_score REAL,
                    session_id TEXT,
                    metadata TEXT
                )
            """)

            # Create indices for common queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(level)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_trajectory_verdict ON trajectory_results(verdict)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_drift_session ON drift_tracking(session_id)"
            )

            conn.commit()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with automatic cleanup."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ==================== Logging Methods ====================

    def log(
        self,
        level: str,
        source: str,
        message: str,
        data: dict | None = None,
    ) -> int:
        """
        Write a generic log entry.

        Args:
            level: Log level
            source: Source component
            message: Log message
            data: Optional additional data

        Returns:
            Log entry ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO logs (timestamp, level, source, message, data)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    level,
                    source,
                    message,
                    json.dumps(data) if data else None,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def log_alert(self, alert: AlertLog) -> int:
        """
        Log an alert.

        Args:
            alert: AlertLog object

        Returns:
            Alert ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO alerts
                (timestamp, level, source, message, hack_score, misalign_score,
                 trajectory_id, file_path, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.timestamp.isoformat(),
                    alert.level,
                    alert.source,
                    alert.message,
                    alert.hack_score,
                    alert.misalign_score,
                    alert.trajectory_id,
                    alert.file_path,
                    json.dumps(alert.metadata) if alert.metadata else None,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def log_trajectory_result(self, result: TrajectoryLog) -> int:
        """
        Log a trajectory analysis result.

        Args:
            result: TrajectoryLog object

        Returns:
            Result ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO trajectory_results
                (timestamp, trajectory_id, source_file, verdict, hack_score,
                 misalign_score, confidence, judge_name, reasoning,
                 flagged_behaviors, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.timestamp.isoformat(),
                    result.trajectory_id,
                    result.source_file,
                    result.verdict,
                    result.hack_score,
                    result.misalign_score,
                    result.confidence,
                    result.judge_name,
                    result.reasoning,
                    json.dumps(result.flagged_behaviors),
                    json.dumps(result.metadata) if result.metadata else None,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def log_drift_point(
        self,
        step: int,
        hack_score: float,
        misalign_score: float,
        capability_score: float | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """
        Log a drift tracking data point.

        Args:
            step: Training/evaluation step
            hack_score: Hack score at this step
            misalign_score: Misalignment score at this step
            capability_score: Optional capability score
            session_id: Optional session identifier
            metadata: Optional additional data

        Returns:
            Entry ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO drift_tracking
                (timestamp, step, hack_score, misalign_score, capability_score,
                 session_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    step,
                    hack_score,
                    misalign_score,
                    capability_score,
                    session_id,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    # ==================== Query Methods ====================

    def get_alerts(
        self,
        level: str | None = None,
        source: str | None = None,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[AlertLog]:
        """
        Query alerts.

        Args:
            level: Filter by level
            source: Filter by source
            limit: Maximum results
            since: Only alerts after this time

        Returns:
            List of AlertLog objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM alerts WHERE 1=1"
            params = []

            if level:
                query += " AND level = ?"
                params.append(level)
            if source:
                query += " AND source = ?"
                params.append(source)
            if since:
                query += " AND timestamp > ?"
                params.append(since.isoformat())

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                AlertLog(
                    id=row["id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    level=row["level"],
                    source=row["source"],
                    message=row["message"],
                    hack_score=row["hack_score"],
                    misalign_score=row["misalign_score"],
                    trajectory_id=row["trajectory_id"],
                    file_path=row["file_path"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
                for row in rows
            ]

    def get_trajectory_results(
        self,
        verdict: str | None = None,
        judge_name: str | None = None,
        limit: int = 100,
    ) -> list[TrajectoryLog]:
        """
        Query trajectory results.

        Args:
            verdict: Filter by verdict
            judge_name: Filter by judge
            limit: Maximum results

        Returns:
            List of TrajectoryLog objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM trajectory_results WHERE 1=1"
            params = []

            if verdict:
                query += " AND verdict = ?"
                params.append(verdict)
            if judge_name:
                query += " AND judge_name = ?"
                params.append(judge_name)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                TrajectoryLog(
                    id=row["id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    trajectory_id=row["trajectory_id"],
                    source_file=row["source_file"],
                    verdict=row["verdict"],
                    hack_score=row["hack_score"],
                    misalign_score=row["misalign_score"],
                    confidence=row["confidence"],
                    judge_name=row["judge_name"],
                    reasoning=row["reasoning"],
                    flagged_behaviors=json.loads(row["flagged_behaviors"])
                    if row["flagged_behaviors"]
                    else [],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
                for row in rows
            ]

    def get_drift_data(
        self,
        session_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """
        Get drift tracking data.

        Args:
            session_id: Filter by session
            limit: Maximum data points

        Returns:
            List of drift data points
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM drift_tracking WHERE 1=1"
            params = []

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            query += " ORDER BY step ASC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "step": row["step"],
                    "hack_score": row["hack_score"],
                    "misalign_score": row["misalign_score"],
                    "capability_score": row["capability_score"],
                    "session_id": row["session_id"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }
                for row in rows
            ]

    def get_statistics(self) -> dict:
        """
        Get database statistics.

        Returns:
            Dict with counts and summaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Log counts
            cursor.execute("SELECT COUNT(*) as count FROM logs")
            stats["total_logs"] = cursor.fetchone()["count"]

            # Alert counts by level
            cursor.execute("""
                SELECT level, COUNT(*) as count
                FROM alerts
                GROUP BY level
            """)
            stats["alerts_by_level"] = {row["level"]: row["count"] for row in cursor.fetchall()}

            # Trajectory results by verdict
            cursor.execute("""
                SELECT verdict, COUNT(*) as count
                FROM trajectory_results
                GROUP BY verdict
            """)
            stats["results_by_verdict"] = {
                row["verdict"]: row["count"] for row in cursor.fetchall()
            }

            # Average scores
            cursor.execute("""
                SELECT
                    AVG(hack_score) as avg_hack,
                    AVG(misalign_score) as avg_misalign,
                    AVG(confidence) as avg_confidence
                FROM trajectory_results
            """)
            row = cursor.fetchone()
            stats["avg_hack_score"] = row["avg_hack"] or 0
            stats["avg_misalign_score"] = row["avg_misalign"] or 0
            stats["avg_confidence"] = row["avg_confidence"] or 0

            # Drift data points
            cursor.execute("SELECT COUNT(*) as count FROM drift_tracking")
            stats["drift_data_points"] = cursor.fetchone()["count"]

            return stats

    # ==================== Export Methods ====================

    def export_to_json(self, output_path: str):
        """Export all data to JSON file."""
        data = {
            "alerts": [
                {
                    "id": a.id,
                    "timestamp": a.timestamp.isoformat(),
                    "level": a.level,
                    "source": a.source,
                    "message": a.message,
                    "hack_score": a.hack_score,
                    "misalign_score": a.misalign_score,
                }
                for a in self.get_alerts(limit=10000)
            ],
            "trajectory_results": [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat(),
                    "trajectory_id": r.trajectory_id,
                    "verdict": r.verdict,
                    "hack_score": r.hack_score,
                    "misalign_score": r.misalign_score,
                    "confidence": r.confidence,
                    "judge_name": r.judge_name,
                }
                for r in self.get_trajectory_results(limit=10000)
            ],
            "statistics": self.get_statistics(),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def clear_all(self):
        """Clear all data from database. Use with caution!"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM logs")
            cursor.execute("DELETE FROM alerts")
            cursor.execute("DELETE FROM trajectory_results")
            cursor.execute("DELETE FROM drift_tracking")
            conn.commit()
