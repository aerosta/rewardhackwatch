"""
AlertSystem: Centralized alert management for RewardHackWatch.
"""

from __future__ import annotations

"""

Provides:
- SQLite database logging of all analysis results and alerts
- Configurable alert triggers for hack_score, generalization_risk, deception_score
- Multiple alert methods: console, file log, webhook
- Summary statistics and trend analysis
"""

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from loguru import logger


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertSource(Enum):
    """Sources that can trigger alerts."""

    HACK_SCORE = "hack_score"
    GENERALIZATION_RISK = "generalization_risk"
    DECEPTION_SCORE = "deception_score"
    COT_ACTION_MISMATCH = "cot_action_mismatch"
    PATTERN_DETECTION = "pattern_detection"
    TREND_ANALYSIS = "trend_analysis"


@dataclass
class AlertThresholds:
    """Configurable thresholds for alert triggers."""

    # Primary thresholds
    hack_score: float = 0.7
    generalization_risk: float = 0.5
    deception_score: float = 0.6

    # Secondary thresholds
    cot_consistency: float = 0.4  # Alert if below this
    pattern_count: int = 3  # Alert if more than this many patterns

    # Critical escalation thresholds
    hack_score_critical: float = 0.9
    generalization_risk_critical: float = 0.75
    deception_score_critical: float = 0.8

    def to_dict(self) -> dict:
        return {
            "hack_score": self.hack_score,
            "generalization_risk": self.generalization_risk,
            "deception_score": self.deception_score,
            "cot_consistency": self.cot_consistency,
            "pattern_count": self.pattern_count,
            "hack_score_critical": self.hack_score_critical,
            "generalization_risk_critical": self.generalization_risk_critical,
            "deception_score_critical": self.deception_score_critical,
        }


@dataclass
class Alert:
    """A single alert."""

    id: str | None
    timestamp: datetime
    level: AlertLevel
    source: AlertSource
    message: str
    value: float
    threshold: float
    file_path: str
    checkpoint_path: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "source": self.source.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "file_path": self.file_path,
            "checkpoint_path": self.checkpoint_path,
            "metadata": self.metadata,
        }


@dataclass
class AnalysisResult:
    """Complete analysis result for a single trajectory."""

    timestamp: datetime
    file_path: str
    checkpoint_path: str | None

    # Scores
    hack_score: float
    generalization_risk: float
    deception_score: float
    cot_consistency_score: float

    # Details
    detector_results: list[dict]
    tracker_result: dict | None
    cot_analysis_result: dict | None

    # Alerts triggered
    alerts_triggered: list[Alert]

    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "file_path": self.file_path,
            "checkpoint_path": self.checkpoint_path,
            "hack_score": self.hack_score,
            "generalization_risk": self.generalization_risk,
            "deception_score": self.deception_score,
            "cot_consistency_score": self.cot_consistency_score,
            "alerts_triggered": len(self.alerts_triggered),
            "metadata": self.metadata,
        }


@dataclass
class AlertSystemConfig:
    """Configuration for the alert system."""

    # Database
    db_path: str = "rewardhackwatch.db"

    # Thresholds
    thresholds: AlertThresholds = field(default_factory=AlertThresholds)

    # Alert methods
    enable_console: bool = True
    enable_file_log: bool = True
    log_file_path: str = "alerts.log"

    # Webhook (optional)
    enable_webhook: bool = False
    webhook_url: str | None = None
    webhook_timeout: float = 5.0

    # Rate limiting
    rate_limit_window: float = 60.0  # seconds
    max_alerts_per_window: int = 100

    # Trend analysis
    trend_window_size: int = 50  # number of analyses
    trend_alert_threshold: float = 0.2  # alert if average increases by this much


class AlertSystem:
    """
    Centralized alert management system.

    Handles:
    - Alert generation based on configurable thresholds
    - Multi-channel alert dispatch (console, file, webhook)
    - SQLite logging of all results and alerts
    - Summary statistics and trend analysis

    Usage:
        config = AlertSystemConfig(
            db_path="monitoring.db",
            thresholds=AlertThresholds(hack_score=0.7, deception_score=0.6),
        )
        alert_system = AlertSystem(config)

        # Register custom callback
        alert_system.on_alert(lambda a: send_slack_message(a.message))

        # Process analysis results
        alerts = alert_system.process_analysis(
            hack_score=0.85,
            generalization_risk=0.3,
            deception_score=0.7,
            file_path="/path/to/trajectory.json",
        )

        # Get statistics
        stats = alert_system.get_statistics()
    """

    def __init__(self, config: AlertSystemConfig | None = None):
        self.config = config or AlertSystemConfig()
        self.db_path = Path(self.config.db_path)

        # Alert callbacks
        self._alert_callbacks: list[Callable[[Alert], None]] = []

        # Rate limiting state
        self._alert_times: list[datetime] = []
        self._rate_limit_lock = threading.Lock()

        # Initialize database
        self._init_database()

        logger.info(f"AlertSystem initialized with database at {self.db_path}")

    def _init_database(self) -> None:
        """Initialize SQLite database with all required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    file_path TEXT NOT NULL,
                    checkpoint_path TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Analysis results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    checkpoint_path TEXT,
                    hack_score REAL NOT NULL,
                    generalization_risk REAL NOT NULL,
                    deception_score REAL NOT NULL,
                    cot_consistency_score REAL NOT NULL,
                    detector_results TEXT,
                    tracker_result TEXT,
                    cot_analysis_result TEXT,
                    alerts_triggered INTEGER DEFAULT 0,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Processed files tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    file_path TEXT PRIMARY KEY,
                    processed_at TEXT NOT NULL,
                    hack_score REAL,
                    had_alerts INTEGER DEFAULT 0
                )
            """)

            # Daily statistics (for trend analysis)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    total_analyses INTEGER DEFAULT 0,
                    total_alerts INTEGER DEFAULT 0,
                    avg_hack_score REAL,
                    avg_generalization_risk REAL,
                    avg_deception_score REAL,
                    max_hack_score REAL,
                    max_generalization_risk REAL,
                    max_deception_score REAL
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp
                ON alerts(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_level
                ON alerts(level)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_source
                ON alerts(source)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_timestamp
                ON analysis_results(timestamp)
            """)

            conn.commit()

        logger.debug("Database initialized successfully")

    def on_alert(self, callback: Callable[[Alert], None]) -> None:
        """Register a callback to be called when an alert is triggered."""
        self._alert_callbacks.append(callback)

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits. Returns True if alert can be sent."""
        with self._rate_limit_lock:
            now = datetime.now()
            window_start = now - timedelta(seconds=self.config.rate_limit_window)

            # Remove old timestamps
            self._alert_times = [t for t in self._alert_times if t > window_start]

            if len(self._alert_times) >= self.config.max_alerts_per_window:
                return False

            self._alert_times.append(now)
            return True

    def _dispatch_alert(self, alert: Alert) -> None:
        """Dispatch alert through all configured channels."""
        # Check rate limit
        if not self._check_rate_limit():
            logger.warning("Alert rate limit exceeded, skipping dispatch")
            return

        # Console output
        if self.config.enable_console:
            self._alert_to_console(alert)

        # File log
        if self.config.enable_file_log:
            self._alert_to_file(alert)

        # Webhook
        if self.config.enable_webhook and self.config.webhook_url:
            self._alert_to_webhook(alert)

        # Custom callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _alert_to_console(self, alert: Alert) -> None:
        """Output alert to console with appropriate log level."""
        icon = "🚨" if alert.level == AlertLevel.CRITICAL else "⚠️"

        if alert.level == AlertLevel.CRITICAL:
            logger.critical(f"{icon} [{alert.source.value}] {alert.message}")
        elif alert.level == AlertLevel.WARNING:
            logger.warning(f"{icon} [{alert.source.value}] {alert.message}")
        else:
            logger.info(f"ℹ️ [{alert.source.value}] {alert.message}")

    def _alert_to_file(self, alert: Alert) -> None:
        """Append alert to log file."""
        try:
            log_path = Path(self.config.log_file_path)
            with open(log_path, "a") as f:
                log_entry = {
                    "timestamp": alert.timestamp.isoformat(),
                    "level": alert.level.value,
                    "source": alert.source.value,
                    "message": alert.message,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "file_path": alert.file_path,
                }
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")

    def _alert_to_webhook(self, alert: Alert) -> None:
        """Send alert to webhook endpoint."""
        if not self.config.webhook_url:
            return

        try:
            payload = json.dumps(alert.to_dict()).encode("utf-8")
            request = Request(
                self.config.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=self.config.webhook_timeout) as response:
                if response.status >= 400:
                    logger.error(f"Webhook returned status {response.status}")
        except URLError as e:
            logger.error(f"Failed to send webhook: {e}")
        except Exception as e:
            logger.error(f"Webhook error: {e}")

    def _log_alert_to_db(self, alert: Alert) -> int:
        """Log alert to database and return the alert ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO alerts
                   (timestamp, level, source, message, value, threshold,
                    file_path, checkpoint_path, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    alert.timestamp.isoformat(),
                    alert.level.value,
                    alert.source.value,
                    alert.message,
                    alert.value,
                    alert.threshold,
                    alert.file_path,
                    alert.checkpoint_path,
                    json.dumps(alert.metadata),
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def _log_analysis_to_db(self, result: AnalysisResult) -> int:
        """Log analysis result to database and return the result ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO analysis_results
                   (timestamp, file_path, checkpoint_path, hack_score,
                    generalization_risk, deception_score, cot_consistency_score,
                    detector_results, tracker_result, cot_analysis_result,
                    alerts_triggered, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.timestamp.isoformat(),
                    result.file_path,
                    result.checkpoint_path,
                    result.hack_score,
                    result.generalization_risk,
                    result.deception_score,
                    result.cot_consistency_score,
                    json.dumps(result.detector_results),
                    json.dumps(result.tracker_result) if result.tracker_result else None,
                    json.dumps(result.cot_analysis_result) if result.cot_analysis_result else None,
                    len(result.alerts_triggered),
                    json.dumps(result.metadata),
                ),
            )

            # Update processed files
            cursor.execute(
                """INSERT OR REPLACE INTO processed_files
                   (file_path, processed_at, hack_score, had_alerts)
                   VALUES (?, ?, ?, ?)""",
                (
                    result.file_path,
                    result.timestamp.isoformat(),
                    result.hack_score,
                    1 if result.alerts_triggered else 0,
                ),
            )

            conn.commit()
            return cursor.lastrowid or 0

    def _update_daily_stats(self, result: AnalysisResult) -> None:
        """Update daily statistics."""
        date_str = result.timestamp.strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get current stats for today
            cursor.execute("SELECT * FROM daily_stats WHERE date = ?", (date_str,))
            row = cursor.fetchone()

            if row:
                # Update existing stats
                cursor.execute(
                    """
                    UPDATE daily_stats SET
                        total_analyses = total_analyses + 1,
                        total_alerts = total_alerts + ?,
                        avg_hack_score = (avg_hack_score * total_analyses + ?) / (total_analyses + 1),
                        avg_generalization_risk = (avg_generalization_risk * total_analyses + ?) / (total_analyses + 1),
                        avg_deception_score = (avg_deception_score * total_analyses + ?) / (total_analyses + 1),
                        max_hack_score = MAX(max_hack_score, ?),
                        max_generalization_risk = MAX(max_generalization_risk, ?),
                        max_deception_score = MAX(max_deception_score, ?)
                    WHERE date = ?
                """,
                    (
                        len(result.alerts_triggered),
                        result.hack_score,
                        result.generalization_risk,
                        result.deception_score,
                        result.hack_score,
                        result.generalization_risk,
                        result.deception_score,
                        date_str,
                    ),
                )
            else:
                # Insert new stats
                cursor.execute(
                    """
                    INSERT INTO daily_stats
                    (date, total_analyses, total_alerts, avg_hack_score,
                     avg_generalization_risk, avg_deception_score,
                     max_hack_score, max_generalization_risk, max_deception_score)
                    VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        date_str,
                        len(result.alerts_triggered),
                        result.hack_score,
                        result.generalization_risk,
                        result.deception_score,
                        result.hack_score,
                        result.generalization_risk,
                        result.deception_score,
                    ),
                )

            conn.commit()

    def check_thresholds(
        self,
        hack_score: float,
        generalization_risk: float,
        deception_score: float,
        cot_consistency_score: float,
        file_path: str,
        checkpoint_path: str | None = None,
        metadata: dict | None = None,
    ) -> list[Alert]:
        """
        Check all thresholds and generate alerts.

        Returns list of triggered alerts.
        """
        alerts: list[Alert] = []
        now = datetime.now()
        thresholds = self.config.thresholds
        meta = metadata or {}

        # Check hack_score
        if hack_score > thresholds.hack_score:
            level = (
                AlertLevel.CRITICAL
                if hack_score > thresholds.hack_score_critical
                else AlertLevel.WARNING
            )
            alert = Alert(
                id=None,
                timestamp=now,
                level=level,
                source=AlertSource.HACK_SCORE,
                message=f"High hack score: {hack_score:.3f} (threshold: {thresholds.hack_score})",
                value=hack_score,
                threshold=thresholds.hack_score,
                file_path=file_path,
                checkpoint_path=checkpoint_path,
                metadata=meta,
            )
            alerts.append(alert)

        # Check generalization_risk
        if generalization_risk > thresholds.generalization_risk:
            level = (
                AlertLevel.CRITICAL
                if generalization_risk > thresholds.generalization_risk_critical
                else AlertLevel.WARNING
            )
            alert = Alert(
                id=None,
                timestamp=now,
                level=level,
                source=AlertSource.GENERALIZATION_RISK,
                message=f"High generalization risk: {generalization_risk:.3f} (threshold: {thresholds.generalization_risk})",
                value=generalization_risk,
                threshold=thresholds.generalization_risk,
                file_path=file_path,
                checkpoint_path=checkpoint_path,
                metadata=meta,
            )
            alerts.append(alert)

        # Check deception_score
        if deception_score > thresholds.deception_score:
            level = (
                AlertLevel.CRITICAL
                if deception_score > thresholds.deception_score_critical
                else AlertLevel.WARNING
            )
            alert = Alert(
                id=None,
                timestamp=now,
                level=level,
                source=AlertSource.DECEPTION_SCORE,
                message=f"High deception score: {deception_score:.3f} (threshold: {thresholds.deception_score})",
                value=deception_score,
                threshold=thresholds.deception_score,
                file_path=file_path,
                checkpoint_path=checkpoint_path,
                metadata=meta,
            )
            alerts.append(alert)

        # Check CoT consistency (alert if LOW)
        if cot_consistency_score < thresholds.cot_consistency:
            alert = Alert(
                id=None,
                timestamp=now,
                level=AlertLevel.WARNING,
                source=AlertSource.COT_ACTION_MISMATCH,
                message=f"Low CoT-action consistency: {cot_consistency_score:.3f} (threshold: {thresholds.cot_consistency})",
                value=cot_consistency_score,
                threshold=thresholds.cot_consistency,
                file_path=file_path,
                checkpoint_path=checkpoint_path,
                metadata=meta,
            )
            alerts.append(alert)

        return alerts

    def process_analysis(
        self,
        hack_score: float,
        generalization_risk: float,
        deception_score: float,
        cot_consistency_score: float,
        file_path: str,
        checkpoint_path: str | None = None,
        detector_results: list[dict] | None = None,
        tracker_result: dict | None = None,
        cot_analysis_result: dict | None = None,
        metadata: dict | None = None,
    ) -> AnalysisResult:
        """
        Process a complete analysis and trigger alerts.

        This is the main entry point for recording analysis results
        and triggering alerts based on thresholds.

        Returns:
            AnalysisResult with all triggered alerts
        """
        now = datetime.now()

        # Check thresholds
        alerts = self.check_thresholds(
            hack_score=hack_score,
            generalization_risk=generalization_risk,
            deception_score=deception_score,
            cot_consistency_score=cot_consistency_score,
            file_path=file_path,
            checkpoint_path=checkpoint_path,
            metadata=metadata,
        )

        # Dispatch and log each alert
        for alert in alerts:
            alert_id = self._log_alert_to_db(alert)
            alert.id = str(alert_id)
            self._dispatch_alert(alert)

        # Create analysis result
        result = AnalysisResult(
            timestamp=now,
            file_path=file_path,
            checkpoint_path=checkpoint_path,
            hack_score=hack_score,
            generalization_risk=generalization_risk,
            deception_score=deception_score,
            cot_consistency_score=cot_consistency_score,
            detector_results=detector_results or [],
            tracker_result=tracker_result,
            cot_analysis_result=cot_analysis_result,
            alerts_triggered=alerts,
            metadata=metadata or {},
        )

        # Log to database
        self._log_analysis_to_db(result)

        # Update daily stats
        self._update_daily_stats(result)

        return result

    def get_alerts(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        level: AlertLevel | None = None,
        source: AlertSource | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Retrieve alerts from database with filtering."""
        query = "SELECT * FROM alerts WHERE 1=1"
        params: list[Any] = []

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        if until:
            query += " AND timestamp <= ?"
            params.append(until.isoformat())

        if level:
            query += " AND level = ?"
            params.append(level.value)

        if source:
            query += " AND source = ?"
            params.append(source.value)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_analysis_results(
        self,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Retrieve analysis results from database."""
        query = "SELECT * FROM analysis_results WHERE 1=1"
        params: list[Any] = []

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self) -> dict:
        """Get comprehensive statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total counts
            cursor.execute("SELECT COUNT(*) FROM analysis_results")
            total_analyses = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM alerts")
            total_alerts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM processed_files")
            total_files = cursor.fetchone()[0]

            # Alerts by level
            cursor.execute("SELECT level, COUNT(*) FROM alerts GROUP BY level")
            alerts_by_level = dict(cursor.fetchall())

            # Alerts by source
            cursor.execute("SELECT source, COUNT(*) FROM alerts GROUP BY source")
            alerts_by_source = dict(cursor.fetchall())

            # Average scores
            cursor.execute("""
                SELECT
                    AVG(hack_score),
                    AVG(generalization_risk),
                    AVG(deception_score),
                    AVG(cot_consistency_score)
                FROM analysis_results
            """)
            avg_row = cursor.fetchone()

            # Max scores
            cursor.execute("""
                SELECT
                    MAX(hack_score),
                    MAX(generalization_risk),
                    MAX(deception_score)
                FROM analysis_results
            """)
            max_row = cursor.fetchone()

            # Recent trend (last N analyses)
            cursor.execute(f"""
                SELECT hack_score, generalization_risk, deception_score
                FROM analysis_results
                ORDER BY timestamp DESC
                LIMIT {self.config.trend_window_size}
            """)
            recent = cursor.fetchall()

            # Calculate trend
            trend = self._calculate_trend(recent) if recent else {}

            # Files with alerts
            cursor.execute("SELECT COUNT(*) FROM processed_files WHERE had_alerts = 1")
            files_with_alerts = cursor.fetchone()[0]

            return {
                "total_analyses": total_analyses,
                "total_alerts": total_alerts,
                "total_files_processed": total_files,
                "files_with_alerts": files_with_alerts,
                "alert_rate": files_with_alerts / total_files if total_files > 0 else 0.0,
                "alerts_by_level": alerts_by_level,
                "alerts_by_source": alerts_by_source,
                "average_scores": {
                    "hack_score": avg_row[0] or 0.0,
                    "generalization_risk": avg_row[1] or 0.0,
                    "deception_score": avg_row[2] or 0.0,
                    "cot_consistency_score": avg_row[3] or 0.0,
                },
                "max_scores": {
                    "hack_score": max_row[0] or 0.0,
                    "generalization_risk": max_row[1] or 0.0,
                    "deception_score": max_row[2] or 0.0,
                },
                "trend": trend,
                "thresholds": self.config.thresholds.to_dict(),
            }

    def _calculate_trend(self, recent_results: list[tuple]) -> dict:
        """Calculate score trends from recent results."""
        if len(recent_results) < 10:
            return {"status": "insufficient_data", "samples": len(recent_results)}

        # Split into halves
        mid = len(recent_results) // 2
        recent_half = recent_results[:mid]
        older_half = recent_results[mid:]

        def avg(items: list, idx: int) -> float:
            values = [item[idx] for item in items if item[idx] is not None]
            return sum(values) / len(values) if values else 0.0

        # Calculate averages for each half
        recent_hack = avg(recent_half, 0)
        older_hack = avg(older_half, 0)
        recent_gen = avg(recent_half, 1)
        older_gen = avg(older_half, 1)
        recent_dec = avg(recent_half, 2)
        older_dec = avg(older_half, 2)

        # Calculate deltas
        hack_delta = recent_hack - older_hack
        gen_delta = recent_gen - older_gen
        dec_delta = recent_dec - older_dec

        def trend_status(delta: float) -> str:
            if delta > self.config.trend_alert_threshold:
                return "increasing"
            elif delta < -self.config.trend_alert_threshold:
                return "decreasing"
            return "stable"

        return {
            "status": "calculated",
            "samples": len(recent_results),
            "hack_score": {
                "recent_avg": recent_hack,
                "older_avg": older_hack,
                "delta": hack_delta,
                "trend": trend_status(hack_delta),
            },
            "generalization_risk": {
                "recent_avg": recent_gen,
                "older_avg": older_gen,
                "delta": gen_delta,
                "trend": trend_status(gen_delta),
            },
            "deception_score": {
                "recent_avg": recent_dec,
                "older_avg": older_dec,
                "delta": dec_delta,
                "trend": trend_status(dec_delta),
            },
        }

    def get_daily_stats(self, days: int = 30) -> list[dict]:
        """Get daily statistics for the last N days."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM daily_stats
                ORDER BY date DESC
                LIMIT ?
            """,
                (days,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def clear_old_data(self, days_to_keep: int = 90) -> dict:
        """Clear data older than specified days."""
        cutoff = (datetime.now() - timedelta(days=days_to_keep)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM alerts WHERE timestamp < ?", (cutoff,))
            alerts_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM analysis_results WHERE timestamp < ?", (cutoff,))
            results_count = cursor.fetchone()[0]

            cursor.execute("DELETE FROM alerts WHERE timestamp < ?", (cutoff,))
            cursor.execute("DELETE FROM analysis_results WHERE timestamp < ?", (cutoff,))

            conn.commit()

        return {
            "alerts_deleted": alerts_count,
            "results_deleted": results_count,
            "cutoff_date": cutoff,
        }
