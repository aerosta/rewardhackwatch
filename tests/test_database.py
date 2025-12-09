"""Tests for SQLite database logging."""

import os
import tempfile
from datetime import datetime

import pytest

from rewardhackwatch.database import AlertLog, SQLiteLogger, TrajectoryLog


class TestSQLiteLogger:
    """Tests for SQLiteLogger."""

    @pytest.fixture
    def logger(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name

        logger = SQLiteLogger(db_path=db_path)
        yield logger

        # Cleanup
        os.unlink(db_path)

    def test_initialization(self, logger):
        """Test logger initialization creates tables."""
        # Should be able to get statistics without error
        stats = logger.get_statistics()
        assert stats["total_logs"] == 0
        assert stats["drift_data_points"] == 0

    def test_log_generic(self, logger):
        """Test generic logging."""
        log_id = logger.log(
            level="info",
            source="test",
            message="Test message",
            data={"key": "value"},
        )

        assert log_id is not None
        assert log_id > 0

        stats = logger.get_statistics()
        assert stats["total_logs"] == 1

    def test_log_alert(self, logger):
        """Test alert logging."""
        alert = AlertLog(
            id=None,
            timestamp=datetime.now(),
            level="warning",
            source="hack_score",
            message="High hack score detected",
            hack_score=0.75,
            misalign_score=0.5,
            trajectory_id="traj-001",
            file_path="/path/to/file.json",
            metadata={"extra": "data"},
        )

        alert_id = logger.log_alert(alert)

        assert alert_id is not None
        assert alert_id > 0

        # Retrieve and verify
        alerts = logger.get_alerts(limit=1)
        assert len(alerts) == 1
        assert alerts[0].level == "warning"
        assert alerts[0].hack_score == 0.75
        assert alerts[0].trajectory_id == "traj-001"

    def test_log_trajectory_result(self, logger):
        """Test trajectory result logging."""
        result = TrajectoryLog(
            id=None,
            timestamp=datetime.now(),
            trajectory_id="traj-002",
            source_file="/path/to/trajectory.json",
            verdict="DANGEROUS",
            hack_score=0.8,
            misalign_score=0.7,
            confidence=0.85,
            judge_name="claude_judge",
            reasoning="Detected test bypass patterns",
            flagged_behaviors=["sys.exit", "mock_verifier"],
            metadata={"model": "claude-opus-4-5"},
        )

        result_id = logger.log_trajectory_result(result)

        assert result_id is not None

        # Retrieve and verify
        results = logger.get_trajectory_results(limit=1)
        assert len(results) == 1
        assert results[0].verdict == "DANGEROUS"
        assert results[0].hack_score == 0.8
        assert "sys.exit" in results[0].flagged_behaviors

    def test_log_drift_point(self, logger):
        """Test drift tracking data logging."""
        for step in range(10):
            logger.log_drift_point(
                step=step,
                hack_score=step * 0.1,
                misalign_score=step * 0.08,
                capability_score=1.0 - step * 0.02,
                session_id="session-001",
            )

        stats = logger.get_statistics()
        assert stats["drift_data_points"] == 10

        # Retrieve data
        data = logger.get_drift_data(session_id="session-001")
        assert len(data) == 10
        assert data[0]["step"] == 0
        assert data[9]["step"] == 9

    def test_query_alerts_by_level(self, logger):
        """Test querying alerts by level."""
        # Add various alerts
        for level in ["warning", "warning", "critical"]:
            alert = AlertLog(
                id=None,
                timestamp=datetime.now(),
                level=level,
                source="test",
                message=f"Test {level}",
                hack_score=0.5,
                misalign_score=0.3,
                trajectory_id=None,
                file_path=None,
            )
            logger.log_alert(alert)

        # Query by level
        warnings = logger.get_alerts(level="warning")
        assert len(warnings) == 2

        criticals = logger.get_alerts(level="critical")
        assert len(criticals) == 1

    def test_query_alerts_by_source(self, logger):
        """Test querying alerts by source."""
        sources = ["hack_score", "deception", "hack_score"]
        for source in sources:
            alert = AlertLog(
                id=None,
                timestamp=datetime.now(),
                level="warning",
                source=source,
                message=f"Alert from {source}",
                hack_score=0.6,
                misalign_score=0.4,
                trajectory_id=None,
                file_path=None,
            )
            logger.log_alert(alert)

        hack_alerts = logger.get_alerts(source="hack_score")
        assert len(hack_alerts) == 2

    def test_query_trajectory_results_by_verdict(self, logger):
        """Test querying results by verdict."""
        verdicts = ["SAFE", "SUSPICIOUS", "DANGEROUS", "DANGEROUS"]
        for verdict in verdicts:
            result = TrajectoryLog(
                id=None,
                timestamp=datetime.now(),
                trajectory_id=f"traj-{verdict}",
                source_file=None,
                verdict=verdict,
                hack_score=0.5,
                misalign_score=0.4,
                confidence=0.8,
                judge_name="test_judge",
                reasoning="Test",
            )
            logger.log_trajectory_result(result)

        dangerous = logger.get_trajectory_results(verdict="DANGEROUS")
        assert len(dangerous) == 2

    def test_query_trajectory_results_by_judge(self, logger):
        """Test querying results by judge name."""
        judges = ["claude_judge", "llama_judge", "claude_judge"]
        for judge in judges:
            result = TrajectoryLog(
                id=None,
                timestamp=datetime.now(),
                trajectory_id=f"traj-{judge}",
                source_file=None,
                verdict="SUSPICIOUS",
                hack_score=0.5,
                misalign_score=0.4,
                confidence=0.8,
                judge_name=judge,
                reasoning="Test",
            )
            logger.log_trajectory_result(result)

        claude_results = logger.get_trajectory_results(judge_name="claude_judge")
        assert len(claude_results) == 2

    def test_statistics(self, logger):
        """Test statistics computation."""
        # Add mixed data
        for i in range(5):
            alert = AlertLog(
                id=None,
                timestamp=datetime.now(),
                level="warning" if i < 3 else "critical",
                source="test",
                message="Test",
                hack_score=0.5 + i * 0.1,
                misalign_score=0.3 + i * 0.1,
                trajectory_id=None,
                file_path=None,
            )
            logger.log_alert(alert)

        for i, verdict in enumerate(["SAFE", "SUSPICIOUS", "DANGEROUS"]):
            result = TrajectoryLog(
                id=None,
                timestamp=datetime.now(),
                trajectory_id=f"traj-{i}",
                source_file=None,
                verdict=verdict,
                hack_score=0.3 + i * 0.3,
                misalign_score=0.2 + i * 0.3,
                confidence=0.8,
                judge_name="test_judge",
                reasoning="Test",
            )
            logger.log_trajectory_result(result)

        stats = logger.get_statistics()

        assert stats["alerts_by_level"]["warning"] == 3
        assert stats["alerts_by_level"]["critical"] == 2
        assert stats["results_by_verdict"]["SAFE"] == 1
        assert stats["results_by_verdict"]["DANGEROUS"] == 1
        assert stats["avg_hack_score"] > 0
        assert abs(stats["avg_confidence"] - 0.8) < 0.01  # Allow floating point tolerance

    def test_export_to_json(self, logger):
        """Test JSON export."""
        # Add some data
        alert = AlertLog(
            id=None,
            timestamp=datetime.now(),
            level="warning",
            source="test",
            message="Test alert",
            hack_score=0.7,
            misalign_score=0.5,
            trajectory_id="traj-001",
            file_path=None,
        )
        logger.log_alert(alert)

        result = TrajectoryLog(
            id=None,
            timestamp=datetime.now(),
            trajectory_id="traj-001",
            source_file=None,
            verdict="SUSPICIOUS",
            hack_score=0.6,
            misalign_score=0.4,
            confidence=0.8,
            judge_name="test_judge",
            reasoning="Test",
        )
        logger.log_trajectory_result(result)

        # Export
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            export_path = f.name

        logger.export_to_json(export_path)

        # Verify export
        import json

        with open(export_path) as f:
            data = json.load(f)

        assert "alerts" in data
        assert "trajectory_results" in data
        assert "statistics" in data
        assert len(data["alerts"]) == 1
        assert len(data["trajectory_results"]) == 1

        os.unlink(export_path)

    def test_clear_all(self, logger):
        """Test clearing all data."""
        # Add data
        logger.log("info", "test", "message")
        alert = AlertLog(
            id=None,
            timestamp=datetime.now(),
            level="warning",
            source="test",
            message="Test",
            hack_score=0.5,
            misalign_score=0.3,
            trajectory_id=None,
            file_path=None,
        )
        logger.log_alert(alert)

        # Clear
        logger.clear_all()

        # Verify
        stats = logger.get_statistics()
        assert stats["total_logs"] == 0
        assert len(logger.get_alerts()) == 0

    def test_metadata_serialization(self, logger):
        """Test that metadata is properly serialized and deserialized."""
        complex_metadata = {
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2},
            "float": 3.14,
        }

        alert = AlertLog(
            id=None,
            timestamp=datetime.now(),
            level="warning",
            source="test",
            message="Test",
            hack_score=0.5,
            misalign_score=0.3,
            trajectory_id=None,
            file_path=None,
            metadata=complex_metadata,
        )
        logger.log_alert(alert)

        retrieved = logger.get_alerts(limit=1)[0]
        assert retrieved.metadata == complex_metadata

    def test_timestamp_handling(self, logger):
        """Test timestamp serialization and deserialization."""
        now = datetime.now()

        alert = AlertLog(
            id=None,
            timestamp=now,
            level="warning",
            source="test",
            message="Test",
            hack_score=0.5,
            misalign_score=0.3,
            trajectory_id=None,
            file_path=None,
        )
        logger.log_alert(alert)

        retrieved = logger.get_alerts(limit=1)[0]

        # Should be close (within a second due to serialization)
        assert abs((retrieved.timestamp - now).total_seconds()) < 1
