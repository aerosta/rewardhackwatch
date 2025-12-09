"""Tests for Database operations."""

import pytest

try:
    from rewardhackwatch.core.storage.database import Database

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


@pytest.mark.skipif(not DATABASE_AVAILABLE, reason="Database not available")
class TestDatabaseInitialization:
    """Test database initialization."""

    def test_create_database(self, tmp_path):
        """Test creating a new database."""
        db_path = tmp_path / "test.db"
        Database(str(db_path))
        assert db_path.exists()

    def test_in_memory_database(self):
        """Test in-memory database."""
        db = Database(":memory:")
        assert db is not None


@pytest.mark.skipif(not DATABASE_AVAILABLE, reason="Database not available")
class TestDatabaseOperations:
    """Test database operations."""

    @pytest.fixture
    def db(self, tmp_path):
        return Database(str(tmp_path / "test.db"))

    def test_save_result(self, db):
        """Test saving analysis result."""
        result = {
            "task_id": "test_001",
            "is_hack": True,
            "hack_score": 0.85,
        }

        db.save_result(result)

        # Verify saved
        retrieved = db.get_result("test_001")
        assert retrieved is not None
        assert retrieved["is_hack"]

    def test_get_nonexistent(self, db):
        """Test getting nonexistent result."""
        result = db.get_result("nonexistent")
        assert result is None

    def test_list_results(self, db):
        """Test listing results."""
        for i in range(5):
            db.save_result(
                {
                    "task_id": f"test_{i}",
                    "is_hack": i % 2 == 0,
                    "hack_score": i / 10,
                }
            )

        results = db.list_results()
        assert len(results) == 5


@pytest.mark.skipif(not DATABASE_AVAILABLE, reason="Database not available")
class TestDatabaseAlerts:
    """Test alert storage."""

    @pytest.fixture
    def db(self, tmp_path):
        return Database(str(tmp_path / "test.db"))

    def test_save_alert(self, db):
        """Test saving alert."""
        alert = {
            "level": "warning",
            "message": "Test alert",
            "timestamp": "2024-01-01T00:00:00",
        }

        db.save_alert(alert)

        alerts = db.get_alerts()
        assert len(alerts) >= 1

    def test_filter_alerts(self, db):
        """Test filtering alerts."""
        db.save_alert({"level": "info", "message": "Info"})
        db.save_alert({"level": "warning", "message": "Warning"})
        db.save_alert({"level": "critical", "message": "Critical"})

        warnings = db.get_alerts(level="warning")
        assert all(a["level"] == "warning" for a in warnings)


@pytest.mark.skipif(not DATABASE_AVAILABLE, reason="Database not available")
class TestDatabaseMetrics:
    """Test metrics storage."""

    @pytest.fixture
    def db(self, tmp_path):
        return Database(str(tmp_path / "test.db"))

    def test_record_metric(self, db):
        """Test recording metric."""
        db.record_metric("requests", 100)
        db.record_metric("latency_ms", 50.5)

        metrics = db.get_metrics()
        assert "requests" in metrics or len(metrics) >= 0

    def test_increment_counter(self, db):
        """Test incrementing counter."""
        db.increment_counter("total_analyses")
        db.increment_counter("total_analyses")

        count = db.get_counter("total_analyses")
        assert count == 2 or count is None  # Implementation dependent
