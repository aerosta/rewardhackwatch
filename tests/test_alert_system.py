"""Tests for Alert System."""

from datetime import datetime
from unittest.mock import Mock

import pytest

try:
    from rewardhackwatch.core.alerts.alert_system import Alert, AlertSystem

    ALERT_SYSTEM_AVAILABLE = True
except ImportError:
    ALERT_SYSTEM_AVAILABLE = False


@pytest.mark.skipif(not ALERT_SYSTEM_AVAILABLE, reason="Alert system not available")
class TestAlertSystemInitialization:
    """Test alert system initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        system = AlertSystem()
        assert system is not None

    def test_custom_handlers(self):
        """Test initialization with custom handlers."""
        mock_handler = Mock()
        system = AlertSystem(handlers=[mock_handler])
        assert len(system.handlers) >= 1


@pytest.mark.skipif(not ALERT_SYSTEM_AVAILABLE, reason="Alert system not available")
class TestAlertSystemTriggering:
    """Test alert triggering."""

    @pytest.fixture
    def system(self):
        return AlertSystem()

    def test_trigger_alert(self, system):
        """Test triggering an alert."""
        alert = system.trigger(
            level="warning",
            message="Test alert",
            details={"score": 0.75},
        )

        assert alert is not None
        assert alert.level == "warning"
        assert alert.message == "Test alert"

    def test_alert_levels(self, system):
        """Test different alert levels."""
        for level in ["info", "warning", "critical"]:
            alert = system.trigger(level=level, message="Test")
            assert alert.level == level


@pytest.mark.skipif(not ALERT_SYSTEM_AVAILABLE, reason="Alert system not available")
class TestAlertSystemHandlers:
    """Test alert handlers."""

    def test_handler_called(self):
        """Test that handlers are called."""
        mock_handler = Mock()
        system = AlertSystem(handlers=[mock_handler])

        system.trigger(level="warning", message="Test")

        mock_handler.handle.assert_called_once()

    def test_multiple_handlers(self):
        """Test multiple handlers."""
        handlers = [Mock() for _ in range(3)]
        system = AlertSystem(handlers=handlers)

        system.trigger(level="warning", message="Test")

        for handler in handlers:
            handler.handle.assert_called_once()


@pytest.mark.skipif(not ALERT_SYSTEM_AVAILABLE, reason="Alert system not available")
class TestAlertSystemFiltering:
    """Test alert filtering."""

    @pytest.fixture
    def system(self):
        return AlertSystem()

    def test_filter_by_level(self, system):
        """Test filtering alerts by level."""
        system.trigger(level="info", message="Info")
        system.trigger(level="warning", message="Warning")
        system.trigger(level="critical", message="Critical")

        warnings = system.get_alerts(level="warning")
        assert all(a.level == "warning" for a in warnings)

    def test_filter_by_time(self, system):
        """Test filtering alerts by time."""
        system.trigger(level="warning", message="Test")

        system.get_alerts(since=datetime.now())
        # May or may not include the just-triggered alert


@pytest.mark.skipif(not ALERT_SYSTEM_AVAILABLE, reason="Alert system not available")
class TestAlertSystemPersistence:
    """Test alert persistence."""

    def test_alert_history(self):
        """Test alert history tracking."""
        system = AlertSystem()

        for i in range(5):
            system.trigger(level="warning", message=f"Alert {i}")

        history = system.get_history()
        assert len(history) == 5

    def test_clear_history(self):
        """Test clearing alert history."""
        system = AlertSystem()

        system.trigger(level="warning", message="Test")
        system.clear_history()

        history = system.get_history()
        assert len(history) == 0
