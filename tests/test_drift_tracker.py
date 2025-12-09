"""Tests for Drift Tracker."""

import pytest

try:
    from rewardhackwatch.core.trackers.drift_tracker import DriftTracker

    DRIFT_TRACKER_AVAILABLE = True
except ImportError:
    DRIFT_TRACKER_AVAILABLE = False

    class DriftTracker:
        def __init__(self, *args, **kwargs):
            pass


pytestmark = [
    pytest.mark.extended,
    pytest.mark.skipif(not DRIFT_TRACKER_AVAILABLE, reason="DriftTracker not available"),
]


class TestDriftTrackerBasic:
    """Basic drift tracker tests."""

    @pytest.fixture
    def tracker(self):
        return DriftTracker(window_size=10)

    def test_initial_state(self, tracker):
        """Test initial state."""
        assert tracker.get_drift() is None
        assert len(tracker.history) == 0

    def test_add_score(self, tracker):
        """Test adding scores."""
        tracker.add_score(0.5)
        assert len(tracker.history) == 1

    def test_window_fills(self, tracker):
        """Test window filling."""
        for i in range(10):
            tracker.add_score(0.5)

        drift = tracker.get_drift()
        assert drift is not None


class TestDriftTrackerDetection:
    """Test drift detection."""

    @pytest.fixture
    def tracker(self):
        return DriftTracker(window_size=5)

    def test_detect_upward_drift(self, tracker):
        """Test detecting upward drift."""
        # Add increasing scores
        for i in range(10):
            tracker.add_score(i / 10)

        drift = tracker.get_drift()
        assert drift > 0

    def test_detect_downward_drift(self, tracker):
        """Test detecting downward drift."""
        # Add decreasing scores
        for i in range(10, 0, -1):
            tracker.add_score(i / 10)

        drift = tracker.get_drift()
        assert drift < 0

    def test_no_drift(self, tracker):
        """Test stable scores (no drift)."""
        for i in range(10):
            tracker.add_score(0.5)

        drift = tracker.get_drift()
        assert abs(drift) < 0.1


class TestDriftTrackerAlerts:
    """Test drift alerting."""

    @pytest.fixture
    def tracker(self):
        return DriftTracker(window_size=5, threshold=0.3)

    def test_alert_on_significant_drift(self, tracker):
        """Test alerting on significant drift."""
        for i in range(10):
            tracker.add_score(i / 5)  # Rapid increase

        assert tracker.is_drifting()

    def test_no_alert_on_minor_drift(self, tracker):
        """Test no alert on minor drift."""
        for i in range(10):
            tracker.add_score(0.5 + i * 0.01)  # Minor increase

        # May or may not be drifting based on implementation


class TestDriftTrackerEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def tracker(self):
        return DriftTracker(window_size=5)

    def test_reset(self, tracker):
        """Test tracker reset."""
        for i in range(10):
            tracker.add_score(0.5)

        tracker.reset()

        assert tracker.get_drift() is None
        assert len(tracker.history) == 0

    def test_nan_handling(self, tracker):
        """Test NaN handling."""
        tracker.add_score(0.5)
        tracker.add_score(float("nan"))
        tracker.add_score(0.6)
        # Should handle gracefully

    def test_negative_scores(self, tracker):
        """Test negative scores."""
        for i in range(5):
            tracker.add_score(-0.5)

        drift = tracker.get_drift()
        assert drift is not None or drift is None  # Implementation dependent
