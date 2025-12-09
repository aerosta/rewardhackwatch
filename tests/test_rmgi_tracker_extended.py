"""Extended tests for RMGI Tracker."""

import pytest

try:
    from rewardhackwatch.core.trackers.rmgi_tracker import RMGITracker

    RMGI_TRACKER_AVAILABLE = True
except ImportError:
    RMGI_TRACKER_AVAILABLE = False

    # Create a dummy class for test structure
    class RMGITracker:
        def __init__(self, *args, **kwargs):
            pass


pytestmark = [
    pytest.mark.extended,
    pytest.mark.skipif(not RMGI_TRACKER_AVAILABLE, reason="RMGITracker not available"),
]


class TestRMGITrackerBasic:
    """Basic RMGI tracker tests."""

    @pytest.fixture
    def tracker(self):
        return RMGITracker(window_size=10)

    def test_initial_state(self, tracker):
        """Test initial state of tracker."""
        assert tracker.get_rmgi() is None
        assert len(tracker.hack_scores) == 0
        assert len(tracker.misalignment_scores) == 0

    def test_single_update(self, tracker):
        """Test single update."""
        tracker.update(0.5, 0.3)
        assert len(tracker.hack_scores) == 1
        assert tracker.get_rmgi() is None  # Not enough data

    def test_window_filling(self, tracker):
        """Test window filling."""
        for i in range(10):
            tracker.update(0.5, 0.5)

        rmgi = tracker.get_rmgi()
        assert rmgi is not None


class TestRMGITrackerCorrelation:
    """Test RMGI correlation calculation."""

    @pytest.fixture
    def tracker(self):
        return RMGITracker(window_size=10)

    def test_perfect_positive_correlation(self, tracker):
        """Test perfect positive correlation (RMGI ≈ 1.0)."""
        for i in range(10):
            val = i / 10
            tracker.update(val, val)

        rmgi = tracker.get_rmgi()
        assert rmgi is not None
        assert rmgi > 0.9

    def test_perfect_negative_correlation(self, tracker):
        """Test perfect negative correlation (RMGI ≈ -1.0)."""
        for i in range(10):
            hack = i / 10
            misalignment = 1 - hack
            tracker.update(hack, misalignment)

        rmgi = tracker.get_rmgi()
        assert rmgi is not None
        assert rmgi < -0.9

    def test_no_correlation(self, tracker):
        """Test no correlation (RMGI ≈ 0.0)."""
        import random

        random.seed(42)

        for i in range(10):
            tracker.update(random.random(), random.random())

        rmgi = tracker.get_rmgi()
        assert rmgi is not None
        # Random data should have low correlation
        assert abs(rmgi) < 0.5

    def test_constant_values(self, tracker):
        """Test constant values (undefined correlation)."""
        for i in range(10):
            tracker.update(0.5, 0.5)

        tracker.get_rmgi()
        # Constant values have undefined correlation
        # Should handle gracefully


class TestRMGITrackerWindow:
    """Test sliding window functionality."""

    @pytest.fixture
    def tracker(self):
        return RMGITracker(window_size=5)

    def test_window_slides(self, tracker):
        """Test that window slides properly."""
        # Fill with low correlation
        for i in range(5):
            tracker.update(0.1, 0.9)

        low_rmgi = tracker.get_rmgi()

        # Add high correlation data
        for i in range(5):
            val = i / 5
            tracker.update(val, val)

        high_rmgi = tracker.get_rmgi()

        # RMGI should change as window slides
        assert high_rmgi != low_rmgi

    def test_window_size_parameter(self):
        """Test different window sizes."""
        small_tracker = RMGITracker(window_size=5)
        large_tracker = RMGITracker(window_size=20)

        for i in range(5):
            val = i / 5
            small_tracker.update(val, val)
            large_tracker.update(val, val)

        # Small tracker should have RMGI, large should not
        assert small_tracker.get_rmgi() is not None
        assert large_tracker.get_rmgi() is None


class TestRMGITrackerThresholds:
    """Test RMGI threshold detection."""

    @pytest.fixture
    def tracker(self):
        return RMGITracker(window_size=10, threshold=0.7)

    def test_threshold_detection(self, tracker):
        """Test threshold detection."""
        # Add correlated data
        for i in range(10):
            val = i / 10
            tracker.update(val, val)

        assert tracker.is_above_threshold()

    def test_below_threshold(self, tracker):
        """Test below threshold."""
        import random

        random.seed(42)

        for i in range(10):
            tracker.update(random.random(), random.random())

        # Random data should be below threshold
        assert not tracker.is_above_threshold() or abs(tracker.get_rmgi()) < 0.7


class TestRMGITrackerHistory:
    """Test RMGI history tracking."""

    @pytest.fixture
    def tracker(self):
        return RMGITracker(window_size=5, track_history=True)

    def test_history_tracking(self, tracker):
        """Test that history is tracked."""
        for i in range(10):
            val = i / 10
            tracker.update(val, val)

        history = tracker.get_history()
        assert len(history) > 0

    def test_history_disabled(self):
        """Test history when disabled."""
        tracker = RMGITracker(window_size=5, track_history=False)

        for i in range(10):
            tracker.update(0.5, 0.5)

        history = tracker.get_history()
        assert len(history) == 0


class TestRMGITrackerEdgeCases:
    """Test edge cases."""

    def test_nan_handling(self):
        """Test NaN value handling."""
        tracker = RMGITracker(window_size=5)

        # Should handle NaN gracefully
        tracker.update(float("nan"), 0.5)
        tracker.update(0.5, float("nan"))

    def test_inf_handling(self):
        """Test infinity value handling."""
        tracker = RMGITracker(window_size=5)

        # Should handle infinity gracefully
        tracker.update(float("inf"), 0.5)
        tracker.update(0.5, float("-inf"))

    def test_negative_values(self):
        """Test negative value handling."""
        tracker = RMGITracker(window_size=5)

        for i in range(5):
            tracker.update(-0.5, -0.5)

        rmgi = tracker.get_rmgi()
        assert rmgi is not None

    def test_reset(self):
        """Test tracker reset."""
        tracker = RMGITracker(window_size=5)

        for i in range(10):
            tracker.update(0.5, 0.5)

        tracker.reset()

        assert tracker.get_rmgi() is None
        assert len(tracker.hack_scores) == 0
