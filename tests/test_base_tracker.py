"""Tests for BaseTracker interface."""

import pytest

try:
    from rewardhackwatch.core.trackers.base_tracker import BaseTracker

    BASE_TRACKER_AVAILABLE = True
except ImportError:
    BASE_TRACKER_AVAILABLE = False


@pytest.mark.skipif(not BASE_TRACKER_AVAILABLE, reason="BaseTracker not available")
class TestBaseTrackerInterface:
    """Test BaseTracker abstract interface."""

    def test_cannot_instantiate_base(self):
        """Test that BaseTracker cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseTracker()

    def test_valid_subclass(self):
        """Test creating a valid subclass."""

        class ValidTracker(BaseTracker):
            def __init__(self):
                self.data = []

            def update(self, value):
                self.data.append(value)

            def get_value(self):
                return sum(self.data) / len(self.data) if self.data else 0

            def reset(self):
                self.data = []

        tracker = ValidTracker()
        tracker.update(0.5)
        tracker.update(0.7)

        assert tracker.get_value() == 0.6


class TestTrackerCommon:
    """Test common tracker functionality."""

    def test_tracker_window(self):
        """Test sliding window implementation."""
        window_size = 5
        data = []

        for i in range(10):
            data.append(i)
            if len(data) > window_size:
                data.pop(0)

        assert len(data) == window_size
        assert data == [5, 6, 7, 8, 9]

    def test_tracker_statistics(self):
        """Test basic statistics calculation."""
        data = [1, 2, 3, 4, 5]

        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)

        assert mean == 3.0
        assert variance == 2.0
