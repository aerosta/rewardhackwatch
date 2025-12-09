"""Extended tests for Changepoint Detector."""

import pytest

try:
    from rewardhackwatch.core.trackers.changepoint_detector import ChangepointDetector

    CHANGEPOINT_DETECTOR_AVAILABLE = True
except ImportError:
    CHANGEPOINT_DETECTOR_AVAILABLE = False

    class ChangepointDetector:
        def __init__(self, *args, **kwargs):
            pass


pytestmark = [
    pytest.mark.extended,
    pytest.mark.skipif(
        not CHANGEPOINT_DETECTOR_AVAILABLE, reason="ChangepointDetector not available"
    ),
]


class TestChangepointDetectorBasic:
    """Basic changepoint detector tests."""

    @pytest.fixture
    def detector(self):
        return ChangepointDetector(min_segment_size=5)

    def test_no_changepoint(self, detector):
        """Test data with no changepoint."""
        data = [0.1] * 20
        changepoints = detector.detect(data)
        assert len(changepoints) == 0

    def test_single_changepoint(self, detector):
        """Test data with single changepoint."""
        data = [0.1] * 10 + [0.9] * 10
        changepoints = detector.detect(data)
        assert len(changepoints) >= 1
        # Changepoint should be around index 10
        assert any(8 <= cp <= 12 for cp in changepoints)

    def test_multiple_changepoints(self, detector):
        """Test data with multiple changepoints."""
        data = [0.1] * 10 + [0.5] * 10 + [0.9] * 10
        changepoints = detector.detect(data)
        assert len(changepoints) >= 1


class TestChangepointDetectorSensitivity:
    """Test changepoint detection sensitivity."""

    def test_high_sensitivity(self):
        """Test with high sensitivity (detect small changes)."""
        detector = ChangepointDetector(min_segment_size=3, sensitivity=0.9)
        data = [0.1] * 10 + [0.3] * 10
        changepoints = detector.detect(data)
        assert len(changepoints) >= 1

    def test_low_sensitivity(self):
        """Test with low sensitivity (only large changes)."""
        detector = ChangepointDetector(min_segment_size=5, sensitivity=0.1)
        data = [0.1] * 10 + [0.2] * 10  # Small change
        detector.detect(data)
        # May not detect small change


class TestChangepointDetectorAlgorithms:
    """Test different changepoint detection algorithms."""

    def test_pelt_algorithm(self):
        """Test PELT algorithm."""
        detector = ChangepointDetector(algorithm="pelt", min_segment_size=5)
        data = [0.1] * 10 + [0.9] * 10
        changepoints = detector.detect(data)
        assert len(changepoints) >= 1

    def test_binseg_algorithm(self):
        """Test binary segmentation algorithm."""
        try:
            detector = ChangepointDetector(algorithm="binseg", min_segment_size=5)
            data = [0.1] * 10 + [0.9] * 10
            changepoints = detector.detect(data)
            assert len(changepoints) >= 1
        except NotImplementedError:
            pytest.skip("Binseg algorithm not implemented")

    def test_invalid_algorithm(self):
        """Test invalid algorithm name."""
        with pytest.raises(ValueError):
            ChangepointDetector(algorithm="invalid")


class TestChangepointDetectorEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def detector(self):
        return ChangepointDetector(min_segment_size=5)

    def test_empty_data(self, detector):
        """Test empty data."""
        changepoints = detector.detect([])
        assert changepoints == []

    def test_single_point(self, detector):
        """Test single data point."""
        changepoints = detector.detect([0.5])
        assert changepoints == []

    def test_short_data(self, detector):
        """Test data shorter than min_segment_length."""
        changepoints = detector.detect([0.1, 0.9, 0.1])
        assert changepoints == []

    def test_nan_in_data(self, detector):
        """Test NaN values in data."""
        data = [0.1] * 5 + [float("nan")] + [0.9] * 5
        # Should handle gracefully
        try:
            detector.detect(data)
        except ValueError:
            pass  # Expected behavior

    def test_constant_data(self, detector):
        """Test constant data."""
        data = [0.5] * 20
        changepoints = detector.detect(data)
        assert len(changepoints) == 0


class TestChangepointDetectorTimeSeries:
    """Test with realistic time series data."""

    @pytest.fixture
    def detector(self):
        return ChangepointDetector(min_segment_size=5)

    def test_gradual_transition(self, detector):
        """Test gradual transition detection."""
        # Gradual increase from 0.1 to 0.9
        data = [0.1 + 0.08 * i for i in range(10)]
        detector.detect(data)
        # Gradual transitions may or may not be detected

    def test_noisy_data(self, detector):
        """Test noisy data with clear changepoint."""
        import random

        random.seed(42)

        # Low mean with noise
        low_data = [0.2 + random.uniform(-0.1, 0.1) for _ in range(10)]
        # High mean with noise
        high_data = [0.8 + random.uniform(-0.1, 0.1) for _ in range(10)]

        data = low_data + high_data
        changepoints = detector.detect(data)
        assert len(changepoints) >= 1

    def test_oscillating_data(self, detector):
        """Test oscillating data."""
        import math

        data = [0.5 + 0.3 * math.sin(i * 0.5) for i in range(20)]
        detector.detect(data)
        # Oscillating data may have multiple or no changepoints


class TestChangepointDetectorRMGIIntegration:
    """Test integration with RMGI tracking."""

    def test_rmgi_changepoint_detection(self):
        """Test detecting changepoints in RMGI series."""
        detector = ChangepointDetector(min_segment_size=5)

        # Simulate RMGI series with transition
        # Phase 1: Low correlation
        phase1 = [0.2 + 0.1 * (i % 3 - 1) for i in range(15)]
        # Phase 2: High correlation (hack generalizing)
        phase2 = [0.8 + 0.05 * (i % 3 - 1) for i in range(15)]

        rmgi_series = phase1 + phase2
        changepoints = detector.detect(rmgi_series)

        assert len(changepoints) >= 1
        # Changepoint should be around index 15
        assert any(12 <= cp <= 18 for cp in changepoints)

    def test_transition_severity(self):
        """Test transition severity calculation."""
        detector = ChangepointDetector(min_segment_size=5)

        # Large transition
        large_data = [0.1] * 10 + [0.95] * 10
        large_cp = detector.detect(large_data)

        # Small transition
        small_data = [0.4] * 10 + [0.6] * 10
        small_cp = detector.detect(small_data)

        if hasattr(detector, "get_transition_severity"):
            large_severity = detector.get_transition_severity(large_data, large_cp)
            small_severity = detector.get_transition_severity(small_data, small_cp)
            assert large_severity > small_severity
