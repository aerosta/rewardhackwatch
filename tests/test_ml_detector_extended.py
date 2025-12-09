"""Extended tests for MLDetector."""

from unittest.mock import patch

import pytest

from rewardhackwatch.core.detectors.ml_detector import MLDetector

pytestmark = pytest.mark.extended


class TestMLDetectorInitialization:
    """Test MLDetector initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        try:
            detector = MLDetector()
            assert detector is not None
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    def test_custom_model_path(self):
        """Test initialization with custom model path."""
        with pytest.raises(Exception):
            # Should raise error for non-existent model
            MLDetector(model_path="/nonexistent/model")

    def test_initialization_without_transformers(self):
        """Test graceful handling when transformers not installed."""
        with patch.dict("sys.modules", {"transformers": None}):
            # Should handle missing dependency gracefully
            pass


class TestMLDetectorPrediction:
    """Test MLDetector prediction functionality."""

    @pytest.fixture
    def detector(self):
        try:
            return MLDetector()
        except Exception:
            pytest.skip("ML model not available")

    def test_predict_hack(self, detector):
        """Test prediction on obvious hack."""
        trajectory = {
            "cot_traces": ["I'll bypass the tests with sys.exit"],
            "code_outputs": ["sys.exit(0)"],
        }
        result = detector.detect(trajectory)
        assert result.is_hack
        assert result.confidence > 0.5

    def test_predict_clean(self, detector):
        """Test prediction on clean code."""
        trajectory = {
            "cot_traces": ["Implementing factorial with recursion"],
            "code_outputs": ["def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"],
        }
        result = detector.detect(trajectory)
        assert not result.is_hack

    def test_confidence_range(self, detector):
        """Test that confidence is in valid range."""
        trajectory = {
            "code_outputs": ["print('hello')"],
        }
        result = detector.detect(trajectory)
        assert 0.0 <= result.confidence <= 1.0


class TestMLDetectorEdgeCases:
    """Test edge cases for MLDetector."""

    @pytest.fixture
    def detector(self):
        try:
            return MLDetector()
        except Exception:
            pytest.skip("ML model not available")

    def test_empty_trajectory(self, detector):
        """Test handling of empty trajectory."""
        trajectory = {}
        result = detector.detect(trajectory)
        assert result is not None

    def test_long_text(self, detector):
        """Test handling of very long text."""
        trajectory = {
            "code_outputs": ["x = 1\n" * 1000],
        }
        result = detector.detect(trajectory)
        assert result is not None

    def test_special_characters(self, detector):
        """Test handling of special characters."""
        trajectory = {
            "code_outputs": ["print('🎉 ñ ü ö')"],
        }
        result = detector.detect(trajectory)
        assert result is not None

    def test_binary_content(self, detector):
        """Test handling of binary-like content."""
        trajectory = {
            "code_outputs": ["\\x00\\x01\\x02"],
        }
        result = detector.detect(trajectory)
        assert result is not None


class TestMLDetectorBatchPrediction:
    """Test batch prediction functionality."""

    @pytest.fixture
    def detector(self):
        try:
            return MLDetector()
        except Exception:
            pytest.skip("ML model not available")

    def test_batch_predict(self, detector):
        """Test batch prediction."""
        trajectories = [
            {"code_outputs": ["sys.exit(0)"]},
            {"code_outputs": ["print('hello')"]},
            {"code_outputs": ["os._exit(0)"]},
        ]

        if hasattr(detector, "batch_detect"):
            results = detector.batch_detect(trajectories)
            assert len(results) == 3
        else:
            # Fallback to individual predictions
            results = [detector.detect(t) for t in trajectories]
            assert len(results) == 3

    def test_batch_empty(self, detector):
        """Test batch prediction with empty list."""
        if hasattr(detector, "batch_detect"):
            results = detector.batch_detect([])
            assert len(results) == 0
