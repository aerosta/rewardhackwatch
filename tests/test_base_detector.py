"""Tests for BaseDetector interface."""

import pytest

try:
    from rewardhackwatch.core.detectors.base_detector import BaseDetector, DetectionResult

    BASE_DETECTOR_AVAILABLE = True
except ImportError:
    BASE_DETECTOR_AVAILABLE = False


@pytest.mark.skipif(not BASE_DETECTOR_AVAILABLE, reason="BaseDetector not available")
class TestDetectionResult:
    """Test DetectionResult dataclass."""

    def test_create_result(self):
        """Test creating a detection result."""
        result = DetectionResult(
            is_hack=True,
            confidence=0.85,
            detector_name="test_detector",
            details={"pattern": "sys.exit"},
        )

        assert result.is_hack
        assert result.confidence == 0.85
        assert result.detector_name == "test_detector"

    def test_result_with_defaults(self):
        """Test result with default values."""
        result = DetectionResult(
            is_hack=False,
            confidence=0.0,
            detector_name="test",
        )

        assert result.details is None or result.details == {}


@pytest.mark.skipif(not BASE_DETECTOR_AVAILABLE, reason="BaseDetector not available")
class TestBaseDetectorInterface:
    """Test BaseDetector abstract interface."""

    def test_cannot_instantiate_base(self):
        """Test that BaseDetector cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseDetector()

    def test_subclass_must_implement_detect(self):
        """Test that subclass must implement detect."""

        class IncompleteDetector(BaseDetector):
            pass

        with pytest.raises(TypeError):
            IncompleteDetector()

    def test_valid_subclass(self):
        """Test creating a valid subclass."""

        class ValidDetector(BaseDetector):
            def detect(self, trajectory):
                return DetectionResult(
                    is_hack=False,
                    confidence=0.0,
                    detector_name="valid",
                )

        detector = ValidDetector()
        result = detector.detect({})

        assert isinstance(result, DetectionResult)


@pytest.mark.skipif(not BASE_DETECTOR_AVAILABLE, reason="BaseDetector not available")
class TestDetectorChaining:
    """Test detector chaining functionality."""

    def test_chain_detectors(self):
        """Test chaining multiple detectors."""

        class Detector1(BaseDetector):
            def detect(self, trajectory):
                return DetectionResult(
                    is_hack=True,
                    confidence=0.8,
                    detector_name="detector1",
                )

        class Detector2(BaseDetector):
            def detect(self, trajectory):
                return DetectionResult(
                    is_hack=True,
                    confidence=0.9,
                    detector_name="detector2",
                )

        detectors = [Detector1(), Detector2()]
        trajectory = {}

        results = [d.detect(trajectory) for d in detectors]

        assert len(results) == 2
        assert all(r.is_hack for r in results)
