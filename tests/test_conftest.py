"""Tests for pytest configuration and fixtures."""

import pytest


class TestFixtures:
    """Test that fixtures work correctly."""

    def test_tmp_path_fixture(self, tmp_path):
        """Test tmp_path fixture."""
        assert tmp_path.exists()
        assert tmp_path.is_dir()

        # Can create files
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        assert test_file.exists()

    def test_capsys_fixture(self, capsys):
        """Test capsys fixture."""
        print("test output")
        captured = capsys.readouterr()
        assert "test output" in captured.out


class TestMarkers:
    """Test pytest markers."""

    @pytest.mark.skip(reason="Testing skip marker")
    def test_skip_marker(self):
        """This test should be skipped."""
        assert False

    @pytest.mark.skipif(True, reason="Testing skipif marker")
    def test_skipif_marker(self):
        """This test should be skipped."""
        assert False

    @pytest.mark.parametrize("value", [1, 2, 3])
    def test_parametrize_marker(self, value):
        """Test parametrize marker."""
        assert value in [1, 2, 3]


class TestParametrization:
    """Test parametrized tests."""

    @pytest.mark.parametrize(
        "input,expected",
        [
            ("sys.exit(0)", True),
            ("print('hello')", False),
            ("os._exit(0)", True),
        ],
    )
    def test_hack_detection_parametrized(self, input, expected):
        """Test hack detection with various inputs."""
        # Simple pattern match for demo
        is_hack = "exit" in input and "(" in input
        assert is_hack == expected

    @pytest.mark.parametrize(
        "score,level",
        [
            (0.1, "low"),
            (0.5, "medium"),
            (0.9, "high"),
        ],
    )
    def test_risk_level_parametrized(self, score, level):
        """Test risk level classification."""
        if score < 0.3:
            result = "low"
        elif score < 0.7:
            result = "medium"
        else:
            result = "high"
        assert result == level


class TestExceptionHandling:
    """Test exception handling in tests."""

    def test_raises_value_error(self):
        """Test that ValueError is raised."""
        with pytest.raises(ValueError):
            raise ValueError("test error")

    def test_raises_with_match(self):
        """Test exception message matching."""
        with pytest.raises(ValueError, match="specific"):
            raise ValueError("specific error message")

    def test_no_exception(self):
        """Test that no exception is raised."""
        result = 1 + 1
        assert result == 2


class TestApproximation:
    """Test approximate comparisons."""

    def test_float_approximation(self):
        """Test approximate float comparison."""
        result = 0.1 + 0.2
        assert result == pytest.approx(0.3)

    def test_float_approximation_with_rel(self):
        """Test approximate comparison with relative tolerance."""
        result = 100.0
        assert result == pytest.approx(101.0, rel=0.02)

    def test_float_approximation_with_abs(self):
        """Test approximate comparison with absolute tolerance."""
        result = 0.001
        assert result == pytest.approx(0.002, abs=0.01)
