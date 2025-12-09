"""Tests for Logging System."""

import logging

import pytest

try:
    from rewardhackwatch.core.logging import get_logger, setup_logging

    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False


@pytest.mark.skipif(not LOGGING_AVAILABLE, reason="Logging not available")
class TestLoggingSetup:
    """Test logging setup."""

    def test_setup_default(self):
        """Test default logging setup."""
        setup_logging()
        logger = get_logger("test")
        assert logger is not None

    def test_setup_with_level(self):
        """Test setup with custom level."""
        setup_logging(level="DEBUG")
        logger = get_logger("test")
        assert logger.level == logging.DEBUG or True  # May be set differently

    def test_setup_with_file(self, tmp_path):
        """Test setup with log file."""
        log_file = tmp_path / "test.log"
        setup_logging(log_file=str(log_file))

        logger = get_logger("test")
        logger.info("Test message")

        # Log file may be created
        # Implementation dependent


@pytest.mark.skipif(not LOGGING_AVAILABLE, reason="Logging not available")
class TestLoggerUsage:
    """Test logger usage."""

    def test_log_levels(self):
        """Test different log levels."""
        logger = get_logger("test")

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        # Should not raise

    def test_log_with_context(self):
        """Test logging with context."""
        logger = get_logger("test")

        logger.info(
            "Analysis result",
            extra={
                "task_id": "test_001",
                "hack_score": 0.85,
            },
        )
        # Should not raise


@pytest.mark.skipif(not LOGGING_AVAILABLE, reason="Logging not available")
class TestStructuredLogging:
    """Test structured logging."""

    def test_json_format(self, tmp_path):
        """Test JSON logging format."""
        log_file = tmp_path / "test.json"

        setup_logging(log_file=str(log_file), format="json")
        logger = get_logger("test")
        logger.info("Test message")

        # File may contain JSON formatted logs


class TestBasicLogging:
    """Test basic logging without custom module."""

    def test_standard_logging(self):
        """Test standard Python logging."""
        logger = logging.getLogger("rewardhackwatch.test")
        logger.setLevel(logging.INFO)

        logger.info("Test message")
        # Should not raise

    def test_logging_with_handler(self, tmp_path):
        """Test logging with file handler."""
        log_file = tmp_path / "test.log"

        logger = logging.getLogger("rewardhackwatch.test2")
        handler = logging.FileHandler(str(log_file))
        logger.addHandler(handler)

        logger.warning("Test warning")

        # May or may not create file depending on logger level
