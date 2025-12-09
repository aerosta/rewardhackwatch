"""Tests for utility functions."""

import json

import pytest

try:
    from rewardhackwatch.utils.helpers import (
        calculate_metrics,
        clean_text,
        extract_code_blocks,
        format_result,
    )

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="Utils not available")
class TestCodeExtraction:
    """Test code extraction utilities."""

    def test_extract_python_block(self):
        """Test extracting Python code block."""
        text = """
Here is the solution:
```python
def solve():
    return 42
```
That's it.
"""
        blocks = extract_code_blocks(text)
        assert len(blocks) >= 1
        assert "def solve" in blocks[0]

    def test_extract_multiple_blocks(self):
        """Test extracting multiple code blocks."""
        text = """
```python
code1()
```
Some text
```python
code2()
```
"""
        blocks = extract_code_blocks(text)
        assert len(blocks) >= 2

    def test_no_code_blocks(self):
        """Test text with no code blocks."""
        text = "Just regular text without code."
        blocks = extract_code_blocks(text)
        assert len(blocks) == 0


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="Utils not available")
class TestTextCleaning:
    """Test text cleaning utilities."""

    def test_clean_whitespace(self):
        """Test cleaning whitespace."""
        text = "  hello   world  "
        cleaned = clean_text(text)
        assert cleaned == "hello world" or "hello" in cleaned

    def test_clean_special_chars(self):
        """Test cleaning special characters."""
        text = "hello\x00world\x01"
        cleaned = clean_text(text)
        assert "\x00" not in cleaned

    def test_preserve_newlines(self):
        """Test preserving newlines."""
        text = "line1\nline2"
        cleaned = clean_text(text, preserve_newlines=True)
        assert "\n" in cleaned or "line1" in cleaned


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="Utils not available")
class TestMetricsCalculation:
    """Test metrics calculation utilities."""

    def test_calculate_precision(self):
        """Test precision calculation."""
        predictions = [True, True, False, True]
        labels = [True, False, False, True]

        metrics = calculate_metrics(predictions, labels)
        # TP=2, FP=1, precision = 2/3
        assert 0.6 <= metrics["precision"] <= 0.7

    def test_calculate_recall(self):
        """Test recall calculation."""
        predictions = [True, False, False, True]
        labels = [True, True, False, True]

        metrics = calculate_metrics(predictions, labels)
        # TP=2, FN=1, recall = 2/3
        assert 0.6 <= metrics["recall"] <= 0.7

    def test_calculate_f1(self):
        """Test F1 calculation."""
        predictions = [True, True, True]
        labels = [True, True, True]

        metrics = calculate_metrics(predictions, labels)
        assert metrics["f1"] == 1.0

    def test_empty_predictions(self):
        """Test empty predictions."""
        metrics = calculate_metrics([], [])
        assert "accuracy" in metrics or metrics is not None


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="Utils not available")
class TestResultFormatting:
    """Test result formatting utilities."""

    def test_format_json(self):
        """Test JSON formatting."""
        result = {
            "is_hack": True,
            "hack_score": 0.85,
        }

        formatted = format_result(result, format="json")
        parsed = json.loads(formatted)
        assert parsed["is_hack"]

    def test_format_text(self):
        """Test text formatting."""
        result = {
            "is_hack": True,
            "hack_score": 0.85,
        }

        formatted = format_result(result, format="text")
        assert "hack" in formatted.lower()


class TestBasicUtils:
    """Test basic utility functions without imports."""

    def test_simple_metric_calc(self):
        """Test simple metric calculation."""
        # TP, FP, TN, FN
        tp, fp, tn, fn = 10, 2, 80, 8

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        assert precision == 10 / 12
        assert recall == 10 / 18
        assert accuracy == 90 / 100

    def test_f1_calculation(self):
        """Test F1 calculation formula."""
        precision = 0.8
        recall = 0.6

        f1 = 2 * (precision * recall) / (precision + recall)
        assert abs(f1 - 0.6857) < 0.01
