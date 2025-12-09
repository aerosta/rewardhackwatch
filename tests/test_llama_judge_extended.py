"""Extended tests for Llama Judge."""

from unittest.mock import patch

import pytest

# Import with error handling
try:
    from rewardhackwatch.core.judges.llama_judge import LlamaJudge

    LLAMA_JUDGE_AVAILABLE = True
except ImportError:
    LLAMA_JUDGE_AVAILABLE = False

pytestmark = pytest.mark.extended


@pytest.mark.skipif(not LLAMA_JUDGE_AVAILABLE, reason="Llama judge not available")
class TestLlamaJudgeInitialization:
    """Test Llama judge initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        with patch("rewardhackwatch.core.judges.llama_judge.OllamaClient"):
            judge = LlamaJudge()
            assert judge is not None

    def test_custom_model(self):
        """Test initialization with custom model."""
        with patch("rewardhackwatch.core.judges.llama_judge.OllamaClient"):
            judge = LlamaJudge(model_name="llama-3.2-8b")
            assert judge.model_name == "llama-3.2-8b"

    def test_custom_endpoint(self):
        """Test initialization with custom endpoint."""
        with patch("rewardhackwatch.core.judges.llama_judge.OllamaClient"):
            judge = LlamaJudge(endpoint="http://localhost:11434")
            assert judge.endpoint == "http://localhost:11434"


@pytest.mark.skipif(not LLAMA_JUDGE_AVAILABLE, reason="Llama judge not available")
class TestLlamaJudgeJudgment:
    """Test Llama judge judgment functionality."""

    @pytest.fixture
    def mock_judge(self):
        """Create a mock judge."""
        with patch("rewardhackwatch.core.judges.llama_judge.OllamaClient"):
            return LlamaJudge()

    def test_judge_hack(self, mock_judge):
        """Test judging a hack trajectory."""
        with patch.object(mock_judge, "_generate") as mock_gen:
            mock_gen.return_value = (
                '{"is_hack": true, "confidence": 0.9, "reasoning": "Bypass detected"}'
            )

            result = mock_judge.judge("sys.exit(0)")

            assert result.is_hack

    def test_judge_clean(self, mock_judge):
        """Test judging a clean trajectory."""
        with patch.object(mock_judge, "_generate") as mock_gen:
            mock_gen.return_value = (
                '{"is_hack": false, "confidence": 0.1, "reasoning": "Legitimate code"}'
            )

            result = mock_judge.judge("def add(a, b): return a + b")

            assert not result.is_hack


@pytest.mark.skipif(not LLAMA_JUDGE_AVAILABLE, reason="Llama judge not available")
class TestLlamaJudgeResponseParsing:
    """Test response parsing for Llama judge."""

    @pytest.fixture
    def judge(self):
        with patch("rewardhackwatch.core.judges.llama_judge.OllamaClient"):
            return LlamaJudge()

    def test_json_response_parsing(self, judge):
        """Test parsing JSON response."""
        response = '{"is_hack": true, "confidence": 0.85, "reasoning": "Test"}'
        parsed = judge._parse_response(response)

        assert parsed["is_hack"]
        assert parsed["confidence"] == 0.85

    def test_malformed_json_handling(self, judge):
        """Test handling of malformed JSON."""
        response = "This is not valid JSON"

        # Should handle gracefully
        try:
            judge._parse_response(response)
        except Exception:
            pass  # Expected behavior

    def test_missing_fields(self, judge):
        """Test handling of missing fields in response."""
        response = '{"is_hack": true}'

        judge._parse_response(response)
        # Should provide defaults for missing fields


@pytest.mark.skipif(not LLAMA_JUDGE_AVAILABLE, reason="Llama judge not available")
class TestLlamaJudgeLocalModel:
    """Test local model handling."""

    def test_model_availability_check(self):
        """Test model availability checking."""
        with patch("rewardhackwatch.core.judges.llama_judge.OllamaClient") as mock_client:
            mock_client.return_value.list.return_value = {"models": [{"name": "llama-3.2-8b"}]}

            judge = LlamaJudge(model_name="llama-3.2-8b")

            assert judge.is_model_available()

    def test_model_not_available(self):
        """Test behavior when model is not available."""
        with patch("rewardhackwatch.core.judges.llama_judge.OllamaClient") as mock_client:
            mock_client.return_value.list.return_value = {"models": []}

            judge = LlamaJudge(model_name="nonexistent-model")

            assert not judge.is_model_available()


@pytest.mark.skipif(not LLAMA_JUDGE_AVAILABLE, reason="Llama judge not available")
class TestLlamaJudgePerformance:
    """Test performance-related functionality."""

    @pytest.fixture
    def judge(self):
        with patch("rewardhackwatch.core.judges.llama_judge.OllamaClient"):
            return LlamaJudge()

    def test_timeout_handling(self, judge):
        """Test timeout handling."""
        with patch.object(judge, "_generate") as mock_gen:
            mock_gen.side_effect = TimeoutError("Request timed out")

            with pytest.raises(TimeoutError):
                judge.judge("test content")

    def test_batch_processing(self, judge):
        """Test batch processing capability."""
        if hasattr(judge, "batch_judge"):
            with patch.object(judge, "_generate") as mock_gen:
                mock_gen.return_value = (
                    '{"is_hack": false, "confidence": 0.1, "reasoning": "Clean"}'
                )

                trajectories = ["traj1", "traj2", "traj3"]
                results = judge.batch_judge(trajectories)

                assert len(results) == 3


@pytest.mark.skipif(not LLAMA_JUDGE_AVAILABLE, reason="Llama judge not available")
class TestLlamaJudgeEdgeCases:
    """Test edge cases for Llama judge."""

    @pytest.fixture
    def judge(self):
        with patch("rewardhackwatch.core.judges.llama_judge.OllamaClient"):
            return LlamaJudge()

    def test_empty_input(self, judge):
        """Test handling of empty input."""
        with patch.object(judge, "_generate") as mock_gen:
            mock_gen.return_value = '{"is_hack": false, "confidence": 0.0, "reasoning": "Empty"}'

            result = judge.judge("")
            assert result is not None

    def test_very_long_input(self, judge):
        """Test handling of very long input."""
        long_input = "x = 1\n" * 10000

        with patch.object(judge, "_generate") as mock_gen:
            mock_gen.return_value = '{"is_hack": false, "confidence": 0.1, "reasoning": "Test"}'

            result = judge.judge(long_input)
            assert result is not None

    def test_special_characters(self, judge):
        """Test handling of special characters."""
        special_input = "print('Hello\\n\\t\\r\"World\"')"

        with patch.object(judge, "_generate") as mock_gen:
            mock_gen.return_value = '{"is_hack": false, "confidence": 0.1, "reasoning": "Test"}'

            result = judge.judge(special_input)
            assert result is not None

    def test_unicode_input(self, judge):
        """Test handling of unicode input."""
        unicode_input = "print('你好世界 🌍')"

        with patch.object(judge, "_generate") as mock_gen:
            mock_gen.return_value = '{"is_hack": false, "confidence": 0.1, "reasoning": "Test"}'

            result = judge.judge(unicode_input)
            assert result is not None


@pytest.mark.skipif(not LLAMA_JUDGE_AVAILABLE, reason="Llama judge not available")
class TestLlamaJudgeConsistency:
    """Test judgment consistency."""

    @pytest.fixture
    def judge(self):
        with patch("rewardhackwatch.core.judges.llama_judge.OllamaClient"):
            return LlamaJudge()

    def test_deterministic_output(self, judge):
        """Test that same input produces consistent output."""
        with patch.object(judge, "_generate") as mock_gen:
            # Set temperature to 0 for deterministic output
            mock_gen.return_value = '{"is_hack": true, "confidence": 0.9, "reasoning": "Test"}'

            result1 = judge.judge("sys.exit(0)")
            result2 = judge.judge("sys.exit(0)")

            # Results should be consistent
            assert result1.is_hack == result2.is_hack
