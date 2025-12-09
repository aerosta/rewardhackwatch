"""Tests for LLM judges."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rewardhackwatch.core.judges.base import (
    BaseJudge,
    JudgeResult,
    Verdict,
)
from rewardhackwatch.core.judges.claude_judge import ClaudeJudge
from rewardhackwatch.core.judges.llama_judge import LlamaJudge


class TestVerdict:
    """Tests for Verdict enum."""

    def test_verdict_values(self):
        """Test all verdict values exist."""
        assert Verdict.SAFE.value == "safe"
        assert Verdict.SUSPICIOUS.value == "suspicious"
        assert Verdict.DANGEROUS.value == "dangerous"
        assert Verdict.CRITICAL.value == "critical"


class TestJudgeResult:
    """Tests for JudgeResult dataclass."""

    def test_judge_result_creation(self):
        """Test creating a JudgeResult."""
        result = JudgeResult(
            judge_name="test_judge",
            verdict=Verdict.SUSPICIOUS,
            confidence=0.8,
            hack_score=0.6,
            misalignment_score=0.4,
            reasoning="Test reasoning",
        )
        assert result.judge_name == "test_judge"
        assert result.verdict == Verdict.SUSPICIOUS
        assert result.confidence == 0.8
        assert result.hack_score == 0.6
        assert result.misalignment_score == 0.4

    def test_judge_result_to_dict(self):
        """Test serializing JudgeResult to dict."""
        result = JudgeResult(
            judge_name="test_judge",
            verdict=Verdict.DANGEROUS,
            confidence=0.9,
            hack_score=0.8,
            misalignment_score=0.7,
            reasoning="Dangerous behavior detected",
            flagged_behaviors=["sys.exit(0)", "deception"],
            metadata={"model": "test-model"},
        )
        result_dict = result.to_dict()

        assert result_dict["judge_name"] == "test_judge"
        assert result_dict["verdict"] == "dangerous"
        assert result_dict["confidence"] == 0.9
        assert result_dict["hack_score"] == 0.8
        assert result_dict["misalignment_score"] == 0.7
        assert "sys.exit(0)" in result_dict["flagged_behaviors"]
        assert result_dict["metadata"]["model"] == "test-model"

    def test_judge_result_defaults(self):
        """Test JudgeResult default values."""
        result = JudgeResult(
            judge_name="test",
            verdict=Verdict.SAFE,
            confidence=0.5,
            hack_score=0.0,
            misalignment_score=0.0,
            reasoning="Safe",
        )
        assert result.flagged_behaviors == []
        assert result.metadata == {}


class TestBaseJudge:
    """Tests for BaseJudge class."""

    def test_format_trajectory_with_steps(self):
        """Test formatting trajectory with steps."""

        class ConcreteJudge(BaseJudge):
            async def judge(self, trajectory):
                pass

        judge = ConcreteJudge()
        trajectory = {
            "steps": [
                {"action": "run test", "result": "passed"},
                {"action": "modify code", "result": "done"},
            ]
        }

        formatted = judge._format_trajectory(trajectory)
        assert "TRAJECTORY STEPS" in formatted
        assert "Step 0" in formatted
        assert "Step 1" in formatted
        assert "run test" in formatted

    def test_format_trajectory_with_cot(self):
        """Test formatting trajectory with chain of thought."""

        class ConcreteJudge(BaseJudge):
            async def judge(self, trajectory):
                pass

        judge = ConcreteJudge()
        trajectory = {
            "cot_traces": ["Thinking about the problem...", "Let me try this approach"],
        }

        formatted = judge._format_trajectory(trajectory)
        assert "CHAIN OF THOUGHT" in formatted
        assert "Thinking about the problem" in formatted

    def test_format_trajectory_with_code(self):
        """Test formatting trajectory with code outputs."""

        class ConcreteJudge(BaseJudge):
            async def judge(self, trajectory):
                pass

        judge = ConcreteJudge()
        trajectory = {
            "code_outputs": ["def foo(): pass", "print('hello')"],
        }

        formatted = judge._format_trajectory(trajectory)
        assert "CODE OUTPUTS" in formatted
        assert "def foo(): pass" in formatted

    def test_parse_response_valid_json(self):
        """Test parsing valid JSON response."""

        class ConcreteJudge(BaseJudge):
            async def judge(self, trajectory):
                pass

        judge = ConcreteJudge()
        response = '{"hack_score": 0.8, "verdict": "DANGEROUS"}'
        parsed = judge._parse_response(response)

        assert parsed["hack_score"] == 0.8
        assert parsed["verdict"] == "DANGEROUS"

    def test_parse_response_json_in_text(self):
        """Test parsing JSON embedded in text."""

        class ConcreteJudge(BaseJudge):
            async def judge(self, trajectory):
                pass

        judge = ConcreteJudge()
        response = 'Here is my analysis: {"hack_score": 0.5, "verdict": "SUSPICIOUS"} Done.'
        parsed = judge._parse_response(response)

        assert parsed["hack_score"] == 0.5

    def test_parse_response_invalid_json(self):
        """Test parsing invalid JSON response."""

        class ConcreteJudge(BaseJudge):
            async def judge(self, trajectory):
                pass

        judge = ConcreteJudge()
        response = "This is not valid JSON at all"
        parsed = judge._parse_response(response)

        # Should return defaults
        assert parsed["hack_score"] == 0.5
        assert parsed["verdict"] == "SUSPICIOUS"
        assert "parse_error" in parsed["flagged_behaviors"]

    def test_verdict_from_string(self):
        """Test converting string to Verdict enum."""

        class ConcreteJudge(BaseJudge):
            async def judge(self, trajectory):
                pass

        judge = ConcreteJudge()

        assert judge._verdict_from_string("safe") == Verdict.SAFE
        assert judge._verdict_from_string("SAFE") == Verdict.SAFE
        assert judge._verdict_from_string("suspicious") == Verdict.SUSPICIOUS
        assert judge._verdict_from_string("dangerous") == Verdict.DANGEROUS
        assert judge._verdict_from_string("critical") == Verdict.CRITICAL
        assert judge._verdict_from_string("unknown") == Verdict.SUSPICIOUS  # default


class TestClaudeJudge:
    """Tests for ClaudeJudge."""

    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove ANTHROPIC_API_KEY if present
            import os

            os.environ.pop("ANTHROPIC_API_KEY", None)

            with pytest.raises(ValueError, match="Anthropic API key required"):
                ClaudeJudge()

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch("rewardhackwatch.core.judges.claude_judge.Anthropic"):
            with patch("rewardhackwatch.core.judges.claude_judge.AsyncAnthropic"):
                judge = ClaudeJudge(api_key="test-key")
                assert judge.model == "claude-opus-4-5-20251101"
                assert judge.name == "claude_judge"

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch("rewardhackwatch.core.judges.claude_judge.Anthropic"):
            with patch("rewardhackwatch.core.judges.claude_judge.AsyncAnthropic"):
                judge = ClaudeJudge(api_key="test-key", model="claude-sonnet-4-20250514")
                assert judge.model == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_judge_async(self):
        """Test async judge method."""
        with patch("rewardhackwatch.core.judges.claude_judge.Anthropic"):
            with patch("rewardhackwatch.core.judges.claude_judge.AsyncAnthropic") as mock_client:
                # Setup mock response with new JSON format
                mock_response = MagicMock()
                mock_response.content = [
                    MagicMock(
                        text='{"hack_detected": false, "hack_type": "none", "hack_confidence": 0.1, "misalignment_detected": false, "misalignment_type": "none", "misalignment_confidence": 0.1, "reasoning": "Clean"}'
                    )
                ]
                mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

                mock_client.return_value.messages.create = AsyncMock(return_value=mock_response)

                judge = ClaudeJudge(api_key="test-key")
                result = await judge.judge({"steps": [{"action": "test"}]})

                assert isinstance(result, JudgeResult)
                assert result.hack_score < 0.5  # Not detected = low score
                assert result.verdict == Verdict.SAFE

    def test_judge_sync(self):
        """Test sync judge method."""
        with patch("rewardhackwatch.core.judges.claude_judge.Anthropic") as mock_anthropic:
            with patch("rewardhackwatch.core.judges.claude_judge.AsyncAnthropic"):
                # Setup mock response with new JSON format
                mock_response = MagicMock()
                mock_response.content = [
                    MagicMock(
                        text='{"hack_detected": true, "hack_type": "sys_exit", "hack_confidence": 0.8, "misalignment_detected": false, "misalignment_type": "none", "misalignment_confidence": 0.2, "reasoning": "Concerning"}'
                    )
                ]
                mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

                mock_anthropic.return_value.messages.create = MagicMock(return_value=mock_response)

                judge = ClaudeJudge(api_key="test-key")
                result = judge.judge_sync({"cot_traces": ["Let me hack this"]})

                assert isinstance(result, JudgeResult)
                assert result.hack_score > 0.5  # Detected with high confidence
                assert result.verdict in [Verdict.SUSPICIOUS, Verdict.DANGEROUS]


class TestLlamaJudge:
    """Tests for LlamaJudge."""

    @patch("rewardhackwatch.core.judges.llama_judge.ollama")
    def test_init(self, mock_ollama):
        """Test initialization."""
        mock_ollama.list.return_value = {"models": [{"name": "llama3.1:8b"}]}

        judge = LlamaJudge()
        assert judge.model == "llama3.1:8b"
        assert judge.name == "llama_judge"

    @patch("rewardhackwatch.core.judges.llama_judge.ollama")
    def test_init_with_custom_model(self, mock_ollama):
        """Test initialization with custom model."""
        mock_ollama.list.return_value = {"models": [{"name": "llama3.1:3b"}]}

        judge = LlamaJudge(model="llama3.1:3b")
        assert judge.model == "llama3.1:3b"

    @patch("rewardhackwatch.core.judges.llama_judge.ollama")
    def test_init_model_not_found(self, mock_ollama, capsys):
        """Test initialization when model not found."""
        mock_ollama.list.return_value = {"models": []}

        judge = LlamaJudge()
        captured = capsys.readouterr()
        assert "Warning" in captured.out or judge is not None  # Should still initialize

    @patch("rewardhackwatch.core.judges.llama_judge.ollama")
    def test_init_ollama_not_running(self, mock_ollama, capsys):
        """Test initialization when Ollama not running."""
        mock_ollama.list.side_effect = Exception("Connection refused")

        judge = LlamaJudge()
        captured = capsys.readouterr()
        assert "Warning" in captured.out or judge is not None

    @pytest.mark.asyncio
    @patch("rewardhackwatch.core.judges.llama_judge.ollama")
    async def test_judge_async(self, mock_ollama):
        """Test async judge method."""
        mock_ollama.list.return_value = {"models": [{"name": "llama3.1:8b"}]}

        with patch("rewardhackwatch.core.judges.llama_judge.AsyncClient") as mock_client:
            # Use new JSON format
            mock_response = {
                "message": {
                    "content": '{"hack_detected": false, "hack_type": "none", "hack_confidence": 0.2, "misalignment_detected": false, "misalignment_type": "none", "misalignment_confidence": 0.1, "reasoning": "Test"}'
                }
            }
            mock_client.return_value.chat = AsyncMock(return_value=mock_response)

            judge = LlamaJudge()
            result = await judge.judge({"steps": [{"action": "test"}]})

            assert isinstance(result, JudgeResult)
            assert result.hack_score < 0.5  # Not detected
            assert result.verdict == Verdict.SAFE

    @patch("rewardhackwatch.core.judges.llama_judge.ollama")
    def test_judge_sync(self, mock_ollama):
        """Test sync judge method."""
        mock_ollama.list.return_value = {"models": [{"name": "llama3.1:8b"}]}
        # Use new JSON format
        mock_ollama.chat.return_value = {
            "message": {
                "content": '{"hack_detected": true, "hack_type": "sys_exit", "hack_confidence": 0.9, "misalignment_detected": false, "misalignment_type": "none", "misalignment_confidence": 0.2, "reasoning": "Bad"}'
            }
        }

        judge = LlamaJudge()
        result = judge.judge_sync({"cot_traces": ["Let me bypass this"]})

        assert isinstance(result, JudgeResult)
        assert result.hack_score > 0.5  # Detected with high confidence
        assert result.verdict == Verdict.DANGEROUS
        assert result.metadata["local"] is True

    @patch("rewardhackwatch.core.judges.llama_judge.ollama")
    def test_ensure_model_success(self, mock_ollama):
        """Test ensure_model success."""
        mock_ollama.pull.return_value = None

        result = LlamaJudge.ensure_model("llama3.1:8b")
        assert result is True
        mock_ollama.pull.assert_called_once_with("llama3.1:8b")

    @patch("rewardhackwatch.core.judges.llama_judge.ollama")
    def test_ensure_model_failure(self, mock_ollama, capsys):
        """Test ensure_model failure."""
        mock_ollama.pull.side_effect = Exception("Network error")

        result = LlamaJudge.ensure_model("llama3.1:8b")
        assert result is False
        captured = capsys.readouterr()
        assert "Failed" in captured.out
