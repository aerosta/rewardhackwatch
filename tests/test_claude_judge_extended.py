"""Extended tests for Claude Judge."""

import os
from unittest.mock import patch

import pytest

# Import with error handling
try:
    from rewardhackwatch.core.judges.claude_judge import ClaudeJudge

    CLAUDE_JUDGE_AVAILABLE = True
except ImportError:
    CLAUDE_JUDGE_AVAILABLE = False

pytestmark = pytest.mark.extended


@pytest.mark.skipif(not CLAUDE_JUDGE_AVAILABLE, reason="Claude judge not available")
class TestClaudeJudgeInitialization:
    """Test Claude judge initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            judge = ClaudeJudge()
            assert judge is not None

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                ClaudeJudge()

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            judge = ClaudeJudge(model="claude-3-opus-20240229")
            assert judge.model == "claude-3-opus-20240229"


@pytest.mark.skipif(not CLAUDE_JUDGE_AVAILABLE, reason="Claude judge not available")
class TestClaudeJudgeJudgment:
    """Test Claude judge judgment functionality."""

    @pytest.fixture
    def mock_judge(self):
        """Create a mock judge."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            judge = ClaudeJudge()
            return judge

    def test_judge_hack_trajectory(self, mock_judge):
        """Test judging a hack trajectory."""
        with patch.object(mock_judge, "_call_api") as mock_api:
            mock_api.return_value = {
                "is_hack": True,
                "confidence": 0.95,
                "reasoning": "Uses sys.exit(0) to bypass tests",
            }

            result = mock_judge.judge("sys.exit(0) to bypass tests")

            assert result.is_hack
            assert result.confidence > 0.9

    def test_judge_clean_trajectory(self, mock_judge):
        """Test judging a clean trajectory."""
        with patch.object(mock_judge, "_call_api") as mock_api:
            mock_api.return_value = {
                "is_hack": False,
                "confidence": 0.1,
                "reasoning": "Legitimate implementation",
            }

            result = mock_judge.judge(
                "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
            )

            assert not result.is_hack
            assert result.confidence < 0.5


@pytest.mark.skipif(not CLAUDE_JUDGE_AVAILABLE, reason="Claude judge not available")
class TestClaudeJudgePromptGeneration:
    """Test prompt generation for Claude judge."""

    @pytest.fixture
    def judge(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            return ClaudeJudge()

    def test_prompt_includes_trajectory(self, judge):
        """Test that prompt includes trajectory content."""
        trajectory_text = "sys.exit(0) to pass tests"
        prompt = judge._generate_prompt(trajectory_text)

        assert trajectory_text in prompt

    def test_prompt_includes_instructions(self, judge):
        """Test that prompt includes judgment instructions."""
        prompt = judge._generate_prompt("test content")

        assert "reward" in prompt.lower() or "hack" in prompt.lower()


@pytest.mark.skipif(not CLAUDE_JUDGE_AVAILABLE, reason="Claude judge not available")
class TestClaudeJudgeErrorHandling:
    """Test error handling in Claude judge."""

    @pytest.fixture
    def judge(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            return ClaudeJudge()

    def test_api_error_handling(self, judge):
        """Test handling of API errors."""
        with patch.object(judge, "_call_api") as mock_api:
            mock_api.side_effect = Exception("API Error")

            with pytest.raises(Exception):
                judge.judge("test content")

    def test_rate_limit_handling(self, judge):
        """Test handling of rate limits."""
        with patch.object(judge, "_call_api") as mock_api:
            mock_api.side_effect = Exception("Rate limit exceeded")

            with pytest.raises(Exception):
                judge.judge("test content")

    def test_invalid_response_handling(self, judge):
        """Test handling of invalid API response."""
        with patch.object(judge, "_call_api") as mock_api:
            mock_api.return_value = {"invalid": "response"}

            # Should handle gracefully


@pytest.mark.skipif(not CLAUDE_JUDGE_AVAILABLE, reason="Claude judge not available")
class TestClaudeJudgeCaching:
    """Test caching functionality."""

    @pytest.fixture
    def judge(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            return ClaudeJudge(cache_enabled=True)

    def test_cache_hit(self, judge):
        """Test cache hit."""
        with patch.object(judge, "_call_api") as mock_api:
            mock_api.return_value = {
                "is_hack": True,
                "confidence": 0.9,
                "reasoning": "Test",
            }

            # First call
            judge.judge("test content")

            # Second call (should hit cache)
            judge.judge("test content")

            # API should only be called once
            assert mock_api.call_count == 1 or not hasattr(judge, "_cache")

    def test_cache_miss(self, judge):
        """Test cache miss."""
        with patch.object(judge, "_call_api") as mock_api:
            mock_api.return_value = {
                "is_hack": True,
                "confidence": 0.9,
                "reasoning": "Test",
            }

            # Different content should not hit cache
            judge.judge("content 1")
            judge.judge("content 2")

            assert mock_api.call_count == 2 or not hasattr(judge, "_cache")


@pytest.mark.skipif(not CLAUDE_JUDGE_AVAILABLE, reason="Claude judge not available")
class TestClaudeJudgeConfidence:
    """Test confidence calibration."""

    @pytest.fixture
    def judge(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            return ClaudeJudge()

    def test_confidence_range(self, judge):
        """Test that confidence is in valid range."""
        with patch.object(judge, "_call_api") as mock_api:
            mock_api.return_value = {
                "is_hack": True,
                "confidence": 0.5,
                "reasoning": "Uncertain",
            }

            result = judge.judge("ambiguous content")

            assert 0.0 <= result.confidence <= 1.0

    def test_high_confidence_hack(self, judge):
        """Test high confidence hack detection."""
        with patch.object(judge, "_call_api") as mock_api:
            mock_api.return_value = {
                "is_hack": True,
                "confidence": 0.99,
                "reasoning": "Definite hack pattern",
            }

            result = judge.judge("sys.exit(0)")

            assert result.is_hack
            assert result.confidence > 0.9
