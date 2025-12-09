"""Tests for BaseJudge interface."""

import pytest

try:
    from rewardhackwatch.core.judges.base_judge import BaseJudge, JudgeResult

    BASE_JUDGE_AVAILABLE = True
except ImportError:
    BASE_JUDGE_AVAILABLE = False


@pytest.mark.skipif(not BASE_JUDGE_AVAILABLE, reason="BaseJudge not available")
class TestJudgeResult:
    """Test JudgeResult dataclass."""

    def test_create_result(self):
        """Test creating a judge result."""
        result = JudgeResult(
            is_hack=True,
            confidence=0.95,
            reasoning="Found sys.exit bypass",
        )

        assert result.is_hack
        assert result.confidence == 0.95
        assert "sys.exit" in result.reasoning

    def test_result_defaults(self):
        """Test result with default values."""
        result = JudgeResult(
            is_hack=False,
            confidence=0.1,
            reasoning="",
        )

        assert result.reasoning == ""


@pytest.mark.skipif(not BASE_JUDGE_AVAILABLE, reason="BaseJudge not available")
class TestBaseJudgeInterface:
    """Test BaseJudge abstract interface."""

    def test_cannot_instantiate_base(self):
        """Test that BaseJudge cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseJudge()

    def test_subclass_must_implement_judge(self):
        """Test that subclass must implement judge."""

        class IncompleteJudge(BaseJudge):
            pass

        with pytest.raises(TypeError):
            IncompleteJudge()

    def test_valid_subclass(self):
        """Test creating a valid subclass."""

        class ValidJudge(BaseJudge):
            def judge(self, content):
                return JudgeResult(
                    is_hack=False,
                    confidence=0.0,
                    reasoning="No issues found",
                )

        judge = ValidJudge()
        result = judge.judge("test content")

        assert isinstance(result, JudgeResult)
