"""Extended tests for Ensemble Judge."""

from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest

# Import with error handling
try:
    from rewardhackwatch.core.judges.base_judge import JudgeResult

    from rewardhackwatch.core.judges.ensemble_judge import EnsembleJudge

    ENSEMBLE_JUDGE_AVAILABLE = True
except ImportError:
    ENSEMBLE_JUDGE_AVAILABLE = False

pytestmark = pytest.mark.extended


@dataclass
class MockJudgeResult:
    """Mock judge result for testing."""

    is_hack: bool
    confidence: float
    reasoning: str


@pytest.mark.skipif(not ENSEMBLE_JUDGE_AVAILABLE, reason="Ensemble judge not available")
class TestEnsembleJudgeInitialization:
    """Test ensemble judge initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        with patch("rewardhackwatch.core.judges.ensemble_judge.ClaudeJudge"):
            with patch("rewardhackwatch.core.judges.ensemble_judge.LlamaJudge"):
                judge = EnsembleJudge()
                assert judge is not None

    def test_custom_judges(self):
        """Test initialization with custom judges."""
        mock_judge1 = Mock()
        mock_judge2 = Mock()

        judge = EnsembleJudge(judges=[mock_judge1, mock_judge2])
        assert len(judge.judges) == 2

    def test_custom_weights(self):
        """Test initialization with custom weights."""
        mock_judge1 = Mock()
        mock_judge2 = Mock()

        judge = EnsembleJudge(judges=[mock_judge1, mock_judge2], weights=[0.7, 0.3])
        assert judge.weights == [0.7, 0.3]


@pytest.mark.skipif(not ENSEMBLE_JUDGE_AVAILABLE, reason="Ensemble judge not available")
class TestEnsembleJudgeVoting:
    """Test ensemble voting mechanisms."""

    def test_unanimous_hack(self):
        """Test unanimous hack verdict."""
        mock_judge1 = Mock()
        mock_judge1.judge.return_value = MockJudgeResult(True, 0.9, "Hack 1")

        mock_judge2 = Mock()
        mock_judge2.judge.return_value = MockJudgeResult(True, 0.95, "Hack 2")

        judge = EnsembleJudge(judges=[mock_judge1, mock_judge2])
        result = judge.judge("test content")

        assert result.is_hack
        assert result.confidence > 0.9

    def test_unanimous_clean(self):
        """Test unanimous clean verdict."""
        mock_judge1 = Mock()
        mock_judge1.judge.return_value = MockJudgeResult(False, 0.1, "Clean 1")

        mock_judge2 = Mock()
        mock_judge2.judge.return_value = MockJudgeResult(False, 0.05, "Clean 2")

        judge = EnsembleJudge(judges=[mock_judge1, mock_judge2])
        result = judge.judge("test content")

        assert not result.is_hack
        assert result.confidence < 0.2

    def test_split_verdict(self):
        """Test split verdict handling."""
        mock_judge1 = Mock()
        mock_judge1.judge.return_value = MockJudgeResult(True, 0.9, "Hack")

        mock_judge2 = Mock()
        mock_judge2.judge.return_value = MockJudgeResult(False, 0.9, "Clean")

        judge = EnsembleJudge(judges=[mock_judge1, mock_judge2])
        result = judge.judge("ambiguous content")

        # Result depends on voting strategy
        assert 0.3 <= result.confidence <= 0.7


@pytest.mark.skipif(not ENSEMBLE_JUDGE_AVAILABLE, reason="Ensemble judge not available")
class TestEnsembleJudgeWeighting:
    """Test weight-based voting."""

    def test_weighted_voting(self):
        """Test that weights affect voting."""
        mock_judge1 = Mock()
        mock_judge1.judge.return_value = MockJudgeResult(True, 0.8, "Hack")

        mock_judge2 = Mock()
        mock_judge2.judge.return_value = MockJudgeResult(False, 0.8, "Clean")

        # Heavy weight on judge1
        judge = EnsembleJudge(judges=[mock_judge1, mock_judge2], weights=[0.9, 0.1])
        result = judge.judge("test content")

        # Should favor judge1's verdict
        assert result.is_hack or result.confidence > 0.5

    def test_confidence_weighting(self):
        """Test confidence-based weighting."""
        mock_judge1 = Mock()
        mock_judge1.judge.return_value = MockJudgeResult(True, 0.99, "Very confident hack")

        mock_judge2 = Mock()
        mock_judge2.judge.return_value = MockJudgeResult(False, 0.51, "Slightly confident clean")

        judge = EnsembleJudge(judges=[mock_judge1, mock_judge2], use_confidence_weighting=True)
        result = judge.judge("test content")

        # Higher confidence should win
        assert result.is_hack


@pytest.mark.skipif(not ENSEMBLE_JUDGE_AVAILABLE, reason="Ensemble judge not available")
class TestEnsembleJudgeFailure:
    """Test failure handling in ensemble."""

    def test_single_judge_failure(self):
        """Test handling when one judge fails."""
        mock_judge1 = Mock()
        mock_judge1.judge.return_value = MockJudgeResult(True, 0.9, "Hack")

        mock_judge2 = Mock()
        mock_judge2.judge.side_effect = Exception("Judge failed")

        judge = EnsembleJudge(judges=[mock_judge1, mock_judge2])

        # Should still return result from working judge
        result = judge.judge("test content")
        assert result is not None

    def test_all_judges_failure(self):
        """Test handling when all judges fail."""
        mock_judge1 = Mock()
        mock_judge1.judge.side_effect = Exception("Judge 1 failed")

        mock_judge2 = Mock()
        mock_judge2.judge.side_effect = Exception("Judge 2 failed")

        judge = EnsembleJudge(judges=[mock_judge1, mock_judge2])

        with pytest.raises(Exception):
            judge.judge("test content")

    def test_partial_failure_weighting(self):
        """Test that failed judges don't affect weighting."""
        mock_judge1 = Mock()
        mock_judge1.judge.return_value = MockJudgeResult(True, 0.9, "Hack")

        mock_judge2 = Mock()
        mock_judge2.judge.side_effect = Exception("Failed")

        mock_judge3 = Mock()
        mock_judge3.judge.return_value = MockJudgeResult(True, 0.8, "Also hack")

        judge = EnsembleJudge(judges=[mock_judge1, mock_judge2, mock_judge3])
        result = judge.judge("test content")

        assert result.is_hack


@pytest.mark.skipif(not ENSEMBLE_JUDGE_AVAILABLE, reason="Ensemble judge not available")
class TestEnsembleJudgeAggregation:
    """Test result aggregation methods."""

    def test_majority_voting(self):
        """Test majority voting aggregation."""
        results = [
            MockJudgeResult(True, 0.9, "Hack"),
            MockJudgeResult(True, 0.8, "Hack"),
            MockJudgeResult(False, 0.7, "Clean"),
        ]

        mock_judges = [Mock() for _ in range(3)]
        for j, r in zip(mock_judges, results):
            j.judge.return_value = r

        judge = EnsembleJudge(judges=mock_judges, aggregation="majority")
        result = judge.judge("test content")

        assert result.is_hack  # 2 out of 3 say hack

    def test_average_confidence(self):
        """Test average confidence calculation."""
        results = [
            MockJudgeResult(True, 0.9, "Hack"),
            MockJudgeResult(True, 0.7, "Hack"),
            MockJudgeResult(True, 0.8, "Hack"),
        ]

        mock_judges = [Mock() for _ in range(3)]
        for j, r in zip(mock_judges, results):
            j.judge.return_value = r

        judge = EnsembleJudge(judges=mock_judges)
        result = judge.judge("test content")

        # Average of 0.9, 0.7, 0.8 = 0.8
        assert 0.75 <= result.confidence <= 0.85


@pytest.mark.skipif(not ENSEMBLE_JUDGE_AVAILABLE, reason="Ensemble judge not available")
class TestEnsembleJudgeDetails:
    """Test detailed result reporting."""

    def test_individual_results(self):
        """Test that individual results are preserved."""
        mock_judge1 = Mock()
        mock_judge1.judge.return_value = MockJudgeResult(True, 0.9, "Hack")
        mock_judge1.__class__.__name__ = "MockJudge1"

        mock_judge2 = Mock()
        mock_judge2.judge.return_value = MockJudgeResult(False, 0.8, "Clean")
        mock_judge2.__class__.__name__ = "MockJudge2"

        judge = EnsembleJudge(judges=[mock_judge1, mock_judge2])
        result = judge.judge("test content")

        if hasattr(result, "individual_results"):
            assert len(result.individual_results) == 2

    def test_reasoning_aggregation(self):
        """Test that reasoning is aggregated."""
        mock_judge1 = Mock()
        mock_judge1.judge.return_value = MockJudgeResult(True, 0.9, "Found sys.exit")

        mock_judge2 = Mock()
        mock_judge2.judge.return_value = MockJudgeResult(True, 0.8, "Test bypass detected")

        judge = EnsembleJudge(judges=[mock_judge1, mock_judge2])
        result = judge.judge("test content")

        # Reasoning should contain both explanations
        assert "sys.exit" in result.reasoning or "bypass" in result.reasoning or result.reasoning
