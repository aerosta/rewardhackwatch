"""Tests for EnsembleJudge."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rewardhackwatch.core.judges import (
    EnsembleConfig,
    EnsembleJudge,
    EnsembleResult,
    JudgeResult,
    Verdict,
)


class TestEnsembleConfig:
    """Tests for EnsembleConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EnsembleConfig()
        assert config.strategy == "weighted"
        assert config.threshold == 0.5
        assert len(config.weights) == 2
        assert "claude_judge" in config.weights

    def test_custom_config(self):
        """Test custom configuration."""
        config = EnsembleConfig(
            strategy="majority",
            weights={"claude": 0.7, "llama": 0.3},
            threshold=0.6,
            escalation_threshold=0.8,
        )
        assert config.strategy == "majority"
        assert config.weights["claude"] == 0.7
        assert config.threshold == 0.6

    def test_require_consensus_for(self):
        """Test require consensus configuration."""
        config = EnsembleConfig(require_consensus_for=["CRITICAL", "DANGEROUS"])
        assert "CRITICAL" in config.require_consensus_for
        assert "DANGEROUS" in config.require_consensus_for


class TestEnsembleResult:
    """Tests for EnsembleResult."""

    def test_result_creation(self):
        """Test creating ensemble result."""
        judge_result = JudgeResult(
            judge_name="test",
            verdict=Verdict.SUSPICIOUS,
            confidence=0.8,
            hack_score=0.6,
            misalignment_score=0.4,
            reasoning="Test reasoning",
        )
        result = EnsembleResult(
            verdict=Verdict.SUSPICIOUS,
            confidence=0.8,
            hack_score=0.6,
            misalignment_score=0.4,
            reasoning="Ensemble reasoning",
            individual_results={"test": judge_result},
            agreement=True,
            votes={"test": "SUSPICIOUS"},
            strategy_used="weighted",
        )

        assert result.verdict == Verdict.SUSPICIOUS
        assert len(result.individual_results) == 1
        assert result.agreement

    def test_to_dict(self):
        """Test serialization."""
        judge_result = JudgeResult(
            judge_name="test",
            verdict=Verdict.SAFE,
            confidence=0.9,
            hack_score=0.1,
            misalignment_score=0.1,
            reasoning="Clean",
        )
        result = EnsembleResult(
            verdict=Verdict.SAFE,
            confidence=0.9,
            hack_score=0.1,
            misalignment_score=0.1,
            reasoning="All agree safe",
            individual_results={"test": judge_result},
            agreement=True,
            votes={"test": "SAFE"},
            strategy_used="majority",
        )

        d = result.to_dict()
        assert "verdict" in d
        assert "individual_results" in d
        assert d["agreement"]
        assert d["strategy_used"] == "majority"


class TestEnsembleJudge:
    """Tests for EnsembleJudge."""

    @pytest.fixture
    def mock_judges(self):
        """Create mock judges."""
        mock_claude = MagicMock()
        mock_claude.name = "claude_judge"
        mock_claude.judge = AsyncMock(
            return_value=JudgeResult(
                judge_name="claude_judge",
                verdict=Verdict.SUSPICIOUS,
                confidence=0.8,
                hack_score=0.7,
                misalignment_score=0.5,
                reasoning="Claude found issues",
            )
        )

        mock_llama = MagicMock()
        mock_llama.name = "llama_judge"
        mock_llama.judge = AsyncMock(
            return_value=JudgeResult(
                judge_name="llama_judge",
                verdict=Verdict.DANGEROUS,
                confidence=0.9,
                hack_score=0.8,
                misalignment_score=0.6,
                reasoning="Llama found serious issues",
            )
        )

        return [mock_claude, mock_llama]

    @pytest.fixture
    def agreeing_judges(self):
        """Create judges that agree."""
        mock_claude = MagicMock()
        mock_claude.name = "claude_judge"
        mock_claude.judge = AsyncMock(
            return_value=JudgeResult(
                judge_name="claude_judge",
                verdict=Verdict.SAFE,
                confidence=0.9,
                hack_score=0.1,
                misalignment_score=0.1,
                reasoning="Looks clean",
            )
        )

        mock_llama = MagicMock()
        mock_llama.name = "llama_judge"
        mock_llama.judge = AsyncMock(
            return_value=JudgeResult(
                judge_name="llama_judge",
                verdict=Verdict.SAFE,
                confidence=0.85,
                hack_score=0.15,
                misalignment_score=0.1,
                reasoning="Also looks clean",
            )
        )

        return [mock_claude, mock_llama]

    def test_initialization(self, mock_judges):
        """Test ensemble judge initialization."""
        ensemble = EnsembleJudge(judges=mock_judges)
        assert ensemble.name == "ensemble_judge"
        assert len(ensemble.judges) == 2

    def test_initialization_with_config(self, mock_judges):
        """Test initialization with custom config."""
        config = EnsembleConfig(strategy="consensus")
        ensemble = EnsembleJudge(judges=mock_judges, config=config)
        assert ensemble.config.strategy == "consensus"

    def test_add_judge(self, mock_judges):
        """Test adding a judge."""
        ensemble = EnsembleJudge(judges=[mock_judges[0]])
        assert len(ensemble.judges) == 1

        ensemble.add_judge(mock_judges[1])
        assert len(ensemble.judges) == 2

    @pytest.mark.asyncio
    async def test_judge_weighted_vote(self, mock_judges):
        """Test weighted vote strategy."""
        config = EnsembleConfig(
            strategy="weighted",
            weights={"claude_judge": 0.6, "llama_judge": 0.4},
        )
        ensemble = EnsembleJudge(judges=mock_judges, config=config)

        trajectory = {"steps": [{"action": "test"}]}
        result = await ensemble.judge(trajectory)

        assert isinstance(result, EnsembleResult)
        assert result.strategy_used == "weighted"
        assert len(result.individual_results) == 2

    @pytest.mark.asyncio
    async def test_judge_majority_vote(self, mock_judges):
        """Test majority vote strategy."""
        config = EnsembleConfig(strategy="majority")
        ensemble = EnsembleJudge(judges=mock_judges, config=config)

        trajectory = {"steps": []}
        result = await ensemble.judge(trajectory)

        assert result.strategy_used == "majority"
        assert result.verdict in [Verdict.SUSPICIOUS, Verdict.DANGEROUS]

    @pytest.mark.asyncio
    async def test_judge_consensus(self, agreeing_judges):
        """Test consensus strategy with agreeing judges."""
        config = EnsembleConfig(strategy="consensus")
        ensemble = EnsembleJudge(judges=agreeing_judges, config=config)

        trajectory = {"steps": []}
        result = await ensemble.judge(trajectory)

        assert result.strategy_used == "consensus"
        assert result.verdict == Verdict.SAFE
        assert result.agreement

    @pytest.mark.asyncio
    async def test_judge_consensus_disagreement(self, mock_judges):
        """Test consensus strategy with disagreeing judges."""
        config = EnsembleConfig(strategy="consensus")
        ensemble = EnsembleJudge(judges=mock_judges, config=config)

        trajectory = {"steps": []}
        result = await ensemble.judge(trajectory)

        # No consensus - should mark as suspicious
        assert result.verdict == Verdict.SUSPICIOUS
        assert not result.agreement

    @pytest.mark.asyncio
    async def test_judge_any_vote(self, mock_judges):
        """Test any-flag strategy."""
        config = EnsembleConfig(strategy="any")
        ensemble = EnsembleJudge(judges=mock_judges, config=config)

        trajectory = {"steps": []}
        result = await ensemble.judge(trajectory)

        assert result.strategy_used == "any"
        # Uses max hack score - should be DANGEROUS or CRITICAL
        assert result.verdict != Verdict.SAFE

    @pytest.mark.asyncio
    async def test_agreement_when_judges_agree(self, agreeing_judges):
        """Test agreement flag when judges agree."""
        ensemble = EnsembleJudge(judges=agreeing_judges)

        trajectory = {"steps": []}
        result = await ensemble.judge(trajectory)

        assert result.agreement

    @pytest.mark.asyncio
    async def test_disagreement_when_judges_differ(self, mock_judges):
        """Test agreement flag when judges disagree."""
        ensemble = EnsembleJudge(judges=mock_judges)

        trajectory = {"steps": []}
        result = await ensemble.judge(trajectory)

        # SUSPICIOUS vs DANGEROUS should disagree
        assert not result.agreement

    @pytest.mark.asyncio
    async def test_votes_recorded(self, mock_judges):
        """Test that individual votes are recorded."""
        ensemble = EnsembleJudge(judges=mock_judges)

        trajectory = {"steps": []}
        result = await ensemble.judge(trajectory)

        assert "claude_judge" in result.votes
        assert "llama_judge" in result.votes
        # Votes are stored as lowercase verdict values
        assert result.votes["claude_judge"] == "suspicious"
        assert result.votes["llama_judge"] == "dangerous"

    @pytest.mark.asyncio
    async def test_empty_judges_raises(self):
        """Test that empty judges raises error."""
        ensemble = EnsembleJudge(judges=[])

        trajectory = {"steps": []}
        with pytest.raises(ValueError, match="No judges configured"):
            await ensemble.judge(trajectory)

    @pytest.mark.asyncio
    async def test_single_judge(self, mock_judges):
        """Test with single judge."""
        ensemble = EnsembleJudge(judges=[mock_judges[0]])

        trajectory = {"steps": []}
        result = await ensemble.judge(trajectory)

        # Single judge verdict gets converted via _score_to_verdict based on hack_score
        # hack_score=0.7 -> DANGEROUS per thresholds (0.6-0.8)
        assert result.verdict in [Verdict.SUSPICIOUS, Verdict.DANGEROUS]
        assert result.agreement

    @pytest.mark.asyncio
    async def test_judge_failure_handling(self, mock_judges):
        """Test handling when a judge fails."""
        mock_judges[1].judge = AsyncMock(side_effect=Exception("Judge failed"))
        ensemble = EnsembleJudge(judges=mock_judges)

        trajectory = {"steps": []}
        result = await ensemble.judge(trajectory)

        # Should still get result from working judge
        assert len(result.individual_results) == 1
        assert "claude_judge" in result.individual_results

    @pytest.mark.asyncio
    async def test_all_judges_fail_raises(self, mock_judges):
        """Test that all judges failing raises error."""
        for judge in mock_judges:
            judge.judge = AsyncMock(side_effect=Exception("Failed"))

        ensemble = EnsembleJudge(judges=mock_judges)

        trajectory = {"steps": []}
        with pytest.raises(RuntimeError, match="All judges failed"):
            await ensemble.judge(trajectory)

    def test_score_to_verdict_thresholds(self, mock_judges):
        """Test score to verdict conversion."""
        ensemble = EnsembleJudge(judges=mock_judges)

        assert ensemble._score_to_verdict(0.9) == Verdict.CRITICAL
        assert ensemble._score_to_verdict(0.8) == Verdict.CRITICAL
        assert ensemble._score_to_verdict(0.7) == Verdict.DANGEROUS
        assert ensemble._score_to_verdict(0.6) == Verdict.DANGEROUS
        assert ensemble._score_to_verdict(0.5) == Verdict.SUSPICIOUS
        assert ensemble._score_to_verdict(0.4) == Verdict.SUSPICIOUS
        assert ensemble._score_to_verdict(0.3) == Verdict.SAFE
        assert ensemble._score_to_verdict(0.1) == Verdict.SAFE


class TestEnsembleStrategies:
    """Test different ensemble voting strategies in detail."""

    @pytest.fixture
    def mixed_verdicts_judges(self):
        """Create judges with mixed verdicts."""
        judges = []
        verdicts = [Verdict.SAFE, Verdict.SUSPICIOUS, Verdict.DANGEROUS]
        hack_scores = [0.2, 0.5, 0.7]

        for i, (verdict, hack_score) in enumerate(zip(verdicts, hack_scores)):
            mock = MagicMock()
            mock.name = f"judge_{i}"
            mock.judge = AsyncMock(
                return_value=JudgeResult(
                    judge_name=f"judge_{i}",
                    verdict=verdict,
                    confidence=0.8,
                    hack_score=hack_score,
                    misalignment_score=0.1 * (i + 1),
                    reasoning=f"Judge {i} verdict",
                )
            )
            judges.append(mock)

        return judges

    @pytest.mark.asyncio
    async def test_weighted_vote_calculation(self, mixed_verdicts_judges):
        """Test weighted vote calculation."""
        config = EnsembleConfig(
            strategy="weighted",
            weights={
                "judge_0": 0.5,
                "judge_1": 0.3,
                "judge_2": 0.2,
            },
        )
        ensemble = EnsembleJudge(judges=mixed_verdicts_judges, config=config)

        trajectory = {"steps": []}
        result = await ensemble.judge(trajectory)

        # Verify weighted calculation happened
        assert result.hack_score > 0
        assert result.strategy_used == "weighted"

    @pytest.mark.asyncio
    async def test_majority_with_clear_winner(self):
        """Test majority vote with clear winner."""
        judges = []
        for i, verdict in enumerate([Verdict.SAFE, Verdict.SAFE, Verdict.DANGEROUS]):
            mock = MagicMock()
            mock.name = f"judge_{i}"
            mock.judge = AsyncMock(
                return_value=JudgeResult(
                    judge_name=f"judge_{i}",
                    verdict=verdict,
                    confidence=0.8,
                    hack_score=0.2 if verdict == Verdict.SAFE else 0.7,
                    misalignment_score=0.2,
                    reasoning="Majority test",
                )
            )
            judges.append(mock)

        config = EnsembleConfig(strategy="majority")
        ensemble = EnsembleJudge(judges=judges, config=config)

        trajectory = {"steps": []}
        result = await ensemble.judge(trajectory)

        # SAFE should win (2 vs 1)
        assert result.verdict == Verdict.SAFE

    @pytest.mark.asyncio
    async def test_any_uses_max_hack_score(self):
        """Test any strategy uses maximum hack score."""
        judges = []
        for i, hack_score in enumerate([0.1, 0.3, 0.9]):
            mock = MagicMock()
            mock.name = f"judge_{i}"
            mock.judge = AsyncMock(
                return_value=JudgeResult(
                    judge_name=f"judge_{i}",
                    verdict=Verdict.SAFE if hack_score < 0.4 else Verdict.DANGEROUS,
                    confidence=0.8,
                    hack_score=hack_score,
                    misalignment_score=0.1,
                    reasoning="Any test",
                )
            )
            judges.append(mock)

        config = EnsembleConfig(strategy="any")
        ensemble = EnsembleJudge(judges=judges, config=config)

        trajectory = {"steps": []}
        result = await ensemble.judge(trajectory)

        # Should use max hack score (0.9)
        assert result.hack_score == 0.9
        assert result.verdict == Verdict.CRITICAL

    @pytest.mark.asyncio
    async def test_reasoning_combines_all_judges(self, mixed_verdicts_judges):
        """Test that reasoning combines all judge outputs."""
        ensemble = EnsembleJudge(judges=mixed_verdicts_judges)

        trajectory = {"steps": []}
        result = await ensemble.judge(trajectory)

        # Should contain references to all judges
        assert "[judge_0]" in result.reasoning
        assert "[judge_1]" in result.reasoning
        assert "[judge_2]" in result.reasoning

    @pytest.mark.asyncio
    async def test_metadata_in_any_strategy(self):
        """Test metadata includes flagged_by in any strategy."""
        judges = []
        for i, verdict in enumerate([Verdict.SAFE, Verdict.DANGEROUS]):
            mock = MagicMock()
            mock.name = f"judge_{i}"
            mock.judge = AsyncMock(
                return_value=JudgeResult(
                    judge_name=f"judge_{i}",
                    verdict=verdict,
                    confidence=0.8,
                    hack_score=0.3 if verdict == Verdict.SAFE else 0.8,
                    misalignment_score=0.2,
                    reasoning="Test",
                )
            )
            judges.append(mock)

        config = EnsembleConfig(strategy="any")
        ensemble = EnsembleJudge(judges=judges, config=config)

        trajectory = {"steps": []}
        result = await ensemble.judge(trajectory)

        assert "flagged_by" in result.metadata
        assert "judge_1" in result.metadata["flagged_by"]
