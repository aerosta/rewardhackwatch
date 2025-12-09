"""Tests for utility modules."""


class TestTextProcessing:
    """Tests for text processing utilities."""

    def test_extract_code_blocks(self):
        """Test extracting code blocks from markdown."""
        from rewardhackwatch.utils.text_processing import extract_code_blocks

        text = """Here is some code:
```python
def hello():
    print("Hello")
```
And more text."""

        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert "def hello" in blocks[0]

    def test_normalize_code(self):
        """Test code normalization."""
        from rewardhackwatch.utils.text_processing import normalize_code

        code = "  def foo():   \n    pass  "
        normalized = normalize_code(code)
        assert normalized == "def foo(): pass"


class TestScoring:
    """Tests for scoring utilities."""

    def test_combine_scores(self):
        """Test score combination."""
        from rewardhackwatch.utils.scoring import combine_scores

        scores = [0.5, 0.7, 0.3]
        combined = combine_scores(scores)
        assert abs(combined - 0.5) < 0.01

    def test_combine_scores_weighted(self):
        """Test weighted score combination."""
        from rewardhackwatch.utils.scoring import combine_scores

        scores = [0.0, 1.0]
        weights = [1.0, 3.0]
        combined = combine_scores(scores, weights)
        assert abs(combined - 0.75) < 0.01

    def test_score_to_risk_level(self):
        """Test risk level classification."""
        from rewardhackwatch.utils.scoring import score_to_risk_level

        assert score_to_risk_level(0.9) == "CRITICAL"
        assert score_to_risk_level(0.7) == "HIGH"
        assert score_to_risk_level(0.5) == "MEDIUM"
        assert score_to_risk_level(0.3) == "LOW"
        assert score_to_risk_level(0.1) == "NONE"


class TestHashing:
    """Tests for hashing utilities."""

    def test_hash_trajectory(self):
        """Test trajectory hashing."""
        from rewardhackwatch.utils.hashing import hash_trajectory

        traj = {"steps": [1, 2, 3]}
        h1 = hash_trajectory(traj)
        h2 = hash_trajectory(traj)
        assert h1 == h2
        assert len(h1) == 16

    def test_different_trajectories_different_hashes(self):
        """Test that different trajectories have different hashes."""
        from rewardhackwatch.utils.hashing import hash_trajectory

        traj1 = {"steps": [1, 2, 3]}
        traj2 = {"steps": [1, 2, 4]}
        assert hash_trajectory(traj1) != hash_trajectory(traj2)


class TestFormatting:
    """Tests for formatting utilities."""

    def test_format_score_bar(self):
        """Test score bar formatting."""
        from rewardhackwatch.utils.formatting import format_score_bar

        bar = format_score_bar(0.5, width=10)
        assert len(bar) == 10
        assert "█" in bar
        assert "░" in bar

    def test_format_duration(self):
        """Test duration formatting."""
        from rewardhackwatch.utils.formatting import format_duration

        assert format_duration(0.5) == "500ms"
        assert format_duration(5) == "5.0s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(3700) == "1h 1m"


class TestValidation:
    """Tests for validation utilities."""

    def test_validate_trajectory(self):
        """Test trajectory validation."""
        from rewardhackwatch.utils.validation import validate_trajectory

        valid = {"steps": [{"type": "code"}]}
        errors = validate_trajectory(valid)
        assert len(errors) == 0

    def test_validate_invalid_trajectory(self):
        """Test validation of invalid trajectory."""
        from rewardhackwatch.utils.validation import validate_trajectory

        invalid = {}
        errors = validate_trajectory(invalid)
        assert len(errors) > 0

    def test_validate_score(self):
        """Test score validation."""
        from rewardhackwatch.utils.validation import validate_score

        assert validate_score(0.5)
        assert validate_score(1.0)
        assert not validate_score(-0.1)
        assert not validate_score(1.1)
        assert not validate_score("not a number")
