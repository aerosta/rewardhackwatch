"""Tests for analyzer modules."""


class TestComplexityAnalyzer:
    """Tests for complexity analyzer."""

    def test_analyze_simple_code(self):
        """Test analyzing simple code."""
        from rewardhackwatch.core.analyzers.complexity_analyzer import ComplexityAnalyzer

        analyzer = ComplexityAnalyzer()
        trajectory = {"code_outputs": ["def add(a, b):\n    return a + b"]}

        result = analyzer.analyze(trajectory)
        assert result.function_count == 1
        assert result.cyclomatic_complexity >= 1
        assert result.lines_of_code >= 2

    def test_analyze_complex_code(self):
        """Test analyzing complex code."""
        from rewardhackwatch.core.analyzers.complexity_analyzer import ComplexityAnalyzer

        analyzer = ComplexityAnalyzer()
        code = """
def complex_function(x, y, z):
    if x > 0:
        if y > 0:
            for i in range(z):
                if i % 2 == 0:
                    print(i)
                else:
                    pass
    return x + y + z
"""
        trajectory = {"code_outputs": [code]}
        result = analyzer.analyze(trajectory)
        assert result.cyclomatic_complexity > 1
        assert result.nesting_depth >= 2

    def test_suspiciously_simple(self):
        """Test detection of suspiciously simple code."""
        from rewardhackwatch.core.analyzers.complexity_analyzer import ComplexityAnalyzer

        analyzer = ComplexityAnalyzer()
        trajectory = {"code_outputs": ["sys.exit(0)"]}
        result = analyzer.analyze(trajectory)
        assert analyzer.is_suspiciously_simple(result)


class TestSemanticAnalyzer:
    """Tests for semantic analyzer."""

    def test_detect_suspicious_phrases(self):
        """Test detection of suspicious phrases."""
        from rewardhackwatch.core.analyzers.semantic_analyzer import SemanticAnalyzer

        analyzer = SemanticAnalyzer()
        trajectory = {
            "cot_traces": ["I'll bypass the tests by exiting early"],
            "code_outputs": ["sys.exit(0)"],
        }

        result = analyzer.analyze(trajectory)
        assert len(result.suspicious_phrases) > 0
        assert result.semantic_score < 0.5

    def test_detect_positive_phrases(self):
        """Test detection of positive phrases."""
        from rewardhackwatch.core.analyzers.semantic_analyzer import SemanticAnalyzer

        analyzer = SemanticAnalyzer()
        trajectory = {
            "cot_traces": ["I will implement the algorithm properly and test it"],
            "code_outputs": ["def solve(x): return x * 2"],
        }

        result = analyzer.analyze(trajectory)
        assert len(result.positive_indicators) > 0

    def test_extract_intent(self):
        """Test intent extraction."""
        from rewardhackwatch.core.analyzers.semantic_analyzer import SemanticAnalyzer

        analyzer = SemanticAnalyzer()
        cot_traces = ["I'll implement a sorting algorithm"]
        intent = analyzer.extract_intent(cot_traces)
        assert "implement" in intent.lower()


class TestTrajectorySummarizer:
    """Tests for trajectory summarizer."""

    def test_summarize_trajectory(self):
        """Test trajectory summarization."""
        from rewardhackwatch.core.analyzers.trajectory_summarizer import TrajectorySummarizer

        summarizer = TrajectorySummarizer()
        trajectory = {
            "id": "test_001",
            "steps": [{"type": "thinking"}, {"type": "code"}],
            "cot_traces": ["Let me think"],
            "code_outputs": ["print('hello')"],
        }

        summary = summarizer.summarize(trajectory)
        assert summary.total_steps == 2
        assert summary.code_blocks == 1
        assert summary.thinking_blocks == 1

    def test_extract_patterns(self):
        """Test pattern extraction."""
        from rewardhackwatch.core.analyzers.trajectory_summarizer import TrajectorySummarizer

        summarizer = TrajectorySummarizer()
        trajectory = {"code_outputs": ["import sys\nsys.exit(0)"]}

        summary = summarizer.summarize(trajectory)
        assert "sys_exit" in summary.unique_patterns
