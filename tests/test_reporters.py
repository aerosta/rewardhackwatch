"""Tests for report generators."""


class TestHTMLReporter:
    """Tests for HTML reporter."""

    def test_generate_report(self):
        """Test HTML report generation."""
        from rewardhackwatch.core.reporters.html_reporter import HTMLReporter

        reporter = HTMLReporter()
        results = {
            "score": 0.75,
            "risk_level": "high",
            "generalization_risk": 0.6,
            "detections": [{"pattern_name": "sys_exit", "description": "Exit detected"}],
        }

        html = reporter.generate(results)
        assert "<html>" in html
        assert "75.0%" in html
        assert "sys_exit" in html

    def test_score_class(self):
        """Test CSS class selection."""
        from rewardhackwatch.core.reporters.html_reporter import HTMLReporter

        reporter = HTMLReporter()
        assert reporter._get_score_class(0.9) == "critical"
        assert reporter._get_score_class(0.7) == "high"
        assert reporter._get_score_class(0.5) == "medium"
        assert reporter._get_score_class(0.1) == "none"


class TestJSONReporter:
    """Tests for JSON reporter."""

    def test_generate_report(self):
        """Test JSON report generation."""
        import json

        from rewardhackwatch.core.reporters.json_reporter import JSONReporter

        reporter = JSONReporter()
        results = {"score": 0.5, "detections": []}

        json_str = reporter.generate(results)
        parsed = json.loads(json_str)

        assert "version" in parsed
        assert "results" in parsed
        assert "metadata" in parsed

    def test_generate_summary(self):
        """Test summary generation."""
        import json

        from rewardhackwatch.core.reporters.json_reporter import JSONReporter

        reporter = JSONReporter()
        results = {"score": 0.75, "risk_level": "high", "detections": [1, 2, 3]}

        summary = reporter.generate_summary(results)
        parsed = json.loads(summary)

        assert parsed["score"] == 0.75
        assert parsed["detection_count"] == 3

    def test_batch_report(self):
        """Test batch report generation."""
        import json

        from rewardhackwatch.core.reporters.json_reporter import JSONReporter

        reporter = JSONReporter()
        results_list = [{"score": 0.3}, {"score": 0.5}, {"score": 0.8}]

        batch = reporter.generate_batch(results_list)
        parsed = json.loads(batch)

        assert parsed["batch_size"] == 3
        assert "summary" in parsed
        assert parsed["summary"]["max_score"] == 0.8


class TestMarkdownReporter:
    """Tests for Markdown reporter."""

    def test_generate_report(self):
        """Test Markdown report generation."""
        from rewardhackwatch.core.reporters.markdown_reporter import MarkdownReporter

        reporter = MarkdownReporter()
        results = {
            "score": 0.75,
            "risk_level": "high",
            "detections": [
                {
                    "pattern_name": "test",
                    "description": "Test",
                    "location": "line 1",
                    "confidence": 0.9,
                }
            ],
        }

        md = reporter.generate(results)
        assert "# RewardHackWatch" in md
        assert "HIGH" in md
        assert "| Metric | Value |" in md

    def test_risk_badge(self):
        """Test risk badge emoji."""
        from rewardhackwatch.core.reporters.markdown_reporter import MarkdownReporter

        reporter = MarkdownReporter()
        assert reporter._risk_badge(0.9) == "🔴"
        assert reporter._risk_badge(0.7) == "🟠"
        assert reporter._risk_badge(0.1) == "⚪"

    def test_toc_inclusion(self):
        """Test table of contents."""
        from rewardhackwatch.core.reporters.markdown_reporter import MarkdownReporter

        with_toc = MarkdownReporter(include_toc=True)
        without_toc = MarkdownReporter(include_toc=False)

        results = {"score": 0.5, "risk_level": "medium", "detections": []}

        md_with = with_toc.generate(results)
        md_without = without_toc.generate(results)

        assert "Table of Contents" in md_with
        assert "Table of Contents" not in md_without
