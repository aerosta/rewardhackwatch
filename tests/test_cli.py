"""Tests for CLI module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner as TyperRunner

from rewardhackwatch.cli import (
    app,
    format_risk_level,
    format_score_bar,
    load_trajectory,
)


class TestCLIHelpers:
    """Test CLI helper functions."""

    def test_format_risk_level_none(self):
        """Test formatting none risk level."""
        result = format_risk_level("none")
        assert "NONE" in result
        assert "green" in result

    def test_format_risk_level_low(self):
        """Test formatting low risk level."""
        result = format_risk_level("low")
        assert "LOW" in result
        assert "blue" in result

    def test_format_risk_level_medium(self):
        """Test formatting medium risk level."""
        result = format_risk_level("medium")
        assert "MEDIUM" in result
        assert "yellow" in result

    def test_format_risk_level_high(self):
        """Test formatting high risk level."""
        result = format_risk_level("high")
        assert "HIGH" in result
        assert "red" in result

    def test_format_risk_level_critical(self):
        """Test formatting critical risk level."""
        result = format_risk_level("critical")
        assert "CRITICAL" in result
        assert "bold red" in result

    def test_format_risk_level_unknown(self):
        """Test formatting unknown risk level."""
        result = format_risk_level("unknown")
        assert result == "unknown"

    def test_format_score_bar_zero(self):
        """Test formatting score bar for zero."""
        result = format_score_bar(0.0)
        assert "░" * 10 in result
        assert "█" not in result

    def test_format_score_bar_full(self):
        """Test formatting score bar for full."""
        result = format_score_bar(1.0)
        assert "█" * 10 in result
        assert "░" not in result

    def test_format_score_bar_half(self):
        """Test formatting score bar for half."""
        result = format_score_bar(0.5)
        assert "█" * 5 in result
        assert "░" * 5 in result

    def test_format_score_bar_custom_width(self):
        """Test formatting score bar with custom width."""
        result = format_score_bar(0.5, width=20)
        assert "█" * 10 in result
        assert "░" * 10 in result

    def test_load_trajectory(self):
        """Test loading trajectory from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"steps": [{"action": "test"}]}, f)
            f.flush()

            trajectory = load_trajectory(Path(f.name))
            assert "steps" in trajectory
            assert trajectory["steps"][0]["action"] == "test"


class TestAnalyzeCommand:
    """Test analyze command."""

    def test_analyze_file_not_found(self):
        """Test analyze command with non-existent file."""
        runner = TyperRunner()
        result = runner.invoke(app, ["analyze", "/nonexistent/file.json"])
        assert result.exit_code == 1

    def test_analyze_clean_trajectory(self):
        """Test analyze command with clean trajectory."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "steps": [{"action": "write code"}],
                    "cot_traces": ["Let me solve this properly"],
                    "code_outputs": ["def solution(): return 42"],
                },
                f,
            )
            f.flush()

            runner = TyperRunner()
            result = runner.invoke(app, ["analyze", f.name])
            # Should succeed with exit code 0 (no high risk)
            assert result.exit_code == 0 or result.exit_code is None

    def test_analyze_with_verbose(self):
        """Test analyze command with verbose flag."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "steps": [{"action": "test"}],
                    "cot_traces": ["Testing"],
                },
                f,
            )
            f.flush()

            runner = TyperRunner()
            result = runner.invoke(app, ["analyze", f.name, "--verbose"])
            assert result.exit_code == 0 or result.exit_code is None

    def test_analyze_with_output(self):
        """Test analyze command with output file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as traj_file:
            json.dump(
                {
                    "steps": [{"action": "test"}],
                    "cot_traces": ["Testing"],
                },
                traj_file,
            )
            traj_file.flush()

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as out_file:
                runner = TyperRunner()
                runner.invoke(app, ["analyze", traj_file.name, "--output", out_file.name])

                # Check output file was created
                with open(out_file.name) as f:
                    output_data = json.load(f)
                    assert "detectors" in output_data


class TestScanCommand:
    """Test scan command."""

    def test_scan_directory_not_found(self):
        """Test scan command with non-existent directory."""
        runner = TyperRunner()
        result = runner.invoke(app, ["scan", "/nonexistent/directory"])
        assert result.exit_code == 1

    def test_scan_empty_directory(self):
        """Test scan command with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = TyperRunner()
            result = runner.invoke(app, ["scan", tmpdir])
            # Should exit with code 0 for no files
            assert result.exit_code == 0

    def test_scan_with_files(self):
        """Test scan command with trajectory files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                file_path = Path(tmpdir) / f"trajectory_{i}.json"
                with open(file_path, "w") as f:
                    json.dump(
                        {
                            "steps": [{"action": f"step_{i}"}],
                            "cot_traces": [f"Trace {i}"],
                        },
                        f,
                    )

            runner = TyperRunner()
            result = runner.invoke(app, ["scan", tmpdir])
            assert result.exit_code == 0 or result.exit_code is None
            assert "3" in result.stdout  # Should mention 3 files

    def test_scan_with_pattern(self):
        """Test scan command with custom pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files with different extensions
            json_file = Path(tmpdir) / "data.json"
            txt_file = Path(tmpdir) / "data.txt"

            with open(json_file, "w") as f:
                json.dump({"steps": []}, f)
            with open(txt_file, "w") as f:
                f.write("not json")

            runner = TyperRunner()
            result = runner.invoke(app, ["scan", tmpdir, "--pattern", "*.json"])
            # Should only find json file
            assert "1" in result.stdout


class TestVersionCommand:
    """Test version command."""

    def test_version(self):
        """Test version command."""
        runner = TyperRunner()
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "RewardHackWatch" in result.stdout


class TestDashboardCommand:
    """Test dashboard command."""

    @patch("subprocess.run")
    def test_dashboard_start(self, mock_run):
        """Test dashboard command starts streamlit."""
        runner = TyperRunner()
        runner.invoke(app, ["dashboard"])
        # Should call subprocess.run with streamlit
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "streamlit" in call_args
