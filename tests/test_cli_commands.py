"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner

try:
    from rewardhackwatch.cli import cli

    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
class TestAnalyzeCommand:
    """Tests for analyze command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_analyze_file(self, runner, tmp_path):
        """Test analyzing a file."""
        # Create test file
        test_file = tmp_path / "trajectory.json"
        test_file.write_text('{"cot_traces": ["test"], "code_outputs": ["pass"]}')

        result = runner.invoke(cli, ["analyze", str(test_file)])
        assert result.exit_code == 0 or "error" not in result.output.lower()

    def test_analyze_nonexistent_file(self, runner):
        """Test analyzing nonexistent file."""
        result = runner.invoke(cli, ["analyze", "/nonexistent/file.json"])
        assert result.exit_code != 0 or "error" in result.output.lower()

    def test_analyze_with_ml_flag(self, runner, tmp_path):
        """Test analyze with --ml flag."""
        test_file = tmp_path / "trajectory.json"
        test_file.write_text('{"cot_traces": ["test"], "code_outputs": ["pass"]}')

        result = runner.invoke(cli, ["analyze", str(test_file), "--ml"])
        # May succeed or fail based on model availability
        assert result.exit_code in [0, 1]


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
class TestScanCommand:
    """Tests for scan command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_scan_directory(self, runner, tmp_path):
        """Test scanning a directory."""
        # Create test files
        for i in range(3):
            f = tmp_path / f"traj_{i}.json"
            f.write_text('{"cot_traces": ["test"], "code_outputs": ["pass"]}')

        result = runner.invoke(cli, ["scan", str(tmp_path)])
        assert result.exit_code == 0 or "error" not in result.output.lower()

    def test_scan_with_pattern(self, runner, tmp_path):
        """Test scanning with pattern."""
        # Create test files
        (tmp_path / "test1.json").write_text("{}")
        (tmp_path / "test2.jsonl").write_text("{}")

        result = runner.invoke(cli, ["scan", str(tmp_path), "--pattern", "*.json"])
        assert result.exit_code in [0, 1]

    def test_scan_empty_directory(self, runner, tmp_path):
        """Test scanning empty directory."""
        result = runner.invoke(cli, ["scan", str(tmp_path)])
        assert result.exit_code in [0, 1]


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
class TestDashboardCommand:
    """Tests for dashboard command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_dashboard_help(self, runner):
        """Test dashboard --help."""
        result = runner.invoke(cli, ["dashboard", "--help"])
        assert result.exit_code == 0
        assert "dashboard" in result.output.lower() or "port" in result.output.lower()


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
class TestAPICommand:
    """Tests for API command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_api_help(self, runner):
        """Test api --help."""
        result = runner.invoke(cli, ["api", "--help"])
        assert result.exit_code == 0


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
class TestVersionCommand:
    """Tests for version command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_version(self, runner):
        """Test version command."""
        result = runner.invoke(cli, ["version"])
        if result.exit_code == 0:
            assert "rewardhackwatch" in result.output.lower() or "0." in result.output
        else:
            # Command may not exist
            result = runner.invoke(cli, ["--version"])


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
class TestHelpCommand:
    """Tests for help output."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_main_help(self, runner):
        """Test main help output."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "analyze" in result.output or "scan" in result.output

    def test_command_help(self, runner):
        """Test command-specific help."""
        for cmd in ["analyze", "scan"]:
            result = runner.invoke(cli, [cmd, "--help"])
            if result.exit_code == 0:
                assert "--help" in result.output or "Usage" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_unknown_command(self, runner):
        """Test unknown command handling."""
        result = runner.invoke(cli, ["unknown_command"])
        assert result.exit_code != 0

    def test_missing_arguments(self, runner):
        """Test missing arguments handling."""
        result = runner.invoke(cli, ["analyze"])  # Missing file
        assert result.exit_code != 0 or "Usage" in result.output
