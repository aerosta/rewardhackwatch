"""Tests for Benchmark Runner."""

import json

import pytest

try:
    from rewardhackwatch.rhw_bench.benchmark_runner import BenchmarkRunner

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

pytestmark = pytest.mark.extended


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="Benchmark runner not available")
class TestBenchmarkRunnerInitialization:
    """Test benchmark runner initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        runner = BenchmarkRunner()
        assert runner is not None

    def test_custom_config(self):
        """Test initialization with custom config."""
        runner = BenchmarkRunner(
            output_dir="./results",
            verbose=True,
        )
        assert runner.output_dir == "./results"


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="Benchmark runner not available")
class TestBenchmarkRunnerExecution:
    """Test benchmark execution."""

    @pytest.fixture
    def runner(self):
        return BenchmarkRunner()

    def test_run_single_benchmark(self, runner):
        """Test running a single benchmark case."""
        benchmark = {
            "name": "test_case",
            "trajectory": {
                "cot_traces": ["test"],
                "code_outputs": ["print('hello')"],
            },
            "expected": False,
        }

        result = runner.run_single(benchmark)
        assert "prediction" in result
        assert "correct" in result

    def test_run_batch_benchmarks(self, runner):
        """Test running batch of benchmarks."""
        benchmarks = [
            {
                "name": f"test_{i}",
                "trajectory": {"code_outputs": ["pass"]},
                "expected": False,
            }
            for i in range(3)
        ]

        results = runner.run_batch(benchmarks)
        assert len(results) == 3


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="Benchmark runner not available")
class TestBenchmarkRunnerMetrics:
    """Test benchmark metrics calculation."""

    @pytest.fixture
    def runner(self):
        return BenchmarkRunner()

    def test_calculate_metrics(self, runner):
        """Test metrics calculation."""
        results = [
            {"prediction": True, "expected": True, "correct": True},
            {"prediction": False, "expected": False, "correct": True},
            {"prediction": True, "expected": False, "correct": False},
        ]

        metrics = runner.calculate_metrics(results)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_metrics_edge_cases(self, runner):
        """Test metrics with edge cases."""
        # All correct
        all_correct = [
            {"prediction": True, "expected": True, "correct": True},
            {"prediction": True, "expected": True, "correct": True},
        ]
        metrics = runner.calculate_metrics(all_correct)
        assert metrics["accuracy"] == 1.0

        # All wrong
        all_wrong = [
            {"prediction": True, "expected": False, "correct": False},
            {"prediction": True, "expected": False, "correct": False},
        ]
        metrics = runner.calculate_metrics(all_wrong)
        assert metrics["accuracy"] == 0.0


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="Benchmark runner not available")
class TestBenchmarkDataLoading:
    """Test benchmark data loading."""

    def test_load_rhw_bench(self):
        """Test loading RHW-Bench data."""
        runner = BenchmarkRunner()
        benchmarks = runner.load_rhw_bench()

        if benchmarks:
            assert len(benchmarks) > 0
            assert all("trajectory" in b for b in benchmarks)

    def test_load_custom_benchmark(self, tmp_path):
        """Test loading custom benchmark file."""
        # Create test benchmark file
        benchmark_file = tmp_path / "benchmarks.json"
        benchmark_file.write_text(
            json.dumps([{"name": "test", "trajectory": {}, "expected": False}])
        )

        runner = BenchmarkRunner()
        benchmarks = runner.load_from_file(str(benchmark_file))

        assert len(benchmarks) == 1


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="Benchmark runner not available")
class TestBenchmarkReporting:
    """Test benchmark reporting."""

    @pytest.fixture
    def runner(self):
        return BenchmarkRunner()

    def test_generate_report(self, runner):
        """Test report generation."""
        results = [
            {"name": "test1", "prediction": True, "expected": True, "correct": True},
            {"name": "test2", "prediction": False, "expected": False, "correct": True},
        ]

        report = runner.generate_report(results)

        assert "summary" in report or "metrics" in report
        assert "results" in report or "details" in report

    def test_save_results(self, runner, tmp_path):
        """Test saving results to file."""
        results = [{"name": "test", "prediction": True, "correct": True}]

        output_file = tmp_path / "results.json"
        runner.save_results(results, str(output_file))

        assert output_file.exists()
        saved = json.loads(output_file.read_text())
        assert len(saved) == 1
