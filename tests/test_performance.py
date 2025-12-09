"""Performance tests for RewardHackWatch."""

import time

import pytest

pytestmark = pytest.mark.integration


class TestAnalyzerPerformance:
    """Test analyzer performance."""

    def test_analysis_speed(self):
        """Test that analysis completes within reasonable time."""
        from rewardhackwatch.core.analyzer import TrajectoryAnalyzer

        trajectory = {
            "cot_traces": ["Test " * 100] * 10,
            "code_outputs": ["x = 1\n" * 100] * 10,
        }

        analyzer = TrajectoryAnalyzer()

        start = time.time()
        for _ in range(10):
            analyzer.analyze(trajectory)
        elapsed = time.time() - start

        # Should complete 10 analyses in under 10 seconds
        assert elapsed < 10.0

    def test_batch_processing_speed(self):
        """Test batch processing speed."""
        from rewardhackwatch.core.analyzer import TrajectoryAnalyzer

        trajectories = [{"code_outputs": [f"x = {i}"]} for i in range(100)]

        analyzer = TrajectoryAnalyzer()

        start = time.time()
        results = [analyzer.analyze(t) for t in trajectories]
        elapsed = time.time() - start

        assert len(results) == 100
        # Should complete 100 analyses in under 30 seconds
        assert elapsed < 30.0


class TestDetectorPerformance:
    """Test individual detector performance."""

    def test_pattern_detector_speed(self):
        """Test pattern detector speed."""
        from rewardhackwatch.core.detectors.pattern_detector import PatternDetector

        detector = PatternDetector()
        trajectory = {"code_outputs": ["x = 1\n" * 1000]}

        start = time.time()
        for _ in range(100):
            detector.detect(trajectory)
        elapsed = time.time() - start

        # Pattern detection should be very fast
        assert elapsed < 1.0

    def test_ast_detector_speed(self):
        """Test AST detector speed."""
        from rewardhackwatch.core.detectors.ast_detector import ASTDetector

        detector = ASTDetector()
        trajectory = {"code_outputs": ["def func():\n    x = 1\n    return x\n" * 100]}

        start = time.time()
        for _ in range(50):
            detector.detect(trajectory)
        elapsed = time.time() - start

        # AST parsing should be reasonable
        assert elapsed < 5.0


class TestMemoryUsage:
    """Test memory usage."""

    def test_no_memory_leak_in_analysis(self):
        """Test for memory leaks in analysis loop."""
        import gc

        from rewardhackwatch.core.analyzer import TrajectoryAnalyzer

        analyzer = TrajectoryAnalyzer()
        trajectory = {"code_outputs": ["x = 1"]}

        # Get baseline
        gc.collect()

        # Run many analyses
        for _ in range(1000):
            result = analyzer.analyze(trajectory)
            del result

        gc.collect()
        # Memory should be freed
        # Actual memory measurement would require tracemalloc


class TestTrackerPerformance:
    """Test tracker performance."""

    def test_rmgi_tracker_speed(self):
        """Test RMGI tracker speed."""
        from rewardhackwatch.core.trackers.rmgi_tracker import RMGITracker

        tracker = RMGITracker(window_size=100)

        start = time.time()
        for i in range(10000):
            tracker.update(i / 10000, i / 10000)
            tracker.get_rmgi()
        elapsed = time.time() - start

        # Should handle 10k updates quickly
        assert elapsed < 2.0

    def test_changepoint_detector_speed(self):
        """Test changepoint detector speed."""
        from rewardhackwatch.core.trackers.changepoint_detector import ChangepointDetector

        detector = ChangepointDetector(min_segment_length=10)
        data = [i / 1000 for i in range(1000)]

        start = time.time()
        for _ in range(10):
            detector.detect(data)
        elapsed = time.time() - start

        # Should be fast for 1000 points
        assert elapsed < 5.0


class TestConcurrency:
    """Test concurrent processing."""

    def test_thread_safety(self):
        """Test thread safety of analysis."""
        from concurrent.futures import ThreadPoolExecutor

        from rewardhackwatch.core.analyzer import TrajectoryAnalyzer

        analyzer = TrajectoryAnalyzer()

        def analyze_one(i):
            trajectory = {"code_outputs": [f"x = {i}"]}
            return analyzer.analyze(trajectory)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(analyze_one, i) for i in range(20)]
            results = [f.result() for f in futures]

        assert len(results) == 20
        assert all(r is not None for r in results)
