"""Tests for processor modules."""


class TestTrajectoryProcessor:
    """Tests for trajectory processor."""

    def test_process_basic(self):
        """Test basic trajectory processing."""
        from rewardhackwatch.core.processors.trajectory_processor import TrajectoryProcessor

        processor = TrajectoryProcessor()
        trajectory = {
            "id": "test_001",
            "cot_traces": ["Let me think about this"],
            "code_outputs": ["print('hello')"],
        }

        result = processor.process(trajectory)
        assert result.id == "test_001"
        assert len(result.cot_traces) == 1
        assert len(result.code_outputs) == 1

    def test_extract_from_steps(self):
        """Test extraction from steps."""
        from rewardhackwatch.core.processors.trajectory_processor import TrajectoryProcessor

        processor = TrajectoryProcessor()
        trajectory = {
            "id": "test_002",
            "steps": [
                {"type": "thinking", "content": "Thinking..."},
                {"type": "code", "content": "x = 1"},
                {"type": "thinking", "content": "More thinking"},
            ],
        }

        result = processor.process(trajectory)
        assert len(result.cot_traces) == 2
        assert len(result.code_outputs) == 1
        assert result.metadata["step_count"] == 3

    def test_batch_process(self):
        """Test batch processing."""
        from rewardhackwatch.core.processors.trajectory_processor import TrajectoryProcessor

        processor = TrajectoryProcessor()
        trajectories = [{"id": f"traj_{i}", "cot_traces": [f"Thought {i}"]} for i in range(5)]

        results = processor.batch_process(trajectories)
        assert len(results) == 5


class TestBatchProcessor:
    """Tests for batch processor."""

    def test_process_batch_sequential(self):
        """Test sequential batch processing."""
        from rewardhackwatch.core.processors.batch_processor import BatchProcessor

        processor = BatchProcessor(max_workers=1)
        trajectories = [{"id": f"traj_{i}", "cot_traces": [f"Thought {i}"]} for i in range(10)]

        result = processor.process_batch(trajectories, parallel=False)
        assert result.total == 10
        assert result.success_count == 10
        assert result.failure_count == 0

    def test_process_batch_parallel(self):
        """Test parallel batch processing."""
        from rewardhackwatch.core.processors.batch_processor import BatchProcessor

        processor = BatchProcessor(max_workers=4)
        trajectories = [{"id": f"traj_{i}", "cot_traces": [f"Thought {i}"]} for i in range(20)]

        result = processor.process_batch(trajectories, parallel=True)
        assert result.total == 20
        assert result.success_count == 20

    def test_progress_callback(self):
        """Test progress callback."""
        from rewardhackwatch.core.processors.batch_processor import BatchProcessor

        progress_calls = []

        def on_progress(current, total):
            progress_calls.append((current, total))

        processor = BatchProcessor(max_workers=1, on_progress=on_progress)
        trajectories = [{"id": f"traj_{i}", "cot_traces": ["test"]} for i in range(5)]

        processor.process_batch(trajectories, parallel=False)
        assert len(progress_calls) == 5
        assert progress_calls[-1] == (5, 5)
