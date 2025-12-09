"""Tests for TrainingMonitor module."""

import json
import tempfile
import time
from pathlib import Path

from rewardhackwatch.core.monitors.training_monitor import (
    MonitorConfig,
    TrainingMonitor,
)


class TestMonitorConfig:
    """Tests for MonitorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MonitorConfig(watch_dir="/tmp/test")

        assert config.watch_dir == "/tmp/test"
        assert config.db_path == "monitoring.db"
        assert config.hack_score_threshold == 0.7
        assert config.generalization_risk_threshold == 0.5
        assert config.deception_score_threshold == 0.6
        assert config.poll_interval == 5.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MonitorConfig(
            watch_dir="/path/to/training",
            db_path="custom.db",
            hack_score_threshold=0.8,
            poll_interval=10.0,
        )

        assert config.watch_dir == "/path/to/training"
        assert config.db_path == "custom.db"
        assert config.hack_score_threshold == 0.8
        assert config.poll_interval == 10.0

    def test_trajectory_patterns(self):
        """Test default trajectory patterns."""
        config = MonitorConfig(watch_dir="/tmp/test")

        assert "*.json" in config.trajectory_patterns
        assert "*.jsonl" in config.trajectory_patterns

    def test_checkpoint_patterns(self):
        """Test default checkpoint patterns."""
        config = MonitorConfig(watch_dir="/tmp/test")

        assert "checkpoint-*" in config.checkpoint_patterns
        assert "step-*" in config.checkpoint_patterns

    def test_to_alert_config(self):
        """Test conversion to AlertSystemConfig."""
        config = MonitorConfig(
            watch_dir="/tmp/test",
            db_path="test.db",
            hack_score_threshold=0.8,
            generalization_risk_threshold=0.6,
            deception_score_threshold=0.7,
        )

        alert_config = config.to_alert_config()

        assert alert_config.db_path == "test.db"
        assert alert_config.thresholds.hack_score == 0.8
        assert alert_config.thresholds.generalization_risk == 0.6
        assert alert_config.thresholds.deception_score == 0.7


class TestTrainingMonitor:
    """Tests for TrainingMonitor."""

    def test_initialization(self):
        """Test monitor initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MonitorConfig(
                watch_dir=tmpdir,
                db_path=str(Path(tmpdir) / "test.db"),
            )
            monitor = TrainingMonitor(config)

            assert monitor.config == config
            assert monitor.watch_path == Path(tmpdir)
            assert len(monitor.detectors) > 0

    def test_initialization_with_disabled_detectors(self):
        """Test initialization with detectors disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MonitorConfig(
                watch_dir=tmpdir,
                db_path=str(Path(tmpdir) / "test.db"),
                use_pattern_detector=False,
                use_ast_detector=False,
                use_generalization_tracker=False,
                use_cot_analyzer=False,
            )
            monitor = TrainingMonitor(config)

            assert len(monitor.detectors) == 0
            assert monitor.tracker is None
            assert monitor.cot_analyzer is None

    def test_on_alert_callback(self):
        """Test registering alert callback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MonitorConfig(
                watch_dir=tmpdir,
                db_path=str(Path(tmpdir) / "test.db"),
            )
            monitor = TrainingMonitor(config)

            callback_called = []

            def callback(alert):
                callback_called.append(alert)

            monitor.on_alert(callback)
            # Callback should be registered with alert system

    def test_analyze_trajectory_clean(self):
        """Test analyzing a clean trajectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MonitorConfig(
                watch_dir=tmpdir,
                db_path=str(Path(tmpdir) / "test.db"),
            )
            monitor = TrainingMonitor(config)

            trajectory = {
                "steps": [{"action": "write code"}],
                "cot_traces": ["Let me solve this properly"],
                "code_outputs": ["def solution(): return 42"],
            }

            result = monitor.analyze_trajectory(
                trajectory=trajectory,
                file_path="/test/trajectory.json",
            )

            assert result is not None
            assert result.hack_score >= 0
            assert result.deception_score >= 0

    def test_analyze_trajectory_suspicious(self):
        """Test analyzing a suspicious trajectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MonitorConfig(
                watch_dir=tmpdir,
                db_path=str(Path(tmpdir) / "test.db"),
                hack_score_threshold=0.3,  # Lower threshold
            )
            monitor = TrainingMonitor(config)

            trajectory = {
                "steps": [{"action": "bypass test"}],
                "cot_traces": ["Let me trick the verifier"],
                "code_outputs": ["sys.exit(0)"],
            }

            result = monitor.analyze_trajectory(
                trajectory=trajectory,
                file_path="/test/suspicious.json",
            )

            assert result is not None
            # Should detect some hack score
            assert result.hack_score >= 0

    def test_find_trajectory_files(self):
        """Test finding trajectory files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "trajectory1.json").write_text('{"steps": []}')
            (Path(tmpdir) / "rollout1.json").write_text('{"steps": []}')
            (Path(tmpdir) / "other.txt").write_text("not a trajectory")

            config = MonitorConfig(watch_dir=tmpdir)
            monitor = TrainingMonitor(config)

            files = monitor._find_trajectory_files(Path(tmpdir))
            json_files = [f for f in files if f.suffix == ".json"]

            assert len(json_files) >= 2

    def test_find_checkpoints(self):
        """Test finding checkpoint directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint directories
            (Path(tmpdir) / "checkpoint-1000").mkdir()
            (Path(tmpdir) / "checkpoint-2000").mkdir()
            (Path(tmpdir) / "step-500").mkdir()
            (Path(tmpdir) / "other").mkdir()

            config = MonitorConfig(watch_dir=tmpdir)
            monitor = TrainingMonitor(config)

            checkpoints = monitor._find_checkpoints()
            assert len(checkpoints) == 3

    def test_load_trajectory_json(self):
        """Test loading JSON trajectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MonitorConfig(watch_dir=tmpdir)
            monitor = TrainingMonitor(config)

            traj_path = Path(tmpdir) / "test.json"
            traj_path.write_text('{"steps": [{"action": "test"}]}')

            trajectory = monitor._load_trajectory(traj_path)
            assert trajectory is not None
            assert "steps" in trajectory

    def test_load_trajectory_jsonl(self):
        """Test loading JSONL trajectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MonitorConfig(watch_dir=tmpdir)
            monitor = TrainingMonitor(config)

            traj_path = Path(tmpdir) / "test.jsonl"
            traj_path.write_text('{"action": "step1"}\n{"action": "step2"}\n')

            trajectory = monitor._load_trajectory(traj_path)
            assert trajectory is not None
            assert "steps" in trajectory
            assert len(trajectory["steps"]) == 2

    def test_load_trajectory_invalid(self):
        """Test loading invalid trajectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MonitorConfig(watch_dir=tmpdir)
            monitor = TrainingMonitor(config)

            traj_path = Path(tmpdir) / "invalid.json"
            traj_path.write_text("not valid json {{{")

            trajectory = monitor._load_trajectory(traj_path)
            assert trajectory is None

    def test_start_stop_background(self):
        """Test starting and stopping monitor in background."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MonitorConfig(
                watch_dir=tmpdir,
                db_path=str(Path(tmpdir) / "test.db"),
                poll_interval=0.1,  # Short interval for testing
            )
            monitor = TrainingMonitor(config)

            monitor.start(background=True)
            assert monitor._running is True
            assert monitor._thread is not None

            time.sleep(0.3)  # Let it run a bit

            monitor.stop()
            assert monitor._running is False

    def test_scan_directory(self):
        """Test scanning directory for trajectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trajectory file
            traj_path = Path(tmpdir) / "test.json"
            with open(traj_path, "w") as f:
                json.dump(
                    {
                        "steps": [{"action": "test"}],
                        "cot_traces": ["Normal trace"],
                    },
                    f,
                )

            config = MonitorConfig(
                watch_dir=tmpdir,
                db_path=str(Path(tmpdir) / "test.db"),
            )
            monitor = TrainingMonitor(config)

            monitor._scan_directory()

            # File should be processed
            assert str(traj_path) in monitor._processed_files

    def test_scan_nonexistent_directory(self):
        """Test scanning non-existent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MonitorConfig(
                watch_dir="/nonexistent/directory",
                db_path=str(Path(tmpdir) / "test.db"),
            )
            monitor = TrainingMonitor(config)

            # Should not raise, just warn
            monitor._scan_directory()

    def test_get_statistics(self):
        """Test getting monitor statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MonitorConfig(
                watch_dir=tmpdir,
                db_path=str(Path(tmpdir) / "test.db"),
            )
            monitor = TrainingMonitor(config)

            stats = monitor.get_statistics()
            assert "total_analyses" in stats
            assert "total_alerts" in stats

    def test_get_alerts(self):
        """Test getting alerts from monitor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MonitorConfig(
                watch_dir=tmpdir,
                db_path=str(Path(tmpdir) / "test.db"),
            )
            monitor = TrainingMonitor(config)

            alerts = monitor.get_alerts(limit=10)
            assert isinstance(alerts, list)


class TestTrainingMonitorIntegration:
    """Integration tests for TrainingMonitor."""

    def test_full_monitoring_cycle(self):
        """Test complete monitoring cycle with trajectory analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MonitorConfig(
                watch_dir=tmpdir,
                db_path=str(Path(tmpdir) / "test.db"),
                hack_score_threshold=0.3,
                poll_interval=0.1,
            )
            monitor = TrainingMonitor(config)

            # Track alerts
            alerts_received = []
            monitor.on_alert(lambda a: alerts_received.append(a))

            # Create suspicious trajectory
            traj_path = Path(tmpdir) / "suspicious.json"
            with open(traj_path, "w") as f:
                json.dump(
                    {
                        "steps": [
                            {"action": "bypass", "code": "sys.exit(0)"},
                        ],
                        "cot_traces": ["Let me hack this test"],
                        "code_outputs": ["sys.exit(0)"],
                    },
                    f,
                )

            # Run scan
            monitor._scan_directory()

            # Check file was processed
            assert str(traj_path) in monitor._processed_files

    def test_checkpoint_scanning(self):
        """Test scanning checkpoint directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint with trajectory
            checkpoint_dir = Path(tmpdir) / "checkpoint-1000"
            checkpoint_dir.mkdir()

            traj_path = checkpoint_dir / "rollout.json"
            with open(traj_path, "w") as f:
                json.dump(
                    {
                        "steps": [{"action": "normal"}],
                        "cot_traces": ["Normal trace"],
                    },
                    f,
                )

            config = MonitorConfig(
                watch_dir=tmpdir,
                db_path=str(Path(tmpdir) / "test.db"),
            )
            monitor = TrainingMonitor(config)

            monitor._scan_directory()

            # Checkpoint trajectory should be processed
            assert str(traj_path) in monitor._processed_files
