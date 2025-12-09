"""Tests for tracker modules."""

import pytest


class TestBehaviorTracker:
    """Tests for behavior tracker."""

    def test_track_behavior(self):
        """Test tracking behavior."""
        from rewardhackwatch.core.trackers.behavior_tracker import BehaviorTracker

        tracker = BehaviorTracker()
        tracker.track("traj_001", {"patterns": ["sys_exit"], "score": 0.8})

        trends = tracker.get_trends()
        assert trends["total_observations"] == 1
        assert trends["trajectories_tracked"] == 1
        assert tracker.pattern_counts["sys_exit"] == 1

    def test_get_high_risk_patterns(self):
        """Test getting high risk patterns."""
        from rewardhackwatch.core.trackers.behavior_tracker import BehaviorTracker

        tracker = BehaviorTracker()
        for _ in range(5):
            tracker.track("traj", {"patterns": ["high_risk"], "score": 0.9})
            tracker.track("traj", {"patterns": ["low_risk"], "score": 0.2})

        high_risk = tracker.get_high_risk_patterns(threshold=0.7)
        assert "high_risk" in high_risk
        assert "low_risk" not in high_risk


class TestSessionTracker:
    """Tests for session tracker."""

    def test_start_session(self):
        """Test starting a session."""
        from rewardhackwatch.core.trackers.session_tracker import SessionTracker

        tracker = SessionTracker()
        session_id = tracker.start_session()

        assert session_id is not None
        assert tracker.current_session == session_id

    def test_record_analysis(self):
        """Test recording analysis."""
        from rewardhackwatch.core.trackers.session_tracker import SessionTracker

        tracker = SessionTracker()
        tracker.start_session("test")
        tracker.record_analysis({"score": 0.5})
        tracker.record_analysis({"score": 0.8})

        stats = tracker.get_session_stats()
        assert stats.total_trajectories == 2
        assert stats.high_risk_count == 1

    def test_end_session(self):
        """Test ending a session."""
        from rewardhackwatch.core.trackers.session_tracker import SessionTracker

        tracker = SessionTracker()
        tracker.start_session("test")
        tracker.record_analysis({"score": 0.6})
        tracker.end_session()

        stats = tracker.get_session_stats("test")
        assert stats.end_time is not None
        assert abs(stats.average_score - 0.6) < 0.01


class TestGeneralizationTracker:
    """Tests for generalization tracker."""

    def test_analyze_trajectory(self):
        """Test trajectory analysis."""
        try:
            from rewardhackwatch.core.trackers import GeneralizationTracker

            tracker = GeneralizationTracker()
            trajectory = {"steps": [{"type": "code"}], "code_outputs": ["print('hello')"]}

            result = tracker.analyze_trajectory(trajectory)
            assert result is not None
            assert hasattr(result, "risk_level")
        except ImportError:
            pytest.skip("GeneralizationTracker not available")

    def test_update_scores(self):
        """Test score updates."""
        try:
            from rewardhackwatch.core.trackers import GeneralizationTracker

            tracker = GeneralizationTracker()
            for i in range(10):
                tracker.update(0.5, 0.5)

            result = tracker.analyze()
            assert result is not None
        except ImportError:
            pytest.skip("GeneralizationTracker not available")
