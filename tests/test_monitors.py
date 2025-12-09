"""Tests for monitors - AlertSystem and TrainingMonitor."""

import os
import tempfile
from datetime import datetime

import pytest

from rewardhackwatch.core.monitors import (
    AlertLevel,
    AlertSource,
    AlertSystem,
    AlertSystemConfig,
    AlertThresholds,
    AnalysisResult,
)


class TestAlertSystem:
    """Tests for AlertSystem."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def alert_system(self, temp_db):
        """Create an AlertSystem with temp database."""
        config = AlertSystemConfig(
            db_path=temp_db,
            enable_console=False,
            enable_file_log=False,
        )
        return AlertSystem(config)

    def test_initialization(self, alert_system):
        """Test alert system initialization."""
        assert alert_system is not None
        assert not alert_system.config.enable_console

    def test_process_low_scores(self, alert_system):
        """Test processing low scores (no alerts)."""
        result = alert_system.process_analysis(
            hack_score=0.1,
            generalization_risk=0.1,
            deception_score=0.1,
            cot_consistency_score=0.9,
            file_path="test.json",
            detector_results=[],
            cot_analysis_result={},
        )

        assert isinstance(result, AnalysisResult)
        assert len(result.alerts_triggered) == 0

    def test_process_high_hack_score(self, alert_system):
        """Test that high hack score triggers alert."""
        result = alert_system.process_analysis(
            hack_score=0.85,
            generalization_risk=0.2,
            deception_score=0.2,
            cot_consistency_score=0.8,
            file_path="test.json",
            detector_results=[],
            cot_analysis_result={},
        )

        assert len(result.alerts_triggered) > 0
        assert any(a.source == AlertSource.HACK_SCORE for a in result.alerts_triggered)

    def test_process_high_deception_score(self, alert_system):
        """Test that high deception score triggers alert."""
        result = alert_system.process_analysis(
            hack_score=0.2,
            generalization_risk=0.2,
            deception_score=0.75,
            cot_consistency_score=0.3,
            file_path="test.json",
            detector_results=[],
            cot_analysis_result={},
        )

        assert len(result.alerts_triggered) > 0
        assert any(a.source == AlertSource.DECEPTION_SCORE for a in result.alerts_triggered)

    def test_process_high_generalization_risk(self, alert_system):
        """Test that high generalization risk triggers alert."""
        result = alert_system.process_analysis(
            hack_score=0.2,
            generalization_risk=0.65,
            deception_score=0.2,
            cot_consistency_score=0.8,
            file_path="test.json",
            detector_results=[],
            cot_analysis_result={},
        )

        assert len(result.alerts_triggered) > 0
        assert any(a.source == AlertSource.GENERALIZATION_RISK for a in result.alerts_triggered)

    def test_alert_levels(self, alert_system):
        """Test alert level classification."""
        # Critical level
        result = alert_system.process_analysis(
            hack_score=0.95,
            generalization_risk=0.1,
            deception_score=0.1,
            cot_consistency_score=0.9,
            file_path="test.json",
            detector_results=[],
            cot_analysis_result={},
        )

        critical_alerts = [a for a in result.alerts_triggered if a.level == AlertLevel.CRITICAL]
        assert len(critical_alerts) > 0

    def test_get_statistics(self, alert_system):
        """Test getting statistics."""
        # Process some analyses first
        for i in range(3):
            alert_system.process_analysis(
                hack_score=0.8,
                generalization_risk=0.2,
                deception_score=0.2,
                cot_consistency_score=0.8,
                file_path=f"test_{i}.json",
                detector_results=[],
                cot_analysis_result={},
            )

        stats = alert_system.get_statistics()

        assert "total_analyses" in stats
        assert "total_alerts" in stats
        assert stats["total_analyses"] >= 3

    def test_get_alerts(self, alert_system):
        """Test getting alerts with filters."""
        # Generate some alerts
        alert_system.process_analysis(
            hack_score=0.9,
            generalization_risk=0.2,
            deception_score=0.2,
            cot_consistency_score=0.8,
            file_path="test.json",
            detector_results=[],
            cot_analysis_result={},
        )

        alerts = alert_system.get_alerts(limit=10)
        assert isinstance(alerts, list)

    def test_custom_thresholds(self, temp_db):
        """Test with custom thresholds."""
        thresholds = AlertThresholds(
            hack_score=0.3,
            hack_score_critical=0.5,
        )
        config = AlertSystemConfig(
            db_path=temp_db,
            thresholds=thresholds,
            enable_console=False,
        )
        alert_system = AlertSystem(config)

        # Should trigger with lower score now
        result = alert_system.process_analysis(
            hack_score=0.4,
            generalization_risk=0.1,
            deception_score=0.1,
            cot_consistency_score=0.9,
            file_path="test.json",
            detector_results=[],
            cot_analysis_result={},
        )

        assert len(result.alerts_triggered) > 0


class TestAnalysisResult:
    """Tests for AnalysisResult."""

    def test_analysis_result_creation(self):
        """Test creating an AnalysisResult."""
        result = AnalysisResult(
            timestamp=datetime.now(),
            file_path="test.json",
            checkpoint_path=None,
            hack_score=0.5,
            generalization_risk=0.3,
            deception_score=0.2,
            cot_consistency_score=0.8,
            detector_results=[],
            tracker_result=None,
            cot_analysis_result=None,
            alerts_triggered=[],
        )

        assert result.file_path == "test.json"
        assert len(result.alerts_triggered) == 0
        assert result.hack_score == 0.5


class TestAlertLevelEnum:
    """Tests for AlertLevel enum."""

    def test_alert_levels(self):
        """Test alert level values."""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"


class TestAlertSourceEnum:
    """Tests for AlertSource enum."""

    def test_alert_sources(self):
        """Test alert source values."""
        assert AlertSource.HACK_SCORE.value == "hack_score"
        assert AlertSource.DECEPTION_SCORE.value == "deception_score"
        assert AlertSource.GENERALIZATION_RISK.value == "generalization_risk"
