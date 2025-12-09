"""Sample result fixtures for testing."""

SAMPLE_DETECTION_RESULT = {
    "score": 0.75,
    "risk_level": "high",
    "detections": [
        {
            "pattern_name": "sys_exit_zero",
            "description": "Exit with success code to bypass tests",
            "location": "code_output_0",
            "confidence": 0.9,
        }
    ],
    "metadata": {"detector": "pattern", "version": "1.0"},
}

SAMPLE_BENCHMARK_RESULT = {
    "precision": 0.92,
    "recall": 0.89,
    "f1": 0.91,
    "accuracy": 0.99,
    "true_positives": 89,
    "false_positives": 8,
    "true_negatives": 900,
    "false_negatives": 11,
}

SAMPLE_ANALYSIS_RESULT = {
    "trajectory_id": "test_001",
    "hack_score": 0.75,
    "generalization_risk": 0.6,
    "deception_score": 0.3,
    "effort_suspicion": 0.4,
    "risk_level": "high",
    "detections": [
        {
            "source": "pattern_detector",
            "pattern": "sys_exit_zero",
            "description": "Exit with success code to bypass tests",
            "location": "code_output_0",
            "confidence": 0.9,
        },
        {
            "source": "ast_detector",
            "pattern": "exit_zero",
            "description": "Exit with success code",
            "location": "code_output_0:line_2",
            "confidence": 0.95,
        },
    ],
    "alerts": [
        {
            "level": "warning",
            "source": "hack_score",
            "message": "High hack score detected",
            "value": 0.75,
        }
    ],
    "metadata": {"analysis_time_ms": 45.2, "detectors_used": ["pattern", "ast"]},
}

SAMPLE_BATCH_RESULT = {
    "results": [
        {"score": 0.1, "risk_level": "low"},
        {"score": 0.5, "risk_level": "medium"},
        {"score": 0.8, "risk_level": "high"},
    ],
    "summary": {"total": 3, "avg_score": 0.47, "max_score": 0.8, "high_risk_count": 1},
    "processing_time_ms": 123.4,
}

SAMPLE_ALERT = {
    "level": "warning",
    "source": "hack_score",
    "message": "High hack score: 0.75",
    "value": 0.75,
    "timestamp": "2024-01-01T12:00:00",
    "trajectory_id": "test_001",
}

SAMPLE_TRACKER_RESULT = {
    "correlation": 0.85,
    "risk_level": "high",
    "generalization_detected": True,
    "transition_points": [10, 25],
    "rmgi_history": [0.2, 0.3, 0.5, 0.7, 0.85],
}
