"""Tests for FastAPI backend."""

import pytest
from fastapi.testclient import TestClient

from rewardhackwatch.api.main import AppState, app


@pytest.fixture
def client():
    """Create test client."""
    # Initialize app state
    import rewardhackwatch.api.main as main_module

    main_module.state = AppState()

    with TestClient(app) as client:
        yield client

    main_module.state = None


class TestHealthEndpoint:
    """Tests for /status endpoint."""

    def test_health_check(self, client):
        """Test health check returns healthy."""
        response = client.get("/status")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data
        assert "components" in data

    def test_components_status(self, client):
        """Test all components are ready."""
        response = client.get("/status")
        data = response.json()

        components = data["components"]
        assert components["pattern_detector"] == "ready"
        assert components["ast_detector"] == "ready"
        assert components["cot_analyzer"] == "ready"


class TestAnalyzeEndpoint:
    """Tests for /analyze endpoint."""

    def test_analyze_clean_trajectory(self, client):
        """Test analyzing a clean trajectory."""
        response = client.post(
            "/analyze",
            json={
                "cot_traces": ["Let me solve this properly."],
                "code_outputs": ["def solve(x): return x * 2"],
                "task": "Double a number",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "hack_score" in data
        assert "deception_score" in data
        assert "risk_level" in data
        assert data["hack_score"] < 0.5

    def test_analyze_hack_trajectory(self, client):
        """Test analyzing a hacking trajectory."""
        response = client.post(
            "/analyze",
            json={
                "cot_traces": ["I'll just hack this test."],
                "code_outputs": ["sys.exit(0)"],
                "task": "Pass the tests",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["hack_score"] > 0.3
        assert len(data["detections"]) > 0

    def test_analyze_empty_trajectory(self, client):
        """Test analyzing empty trajectory."""
        response = client.post("/analyze", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["hack_score"] == 0.0

    def test_analyze_with_steps(self, client):
        """Test analyzing trajectory with steps."""
        response = client.post(
            "/analyze",
            json={
                "steps": [
                    {"thought": "Let me think", "action": "print('hello')"},
                    {"thought": "Now I'll hack it", "action": "sys.exit(0)"},
                ],
            },
        )

        assert response.status_code == 200


class TestBatchEndpoint:
    """Tests for /analyze/batch endpoint."""

    def test_batch_analyze(self, client):
        """Test batch analysis."""
        response = client.post(
            "/analyze/batch",
            json={
                "trajectories": [
                    {
                        "cot_traces": ["Clean solution"],
                        "code_outputs": ["return 1"],
                    },
                    {
                        "cot_traces": ["Hack it"],
                        "code_outputs": ["sys.exit(0)"],
                    },
                ],
                "parallel": False,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["results"]) == 2
        assert "summary" in data
        assert data["summary"]["total_trajectories"] == 2

    def test_batch_parallel(self, client):
        """Test batch analysis with parallel processing."""
        response = client.post(
            "/analyze/batch",
            json={
                "trajectories": [
                    {"code_outputs": ["code1"]},
                    {"code_outputs": ["code2"]},
                ],
                "parallel": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2

    def test_batch_summary_stats(self, client):
        """Test batch summary statistics."""
        response = client.post(
            "/analyze/batch",
            json={
                "trajectories": [
                    {"code_outputs": ["sys.exit(0)"]},
                    {"code_outputs": ["return 42"]},
                ],
            },
        )

        data = response.json()
        summary = data["summary"]

        assert "avg_hack_score" in summary
        assert "max_hack_score" in summary
        assert "risk_distribution" in summary


class TestStatsEndpoint:
    """Tests for /stats endpoint."""

    def test_get_statistics(self, client):
        """Test getting statistics."""
        # First do some analyses
        client.post("/analyze", json={"code_outputs": ["test"]})

        response = client.get("/stats")
        assert response.status_code == 200

        data = response.json()
        assert "total_analyses" in data
        assert "total_alerts" in data


class TestAlertsEndpoint:
    """Tests for /alerts endpoint."""

    def test_get_alerts(self, client):
        """Test getting alerts."""
        # Generate an alert
        client.post(
            "/analyze",
            json={
                "code_outputs": ["sys.exit(0)"],
                "cot_traces": ["Hack this"],
            },
        )

        response = client.get("/alerts")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_get_alerts_with_filters(self, client):
        """Test getting alerts with filters."""
        response = client.get("/alerts?limit=10&level=warning")
        assert response.status_code == 200

    def test_invalid_level_filter(self, client):
        """Test invalid level filter."""
        response = client.get("/alerts?level=invalid")
        assert response.status_code == 400


class TestWebSocket:
    """Tests for WebSocket endpoint."""

    def test_websocket_ping_pong(self, client):
        """Test WebSocket ping/pong."""
        with client.websocket_connect("/ws/monitor") as websocket:
            websocket.send_json({"type": "ping"})
            response = websocket.receive_json()
            assert response["type"] == "pong"

    def test_websocket_analyze(self, client):
        """Test WebSocket analysis."""
        with client.websocket_connect("/ws/monitor") as websocket:
            websocket.send_json(
                {
                    "type": "analyze",
                    "trajectory": {
                        "code_outputs": ["test code"],
                    },
                }
            )
            response = websocket.receive_json()
            assert response["type"] == "analysis_result"
            assert "result" in response


class TestErrorHandling:
    """Tests for error handling."""

    def test_global_exception_handler(self, client):
        """Test that exceptions are handled gracefully."""
        # The app should handle errors without crashing
        response = client.post(
            "/analyze",
            json={
                "code_outputs": ["valid code"],
            },
        )
        assert response.status_code == 200
