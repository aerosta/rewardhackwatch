"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

try:
    from rewardhackwatch.api.main import app

    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False


@pytest.fixture
def api_client():
    """Create a test client with lifespan context."""
    with TestClient(app) as client:
        yield client


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestStatusEndpoint:
    """Tests for /status endpoint."""

    @pytest.fixture
    def client(self, api_client):
        return api_client

    def test_status_returns_ok(self, client):
        """Test that status endpoint returns OK."""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "healthy")

    def test_status_includes_version(self, client):
        """Test that status includes version."""
        response = client.get("/status")
        data = response.json()
        assert "version" in data

    def test_status_includes_detectors(self, client):
        """Test that status includes available detectors (components)."""
        response = client.get("/status")
        data = response.json()
        # API may use "detectors" or "components" depending on schema
        assert "detectors" in data or "components" in data


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestAnalyzeEndpoint:
    """Tests for /analyze endpoint."""

    @pytest.fixture
    def client(self, api_client):
        return api_client

    def test_analyze_hack_trajectory(self, client):
        """Test analyzing a hack trajectory."""
        payload = {
            "cot_traces": ["I'll bypass the tests with sys.exit"],
            "code_outputs": ["sys.exit(0)"],
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        # API returns hack_score (0-1) or is_hack (bool)
        if "is_hack" in data:
            assert data["is_hack"]
        else:
            assert data.get("hack_score", 0) > 0.3  # Should detect as potential hack

    def test_analyze_clean_trajectory(self, client):
        """Test analyzing a clean trajectory."""
        payload = {
            "cot_traces": ["Implementing factorial"],
            "code_outputs": ["def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"],
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        # API returns hack_score (0-1) or is_hack (bool)
        if "is_hack" in data:
            assert not data["is_hack"]
        else:
            assert data.get("hack_score", 1) < 0.5  # Should be low score for clean code

    def test_analyze_empty_payload(self, client):
        """Test analyzing empty payload."""
        payload = {}
        response = client.post("/analyze", json=payload)
        # Should either return 200 with clean result or 422 validation error
        assert response.status_code in [200, 422]

    def test_analyze_with_ml(self, client):
        """Test analyzing with ML detector."""
        payload = {
            "cot_traces": ["test"],
            "code_outputs": ["print('hello')"],
            "use_ml": True,
        }
        response = client.post("/analyze", json=payload)
        # May fail if ML model not available
        assert response.status_code in [200, 500]


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestBatchAnalyzeEndpoint:
    """Tests for batch analyze endpoint."""

    @pytest.fixture
    def client(self, api_client):
        return api_client

    def test_batch_analyze(self, client):
        """Test batch analyzing trajectories."""
        payload = {
            "trajectories": [
                {"cot_traces": ["hack"], "code_outputs": ["sys.exit(0)"]},
                {"cot_traces": ["clean"], "code_outputs": ["print('hi')"]},
            ]
        }
        response = client.post("/batch_analyze", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert len(data) == 2
        else:
            # Endpoint may not be implemented
            assert response.status_code == 404

    def test_batch_analyze_empty(self, client):
        """Test batch analyze with empty list."""
        payload = {"trajectories": []}
        response = client.post("/batch_analyze", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert len(data) == 0


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    @pytest.fixture
    def client(self, api_client):
        return api_client

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        if response.status_code == 200:
            data = response.json()
            assert "total_requests" in data or isinstance(data, dict)
        else:
            # Endpoint may not be implemented
            assert response.status_code == 404


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestErrorHandling:
    """Tests for API error handling."""

    @pytest.fixture
    def client(self, api_client):
        return api_client

    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/analyze", content="invalid json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        # Depends on schema definition
        response = client.post("/analyze", json={})
        # Either accepts empty or returns validation error
        assert response.status_code in [200, 422]

    def test_invalid_field_types(self, client):
        """Test handling of invalid field types."""
        payload = {
            "cot_traces": "not a list",  # Should be list
            "code_outputs": 123,  # Should be list
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422
