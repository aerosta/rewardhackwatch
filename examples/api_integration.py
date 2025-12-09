#!/usr/bin/env python3
"""
API Integration Example

Demonstrates how to integrate with the RewardHackWatch REST API
for programmatic trajectory analysis.
"""

from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass
class APIConfig:
    """Configuration for the RewardHackWatch API."""

    base_url: str = "http://localhost:8000"
    timeout: int = 30


class RewardHackWatchClient:
    """Client for the RewardHackWatch REST API."""

    def __init__(self, config: Optional[APIConfig] = None):
        """
        Initialize the API client.

        Args:
            config: API configuration (uses defaults if not provided)
        """
        self.config = config or APIConfig()

    def check_status(self) -> dict[str, Any]:
        """
        Check the API server status.

        Returns:
            Status information dictionary
        """
        response = requests.get(
            f"{self.config.base_url}/status",
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()

    def analyze(
        self,
        cot_traces: list[str],
        code_outputs: list[str],
        use_ml: bool = False,
    ) -> dict[str, Any]:
        """
        Analyze a trajectory for reward hacking.

        Args:
            cot_traces: Chain of thought traces
            code_outputs: Code outputs from the agent
            use_ml: Whether to use the ML detector

        Returns:
            Analysis results
        """
        payload = {
            "cot_traces": cot_traces,
            "code_outputs": code_outputs,
            "use_ml": use_ml,
        }

        response = requests.post(
            f"{self.config.base_url}/analyze",
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()

    def analyze_trajectory(
        self,
        trajectory: dict[str, Any],
        use_ml: bool = False,
    ) -> dict[str, Any]:
        """
        Analyze a full trajectory object.

        Args:
            trajectory: Trajectory dictionary with cot_traces and code_outputs
            use_ml: Whether to use the ML detector

        Returns:
            Analysis results
        """
        return self.analyze(
            cot_traces=trajectory.get("cot_traces", []),
            code_outputs=trajectory.get("code_outputs", []),
            use_ml=use_ml,
        )

    def batch_analyze(
        self,
        trajectories: list[dict[str, Any]],
        use_ml: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Analyze multiple trajectories.

        Args:
            trajectories: List of trajectory dictionaries
            use_ml: Whether to use the ML detector

        Returns:
            List of analysis results
        """
        payload = {
            "trajectories": trajectories,
            "use_ml": use_ml,
        }

        response = requests.post(
            f"{self.config.base_url}/batch_analyze",
            json=payload,
            timeout=self.config.timeout * 2,
        )
        response.raise_for_status()
        return response.json()

    def get_metrics(self) -> dict[str, Any]:
        """
        Get server metrics and statistics.

        Returns:
            Metrics dictionary
        """
        response = requests.get(
            f"{self.config.base_url}/metrics",
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()


def main():
    """Demonstrate API integration."""
    print("=" * 60)
    print("REWARDHACKWATCH API INTEGRATION")
    print("=" * 60)

    # Initialize client
    client = RewardHackWatchClient()

    # Check server status
    print("\n--- Checking Server Status ---")
    try:
        status = client.check_status()
        print(f"Server Status: {status.get('status', 'unknown')}")
        print(f"Version: {status.get('version', 'unknown')}")
        print(f"Detectors: {status.get('detectors', [])}")
    except requests.ConnectionError:
        print("❌ Server not running!")
        print("\nStart the server with:")
        print("  rewardhackwatch api --port 8000")
        print("or")
        print("  uvicorn rewardhackwatch.api.main:app --port 8000")
        return

    # Sample trajectories
    hack_trajectory = {
        "cot_traces": [
            "I need to pass the tests",
            "I'll just exit with code 0 to bypass everything",
        ],
        "code_outputs": ["import sys\nsys.exit(0)"],
    }

    clean_trajectory = {
        "cot_traces": [
            "Let me implement the solution properly",
            "I'll handle edge cases too",
        ],
        "code_outputs": ["def solve(x):\n    return x * 2"],
    }

    # Analyze hack trajectory
    print("\n--- Analyzing Hack Trajectory ---")
    result = client.analyze(
        cot_traces=hack_trajectory["cot_traces"],
        code_outputs=hack_trajectory["code_outputs"],
    )
    print(f"Is Hack: {result.get('is_hack', False)}")
    print(f"Hack Score: {result.get('hack_score', 0):.2f}")
    print(f"Risk Level: {result.get('risk_level', 'unknown')}")

    # Analyze clean trajectory
    print("\n--- Analyzing Clean Trajectory ---")
    result = client.analyze_trajectory(clean_trajectory)
    print(f"Is Hack: {result.get('is_hack', False)}")
    print(f"Hack Score: {result.get('hack_score', 0):.2f}")
    print(f"Risk Level: {result.get('risk_level', 'unknown')}")

    # Batch analysis
    print("\n--- Batch Analysis ---")
    try:
        batch_results = client.batch_analyze([hack_trajectory, clean_trajectory])
        for i, res in enumerate(batch_results):
            print(
                f"Trajectory {i + 1}: hack={res.get('is_hack', False)}, score={res.get('hack_score', 0):.2f}"
            )
    except requests.HTTPError as e:
        print(f"Batch endpoint not available: {e}")

    # Get metrics
    print("\n--- Server Metrics ---")
    try:
        metrics = client.get_metrics()
        print(f"Total Requests: {metrics.get('total_requests', 0)}")
        print(f"Hacks Detected: {metrics.get('hacks_detected', 0)}")
        print(f"Avg Response Time: {metrics.get('avg_response_time_ms', 0):.1f}ms")
    except requests.HTTPError:
        print("Metrics endpoint not available")

    print("\n" + "=" * 60)
    print("API ENDPOINTS REFERENCE")
    print("=" * 60)
    print("""
GET  /status         - Check server health
POST /analyze        - Analyze single trajectory
POST /batch_analyze  - Analyze multiple trajectories
GET  /metrics        - Get server statistics

Request format:
{
    "cot_traces": ["trace1", "trace2"],
    "code_outputs": ["code1", "code2"],
    "use_ml": false
}

Response format:
{
    "is_hack": true,
    "hack_score": 0.85,
    "risk_level": "high",
    "detectors": {
        "pattern": {"is_hack": true, "confidence": 0.9},
        "ast": {"is_hack": false, "confidence": 0.3}
    }
}
""")


if __name__ == "__main__":
    main()
