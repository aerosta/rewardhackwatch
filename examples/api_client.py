#!/usr/bin/env python3
"""REST API client example for RewardHackWatch.

This example shows how to use the RewardHackWatch API server.
First, start the server: rewardhackwatch serve --port 8000
"""

import json

import requests

API_BASE = "http://localhost:8000"


def analyze_code(cot_traces: list, code_outputs: list):
    """Send code to the API for analysis."""
    response = requests.post(
        f"{API_BASE}/analyze", json={"cot_traces": cot_traces, "code_outputs": code_outputs}
    )
    return response.json()


def batch_analyze(trajectories: list):
    """Batch analyze multiple trajectories."""
    response = requests.post(f"{API_BASE}/batch", json={"trajectories": trajectories})
    return response.json()


def check_health():
    """Check API server health."""
    response = requests.get(f"{API_BASE}/status")
    return response.json()


def main():
    print("=" * 60)
    print("RewardHackWatch API Client Example")
    print("=" * 60)

    # Check server health
    try:
        health = check_health()
        print(f"Server status: {health.get('status', 'unknown')}")
        print(f"Version: {health.get('version', 'unknown')}")
    except requests.ConnectionError:
        print("ERROR: Cannot connect to API server.")
        print("Start the server first: rewardhackwatch serve --port 8000")
        return

    # Analyze suspicious code
    print("\n--- Analyzing suspicious code ---")
    result = analyze_code(
        cot_traces=["Let me bypass the test..."], code_outputs=["import sys\nsys.exit(0)"]
    )
    print(json.dumps(result, indent=2))

    # Batch analyze
    print("\n--- Batch analysis ---")
    batch_result = batch_analyze(
        [
            {
                "cot_traces": ["Implementing factorial..."],
                "code_outputs": ["def factorial(n): return 1 if n<=1 else n*factorial(n-1)"],
            },
            {
                "cot_traces": ["Bypassing checks..."],
                "code_outputs": ["assert True  # skip all tests"],
            },
        ]
    )
    print(f"Batch analyzed {len(batch_result.get('results', []))} trajectories")
    for i, r in enumerate(batch_result.get("results", [])):
        print(
            f"  [{i + 1}] Risk: {r.get('risk_level', 'unknown')}, Score: {r.get('ml_score', 0):.3f}"
        )


if __name__ == "__main__":
    main()
