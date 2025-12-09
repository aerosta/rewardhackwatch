#!/usr/bin/env python3
"""
Check Health

Checks the health of all RewardHackWatch services.
"""

import argparse
import sys
from typing import Any


def check_api(port: int = 8000) -> dict[str, Any]:
    """Check API server health."""
    try:
        import requests

        response = requests.get(f"http://localhost:{port}/status", timeout=5)
        if response.status_code == 200:
            return {
                "status": "healthy",
                "data": response.json(),
            }
        else:
            return {
                "status": "unhealthy",
                "error": f"Status code: {response.status_code}",
            }
    except requests.ConnectionError:
        return {"status": "offline", "error": "Connection refused"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_dashboard(port: int = 8501) -> dict[str, Any]:
    """Check dashboard health."""
    try:
        import requests

        response = requests.get(f"http://localhost:{port}", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy"}
        else:
            return {
                "status": "unhealthy",
                "error": f"Status code: {response.status_code}",
            }
    except requests.ConnectionError:
        return {"status": "offline", "error": "Connection refused"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_ollama() -> dict[str, Any]:
    """Check Ollama health."""
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return {
                "status": "healthy",
                "models": models,
            }
        else:
            return {
                "status": "unhealthy",
                "error": f"Status code: {response.status_code}",
            }
    except requests.ConnectionError:
        return {"status": "offline", "error": "Connection refused"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_ml_model(model_path: str = "./models/distilbert") -> dict[str, Any]:
    """Check ML model availability."""
    from pathlib import Path

    model_dir = Path(model_path)

    if not model_dir.exists():
        return {"status": "missing", "error": f"Model not found at {model_path}"}

    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    missing = [f for f in required_files if not (model_dir / f).exists()]

    if missing:
        # Check for alternative model formats
        if (model_dir / "pytorch_model.bin").exists():
            return {"status": "healthy", "format": "pytorch"}

    if not missing:
        return {"status": "healthy", "format": "safetensors"}

    return {"status": "incomplete", "missing": missing}


def print_status(name: str, result: dict[str, Any]) -> None:
    """Print status with color."""
    status = result.get("status", "unknown")

    if status == "healthy":
        icon = "✅"
        color = "\033[92m"  # Green
    elif status == "offline":
        icon = "⚫"
        color = "\033[90m"  # Gray
    else:
        icon = "❌"
        color = "\033[91m"  # Red

    reset = "\033[0m"

    print(f"{icon} {color}{name}: {status}{reset}")

    if "error" in result:
        print(f"   Error: {result['error']}")
    if "models" in result:
        print(f"   Models: {result['models']}")
    if "missing" in result:
        print(f"   Missing: {result['missing']}")


def main():
    parser = argparse.ArgumentParser(description="Check health")
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="API port",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8501,
        help="Dashboard port",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/distilbert",
        help="ML model path",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("REWARDHACKWATCH HEALTH CHECK")
    print("=" * 60)
    print()

    # Check services
    services = [
        ("API Server", lambda: check_api(args.api_port)),
        ("Dashboard", lambda: check_dashboard(args.dashboard_port)),
        ("Ollama", check_ollama),
        ("ML Model", lambda: check_ml_model(args.model_path)),
    ]

    all_healthy = True
    for name, check_fn in services:
        result = check_fn()
        print_status(name, result)
        if result.get("status") not in ["healthy", "offline"]:
            all_healthy = False

    print()
    print("=" * 60)

    if all_healthy:
        print("All services are healthy or offline (as expected)")
        sys.exit(0)
    else:
        print("Some services have issues")
        sys.exit(1)


if __name__ == "__main__":
    main()
