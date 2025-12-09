#!/usr/bin/env python3
"""
Setup Ollama

Sets up Ollama for local LLM judge inference.
"""

import argparse
import subprocess
import sys
import time


def check_ollama_installed() -> bool:
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def list_models() -> list:
    """List installed models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            return [line.split()[0] for line in lines if line]
        return []
    except:
        return []


def pull_model(model_name: str) -> bool:
    """Pull a model."""
    print(f"Pulling model: {model_name}")
    result = subprocess.run(
        ["ollama", "pull", model_name],
    )
    return result.returncode == 0


def start_server() -> bool:
    """Start Ollama server."""
    print("Starting Ollama server...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for server to start
    for _ in range(10):
        time.sleep(1)
        if check_ollama_running():
            return True

    return False


def test_model(model_name: str) -> bool:
    """Test a model with a simple prompt."""
    print(f"Testing model: {model_name}")

    try:
        import requests

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "Say 'hello' in one word.",
                "stream": False,
            },
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data.get('response', '')[:100]}")
            return True
    except Exception as e:
        print(f"Test failed: {e}")

    return False


def main():
    parser = argparse.ArgumentParser(description="Setup Ollama")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:8b",
        help="Model to install",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip model test",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("OLLAMA SETUP")
    print("=" * 60)

    # Check if Ollama is installed
    print("\n1. Checking Ollama installation...")
    if not check_ollama_installed():
        print("   ❌ Ollama not installed!")
        print("   Install from: https://ollama.ai")
        sys.exit(1)
    print("   ✅ Ollama is installed")

    # Check if server is running
    print("\n2. Checking Ollama server...")
    if not check_ollama_running():
        print("   Server not running, starting...")
        if not start_server():
            print("   ❌ Failed to start server")
            sys.exit(1)
    print("   ✅ Server is running")

    # List installed models
    print("\n3. Checking installed models...")
    models = list_models()
    print(f"   Installed: {models or 'none'}")

    # Pull model if needed
    print(f"\n4. Ensuring {args.model} is available...")
    if args.model not in models:
        if not pull_model(args.model):
            print(f"   ❌ Failed to pull {args.model}")
            sys.exit(1)
    print(f"   ✅ {args.model} is available")

    # Test model
    if not args.skip_test:
        print(f"\n5. Testing {args.model}...")
        if test_model(args.model):
            print("   ✅ Model test passed")
        else:
            print("   ⚠️ Model test failed (may still work)")

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print(f"\nOllama is ready with {args.model}")
    print("API endpoint: http://localhost:11434")


if __name__ == "__main__":
    main()
