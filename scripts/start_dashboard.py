#!/usr/bin/env python3
"""
Start Dashboard

Starts the RewardHackWatch Streamlit dashboard.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def start_dashboard(
    port: int = 8501,
    host: str = "localhost",
    browser: bool = True,
) -> None:
    """
    Start the Streamlit dashboard.

    Args:
        port: Port to run on
        host: Host to bind to
        browser: Whether to open browser automatically
    """
    # Find dashboard path
    dashboard_path = Path(__file__).parent.parent / "rewardhackwatch" / "dashboard" / "app.py"

    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)

    print("Starting RewardHackWatch Dashboard...")
    print(f"  Port: {port}")
    print(f"  Host: {host}")
    print(f"  URL: http://{host}:{port}")
    print()

    # Build streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
    ]

    if not browser:
        cmd.extend(["--server.headless", "true"])

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except FileNotFoundError:
        print("Error: Streamlit not installed. Run: pip install streamlit")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Start dashboard")
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    args = parser.parse_args()
    start_dashboard(args.port, args.host, not args.no_browser)


if __name__ == "__main__":
    main()
