#!/usr/bin/env python3
"""
Start API Server

Starts the RewardHackWatch API server with uvicorn.
"""

import argparse
import sys


def start_api(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
    log_level: str = "info",
) -> None:
    """
    Start the API server.

    Args:
        host: Host to bind to
        port: Port to bind to
        workers: Number of workers
        reload: Enable auto-reload
        log_level: Logging level
    """
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    print("Starting RewardHackWatch API server...")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Workers: {workers}")
    print(f"  Reload: {reload}")
    print(f"  Log Level: {log_level}")
    print()

    uvicorn.run(
        "rewardhackwatch.api.main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level=log_level,
    )


def main():
    parser = argparse.ArgumentParser(description="Start API server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level",
    )

    args = parser.parse_args()
    start_api(args.host, args.port, args.workers, args.reload, args.log_level)


if __name__ == "__main__":
    main()
