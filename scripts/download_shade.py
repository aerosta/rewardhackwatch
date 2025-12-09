#!/usr/bin/env python3
"""
Download SHADE-Arena Dataset

Downloads the SHADE-Arena (ScaleAI MRT) dataset for benchmarking.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import hf_hub_download, list_repo_files

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed")


def download_shade(
    output_dir: str = "./data/shade",
    limit: Optional[int] = None,
) -> None:
    """
    Download SHADE-Arena dataset.

    Args:
        output_dir: Directory to save the dataset
        limit: Maximum number of transcripts to download
    """
    if not HF_AVAILABLE:
        print("Error: huggingface_hub not installed")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Listing files in ScaleAI/mrt...")

    try:
        files = list_repo_files("ScaleAI/mrt", repo_type="dataset")
    except Exception as e:
        print(f"Error listing files: {e}")
        return

    # Find transcript files
    transcript_files = [f for f in files if f.endswith(".json") and "transcript" in f]
    print(f"Found {len(transcript_files)} transcript files")

    if limit:
        transcript_files = transcript_files[:limit]

    transcripts = []
    for i, file_path in enumerate(transcript_files):
        print(f"[{i + 1}/{len(transcript_files)}] Downloading {file_path}...")

        try:
            local_path = hf_hub_download(
                "ScaleAI/mrt",
                file_path,
                repo_type="dataset",
            )

            with open(local_path) as f:
                data = json.load(f)

            transcripts.append(
                {
                    "file": file_path,
                    "data": data,
                }
            )
        except Exception as e:
            print(f"  Error: {e}")

    # Save combined data
    output_file = output_path / "shade_transcripts.json"
    print(f"\nSaving {len(transcripts)} transcripts to {output_file}")

    with open(output_file, "w") as f:
        json.dump(transcripts, f, indent=2)

    print("\nDownload complete!")
    print(f"Data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download SHADE-Arena dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/shade",
        help="Output directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum transcripts to download",
    )

    args = parser.parse_args()
    download_shade(args.output_dir, args.limit)


if __name__ == "__main__":
    main()
