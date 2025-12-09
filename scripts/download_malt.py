#!/usr/bin/env python3
"""
Download MALT Dataset

Downloads the MALT dataset from HuggingFace for training.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Run: pip install datasets huggingface_hub")


def download_malt(
    output_dir: str = "./data/malt",
    split: Optional[str] = None,
    limit: Optional[int] = None,
) -> None:
    """
    Download MALT dataset.

    Args:
        output_dir: Directory to save the dataset
        split: Specific split to download (train/validation/test)
        limit: Maximum number of samples to download
    """
    if not HF_AVAILABLE:
        print("Error: huggingface_hub not installed")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading MALT dataset to {output_path}")

    # Try to load the dataset
    try:
        dataset = load_dataset("ScaleAI/malt", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative method...")
        try:
            # Alternative: download specific files
            dataset = load_dataset(
                "json",
                data_files={
                    "train": "hf://datasets/ScaleAI/malt/train.jsonl",
                },
            )
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            return

    splits_to_download = [split] if split else ["train", "validation", "test"]

    for split_name in splits_to_download:
        if split_name not in dataset:
            print(f"Split '{split_name}' not found, skipping")
            continue

        split_data = dataset[split_name]

        if limit:
            split_data = split_data.select(range(min(limit, len(split_data))))

        # Save to JSON
        output_file = output_path / f"malt_{split_name}.json"
        print(f"Saving {len(split_data)} samples to {output_file}")

        with open(output_file, "w") as f:
            json.dump(list(split_data), f, indent=2)

        print(f"✓ Saved {split_name} split")

    print("\nDownload complete!")
    print(f"Data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download MALT dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/malt",
        help="Output directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        help="Specific split to download",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum samples to download",
    )

    args = parser.parse_args()
    download_malt(args.output_dir, args.split, args.limit)


if __name__ == "__main__":
    main()
