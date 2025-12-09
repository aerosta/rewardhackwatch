#!/usr/bin/env python3
"""
Export Model

Exports a trained model for deployment.
"""

import argparse
import json
from pathlib import Path


def export_model(
    model_path: str,
    output_path: str,
    format: str = "huggingface",
    quantize: bool = False,
) -> None:
    """
    Export model for deployment.

    Args:
        model_path: Path to trained model
        output_path: Output path for exported model
        format: Export format (huggingface, onnx, torchscript)
        quantize: Whether to quantize the model
    """
    model_dir = Path(model_path)
    output_dir = Path(output_path)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting model from {model_dir} to {output_dir}")
    print(f"Format: {format}")
    print(f"Quantize: {quantize}")

    if format == "huggingface":
        export_huggingface(model_dir, output_dir, quantize)
    elif format == "onnx":
        export_onnx(model_dir, output_dir, quantize)
    elif format == "torchscript":
        export_torchscript(model_dir, output_dir)
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"\nModel exported to: {output_dir}")


def export_huggingface(model_dir: Path, output_dir: Path, quantize: bool) -> None:
    """Export as HuggingFace model."""
    try:
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
    except ImportError:
        print("Error: transformers not installed")
        return

    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)

    if quantize:
        print("Quantizing model...")
        try:
            import torch

            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        except Exception as e:
            print(f"Quantization failed: {e}")

    print("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save config
    config = {
        "model_type": "distilbert",
        "task": "sequence_classification",
        "num_labels": 2,
        "quantized": quantize,
    }
    with open(output_dir / "export_config.json", "w") as f:
        json.dump(config, f, indent=2)


def export_onnx(model_dir: Path, output_dir: Path, quantize: bool) -> None:
    """Export as ONNX model."""
    try:
        import torch
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
    except ImportError:
        print("Error: transformers or torch not installed")
        return

    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)

    model.eval()

    # Create dummy input
    dummy_input = tokenizer("test input", return_tensors="pt")

    print("Exporting to ONNX...")
    onnx_path = output_dir / "model.onnx"

    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"},
        },
        opset_version=12,
    )

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    if quantize:
        print("Quantizing ONNX model...")
        try:
            from onnxruntime.quantization import quantize_dynamic

            quantize_dynamic(
                str(onnx_path),
                str(output_dir / "model_quantized.onnx"),
            )
        except Exception as e:
            print(f"ONNX quantization failed: {e}")


def export_torchscript(model_dir: Path, output_dir: Path) -> None:
    """Export as TorchScript model."""
    try:
        import torch
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
    except ImportError:
        print("Error: transformers or torch not installed")
        return

    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)

    model.eval()

    # Trace model
    dummy_input = tokenizer("test input", return_tensors="pt")

    print("Tracing model...")
    traced = torch.jit.trace(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
    )

    # Save
    traced.save(output_dir / "model.pt")
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Export model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["huggingface", "onnx", "torchscript"],
        default="huggingface",
        help="Export format",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize model",
    )

    args = parser.parse_args()
    export_model(args.model, args.output, args.format, args.quantize)


if __name__ == "__main__":
    main()
