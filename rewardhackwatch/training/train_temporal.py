"""Training pipeline for temporal (sequence) trajectory modeling.

Uses the AttentionClassifier to model hacking as an escalation process
across trajectory steps, rather than single-snapshot classification.

Usage:
    python -m rewardhackwatch.training.train_temporal
    python -m rewardhackwatch.training.train_temporal --data-dir data/hackbench
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from .dataset import StepSequenceDataset, TrajectoryFeatures
from .models import AttentionClassifier

logger = logging.getLogger(__name__)


@dataclass
class TemporalTrainingConfig:
    """Configuration for temporal model training."""

    # Data
    data_dir: str = "rewardhackwatch/rhw_bench/test_cases/expanded"
    max_steps: int = 50

    # Model
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    # Training
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Early stopping
    patience: int = 10
    min_delta: float = 0.001

    # Output
    output_dir: str = "models/temporal"
    seed: int = 42


class TemporalTrainer:
    """Train AttentionClassifier on step-level trajectory sequences."""

    def __init__(self, config: TemporalTrainingConfig):
        self.config = config
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    def train(self) -> dict:
        """Run full training loop. Returns results dict."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load datasets
        logger.info(f"Loading data from {self.config.data_dir}")
        train_ds = StepSequenceDataset(
            self.config.data_dir,
            split="train",
            max_steps=self.config.max_steps,
            seed=self.config.seed,
        )
        val_ds = StepSequenceDataset(
            self.config.data_dir,
            split="val",
            max_steps=self.config.max_steps,
            seed=self.config.seed,
        )

        logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Build model
        model = AttentionClassifier(
            input_dim=TrajectoryFeatures.feature_dim(),
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            max_steps=self.config.max_steps,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.BCELoss()

        # Training loop
        best_f1 = 0.0
        patience_counter = 0
        history = []

        start_time = time.time()

        for epoch in range(self.config.epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_preds, train_labels = [], []

            for batch in train_loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                output = model(features, mask=mask)
                hack_prob = output["hack_prob"]

                loss = criterion(hack_prob, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                optimizer.step()

                train_loss += loss.item() * len(labels)
                train_preds.extend((hack_prob > 0.5).cpu().numpy().tolist())
                train_labels.extend(labels.cpu().numpy().tolist())

            train_loss /= max(len(train_ds), 1)

            # Validate
            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for batch in val_loader:
                    features = batch["features"].to(self.device)
                    mask = batch["mask"].to(self.device)
                    labels = batch["label"].to(self.device)

                    output = model(features, mask=mask)
                    hack_prob = output["hack_prob"]
                    loss = criterion(hack_prob, labels)

                    val_loss += loss.item() * len(labels)
                    val_preds.extend((hack_prob > 0.5).cpu().numpy().tolist())
                    val_labels.extend(labels.cpu().numpy().tolist())

            val_loss /= max(len(val_ds), 1)

            # Metrics
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)
            val_acc = accuracy_score(val_labels, val_preds)
            train_f1 = f1_score(train_labels, train_preds, zero_division=0)

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "train_f1": round(train_f1, 4),
                    "val_f1": round(val_f1, 4),
                    "val_acc": round(val_acc, 4),
                }
            )

            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} — "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_f1={val_f1:.4f} val_acc={val_acc:.4f}"
            )

            # Early stopping
            if val_f1 > best_f1 + self.config.min_delta:
                best_f1 = val_f1
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), out_dir / "temporal_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        elapsed = time.time() - start_time

        # Final evaluation on best model
        model.load_state_dict(torch.load(out_dir / "temporal_best.pt", weights_only=True))
        model.eval()

        val_preds_final, val_labels_final, val_scores = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                labels = batch["label"].to(self.device)

                output = model(features, mask=mask)
                hack_prob = output["hack_prob"]

                val_scores.extend(hack_prob.cpu().numpy().tolist())
                val_preds_final.extend((hack_prob > 0.5).cpu().numpy().tolist())
                val_labels_final.extend(labels.cpu().numpy().tolist())

        results = {
            "model": "AttentionClassifier",
            "best_val_f1": round(best_f1, 4),
            "final_val_f1": round(f1_score(val_labels_final, val_preds_final, zero_division=0), 4),
            "final_val_precision": round(
                precision_score(val_labels_final, val_preds_final, zero_division=0), 4
            ),
            "final_val_recall": round(
                recall_score(val_labels_final, val_preds_final, zero_division=0), 4
            ),
            "final_val_accuracy": round(accuracy_score(val_labels_final, val_preds_final), 4),
            "epochs_trained": len(history),
            "training_time_seconds": round(elapsed, 1),
            "config": {
                "hidden_dim": self.config.hidden_dim,
                "num_heads": self.config.num_heads,
                "num_layers": self.config.num_layers,
                "max_steps": self.config.max_steps,
                "learning_rate": self.config.learning_rate,
            },
            "history": history,
        }

        # Save results
        with open(out_dir / "temporal_results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Training complete. Best F1: {best_f1:.4f}, Time: {elapsed:.1f}s")
        return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train temporal trajectory model")
    parser.add_argument("--data-dir", default="rewardhackwatch/rhw_bench/test_cases/expanded")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dir", default="models/temporal")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = TemporalTrainingConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    trainer = TemporalTrainer(config)
    results = trainer.train()

    print("\nResults:")
    print(f"  Best Val F1: {results['best_val_f1']}")
    print(f"  Final Val F1: {results['final_val_f1']}")
    print(f"  Precision: {results['final_val_precision']}")
    print(f"  Recall: {results['final_val_recall']}")
    print(f"  Epochs: {results['epochs_trained']}")
    print(f"  Time: {results['training_time_seconds']}s")


if __name__ == "__main__":
    main()
