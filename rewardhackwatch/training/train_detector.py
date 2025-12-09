"""Training script for RewardHackWatch ML models."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .dataset import CombinedDataset, RHWBenchDataset
from .models import FeatureClassifier, MLPClassifier


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Data
    data_dir: str = "rewardhackwatch/rhw_bench/test_cases"
    trajectories_dir: str = "rewardhackwatch/rhw_bench/trajectories"

    # Combined training options
    use_combined: bool = False  # Use combined RHW + SHADE-Arena data
    shade_limit: int = 100  # Max SHADE-Arena trajectories to include

    # Model
    model_type: str = "mlp"  # "mlp", "feature", "sklearn_rf", "sklearn_gb", "sklearn_lr"
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.3

    # Training
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    early_stopping_patience: int = 10

    # Output
    output_dir: str = "models"
    model_name: str = "hack_classifier"


class Trainer:
    """Trainer for RewardHackWatch ML models."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._get_device()

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def train_pytorch(self) -> dict[str, Any]:
        """Train PyTorch model."""
        print(f"Training PyTorch {self.config.model_type} model...")
        print(f"Device: {self.device}")

        # Load datasets
        train_dataset = RHWBenchDataset(self.config.data_dir, split="train")
        val_dataset = RHWBenchDataset(self.config.data_dir, split="val")

        # Add trajectory data if available
        if Path(self.config.trajectories_dir).exists():
            traj_dataset = RHWBenchDataset(self.config.trajectories_dir, split="all")
            # Combine datasets
            train_dataset.trajectories.extend(traj_dataset.trajectories[: len(traj_dataset) // 2])
            train_dataset.labels.extend(traj_dataset.labels[: len(traj_dataset) // 2])

        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

        if len(train_dataset) == 0:
            print("No training data found!")
            return {"error": "No training data"}

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )
            if len(val_dataset) > 0
            else None
        )

        # Create model
        if self.config.model_type == "mlp":
            model = MLPClassifier(
                hidden_dims=self.config.hidden_dims,
                dropout=self.config.dropout,
            )
        else:
            model = FeatureClassifier(dropout=self.config.dropout)

        model = model.to(self.device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(self.config.epochs):
            # Train
            model.train()
            train_loss = 0.0

            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(features)

                loss = criterion(outputs["hack_prob"], labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validate
            if val_loader:
                model.eval()
                val_loss = 0.0
                val_preds = []
                val_labels = []

                with torch.no_grad():
                    for features, labels in val_loader:
                        features = features.to(self.device)
                        labels = labels.to(self.device)

                        outputs = model(features)
                        loss = criterion(outputs["hack_prob"], labels)

                        val_loss += loss.item()
                        val_preds.extend((outputs["hack_prob"] > 0.5).cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())

                val_loss /= len(val_loader)
                val_acc = accuracy_score(val_labels, val_preds)

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    model_path = Path(self.config.output_dir) / f"{self.config.model_name}.pt"
                    torch.save(model.state_dict(), model_path)
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    print(
                        f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.2%}"
                    )

        # Load best model
        model_path = Path(self.config.output_dir) / f"{self.config.model_name}.pt"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Final evaluation
        results = self._evaluate_pytorch(model, val_loader if val_loader else train_loader)
        results["history"] = history

        return results

    def train_sklearn(self) -> dict[str, Any]:
        """Train sklearn baseline model."""
        print(f"Training sklearn {self.config.model_type} model...")

        # Load data
        if self.config.use_combined:
            print("Using combined RHW + SHADE-Arena data")
            train_dataset = CombinedDataset(
                self.config.data_dir,
                shade_limit=self.config.shade_limit,
                split="train",
            )
            test_dataset = CombinedDataset(
                self.config.data_dir,
                shade_limit=self.config.shade_limit,
                split="test",
            )
            source_counts = train_dataset.get_source_counts()
            print(f"Training data sources: {source_counts}")
        else:
            train_dataset = RHWBenchDataset(self.config.data_dir, split="train")
            test_dataset = RHWBenchDataset(self.config.data_dir, split="test")

            if Path(self.config.trajectories_dir).exists():
                traj_dataset = RHWBenchDataset(self.config.trajectories_dir, split="all")
                train_dataset.trajectories.extend(traj_dataset.trajectories)
                train_dataset.labels.extend(traj_dataset.labels)

        X_train, y_train = train_dataset.get_all_features()
        X_test, y_test = test_dataset.get_all_features()

        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        if len(X_train) == 0:
            return {"error": "No training data"}

        # Create model
        if self.config.model_type == "sklearn_rf":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.config.model_type == "sklearn_gb":
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:  # sklearn_lr
            model = LogisticRegression(random_state=42, max_iter=1000)

        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "train_time": train_time,
            "model_type": self.config.model_type,
        }

        if len(np.unique(y_test)) > 1:
            results["auc"] = roc_auc_score(y_test, y_prob)

        # Save model
        import joblib

        model_path = (
            Path(self.config.output_dir) / f"{self.config.model_name}_{self.config.model_type}.pkl"
        )
        joblib.dump(model, model_path)

        print(f"Results: {results}")
        return results

    def _evaluate_pytorch(self, model: nn.Module, data_loader: DataLoader) -> dict[str, Any]:
        """Evaluate PyTorch model."""
        model.eval()

        all_preds = []
        all_probs = []
        all_labels = []
        latencies = []

        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)

                start = time.time()
                outputs = model(features)
                latencies.append((time.time() - start) * 1000)  # ms

                probs = outputs["hack_prob"].cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        results = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
            "avg_latency_ms": np.mean(latencies),
            "model_type": self.config.model_type,
        }

        if len(np.unique(all_labels)) > 1:
            results["auc"] = roc_auc_score(all_labels, all_probs)

        cm = confusion_matrix(all_labels, all_preds)
        results["confusion_matrix"] = cm.tolist()

        return results

    def train(self) -> dict[str, Any]:
        """Train model based on config."""
        if self.config.model_type.startswith("sklearn"):
            return self.train_sklearn()
        else:
            return self.train_pytorch()


def run_all_experiments(config: TrainingConfig = None) -> dict[str, Any]:
    """Run all training experiments for comparison."""
    if config is None:
        config = TrainingConfig()

    results = {}

    # PyTorch models
    for model_type in ["mlp", "feature"]:
        print(f"\n{'=' * 60}")
        print(f"Training {model_type} model")
        print(f"{'=' * 60}")

        config.model_type = model_type
        config.model_name = f"hack_classifier_{model_type}"

        trainer = Trainer(config)
        results[model_type] = trainer.train()

    # Sklearn baselines
    for model_type in ["sklearn_lr", "sklearn_rf", "sklearn_gb"]:
        print(f"\n{'=' * 60}")
        print(f"Training {model_type} model")
        print(f"{'=' * 60}")

        config.model_type = model_type
        config.model_name = f"hack_classifier_{model_type}"

        trainer = Trainer(config)
        results[model_type] = trainer.train()

    return results


if __name__ == "__main__":
    config = TrainingConfig()
    results = run_all_experiments(config)

    # Save results
    output_path = Path(config.output_dir) / "training_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
