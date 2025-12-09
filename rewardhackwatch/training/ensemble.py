"""
Ensemble methods for reward hacking detection.

This module provides:
1. Training multiple models with different seeds
2. Averaging predictions (reduces variance)
3. Weighted ensemble voting
4. Uncertainty estimation via disagreement
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


@dataclass
class EnsembleConfig:
    """Configuration for ensemble training."""

    # Ensemble
    n_models: int = 5
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    # Model architecture
    model_type: str = "mlp"  # "mlp" or "distilbert"
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.3

    # Training
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 0.001

    # Data
    data_dir: str = "rewardhackwatch/rhw_bench/test_cases/expanded"

    # Output
    output_dir: str = "models/ensemble"


class EnsemblePredictor:
    """Ensemble of models for prediction with uncertainty."""

    def __init__(self, models: list[nn.Module], device: torch.device):
        self.models = models
        self.device = device

    def predict(
        self,
        features: torch.Tensor,
        return_uncertainty: bool = True,
    ) -> dict[str, Any]:
        """
        Make ensemble prediction.

        Returns:
            dict with:
                - hack_prob: mean probability
                - is_hack: binary prediction (majority vote)
                - uncertainty: std of predictions
                - individual_probs: list of per-model probs
        """
        features = features.to(self.device)

        individual_probs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(features)
                probs = output["hack_prob"].cpu().numpy()
                individual_probs.append(probs)

        individual_probs = np.array(individual_probs)

        # Mean prediction
        mean_prob = np.mean(individual_probs, axis=0)

        # Majority vote
        votes = (individual_probs > 0.5).sum(axis=0)
        is_hack = votes > len(self.models) / 2

        result = {
            "hack_prob": mean_prob,
            "is_hack": is_hack,
            "individual_probs": individual_probs.tolist(),
        }

        if return_uncertainty:
            # Uncertainty = std across models
            result["uncertainty"] = np.std(individual_probs, axis=0)

            # Agreement = fraction of models that agree with majority
            agreement = np.maximum(votes, len(self.models) - votes) / len(self.models)
            result["agreement"] = agreement

        return result

    def predict_with_rejection(
        self,
        features: torch.Tensor,
        uncertainty_threshold: float = 0.2,
    ) -> dict[str, Any]:
        """
        Predict with option to reject uncertain cases.

        Args:
            features: Input features
            uncertainty_threshold: Reject if uncertainty > threshold

        Returns:
            dict with prediction and rejection status
        """
        result = self.predict(features, return_uncertainty=True)

        # Reject high-uncertainty predictions
        result["rejected"] = result["uncertainty"] > uncertainty_threshold
        result["rejection_rate"] = np.mean(result["rejected"])

        return result


class EnsembleTrainer:
    """Trainer for ensemble of models."""

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.device = self._get_device()
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _get_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def train(self) -> dict[str, Any]:
        """Train ensemble of models."""
        from .dataset import RHWBenchDataset
        from .models import MLPClassifier

        print(f"Training ensemble of {self.config.n_models} models on {self.device}")

        # Load data (same for all models)
        test_dataset = RHWBenchDataset(self.config.data_dir, split="test")
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)

        all_results = []
        models = []

        for i, seed in enumerate(self.config.seeds[: self.config.n_models]):
            print(f"\n{'=' * 60}")
            print(f"Training model {i + 1}/{self.config.n_models} (seed={seed})")
            print(f"{'=' * 60}")

            # Set seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Create model
            model = MLPClassifier(
                hidden_dims=self.config.hidden_dims,
                dropout=self.config.dropout,
            ).to(self.device)

            # Train
            result = self._train_single_model(model, seed)
            all_results.append(result)

            # Save model
            model_path = Path(self.config.output_dir) / f"model_seed_{seed}.pt"
            torch.save(model.state_dict(), model_path)
            models.append(model)

        # Create ensemble
        ensemble = EnsemblePredictor(models, self.device)

        # Evaluate ensemble
        ensemble_metrics = self._evaluate_ensemble(ensemble, test_loader)

        # Report results
        individual_f1s = [r.get("f1", 0) for r in all_results]
        print(f"\n{'=' * 60}")
        print("ENSEMBLE RESULTS")
        print(f"{'=' * 60}")
        print(f"Individual F1 scores: {individual_f1s}")
        print(
            f"Individual F1 mean ± std: {np.mean(individual_f1s):.3f} ± {np.std(individual_f1s):.3f}"
        )
        print(f"Ensemble F1: {ensemble_metrics['f1']:.3f}")

        final_results = {
            "individual_results": all_results,
            "individual_f1_mean": np.mean(individual_f1s),
            "individual_f1_std": np.std(individual_f1s),
            "ensemble_metrics": ensemble_metrics,
        }

        # Save results
        results_path = Path(self.config.output_dir) / "ensemble_results.json"
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2, default=str)

        return final_results

    def _train_single_model(self, model: nn.Module, seed: int) -> dict[str, Any]:
        """Train a single model."""
        from .dataset import RHWBenchDataset

        # Load data with seed for reproducibility
        np.random.seed(seed)
        train_dataset = RHWBenchDataset(self.config.data_dir, split="train")
        val_dataset = RHWBenchDataset(self.config.data_dir, split="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
        )

        # Training
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
        )

        best_val_f1 = 0.0
        best_state = None

        for epoch in range(self.config.epochs):
            # Train
            model.train()
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs["hack_prob"], labels)
                loss.backward()
                optimizer.step()

            # Validate
            val_metrics = self._evaluate_single(model, val_loader)

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}: val_f1={val_metrics['f1']:.3f}")

        # Load best state
        if best_state:
            model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return {"seed": seed, "f1": best_val_f1}

    def _evaluate_single(self, model: nn.Module, data_loader: DataLoader) -> dict[str, float]:
        """Evaluate single model."""
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                outputs = model(features)
                preds = (outputs["hack_prob"] > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        return {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
        }

    def _evaluate_ensemble(
        self,
        ensemble: EnsemblePredictor,
        data_loader: DataLoader,
    ) -> dict[str, Any]:
        """Evaluate ensemble on data."""
        all_preds = []
        all_probs = []
        all_labels = []
        all_uncertainties = []

        for features, labels in data_loader:
            result = ensemble.predict(features, return_uncertainty=True)

            all_preds.extend(result["is_hack"])
            all_probs.extend(result["hack_prob"])
            all_labels.extend(labels.numpy())
            all_uncertainties.extend(result["uncertainty"])

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_uncertainties = np.array(all_uncertainties)

        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
            "avg_uncertainty": float(np.mean(all_uncertainties)),
        }

        if len(np.unique(all_labels)) > 1:
            metrics["auc"] = roc_auc_score(all_labels, all_probs)

        return metrics


if __name__ == "__main__":
    config = EnsembleConfig()
    trainer = EnsembleTrainer(config)
    results = trainer.train()
