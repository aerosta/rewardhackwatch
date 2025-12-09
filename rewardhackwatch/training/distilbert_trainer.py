"""
DistilBERT-based classifier for reward hacking detection.

This module provides:
1. Text embedding using DistilBERT tokenizer
2. Fine-tuning DistilBERT on trajectory data
3. Hyperparameter search utilities
4. Checkpoint saving and loading
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import (
        DistilBertConfig,
        DistilBertModel,
        DistilBertTokenizer,
        get_linear_schedule_with_warmup,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Run: pip install transformers")


@dataclass
class DistilBertTrainingConfig:
    """Configuration for DistilBERT training."""

    # Model
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    hidden_size: int = 256
    dropout: float = 0.2
    freeze_bert: bool = False  # If True, only train classifier head

    # Training
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0

    # Early stopping
    early_stopping_patience: int = 5
    min_delta: float = 0.001

    # Output
    output_dir: str = "models/distilbert"
    checkpoint_every: int = 1  # Save checkpoint every N epochs
    model_name_prefix: str = "rhw_distilbert"

    # Data
    data_dir: str = "rewardhackwatch/rhw_bench/test_cases/expanded"
    use_shade: bool = False
    shade_ratio: float = 0.3  # Ratio of SHADE-Arena data to use


class TrajectoryTextDataset(Dataset):
    """Dataset that converts trajectories to text for BERT."""

    def __init__(
        self,
        trajectories: list[dict],
        labels: list[int],
        tokenizer,
        max_length: int = 512,
    ):
        self.trajectories = trajectories
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.trajectories)

    def _trajectory_to_text(self, trajectory: dict) -> str:
        """Convert trajectory to text for BERT input."""
        parts = []

        # Add task description
        if "task" in trajectory:
            parts.append(f"Task: {trajectory['task']}")

        # Add CoT traces
        if "cot_traces" in trajectory:
            cot_text = " ".join(str(t) for t in trajectory["cot_traces"])
            parts.append(f"Reasoning: {cot_text}")

        # Add code
        if "code_outputs" in trajectory:
            code_text = " ".join(str(c) for c in trajectory["code_outputs"])
            parts.append(f"Code: {code_text}")

        # Add steps
        if "steps" in trajectory:
            step_texts = []
            for step in trajectory["steps"]:
                if isinstance(step, dict):
                    if "thought" in step:
                        step_texts.append(str(step["thought"]))
                    if "action" in step:
                        step_texts.append(str(step["action"]))
            if step_texts:
                parts.append(f"Steps: {' '.join(step_texts)}")

        return " [SEP] ".join(parts)

    def __getitem__(self, idx: int) -> dict:
        trajectory = self.trajectories[idx]
        label = self.labels[idx]

        # Convert to text
        text = self._trajectory_to_text(trajectory)

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
        }


class DistilBertClassifier(nn.Module):
    """DistilBERT-based classifier for reward hacking detection."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        hidden_size: int = 256,
        dropout: float = 0.2,
        freeze_bert: bool = False,
    ):
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers package required")

        self.bert = DistilBertModel.from_pretrained(model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        bert_hidden = self.bert.config.hidden_size  # 768 for distilbert-base

        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        # Get BERT output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Classify
        hack_prob = self.classifier(cls_output).squeeze(-1)

        return {
            "hack_prob": hack_prob,
            "cls_embedding": cls_output,
        }


class DistilBertTrainer:
    """Trainer for DistilBERT reward hacking classifier."""

    def __init__(self, config: DistilBertTrainingConfig):
        self.config = config
        self.device = self._get_device()

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize tokenizer
        if HAS_TRANSFORMERS:
            self.tokenizer = DistilBertTokenizer.from_pretrained(config.model_name)
        else:
            self.tokenizer = None

    def _get_device(self) -> torch.device:
        """Get best available device."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def load_data(self) -> tuple[list[dict], list[int], list[dict], list[int]]:
        """Load training and validation data."""
        from .dataset import RHWBenchDataset

        # Load RHW-Bench data
        train_dataset = RHWBenchDataset(self.config.data_dir, split="train")
        val_dataset = RHWBenchDataset(self.config.data_dir, split="val")

        train_trajectories = train_dataset.trajectories
        train_labels = train_dataset.labels
        val_trajectories = val_dataset.trajectories
        val_labels = val_dataset.labels

        # Optionally add SHADE-Arena data
        if self.config.use_shade:
            try:
                from .dataset import ShadeArenaDataset

                shade_dataset = ShadeArenaDataset(limit=200, split="train")

                # Add fraction of SHADE data
                n_shade = int(len(shade_dataset) * self.config.shade_ratio)
                train_trajectories.extend(shade_dataset.trajectories[:n_shade])
                train_labels.extend(shade_dataset.labels[:n_shade])

                print(f"Added {n_shade} SHADE-Arena trajectories to training")
            except Exception as e:
                print(f"Could not load SHADE-Arena data: {e}")

        return train_trajectories, train_labels, val_trajectories, val_labels

    def create_dataloaders(
        self,
        train_trajectories: list[dict],
        train_labels: list[int],
        val_trajectories: list[dict],
        val_labels: list[int],
    ) -> tuple[DataLoader, DataLoader]:
        """Create DataLoaders for training."""
        train_dataset = TrajectoryTextDataset(
            train_trajectories,
            train_labels,
            self.tokenizer,
            self.config.max_length,
        )
        val_dataset = TrajectoryTextDataset(
            val_trajectories,
            val_labels,
            self.tokenizer,
            self.config.max_length,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        return train_loader, val_loader

    def train(self) -> dict[str, Any]:
        """Train the model."""
        if not HAS_TRANSFORMERS:
            return {"error": "transformers package not installed"}

        print(f"Training DistilBERT classifier on {self.device}")

        # Load data
        train_traj, train_labels, val_traj, val_labels = self.load_data()
        print(f"Train: {len(train_traj)}, Val: {len(val_traj)}")

        if len(train_traj) == 0:
            return {"error": "No training data found"}

        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(
            train_traj, train_labels, val_traj, val_labels
        )

        # Create model
        model = DistilBertClassifier(
            model_name=self.config.model_name,
            hidden_size=self.config.hidden_size,
            dropout=self.config.dropout,
            freeze_bert=self.config.freeze_bert,
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_f1": []}
        best_val_f1 = 0.0
        patience_counter = 0
        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Train
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs["hack_prob"], labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validate
            val_metrics = self._evaluate(model, val_loader, criterion)
            history["val_loss"].append(val_metrics["loss"])
            history["val_f1"].append(val_metrics["f1"])

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_f1={val_metrics['f1']:.3f} "
                f"({epoch_time:.1f}s)"
            )

            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self._save_checkpoint(model, optimizer, epoch, val_metrics)

            # Early stopping
            if val_metrics["f1"] > best_val_f1 + self.config.min_delta:
                best_val_f1 = val_metrics["f1"]
                patience_counter = 0
                self._save_best_model(model)
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        best_model_path = Path(self.config.output_dir) / f"{self.config.model_name_prefix}_best.pt"
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        # Final evaluation
        final_metrics = self._evaluate(model, val_loader, criterion)
        final_metrics["training_time"] = time.time() - start_time
        final_metrics["history"] = history
        final_metrics["best_val_f1"] = best_val_f1

        # Save final results
        results_path = Path(self.config.output_dir) / "training_results.json"
        with open(results_path, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in final_metrics.items()
            }
            json.dump(serializable_metrics, f, indent=2)

        return final_metrics

    def _evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
    ) -> dict[str, Any]:
        """Evaluate model on data."""
        model.eval()

        all_preds = []
        all_probs = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs["hack_prob"], labels)

                total_loss += loss.item()

                probs = outputs["hack_prob"].cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
        }

        if len(np.unique(all_labels)) > 1:
            metrics["auc"] = roc_auc_score(all_labels, all_probs)

        metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()

        return metrics

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
    ):
        """Save training checkpoint."""
        checkpoint_path = Path(self.config.output_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            },
            checkpoint_path,
        )

    def _save_best_model(self, model: nn.Module):
        """Save best model."""
        model_path = Path(self.config.output_dir) / f"{self.config.model_name_prefix}_best.pt"
        torch.save(model.state_dict(), model_path)


def run_hyperparameter_search(
    data_dir: str,
    output_dir: str = "models/distilbert_search",
    n_trials: int = 12,
) -> dict[str, Any]:
    """Run hyperparameter search."""
    if not HAS_TRANSFORMERS:
        return {"error": "transformers package not installed"}

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Hyperparameter grid
    learning_rates = [1e-5, 2e-5, 5e-5, 1e-4]
    hidden_sizes = [128, 256, 512]
    dropouts = [0.1, 0.2, 0.3]

    results = []
    best_f1 = 0.0
    best_config = None

    trial = 0
    for lr in learning_rates:
        for hidden in hidden_sizes:
            for dropout in dropouts:
                if trial >= n_trials:
                    break

                trial += 1
                print(f"\n{'=' * 60}")
                print(f"Trial {trial}/{n_trials}")
                print(f"lr={lr}, hidden={hidden}, dropout={dropout}")
                print(f"{'=' * 60}")

                config = DistilBertTrainingConfig(
                    data_dir=data_dir,
                    output_dir=f"{output_dir}/trial_{trial}",
                    learning_rate=lr,
                    hidden_size=hidden,
                    dropout=dropout,
                    epochs=10,  # Shorter for search
                    batch_size=8,
                )

                trainer = DistilBertTrainer(config)
                try:
                    metrics = trainer.train()

                    trial_result = {
                        "trial": trial,
                        "learning_rate": lr,
                        "hidden_size": hidden,
                        "dropout": dropout,
                        "val_f1": metrics.get("f1", 0),
                        "val_accuracy": metrics.get("accuracy", 0),
                        "training_time": metrics.get("training_time", 0),
                    }
                    results.append(trial_result)

                    if metrics.get("f1", 0) > best_f1:
                        best_f1 = metrics["f1"]
                        best_config = config

                except Exception as e:
                    print(f"Trial {trial} failed: {e}")
                    results.append({"trial": trial, "error": str(e)})

    # Save search results
    search_results = {
        "trials": results,
        "best_f1": best_f1,
        "best_config": {
            "learning_rate": best_config.learning_rate if best_config else None,
            "hidden_size": best_config.hidden_size if best_config else None,
            "dropout": best_config.dropout if best_config else None,
        }
        if best_config
        else None,
    }

    results_path = Path(output_dir) / "search_results.json"
    with open(results_path, "w") as f:
        json.dump(search_results, f, indent=2)

    print(f"\nSearch complete! Best F1: {best_f1:.3f}")
    print(f"Results saved to: {results_path}")

    return search_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="rewardhackwatch/rhw_bench/test_cases/expanded")
    parser.add_argument("--output-dir", default="models/distilbert")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--search", action="store_true", help="Run hyperparameter search")
    args = parser.parse_args()

    if args.search:
        run_hyperparameter_search(args.data_dir, args.output_dir)
    else:
        config = DistilBertTrainingConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        trainer = DistilBertTrainer(config)
        results = trainer.train()
        print(f"\nFinal results: {results}")
