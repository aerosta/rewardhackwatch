"""PyTorch model architectures for RewardHackWatch."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import TrajectoryFeatures


class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron classifier for trajectory features.

    Takes extracted features (code patterns, CoT metrics, etc.) and
    predicts hack probability and misalignment probability.
    """

    def __init__(
        self,
        input_dim: int = None,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
    ):
        """
        Initialize MLP classifier.

        Args:
            input_dim: Input feature dimension (auto-detected if None)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        if input_dim is None:
            input_dim = TrajectoryFeatures.feature_dim()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # Two output heads
        self.hack_head = nn.Linear(prev_dim, 1)
        self.misalign_head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Dictionary with 'hack_prob' and 'misalign_prob'
        """
        h = self.shared(x)

        hack_logits = self.hack_head(h)
        misalign_logits = self.misalign_head(h)

        return {
            "hack_prob": torch.sigmoid(hack_logits).squeeze(-1),
            "misalign_prob": torch.sigmoid(misalign_logits).squeeze(-1),
            "hack_logits": hack_logits.squeeze(-1),
            "misalign_logits": misalign_logits.squeeze(-1),
        }


class FeatureClassifier(nn.Module):
    """
    Simple feature-based classifier using rule-derived features.

    This is a lightweight alternative to the transformer-based MLDetector,
    suitable when compute resources are limited or for fast inference.
    """

    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        """
        Initialize feature classifier.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()

        if input_dim is None:
            input_dim = TrajectoryFeatures.feature_dim()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 2)  # [hack, misalign]

        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass."""
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.dropout(h)

        out = self.fc_out(h)
        probs = torch.sigmoid(out)

        return {
            "hack_prob": probs[:, 0],
            "misalign_prob": probs[:, 1],
            "logits": out,
        }


class AttentionClassifier(nn.Module):
    """
    Attention-based classifier for variable-length trajectory inputs.

    Uses self-attention to combine features from multiple steps/traces.
    """

    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize attention classifier.

        Args:
            input_dim: Per-step feature dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()

        if input_dim is None:
            input_dim = TrajectoryFeatures.feature_dim()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, seq_len, input_dim]
            mask: Optional attention mask [batch_size, seq_len]

        Returns:
            Dictionary with probabilities
        """
        # Project to hidden dim
        h = self.input_proj(x)

        # Apply transformer
        h = self.transformer(h, src_key_padding_mask=mask)

        # Pool: mean over sequence
        if mask is not None:
            # Mask out padded positions
            mask_expanded = mask.unsqueeze(-1).expand_as(h)
            h = h.masked_fill(mask_expanded, 0)
            lengths = (~mask).sum(dim=1, keepdim=True).float()
            h = h.sum(dim=1) / lengths
        else:
            h = h.mean(dim=1)

        # Classify
        out = self.classifier(h)
        probs = torch.sigmoid(out)

        return {
            "hack_prob": probs[:, 0],
            "misalign_prob": probs[:, 1],
            "logits": out,
        }


class EnsembleClassifier(nn.Module):
    """
    Ensemble of multiple classifiers.

    Combines predictions from MLP, feature classifier, and optionally
    attention-based classifier for more robust predictions.
    """

    def __init__(
        self,
        classifiers: list[nn.Module] = None,
        weights: list[float] = None,
    ):
        """
        Initialize ensemble.

        Args:
            classifiers: List of classifier modules
            weights: Weights for each classifier (uniform if None)
        """
        super().__init__()

        if classifiers is None:
            classifiers = [
                MLPClassifier(),
                FeatureClassifier(),
            ]

        self.classifiers = nn.ModuleList(classifiers)

        if weights is None:
            weights = [1.0 / len(classifiers)] * len(classifiers)

        self.register_buffer(
            "weights",
            torch.tensor(weights, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass with ensemble averaging."""
        hack_probs = []
        misalign_probs = []

        for classifier in self.classifiers:
            out = classifier(x)
            hack_probs.append(out["hack_prob"])
            misalign_probs.append(out["misalign_prob"])

        # Stack and weight
        hack_stack = torch.stack(hack_probs, dim=-1)
        misalign_stack = torch.stack(misalign_probs, dim=-1)

        # Weighted average
        hack_prob = (hack_stack * self.weights).sum(dim=-1)
        misalign_prob = (misalign_stack * self.weights).sum(dim=-1)

        return {
            "hack_prob": hack_prob,
            "misalign_prob": misalign_prob,
        }
