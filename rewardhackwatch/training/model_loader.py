"""Universal model loader that handles different architectures."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

try:
    from transformers import DistilBertModel, DistilBertTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class LegacyDistilBertClassifier(nn.Module):
    """
    Legacy architecture matching saved best_model.pt.

    Architecture:
    - self.bert: DistilBertModel
    - self.fc: Sequential(
        Linear(768, 256),   # fc.0
        ReLU(),             # fc.1
        Dropout(0.2),       # fc.2
        Linear(256, 2)      # fc.3
      )
    """

    def __init__(self, hidden_size: int = 256, dropout: float = 0.2):
        super().__init__()
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers package required")

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Match the saved model's fc architecture exactly
        self.fc = nn.Sequential(
            nn.Linear(768, hidden_size),  # fc.0
            nn.ReLU(),  # fc.1
            nn.Dropout(dropout),  # fc.2
            nn.Linear(hidden_size, 2),  # fc.3 - 2 classes
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for 2 classes."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.fc(pooled)
        return logits

    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
        """Get probability of being a hack (class 1)."""
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            # Class 0 = clean, Class 1 = hack
            return probs[0, 1].item()


class CurrentDistilBertClassifier(nn.Module):
    """Current architecture with classifier.* naming and BCE output."""

    def __init__(self, hidden_size: int = 256, dropout: float = 0.2):
        super().__init__()
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers package required")

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass returning single probability."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        return torch.sigmoid(logits)

    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
        """Get probability of being a hack."""
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)
            return output[0, 0].item()


def load_model(model_path: str, device: str = "cpu") -> nn.Module:
    """
    Load model from checkpoint, automatically detecting architecture.

    Args:
        model_path: Path to the saved model weights
        device: Device to load model on ('cpu', 'cuda', 'mps')

    Returns:
        A model with a predict_proba(input_ids, attention_mask) method
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Extract state dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Detect architecture by key naming
    has_fc = any("fc." in k and "bert" not in k for k in state_dict.keys())
    has_classifier = any("classifier." in k for k in state_dict.keys())

    if has_fc and not has_classifier:
        # Legacy architecture with fc.* naming
        print("Loading legacy model (fc.* naming, 2-class output)")

        # Detect hidden size from fc.0.weight shape
        fc_0_weight = state_dict.get("fc.0.weight")
        if fc_0_weight is not None:
            hidden_size = fc_0_weight.shape[0]
        else:
            hidden_size = 256

        model = LegacyDistilBertClassifier(hidden_size=hidden_size)
        model.load_state_dict(state_dict)

    elif has_classifier:
        # Current architecture with classifier.* naming
        print("Loading current model (classifier.* naming)")
        model = CurrentDistilBertClassifier()
        model.load_state_dict(state_dict, strict=False)

    else:
        raise ValueError(f"Unknown model architecture. Keys: {list(state_dict.keys())[:10]}")

    model.to(device)
    model.eval()
    return model


def get_tokenizer() -> DistilBertTokenizer:
    """Get the DistilBERT tokenizer."""
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers package required")
    return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def trajectory_to_text(trajectory: dict) -> str:
    """
    Convert a trajectory to text format for model input.

    The model was trained on raw agent transcripts, so this function
    concatenates all available text fields without special formatting.
    """
    # If there's already a "text" field, use it directly (matches training format)
    if "text" in trajectory and trajectory["text"]:
        return str(trajectory["text"])

    # Otherwise, concatenate all available content as raw text
    parts = []

    # Add task/prompt if present
    for key in ["task", "prompt", "instruction", "goal"]:
        if key in trajectory and trajectory[key]:
            parts.append(str(trajectory[key]))

    # Add CoT traces - raw concatenation
    cot_traces = trajectory.get("cot_traces", [])
    if isinstance(cot_traces, str):
        parts.append(cot_traces)
    elif cot_traces:
        parts.extend(str(t) for t in cot_traces if t)

    # Add reasoning/thought fields
    for key in ["reasoning", "thought", "thinking"]:
        if key in trajectory and trajectory[key]:
            parts.append(str(trajectory[key]))

    # Add code outputs - raw concatenation
    code_outputs = trajectory.get("code_outputs", [])
    if isinstance(code_outputs, str):
        parts.append(code_outputs)
    elif code_outputs:
        parts.extend(str(c) for c in code_outputs if c)

    # Add actions
    actions = trajectory.get("actions", [])
    if isinstance(actions, str):
        parts.append(actions)
    elif actions:
        parts.extend(str(a) for a in actions if a)

    # Add steps - extract all text content
    steps = trajectory.get("steps", [])
    if steps:
        for step in steps:
            if isinstance(step, dict):
                for key in ["thought", "action", "content", "output", "reasoning"]:
                    if key in step and step[key]:
                        parts.append(str(step[key]))
            elif isinstance(step, str):
                parts.append(step)

    # Add observations
    observations = trajectory.get("observations", [])
    if isinstance(observations, str):
        parts.append(observations)
    elif observations:
        parts.extend(str(o) for o in observations if o)

    # Concatenate with newlines (similar to agent transcript format)
    return "\n".join(parts) if parts else ""


class MLPredictor:
    """High-level predictor class for easy use."""

    def __init__(self, model_path: str = "models/best_model.pt", device: str = "cpu"):
        self.model = load_model(model_path, device)
        self.tokenizer = get_tokenizer()
        self.device = device

    def predict(self, trajectory: dict) -> float:
        """
        Predict hack probability for a trajectory.

        Args:
            trajectory: Trajectory dict with cot_traces, code_outputs, steps, etc.

        Returns:
            Probability of being a hack (0.0 - 1.0)
        """
        text = trajectory_to_text(trajectory)
        if not text.strip():
            return 0.0

        inputs = self.tokenizer(
            text, max_length=512, truncation=True, padding="max_length", return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return self.model.predict_proba(inputs["input_ids"], inputs["attention_mask"])

    def predict_text(self, text: str) -> float:
        """
        Predict hack probability for raw text.

        Args:
            text: Raw text to classify

        Returns:
            Probability of being a hack (0.0 - 1.0)
        """
        if not text.strip():
            return 0.0

        inputs = self.tokenizer(
            text, max_length=512, truncation=True, padding="max_length", return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return self.model.predict_proba(inputs["input_ids"], inputs["attention_mask"])
