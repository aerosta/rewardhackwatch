"""Training module for RewardHackWatch ML models."""

from .dataset import RHWBenchDataset, TrajectoryFeatureExtractor
from .models import FeatureClassifier, MLPClassifier
from .train_detector import Trainer, TrainingConfig

__all__ = [
    "RHWBenchDataset",
    "TrajectoryFeatureExtractor",
    "FeatureClassifier",
    "MLPClassifier",
    "Trainer",
    "TrainingConfig",
]
