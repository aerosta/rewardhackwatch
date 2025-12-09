"""
RHW-Bench: RewardHackWatch Benchmark Suite.

Our internal benchmark for evaluating reward hacking → misalignment detection.
Not to be confused with CheatBench.com (external benchmark).

Includes:
- Synthetic test cases based on Anthropic's reward hacking research
- Real research trajectories from Denison et al. environments
- SHADE-Arena external benchmark (ScaleAI/mrt dataset)
- RMGI (Reward-Misalignment Generalization Index) computation
"""

from .benchmark_runner import BenchmarkMetrics, BenchmarkResult, BenchmarkRunner
from .shade_loader import ShadeArenaLoader, ShadeTrajectory

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkMetrics",
    "ShadeArenaLoader",
    "ShadeTrajectory",
]
