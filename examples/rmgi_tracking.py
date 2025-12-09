#!/usr/bin/env python3
"""
RMGI Tracking Example

Demonstrates the Reward-Misalignment Generalization Index (RMGI) tracking
to detect when reward hacking generalizes into misalignment.
"""

import random
from dataclasses import dataclass

from rewardhackwatch.core.trackers.rmgi_tracker import RMGITracker

from rewardhackwatch.core.trackers.changepoint_detector import ChangepointDetector


@dataclass
class SimulatedTrajectory:
    """A simulated trajectory for demonstration."""

    timestep: int
    hack_score: float
    misalignment_score: float


def simulate_normal_training(num_steps: int = 50) -> list[SimulatedTrajectory]:
    """
    Simulate normal training without hack-to-misalignment generalization.

    Hack scores and misalignment scores are uncorrelated.
    """
    trajectories = []
    for t in range(num_steps):
        # Random, uncorrelated scores
        hack = random.uniform(0.0, 0.3)
        misalignment = random.uniform(0.0, 0.2)
        trajectories.append(SimulatedTrajectory(t, hack, misalignment))
    return trajectories


def simulate_generalization(num_steps: int = 100) -> list[SimulatedTrajectory]:
    """
    Simulate training where reward hacking generalizes to misalignment.

    After a transition point (~step 40), hack and misalignment scores
    become highly correlated.
    """
    trajectories = []
    transition_point = 40

    for t in range(num_steps):
        if t < transition_point:
            # Phase 1: Normal training, low correlation
            hack = random.uniform(0.1, 0.4) + 0.01 * t  # Slowly increasing
            misalignment = random.uniform(0.0, 0.2)
        else:
            # Phase 2: Generalization, high correlation
            base = 0.4 + 0.005 * (t - transition_point)
            hack = min(0.95, base + random.uniform(-0.05, 0.05))
            # Misalignment now tracks hack score
            misalignment = min(0.9, hack * 0.8 + random.uniform(-0.1, 0.1))

        trajectories.append(SimulatedTrajectory(t, hack, misalignment))

    return trajectories


def track_rmgi(
    trajectories: list[SimulatedTrajectory], window_size: int = 10
) -> list[tuple[int, float]]:
    """
    Calculate RMGI over a sliding window.

    Args:
        trajectories: List of simulated trajectories
        window_size: Size of the sliding window

    Returns:
        List of (timestep, rmgi_value) tuples
    """
    tracker = RMGITracker(window_size=window_size)
    rmgi_values = []

    for traj in trajectories:
        tracker.update(traj.hack_score, traj.misalignment_score)
        rmgi = tracker.get_rmgi()
        if rmgi is not None:
            rmgi_values.append((traj.timestep, rmgi))

    return rmgi_values


def detect_transition(rmgi_values: list[tuple[int, float]], threshold: float = 0.7) -> int:
    """
    Detect the transition point where RMGI exceeds threshold.

    Args:
        rmgi_values: List of (timestep, rmgi) tuples
        threshold: RMGI threshold for transition detection

    Returns:
        Timestep of first transition, or -1 if none detected
    """
    consecutive_count = 0
    required_consecutive = 5

    for timestep, rmgi in rmgi_values:
        if rmgi > threshold:
            consecutive_count += 1
            if consecutive_count >= required_consecutive:
                # Return the timestep where the streak started
                return timestep - required_consecutive + 1
        else:
            consecutive_count = 0

    return -1


def main():
    """Demonstrate RMGI tracking and transition detection."""
    random.seed(42)  # For reproducibility

    print("=" * 60)
    print("RMGI TRACKING DEMONSTRATION")
    print("=" * 60)

    # Scenario 1: Normal training
    print("\n--- Scenario 1: Normal Training (No Generalization) ---")
    normal_trajs = simulate_normal_training(50)
    normal_rmgi = track_rmgi(normal_trajs)

    if normal_rmgi:
        avg_rmgi = sum(r[1] for r in normal_rmgi) / len(normal_rmgi)
        max_rmgi = max(r[1] for r in normal_rmgi)
        print(f"Average RMGI: {avg_rmgi:.3f}")
        print(f"Max RMGI: {max_rmgi:.3f}")

        transition = detect_transition(normal_rmgi)
        if transition == -1:
            print("✅ No transition detected (expected)")
        else:
            print(f"⚠️ Transition detected at step {transition}")

    # Scenario 2: Generalization
    print("\n--- Scenario 2: Hack→Misalignment Generalization ---")
    gen_trajs = simulate_generalization(100)
    gen_rmgi = track_rmgi(gen_trajs)

    if gen_rmgi:
        # Show RMGI progression
        print("\nRMGI Progression (every 10 steps):")
        for t, rmgi in gen_rmgi[::10]:
            bar = "█" * int(rmgi * 20)
            status = "🚨" if rmgi > 0.7 else "  "
            print(f"  Step {t:3d}: {rmgi:.3f} {bar} {status}")

        transition = detect_transition(gen_rmgi)
        if transition != -1:
            print(f"\n🚨 TRANSITION DETECTED at step {transition}")
            print("   Reward hacking is generalizing to misalignment!")
        else:
            print("\n✅ No transition detected")

    # Scenario 3: Using changepoint detector
    print("\n--- Scenario 3: Changepoint Detection ---")
    detector = ChangepointDetector(min_segment_length=5)

    # Feed RMGI values to changepoint detector
    rmgi_series = [r[1] for r in gen_rmgi]
    changepoints = detector.detect(rmgi_series)

    if changepoints:
        print(f"Changepoints detected at indices: {changepoints}")
        # Map back to timesteps
        for cp in changepoints:
            if cp < len(gen_rmgi):
                timestep = gen_rmgi[cp][0]
                print(f"  → Timestep {timestep}: RMGI = {gen_rmgi[cp][1]:.3f}")
    else:
        print("No changepoints detected")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
RMGI (Reward-Misalignment Generalization Index) measures the correlation
between hack scores and misalignment scores over a sliding window.

- RMGI < 0.3: Low correlation, hacks are isolated
- RMGI 0.3-0.7: Moderate correlation, potential concern
- RMGI > 0.7: High correlation, hacks generalizing to misalignment

When RMGI stays above 0.7 for 5+ consecutive steps, this indicates
that reward hacking behavior is systematically producing misalignment.
""")


if __name__ == "__main__":
    main()
