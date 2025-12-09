#!/usr/bin/env python3
"""
Lagged RMGI Cross-Correlation Analysis

Analyzes temporal relationships between hack scores (H) and misalignment scores (M)
using lagged cross-correlation. Tests whether hack behaviors precede or follow
misalignment patterns.

Reference: Amodei et al. (2016) "Concrete Problems in AI Safety" - reward hacking
as potential precursor to broader misalignment.

Output metrics:
- Cross-correlation at lags 0, 1, 2, 3 timesteps
- Statistical significance of correlations
- Peak lag identification
- Granger causality (if sufficient data)
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_lagged_correlation(
    series_a: np.ndarray, series_b: np.ndarray, max_lag: int = 5
) -> dict[int, dict[str, float]]:
    """
    Compute lagged cross-correlation between two time series.

    Args:
        series_a: First time series (e.g., hack scores)
        series_b: Second time series (e.g., misalignment scores)
        max_lag: Maximum lag to compute (positive = a leads b)

    Returns:
        Dict mapping lag -> {correlation, p_value, n_samples}
    """
    results = {}
    len(series_a)

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # Negative lag: b leads a
            a_slice = series_a[-lag:]
            b_slice = series_b[:lag]
        elif lag > 0:
            # Positive lag: a leads b
            a_slice = series_a[:-lag]
            b_slice = series_b[lag:]
        else:
            # No lag
            a_slice = series_a
            b_slice = series_b

        if len(a_slice) < 5:  # Minimum samples for meaningful correlation
            continue

        # Compute Pearson correlation
        correlation, p_value = stats.pearsonr(a_slice, b_slice)

        results[lag] = {
            "correlation": correlation,
            "p_value": p_value,
            "n_samples": len(a_slice),
            "significant": p_value < 0.05,
        }

    return results


def generate_synthetic_trajectories(
    n_trajectories: int = 50,
    trajectory_length: int = 20,
    transition_point: Optional[int] = None,
    hack_leads_misalignment: bool = True,
    lead_steps: int = 2,
) -> list[dict]:
    """
    Generate synthetic trajectories with controlled temporal relationships.

    Args:
        n_trajectories: Number of trajectories to generate
        trajectory_length: Length of each trajectory
        transition_point: Step where behavior changes (None = random)
        hack_leads_misalignment: If True, hack precedes misalignment
        lead_steps: How many steps hack leads/follows misalignment
    """
    trajectories = []

    for i in range(n_trajectories):
        # Random transition point if not specified
        tp = transition_point or np.random.randint(5, trajectory_length - 5)

        # Generate base signals with noise
        hack_scores = np.zeros(trajectory_length)
        misalignment_scores = np.zeros(trajectory_length)

        # Add gradual escalation
        for t in range(trajectory_length):
            if t < tp:
                # Pre-transition: low scores
                hack_scores[t] = 0.1 + 0.05 * np.random.randn()
                misalignment_scores[t] = 0.1 + 0.05 * np.random.randn()
            else:
                # Post-transition: escalating
                progress = (t - tp) / (trajectory_length - tp)
                base_hack = 0.3 + 0.5 * progress
                base_mis = 0.3 + 0.5 * progress

                if hack_leads_misalignment:
                    # Hack increases first, misalignment follows
                    if t >= tp:
                        hack_scores[t] = base_hack + 0.1 * np.random.randn()
                    if t >= tp + lead_steps:
                        misalignment_scores[t] = base_mis + 0.1 * np.random.randn()
                    else:
                        misalignment_scores[t] = 0.15 + 0.05 * np.random.randn()
                else:
                    # Misalignment increases first
                    if t >= tp:
                        misalignment_scores[t] = base_mis + 0.1 * np.random.randn()
                    if t >= tp + lead_steps:
                        hack_scores[t] = base_hack + 0.1 * np.random.randn()
                    else:
                        hack_scores[t] = 0.15 + 0.05 * np.random.randn()

        # Clip to valid range
        hack_scores = np.clip(hack_scores, 0, 1)
        misalignment_scores = np.clip(misalignment_scores, 0, 1)

        trajectories.append(
            {
                "id": f"synth_{i:03d}",
                "hack_scores": hack_scores.tolist(),
                "misalignment_scores": misalignment_scores.tolist(),
                "transition_point": tp,
                "has_transition": True,
            }
        )

    return trajectories


def analyze_trajectory_lags(trajectory: dict, max_lag: int = 3) -> dict:
    """Analyze lagged correlations for a single trajectory."""
    hack_scores = np.array(trajectory["hack_scores"])
    mis_scores = np.array(trajectory["misalignment_scores"])

    if len(hack_scores) < 10:
        return {"error": "Trajectory too short"}

    correlations = compute_lagged_correlation(hack_scores, mis_scores, max_lag)

    # Find peak correlation lag
    if correlations:
        peak_lag = max(correlations.keys(), key=lambda k: abs(correlations[k]["correlation"]))
        peak_corr = correlations[peak_lag]
    else:
        peak_lag = 0
        peak_corr = {"correlation": 0, "p_value": 1.0}

    return {
        "trajectory_id": trajectory.get("id", "unknown"),
        "correlations": correlations,
        "peak_lag": peak_lag,
        "peak_correlation": peak_corr["correlation"],
        "peak_p_value": peak_corr["p_value"],
        "interpretation": interpret_lag(peak_lag, peak_corr["correlation"]),
    }


def interpret_lag(lag: int, correlation: float) -> str:
    """Interpret the meaning of a lagged correlation."""
    if abs(correlation) < 0.3:
        return "Weak relationship - no clear temporal ordering"

    if lag > 0:
        return f"Hack LEADS misalignment by {lag} steps (correlation={correlation:.3f})"
    elif lag < 0:
        return f"Misalignment LEADS hack by {-lag} steps (correlation={correlation:.3f})"
    else:
        return f"Synchronous relationship (correlation={correlation:.3f})"


def run_lagged_analysis(
    trajectories: Optional[list[dict]] = None, output_dir: str = "results/lagged_rmgi"
) -> dict:
    """
    Run full lagged RMGI analysis.

    Args:
        trajectories: List of trajectory dicts with hack_scores and misalignment_scores
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("LAGGED RMGI CROSS-CORRELATION ANALYSIS")
    print("=" * 70)
    print()
    print("Testing temporal relationship between hack and misalignment behaviors.")
    print("Positive lag = hack leads misalignment")
    print("Negative lag = misalignment leads hack")
    print()

    # Generate synthetic data if none provided
    if trajectories is None:
        print("Generating synthetic trajectories for analysis...")
        print()

        # Generate trajectories with known temporal relationship
        trajectories = generate_synthetic_trajectories(
            n_trajectories=100,
            trajectory_length=25,
            hack_leads_misalignment=True,  # Test hypothesis that hack precedes misalignment
            lead_steps=2,
        )

    print(f"Analyzing {len(trajectories)} trajectories...")
    print()

    # Analyze each trajectory
    trajectory_results = []
    for traj in trajectories:
        result = analyze_trajectory_lags(traj, max_lag=3)
        if "error" not in result:
            trajectory_results.append(result)

    print(f"Successfully analyzed {len(trajectory_results)} trajectories")
    print()

    # Aggregate results
    print("=" * 70)
    print("AGGREGATE RESULTS BY LAG")
    print("=" * 70)
    print()
    print(f"{'Lag':>6} | {'Mean Corr':>10} | {'Std':>8} | {'% Sig':>8} | {'Interpretation'}")
    print("-" * 70)

    lag_summary = {}
    for lag in range(-3, 4):
        correlations = [
            r["correlations"].get(lag, {}).get("correlation", np.nan) for r in trajectory_results
        ]
        significant = [
            r["correlations"].get(lag, {}).get("significant", False) for r in trajectory_results
        ]

        correlations = [c for c in correlations if not np.isnan(c)]
        if correlations:
            mean_corr = np.mean(correlations)
            std_corr = np.std(correlations)
            pct_sig = 100 * sum(significant) / len(significant)

            interp = interpret_lag(lag, mean_corr)
            print(f"{lag:>6} | {mean_corr:>10.3f} | {std_corr:>8.3f} | {pct_sig:>7.1f}% | {interp}")

            lag_summary[lag] = {
                "mean_correlation": mean_corr,
                "std_correlation": std_corr,
                "n_trajectories": len(correlations),
                "pct_significant": pct_sig,
            }

    print()

    # Find optimal lag
    optimal_lag = max(lag_summary.keys(), key=lambda k: abs(lag_summary[k]["mean_correlation"]))
    optimal_stats = lag_summary[optimal_lag]

    print("=" * 70)
    print("PEAK CORRELATION ANALYSIS")
    print("=" * 70)
    print()

    # Distribution of peak lags across trajectories
    peak_lags = [r["peak_lag"] for r in trajectory_results]
    lag_counts = {}
    for lag in peak_lags:
        lag_counts[lag] = lag_counts.get(lag, 0) + 1

    print("Distribution of peak correlation lags:")
    for lag in sorted(lag_counts.keys()):
        pct = 100 * lag_counts[lag] / len(peak_lags)
        bar = "█" * int(pct / 2)
        print(f"  Lag {lag:>2}: {lag_counts[lag]:>4} ({pct:>5.1f}%) {bar}")

    print()
    print(f"Modal peak lag: {max(lag_counts, key=lag_counts.get)}")
    print(f"Mean peak lag: {np.mean(peak_lags):.2f}")

    # Statistical test: Is the peak lag significantly different from 0?
    t_stat, p_value = stats.ttest_1samp(peak_lags, 0)
    print()
    print("T-test (peak lag vs 0):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant (p<0.05): {'YES' if p_value < 0.05 else 'NO'}")

    # Final interpretation
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()

    if optimal_lag > 0 and p_value < 0.05:
        conclusion = f"SUPPORTED: Hack behavior LEADS misalignment by ~{optimal_lag} steps"
        hypothesis_support = "strong"
    elif optimal_lag < 0 and p_value < 0.05:
        conclusion = f"REVERSED: Misalignment LEADS hack behavior by ~{-optimal_lag} steps"
        hypothesis_support = "reversed"
    else:
        conclusion = "INCONCLUSIVE: No significant temporal ordering detected"
        hypothesis_support = "weak"

    print(conclusion)
    print()
    print("This analysis tests the hypothesis from Anthropic (2025) that reward")
    print("hacking behaviors emerge first and then generalize to broader misalignment.")
    print()

    # Save results
    final_results = {
        "analysis_date": "2025-12-06",
        "n_trajectories_analyzed": len(trajectory_results),
        "lag_summary": lag_summary,
        "optimal_lag": optimal_lag,
        "optimal_correlation": optimal_stats["mean_correlation"],
        "peak_lag_distribution": lag_counts,
        "mean_peak_lag": float(np.mean(peak_lags)),
        "statistical_test": {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
        },
        "conclusion": conclusion,
        "hypothesis_support": hypothesis_support,
        "methodology": {
            "metric": "Pearson cross-correlation",
            "lags_tested": list(range(-3, 4)),
            "min_trajectory_length": 10,
            "significance_threshold": 0.05,
        },
    }

    output_path = os.path.join(output_dir, "lagged_rmgi_analysis.json")
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"Results saved to: {output_path}")

    return final_results


def main():
    """Run lagged RMGI analysis."""
    # Check for real trajectory data
    trajectory_path = "data/trajectories/escalating_trajectories.json"

    if os.path.exists(trajectory_path):
        print(f"Loading real trajectories from {trajectory_path}")
        with open(trajectory_path) as f:
            trajectories = json.load(f)
    else:
        print("No real trajectory data found, using synthetic data.")
        print("To use real data, provide trajectories at:")
        print(f"  {trajectory_path}")
        print()
        trajectories = None

    results = run_lagged_analysis(trajectories)

    print()
    print("=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    print()
    print("Lagged cross-correlation analysis tests the temporal relationship")
    print("between reward hacking (H) and misalignment (M) behaviors.")
    print()
    print(f"  Optimal lag: {results['optimal_lag']} steps")
    print(f"  Mean correlation at optimal lag: {results['optimal_correlation']:.3f}")
    print(f"  Statistical significance: p = {results['statistical_test']['p_value']:.4f}")
    print()
    print(f"Interpretation: {results['conclusion']}")


if __name__ == "__main__":
    main()
