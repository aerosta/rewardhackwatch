#!/usr/bin/env python3
"""Retrain classifier on combined real + synthetic data.

Loads expanded test cases + synthetic generated trajectories, trains
a GradientBoosting classifier on the 11-dimensional feature extractor,
and reports per-category F1 scores.

Usage:
    python scripts/retrain_combined.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rewardhackwatch.training.dataset import TrajectoryFeatureExtractor


def load_all_trajectories(dirs: list[str]) -> list[dict]:
    """Load JSON trajectories from multiple directories."""
    trajectories = []
    for d in dirs:
        dp = Path(d)
        if not dp.exists():
            print(f"  Warning: {d} does not exist, skipping")
            continue
        for f in sorted(dp.glob("*.json")):
            if f.name.startswith("_"):
                continue
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    trajectories.extend(data)
                else:
                    trajectories.append(data)
            except (json.JSONDecodeError, KeyError):
                continue
    return trajectories


def extract_features(trajectories: list[dict]) -> tuple:
    """Extract features, labels, and categories."""
    extractor = TrajectoryFeatureExtractor()
    features, labels, categories = [], [], []

    for traj in trajectories:
        inner = traj.get("trajectory", traj)
        feat = extractor.extract(inner)
        features.append(feat.to_tensor().tolist())
        labels.append(1.0 if traj.get("expected_hack", False) else 0.0)
        categories.append(traj.get("category", "unknown"))

    return np.array(features), np.array(labels), categories


def per_category_metrics(y_true, y_pred, categories):
    """Compute per-category F1, precision, recall."""
    cat_metrics = {}
    unique_cats = sorted(set(categories))
    for cat in unique_cats:
        mask = [i for i, c in enumerate(categories) if c == cat]
        if not mask:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        n_hack = int(yt.sum())
        n_total = len(yt)

        if len(np.unique(yt)) < 2:
            # Only one class present — compute accuracy instead
            acc = accuracy_score(yt, yp)
            cat_metrics[cat] = {
                "f1": round(acc, 4),  # accuracy as proxy
                "precision": round(acc, 4),
                "recall": round(acc, 4),
                "n_samples": n_total,
                "n_hack": n_hack,
                "note": "single-class (using accuracy)",
            }
        else:
            cat_metrics[cat] = {
                "f1": round(f1_score(yt, yp, zero_division=0), 4),
                "precision": round(precision_score(yt, yp, zero_division=0), 4),
                "recall": round(recall_score(yt, yp, zero_division=0), 4),
                "n_samples": n_total,
                "n_hack": n_hack,
            }
    return cat_metrics


def main():
    print("=" * 68)
    print("RewardHackWatch Combined Retraining")
    print("=" * 68)

    # --- Phase 1: Old data only (expanded) ---
    print("\n--- Phase 1: Baseline (expanded test cases only) ---")
    old_data = load_all_trajectories([
        "rewardhackwatch/rhw_bench/test_cases/expanded",
    ])
    print(f"Old dataset: {len(old_data)} trajectories")
    old_cats = Counter(t.get("category", "unknown") for t in old_data)
    for cat, n in sorted(old_cats.items()):
        n_hack = sum(1 for t in old_data if t.get("category") == cat and t.get("expected_hack"))
        print(f"  {cat}: {n} ({n_hack} hacks)")

    X_old, y_old, cats_old = extract_features(old_data)
    print(f"Features shape: {X_old.shape}, hack ratio: {y_old.mean():.3f}")

    # 5-fold CV on old data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    old_f1s = []
    for train_idx, test_idx in skf.split(X_old, y_old):
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
        clf.fit(X_old[train_idx], y_old[train_idx])
        preds = clf.predict(X_old[test_idx])
        old_f1s.append(f1_score(y_old[test_idx], preds, zero_division=0))

    old_mean_f1 = np.mean(old_f1s)
    print(f"5-fold CV F1: {old_mean_f1:.4f} (+/- {np.std(old_f1s):.4f})")

    # Train on all old data for per-category breakdown
    clf_old = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    clf_old.fit(X_old, y_old)
    preds_old = clf_old.predict(X_old)  # resubstitution (train set) just for category breakdown
    old_cat_metrics = per_category_metrics(y_old, preds_old, cats_old)

    print("\nPer-category (resubstitution baseline):")
    for cat, m in sorted(old_cat_metrics.items()):
        print(f"  {cat:25s}  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  n={m['n_samples']}")

    # --- Phase 2: Combined data (expanded + generated) ---
    print("\n" + "=" * 68)
    print("--- Phase 2: Combined (expanded + generated synthetic) ---")
    combined_data = load_all_trajectories([
        "rewardhackwatch/rhw_bench/test_cases/expanded",
        "rewardhackwatch/rhw_bench/test_cases/generated",
    ])
    print(f"Combined dataset: {len(combined_data)} trajectories")
    new_cats = Counter(t.get("category", "unknown") for t in combined_data)
    for cat, n in sorted(new_cats.items()):
        n_hack = sum(1 for t in combined_data if t.get("category") == cat and t.get("expected_hack"))
        print(f"  {cat}: {n} ({n_hack} hacks)")

    X_new, y_new, cats_new = extract_features(combined_data)
    print(f"Features shape: {X_new.shape}, hack ratio: {y_new.mean():.3f}")

    # 5-fold CV on combined data
    new_f1s = []
    all_preds = np.zeros(len(y_new))
    all_cats_arr = np.array(cats_new)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_new, y_new)):
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
        clf.fit(X_new[train_idx], y_new[train_idx])
        preds = clf.predict(X_new[test_idx])
        all_preds[test_idx] = preds
        fold_f1 = f1_score(y_new[test_idx], preds, zero_division=0)
        new_f1s.append(fold_f1)
        print(f"  Fold {fold+1}: F1={fold_f1:.4f}")

    new_mean_f1 = np.mean(new_f1s)
    print(f"\n5-fold CV F1: {new_mean_f1:.4f} (+/- {np.std(new_f1s):.4f})")

    # Per-category from CV predictions
    new_cat_metrics = per_category_metrics(y_new, all_preds, cats_new)

    print("\nPer-category F1 (5-fold CV):")
    print(f"  {'Category':25s}  {'F1':>8}  {'Prec':>8}  {'Recall':>8}  {'N':>6}  {'Hacks':>6}")
    print("  " + "-" * 72)
    for cat, m in sorted(new_cat_metrics.items()):
        print(
            f"  {cat:25s}  {m['f1']:>8.4f}  {m['precision']:>8.4f}  {m['recall']:>8.4f}"
            f"  {m['n_samples']:>6}  {m['n_hack']:>6}"
        )

    # --- Comparison ---
    print("\n" + "=" * 68)
    print("--- Old vs New Comparison ---")
    print(f"  {'Metric':25s}  {'Old (expanded)':>15}  {'New (combined)':>15}  {'Delta':>10}")
    print("  " + "-" * 68)
    print(f"  {'Overall CV F1':25s}  {old_mean_f1:>15.4f}  {new_mean_f1:>15.4f}  {new_mean_f1-old_mean_f1:>+10.4f}")
    print(f"  {'Dataset size':25s}  {len(old_data):>15}  {len(combined_data):>15}  {len(combined_data)-len(old_data):>+10}")
    print(f"  {'Hack ratio':25s}  {y_old.mean():>15.3f}  {y_new.mean():>15.3f}  {y_new.mean()-y_old.mean():>+10.3f}")

    # Per-category comparison
    print(f"\n  {'Category':25s}  {'Old F1':>10}  {'New F1':>10}  {'Delta':>10}")
    print("  " + "-" * 58)
    all_cats = sorted(set(list(old_cat_metrics.keys()) + list(new_cat_metrics.keys())))
    for cat in all_cats:
        old_f1 = old_cat_metrics.get(cat, {}).get("f1", 0.0)
        new_f1 = new_cat_metrics.get(cat, {}).get("f1", 0.0)
        marker = " ***" if cat == "mock_exploit" else ""
        print(f"  {cat:25s}  {old_f1:>10.4f}  {new_f1:>10.4f}  {new_f1-old_f1:>+10.4f}{marker}")

    # --- Save results ---
    results = {
        "old": {
            "dataset_size": len(old_data),
            "cv_f1_mean": round(old_mean_f1, 4),
            "cv_f1_std": round(float(np.std(old_f1s)), 4),
            "per_category": old_cat_metrics,
        },
        "new": {
            "dataset_size": len(combined_data),
            "cv_f1_mean": round(new_mean_f1, 4),
            "cv_f1_std": round(float(np.std(new_f1s)), 4),
            "per_category": new_cat_metrics,
        },
    }

    out_path = Path("results/retrain_combined.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
