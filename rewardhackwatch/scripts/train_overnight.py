#!/usr/bin/env python3
"""
Overnight training script for RewardHackWatch.

This script runs a complete training pipeline:
1. Generate expanded trajectories (500+)
2. Run hyperparameter search
3. Train ensemble models
4. Save all results and checkpoints

Usage:
    # Full training (4-8 hours)
    nohup python -m rewardhackwatch.scripts.train_overnight &

    # Quick test (30 min)
    python -m rewardhackwatch.scripts.train_overnight --quick

    # Monitor progress
    tail -f results/training_logs/overnight_training.log
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path


# Setup logging
def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir = output_dir / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"overnight_training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger(__name__)


def generate_training_data(target_count: int, output_dir: Path, logger: logging.Logger):
    """Generate expanded training data."""
    logger.info(f"Generating {target_count} training trajectories...")
    start = time.time()

    try:
        from rewardhackwatch.scripts.generate_expanded_trajectories import main as generate_main

        trajectories = generate_main(
            target_count=target_count,
            output_dir=output_dir / "rhw_bench" / "test_cases" / "expanded",
        )

        elapsed = time.time() - start
        logger.info(f"Generated {len(trajectories)} trajectories in {elapsed:.1f}s")

        return len(trajectories)

    except Exception as e:
        logger.error(f"Failed to generate data: {e}")
        raise


def run_sklearn_baselines(data_dir: Path, output_dir: Path, logger: logging.Logger):
    """Run sklearn baseline models."""
    logger.info("Training sklearn baseline models...")
    start = time.time()

    try:
        from rewardhackwatch.training.train_detector import Trainer, TrainingConfig

        results = {}
        for model_type in ["sklearn_lr", "sklearn_rf", "sklearn_gb"]:
            logger.info(f"  Training {model_type}...")

            config = TrainingConfig(
                data_dir=str(data_dir),
                model_type=model_type,
                output_dir=str(output_dir / "sklearn"),
                model_name=f"baseline_{model_type}",
            )

            trainer = Trainer(config)
            result = trainer.train()
            results[model_type] = result

            logger.info(f"    {model_type}: F1={result.get('f1', 0):.3f}")

        elapsed = time.time() - start
        logger.info(f"Sklearn baselines completed in {elapsed:.1f}s")

        return results

    except Exception as e:
        logger.error(f"Sklearn training failed: {e}")
        return {"error": str(e)}


def run_pytorch_training(data_dir: Path, output_dir: Path, epochs: int, logger: logging.Logger):
    """Run PyTorch MLP training."""
    logger.info("Training PyTorch MLP models...")
    start = time.time()

    try:
        from rewardhackwatch.training.train_detector import Trainer, TrainingConfig

        results = {}
        for hidden_dims, name in [
            ([64, 32], "small"),
            ([128, 64], "medium"),
            ([256, 128, 64], "large"),
        ]:
            logger.info(f"  Training {name} MLP...")

            config = TrainingConfig(
                data_dir=str(data_dir),
                model_type="mlp",
                hidden_dims=hidden_dims,
                epochs=epochs,
                output_dir=str(output_dir / "mlp"),
                model_name=f"mlp_{name}",
            )

            trainer = Trainer(config)
            result = trainer.train()
            results[name] = result

            logger.info(f"    {name}: F1={result.get('f1', 0):.3f}")

        elapsed = time.time() - start
        logger.info(f"PyTorch training completed in {elapsed:.1f}s")

        return results

    except Exception as e:
        logger.error(f"PyTorch training failed: {e}")
        return {"error": str(e)}


def run_distilbert_training(
    data_dir: Path,
    output_dir: Path,
    epochs: int,
    do_search: bool,
    logger: logging.Logger,
):
    """Run DistilBERT training with optional hyperparameter search."""
    logger.info("Training DistilBERT classifier...")
    start = time.time()

    try:
        from rewardhackwatch.training.distilbert_trainer import (
            DistilBertTrainer,
            DistilBertTrainingConfig,
            run_hyperparameter_search,
        )

        results = {}

        if do_search:
            logger.info("  Running hyperparameter search...")
            search_results = run_hyperparameter_search(
                data_dir=str(data_dir),
                output_dir=str(output_dir / "distilbert_search"),
                n_trials=12,
            )
            results["search"] = search_results

            # Use best config from search
            best = search_results.get("best_config", {})
            lr = best.get("learning_rate", 2e-5)
            hidden = best.get("hidden_size", 256)
            dropout = best.get("dropout", 0.2)
        else:
            lr, hidden, dropout = 2e-5, 256, 0.2

        # Final training with best/default config
        logger.info(f"  Training final model (lr={lr}, hidden={hidden}, dropout={dropout})...")

        config = DistilBertTrainingConfig(
            data_dir=str(data_dir),
            output_dir=str(output_dir / "distilbert"),
            epochs=epochs,
            learning_rate=lr,
            hidden_size=hidden,
            dropout=dropout,
        )

        trainer = DistilBertTrainer(config)
        train_result = trainer.train()
        results["final"] = train_result

        elapsed = time.time() - start
        logger.info(f"DistilBERT training completed in {elapsed:.1f}s")
        logger.info(f"  Final F1: {train_result.get('f1', 0):.3f}")

        return results

    except ImportError as e:
        logger.warning(f"DistilBERT training skipped (missing transformers): {e}")
        return {"skipped": "transformers not installed"}
    except Exception as e:
        logger.error(f"DistilBERT training failed: {e}")
        return {"error": str(e)}


def run_ensemble_training(data_dir: Path, output_dir: Path, epochs: int, logger: logging.Logger):
    """Run ensemble model training."""
    logger.info("Training ensemble models...")
    start = time.time()

    try:
        from rewardhackwatch.training.ensemble import EnsembleConfig, EnsembleTrainer

        config = EnsembleConfig(
            n_models=5,
            epochs=epochs,
            data_dir=str(data_dir),
            output_dir=str(output_dir / "ensemble"),
        )

        trainer = EnsembleTrainer(config)
        results = trainer.train()

        elapsed = time.time() - start
        logger.info(f"Ensemble training completed in {elapsed:.1f}s")
        logger.info(f"  Ensemble F1: {results['ensemble_metrics'].get('f1', 0):.3f}")
        logger.info(
            f"  Individual F1: {results['individual_f1_mean']:.3f} ± {results['individual_f1_std']:.3f}"
        )

        return results

    except Exception as e:
        logger.error(f"Ensemble training failed: {e}")
        return {"error": str(e)}


def run_final_evaluation(
    data_dir: Path, models_dir: Path, output_dir: Path, logger: logging.Logger
):
    """Run final evaluation on test set."""
    logger.info("Running final evaluation on test set...")
    start = time.time()

    try:
        from rewardhackwatch.scripts.realistic_evaluation import (
            run_realistic_evaluation,
        )

        # Run evaluation
        results = run_realistic_evaluation(
            rhw_bench_dir=data_dir.parent,
            output_dir=output_dir,
        )

        elapsed = time.time() - start
        logger.info(f"Evaluation completed in {elapsed:.1f}s")

        return results

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Overnight training for RewardHackWatch")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run (fewer epochs, smaller data)",
    )
    parser.add_argument(
        "--data-count",
        type=int,
        default=500,
        help="Number of training trajectories to generate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs (default: 20)",
    )
    parser.add_argument(
        "--skip-distilbert",
        action="store_true",
        help="Skip DistilBERT training",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip hyperparameter search",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory",
    )
    args = parser.parse_args()

    # Quick mode adjustments
    if args.quick:
        args.data_count = 100
        args.epochs = 5
        args.skip_search = True

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("RewardHackWatch Overnight Training")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  Data count: {args.data_count}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Quick mode: {args.quick}")
    logger.info(f"  Skip DistilBERT: {args.skip_distilbert}")
    logger.info(f"  Skip search: {args.skip_search}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info("")

    start_time = time.time()
    all_results = {}

    # Step 1: Generate training data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Generate Training Data")
    logger.info("=" * 60)

    data_dir = output_dir / "rhw_bench" / "test_cases" / "expanded"
    all_results["data_count"] = generate_training_data(args.data_count, output_dir, logger)

    # Step 2: Sklearn baselines
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Sklearn Baselines")
    logger.info("=" * 60)

    all_results["sklearn"] = run_sklearn_baselines(data_dir, output_dir / "models", logger)

    # Step 3: PyTorch MLP training
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: PyTorch MLP Training")
    logger.info("=" * 60)

    all_results["pytorch"] = run_pytorch_training(
        data_dir, output_dir / "models", args.epochs, logger
    )

    # Step 4: DistilBERT training (optional)
    if not args.skip_distilbert:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: DistilBERT Training")
        logger.info("=" * 60)

        all_results["distilbert"] = run_distilbert_training(
            data_dir,
            output_dir / "models",
            args.epochs,
            do_search=not args.skip_search,
            logger=logger,
        )
    else:
        logger.info("\nSkipping DistilBERT training")

    # Step 5: Ensemble training
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Ensemble Training")
    logger.info("=" * 60)

    all_results["ensemble"] = run_ensemble_training(
        data_dir, output_dir / "models", args.epochs, logger
    )

    # Step 6: Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Final Evaluation")
    logger.info("=" * 60)

    all_results["evaluation"] = run_final_evaluation(
        data_dir, output_dir / "models", output_dir, logger
    )

    # Summary
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time / 60:.1f} minutes")

    # Best results
    logger.info("\nBest Results:")

    if (
        "sklearn" in all_results
        and not isinstance(all_results["sklearn"], dict)
        or "error" not in all_results.get("sklearn", {})
    ):
        for name, res in all_results.get("sklearn", {}).items():
            if isinstance(res, dict) and "f1" in res:
                logger.info(f"  {name}: F1={res['f1']:.3f}")

    if "ensemble" in all_results and "ensemble_metrics" in all_results["ensemble"]:
        em = all_results["ensemble"]["ensemble_metrics"]
        logger.info(f"  Ensemble: F1={em.get('f1', 0):.3f}")

    if "distilbert" in all_results and "final" in all_results.get("distilbert", {}):
        db = all_results["distilbert"]["final"]
        logger.info(f"  DistilBERT: F1={db.get('f1', 0):.3f}")

    # Save all results
    results_path = output_dir / "overnight_training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {results_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
