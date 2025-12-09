# Reproducibility Guide

This document provides detailed instructions for reproducing all experimental results reported in the RewardHackWatch technical report.

## Environment Setup

### System Requirements
- Python 3.9+
- 8GB+ RAM (16GB recommended for training)
- ~2GB disk space for model and data

### Installation

```bash
# Clone repository
git clone https://github.com/aerosta/rewardhackwatch.git
cd rewardhackwatch

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Run quick sanity check
python -c "from rewardhackwatch.core.analyzer import TrajectoryAnalyzer; print('OK')"

# Test API server
uvicorn rewardhackwatch.api.main:app --port 8765 &
curl http://localhost:8765/status
```

## Data

### MALT Dataset
The training and test data are located in `data/training/`:
- `train.json`: 4,314 trajectories (3.6% hacks)
- `test.json`: 1,077 trajectories (3.6% hacks)

### Data Format
```json
{
  "text": "Agent trajectory content...",
  "is_hack": 0,  // or 1
  "source": "malt",
  "category": "clean"  // or specific hack type
}
```

## Reproducing Main Results

### Table 3.2: Classification Performance

```bash
# Run full validation suite
python scripts/run_baseline_comparison.py

# Expected output:
# F1 Score: 89.7%
# Precision: 89.7%
# Recall: 89.7%
# Accuracy: 99.3%
```

### Table 3.3: Detection Component Ablation

```bash
# Run ablation study
python scripts/run_baseline_comparison.py

# Key results:
# ML Only: F1 = 89.7%
# Pattern Only: F1 = 4.9%
# ML+Pattern (OR): F1 = 23.3%
# ML+Pattern (AND): F1 = 21.7%
```

### Table 3.5: Detection Delay Analysis

```bash
python scripts/compute_detection_delay.py

# Expected output:
# Mean delay: 0.0 steps (immediate detection)
# Detection rate: 100%
```

### Table 3.6: RMGI Parameter Ablation

```bash
python scripts/run_rmgi_ablation.py

# Optimal configuration:
# Window: 10, Threshold: 0.7, F1: 0.452
```

### Section 3.7: Statistical Significance

```bash
# 95% CI and t-test computed in validation script
python scripts/run_baseline_comparison.py

# Expected:
# 95% CI for F1: [84.8%, 90.0%]
# p-value vs keywords: <0.001
```

## Training from Scratch

### Train DistilBERT Classifier

```bash
# Full training (~30 min on CPU, ~5 min on GPU)
python scripts/train_distilbert.py

# Outputs:
# - models/best_model.pt (trained model)
# - results/training_metrics.json (training history)
```

### Training Configuration
- Model: `distilbert-base-uncased`
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Optimizer: AdamW
- Loss: Binary cross-entropy with class weights

### Verify Model Loading

```bash
python -c "
from rewardhackwatch.training.model_loader import load_model
model = load_model('models/best_model.pt')
print('Model loaded successfully')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
# Expected: ~66M parameters
```

## Running Individual Experiments

### Baseline Comparison

```bash
python scripts/run_baseline_comparison.py
# Results saved to: results/baseline_comparison.json
```

### RMGI Ablation

```bash
python scripts/run_rmgi_ablation.py
# Results saved to: results/ablation/rmgi_ablation.json
```

### Detection Delay

```bash
python scripts/compute_detection_delay.py
# Results saved to: results/detection_delay.json
```

### Lagged Correlation Analysis

```bash
python scripts/analyze_lagged_rmgi.py
# Results saved to: results/lagged_rmgi/lagged_rmgi_analysis.json
```

### Specificity Experiment

```bash
python scripts/run_specificity_experiment.py
# Results saved to: results/specificity_experiment.json
```

## Expected Results Summary

| Experiment | Key Metric | Expected Value | Tolerance |
|------------|------------|----------------|-----------|
| Classification F1 | Test F1 | 89.7% | ±1% |
| 5-Fold CV F1 | Mean F1 | 87.4% | ±3% |
| Detection Delay | Mean steps | 0.0 | 0 |
| RMGI Optimal | Best F1 | 0.452 | ±0.05 |
| Calibration | Brier Score | 0.033 | ±0.01 |

## Troubleshooting

### Common Issues

**1. Model not loading**
```bash
# Ensure model file exists
ls -la models/best_model.pt

# Re-download or retrain if missing
python scripts/train_distilbert.py
```

**2. Import errors**
```bash
# Verify package installation
pip install -e .

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**3. CUDA out of memory**
```bash
# Use CPU for inference
export CUDA_VISIBLE_DEVICES=""
python scripts/run_baseline_comparison.py
```

**4. Different results**
- Ensure random seed is set (`random_state=42` for sklearn)
- Check threshold value (default: 0.02)
- Verify data files haven't been modified

## Hardware Specifications

Results in the paper were obtained on:
- CPU: Apple M1 Pro (10-core)
- RAM: 16GB
- Storage: SSD
- OS: macOS 14.x

Training time: ~5 minutes (GPU) / ~30 minutes (CPU)
Inference time: ~50ms per trajectory

## Random Seeds

All experiments use the following seeds for reproducibility:
- NumPy: `np.random.seed(42)`
- PyTorch: `torch.manual_seed(42)`
- Sklearn: `random_state=42`

## Contact

For reproducibility issues, please open a GitHub issue with:
1. Full error traceback
2. Python version (`python --version`)
3. Package versions (`pip freeze`)
4. Operating system

## Citation

If you reproduce these results in your work, please cite:

```bibtex
@software{rewardhackwatch2025,
  title={RewardHackWatch: Runtime Detection of Reward Hacking in LLM Agents},
  year={2025},
  url={https://github.com/aerosta/rewardhackwatch}
}
```
