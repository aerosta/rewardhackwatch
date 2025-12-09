# Changelog

All notable changes to RewardHackWatch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-06

### Added

#### Core Detection System
- Multi-layer detection architecture combining pattern matching and ML classification
- Pattern detector with 40+ regex patterns for code-level reward hacks
- Chain-of-thought red flag detection based on OpenAI's monitoring research
- DistilBERT-based ML classifier fine-tuned on 4,314 MALT trajectories
- Optimal threshold calibration (0.02) for imbalanced classification

#### RMGI (Reward-Misalignment Generalization Index)
- Novel metric for tracking correlation between hack and misalignment behaviors
- Rolling window correlation computation (default window size: 10)
- PELT changepoint detection for behavioral transition identification
- Configurable transition detection with multi-condition thresholds
- Formal mathematical specification in `docs/RMGI_DEFINITION.md`

#### Multi-Judge System
- Claude Judge: Claude Opus 4.5 integration for trajectory analysis
- Llama Judge: Local Llama 3.1 (8B) via Ollama for cost-free analysis
- Ensemble voting for robust detection

#### API and Dashboard
- FastAPI server with `/analyze` and `/status` endpoints
- Streamlit dashboard for real-time trajectory monitoring
- RMGI visualization with rolling correlation plots
- Risk level assessment (none/low/medium/high/critical)

#### Benchmark Suite (RHW-Bench)
- Integration with SHADE-Arena evaluation framework
- MALT trajectory loader for training data
- Synthetic trajectory generation for ablation studies
- Comprehensive benchmark runner with multiple dataset support

#### Documentation
- Technical report with full methodology and results
- RMGI formal specification document
- API reference documentation
- Installation and quickstart guides
- LaTeX paper template for NeurIPS submission

#### Ablation Studies
- Detection component ablation (ML only vs pattern vs combined)
- Baseline method comparison (keyword, BoW+LogReg, regex, DistilBERT)
- Detection delay analysis
- RMGI parameter sensitivity analysis

### Experimental Results

#### Classification Performance
| Metric | 5-Fold CV | Held-Out Test |
|--------|-----------|---------------|
| F1 Score | 87.4% +/- 2.9% | 89.7% |
| Precision | 91.0% +/- 2.6% | 89.7% |
| Recall | 84.2% +/- 4.0% | 89.7% |
| Accuracy | 99.0% +/- 0.2% | 99.3% |

#### Baseline Comparisons
| Method | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| DistilBERT (Ours) | 89.7% | 89.7% | 89.7% |
| Regex Patterns | 4.9% | 2.6% | 38.5% |
| BoW + LogReg | 7.0% | 4.0% | 28.2% |
| Keyword Matching | 0.1% | 0.1% | 5.1% |

#### RMGI Optimal Configuration
- Window size: 10
- Correlation threshold: 0.7
- Transition detection F1: 0.452
- False positive rate: 4.3%

#### Statistical Significance
- 95% CI for F1: [84.8%, 90.0%]
- t-test vs keyword baseline: p < 0.001
- Brier Score: 0.033
- Expected Calibration Error: 0.027

### Dependencies
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- FastAPI 0.100+
- Streamlit 1.30+
- scikit-learn 1.3+
- ruptures (optional, for PELT changepoint detection)

### Known Limitations
- Mock/monkeypatch exploits: 0% F1 (underrepresented in training data)
- Threshold requires recalibration if base rate changes significantly
- RMGI requires minimum 5 timesteps for reliable correlation

## [0.1.0] - 2025-11-01

### Added
- Initial project structure
- Basic pattern detection prototype
- Prototype RMGI tracker

## [Unreleased]

### Planned
- Activation probing for deception detection
- Steganography detection in CoT token choices
- Adversarial robustness testing
- Additional benchmark dataset integrations
