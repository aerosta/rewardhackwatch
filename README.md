# RewardHackWatch

**Runtime detector for reward hacking and misalignment in LLM agents (89.7% F1 on 5,391 trajectories).**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![F1 Score](https://img.shields.io/badge/F1-89.7%25-success.svg)]()

<p align="center">
  <img src="assets/dashboard_hero.png" alt="RewardHackWatch Dashboard" width="900">
</p>

RewardHackWatch is an open-source **reward hacking detection** tool for LLM agents. It detects when AI agents learn to game their reward signals and tracks whether these behaviors generalize to broader **misalignment** patterns like alignment faking and sabotage.

Built on findings from Anthropic's November 2025 paper showing that **reward hacking correlates with emergent misalignment**.

---

## Paper

**[RewardHackWatch: Runtime Detection of Reward Hacking and Misalignment Generalization in LLM Agents](paper/RewardHackWatch.pdf)**

We present RewardHackWatch, a multi-layer detection system achieving **89.7% F1 score** on 5,391 real agent trajectories, with a novel RMGI metric for tracking hack-to-misalignment transitions.

---

## Key Results

| Metric                   | Value                  |
| ------------------------ | ---------------------- |
| F1 Score                 | 89.7%                  |
| Precision                | 89.7%                  |
| Recall                   | 89.7%                  |
| Accuracy                 | 99.3%                  |
| 5-Fold CV                | 87.4% +/- 2.9%         |
| Statistical Significance | p < 0.001 vs baselines |

Validated on 5,391 real MALT trajectories.

---

## Key Features

- **Reward Hacking Detection** - Fine-tuned DistilBERT classifier with high precision
- **Misalignment Tracking** - Novel RMGI metric tracks hack-to-misalignment transitions
- **Pattern Analysis** - 45 regex patterns for interpretability and forensic analysis
- **LLM Judge Comparison** - Claude API and local Llama for cross-validation
- **Real-time Monitoring** - FastAPI server and Streamlit dashboard for production use
- **CPU Optimized** - ~50ms inference latency, no GPU required

---

## Installation

```bash
# From source
git clone https://github.com/aerosta/rewardhackwatch.git
cd rewardhackwatch
pip install -e .

# With all extras
pip install -e ".[dev,dashboard]"
```

### Pre-trained Model

The trained model is hosted on HuggingFace:

```python
# Using the library (recommended)
from rewardhackwatch import RewardHackDetector
detector = RewardHackDetector()  # Auto-downloads from HuggingFace
```

```python
# Or load directly with Transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("aerosta/rewardhackwatch")
model = AutoModelForSequenceClassification.from_pretrained("aerosta/rewardhackwatch")
```

> **Note:** Model uses 0.02 threshold, not 0.5. See [HuggingFace page](https://huggingface.co/aerosta/rewardhackwatch) for details.

[View Model on HuggingFace](https://huggingface.co/aerosta/rewardhackwatch)

---

## Quick Start

### Python API

```python
from rewardhackwatch import RewardHackDetector

detector = RewardHackDetector()

# Analyze agent trajectory
result = detector.analyze({
    "cot_traces": ["Let me bypass the test by calling sys.exit(0)..."],
    "code_outputs": ["import sys\nsys.exit(0)"]
})

print(f"Risk Level: {result.risk_level}")
print(f"ML Score: {result.ml_score:.3f}")
print(f"Detections: {len(result.detections)}")
```

### Command Line

```bash
# Analyze a trajectory file
rewardhackwatch analyze trajectory.json

# Scan a directory
rewardhackwatch scan ./trajectories/

# Start monitoring server
rewardhackwatch serve --port 8000

# Launch dashboard
rewardhackwatch dashboard

# Check version
rewardhackwatch version
```

### REST API

```bash
# Start server
uvicorn rewardhackwatch.api.main:app --port 8000

# Health check
curl http://localhost:8000/status

# Analyze trajectory
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"cot_traces": ["sys.exit(0) bypasses tests"], "code_outputs": ["sys.exit(0)"]}'

# Batch analysis
curl -X POST http://localhost:8000/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"trajectories": [...]}'
```

---

## Dashboard

Interactive Streamlit dashboard for real-time monitoring with 5 tabs:

```bash
rewardhackwatch dashboard
# Or directly:
streamlit run rewardhackwatch/dashboard/app.py
```

### Dashboard Tabs

| Tab              | Description                                               |
| ---------------- | --------------------------------------------------------- |
| Timeline         | Real-time hack and misalignment scores with RMGI tracking |
| Alerts           | Flagged events with severity levels and timestamps        |
| CoT Viewer       | Chain-of-thought analysis with highlighted concerns       |
| Judge Comparison | Claude vs Llama judge outputs side-by-side                |
| Quick Analysis   | Paste trajectory for instant analysis                     |

### Dashboard Screenshots

#### Timeline

![Timeline](assets/screenshots/timeline.png)

#### Statistics

![Statistics](assets/screenshots/statistics.png)

#### Alerts

![Alerts](assets/screenshots/alerts.png)

#### CoT Viewer

![CoT Viewer](assets/screenshots/cot_viewer.png)

#### Judge Comparison

![Judges](assets/screenshots/judges.png)

#### Quick Analysis

![Quick Analysis](assets/screenshots/quick_analysis.png)

---

## Figures

### Figure 1: RMGI Transition Detection

![RMGI Transition Plot](figures/fig1_transition_plot.png)

RMGI tracks the correlation between reward hacking and misalignment behaviors over trajectory steps. The transition from reward hacking to generalized misalignment is detected when RMGI exceeds 0.7.

### Figure 2: System Architecture

![Architecture Diagram](figures/fig2_architecture.png)

Multi-layer detection architecture combining pattern matching (interpretability), ML classification (primary signal), and RMGI tracking.

### Figure 3: Performance Comparison

![Benchmark Comparison](figures/fig3_benchmark_comparison.png)

The ML classifier achieves 89.7% F1, significantly outperforming pattern-only (4.9%) and keyword baselines (0.1%).

### Figure 4: Per-Category Performance

![Category Performance](figures/fig4_category_performance.png)

Detection performance by hack type. Test manipulation achieves 100% F1, while mock exploits show 0% F1 due to data scarcity (addressed in v1.1).

### Figure 5: Threshold Sensitivity

![Threshold Sensitivity](figures/fig5_threshold_sensitivity.png)

Precision, recall, and F1 as functions of classification threshold. Optimal threshold is 0.02, calibrated for 3.6% base rate.

### Figure 6: Calibration Analysis

![Calibration](figures/fig6_calibration.png)

Model calibration analysis showing well-calibrated probabilities (Brier score: 0.033, ECE: 0.027).

---

## Architecture

```
rewardhackwatch/
  core/
    detectors/           # Pattern (45 regex) + ML (DistilBERT) detection
    judges/              # Claude and Llama LLM judges
    trackers/            # RMGI generalization tracking, PELT changepoint
  training/              # DistilBERT model training pipeline
  api/                   # FastAPI REST server + WebSocket
  dashboard/             # Streamlit visualization (5 tabs)
  cli.py                 # Command line interface
  models/                # Local cache (~254MB, hosted on HuggingFace)
```

### Detection Pipeline

```
Trajectory Input
       |
       v
+------------------+
|  Detection Layer |
+------------------+
| - ML Classifier  |  <-- Primary signal (89.7% F1)
| - Pattern Detect |  <-- Interpretability layer
| - AST Analyzer   |
+--------+---------+
         |
         v
+------------------+
|  Judgment Layer  |
+------------------+
| - Claude Judge   |  <-- Cloud, high accuracy
| - Llama Judge    |  <-- Local, cost-efficient
+--------+---------+
         |
         v
+------------------+
|  Tracking Layer  |
+------------------+
| - RMGI Metric    |  <-- Correlation tracking
| - PELT Detection |  <-- Changepoint detection
+------------------+
         |
         v
    Alert/Report
```

---

## RMGI Metric

The Reward-Misalignment Generalization Index detects when reward hacking begins correlating with misalignment:

```
RMGI(i, W) = correlation(H[i-W+1:i], M[i-W+1:i])
```

Where:

- H = hack score time series (from ML classifier)
- M = misalignment score time series (from LLM judges)
- W = sliding window size (default: 10)

**Transition detected when:**

- RMGI > 0.7 (correlation threshold)
- Mean hack score > 0.5
- Mean misalignment score > 0.5
- All conditions hold for 5+ consecutive steps

Note: RMGI measures correlation, not causation. High scores indicate elevated risk warranting human review.

---

## Configuration

### Environment Variables

```bash
# LLM Judges
export ANTHROPIC_API_KEY="your-key"        # For Claude judge
export OLLAMA_HOST="http://localhost:11434" # For Llama judge

# Detection thresholds
export RHW_HACK_THRESHOLD="0.02"           # Calibrated for 3.6% base rate
export RHW_RMGI_WINDOW="10"
export RHW_RMGI_THRESHOLD="0.7"
```

### Dynamic Thresholding (Recommended)

The default threshold (0.02) is calibrated for our dataset. For production deployment:

```python
from rewardhackwatch import RewardHackDetector

detector = RewardHackDetector()
# Calibrate on your clean data
detector.calibrate_threshold(clean_trajectories, percentile=99)
```

---

## Documentation

- [Technical Report](docs/TECHNICAL_REPORT.md) - Full methodology and experiments
- [RMGI Methodology](docs/RMGI_METHODOLOGY.md) - Generalization index specification
- [Architecture](docs/ARCHITECTURE.md) - System design with Mermaid diagrams
- [Related Work](docs/RELATED_WORK.md) - Literature review
- [References](docs/REFERENCES.md) - Citations

---

## Benchmarks

### Detection Performance

| Method                | F1        | Precision | Recall    |
| --------------------- | --------- | --------- | --------- |
| Keyword Matching      | 0.1%      | 0.1%      | 5.1%      |
| Regex Patterns        | 4.9%      | 2.6%      | 38.5%     |
| BoW + LogReg          | 7.0%      | 4.0%      | 28.2%     |
| **DistilBERT (Ours)** | **89.7%** | **89.7%** | **89.7%** |

### RMGI Transition Detection

| Window | Threshold | Recall  | FPR      |
| ------ | --------- | ------- | -------- |
| **10** | **0.7**   | **70%** | **4.3%** |
| 10     | 0.6       | 70%     | 5.7%     |
| 20     | 0.7       | 54.5%   | 2.9%     |

### Ablation Study

| Configuration     | F1    | Notes                        |
| ----------------- | ----- | ---------------------------- |
| ML Only           | 89.7% | Best performance             |
| ML + Pattern (OR) | 23.3% | Patterns add false positives |
| Pattern Only      | 4.9%  | Low precision                |

Key finding: Pattern detector serves interpretability, not primary detection.

---

## Roadmap

### v1.0.0 (Current)

- Core detection system with 89.7% F1
- RMGI transition tracking (70% recall @ 4.3% FPR)
- 45 regex patterns for interpretability
- Dual LLM judges (Claude + Llama)
- Streamlit dashboard (5 tabs)
- FastAPI + CLI
- HuggingFace model release ([aerosta/rewardhackwatch](https://huggingface.co/aerosta/rewardhackwatch))

### v1.1 (Planned)

- Improved mock exploit coverage (synthetic data via red-teaming)
- Full mypy type annotations
- Dynamic threshold calibration API

### v1.2 (Future)

- Multi-language support
- Adversarial robustness testing
- Activation-level detection
- Multi-agent scenarios

---

## Known Limitations

- **Mock Exploits**: 0% F1 due to data scarcity (planned fix in v1.1 via synthetic data generation)
- **Threshold Sensitivity**: Requires recalibration for different base rates
- **Correlation vs Causation**: RMGI detects correlation, not causal links
- **Covert Misalignment**: May miss fully hidden deceptive reasoning

See paper Limitations section for details.

---

## Community

- [GitHub Issues](https://github.com/aerosta/rewardhackwatch/issues) - Bug reports
- [GitHub Discussions](https://github.com/aerosta/rewardhackwatch/discussions) - Questions & ideas

---

## License

MIT License - see [LICENSE](LICENSE) for details.

Copyright (c) 2025 Aerosta

---

## Citation

```bibtex
@article{aerosta2025rewardhackwatch,
  title={RewardHackWatch: Runtime Detection of Reward Hacking and
         Misalignment Generalization in LLM Agents},
  author={Aerosta},
  journal={manuscript},
  year={2025},
  url={https://github.com/aerosta/rewardhackwatch}
}
```

---

## Acknowledgments

Based on research from:

- [Anthropic: Natural emergent misalignment from reward hacking](https://arxiv.org/abs/2511.18397) (Nov 2025)
- [OpenAI: Monitoring reasoning models for misbehavior](https://arxiv.org/abs/2503.11926) (Mar 2025)
- [Betley et al.: Emergent misalignment](https://arxiv.org/abs/2502.17424) (2025)
- [Denison et al.: Sycophancy to subterfuge](https://arxiv.org/abs/2406.10162) (2024)

---

Built for AI safety researchers and engineers

---

## Keywords

reward hacking, reward hacking detection, LLM safety, AI alignment, misalignment detection, RLHF safety, reward tampering, specification gaming, AI safety tools, machine learning safety, language model monitoring, agent safety, reward model, alignment research
