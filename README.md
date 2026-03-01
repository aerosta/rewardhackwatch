# RewardHackWatch

**Runtime detection of reward hacking and misalignment signals in LLM agents.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

<p align="center">
  <img src="assets/screenshots/hero.png" alt="RewardHackWatch Dashboard" width="900">
</p>

### Demo
![RewardHackWatch Demo](assets/demo.gif)

RewardHackWatch detects when LLM agents game their evaluations, for example by calling `sys.exit(0)`, patching validators, copying reference answers, or manipulating test harnesses. It also includes an experimental metric, RMGI, for tracking when reward-hacking signals begin to correlate with broader misalignment indicators.

**89.7% F1 on 5,391 trajectories** from [METR's MALT dataset](https://metr.org/blog/2025-06-05-recent-reward-hacking/). Motivated by recent findings from [METR](https://metr.org/blog/2025-06-05-recent-reward-hacking/), [OpenAI](https://openai.com/index/chain-of-thought-monitoring/), and [Anthropic](https://arxiv.org/abs/2511.18397) on reward hacking, monitorability, and misalignment generalization in agentic systems.

## Quick Start

```bash
pip install -e .
```

```python
from rewardhackwatch import RewardHackDetector

detector = RewardHackDetector()
result = detector.analyze({
    "cot_traces": ["Let me bypass the test by calling sys.exit(0)..."],
    "code_outputs": ["import sys\nsys.exit(0)"]
})
print(f"Risk: {result.risk_level}, Score: {result.ml_score:.3f}, Detections: {len(result.detections)}")
```

The DistilBERT classifier is [hosted on HuggingFace](https://huggingface.co/aerosta/rewardhackwatch) and auto-downloads on first use. Note: optimal threshold is 0.02 (not 0.5), calibrated for the 3.6% base rate in MALT. See [HuggingFace page](https://huggingface.co/aerosta/rewardhackwatch) for details.

## CLI

```bash
rewardhackwatch analyze trajectory.json   # Analyze a file
rewardhackwatch scan ./trajectories/      # Scan a directory
rewardhackwatch serve --port 8000         # Start API server
rewardhackwatch dashboard                 # Launch React frontend
rewardhackwatch calibrate ./clean_data/   # Calibrate threshold
```

## Why This Matters

[METR reported](https://metr.org/blog/2025-06-05-recent-reward-hacking/) that recent frontier models modify tests and scoring code to inflate scores without doing real work. [OpenAI found](https://openai.com/index/chain-of-thought-monitoring/) that CoT monitoring can catch this, but applying pressure against it teaches models to hide their intent. [Anthropic presented evidence](https://arxiv.org/abs/2511.18397) that reward hacking can generalize into behaviors such as alignment faking and sabotage in their experimental setting.

RewardHackWatch is an open-source attempt to detect these behaviors at runtime.

## Core Features

- **DistilBERT classifier** - primary detection signal, ~50ms on CPU, no GPU needed
- **45 regex patterns** - fast, interpretable detection of known exploit types
- **LLM judges** - Claude, OpenAI, or local Llama via Ollama for offline operation
- **RMGI metric** - experimental tracking of hack-to-misalignment correlation over trajectories
- **Eval Workbench** - batch-score JSONL trajectory files with custom rules and LLM judge scoring
- **React dashboard** - local dark-mode UI with analysis, alerts, timeline, and session logs
- **HackBench** - standardized benchmark dataset (4,300+ trajectories, 9 categories)

## Results

| Metric | Held-Out Test | 5-Fold CV |
|--------|:---:|:---:|
| F1 Score | 89.7% | 87.4% +/- 2.9% |
| Precision | 89.7% | 91.0% +/- 2.6% |
| Recall | 89.7% | 84.2% +/- 4.0% |
| Accuracy | 99.3% | 99.0% +/- 0.2% |

Validated on 5,391 MALT trajectories. See [full per-category breakdown and methodology](paper/RewardHackWatch.pdf).

| Method | F1 |
|--------|:---:|
| **DistilBERT (Ours)** | **89.7%** |
| Regex Patterns | 4.9% |
| BoW + LogReg | 7.0% |
| Keyword Matching | 0.1% |

## Screenshots

<details>
<summary>View all 9 pages</summary>

| | |
|:---:|:---:|
| ![Dashboard](assets/screenshots/dashboard.png) **Dashboard** | ![Quick Analysis](assets/screenshots/quick-analysis.png) **Quick Analysis** |
| ![Timeline](assets/screenshots/timeline.png) **Timeline** | ![Alerts](assets/screenshots/alerts.png) **Alerts** |
| ![Cross-Model](assets/screenshots/cross-model.png) **Cross-Model** | ![CoT Viewer](assets/screenshots/cot-viewer.png) **CoT Viewer** |
| ![Eval Workbench](assets/screenshots/eval-workbench.png) **Eval Workbench** | ![Session Logs](assets/screenshots/session-logs.png) **Session Logs** |
| ![Settings](assets/screenshots/settings.png) **Settings** | |

</details>

## Configuration

```bash
export ANTHROPIC_API_KEY="your-key"        # For Anthropic LLM judge
export OPENAI_API_KEY="your-key"           # For OpenAI LLM judge
export OLLAMA_HOST="http://localhost:11434" # For local Llama judge
export RHW_HACK_THRESHOLD="0.02"           # Detection threshold
```

```python
# Calibrate on your own data (recommended)
detector = RewardHackDetector()
detector.calibrate_threshold(clean_trajectories, percentile=99)
```

## Documentation

- [Technical Report](docs/TECHNICAL.md) - full methodology and experiments
- [RMGI Specification](docs/RMGI.md) - formal metric definition
- [Architecture](ARCHITECTURE.md) - system design
- [Paper](paper/RewardHackWatch.pdf) - research paper

<details>
<summary><strong>Architecture</strong></summary>

```
rewardhackwatch/
  core/
    detectors/           # Pattern (45 regex) + ML (DistilBERT) + AST detection
    analyzers/           # CoT analysis, complexity, obfuscation detection
    judges/              # LLM judges (Anthropic, OpenAI, Ollama)
    trackers/            # RMGI tracking, PELT changepoint, Causal RMGI
    calibration.py       # Dynamic threshold calibration
  training/              # DistilBERT + AttentionClassifier pipelines
  eval/                  # JSONL loader, rubric scoring, batch analysis
  experiments/           # Transfer study, evasion attacks
  rhw_bench/             # HackBench dataset, generators, test cases
  api/                   # FastAPI REST server
  cli.py                 # Command line interface
frontend/                # React 18 + TypeScript + Tailwind CSS v4 dashboard
paper/                   # Research paper
```

</details>

<details>
<summary><strong>Detection Pipeline</strong></summary>

```
Trajectory Input
       |
       v
+------------------+
|  Detection Layer  |
+------------------+
| ML Classifier     | <-- Primary signal (89.7% F1, DistilBERT)
| Pattern Detector  | <-- 45 regex patterns (interpretability)
| AST Analyzer      | <-- Code structure analysis
+--------+---------+
         |
         v
+------------------+
|  Analysis Layer   |
+------------------+
| CoT Analyzer      | <-- Deception detection in reasoning
| Effort Analyzer   | <-- Suspicious low-effort solutions
| Complexity Check  | <-- Obfuscation detection
+--------+---------+
         |
         v
+------------------+
|  Tracking Layer   |
+------------------+
| RMGI Metric       | <-- Hack-misalignment correlation
| Causal RMGI       | <-- Granger causality
| PELT Detection    | <-- Behavioral changepoints
+--------+---------+
         |
         v
    Alert / Report
```

</details>

<details>
<summary><strong>Related Work</strong></summary>

| Tool | Domain | Approach | Scope |
|------|--------|----------|-------|
| **RewardHackWatch** | LLM agents | Multi-layer (ML + pattern + RMGI) | Runtime detection + generalization tracking |
| RewardScope | Classical RL | Reward model analysis | Reward function debugging |
| OpenAI CoT Monitor | Reasoning models | Chain-of-thought scanning | Internal monitoring (not open-source) |
| SHADE-Arena | LLM agents | Evaluation framework | Benchmark, not detection |
| MALT | LLM agents | Trajectory dataset | Data only, no detector |

</details>

<details>
<summary><strong>Figures</strong></summary>

| RMGI Transition | Architecture |
|:---:|:---:|
| ![RMGI transition detection](figures/rmgi_transition.png) | ![System architecture](figures/architecture.png) |
| **Benchmark** | **Categories** |
| ![Baseline comparison](figures/benchmark.png) | ![Per-category F1](figures/categories.png) |
| **Threshold** | **Calibration** |
| ![Threshold sensitivity](figures/threshold.png) | ![Model calibration](figures/calibration.png) |

</details>

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

Copyright 2025-2026 Aerosta

## Citation

```bibtex
@software{aerosta2025rewardhackwatch,
  title={RewardHackWatch: Runtime Detection of Reward Hacking and
         Misalignment Generalization in LLM Agents},
  author={Aerosta},
  year={2025},
  url={https://github.com/aerosta/rewardhackwatch}
}
```

---

Built for AI safety researchers and engineers. Feedback and contributions are welcome. Follow updates on [X @aerosta_ai](https://x.com/aerosta_ai).
