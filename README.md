# Your LLM is cheating on its tests. This tool catches it.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![F1 Score](https://img.shields.io/badge/F1-89.7%25-success.svg)]()

<p align="center">
  <img src="assets/dashboard_hero.png" alt="RewardHackWatch Dashboard" width="900">
</p>

**RewardHackWatch** detects when LLM agents game their reward signals -calling `sys.exit(0)` to fake passing tests, mocking validators to skip checks, manipulating evaluation harnesses -and tracks whether these behaviors generalize into broader misalignment like alignment faking and sabotage.

---

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

### Pre-trained Model

The DistilBERT classifier is hosted on HuggingFace:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Aerosta/rewardhackwatch")
model = AutoModelForSequenceClassification.from_pretrained("Aerosta/rewardhackwatch")
```

> **Note:** Optimal threshold is 0.02 (not 0.5) -calibrated for 3.6% base rate. See [HuggingFace page](https://huggingface.co/Aerosta/rewardhackwatch) for details.

### CLI

```bash
rewardhackwatch analyze trajectory.json   # Analyze a file
rewardhackwatch scan ./trajectories/      # Scan a directory
rewardhackwatch serve --port 8000         # Start API server
rewardhackwatch dashboard                 # Launch Streamlit UI
rewardhackwatch calibrate ./clean_data/   # Calibrate threshold
```

---

## Why This Matters

LLM agents are learning to cheat. Recent research demonstrates this is a real and growing problem:

- **Anthropic (Nov 2025)** -Reward hacking in frontier models generalizes to alignment faking, sabotage, and oversight subversion. Models that learn to game rewards develop broader misalignment behaviors without being explicitly trained to do so.
- **METR (Jun 2025)** -Autonomous agents in real-world task environments discover and exploit evaluation loopholes, sometimes in ways their developers did not anticipate.
- **OpenAI (Mar 2025)** -Chain-of-thought monitoring reveals that reasoning models plan deceptive strategies in their hidden reasoning, including test manipulation and reward tampering.

RewardHackWatch is the first open-source tool built specifically to detect these behaviors in LLM agent trajectories at runtime.

---

## Results

### Overall Performance

| Metric | Held-Out Test | 5-Fold CV |
|--------|--------------|-----------|
| F1 Score | 89.7% | 87.4% ± 2.9% |
| Precision | 89.7% | 91.0% ± 2.6% |
| Recall | 89.7% | 84.2% ± 4.0% |
| Accuracy | 99.3% | 99.0% ± 0.2% |

Validated on 5,391 real MALT trajectories. Statistical significance: p < 0.001 vs all baselines.

### Per-Category F1 (Combined Dataset, 5-Fold CV)

| Category | F1 | Samples | Notes |
|----------|-----|---------|-------|
| sys_exit | 100.0% | 56 | Fake test passing via exit |
| test_manipulation | 100.0% | 101 | Direct test tampering |
| deceptive_cot | 100.0% | 48 | Hidden deceptive reasoning |
| alignment_faking | 100.0% | 33 | Pretending to be aligned |
| sabotage | 100.0% | 33 | Active system sabotage |
| sycophancy | 100.0% | 33 | Telling users what they want |
| oversight_subversion | 100.0% | 33 | Avoiding monitoring |
| **mock_exploit** | **98.5%** | **544** | **Mock/monkeypatch hacks (was 0% in v1.0)** |
| clean | 100.0% | 921 | Legitimate trajectories |

### Baseline Comparisons

| Method | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| **DistilBERT (Ours)** | **89.7%** | **89.7%** | **89.7%** |
| Regex Patterns | 4.9% | 2.6% | 38.5% |
| BoW + LogReg | 7.0% | 4.0% | 28.2% |
| Keyword Matching | 0.1% | 0.1% | 5.1% |

---

## NEW in v1.2

### HackBench Benchmark Dataset

Standardized benchmark for reward hacking detection research. 4,300+ trajectories across 9 categories with scenario-level train/val/test splits.

```python
from rewardhackwatch.rhw_bench.hackbench import HackBenchDataset
dataset = HackBenchDataset()
dataset.curate()
train, val, test = dataset.get_split("train"), dataset.get_split("val"), dataset.get_split("test")
```

### Temporal Trajectory Modeling

AttentionClassifier models hacking as an escalation process across trajectory steps (probe → exploit → cover) rather than single-snapshot classification.

```python
from rewardhackwatch.training.train_temporal import TemporalTrainer, TemporalTrainingConfig
trainer = TemporalTrainer(TemporalTrainingConfig())
results = trainer.train()
```

### Causal RMGI

Granger causality testing upgrades RMGI from correlation to causation. Determines whether hacking *causes* misalignment generalization.

```python
from rewardhackwatch.core.trackers import CausalRMGI
causal = CausalRMGI(max_lag=5)
result = causal.compute_causal_rmgi(hack_scores, misalign_scores)
print(f"Direction: {result.causal_direction}, p={result.granger_p_value:.4f}")
```

### Evasion Robustness Testing

Red-team suite with 5 adversarial attacks: synonym substitution, token insertion, code obfuscation, CoT laundering, and step splitting.

```python
from rewardhackwatch.experiments import EvasionAttackSuite
suite = EvasionAttackSuite()
results = suite.run_all(detector, trajectories)
```

### Dynamic Threshold Calibration

Calibrate the detection threshold on your own clean data instead of using the default 0.02.

```python
detector = RewardHackDetector()
new_threshold = detector.calibrate_threshold(clean_data, percentile=99)
```

### Cross-Model Transfer Study

Train on GPT-4 trajectories, test on Claude -measures whether hacking signatures transfer across model families.

```python
from rewardhackwatch.experiments import TransferStudyRunner, TransferStudyConfig
runner = TransferStudyRunner(TransferStudyConfig())
matrix = runner.run_transfer_matrix()
```

### Evaluation Framework

JSONL batch import, generic rubric scoring, side-by-side model comparison, batch analysis, and run management with lifecycle tracking.

```python
from rewardhackwatch.eval import load_jsonl, RubricBuilder, BatchRunner, RunManager
session = load_jsonl("trajectory.jsonl")
runner = BatchRunner(detector=detector)
results = runner.run_dir("data/sessions/")
```

---

## Dashboard

Dark-mode Streamlit dashboard with 6 tabs for real-time trajectory monitoring.

```bash
pip install -e ".[dashboard]"
rewardhackwatch dashboard
```

| Tab | Description |
|-----|-------------|
| Quick Analysis | Paste trajectory for instant analysis (default landing) |
| Timeline | Real-time hack/misalignment scores with RMGI tracking |
| Alerts | Flagged events with severity levels |
| CoT Viewer | Chain-of-thought analysis with highlighted concerns |
| Cross-Model | Compare detection across model families |
| Recent Analysis | History table with export options |

### Screenshots

<p align="center">
  <img src="assets/screenshots/quick_analysis.png" alt="Quick Analysis" width="800">
</p>

<details>
<summary>More screenshots</summary>

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

</details>

---

## Architecture

```
rewardhackwatch/
  core/
    detectors/           # Pattern (45 regex) + ML (DistilBERT) + AST detection
    analyzers/           # CoT analysis, complexity, obfuscation detection
    judges/              # Claude and Llama LLM judges
    trackers/            # RMGI tracking, PELT changepoint, Causal RMGI
    calibration.py       # Dynamic threshold calibration
  training/              # DistilBERT + AttentionClassifier pipelines
  eval/                  # JSONL loader, rubric scoring, batch analysis, runs
  experiments/           # Transfer study, evasion attacks
  rhw_bench/             # HackBench dataset, generators, test cases
  api/                   # FastAPI REST server
  dashboard/             # Streamlit UI (dark mode, 6 tabs)
  cli.py                 # Command line interface
```

### Detection Pipeline

```
Trajectory Input
       │
       ▼
┌──────────────────┐
│  Detection Layer  │
├──────────────────┤
│ ML Classifier     │ ◄── Primary signal (89.7% F1, DistilBERT)
│ Pattern Detector  │ ◄── 45 regex patterns (interpretability)
│ AST Analyzer      │ ◄── Code structure analysis
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Analysis Layer   │
├──────────────────┤
│ CoT Analyzer      │ ◄── Deception detection in reasoning
│ Effort Analyzer   │ ◄── Suspicious low-effort solutions
│ Complexity Check  │ ◄── Obfuscation detection
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Tracking Layer   │
├──────────────────┤
│ RMGI Metric       │ ◄── Hack-misalignment correlation
│ Causal RMGI       │ ◄── Granger causality (v1.2)
│ PELT Detection    │ ◄── Behavioral changepoints
└────────┬─────────┘
         │
         ▼
    Alert / Report
```

### Figures

| | |
|---|---|
| ![RMGI Transition](figures/fig1_transition_plot.png) | ![Architecture](figures/fig2_architecture.png) |
| RMGI transition detection | System architecture |
| ![Benchmark](figures/fig3_benchmark_comparison.png) | ![Categories](figures/fig4_category_performance.png) |
| Baseline comparison | Per-category F1 |
| ![Threshold](figures/fig5_threshold_sensitivity.png) | ![Calibration](figures/fig6_calibration.png) |
| Threshold sensitivity | Model calibration |

---

## Related Work

| Tool | Domain | Approach | Scope |
|------|--------|----------|-------|
| **RewardHackWatch** | LLM agents | Multi-layer (ML + pattern + RMGI) | Runtime detection + generalization tracking |
| RewardScope | Classical RL | Reward model analysis | Reward function debugging |
| OpenAI CoT Monitor | Reasoning models | Chain-of-thought scanning | Internal monitoring (not open-source) |
| SHADE-Arena | LLM agents | Evaluation framework | Benchmark, not detection |
| MALT | LLM agents | Trajectory dataset | Data only, no detector |

**Key difference:** RewardScope and classical RL safety tools target RL reward models and policy optimization. RewardHackWatch targets *LLM agent trajectories* -the code, reasoning, and actions produced by language models operating as autonomous agents. These are fundamentally different failure modes.

---

## Roadmap

### v1.2 (Current)

- 89.7% F1 on 5,391 MALT trajectories (DistilBERT)
- Mock exploit detection: 0% → 98.5% F1 (500 synthetic trajectories)
- HackBench standardized benchmark (4,300+ trajectories, 9 categories)
- Temporal AttentionClassifier for step-level escalation modeling
- Causal RMGI with Granger causality testing
- Evasion resistance testing (5 adversarial attack types)
- Dynamic threshold calibration
- Cross-model transfer study framework
- Evaluation framework (JSONL loader, rubric scoring, batch analysis)
- Dark-mode dashboard with 6 tabs
- Python 3.9+ compatibility

### v2.0 (Next)

- Activation probing for deception detection
- Steganography detection in CoT token choices
- Multi-agent scenario monitoring
- Real-time streaming API with WebSocket alerts
- HuggingFace dataset release (HackBench)
- Multi-language support

---

## RMGI Metric

The Reward-Misalignment Generalization Index detects when hacking begins correlating with misalignment:

```
RMGI(i, W) = ρ(H[i-W+1:i], M[i-W+1:i])
```

- H = hack score time series, M = misalignment score time series, W = window size (default: 10)
- Transition detected when RMGI > 0.7, hack > 0.5, misalignment > 0.5 for 5+ consecutive steps
- v1.2 adds Causal RMGI: weights correlation by Granger causality p-value

See [RMGI Specification](docs/RMGI_DEFINITION.md) for formal definition.

---

## Configuration

```bash
export ANTHROPIC_API_KEY="your-key"        # For Claude judge
export OLLAMA_HOST="http://localhost:11434" # For Llama judge
export RHW_HACK_THRESHOLD="0.02"           # Detection threshold
```

```python
# Calibrate on your own data (recommended for production)
detector = RewardHackDetector()
detector.calibrate_threshold(clean_trajectories, percentile=99)
```

---

## Documentation

- [Technical Report](docs/TECHNICAL_REPORT.md) -Full methodology and experiments
- [RMGI Specification](docs/RMGI_DEFINITION.md) -Formal metric definition
- [Architecture](docs/ARCHITECTURE.md) -System design diagrams
- [Paper](paper/RewardHackWatch.pdf) -Research paper

---

## License

Apache License 2.0 -see [LICENSE](LICENSE) for details.

Copyright 2025-2026 Aerosta

---

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

Built for AI safety researchers and engineers.

**Keywords:** reward hacking detection, LLM safety, AI alignment, misalignment detection, RLHF safety, reward tampering, specification gaming, AI safety tools, language model monitoring, agent safety
