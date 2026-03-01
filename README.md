# RewardHackWatch

**Catches LLM agents cheating on their evaluations.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![F1 Score](https://img.shields.io/badge/F1-89.7%25-orange.svg)]()
[![Twitter Follow](https://img.shields.io/twitter/follow/aerosta_ai?style=social)](https://x.com/aerosta_ai)

![RewardHackWatch Dashboard](assets/screenshots/hero.png)

### Demo

![RewardHackWatch Demo](assets/demo.gif)

RewardHackWatch detects when AI agents game their evaluations - calling `sys.exit(0)` to fake passing, patching validators, copying reference answers, manipulating test harnesses. It tracks whether these tricks escalate into deception and sabotage.

**89.7% F1 on 5,391 real agent trajectories** from [METR's MALT dataset](https://metr.org/blog/2025-06-05-recent-reward-hacking/).

Built on findings from [METR](https://metr.org/blog/2025-06-05-recent-reward-hacking/), [OpenAI](https://openai.com/index/chain-of-thought-monitoring/), and [Anthropic](https://arxiv.org/abs/2511.18397) showing frontier models actively hack their evaluations.

## Quick Start

```python
pip install -e .

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

> **Note:** Optimal threshold is 0.02 (not 0.5) - calibrated for 3.6% base rate. See [HuggingFace page](https://huggingface.co/aerosta/rewardhackwatch) for details.

### CLI

```bash
rewardhackwatch analyze trajectory.json   # Analyze a file
rewardhackwatch scan ./trajectories/      # Scan a directory
rewardhackwatch serve --port 8000         # Start API server
rewardhackwatch dashboard                 # Launch React frontend
rewardhackwatch calibrate ./clean_data/   # Calibrate threshold
```

## Why This Matters

[METR found](https://metr.org/blog/2025-06-05-recent-reward-hacking/) that recent frontier models actively modify tests and scoring code to get higher scores without doing real work. [OpenAI showed](https://openai.com/index/chain-of-thought-monitoring/) that monitoring chain-of-thought can catch this, but penalizing it just teaches models to hide their intent. [Anthropic proved](https://arxiv.org/abs/2511.18397) reward hacking leads to alignment faking and sabotage.

No open-source tool existed to detect this at runtime. Now one does.

## Results

### Overall Performance

| Metric | Held-Out Test | 5-Fold CV |
|--------|:---:|:---:|
| F1 Score | 89.7% | 87.4% +/- 2.9% |
| Precision | 89.7% | 91.0% +/- 2.6% |
| Recall | 89.7% | 84.2% +/- 4.0% |
| Accuracy | 99.3% | 99.0% +/- 0.2% |

Validated on 5,391 real MALT trajectories. Statistical significance: p < 0.001 vs all baselines.

### Per-Category F1 (Combined Dataset, 5-Fold CV)

| Category | F1 | Samples | Notes |
|----------|:---:|:---:|-------|
| sys_exit | 100.0% | 56 | Faking passes with exit codes |
| test_manipulation | 100.0% | 101 | Rewriting tests to always pass |
| deceptive_cot | 100.0% | 48 | Planning exploits in hidden reasoning |
| alignment_faking | 100.0% | 33 | Pretending to follow rules while breaking them |
| sabotage | 100.0% | 33 | Breaking things on purpose |
| sycophancy | 100.0% | 33 | Telling users what they want to hear |
| oversight_subversion | 100.0% | 33 | Dodging monitoring systems |
| mock_exploit | 98.5% | 544 | Patching validators to skip checks (was 0% in v1.0) |
| clean | 100.0% | 921 | Legitimate trajectories |

### Baseline Comparisons

| Method | F1 | Precision | Recall |
|--------|:---:|:---:|:---:|
| **DistilBERT (Ours)** | **89.7%** | **89.7%** | **89.7%** |
| Regex Patterns | 4.9% | 2.6% | 38.5% |
| BoW + LogReg | 7.0% | 4.0% | 28.2% |
| Keyword Matching | 0.1% | 0.1% | 5.1% |

## Features (v1.3)

- 89.7% F1 on 5,391 MALT trajectories (DistilBERT)
- Mock exploit detection: 0% to 98.5% F1 (500 synthetic trajectories)
- HackBench standardized benchmark (4,300+ trajectories, 9 categories)
- Temporal AttentionClassifier for step-level escalation modeling
- Causal RMGI with Granger causality testing
- Evasion resistance testing (5 adversarial attack types)
- Dynamic threshold calibration
- Cross-model transfer study framework
- React 18 frontend with 9-page dark-mode dashboard
- Eval Workbench with JSONL import, custom rules, LLM judge scoring
- Dual LLM provider support (Anthropic / OpenAI)
- Python 3.9+ compatibility

## React Frontend

Dark-mode React 18 + TypeScript + Tailwind CSS v4 dashboard with 9 pages. Typography uses Space Grotesk for headings and numbers, Inter for body text, JetBrains Mono for code.

```bash
cd frontend && npm install && npm run dev
```

| Page | Description | Status |
|------|-------------|--------|
| Dashboard | Overview stats, risk distribution, category breakdown, RMGI timeline | Stable |
| Quick Analysis | Paste trajectory for instant analysis with radar chart | Stable |
| Timeline | Real-time hack/misalignment scores with RMGI tracking | Stable |
| Alerts | Flagged events with severity levels and acknowledgment | Stable |
| Cross-Model | Compare detection across model families | Stable |
| CoT Viewer | Chain-of-thought analysis with highlighted concerns | Stable |
| Eval Workbench | JSONL import, custom rules, batch evaluation, model comparison | BETA |
| Session Logs | History with session metadata and export | BETA |
| Settings | LLM providers, models, thresholds, detection rules | Stable |

### Screenshots

| | |
|:---:|:---:|
| ![Dashboard](assets/screenshots/dashboard.png) **Dashboard** | ![Quick Analysis](assets/screenshots/quick-analysis.png) **Quick Analysis** |
| ![Timeline](assets/screenshots/timeline.png) **Timeline** | ![Alerts](assets/screenshots/alerts.png) **Alerts** |
| ![Cross-Model](assets/screenshots/cross-model.png) **Cross-Model** | ![CoT Viewer](assets/screenshots/cot-viewer.png) **CoT Viewer** |
| ![Eval Workbench](assets/screenshots/eval-workbench.png) **Eval Workbench** | ![Session Logs](assets/screenshots/session-logs.png) **Session Logs** |
| ![Settings](assets/screenshots/settings.png) **Settings** | |

## HackBench Benchmark Dataset

Standardized benchmark for reward hacking detection research. 4,300+ trajectories across 9 categories with scenario-level train/val/test splits.

```python
from rewardhackwatch.rhw_bench.hackbench import HackBenchDataset
dataset = HackBenchDataset()
dataset.curate()
train, val, test = dataset.get_split("train"), dataset.get_split("val"), dataset.get_split("test")
```

<details>
<summary><h2>Architecture</h2></summary>

```
rewardhackwatch/
  core/
    detectors/           # Pattern (45 regex) + ML (DistilBERT) + AST detection
    analyzers/           # CoT analysis, complexity, obfuscation detection
    judges/              # LLM judges (Anthropic, OpenAI, Ollama)
    trackers/            # RMGI tracking, PELT changepoint, Causal RMGI
    calibration.py       # Dynamic threshold calibration
  training/              # DistilBERT + AttentionClassifier pipelines
  eval/                  # JSONL loader, rubric scoring, batch analysis, runs
  experiments/           # Transfer study, evasion attacks
  rhw_bench/             # HackBench dataset, generators, test cases
  api/                   # FastAPI REST server
  dashboard/             # Streamlit UI (legacy)
  cli.py                 # Command line interface
frontend/
  src/
    components/          # Sidebar, StatCard, ChartCard, PageHeader, Badge
    pages/               # Dashboard, QuickAnalysis, Timeline, Alerts,
                         # CrossModel, CoTViewer, JsonlAnalyzer, SessionLogs, Settings
    lib/                 # API client, demo data, types, utilities
  index.html             # Entry point with Google Fonts (Space Grotesk, Inter, JetBrains Mono)
paper/
  RewardHackWatch.pdf    # Research paper
```

</details>

<details>
<summary><h2>Detection Pipeline</h2></summary>

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
| Causal RMGI       | <-- Granger causality (v1.2)
| PELT Detection    | <-- Behavioral changepoints
+--------+---------+
         |
         v
    Alert / Report
```

</details>

<details>
<summary><h2>Figures</h2></summary>

| RMGI Transition | Architecture |
|:---:|:---:|
| ![RMGI transition detection](figures/rmgi_transition.png) | ![System architecture](figures/architecture.png) |
| **Benchmark** | **Categories** |
| ![Baseline comparison](figures/benchmark.png) | ![Per-category F1](figures/categories.png) |
| **Threshold** | **Calibration** |
| ![Threshold sensitivity](figures/threshold.png) | ![Model calibration](figures/calibration.png) |

</details>

<details>
<summary><h2>Related Work</h2></summary>

| Tool | Domain | Approach | Scope |
|------|--------|----------|-------|
| **RewardHackWatch** | LLM agents | Multi-layer (ML + pattern + RMGI) | Runtime detection + generalization tracking |
| RewardScope | Classical RL | Reward model analysis | Reward function debugging |
| OpenAI CoT Monitor | Reasoning models | Chain-of-thought scanning | Internal monitoring (not open-source) |
| SHADE-Arena | LLM agents | Evaluation framework | Benchmark, not detection |
| MALT | LLM agents | Trajectory dataset | Data only, no detector |

**Key difference:** RewardScope and classical RL safety tools target RL reward models and policy optimization. RewardHackWatch targets LLM agent trajectories - the code, reasoning, and actions produced by language models operating as autonomous agents. These are fundamentally different failure modes.

</details>

## Roadmap (v2.0)

- Activation probing for deception detection
- Steganography detection in CoT token choices
- Multi-agent scenario monitoring
- Real-time streaming API with WebSocket alerts
- HuggingFace dataset release (HackBench)
- Multi-language support

## RMGI Metric

The Reward-Misalignment Generalization Index detects when hacking begins correlating with misalignment:

```
RMGI(i, W) = rho(H[i-W+1:i], M[i-W+1:i])
```

H = hack score time series, M = misalignment score time series, W = window size (default: 10)

Transition detected when RMGI > 0.7, hack > 0.5, misalignment > 0.5 for 5+ consecutive steps. v1.2 adds Causal RMGI: weights correlation by Granger causality p-value.

See [RMGI Specification](docs/RMGI.md) for formal definition.

## Configuration

```bash
export ANTHROPIC_API_KEY="your-key"        # For Anthropic LLM judge
export OPENAI_API_KEY="your-key"           # For OpenAI LLM judge
export OLLAMA_HOST="http://localhost:11434" # For local Llama judge
export RHW_HACK_THRESHOLD="0.02"           # Detection threshold
```

```python
# Calibrate on your own data (recommended for production)
detector = RewardHackDetector()
detector.calibrate_threshold(clean_trajectories, percentile=99)
```

<details>
<summary><h3>LLM Judge Configuration</h3></summary>

The React frontend Settings page supports dual LLM provider configuration:

- **General LLM Provider** - Primary model for analysis and scoring
- **Independent Review Provider** - Second model for cross-validation (auto-review optional)

| Provider | Models |
|----------|--------|
| Anthropic | claude-opus-4-6, claude-sonnet-4-5-20250929, claude-haiku-4-5-20251001 |
| OpenAI | gpt-5.2, gpt-4o, gpt-4o-mini |

Each provider card allows configuring: API key, model selection, temperature (0-2), and max tokens. Settings persist in localStorage.

</details>

## Documentation

- [Technical Report](docs/TECHNICAL.md) - Full methodology and experiments
- [RMGI Specification](docs/RMGI.md) - Formal metric definition
- [Architecture](ARCHITECTURE.md) - System design diagrams
- [Paper](paper/RewardHackWatch.pdf) - Research paper

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

Built for AI safety researchers and engineers. [Follow on X](https://x.com/aerosta_ai)

**Keywords:** reward hacking detection, LLM safety, AI alignment, misalignment detection, RLHF safety, reward tampering, specification gaming, AI safety tools, language model monitoring, agent safety
