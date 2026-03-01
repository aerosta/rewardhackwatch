# Changelog

All notable changes to RewardHackWatch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-03-01

### Added

#### React Frontend
- Full React 18 + TypeScript + Tailwind CSS v4 dashboard replacing Streamlit
- 9-page SPA: Dashboard, Quick Analysis, Timeline, Alerts, Cross-Model, CoT Viewer, Eval Workbench, Session Logs, Settings
- Dark-mode developer tool UI with `#20232B` theme and JetBrains Mono
- Stat cards with accent-colored left borders
- Recharts-based visualizations (donut, bar, area, line, radar charts)
- Code splitting with manual chunks for react and recharts

#### Eval Workbench (formerly JSONL Analyzer)

- Drag-drop JSONL import with auto-detect schema (conversation, completion, I/O, Q&A)
- Raw JSONL paste input with parse preview
- Custom evaluation rules: regex, keyword, length, LLM judge types
- 8 built-in rule templates (rogue code, mock exploit, safety, deceptive CoT, etc.)
- Rule management: add, toggle, delete, weight and severity configuration
- Batch evaluation engine with progress bar
- Weighted scoring system with A-F grade badges
- Per-entry detail view with full conversation and rule results
- Per-rule pass rate charts and grade distribution visualization
- Model 1 vs Model 2 side-by-side comparison with winner indicator
- JSON and CSV export

#### Settings

- Dual LLM provider cards: General LLM Provider and Independent Review Provider
- Provider dropdown (Anthropic / OpenAI) with dynamic model list per provider
- Eye icon toggle for API key visibility
- Number inputs for temperature and max tokens
- Auto-review toggle on Independent Review Provider card
- Per-section save buttons
- Settings persistence via localStorage

#### Typography

- Space Grotesk font for headings, stat numbers, chart titles, sidebar logo
- Inter font for body text and labels
- JetBrains Mono for code blocks and monospace elements

### Changed

- License changed to Apache 2.0
- Version bumped to 1.3.0
- Color scheme updated: bg `#20232B`, cards `#292A30`, sidebar `#1B1D24`, borders `#363840`
- All hardcoded colors in chart components replaced with theme variables
- Sidebar label: "JSONL Analyzer" renamed to "Eval Workbench"
- Sidebar logo: "RewardHackWatch" as single word
- All chart containers minimum 320px height
- American spelling throughout ("analyzed" not "analysed")
- Dashboard category bar chart: shorter axis labels, wider bar area
- Dashboard chart title: "Detections by Category" shortened to "Category Breakdown" to prevent truncation in 3-column layout
- Model labels in Settings now display exact model strings instead of friendly names
- Settings cards now full-width layout instead of narrow centered column
- Eval Workbench format labels: generic "Model 1" / "Model 2" / "Custom Format" instead of provider-specific names

### Fixed

- CITATION.cff placeholder values replaced with actual metadata
- SECURITY.md placeholder email replaced
- Organization name normalized to "Aerosta" across all files
- Sidebar truncation of long nav labels
- Removed all em dashes from README
- CHANGELOG v1.0.0 date corrected to 2025-12-09
- Category label "Deceptive Cot" corrected to "Deceptive CoT" via acronym-aware formatter
- TypeScript build error in Eval Workbench chart tooltip formatter

## [1.2.0] - 2026-02-28

### Added

#### Public API
- `RewardHackDetector` high-level facade class matching README Quick Start examples
- `AnalysisResult.detections` property (alias for `pattern_matches`)
- `RewardHackDetector.calibrate_threshold()` for dynamic threshold calibration
- `ThresholdCalibrator` with percentile, isotonic, and beta calibration methods
- `rewardhackwatch calibrate` CLI command for threshold calibration on clean data

#### Synthetic Data & Benchmarks
- `MockExploitGenerator` with 8 code templates for mock/monkeypatch exploits
- `CleanTrajectoryGenerator` for balanced dataset creation
- `HackBenchDataset` standardized benchmark (4,300+ trajectories, 9 categories)
- `scripts/generate_synthetic_data.py` and `scripts/build_hackbench.py`

#### Temporal Modeling
- `StepLevelFeatureExtractor` for per-step feature extraction
- `StepSequenceDataset` with padding and attention masks
- `TemporalTrainer` training pipeline for `AttentionClassifier`
- Positional encoding in `AttentionClassifier` for step ordering

#### Research Tools
- `TransferStudyRunner` for cross-model transfer matrix experiments
- `EvasionAttackSuite` with 5 adversarial attack types
- `CausalRMGI` with Granger causality testing and intervention analysis
- `scripts/run_transfer_study.py` and `scripts/run_evasion_tests.py`

#### Dashboard
- Dark-mode developer tool UI (#1a1a2e theme)
- Stat cards with accent-colored borders
- Donut charts for category distribution
- Per-category F1 bar charts
- Cross-Model comparison tab
- Recent Analysis table
- Quick Analysis as default landing tab
- PDF and JSON export buttons

### Fixed
- `__version__` was "0.1.0" instead of matching pyproject.toml
- `__author__` was "Your Name" instead of "Aerosta"
- `RewardHackDetector` class referenced in README didn't exist
- `result.detections` referenced in Quick Start didn't exist
- Python 3.9 compatibility: added `from __future__ import annotations` to 52 files
- Missing `[dashboard]` optional dependency group in pyproject.toml

### Changed
- Version bumped to 1.2.0
- Development status upgraded from Alpha to Beta

## [1.0.0] - 2025-12-09

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
- Ensemble voting for multi-signal detection

#### API and Dashboard
- FastAPI server with `/analyze` and `/status` endpoints
- Streamlit dashboard for real-time trajectory monitoring
- RMGI visualization with rolling correlation plots
- Risk level assessment (none/low/medium/high/critical)

#### Benchmark Suite (RHW-Bench)
- Integration with SHADE-Arena evaluation framework
- MALT trajectory loader for training data
- Synthetic trajectory generation for ablation studies
- Benchmark runner with multiple dataset support

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

## [Unreleased]

### Planned
- Activation probing for deception detection
- Steganography detection in CoT token choices
- Multi-agent scenario monitoring
- Real-time streaming API with WebSocket alerts
