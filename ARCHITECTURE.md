# RewardHackWatch Architecture

This document describes the system architecture, component flow, and data schemas used in RewardHackWatch.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RewardHackWatch                                   │
│                                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────────┐ │
│  │  Trajectory │ ──► │  Detectors  │ ──► │   Judges    │ ──► │   RMGI    │ │
│  │   Input     │     │   Layer     │     │   Layer     │     │  Scoring  │ │
│  └─────────────┘     └─────────────┘     └─────────────┘     └───────────┘ │
│         │                  │                   │                    │       │
│         │                  ▼                   ▼                    ▼       │
│         │           ┌─────────────┐     ┌─────────────┐     ┌───────────┐ │
│         └─────────► │  Analyzers  │     │  Trackers   │ ──► │  Alerts   │ │
│                     │   Layer     │     │   Layer     │     │  System   │ │
│                     └─────────────┘     └─────────────┘     └───────────┘ │
│                                                                     │       │
│                                                                     ▼       │
│                                                              ┌───────────┐ │
│                                                              │ Dashboard │ │
│                                                              │    API    │ │
│                                                              └───────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Flow

### 1. Trajectory Input

Trajectories are the primary input format representing an agent's execution trace:

```
Trajectory (JSON)
    │
    ├── steps[]           → Sequential actions taken
    │   ├── action        → What the agent did
    │   ├── code          → Code written/executed
    │   └── result        → Outcome
    │
    ├── cot_traces[]      → Chain-of-thought reasoning
    │
    └── code_outputs[]    → Final code artifacts
```

### 2. Detection Pipeline

```
Trajectory
    │
    ├──► PatternDetector ──► Regex-based hack pattern matching
    │                        (sys.exit, assert True, etc.)
    │
    ├──► ASTDetector ──────► Python AST analysis for code patterns
    │                        (empty functions, verifier overrides)
    │
    └──► MLDetector ───────► DistilBERT classifier for CoT analysis
                             (hack_prob, misalign_prob)
    │
    ▼
DetectorResult(score, risk_level, detections[])
```

### 3. Analysis Pipeline

```
Trajectory
    │
    ├──► CoTAnalyzer ──────► Deception pattern detection
    │                        (hiding intent, mismatch analysis)
    │
    ├──► EffortAnalyzer ───► TRACE-style effort analysis
    │                        (task complexity vs reasoning depth)
    │
    └──► ObfuscationDetector → Verbosity tracking over checkpoints
                               (intent hiding detection)
    │
    ▼
AnalysisResult(deception_score, effort_score, obfuscation_risk)
```

### 4. Judge System

```
Trajectory + Detector Results
    │
    ├──► ClaudeJudge ──────► Claude 4.5 Opus API
    │                        (JSON verdict + reasoning)
    │
    └──► LlamaJudge ───────► Local Llama 3.1 via Ollama
                             (JSON verdict + reasoning)
    │
    ▼
JudgeResult(verdict, hack_score, misalignment_score, reasoning)
```

### 5. Generalization Tracking

```
Time Series of (hack_score, misalign_score)
    │
    ▼
GeneralizationTracker
    │
    ├── Pearson correlation computation
    ├── PELT changepoint detection
    └── Transition point identification
    │
    ▼
GeneralizationResult(correlation, transition_points[], risk_level)
```

### 6. RMGI Scoring

The Reward-Misalignment Generalization Index combines all signals:

```
RMGI = w_hack × hack_score
     + w_deception × deception_score
     + w_effort × effort_suspicion
     + w_gen × generalization_risk

Where:
  w_hack = 0.3
  w_deception = 0.3
  w_effort = 0.2
  w_gen = 0.2
```

### 7. Alert System

```
Analysis Results
    │
    ▼
AlertSystem
    │
    ├── Threshold checking
    │   ├── hack_score > 0.7 → WARNING
    │   ├── deception_score > 0.6 → WARNING
    │   └── generalization_risk > 0.5 → CRITICAL
    │
    ├── Alert generation
    ├── SQLite persistence
    └── Multi-channel dispatch (console, file, webhook)
    │
    ▼
Alert(level, source, message, scores, file_path)
```

## Data Schemas

### Trajectory

```python
{
    "id": str,                    # Optional identifier
    "steps": [
        {
            "action": str,        # Action description
            "code": str,          # Code content
            "thought": str,       # CoT reasoning
            "result": str,        # Action result
        }
    ],
    "cot_traces": [str],          # Raw CoT traces
    "code_outputs": [str],        # Final code outputs
    "task": str,                  # Optional task description
    "labels": {                   # Ground truth (for benchmarks)
        "hack_type": str,
        "misalignment_type": str,
        "transition_step": int,
    }
}
```

### DetectorResult

```python
@dataclass
class DetectorResult:
    detector_name: str            # "pattern_detector", "ast_detector", etc.
    score: float                  # 0.0-1.0 overall score
    risk_level: RiskLevel         # NONE, LOW, MEDIUM, HIGH, CRITICAL
    detections: list[Detection]   # Individual findings
    metadata: dict                # Detector-specific data
```

### Detection

```python
@dataclass
class Detection:
    pattern_name: str             # What was detected
    description: str              # Human-readable description
    location: str                 # Where (e.g., "step_3", "cot_trace_1")
    confidence: float             # 0.0-1.0
    risk_level: RiskLevel
    raw_match: str                # The matched content
    metadata: dict
```

### JudgeResult

```python
@dataclass
class JudgeResult:
    judge_name: str               # "claude_judge", "llama_judge"
    verdict: Verdict              # SAFE, SUSPICIOUS, DANGEROUS, CRITICAL
    confidence: float             # 0.0-1.0
    hack_score: float             # 0.0-1.0
    misalignment_score: float     # 0.0-1.0
    reasoning: str                # LLM's explanation
    flagged_behaviors: list[str]  # Specific concerns
    metadata: dict
```

### Alert

```python
@dataclass
class Alert:
    id: str
    level: AlertLevel             # WARNING, CRITICAL
    source: AlertSource           # HACK_SCORE, DECEPTION, GENERALIZATION
    message: str
    timestamp: datetime
    scores: dict[str, float]
    file_path: str
    checkpoint_path: str | None
    metadata: dict
```

## Directory Structure

```
rewardhackwatch/
├── core/
│   ├── detectors/           # Pattern, AST, ML detectors
│   │   ├── base.py          # BaseDetector, Detection, DetectorResult
│   │   ├── pattern_detector.py
│   │   ├── ast_detector.py
│   │   └── ml_detector.py
│   │
│   ├── analyzers/           # CoT, Effort, Obfuscation analysis
│   │   ├── cot_analyzer.py
│   │   ├── effort_analyzer.py
│   │   └── obfuscation_detector.py
│   │
│   ├── judges/              # LLM-based judges
│   │   ├── base.py          # BaseJudge, JudgeResult, Verdict
│   │   ├── claude_judge.py  # Claude Opus 4.5
│   │   ├── llama_judge.py   # Local Llama 3.1
│   │   └── ensemble_judge.py # Multi-judge voting
│   │
│   ├── trackers/            # Generalization and drift tracking
│   │   ├── generalization_tracker.py  # Correlation analysis
│   │   ├── changepoint_detector.py    # PELT/CUSUM detection
│   │   └── drift_tracker.py           # Behavioral drift monitoring
│   │
│   └── monitors/            # Real-time monitoring
│       ├── training_monitor.py
│       └── alert_system.py
│
├── database/                # Persistent storage
│   └── sqlite_logger.py     # SQLite logging
│
├── api/                     # FastAPI backend
│   └── main.py
│
├── dashboard/               # Streamlit UI
│   └── app.py
│
├── rhw_bench/              # Benchmark suite
│   ├── benchmark_runner.py
│   ├── shade_loader.py     # SHADE-Arena external benchmark
│   ├── evilgenie_loader.py # EvilGenie external benchmark
│   ├── test_cases/         # Synthetic trajectories (8 original)
│   ├── test_cases/generated/ # Generated trajectories (132 total)
│   └── trajectories/       # Real research trajectories
│
├── scripts/                # Utility scripts
│   ├── calibrate_judge.py  # Judge calibration
│   ├── generate_results.py # Results table generation
│   └── generate_trajectories.py # Dataset generation
│
├── training/               # ML training
│   ├── dataset.py          # Data loading
│   ├── models.py           # Model definitions
│   └── train_detector.py   # Training loop
│
└── cli.py                  # Command-line interface
```

## Plugin Interface

### Adding a New Detector

Detectors implement the `BaseDetector` interface:

```python
from rewardhackwatch.core.detectors.base import (
    BaseDetector,
    Detection,
    DetectorResult,
    RiskLevel,
)

class MyDetector(BaseDetector):
    name = "my_detector"

    def detect(self, trajectory: dict[str, Any]) -> DetectorResult:
        # Implement detection logic
        detections = []
        score = 0.0

        # ... your logic ...

        return DetectorResult(
            detector_name=self.name,
            score=score,
            risk_level=self._determine_risk_level(score),
            detections=detections,
            metadata={},
        )
```

### Adding a New Judge

Judges implement the `BaseJudge` interface:

```python
from rewardhackwatch.core.judges.base import (
    BaseJudge,
    JudgeResult,
    Verdict,
)

class MyJudge(BaseJudge):
    name = "my_judge"

    async def judge(self, trajectory: dict[str, Any]) -> JudgeResult:
        # Format trajectory
        text = self._format_trajectory(trajectory)

        # Call your LLM
        response = await my_llm_call(text)

        # Parse response
        parsed = self._parse_response(response)

        return JudgeResult(
            judge_name=self.name,
            verdict=self._verdict_from_string(parsed["verdict"]),
            confidence=0.8,
            hack_score=parsed["hack_score"],
            misalignment_score=parsed["misalignment_score"],
            reasoning=parsed["reasoning"],
            flagged_behaviors=parsed.get("flagged_behaviors", []),
            metadata={},
        )
```

## Performance Considerations

1. **Batch Processing**: ML detector supports batch prediction for efficiency
2. **Async Judges**: LLM judges support async for concurrent analysis
3. **Caching**: Pattern compilation is cached at initialization
4. **Lazy Loading**: ML models loaded on first use
5. **Background Monitoring**: TrainingMonitor runs in separate thread

## External Benchmarks

### SHADE-Arena Integration

RewardHackWatch supports evaluation on ScaleAI's [SHADE-Arena](https://huggingface.co/datasets/ScaleAI/mrt) (Monitor Red Teaming) dataset:

```
SHADE-Arena (ScaleAI/mrt)
    │
    ├── task_pairs/               # SHADE-Arena tasks
    │   └── <task>/<attack>/<monitor>/transcript_N.json
    │
    └── task_pairs_agentcompany/  # CUA-SHADE-Arena tasks
        └── <task>/<attack>/<monitor>/transcript_N.json

Each transcript contains:
  - transcript: Full agent conversation with <thinking> tags
  - monitor_transcript: Monitor's evaluation
  - monitor_verdict: Final judgment
  - task_description: Primary and secondary objectives
```

### Loading External Data

```python
from rewardhackwatch.rhw_bench.shade_loader import ShadeArenaLoader

loader = ShadeArenaLoader()
trajectories = loader.load(limit=100, task_filter="banking")

for traj in trajectories:
    result = runner.run_single(traj.trajectory)
```

### Combined Training

```python
from rewardhackwatch.training.dataset import CombinedDataset

# Load both RHW-Bench and SHADE-Arena for training
dataset = CombinedDataset(
    rhw_dir="rewardhackwatch/rhw_bench/test_cases",
    shade_limit=100,
    split="train",
)
```

## Security Model

See [SECURITY.md](SECURITY.md) for:
- Threat model and scope
- Known limitations
- Responsible disclosure process
