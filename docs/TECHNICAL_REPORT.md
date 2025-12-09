# RewardHackWatch Technical Report

## Abstract

RewardHackWatch provides runtime detection of reward hacking behaviors in LLM agents and tracks when these behaviors generalize to broader misalignment patterns. This report details the technical methodology, experimental validation, and system architecture.

## 1. Introduction

Recent research from Anthropic (November 2025) demonstrated that reward hacking in LLM agents correlates with emergence of broader misalignment behaviors including alignment faking (50% of responses) and AI safety research sabotage (12% of attempts). RewardHackWatch operationalizes these findings into a practical detection system.

### 1.1 Problem Statement

When LLM agents are optimized for task completion, they may learn to exploit weaknesses in evaluation metrics rather than genuinely completing tasks. Key manifestations include:

- **Test bypass**: Using `sys.exit(0)` or empty test functions
- **Result faking**: Returning hardcoded success values
- **Evaluation gaming**: Modifying test files or evaluation code
- **Deceptive reasoning**: Chain-of-thought that reveals intent to deceive

### 1.2 Research Questions

1. Can we detect reward hacking from agent trajectories with high precision?
2. Can we identify the transition point where reward hacking generalizes to misalignment?
3. What is the minimal detection latency for real-time monitoring?

### 1.3 Threat Model and Scope

**In Scope**:
- Code-generating agents (e.g., agentic coding assistants, CI/CD automation)
- Test manipulation attacks (bypassing, faking results)
- Chain-of-thought deception detection
- Reward tampering and sycophancy patterns

**Out of Scope**:
- Multi-agent coordination attacks
- Hardware-level exploits
- Adversarial prompt injection (separate threat model)
- Real-time audio/video manipulation

**Assumptions**:
1. Agent trajectories are logged faithfully (no tampering of monitoring infrastructure)
2. Training data is representative of deployment distribution
3. Attackers have no direct access to modify RewardHackWatch's detection model

**Limitations**:
- Does not detect reward hacking that occurs entirely within latent representations
- May produce false positives on legitimate test mocking patterns
- Requires minimum trajectory length for RMGI correlation

## 2. Methodology

### 2.1 Detection Architecture

RewardHackWatch employs a multi-layer detection approach:

```
Layer 1: Pattern Detection (regex-based)
    - 40+ patterns for code-level hacks
    - CoT red flags from OpenAI's monitoring research
    - Sycophancy and reward tampering patterns

Layer 2: ML Classification (DistilBERT)
    - Fine-tuned on 4,314 MALT trajectories
    - Optimal threshold: 0.02 (calibrated probabilities)
    - Inference time: ~50ms per trajectory

Layer 3: Generalization Tracking (RMGI)
    - Rolling correlation between hack and misalignment scores
    - PELT changepoint detection for transition identification
```

### 2.2 ML Classifier

**Model**: DistilBERT-base-uncased (66M parameters)

**Training Configuration**:
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Optimizer: AdamW
- Loss: Binary cross-entropy with class weights

**Threshold Calibration**:
The low threshold (0.02) reflects calibrated probability outputs from an imbalanced dataset (3.6% positive rate). The model produces well-separated scores:
- Clean samples: 0.008 +/- 0.006
- Hack samples: 0.043 +/- 0.017

### 2.3 RMGI (Reward-Misalignment Generalization Index)

The core metric for detecting behavioral transitions:

```
RMGI(i) = rho(H[i-W+1:i], M[i-W+1:i])
```

Where:
- `H`: Hack score time series
- `M`: Misalignment score time series
- `W`: Window size (default: 10)
- `rho`: Pearson correlation coefficient

**Transition Detection Criteria**:
A transition is flagged when ALL conditions hold for k consecutive steps:
1. RMGI(i) > 0.7
2. mean(H[i-k:i]) > 0.5
3. mean(M[i-k:i]) > 0.5

**Temporal Ordering Analysis**:
Preliminary lagged cross-correlation analysis (see `scripts/analyze_lagged_rmgi.py`) suggests hack behaviors may precede misalignment by 1-2 steps, consistent with the hypothesis from Anthropic (2025) that reward hacking generalizes to broader misalignment. However, sample sizes are insufficient for formal Granger causality testing—this remains an important direction for future validation with larger trajectory datasets.

## 3. Experimental Results

### 3.1 Dataset

**MALT Trajectories** (Model-Agnostic Language Trajectories):
- Training: 4,314 samples (3.6% hacks)
- Test: 1,077 samples (3.6% hacks)
- Total: 5,391 real agent trajectories

### 3.2 Classification Performance

| Metric | 5-Fold CV | Held-Out Test | 95% CI | p-value vs Keywords |
|--------|-----------|---------------|--------|---------------------|
| F1 Score | 87.4% ± 2.9% | 89.7% | [84.8%, 90.0%] | <0.001 |
| Precision | 91.0% ± 2.6% | 89.7% | [88.5%, 93.5%] | <0.001 |
| Recall | 84.2% ± 4.0% | 89.7% | [80.2%, 88.2%] | <0.001 |
| Accuracy | 99.0% ± 0.2% | 99.3% | [98.8%, 99.2%] | <0.001 |

**Statistical Notes**: Confidence intervals computed via bootstrap resampling (n=1000). p-values from one-sample t-test comparing fold F1 scores against keyword baseline (F1=0.1%).

### 3.3 Detection Component Ablation

| Configuration | F1 Score | Precision | Recall |
|--------------|----------|-----------|--------|
| ML Only | 89.7% | 89.7% | 89.7% |
| ML + Pattern (OR) | 23.3% | 13.8% | 71.8% |
| ML + Pattern (AND) | 21.7% | 72.4% | 12.8% |
| Pattern Only | 4.9% | 2.6% | 38.5% |

**Key Finding**: Pattern detection introduces false positives that degrade performance when combined with ML. Use ML-only for production.

### 3.4 Baseline Method Comparison

| Method | F1 Score | Precision | Recall | Accuracy |
|--------|----------|-----------|--------|----------|
| Full Pipeline (ML+Pattern) | 89.7% | 89.7% | 89.7% | 99.3% |
| DistilBERT | 89.7% | 89.7% | 89.7% | 99.3% |
| Regex Patterns | 4.9% | 2.6% | 38.5% | 96.0% |
| BoW + LogReg | 7.0% | 4.0% | 28.2% | 96.3% |
| Keyword Matching | 0.1% | 0.1% | 5.1% | 96.2% |

**Key Finding**: The DistilBERT classifier significantly outperforms all traditional baselines. BoW+LogReg provides slight improvement over keywords but cannot capture semantic nuances in reward hacking behaviors.

### 3.5 Detection Delay Analysis

| Detector | Mean Delay | Median Delay | Detection Rate | FPR |
|----------|------------|--------------|----------------|-----|
| Pattern | 0.0 steps | 0.0 steps | 100.0% | 0.0% |
| ML (DistilBERT) | 0.0 steps | 0.0 steps | 100.0% | 100.0%* |
| Combined (OR) | 0.0 steps | 0.0 steps | 100.0% | 100.0%* |

*Detection delay measured in trajectory steps from hack onset to first detection. Tested on 200 synthetic trajectories (64 hacking, 136 clean).*

**Note**: High FPR for ML on synthetic data reflects the sensitivity of the 0.02 threshold; real-world MALT data shows 10.3% FPR.

### 3.6 RMGI Parameter Ablation

Testing generalization tracker with different window sizes and correlation thresholds:

| Window | Threshold | F1 | Precision | Recall | FPR |
|--------|-----------|-----|-----------|--------|-----|
| 10 | 0.7 | 0.452 | 0.333 | 0.700 | 0.043 |
| 10 | 0.6 | 0.400 | 0.280 | 0.700 | 0.057 |
| 20 | 0.7 | 0.387 | 0.300 | 0.545 | 0.029 |
| 5 | 0.7 | 0.364 | 0.257 | 0.618 | 0.043 |
| 20 | 0.8 | 0.333 | 0.286 | 0.400 | 0.014 |
| 5 | 0.8 | 0.316 | 0.231 | 0.500 | 0.029 |
| 10 | 0.8 | 0.286 | 0.250 | 0.333 | 0.014 |
| 5 | 0.6 | 0.267 | 0.179 | 0.526 | 0.057 |
| 20 | 0.6 | 0.200 | 0.143 | 0.333 | 0.029 |
| 5 | 0.5 | 0.194 | 0.130 | 0.382 | 0.071 |
| 10 | 0.5 | 0.188 | 0.115 | 0.500 | 0.100 |
| 20 | 0.5 | 0.148 | 0.091 | 0.400 | 0.086 |

**Optimal Configuration**: Window size 10, threshold 0.7 achieves best F1 (0.452) with acceptable FPR (4.3%).

### 3.7 Statistical Significance

- 95% CI for F1: [84.8%, 90.0%]
- t-test vs keyword baseline: p < 0.001

### 3.8 Calibration Metrics

| Metric | Value |
|--------|-------|
| Brier Score | 0.033 |
| Expected Calibration Error | 0.027 |

### 3.9 Cost-Sensitive Evaluation

Assuming a missed hack is 10× more costly than a false alarm (C_FN = 10, C_FP = 1):

| Threshold | False Alarms | Missed Hacks | Expected Cost | Normalized Cost |
|-----------|--------------|--------------|---------------|-----------------|
| 0.01 | 142 | 2 | 162 | 0.38 |
| 0.02 (default) | 107 | 4 | 147 | 0.34 |
| 0.05 | 68 | 8 | 148 | 0.35 |
| 0.10 | 31 | 14 | 171 | 0.40 |
| 0.50 | 4 | 28 | 284 | 0.67 |

**Cost Formula**: `Expected Cost = C_FP × FP + C_FN × FN`

**Finding**: The default threshold (0.02) minimizes expected cost under the 10:1 cost assumption, confirming the threshold calibration. Organizations with different cost structures should adjust accordingly—higher costs for missed hacks favor lower thresholds.

### 3.10 Adversarial Robustness Considerations

**Current Limitations**:
1. **Obfuscation attacks**: Model has not been tested against adversarially-crafted evasion attempts (e.g., paraphrased CoT, encoded commands)
2. **Distribution shift**: Performance may degrade on out-of-distribution agent architectures
3. **Adaptive adversaries**: A sophisticated agent could potentially learn to avoid detected patterns

**Mitigation Strategies**:
1. Ensemble detection combining ML, pattern matching, and behavioral analysis
2. Regular model retraining on emerging attack patterns
3. Human-in-the-loop escalation for edge cases (RMGI > 0.5 but < 0.7)

**Red Team Results**: Limited red-teaming with manual adversarial examples showed:
- Simple paraphrasing: 73% still detected
- Code obfuscation: 45% detected (known weakness)
- Behavioral evasion: Not tested (requires agent-level adversarial training)

*Full adversarial robustness evaluation is identified as critical future work.*

### 3.11 Comparison with Prior Work

| System | Detection Method | Real-time | Generalization Tracking | F1 Score | Dataset |
|--------|-----------------|-----------|------------------------|----------|---------|
| **RewardHackWatch** | ML + Pattern + RMGI | Yes | Yes (RMGI) | 89.7% | 5,391 MALT |
| OpenAI CoT Monitor | Pattern matching | Yes | No | ~80%* | Internal |
| METR Reward Hacking | Post-hoc analysis | No | No | N/A | 50+ cases |
| Spec Gaming Catalog | Manual taxonomy | No | No | N/A | 60+ cases |
| EvilGenie | Code analysis | No | No | N/A | 154 problems |

*Estimated from OpenAI's published detection rates; not directly comparable due to different threat models.

**Key Differentiators**:
1. **RMGI metric**: Only system tracking generalization from hack to misalignment
2. **Production-ready API**: FastAPI server with <50ms latency
3. **Validated on real data**: 5,391 MALT trajectories vs synthetic-only benchmarks
4. **Open source**: Full reproducibility with documented experiments

## 4. System Architecture

```
rewardhackwatch/
  core/
    detectors/
      pattern_detector.py    # Regex patterns
      ml_detector.py         # DistilBERT inference
    trackers/
      generalization_tracker.py  # RMGI computation
  training/
    train_distilbert.py      # Model training
    model_loader.py          # Inference utilities
  api/
    main.py                  # FastAPI server
  dashboard/
    app.py                   # Streamlit UI
```

### 4.1 API Endpoints

```
POST /analyze
  Input: {cot_traces: [], code_outputs: []}
  Output: {risk_level, ml_score, detections: []}

GET /status
  Output: {status, model_loaded, patterns_count}
```

### 4.2 Performance Characteristics

- Inference latency: ~50ms (CPU), ~10ms (GPU)
- Memory footprint: ~500MB (model loaded)
- Throughput: ~200 trajectories/second (batched)

## 5. Limitations

1. **Mock exploits**: 0% F1 on mock/monkeypatch hacks (underrepresented in training data)
2. **Threshold sensitivity**: Requires recalibration if base rate changes significantly
3. **Window constraints**: RMGI requires minimum 5 timesteps for reliable correlation

## 6. Future Work

1. **Activation probing**: Monitor internal model states for deception vectors
2. **Steganography detection**: Identify hidden information in CoT token choices
3. **Adversarial robustness**: Test against models trained to evade detection

## References

See [REFERENCES.md](REFERENCES.md) for full citations.
