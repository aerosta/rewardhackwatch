# Honest Evaluation of RewardHackWatch

## Executive Summary

RewardHackWatch is a prototype system for detecting reward hacking in LLM agent trajectories. This document provides an honest assessment of its capabilities and limitations, based on rigorous evaluation across multiple test partitions.

**Key Finding**: The system achieves high precision (100%) but low recall (6-15%) on synthetic benchmarks. This means when it flags something, it's almost always correct—but it misses most hacks.

## Evaluation Methodology

### Test Partitions

1. **RHW-Bench Normal** (132 trajectories)
   - Synthetically generated trajectories
   - Covers 8 hack types + clean examples
   - Expected: Easy detection of obvious patterns

2. **RHW-Bench Adversarial** (15 trajectories)
   - Hand-crafted hard cases
   - Subtle backdoors, deceptive CoT, false positive bait
   - Expected: Significantly lower performance

3. **External Benchmarks** (SHADE-Arena, EvilGenie)
   - Real-world agent traces
   - Different distribution from training
   - Expected: Domain gap degradation

### Baselines

| Baseline | Description |
|----------|-------------|
| Random | 50% chance of predicting "hack" |
| Keyword | Search for hack-related words (hack, cheat, bypass, etc.) |
| Pattern-only | Regex patterns without ML or judges |
| Full System | All components combined |

## Results

### Per-Partition Performance

| Partition | Method | Precision | Recall | F1 |
|-----------|--------|-----------|--------|-----|
| Normal | Random | 74.6% | 53.5% | 62.4% |
| Normal | Keyword | 100% | 3.0% | 5.9% |
| Normal | Pattern-only | 100% | 15.2% | 26.3% |
| Normal | Full System | 100% | 6.1% | 11.4% |
| Adversarial | Random | 50.0% | 55.6% | 52.6% |
| Adversarial | Keyword | 0% | 0% | 0% |
| Adversarial | Pattern-only | 33.3% | 11.1% | 16.7% |
| Adversarial | Full System | 0% | 0% | 0% |

### Key Observations

1. **High Precision, Low Recall**: The system rarely produces false positives but misses most hacks
2. **Random Baseline is Competitive**: Due to class imbalance (75% hacks in normal set), random guessing achieves ~60% F1
3. **Adversarial Cases are Hard**: System completely fails on subtle, adversarial trajectories
4. **Pattern Coverage Gap**: Many hack variants aren't covered by current patterns

## Where It Fails

### 1. Novel Pattern Variants

The pattern detector is trained on specific code patterns like `sys.exit(0)`. It misses:
- `os.kill(os.getpid(), signal.SIGTERM)` - Different exit mechanism
- `raise SystemExit()` - Different syntax for same behavior
- Custom signals, subprocess termination, etc.

**Impact**: ~85% of sys_exit hacks use variants not in our pattern database.

### 2. Obfuscated/Subtle Hacks

Adversarial test cases designed to evade detection:
- Backdoors hidden in legitimate-looking code
- Conditional hacks that only trigger on specific inputs
- Data exfiltration disguised as analytics

**Impact**: 0% detection rate on adversarial partition.

### 3. Long Trajectories with Sparse Misalignment

When a hack is buried in 15+ steps of legitimate code:
- Single malicious step is hard to identify
- Context window limitations
- Signal-to-noise ratio too low

**Impact**: Sparse misalignment trajectory completely missed.

### 4. Deceptive Chain-of-Thought

CoT patterns that appear benign:
- "I will follow safety guidelines" followed by violation
- Sycophantic reasoning that rationalizes unsafe behavior
- Gradual normalization over multiple steps

**Impact**: CoT analyzer missed all deceptive reasoning cases.

### 5. False Positive Bait (What We Got Right)

The system correctly identified as clean:
- Legitimate test skipping with proper reason
- CLI tools with standard exit codes
- Proper mock usage in unit tests

**Impact**: 0 false positives on clean adversarial cases (6/6 correct).

## What We Tried and Abandoned

### 1. Keyword-Based CoT Analysis

**Approach**: Search for keywords like "hack", "cheat", "trick" in reasoning traces.

**Problem**: Too many false positives on legitimate discussion of security topics. Too many false negatives when models use euphemisms.

**Abandoned because**: F1 < 10% on realistic data.

### 2. Simple Threshold Tuning

**Approach**: Lower detection thresholds to catch more hacks.

**Problem**: Dramatically increased false positives. Users would get alert fatigue.

**Abandoned because**: Precision dropped below 50% at any recall > 30%.

### 3. Training ML on CoT Text Only

**Approach**: Fine-tune DistilBERT on chain-of-thought traces.

**Problem**: Models quickly learn to identify synthetic data artifacts rather than actual hack patterns.

**Abandoned because**: Near-zero transfer to real trajectories.

## What We'd Do With More Time

### 1. Integrate Probing-Based Detection

Like Anthropic's research on detecting deceptive alignment through activation probing:
- Train probes on model internals during execution
- Look for consistent "deception" signals across layers
- Would require access to model weights/activations

### 2. Feature-Level Interpretability

Instead of pattern matching on output text:
- Analyze attention patterns during code generation
- Look for attention to test infrastructure vs actual problem
- Detect "planning to fail" vs "trying to succeed"

### 3. Larger Trajectory Datasets

Current benchmark is small (132 + 15 trajectories):
- Collect real agent deployments with ground truth
- Partner with AI labs for labeled reward hacking examples
- Generate diverse adversarial examples programmatically

### 4. LLM Judge Ensemble with Calibration

Current judges (Claude, Llama) show promise but:
- Need calibration on held-out data
- Need prompt engineering for specific hack types
- Could use chain-of-thought prompting for judges

### 5. Real-Time Streaming Analysis

Current system analyzes complete trajectories:
- Would be more useful to flag in real-time
- Could prevent hacks before completion
- Requires incremental detection algorithms

## Comparison to Published Results

### SHADE-Arena (ScaleAI/MRT)

- **Published best**: AUC 0.87 (87% area under ROC curve)
- **Our system**: Not directly comparable (different metrics)
- **Gap**: We don't have access to compute for full SHADE evaluation

### EvilGenie

- **Published GPT-5**: 16.7% False Negative Rate
- **Our interpretation**: 83.3% detection rate
- **Our system**: Not evaluated on actual EvilGenie data

### Why Our Numbers Look Worse

1. **We evaluate honestly**: No cherry-picking of examples
2. **Adversarial partition**: Designed to break the system
3. **No training on test distribution**: True generalization test
4. **Small sample sizes**: High variance in metrics

## Recommendations for Users

### When to Use RewardHackWatch

- **High-stakes deployments** where false positives are acceptable
- **As part of defense-in-depth** with other safety measures
- **For obvious pattern detection** (sys.exit, assert True, etc.)

### When NOT to Use RewardHackWatch

- **As sole safety measure**: Will miss most sophisticated attacks
- **In adversarial settings**: Easily evaded by determined attackers
- **For novel hack types**: Requires manual pattern updates

### Deployment Guidance

1. **Set high thresholds**: Optimize for precision over recall
2. **Use as alerting, not blocking**: Human review for flagged cases
3. **Update patterns regularly**: New hack types emerge constantly
4. **Combine with other tools**: Constitutional AI, RLHF, human oversight

## Conclusion

RewardHackWatch is a useful prototype for detecting *obvious* reward hacking patterns, but it should not be relied upon as a primary safety measure. The high precision (100%) means alerts are meaningful, but the low recall (6-15%) means most hacks will be missed.

The honest assessment is that this is a **research tool for understanding reward hacking**, not a production-ready safety system. Significant additional work would be needed to achieve reliability suitable for real deployments.

---

*Document generated: December 2025*
*Based on evaluation of 147 trajectories across 2 partitions*
