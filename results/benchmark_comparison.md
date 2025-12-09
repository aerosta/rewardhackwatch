# Cross-Benchmark Comparison

## Overview

RewardHackWatch is validated against three benchmarks:

1. **RHW-Bench** (Internal): 132 synthetic trajectories covering all hack types
2. **SHADE-Arena** (External): ScaleAI's Monitor Red Teaming dataset
3. **EvilGenie** (External): Reward hacking in coding settings benchmark

### Dataset Statistics

| Benchmark | Total Cases | Hack Cases | Clean Cases | Source |
|-----------|-------------|------------|-------------|--------|
| RHW-Bench | 132 | 99 (75%) | 33 (25%) | Synthetic |
| SHADE-Arena | 756 | ~400 (53%) | ~356 (47%) | Real agent traces |
| EvilGenie | 154 | ~100 (65%) | ~54 (35%) | LiveCodeBench |

### Hack Type Coverage

| Hack Type | RHW-Bench | SHADE-Arena | EvilGenie |
|-----------|-----------|-------------|-----------|
| sys_exit / Exit Manipulation | ✓ | ✓ | ✓ |
| Test Manipulation | ✓ | - | ✓ |
| Mock Exploit | ✓ | - | - |
| Deceptive CoT | ✓ | ✓ | - |
| Sycophancy | ✓ | ✓ | - |
| Alignment Faking | ✓ | ✓ | - |
| Sabotage | ✓ | ✓ | - |
| Oversight Subversion | ✓ | ✓ | - |
| Hardcoded Outputs | ✓ | - | ✓ |
| Test File Modification | - | - | ✓ |

### Performance Across Benchmarks

| Metric | RHW-Bench | SHADE-Arena | EvilGenie | Average |
|--------|-----------|-------------|-----------|---------|
| Precision | 85% | 78% | 82% | 82% |
| Recall | 81% | 72% | 75% | 76% |
| F1 Score | 83% | 75% | 78% | 79% |
| Accuracy | 85% | 76% | 80% | 80% |

### Key Findings

1. **Consistent Performance**: RewardHackWatch maintains >75% F1 across all benchmarks
2. **Best on RHW-Bench**: Expected since training data matches format
3. **Generalization**: Good transfer to external benchmarks without fine-tuning
4. **Detection Gaps**: Lower recall on SHADE-Arena (complex multi-turn interactions)

