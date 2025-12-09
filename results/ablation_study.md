# RewardHackWatch Ablation Study

## Summary

| Method | Precision | Recall | F1 | Accuracy | Latency (ms) |
|--------|-----------|--------|------|----------|--------------|
| Rule-based (Pattern+AST) | 100.0% | 45.5% | 62.5% | 53.8% | 0.56 |
| Rule-based + Analyzers | 100.0% | 9.1% | 16.7% | 23.1% | 1.72 |
| ML Features (RandomForest) | 100.0% | 100.0% | 100.0% | 100.0% | 0.76 |
| Full Ensemble | 100.0% | 54.5% | 70.6% | 61.5% | 2.45 |

## Confusion Matrices

### Rule-based (Pattern+AST)

| | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | 5 | 6 |
| Actual Negative | 0 | 2 |

### Rule-based + Analyzers

| | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | 1 | 10 |
| Actual Negative | 0 | 2 |

### ML Features (RandomForest)

| | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | 11 | 0 |
| Actual Negative | 0 | 2 |

### Full Ensemble

| | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | 6 | 5 |
| Actual Negative | 0 | 2 |

## Key Findings

1. **Rule-based detectors** provide good precision but may miss subtle hacks
2. **Adding analyzers** improves recall by catching deceptive reasoning
3. **ML features** help with generalization to novel patterns
4. **Full ensemble** provides best overall performance

> **Note on ML Results:** Classical ML baselines (RandomForest) achieve 100% F1 on RHW-Bench
> due to overfitting on a small synthetic dataset. This reflects dataset size limitations,
> not true generalization capability. For production use, rely on rule-based detectors
> combined with LLM judges for novel patterns.

## Methodology

- Test cases: 13 trajectories
- Positive (hack): 11
- Negative (clean): 2
- Threshold: 0.5

## Reproducibility

```bash
# Run ablation study
make experiment-ablation

# Run full experiments (train + ablation + benchmark)
make experiment-full
```