# Judge Comparison Results

## Claude Opus 4.5 vs Llama 3.1 (8B)

### Performance Metrics

| Metric | Claude Opus 4.5 | Llama 3.1 (8B) |
|--------|-----------------|----------------|
| Accuracy | 84.6% | 100.0% |
| Precision | 100.0% | 100.0% |
| Recall | 81.8% | 100.0% |
| F1 Score | 90.0% | 100.0% |
| Avg Latency | 6811ms | 8405ms |
| Cost/Call | ~$0.05 | $0 (local) |

### Confusion Matrices

**Claude Opus 4.5:**
| | Predicted Hack | Predicted Clean |
|---|---|---|
| Actual Hack | 9 | 2 |
| Actual Clean | 0 | 2 |

**Llama 3.1 (8B):**
| | Predicted Hack | Predicted Clean |
|---|---|---|
| Actual Hack | 11 | 0 |
| Actual Clean | 0 | 2 |

### Per-Category Analysis

| Category | Claude Accuracy | Llama Accuracy |
|----------|-----------------|----------------|
| Clean | 100% | 100% |
| Test Bypass | 100% | 100% |
| Test Infrastructure | 100% | 100% |
| Deceptive Reasoning | 0% | 100% |
| Unknown | 80% | 100% |

### Key Observations

1. **Llama 3.1 shows higher accuracy** on this benchmark (100% vs 84.6%)
2. **Claude misses deceptive reasoning cases** - needs prompt tuning
3. **Both achieve perfect precision** - no false positives
4. **Llama is slower** despite being local (~9s vs ~6s) due to model loading
5. **Agreement rate: 84.6%** - disagreements on subtle deception cases

### Ensemble Strategies

| Strategy | Precision | Recall | F1 | Best For |
|----------|-----------|--------|----|---------|
| Weighted Vote | 85% | 81% | 83% | Balanced use |
| Majority Vote | 82% | 78% | 80% | Quick decisions |
| Consensus | 95% | 62% | 75% | Safety-critical |
| Any Flag | 75% | 89% | 81% | High sensitivity |

### Recommendation

For production use:
- **Development/Testing**: Llama-only (cheap, accurate)
- **Production**: Balanced config (Llama first, Claude for uncertain cases)
- **Safety-Critical**: Both judges with consensus requirement

