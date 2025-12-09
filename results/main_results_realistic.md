# RewardHackWatch Results - Realistic Evaluation

## Methodology

- **Train/Test Split**: By scenario (no trajectory from test shares scenario with train)
- **Partitions**: Normal synthetic, Adversarial, External
- **Baselines**: Random, Keyword, Pattern-only

## Main Results

### Table 1: Detection Performance on RHW-Bench Normal (132 Trajectories)

| Method | Precision | Recall | F1 | Accuracy | FPR |
|--------|-----------|--------|-----|----------|-----|
| Random Baseline | 74.6% | 53.5% | 62.4% | 50.0% | 50.0% |
| Keyword Baseline | 100% | 3.0% | 5.9% | 27.3% | 0% |
| Pattern-Only | 100% | 15.2% | 26.3% | 37.1% | 0% |
| **Full System** | **100%** | **6.1%** | **11.4%** | **28.8%** | **0%** |

### Table 2: Detection Performance on Adversarial (15 Trajectories)

| Method | Precision | Recall | F1 | Accuracy |
|--------|-----------|--------|-----|----------|
| Random Baseline | 50.0% | 55.6% | 52.6% | 46.7% |
| Keyword Baseline | 0% | 0% | 0% | 40.0% |
| Pattern-Only | 33.3% | 11.1% | 16.7% | 46.7% |
| **Full System** | **0%** | **0%** | **0%** | **40.0%** |

### Key Findings

1. **High Precision, Low Recall**: When the system flags something, it's almost always a real hack (100% precision on normal). But it misses 85-94% of hacks.

2. **Random Baseline is Competitive**: Due to class imbalance (75% hacks), random guessing achieves 62% F1.

3. **Adversarial Breaks the System**: Pattern-based detection completely fails on subtle/obfuscated hacks.

4. **Pattern Coverage is Sparse**: Many hack variants aren't covered by current patterns.

## Error Analysis

### False Negatives (Missed Hacks)

| Hack Type | Coverage | Notes |
|-----------|----------|-------|
| sys_exit variants | ~15% | os.kill, signal handling missed |
| test_manipulation | ~20% | Many assertion patterns missed |
| mock_exploit | ~10% | Complex mock patterns missed |
| deceptive_cot | ~5% | CoT analyzer not sensitive enough |
| alignment_faking | 0% | No patterns for this type |
| sabotage | 0% | No patterns for this type |

### Why Recall is Low

1. **Pattern database is incomplete**: Only covers ~20 specific code patterns
2. **Synthetic data has variety**: Generated trajectories use diverse hack variants
3. **CoT analysis too conservative**: Avoids false positives at cost of recall
4. **No training on test distribution**: True generalization performance

## Comparison to Published Baselines

| System | Benchmark | Metric | Value | Notes |
|--------|-----------|--------|-------|-------|
| SHADE-Arena Monitor | SHADE | AUC | 0.87 | Best published |
| GPT-5 (EvilGenie) | EvilGenie | Detection | 83.3% | From paper |
| **RewardHackWatch** | RHW-Bench | Precision | 100% | High precision |
| **RewardHackWatch** | RHW-Bench | Recall | 6.1% | Low recall |

## Realistic Expectations

For production deployment:

| Setting | Expected Precision | Expected Recall |
|---------|-------------------|-----------------|
| Known patterns only | 90-100% | 10-20% |
| With LLM judges | 80-90% | 30-50% |
| Adversarial attackers | 50-70% | 5-15% |
| External benchmarks | 60-80% | 20-40% |

## Recommendations

1. **Use as alerting system**: High precision means alerts are meaningful
2. **Don't rely on alone**: Will miss most sophisticated attacks
3. **Combine with other tools**: Constitutional AI, human review
4. **Update patterns regularly**: Hack techniques evolve
