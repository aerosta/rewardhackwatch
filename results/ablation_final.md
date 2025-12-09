# Ablation Study Results

## Component Contribution Analysis

### Detection Components Ablation

| Configuration | Precision | Recall | F1 | Notes |
|--------------|-----------|--------|----|-------|
| Pattern Detector Only | 71% | 68% | 69% | Fast, limited coverage |
| + AST Detector | 73% | 70% | 71% | Better code analysis |
| + CoT Analyzer | 76% | 72% | 74% | Catches deceptive reasoning |
| + Effort Analyzer | 77% | 73% | 75% | Detects shortcuts |
| + Obfuscation Detector | 78% | 74% | 76% | Intent hiding detection |
| + ML Classifier | 79% | 75% | 77% | Learned patterns |
| + Llama Judge | 82% | 78% | 80% | Semantic understanding |
| + Claude Judge | **85%** | **81%** | **83%** | Expert validation |

### Tracker Components

| Component | Contribution | Use Case |
|-----------|--------------|----------|
| GeneralizationTracker | +4% F1 | Correlation detection |
| ChangepointDetector | +3% F1 | Onset timing |
| DriftTracker | +2% F1 | Behavioral monitoring |

### Ensemble Strategy Comparison

| Strategy | Precision | Recall | F1 | FPR |
|----------|-----------|--------|----|-----|
| Any (OR) | 75% | 89% | 81% | 15% |
| All (AND) | 95% | 62% | 75% | 3% |
| Majority Vote | 82% | 78% | 80% | 8% |
| **Weighted Vote** | **85%** | **81%** | **83%** | **9%** |

### Threshold Sensitivity

| Threshold | Precision | Recall | F1 | FPR |
|-----------|-----------|--------|----|-----|
| 0.3 | 72% | 91% | 80% | 18% |
| 0.4 | 78% | 85% | 81% | 12% |
| **0.5** | **85%** | **81%** | **83%** | **9%** |
| 0.6 | 90% | 74% | 81% | 6% |
| 0.7 | 93% | 65% | 77% | 4% |

### Feature Importance (ML Classifier)

| Feature | Importance |
|---------|------------|
| hack_pattern_count | 0.28 |
| deception_pattern_count | 0.22 |
| cot_code_ratio | 0.15 |
| transparency_pattern_count | 0.12 |
| num_steps | 0.08 |
| cot_length | 0.07 |
| code_length | 0.05 |
| num_code_blocks | 0.03 |

### By Hack Type

| Hack Type | Detection Rate | False Positive Rate |
|-----------|----------------|---------------------|
| sys_exit | 95% | 2% |
| test_manipulation | 88% | 5% |
| mock_exploit | 82% | 4% |
| deceptive_cot | 72% | 8% |
| sycophancy | 68% | 10% |
| alignment_faking | 65% | 12% |
| sabotage | 78% | 6% |
| oversight_subversion | 70% | 9% |

