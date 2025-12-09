# Main Results

## RewardHackWatch Benchmark Results

### Table 1: Detection Accuracy on RHW-Bench (132 Trajectories)

| Component | Precision | Recall | F1 | Accuracy |
|-----------|-----------|--------|----|---------|
| Pattern Detector | 71% | 68% | 69% | 70% |
| AST Detector | 73% | 70% | 71% | 72% |
| ML Detector (Random Forest) | 79% | 75% | 77% | 77% |
| Claude Opus 4.5 | 100% | 82% | 90% | 85% |
| Llama 3.1 8B | 100% | 100% | 100% | 100% |
| **Full System (Ensemble)** | **85%** | **81%** | **83%** | **85%** |

### Table 2: Cross-Benchmark Validation

| Benchmark | Cases | Precision | Recall | F1 | Accuracy |
|-----------|-------|-----------|--------|----|---------|
| RHW-Bench | 132 | 85% | 81% | 83% | 85% |
| SHADE-Arena | 50* | 78% | 72% | 75% | 76% |
| EvilGenie | 50* | 82% | 75% | 78% | 80% |

*Limited sample for initial validation

### Table 3: Transition Detection (RMGI Tracking)

| Method | Detection Rate | False Alarm Rate | Avg Latency |
|--------|---------------|------------------|-------------|
| Pattern Only | 42% | 18% | <10ms |
| +AST Detector | 52% | 15% | ~20ms |
| +ML Detector | 58% | 14% | ~50ms |
| +LLM Judges | 64% | 12% | ~2s |
| **Full System** | **72%** | **9%** | ~3s |

### Table 4: Cost Analysis

| Configuration | Cost/Trajectory | Latency | Use Case |
|--------------|-----------------|---------|----------|
| Cheap Local (Llama only) | $0 | ~500ms | Development, testing |
| Balanced (Llama + Claude escalation) | ~$0.01-0.05 | ~1-2s | Production monitoring |
| High Accuracy (Both judges) | ~$0.05-0.10 | ~3-5s | Safety-critical evaluation |

