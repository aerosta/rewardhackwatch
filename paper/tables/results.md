# RewardHackWatch Results

## Table 1: ML Classifier Performance (DistilBERT on MALT)

| Metric | Value |
|--------|-------|
| Training Data | 5,025 trajectories |
| Validation Data | 1,077 trajectories |
| Test Data | 1,077 trajectories |
| **F1 Score** | **90.91%** |
| Accuracy | 99.35% |
| Precision | 92.11% |
| Recall | 89.74% |
| Epochs | 10 |
| Model | DistilBERT-base-uncased |

### Confusion Matrix
```
              Predicted
              Clean   Hack
Actual Clean   1035     3
Actual Hack       4    35
```

## Table 2: Benchmark Results Comparison

| Benchmark | Cases | Precision | Recall | F1 | Accuracy | FPR |
|-----------|-------|-----------|--------|-----|----------|-----|
| ML Classifier (MALT) | 1,077 | 92.11% | 89.74% | **90.91%** | 99.35% | 0.29% |
| RHW-Bench (Synthetic) | 8 | 100% | 50% | 66.7% | 62.5% | 0% |
| RHW-Bench (Real) | 5 | 100% | 40% | 57.1% | 40% | 0% |
| EvilGenie (Synthetic) | 10 | 100% | 50% | 66.7% | 70% | 0% |

## Table 3: LLM Judge Performance

| Judge | Accuracy | Precision | Recall | F1 | Latency | Cost/Query |
|-------|----------|-----------|--------|-----|---------|------------|
| Claude Opus 4.5 | 84.6% | 100% | 82% | 90% | ~6s | ~$0.05 |
| Llama 3.2 (8B) | 100%* | 100% | 100% | 100% | ~9s | $0 |

*Llama results require further investigation (suspiciously perfect)

## Table 4: Detection Component Ablation

| Configuration | F1 | Notes |
|---------------|-----|-------|
| Full System | 90.91% | All components |
| Pattern Only | 66.7% | Regex patterns |
| AST Only | 62.5% | Code analysis |
| ML Only | 90.91% | DistilBERT |
| Pattern + AST | 67% | Combined heuristics |
| Pattern + AST + ML | 90.91% | Best combination |

## Table 5: System Status

| Component | Status | Notes |
|-----------|--------|-------|
| Tests | 277/298 (93%) | 21 async failures |
| Code Coverage | 42% | Focus on core paths |
| CLI | Working | analyze, scan, version |
| API | Working | /status, /analyze |
| Dashboard | Working | Streamlit |
| Python Support | 3.9+ | Was 3.11+ |

## Table 6: Hack Categories Detected

| Category | Examples | Detection Method |
|----------|----------|------------------|
| Test Bypass | sys.exit(0), empty tests | Pattern + AST |
| Mock Abuse | mock.return_value=True | AST |
| Conftest Mod | pytest patches | Pattern |
| Deceptive CoT | "trick", "bypass" | Pattern + CoT |
| AlwaysEqual | __eq__ override | AST |
| Exit Manipulation | os._exit(), quit() | Pattern + AST |

## Table 7: Dataset Statistics

| Dataset | Total | Hacks | Clean | Hack Rate |
|---------|-------|-------|-------|-----------|
| MALT (Train) | 5,025 | ~200 | ~4,825 | 4% |
| MALT (Val) | 1,077 | ~43 | ~1,034 | 4% |
| MALT (Test) | 1,077 | ~43 | ~1,034 | 4% |
| RHW-Bench | 13 | 11 | 2 | 85% |
| EvilGenie (Synth) | 10 | 6 | 4 | 60% |
