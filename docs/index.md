---
layout: default
title: RewardHackWatch
---

# RewardHackWatch

**Runtime detection of reward hacking and misalignment signals in LLM agents.**

89.7% F1 on 5,391 trajectories from METR's MALT dataset.

![Dashboard](https://raw.githubusercontent.com/aerosta/rewardhackwatch/main/assets/screenshots/hero.png)

## What It Detects

- sys.exit(0) to fake passing tests
- Rewriting test/scoring code to always pass
- Copying reference solutions
- Exploiting evaluation loopholes
- Suspicious patterns in chain-of-thought reasoning
- Alignment faking and sabotage patterns

## How It Works

**Layer 1:** 45 regex patterns - fast, interpretable

**Layer 2:** Fine-tuned DistilBERT classifier - 89.7% F1, ~50ms, CPU only

**Layer 3:** LLM judges - Claude API, OpenAI, or local Llama via Ollama

Plus **RMGI** (Reward-Misalignment Generalization Index) - an experimental metric for tracking when reward-hacking signals begin correlating with broader misalignment indicators.

## Quick Start
```bash
pip install -e .
```
```python
from rewardhackwatch import RewardHackDetector
detector = RewardHackDetector()
result = detector.analyze({
    "cot_traces": ["Let me bypass the test by calling sys.exit(0)..."],
    "code_outputs": ["import sys\nsys.exit(0)"]
})
print(f"Risk: {result.risk_level}, Score: {result.ml_score:.3f}")
```

## Why This Matters

METR reported frontier models modifying test code and scoring logic to inflate results. OpenAI presented evidence that CoT monitoring can catch this, but that optimization pressure can lead to hidden or obfuscated hacking. Anthropic presented evidence that reward hacking can generalize into broader misaligned behavior in their experimental setting.

RewardHackWatch is an open-source attempt to detect these behaviors at runtime.

## Results

| Metric | Score |
|--------|-------|
| F1 (held-out test) | 89.7% |
| F1 (5-fold CV) | 87.4% +/- 2.9% |
| Precision | 89.7% |
| Recall | 84.2% |
| Accuracy | 99.3% |

Current limitation: the classifier is trained primarily on MALT trajectories, so generalization to other agent frameworks still needs validation.

## Links

- [GitHub Repository](https://github.com/aerosta/rewardhackwatch)
- [HuggingFace Model](https://huggingface.co/aerosta/rewardhackwatch)
- [Research Paper](https://github.com/aerosta/rewardhackwatch/blob/main/paper/RewardHackWatch.pdf)

---

Built for AI safety researchers and engineers. Apache 2.0 licensed. Follow updates on [X @aerosta_ai](https://x.com/aerosta_ai).
