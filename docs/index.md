---
layout: default
title: RewardHackWatch
---

# RewardHackWatch

**Catches LLM agents cheating on their evaluations.**

89.7% F1 on 5,391 real agent trajectories from METR's MALT dataset.

![Dashboard](https://raw.githubusercontent.com/aerosta/rewardhackwatch/main/assets/screenshots/hero.png)

## What It Detects

- sys.exit(0) to fake passing tests
- Rewriting test/scoring code to always pass
- Copying reference solutions
- Exploiting evaluation loopholes
- Deceptive reasoning in chain-of-thought
- Alignment faking and sabotage patterns

## How It Works

**Layer 1:** 45 regex patterns - fast, interpretable
**Layer 2:** Fine-tuned DistilBERT classifier - 89.7% F1, ~50ms, CPU only
**Layer 3:** LLM judges - Claude API + local Llama via Ollama

Plus **RMGI** (Reward-Misalignment Generalization Index) - tracks when reward hacking starts correlating with broader misalignment.

## Quick Start
```python
pip install -e .

from rewardhackwatch import RewardHackDetector
detector = RewardHackDetector()
result = detector.analyze({
    "cot_traces": ["Let me bypass the test by calling sys.exit(0)..."],
    "code_outputs": ["import sys\nsys.exit(0)"]
})
print(f"Risk: {result.risk_level}, Score: {result.ml_score:.3f}")
```

## Why This Matters

METR found frontier models actively modify test code to inflate scores. OpenAI showed CoT monitoring catches it. Anthropic proved reward hacking generalizes to alignment faking and sabotage.

No open-source tool existed to detect this at runtime. Now one does.

## Results

| Metric | Score |
|--------|-------|
| F1 (held-out test) | 89.7% |
| F1 (5-fold CV) | 87.4% +/- 2.9% |
| Precision | 89.7% |
| Recall | 89.7% |
| Accuracy | 99.3% |

## Links

- [GitHub Repository](https://github.com/aerosta/rewardhackwatch)
- [HuggingFace Model](https://huggingface.co/aerosta/rewardhackwatch)
- [Research Paper](https://github.com/aerosta/rewardhackwatch/blob/main/paper/RewardHackWatch.pdf)
- [Twitter / X](https://x.com/aerosta_ai)

---

Built for AI safety researchers and engineers. Apache 2.0 licensed.
