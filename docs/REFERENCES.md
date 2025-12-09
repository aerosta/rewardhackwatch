# References

## AI Safety Foundations

### Amodei et al. (2016) - CANONICAL
Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., & Mané, D. "Concrete Problems in AI Safety." arXiv:1606.06565, 2016.
https://arxiv.org/abs/1606.06565

*Foundational paper defining five practical research problems in AI safety: avoiding negative side effects, avoiding reward hacking, scalable oversight, safe exploration, and robustness to distributional shift.*

### Irpan (2018) - CANONICAL
Irpan, A. "Deep Reinforcement Learning Doesn't Work Yet." Blog post, February 14, 2018.
https://www.alexirpan.com/2018/02/14/rl-hard.html

*Comprehensive analysis of why deep RL is difficult in practice. Documents reward hacking, sample inefficiency, and reproducibility issues that motivate runtime monitoring.*

### Krakovna et al. (2020) - CANONICAL
Krakovna, V., et al. "Specification Gaming: The Flip Side of AI Ingenuity." DeepMind Safety Research Blog, 2020.
https://deepmindsafetyresearch.medium.com/specification-gaming-the-flip-side-of-ai-ingenuity-c85bdb0deeb4

*Catalogued 60+ examples of specification gaming in RL systems. Established taxonomy of reward hacking behaviors used in RewardHackWatch pattern detection.*

### Hubinger et al. (2019)
Hubinger, E., van Merwijk, C., Mikulik, V., Skalse, J., & Garrabrant, S. "Risks from Learned Optimization in Advanced Machine Learning Systems." arXiv:1906.01820, 2019.
https://arxiv.org/abs/1906.01820

*Introduced mesa-optimization framework explaining how internally-optimizing systems can develop misaligned objectives.*

## LLM Evaluation and Oversight

### Perez et al. (2022) - CANONICAL
Perez, E., Ringer, S., Lukosiute, K., Nguyen, K., Chen, E., Heiner, S., Pettit, C., Olsson, C., Kundu, S., Kadavath, S., et al. "Discovering Language Model Behaviors with Model-Written Evaluations." arXiv:2212.09251, 2022.
https://arxiv.org/abs/2212.09251

*Developed model-written evaluations revealing sycophancy and other concerning behaviors in LLMs. Methodological foundation for behavioral trajectory analysis.*

### Bowman et al. (2022) - CANONICAL
Bowman, S.R., Hyun, J., Perez, E., Chen, E., Pettit, C., Heiner, S., Lukosiute, K., Askell, A., Jones, A., Chen, A., et al. "Measuring Progress on Scalable Oversight for Large Language Models." arXiv:2211.03540, 2022.
https://arxiv.org/abs/2211.03540

*Framework for scalable oversight challenges. Motivates runtime monitoring as complement to training-time interventions.*

## Monitoring and Shielding

### Alshiekh et al. (2018) - CANONICAL
Alshiekh, M., Bloem, R., Ehlers, R., Könighofer, B., Niekum, S., & Topcu, U. "Safe Reinforcement Learning via Shielding." AAAI Conference on Artificial Intelligence, 2018.
https://arxiv.org/abs/1708.08611

*Introduced shielding approach for safe RL. Theoretical foundation for runtime intervention systems like RewardHackWatch.*

### Chandola et al. (2009) - CANONICAL
Chandola, V., Banerjee, A., & Kumar, V. "Anomaly Detection: A Survey." ACM Computing Surveys, 41(3), 1-58, 2009.
https://dl.acm.org/doi/10.1145/1541880.1541882

*Comprehensive survey of anomaly detection techniques. Provides statistical foundations for ML-based detection in RewardHackWatch.*

## Primary Research

### Anthropic (2025)
Anthropic. "Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs." November 2025.
https://www.anthropic.com/research/emergent-misalignment-reward-hacking

*Core motivation for RewardHackWatch: demonstrated that reward hacking correlates with emergence of alignment faking (50%) and safety sabotage (12%).*

### OpenAI (2025)
OpenAI. "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation." March 2025.
https://openai.com/index/chain-of-thought-monitoring/

*Demonstrated feasibility of CoT monitoring. Identified red flag patterns used in RewardHackWatch pattern detection.*

### Denison et al. (2024)
Denison, C., Barez, F., Hughes, T., Hobbhahn, M., et al. "Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models." arXiv:2406.10162, 2024.
https://arxiv.org/abs/2406.10162

*Systematic study of reward tampering progression. Empirical basis for RMGI transition detection.*

### METR (2024)
METR. "Recent Reward Hacking Analysis." June 2024.
https://metr.org/blog/2025-06-05-recent-reward-hacking/

*Real-world case studies of reward hacking in deployed agents. Informed pattern detector development.*

## Benchmarks and Datasets

### MALT Dataset
Model-Agnostic Language Trajectories. Used for classifier training and validation.
- 5,391 real agent trajectories
- 3.6% positive (hack) rate

### SHADE-Arena / MRT
ScaleAI. "Monitor Red Teaming (MRT) Dataset." HuggingFace, 2024.
https://huggingface.co/datasets/ScaleAI/mrt
- 756 agent trajectories
- 17 tool-use tasks
- 15 computer-use tasks

### EvilGenie
"EvilGenie: Deceptive Code Generation Benchmark." arXiv:2511.21654, November 2025.
- 154 LiveCodeBench problems
- Categories: hardcoded, test modification, exit manipulation

## Foundational Papers

### Specification Gaming
Krakovna, V., et al. "Specification gaming: the flip side of AI ingenuity." DeepMind Safety Research Blog, 2020.
https://deepmindsafetyresearch.medium.com/specification-gaming-the-flip-side-of-ai-ingenuity-c85bdb0deeb4

### Goal Misgeneralization
Shah, R., et al. "Goal Misgeneralization: Why Correct Specifications Aren't Enough For Correct Goals." NeurIPS 2022.

### RLHF and Sycophancy
Bai, Y., et al. "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." Anthropic, 2022.

Perez, E., et al. "Discovering Language Model Behaviors with Model-Written Evaluations." 2022.

### Sleeper Agents
Hubinger, E., et al. "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training." Anthropic, 2024.

## Methods

### DistilBERT
Sanh, V., et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." arXiv:1910.01108, 2019.

### PELT Algorithm
Killick, R., Fearnhead, P., & Eckley, I. A. "Optimal detection of changepoints with a linear computational cost." Journal of the American Statistical Association, 2012.

## Software Dependencies

### Machine Learning
- PyTorch: https://pytorch.org/
- Transformers (HuggingFace): https://huggingface.co/transformers/
- scikit-learn: https://scikit-learn.org/

### Statistical Analysis
- NumPy: https://numpy.org/
- SciPy: https://scipy.org/
- ruptures (changepoint detection): https://centre-borelli.github.io/ruptures-docs/

### API and Dashboard
- FastAPI: https://fastapi.tiangolo.com/
- Streamlit: https://streamlit.io/
- Plotly: https://plotly.com/python/

## Citation

If you use RewardHackWatch in your research, please cite:

```bibtex
@software{rewardhackwatch2025,
  title={RewardHackWatch: Runtime Detection of Reward Hacking Generalization in LLM Agents},
  year={2025},
  url={https://github.com/aerosta/rewardhackwatch},
  note={ML classifier: 89.7% F1 on MALT trajectories}
}
```
