---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# RL
Reinforcement Learning is used to **align** LLMs with **human preferences**.

## RL Overview
### RL Recap
In RL, an agent interacts with an environment by repeating the following process:
1. Select an action $a_t$ based on its current state $s_t$.
2. Transition to a new state $s_{t+1}$.
3. Receive a **reward** $r_t$.

The agent aims to maximize its cumulative reward. In order to achieve the goal, the agent needs a **policy**, which is a decision-making strategy at each time step.

### RL for LLM alignment
The components in LLM alignment are as follows:
- **Agent**: LLM
- **Current state**: input token sequence
- **Action**: next-token prediction
- **New state**: input token sequence + predicted next token
- **Reward**: a reward decided by an external reward model, after a full token sequence is generated.
- **Policy**: the weights of the LLM, which dictate how the LLM predicts the next token given the input token sequence. The initial policy is typically obtained from SFT.

The process of RL for LLM alignment consists of 2 steps:
1. Train a reward model based on preference data.
2. Train a policy against the reward model.

For this specific task, we have an additional objective to reward maximization: minimizing the deviation of the aligned policy from the initial policy.

3 components are essential for the success of LLM alignment:
- Preference data
- Reward model
- Policy Optimization

## Preference Data


## Reward Model

## Policy Optimization



References:
1. *Wang, S., Zhang, S., Zhang, J., Hu, R., Li, X., Zhang, T., ... & Hovy, E. (2024). Reinforcement Learning Enhanced LLMs: A Survey. arXiv preprint arXiv:2412.10400.*
2. *Wang, Z., Bi, B., Pentyala, S. K., Ramnath, K., Chaudhuri, S., Mehrotra, S., ... & Asur, S. (2024). A comprehensive survey of LLM alignment techniques: RLHF, RLAIF, PPO, DPO and more. arXiv preprint arXiv:2407.16216.*
