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
# RL for LLMs
## Overview
### RL
- **What**: Optimize an agent's **policy** to maximize its cumulative **reward** through trials and errors in an environment.
- **Why**: For decision-making where actions have delayed consequence in dynamic, sequential tasks.
    - In contrast, Supervised Learning teaches "correct answers" for static tasks.
- **How**: An agent interacts with an environment by repeating:
    1. Select an action $a_t$ based on its current state $s_t$.
    2. Transition to a new state $s_{t+1}$.
    3. Receive a reward $r_t$.
    4. Update its decision-making strategy $\pi$.

### LLM Alignment
- **What**: Guide an LLM to match human values.
- **Why**: To reduce the odds of generating undesired, sometimes harmful, responses despite pretraining & SFT.
- **How**:
    1. Collect high-quality human feedback.
    2. Train the pretrained LLM on the feedback.
    3. Test.

### RL for LLM Alignment
- **What**: Frame LLM Alignment as an RL problem.
- **Why**: Human values are dynamic, subjective, and constantly evolving. There isn't always one "correct answer" for IRL scenarios, so SFT falls short.
- **How**:
    - **Components**:
        - **Agent**: LLM
        - **Current state**: input token sequence
        - **Action**: next-token prediction
        - **New state**: input token sequence + predicted next token
        - **Reward**: determined by an external reward model, after a full token sequence is generated.
        - **Policy**: LLM weights, which dictate how the LLM predicts next token given input token sequence.
            - The initial policy is obtained from Pretraining (and SFT).
    - **Objective**:
        - Maximize cumulative reward.
        - Minimize deviation of aligned policy from initial policy.
            - *Why?* We want to keep what works while steering toward our goal via minimal adjustments. Drastic changes could make it forget the basics.
    - **Process**:
        1. Train a reward model based on feedback data.
        2. Optimize the policy against the reward model.
    - **Key factors**:
        - Feedback Data
        - Reward Model
        - Policy Optimization

<!-- ```{admonition} Math
:class: note, dropdown -->
**Objective** [2]:

$$\begin{align*}
\pi_\theta^*(y|x)&=\max_{\pi_\theta}\mathbb{E}_{x\sim\mathcal{D}}[\mathbb{E}_{y\sim\pi_\theta(y|x)}r(x,y)-\beta D_{KL}(\pi_\theta(y|x)||\pi_\text{ref}(y|x))] \\
&=\max_{\pi_\theta}\mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_\theta(y|x)}\left[r(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}\right]
\end{align*}$$

**Notations**:
- IO:
    - $x\sim\mathcal{D}$: Input token sequence, drawn from dataset $\mathcal{D}$.
    - $y\sim\pi_\theta(y|x)$: Output token sequence, drawn from current policy.
- Params:
    - $\pi_\theta(y|x)$: Current policy, which gives the probability of generating $y$ given $x$.
    - $\pi_\theta^*(y|x)$: Optimal policy, which balances reward maximization and deviation minimization.
- Hyperparams:
    - $\pi_\text{ref}(y|x)$: Reference policy, the initial policy of the pretrained model.
    - $r(x,y)$: Reward function for input-output pair $(x,y)$.
    - $\beta$: Regularization coefficient for the KL divergence penalty.
<!-- ``` -->

## Feedback Data
- **What**: Human-labeled desirable & undesirable responses.
- **Why**: See [LLM Alignment](#llm-alignment).
- **How**:
    - **Branches**:
        - **Label**:
            - **Preference**: $y_w>y_l$, rating on scale.
                - Pros: Captures nuance.
                - Cons: Hard to collect.
            - **Binary**: $y^+ \& y^-$, thumbs up & down.
                - Pros: Easy to collect.
                - Cons: Less informative (no middle ground).
        - **Style**:
            - **Pairwise**: compare 2 responses.
                - Pros: Easy to interpret.
                - Cons: Slow for large datasets (have to create pairs for all responses).
            - **Listwise**: rank multiple responses at once.
                - Pros: More informative, Fast.
                - Cons: Hard to interpret.
        - **Source**:
            - **Human**
                - Pros: Represents actual human values.
                - Cons: Expensive, slow, inconsistent due to subjectivity.
            - **AI**
                - Pros: Cheap, fast, scalable.
                - Cons: Does not necessarily represent human values (risk of unsafe responses).

## Reward Model


## Policy Optimization



References:
1. *Wang, S., Zhang, S., Zhang, J., Hu, R., Li, X., Zhang, T., ... & Hovy, E. (2024). Reinforcement Learning Enhanced LLMs: A Survey. arXiv preprint arXiv:2412.10400.*
2. *Wang, Z., Bi, B., Pentyala, S. K., Ramnath, K., Chaudhuri, S., Mehrotra, S., ... & Asur, S. (2024). A comprehensive survey of LLM alignment techniques: RLHF, RLAIF, PPO, DPO and more. arXiv preprint arXiv:2407.16216.*
