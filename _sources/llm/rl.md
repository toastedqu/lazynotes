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
- **What**: $U \stackrel[\beta]{\alpha}\rightleftarrows V$
- [Agent $\xleftrightarrow{\text{interact}}$ Environment] $\xrightarrow{\text{optimize}}$ Policy $\xrightarrow{\text{maximize}}$ Cumulative Reward
- **Why**: For decision-making where actions have delayed consequence in dynamic, sequential tasks.
    - In contrast, Supervised Learning teaches "correct answers" for static tasks.
- **How**: An agent interacts with an environment by repeating:
    - $s_t$ $\xrightarrow{a_t}$ $s_{t+1}$ $\xrightarrow{\text{get}}$ $r_t$ $\xrightarrow{\text{update}}$ $\pi$

### LLM Alignment
- **What**: LLM $\xleftarrow{\text{match}}$ human values
- **Why**: To reduce the odds of generating undesired, sometimes harmful, responses despite pretraining & SFT.
- **How**: Humans $\xrightarrow{\text{collect}}$ Feedback $\xrightarrow{\text{train}}$ Pretrained LLM

### RL for LLM Alignment
- **What**: Frame LLM Alignment as an RL problem:
    - **Agent**: LLM.
    - **State**: Input token sequence.
    - **Action**: Next-token prediction.
    - **Next State**: Input token sequence + Predicted next token.
    - **Reward**: Reward.
        - Determined by an external reward model OR preference labels.
        - Typically computed after a full token sequence is generated.
    - **Policy**: LLM weights.
        - Dictates how the LLM predicts next token given input token sequence.
        - The initial policy is obtained from Pretraining (and SFT).
- **Why**: Human values are dynamic, subjective, and constantly evolving. There isn't always one "correct answer" for IRL scenarios, so SFT falls short.
- **How**:
    - **Objective**:
        - Maximize cumulative reward.
        - Minimize deviation of aligned policy from initial policy.
            - *Why?* We want to keep what works while steering toward our goal via minimal adjustments. Drastic changes could make it forget the basics.
    - **Key factors**:
        - Feedback Data
        - Reward Model
        - Policy Optimization
    - **Process**: Feedback $\xrightarrow{\text{train}}$ RM $\xrightarrow{\text{train}}$ Policy
    - **Feedback Data**:
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
    - **RM**:
        - **Form**:
            - **Explicit**: An external model, typically from SFT of a pretrained LLM.
                - Pros: Interpretable & Scalable.
                - Cons: High computational cost.
            - **Implicit**: No external model (e.g., DPO).
                - Pros: Low computational cost. No reward overfitting.
                - Cons: Less control.
        - **Style**:
            - **Pointwise**: Outputs a reward score $r(x,y)$ given an input-output pair.
                - Pros: Simple & Interpretable.
                - Cons: Ignores relative preferences.
            - **Preferencewise**: Outputs a probability of the desired response being preferred over the undesired response: $P(y_w>y_l|x)=\sigma(r(x,y_w)-r(x,y_l))$.
                - Pros: Provides comparisons.
                - Cons: No pairwise preferences, Sensitive to human label inconsistencies.
        - **Level**:
            - **Token-level**: Reward is given per token/action.
                - Pros: Fine-grained feedback.
                - Cons: High computational cost, Noisy rewards.
            - **Response-level**: Reward is given per response (most commonly used).
                - Pros: Simple.
                - Cons: Coarse feedback.
        - **Source**:
            - **Positive**: Humans label both desired and undesired responses.
                - Pros: More control.
                - Cons: Expensive & Time-consuming.
            - **Negative**: Humans label undesired responses. LLMs generate desired responses.
                - Pros: Cheap & Scalable.
                - Cons: Less control.

<!-- ```{admonition} Math
:class: note, dropdown -->
Sample Objective [2]: Reward Max + Deviation Min.

$$\begin{align*}
\pi_\theta^*(y|x)&=\max_{\pi_\theta}\mathbb{E}_{x\sim\mathcal{D}}[\mathbb{E}_{y\sim\pi_\theta(y|x)}r(x,y)-\beta D_\text{KL}(\pi_\theta(y|x)||\pi_\text{ref}(y|x))] \\
&=\max_{\pi_\theta}\mathbb{E}_{x\sim\mathcal{D}}\left[\mathbb{E}_{y\sim\pi_\theta(y|x)}r(x,y)-\beta\mathbb{E}_{y\sim\pi_\theta(y|x)}\left[\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}\right]\right] \\
&=\max_{\pi_\theta}\mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_\theta(y|x)}\left[r(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}\right]
\end{align*}$$

Notations:
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

## Variations
### InstructGPT (The OG RLHF)
- **What**: The backbone of all ChatGPT variations in OpenAI.
- **Why**: OpenAI researchers realized traditional NLG evaluation metrics do NOT align with human preferences, so they came up with a way to directly teach LLMs human preferences.
- **How**:
    - **Training**:
        - Process:
            1. Data Collection: Pairwise, human feedback.
            2. RM: Explicit, pointwise RM.
            3. PPO
        - **Objective**:
            - Reward maximization.
            - Deviation minimization.
            - Alignment tax minimization: Minimize degradation of downstream task performance due to alignment.
        - **Datasets**:
            - SFT data: OG labeler samples.
            - RM data: Labeler rankings of model outputs.
            - PPO data: Prompts for RLHF.
    - **Evaluation**: Human metrics: Helpful, Honest, Harms.

<!-- ```{admonition} Math
:class: note, dropdown -->
RM:

$$
L_\text{RM}(r_\phi)=-\frac{1}{C_K^2}\mathbb{E}_{(x,y_w,y_l)~\mathcal{D}}\left[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))\right]
$$

PPO:

$$
\pi_\theta^*(y|x)=\max_{\pi_\theta}\mathbb{E}_{x\sim\mathcal{D}}\left[\mathbb{E}_{y\sim\pi_\theta(y|x)}r_\phi(x,y)-\beta D_\text{KL}\left(\pi_\theta(y|x)||\pi_\text{ref}(y|x)\right)\right]+\gamma\mathbb{E}_{x\sim\mathcal{D}_\text{pretrain}}[\log\pi_\theta(x)]
$$


Notations:
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



References:
1. Wang, S., Zhang, S., Zhang, J., Hu, R., Li, X., Zhang, T., ... & Hovy, E. (2024). Reinforcement Learning Enhanced LLMs: A Survey. arXiv preprint arXiv:2412.10400.
2. Wang, Z., Bi, B., Pentyala, S. K., Ramnath, K., Chaudhuri, S., Mehrotra, S., ... & Asur, S. (2024). A comprehensive survey of LLM alignment techniques: RLHF, RLAIF, PPO, DPO and more. arXiv preprint arXiv:2407.16216.
3. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022). Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35, 27730-27744.
