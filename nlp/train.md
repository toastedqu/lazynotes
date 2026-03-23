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
# Training
Post-training adapts (or aligns) a pretrained general-purpose LM to a specific data distribution (or task).

Post-training is necessary because foundation models are pretrained on vast, general-domain corpora, meaning they may mismatch downstream tasks.

&nbsp;

## SFT
- **What**: Supervised Fine-Tuning.
	- Fine-tune a pretrained LM on <**input → desired output**> pairs with supervised learning objectives.
- **Why**: SFT directly steers the model to the exact behavior spec.
- **How**: Minimize NLL of reference answer tokens.

```{attention} Q&A
:class: dropdown
*Pros?*
- Simple, stable training objective.
- Effective ← Even small curated datasets can strongly steer tone/format.
- Typically first step before preference optimization.

*Cons?*
- **Imitation ceiling**: It can only learn what's in the dataset.
- Overfitting if data is narrow.
- ⬇️⬇️Generalization.

*When is SFT NOT enough?*
- **Preference trade-offs** ("helpful vs safe", "concise vs thorough") are NOT captured by "gold" responses.
- **Policy compliance** in adversarial settings → often paired with preference optimization + red-teaming.
```

&nbsp;

### PEFT
- **What**: Parameter-Efficient Fine-Tuning.
	- Fine-tune ONLY a small subset of params.
- **Why**: **Adaptation** + **Efficiency**.
	- *Why use PEFT?*
		- Full Fine-tuning:
			- HIGH computational cost (time & resource).
			- HIGH storage cost (one full copy per task).
			- Catastrophic Forgetting.
		- PEFT solves all 3 problems above.
	- *Why PEFT works?*
		- Assumption: The adaptation of a pretrained LM to a downstream task exists in a **low-intrinsic dimension**.
		- The assumption is verified by empirical success.
- **How**:
	1. Freeze pretrained params.
	2. Inject trainable adapters/modules with a small number of params.
	3. Train the injected params.

&nbsp;

#### LoRA
- **What**: Low-Rank Adaptation.
	- Train **low-rank** matrices for adaptation.
- **Why**: #params ⬇️ → Finetuning efficiency ⬆️.
- **How**: Inject low-rank matrices into the weights of specific layers, w/o modifying the original weights directly.

```{note} Math
:class: dropdown
Notations:
- IO:
	- $\mathbf{x}\in\mathbb{R}^{H_{in}}$: Input vector.
	- $\mathbf{y}\in\mathbb{R}^{H_{out}}$: Output vector.
- Params:
	- $B\in\mathbb{R}^{H_{out}\times r}$: Low-rank decomposed matrix (left)
	- $A\in\mathbb{R}^{r\times H_{in}}$: Low-rank decomposed matrix (right)
	- $\Delta W=BA$: Additional, trainable weight matrix.
- Hyperparams:
	- $W_{0}\in\mathbb{R}^{H_{out}\times H_{in}}$: Original, frozen weight matrix.
	- $r\ll\min(H_{in}, H_{out})$: Rank of the original weight matrix.
	- $\alpha\in[0,r]$: Scaling factor of the additional weights.

Forward:

$$
\mathbf{y}=W_{0}\mathbf{x}+\frac{\alpha}{r}\Delta W\mathbf{x}=W_{0}\mathbf{x}+\frac{\alpha}{r}BA\mathbf{x}
$$

Backward:

$$\begin{align}
g_{B}=\frac{\alpha}{r}g_{\Delta W}A^T \\
g_{A}=\frac{\alpha}{r}B^Tg_{\Delta W}
\end{align}$$
```

```{attention} Q&A
:class: dropdown
*Pros*:
- No overfitting ← Task-specific adaptation w/o modifying the original params

*Cons*:
- ONLY applicable to linear transformation.
- ONLY great performance with pretrained models → Lower performance relative to full finetuning
- High sensitivity to hyperparameters.

*Where do I apply LoRA?*
- OG paper: Attention weights ONLY → Simplicity & Param Efficiency
	- ← Empirical performance
```

<!-- #### QLoRA
- **Name**: LoRA for Quantized LLMs.
- **What**: Insert learnable LoRAs into each layer of a quantized pretrained LM.
- **Why**: Standard fine-tuning scales linearly with model size in memory → HUGE memory cost
	- QLoRA dramatically cuts down memory WHILE preserving full-precision performance.
- **How**:
	1. **4-bit NormalFloat Quantization**:  -->

<!-- ### Adapters

### Prompt Tuning

#### Prefix Tuning

#### Instruction Tuning

## Alignment
### RLHF/PPO

### RLAIF

### DPO

### GRPO -->

&nbsp;

## RL
```{dropdown} Table: Notations
| Concept | Notation |
|------|-----|
| State | $s\in\mathcal{S}$ |
| Action | $a\in\mathcal{A}$ |
| Policy | $\pi(a_t\|s_t):\mathcal{S}\times\mathcal{A}\rightarrow[0,1]$ |
| Reward | $r(s,a):\mathcal{S}\times\mathcal{A}\rightarrow\mathbb{R}$ |
| Transition Probability | $p(s_{t+1}\|s_t,a_t)\in[0,1]$ |
| Discount Factor | $\gamma\in[0,1)$ |
| #Steps | $T\in[0,\infty)$ ($T=\infty$ if endless) |
| Trajectory | $\tau=(s_0,a_0,s_1,a_1,\dots)$ |
| Discounted Return of a trajectory | $R(\tau)=G_0=\sum_{t=0}^T\gamma^tr_t$ |
| Discounted Return at a time step | $G_t=R(\tau_{t:})=\sum_{k=0}^{T-t}\gamma^kr_{t+k}$ |
| State Value Function | $V^\pi(s)=E_\pi[G_t\|s_t=s]$ |
| Action Value Function | $Q^\pi(s,a)=E_\pi[G_t\|s_t=s,a_t=a]$ |
| Advantage | $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ |
| Expected Discounted Return | $J(\pi)=E_{\tau\sim\pi}\left[R(\tau)\right]=E_{s_0\in\mathcal{S}}[V^\pi(s_0)]$ |
```

```{dropdown} Table: Intuition
| Concept | Intuition |
|---------|-----------|
| Policy | How likely the agent chooses a specific action given curr state. |
| Reward | The immediate feedback from the env given curr action & state. |
| Transition Probability | How likely the env moves to a specific state given curr action & state. |
| Discount Factor | How much the agent cares about future vs immediate rewards.<br>(1: yes; 0: no) |
| Discounted Return of a trajectory | How good a full trajectory is. |
| Discounted Return at a time step | How good the future trajectory is from a specific moment. |
| State Value Function | How good it is to be in a specific curr state, following curr policy. |
| Action Value Function | How good it is to take a specific action in a specific curr state, if following curr policy. |
| Advantage | How much a specific action is better/worse than expected. |
| Expected Discounted Return | The overall performance of a policy across all of its possible trajectories.<br>(i.e., **RL's main objective**) |
```

&nbsp;

### REINFORCE
- **What**:
	- State $x$: Input token seq.
	- Action $y$: Output token seq.
	- Policy $\pi_\theta(y|x)$: Model, parametrized by $\theta$.
	- Reward $r(x,y)$: Scalar reward.
- **How**:
	- Objective: **Expected Discounted Return**
$$
J(\theta)=E_{y\sim\pi_\theta(\cdot|x)}[r(x,y)]=\sum_y\pi_\theta(y|x)r(x,y)
$$
	- Optimization: **Policy Gradient**
$$
\nabla_\theta J(\theta)=E_{y\sim\pi_\theta(\cdot|x)}[\nabla_\theta\log\pi_\theta(y|x)r(x,y)]
$$
- **How** (Reduced Variance w/ Advantage Estimate):
	- Objective:
$$
J(\theta)=E_{y\sim\pi_\theta(\cdot|x)}[\hat{A}(x,y)]=\sum_y\pi_\theta(y|x)\hat{A}(x,y)
$$
	- Optimization:
$$
\nabla_\theta J(\theta)=E_{y\sim\pi_\theta(\cdot|x)}[\nabla_\theta\log\pi_\theta(y|x)\hat{A}(x,y)]
$$

```{tip} Derivation
:class: dropdown
Prerequisite: **Log Derivative**

$$
\nabla_\theta\log f(\theta)=\frac{\nabla_\theta f(\theta)}{f(\theta)}
$$

Gradient:

$$\begin{align*}
\nabla_\theta J(\theta)&=\sum_y\nabla_\theta\pi_\theta(y|x)r(x,y) \\
&=\sum_y\pi_\theta(y|x)\nabla_\theta\log\pi_\theta(y|x)r(x,y) \\
&=E_{y\sim\pi_\theta(\cdot|x)}[\nabla_\theta\log\pi_\theta(y|x)r(x,y)]
\end{align*}$$
```

```{attention} Q&A
:class: dropdown
*Why reduce variance?*
- In RL for LLMs, we first sample complete trajectories (i.e., full $y$), then compute reward & estimate grads.
- → High variance across samples: Some outputs get super high reward while others super low.
- → High variance across tokens: All tokens get same high/low reward even if some of them are bad/good.

*What is advantage estimate?*
$$
\hat{A}(x,y)=r(x,y)-b(x)
$$
- $b(x)$: Baseline estimate of expected reward for input seq $x$. (often $V(x)$)
- $\hat{A}(x,y)$: How much is the reward better/worse than expected?

*Why advantage estimate?*
- Centering → Reduce scaling on rewards per token.
- No change in grad ← $b(x)$ not dependent on $y$.
$$
E_{y\sim\pi_\theta(\cdot|x)}[\nabla_\theta\log\pi_\theta(y|x)b(x)]=0
$$
```

&nbsp;

### PPO
- **Name**: Proximal Policy Optimization.
- **What**:
	1. **Importance Sampling (IS) ratio**: How much more/less likely an action is under $\pi_\text{new}$ vs $\pi_\text{old}$?
	2. **Clip** IS ratio → $\pi_\text{new}$ never strays too far from $\pi_\text{old}$.

<!-- # RL for LLMs
## RL
- **What**: Agent $\overset{\text{action}}{\underset{\text{reward}}{\rightleftarrows}}$ Environment
- **Why**: For decision-making where actions have delayed consequence in dynamic, sequential tasks.
    - In contrast, Supervised Learning teaches "correct answers" for static tasks.
- **How**:
    - **Objective**: Cumulative Reward $\xleftarrow{\text{maximize}}$ Optimal Policy
    - **Process**: Repeat: $s_t$ $\xrightarrow{a_t}$ $s_{t+1}$ $\xrightarrow{\text{get}}$ $r_t$ $\xrightarrow{\text{update}}$ $\pi$

## LLM Alignment
- **What**: LLM $\xrightarrow{\text{match}}$ human preferences {cite:p}`wang2024comprehensive`
- **Why**: ⬇️Undesired, sometimes harmful responses.
- **How**: Humans $\xrightarrow{\text{collect}}$ Feedback $\xrightarrow{\text{train}}$ Pretrained LLM

## RL for LLM Alignment
- **What**: Frame LLM Alignment as an RL problem {cite:p}`wang2024reinforcement`:
    - **Agent**: LLM.
    - **State**: Input token sequence.
    - **Action**: Next-token prediction.
    - **Next State**: Input token sequence + Predicted next token.
    - **Reward**: Reward.
        - Determined by an external reward model OR preference labels.
        - Typically computed after a full token sequence is generated.
    - **Policy**: LLM weights.
        - Dictates how LLM predicts next token given input token sequence.
        - Initial policy ← Pretraining (& SFT).
- **Why**: Human values are dynamic, subjective, and constantly evolving. There isn't always one "correct answer" for IRL scenarios, so SFT falls short.
- **How**:
    - **Process**: Feedback $\xrightarrow{\text{train}}$ RM $\xrightarrow{\text{train}}$ Policy
    - **Key factors**:
        - Feedback Data.
        - Reward Model.
        - Policy Optimization.

```{dropdown} Table 1: Feedback Data
| Subcategory | Type | Description | Pros | Cons |
|------------|------|-------------|------|------|
| **Label** | **Preference** | Rating on a scale ($y_w>y_l$) | Captures nuance | Hard to collect |
|  | **Binary** | Thumbs up & down ($y^+\ \&\ y^-$) | Easy to collect | Less informative (no middle ground) |
| **Style** | **Pairwise** | Compare 2 responses | Easy to interpret | Slow for large datasets (have to create pairs for all responses) |
|  | **Listwise** | Rank multiple responses at once | More informative, Fast | Hard to interpret |
| **Source** | **Human** | Feedback from human evaluators | Represents actual human values | Expensive, slow, inconsistent due to subjectivity |
|  | **AI** | Feedback generated by AI models | Cheap, fast, scalable | Does not necessarily represent human values (risk of unsafe responses) |
```

```{dropdown} Table 2: Reward Model
| Subcategory | Type | Description | Pros | Cons |
|------------|------|-------------|------|------|
| **Form** | **Explicit** | An external model, typically from SFT of a pretrained LLM | Interpretable & Scalable | High computational cost |
|  | **Implicit** | No external model (e.g., DPO) | Low computational cost, No reward overfitting | Less control |
| **Style** | **Pointwise** | Outputs a reward score $r(x,y)$ given an input-output pair | Simple & Interpretable | Ignores relative preferences |
|  | **Preferencewise** | Outputs probability of desired response being preferred over undesired response:<br>$P(y_w>y_l\|x)=\sigma(r(x,y_w)-r(x,y_l))$ | Provides comparisons | No pairwise preferences, Sensitive to human label inconsistencies |
| **Level** | **Token-level** | Reward is given per token/action | Fine-grained feedback | High computational cost, Noisy rewards |
|  | **Response-level** | Reward is given per response (most commonly used) | Simple | Coarse feedback |
| **Source** | **Positive** | Humans label both desired and undesired responses | More control | Expensive & Time-consuming |
|  | **Negative** | Humans label undesired responses, LLMs generate desired responses | Cheap & Scalable | Less control |
```

## RLHF (InstructGPT)
- **Name**: Reinforcement Learning from Human Feedback {cite:p}`ouyang2022training`
- **What**: RLHF + PPO/PPO-ptx.
- **Why**: LLM $\xrightarrow{\text{match}}$ human preferences
- **How**:
    - **Data**: Pairwise + Human.
    - **RM**: Explicit + Pointwise.
    - **PO**:
        1. **PPO**: Max Reward + **Min Deviation**
            - Deviation Minimization: Aligned policy $\Leftrightarrow$ Initial policy ← Trust Region Constraint
                - *Why?* To keep what works while aiming at what we want, via minimal changes. Drastic changes could make it forget what worked.
        2. **PPO-ptx**: Max Reward + Min Deviation + **Min Alignment Tax**
            - Alignment Tax Minimization: ❌Degradation of Pre/SFT task performance.

```{note} Math
:class: dropdown
RM:
- Reward Estimation (**Bradley-Terry Model**):

    $$
    p^*(y_w\succ y_l|x)=\frac{\exp{r^*(x,y_w)}}{\exp{r^*(x,y_w)}+\exp{r^*(x,y_l)}}
    $$
    - $x$: Input token sequence.
    - $y_w$: Desired (W) output token sequence.
    - $y_l$: Undesired (L) output token sequence.
    - $r^*(x,y)$: Latent reward function for input $x$ and output $y$.
    - $p^*(y_w\succ y_l|x)$: Probability that humans prefer $y_w$ over $y_l$.
- Objective:

    $$
    L_\text{RM}(r_\phi)=-\frac{1}{\binom{K}{2}}\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)\right]
    $$
    - $r_\phi(x,y)$: Reward model for input $x$ and output $y$, parameterized by $\phi$.
    - $\binom{K}{2}=\frac{K(K-1)}{2}$: #Comparisons for each prompt shown to each labeler.
    - $\mathcal{D}$: RL Dataset.
    - $\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)$: Sigmoid function → Bradley-Terry model.
---

PO:
- PPO:

    $$\begin{align*}
    L_\text{PPO}(\pi_\theta)&=\mathbb{E}_{x\sim\mathcal{D}}[\mathbb{E}_{y\sim\pi_\theta(y|x)}r_\phi(x,y)-\beta\text{KL}\left[\pi_\theta(y|x)||\pi_\text{ref}(y|x)\right]] \\
    &=\mathbb{E}_{x\sim\mathcal{D}}\left[\mathbb{E}_{y\sim\pi_\theta(y|x)}r_\phi(x,y)-\beta\mathbb{E}_{y\sim\pi_\theta(y|x)}\left[\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}\right]\right] \\
    &=\mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_\theta(y|x)}\left[r_\phi(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}\right]
    \end{align*}$$
    - $\pi_\theta(y|x)$: Curr policy, which gives the probability of generating $y$ given $x$.
    - $\pi_\text{ref}(y|x)$: Reference policy (initial policy of pretrained LLM).
    - $\beta$: KL divergence penalty coefficient.
    - $\text{KL}[\pi_\theta||\pi_\text{ref}]$: Per-token KL divergence.
- PPO-ptx:

    $$
    L_\text{PPO-ptx}(\pi_\theta)=\mathbb{E}_{x\sim\mathcal{D},y\sim\pi_\theta(y|x)}\left[r_\phi(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}\right]+\gamma\mathbb{E}_{x\sim\mathcal{D}_\text{pretrain}}[\log\pi_\theta(x)]
    $$
    - $\gamma$: Pretrain loss coefficient.
    - $\mathcal{D}_\text{pretrain}$: Pretraining dataset.
```

<br><br>

# Policy Optimization
## PPO
- **Name**: Proximal Policy Optimization {cite:p}`schulman2017proximalpolicyoptimizationalgorithms`
- **What**: Policy gradient, but **proximal** (close) to current policy.
- **Why**:
    1. Stable gradients.
    2. No optimal KL penalty coefficient for **TRPO** to work well.
- **How**: TRPO + **Better Penalty**
    1. **Clipped Surrogate Objective**: Trap the probability ratio in a range.
        - $\xrightarrow{\text{penalize}}$ deviation from current policy
    2. **Adaptive KL coefficient**: Adapt the coefficient to match a target KL divergence value per update.
        - $\xrightarrow{\text{minimize}}$ hyperparam tuning impact
    - (Empirically, 1>2)
    - (Empirically by Anthropic, ❌1&2, ✅$\beta=0.001$) {cite:p}`bai2022traininghelpfulharmlessassistant`

```{note} Math
:class: dropdown
TRPO (quick recap):
- **Probability Ratio**: How much more/less likely to take the given action under new policy vs old policy.

    $$
    \rho_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}
    $$
    - $\theta$: Policy parameter(s).
    - $t$: Curr time step.
    - $a_t$: Curr action.
    - $s_t$: Curr state.
    - $\pi_\theta$: New policy.
    - $\pi_{\theta_\text{old}}$: Old policy.
- **Advantage Estimate**: (roughly) How much better/worse of the given action compared to baseline.

    $$
    \hat{A}_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\left[r_{t+l}+\gamma V(s_{t+l+1})-V(s_{t+l})\right]
    $$
    - $T$: Total time steps.
    - $l$: Time step increment.
    - $\gamma$: Discount factor for future rewards.
    - $\lambda\in[0,1]$: Bias-variance tradeoff coefficient.
        - $\lambda=0$: One-step TD → ⬆️Bias, ⬇️Variance
        - $\lambda=1$: Full Monte Carlo return → ⬇️Bias, ⬆️Variance
    - $r_t$: Curr reward.
    - $V(s)$: Value function at state $s$.
- **KL Divergence**:

    $$
    \text{KL}[\pi_\theta(\cdot|s_t)||\pi_{\theta_\text{old}}(\cdot|s_t)]=\mathbb{E}_{a_t\in\mathcal{A}_t}\left[\log\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}\right]
    $$
    - $\mathcal{A}_t$: Curr action space.
- Objective:

    $$
    L_\text{TRPO}=\hat{\mathbb{E}}_t\left[\rho_t(\theta)\hat{A}_t-\beta\text{KL}[\pi_\theta(\cdot|s_t)||\pi_{\theta_\text{old}}(\cdot|s_t)]\right]
    $$

---
PPO:
- **Clipped Surrogate Objective**:
    - Clip Function:

        $$
        \text{clip}(x, a, b)=\begin{cases}
        a & \text{if } x\leq a \\
        x & \text{if } x\in(a,b) \\
        b & \text{if } x\geq b
        \end{cases}
        $$
    - Objective:

        $$
        L_\text{CLIP}(\theta)=\hat{\mathbb{E}}_t\left[\min\left(\rho_t(\theta)\hat{A}_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\right)\right]
        $$
        - $\epsilon$: Tiny value to control ratio change.
- **Adaptive KL Penalty Coefficient**:
    - For each policy update:
        1. Optimize TRPO objective.
        2. Compute **divergence**:

            $$
            d=\hat{\mathbb{E}_t}\left[\text{KL}[\pi_\theta(\cdot|s_t)||\pi_{\theta_\text{old}}(\cdot|s_t)]\right]
            $$
        3. Update $\beta$ via case switch:

            $$\begin{align*}
            &d<d_\text{tar} /1.5 &\Longrightarrow\ &\beta\leftarrow\beta/2 \\
            &d>d_\text{tar}\cdot 1.5 &\Longrightarrow\ &\beta\leftarrow\beta\cdot 2
            \end{align*}$$
            - $d_\text{tar}$: Target divergence.
```

## DPO
- **Name**: Direct Preference Optimization {cite:p}`rafailov2023direct`
- **What**: ❌RM → LM=RM → ✅Classification
- **Why**:
    - *Why do we need it?*
        - RLHF is unstable ← RM underfitting/overfitting
        - PPO is expensive ← Extra RM, Hyperparameter tuning, On-policy sampling, etc.
    - *Why does it even work?*
        1. PPO's KL-constrained reward maximization objective actually has a **closed-form solution**.
        2. The solution actually satisfies **Bradley-Terry model** (or other pairwise preference models).
        3. The model provides the **probability of human preference data in terms of the optimal policy** (❌RM).
        3. Yo, probability of data? → MLE → BCE
- **How**: Get data → Train LM to minimize BCE

```{note} Math
:class: dropdown
Optimal Policy:

$$\begin{align*}
\pi^*(y|x)=&\frac{1}{Z(x)}\pi_\text{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)\text{, where} \\
&r(x,y)=r_\phi(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} \\
&Z(x)=\sum_y\pi_\text{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) \\
\end{align*}$$
- $r(x,y)$: The reward function that gets maximized in PPO.
- $Z(x)$: Partition function, used for
    - normalization of optimal policy to a valid probability distribution.
    - dirty tricks in Derivation.

Reward Function reparameterized:

$$
r^*(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)}+\beta\log Z(x)
$$

Bradley-Terry Model reparameterized:

$$
p^*(y_w\succ y_l|x)=\frac{1}{1+\exp\left(\beta\log\frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)}-\beta\log\frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)}\right)}
$$

Objective (BCE):

$$
L_\text{DPO}(\pi_\theta|\pi_\text{ref})=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)}-\beta\log\frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]
$$

Gradient:

$$
\nabla_\theta L_\text{DPO}(\pi_\theta|\pi_\text{ref})=-\beta\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left\{\sigma\left(\hat{r}_\theta(x,y_l)-\hat{r}_\theta(x,y_w)\right)\left[\nabla_\theta\log\pi_\theta(y_w|x)-\nabla_\theta\log\pi_\theta(y_l|x)\right]\right\}
$$
- $\hat{r}_\theta(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)}$: Implicit reward model via LM.
- $\hat{r}_\theta(x,y_l)-\hat{r}_\theta(x,y_w)$: Higher update when reward estimate is wrong ($y_l\succ y_w$).
- $\log\pi_\theta(y|x)$: Log likelihood of $y|x$ → ⬆️$y_w$, ⬇️$y_l$.
```

```{tip} Derivation: PPO -> BCE 
:class: dropdown
1. Solve PPO for arbitrary reward function $r(x,y)$:

    $$\begin{align*}
    &\max_{\pi_\theta}\mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_\theta(y|x)}\left[r(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}\right] \\
    &=\min_{\pi_\theta}\mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_\theta(y|x)}\left[\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}-\frac{1}{\beta}r(x,y)\right]\\
    &=\min_{\pi_\theta}\mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_\theta(y|x)}\left[\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}+\log Z(x)-\frac{1}{\beta}r(x,y)-\log Z(x)\right] \\
    &=\min_{\pi_\theta}\mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_\theta(y|x)}\left[\log\frac{\pi_\theta(y|x)}{\frac{1}{Z(x)}\pi_\text{ref}(y|x)}-\log\exp\left(\frac{1}{\beta}r(x,y)\right)-\log Z(x)\right]\\
    &=\min_{\pi_\theta}\mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_\theta(y|x)}\left[\log\frac{\pi_\theta(y|x)}{\frac{1}{Z(x)}\pi_\text{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)}-\log Z(x)\right]\\
    &=\min_{\pi_\theta}\mathbb{E}_{x\sim\mathcal{D}}\left\{\left[\sum_y\pi_\theta(y|x)\log\frac{\pi_\theta(y|x)}{\frac{1}{Z(x)}\pi_\text{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)}\right]-\log Z(x)\right\}\\
    &=\min_{\pi_\theta}\mathbb{E}_{x\sim\mathcal{D}}\left\{\text{KL}\left[\pi_\theta(y|x)||\frac{1}{Z(x)}\pi_\text{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)\right]\right\}\\
    \end{align*}$$

    The minimal value of KL divergence is 0, where left = right. Thus,

    $$
    \pi^*(y|x)=\frac{1}{Z(x)}\pi_\text{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)
    $$

    This is a valid probability distribution because $\forall y: \pi^*(y|x)\geq0$ and $\sum_y\pi^*(y|x)=1$.

2. Derive $r(x,y)$ from the equation above.

3. Reformat Bradley-Terry Model to sigmoid function.

4. Formulate MLE, switch to BCE, calculate gradient via sigmoid derivative tricks.
```

## GRPO
- **Name**: Group Relative Policy Optimization {cite:p}`shao2024deepseekmath`
- **What**: ❌Value Function → ✅Average of multisampling from reference policy = baseline for Advantage Estimation.
- **Why**: PPO objective $\xleftarrow{\text{compute}}$ Advantage $\xleftarrow{\text{compute}}$ Extra value model (i.e., critic)
    - → ⬆️**Computational cost**
        - Value models can be as large as reward models.
        - e.g., OG RLHF/PPO initialized a 6B value function from a 6B RM {cite:p}`ouyang2022training`.
    - → ⬇️**Value model accuracy**
        - Value model is supposed to be accurate at each token.
        - BUT reward score is only computed at the last token.
        - SO there's no reward signal for intermediate tokens.
        - SO value model has to guess the final reward based on incomplete information.
        - → High variance + Training instability.

- **How**:
    - **Outcome Supervision**: For each input,
        1. Group of outputs $\xleftarrow{\text{sample}}$ reference policy
        2. $\xrightarrow{\text{RM on }\textbf{full sequence}}$ Rewards
        3. $\xrightarrow{\text{normalize}}$ Normalized rewards
        4. For **all tokens**, Advantages = Normalized rewards
        4. Optimize.
    - **Process Supervision**: For each input,
        1. Group of outputs $\xleftarrow{\text{sample}}$ reference policy
        2. $\xrightarrow{\text{RM on }\textbf{each step}}$ Rewards
        3. $\xrightarrow{\text{normalize}\textbf{ across all dims}}$ Normalized rewards
        4. For **each token**, Advantage = $\sum$ Normalized rewards after it.
        5. Optimize. -->

<!-- # Misc
This page collects miscellaneous techniques in LLM training.

## Knowledge Distillation
- **What**: Large, pre-trained **teacher** model $\xrightarrow{\text{transfer knowledge}}$ Small **student** model
    - Teacher & Student are trained on the same data.
    - Teacher sees true labels (i.e., **hard targets**).
    - Student sees both true labels and teacher's outputs (i.e., **soft targets**).
- **Why**: Model compression + Domain-specific knowledge transfer. -->