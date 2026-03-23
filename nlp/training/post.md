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
# Post-Training
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
