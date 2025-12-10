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