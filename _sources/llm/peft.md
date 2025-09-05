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
# PEFT
- **What**: Modify a small part of a large model $\xrightarrow{\text{adapt to}}$ Downstream tasks
- **Why**: Reduce cost & Retain pretraining performance.

## LoRA (Low-Rank Adaptation)
- **What**: Train **low-rank** matrices to adapt to downstream tasks.
- **Why**: #params ⇓ $\rightarrow$ Finetuning efficiency ⇑.
- **How**: Inject low-rank matrices into the weights of specific layers, w/o modifying the original weights directly.

```{admonition} Math
:class: note, dropdown
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

```{admonition} Q&A
:class: tip, dropdown
*Pros*:
- No overfitting $\leftarrow$ Task-specific adaptation w/o modifying the original params

*Cons*:
- ONLY applicable to linear transformation.
- ONLY great performance with pretrained models $\rightarrow$ Lower performance relative to full finetuning
- High sensitivity to hyperparameters.
```

## QLoRA (LoRA for Quantized LLMs)
- **What**: Insert learnable LoRAs into each layer of a quantized pretrained LM.
- **Why**: Standard fine-tuning scales linearly with model size in memory $\rightarrow$ HUGE memory cost
	- QLoRA dramatically cuts down memory WHILE preserving full-precision performance.
- **How**:
	1. **4-bit NormalFloat Quantization**: 