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
**[Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2403.14608)** fine-tunes a small part of a large model to save resources while retaining performance.

## LoRA
- **Name**: **[Low-Rank Adaptation](https://arxiv.org/pdf/2106.09685)**
- **What**: Injects **low-rank** matrices into the weights of specific layers, without modifying the original weights directly.
- **Why**: #params $\downarrow$ -> Finetuning becomes more **efficient**.
- **When**:
	- Linear transformation dominates the model architecture.
	- Optimal weight updates for downstream tasks lie in a low-rank subspace.
	- Initial weights are already optimized via **pretraining** -> only **small adaptations** are required for downstream tasks.
	- Parameter efficiency matters more than Performance (e.g., limited computational time & resources).
- **Where**: All transformer-based models.
- **Pros**:
	- High parameter efficiency -> High computational efficiency.
	- High scalability.
	- Task-specific adaptation without modifying the original params -> No overfitting.
- **Cons**:
	- ONLY applicable to linear transformation.
	- Lower performance relative to full finetuning.
	- ONLY great performance with pretrained models.
	- High sensitivity to hyperparameters.

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