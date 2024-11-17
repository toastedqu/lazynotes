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
# Activations
Activation functions adds non-linearity to enable layers to learn complex patterns.

## Softmax
- **What**: Raw logits -> Probabilities.
- **Why**: Provides a **probability distribution** for **multi-class classification**.
- **How**:
    1. Adjust the input scale to exponential.
    2. Normalize the exponential scores as probabilities.
- **When**:
	- Mutually exclusive classes.
	- Cross-entropy loss.
- **Where**: Final layer in multi-class classification NNs.
- **Pros**:
	- Ensures probability outputs (between 0 and 1, sums up to 1).
	- High class differentiation (higher -> higher; lower -> lower).
	- High interpretability.
	- Differentiable.
- **Cons**:
	- Prone to overconfidence.
	- Sensitive to class imbalance.
	- Sensitive to input scale -> Prefers normalized logits.
	- Probabilities are NOT calibrated to actual distribution.
	- No sparsity.

```{admonition} Math
:class: note, dropdown

Notations:
- IO:
	- $\mathbf{x}\in\mathbb{R}^{n}$: Input vector of logits.
    - $\mathbf{y}\in\mathbb{R}^{n}$: Output vector of probabilities.

Forward:

$$
y_j=\frac{e^{x_j}}{\sum_{k=1}^{n}e^{x_{k}}}
$$

Backward:

$$
g_{x_j}=y_j(g_{y_j}-\mathbf{g}\mathbf{y})
$$
```

```{admonition} Derivation
:class: tip, dropdown
Assume $C=\sum_{k=1}^{n}e^{x_k}$, then $C'(x_j)=e^{x_j}$, then

$$\begin{align}
\frac{\partial y_{j}}{\partial x_{j}}&=\frac{Ce^{x_{j}}-e^{x_{j}}e^{x_{j}}}{C^2} \\
&=y_{j}-y_{j}^2 \\
&=y_{j}(1-y_{j}) \\
 \\
\frac{\partial y_{k}}{\partial x_{j}}&=-\frac{e^{x_{k}}e^{x_{j}}}{C^2} \\
&=-y_{k}y_{j}
\end{align}$$

Then,

$$\begin{align}
\frac{\partial L}{\partial x_{j}}&=\sum_{k=1}^n\frac{\partial L}{\partial y_{k}}\frac{\partial y_{k}}{\partial x_{j}} \\
&=\frac{\partial L}{\partial y_{j}}\frac{\partial y_{j}}{\partial x_{j}}+\sum_{k\neq j}\frac{\partial L}{\partial y_{k}}\frac{\partial y_{k}}{\partial x_{j}} \\
&=y_{j}\frac{\partial L}{\partial y_{j}}(1-y_{j})-y_{j}\sum_{k\neq j}\frac{\partial L}{\partial y_{k}}y_{k} \\
&=y_{j}\left(\frac{\partial L}{\partial y_{j}}-\mathbf{g}\mathbf{y}\right)
\end{align}$$
```