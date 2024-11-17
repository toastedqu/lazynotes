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
# Linear
- **What**: Linear transformation.
- **Why**: The simplest way to transform data, learn patterns, and make predictions.
- **Conditions**: The input-output relationship is linear.
- **Applications**: Anywhere. Typically used for feature dimension transformation.
- **Pros**: Simple, high interpretability, high computational efficiency, widely applicable.
- **Cons**: Cannot capture non-linear/complex patterns.

```{admonition} Math
:class: note, dropdown
Notations:
- IO:
	- $X\in\mathbb{R}^{*\times H_{in}}$: Input tensor.
	- $Y\in\mathbb{R}^{*\times H_{out}}$: Output tensor.
- Params:
	- $W\in\mathbb{R}^{H_{in}\times H_{out}}$: Weight matrix.
	- $\textbf{b}\in\mathbb{R}^{H_{out}}$: Bias vector.
- Hyperparams:
	- $H_{in}$: Input feature dimension.
	- $H_{out}$: Output feature dimension.

Forward:

$$
Y=XW+\textbf{b}
$$

Backward:

$$\begin{align}
&g_{W}=X^Tg_{Y} \\
&g_{\mathbf{b}}=\sum_{*}g_{Y}\\
&g_{X}=g_{Y}W^T
\end{align}$$
```

```{admonition} Derivation
:class: tip, dropdown
$g_{W}\in\mathbb{R}^{H_{in}\times H_{out}}$:

$$\begin{align}
\frac{\partial L}{\partial W}&=\frac{\partial Y}{\partial W}\frac{\partial L}{\partial Y} \\
&=X^T\frac{\partial L}{\partial Y}
\end{align}$$

$g_{b}\in\mathbb{R}^{H_{out}}$:

$$\begin{align}
\frac{\partial L}{\partial \textbf{b}}&=\frac{\partial Y}{\partial \textbf{b}}\frac{\partial L}{\partial Y} \\
&=\mathbf{1}\cdot\frac{\partial L}{\partial Y} \\
&=\sum_{*}\frac{\partial L}{\partial Y}
\end{align}$$

$g_{\mathbf{x}}\in\mathbb{R}^{H_{in}}$:

$$\begin{align}
\frac{\partial L}{\partial X}&=\frac{\partial L}{\partial Y}\frac{\partial Y}{\partial X} \\
&=\frac{\partial L}{\partial \mathbf{y}}W^T
\end{align}$$
```