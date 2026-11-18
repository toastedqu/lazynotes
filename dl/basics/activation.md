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
# Activation
Element-wise non-linearities applied after a linear layer.

Notations:
- $z$: Pre-activation (input to the activation).
- $y$: Activation output.
- $\mathcal{L}$: Loss.

&nbsp;

## Activation
- **What**: An element-wise non-linear function over a layer's output.
- **Why**: Non-linearity.
    - No activation → NN = LinReg.

### Binary-like
- **What**: Functions with near-binary outputs.
- **Why**: Biological neurons generally:
    - react little to small inputs.
    - react rapidly after input stimulus passes a threshold.
    - converge to a max as stimulus increases.

&nbsp;

#### Sigmoid
- **What**: Sigmoid function.
- **Why**: Mathematically convenient ← Smooth gradient.

```{note} Math
:class: dropdown
Forward:

$$
y=\sigma(z)=\frac{1}{1+e^{-z}}
$$
- $\sigma(z)\in(0,1)$
- $\sigma(0)=0.5$


Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\frac{\partial\mathcal{L}}{\partial y}y(1-y)
$$
```

```{attention} Q&A
:class: dropdown
*Cons?*
-  Vanishing gradient.
-  Non-zero centric bias → Non-zero mean activations
```

&nbsp;

#### Tanh
- **What**: Tanh function.
- **Why**: Mathematically convenient ← Smooth gradient.

```{note} Math
:class: dropdown
Forward:

$$
y=\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$
- $\tanh(z)\in(-1,1)$
- $\tanh(0)=0$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\frac{\partial\mathcal{L}}{\partial y}(1-y^2)
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Zero-centered.

*Cons?*
-  Vanishing gradient.
```

&nbsp;

### ReLU
- **What**: Rectified Linear Unit
- **Why**:
    - Binary-like activation functions suffered from vanishing gradients.
    - Biological neurons either fire or remain inactive.
    - ReLU-like functions existed long ago ([Householder, 1941](https://link.springer.com/article/10.1007/BF02478220)).
- **How**: Linear for positive, 0 for negative.

```{note} Math
:class: dropdown
Forward:

$$
y=\text{ReLU}(z)=\max{(0,z)}
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\begin{cases}
\frac{\partial\mathcal{L}}{\partial y} & z\geq0 \\
0 & z<0
\end{cases}
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- ❌Vanishing gradient.
- ✅Sparsity.
- ✅Computational efficiency.

*Cons?*
- **Dying ReLU**: If most inputs are negative, then most neurons output 0 → No gradient → No param update $\rightarrow $ Dead. (NOTE: A SOLVABLE DISADVANTAGE)
    - Cause 1: High learning rate $ \rightarrow$ Too much subtraction in param update → Weight⬇️⬇️ → Input for neuron⬇️⬇️.
    - Cause 2: Bias too negative → Input for neuron⬇️⬇️.
- Activation explosion $\longleftarrow$ $z\rightarrow\infty$. (NOTE: NOT A SEVERE DISADVANTAGE SO FAR)
```

&nbsp;

#### LReLU
- **What**: Leaky ReLU.
- **Why**: Dying ReLU.
- **How**: Linear for positive, tiny linear for negative.

```{note} Math
:class: dropdown
Forward:

$$
y=\text{LReLU}(z)=\max{(\alpha z,z)}
$$
- $\alpha\in(0,1)$: Negative slope hyperparam, default 0.01.

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\begin{cases}
\frac{\partial\mathcal{L}}{\partial y} & z\geq0 \\
\alpha\frac{\partial\mathcal{L}}{\partial y} & z<0
\end{cases}
$$
```

```{attention} Q&A
:class: dropdown
*Why aren't we using LReLU in place of ReLU?*

Because Dying ReLU became insignificant.
- Empirical performance: ReLU >> LReLU ← Sparsity
- Dying ReLU is solvable with other structural changes:
    - Weight Init → Ensure sufficient initial positive weights.
    - Batch Norm → Ensure $\sim$50% input values are positive.
    - Residual Connection → Gradients can flow directly back to input even if ReLU is dead.
```

&nbsp;

#### PReLU
- **What**: Parametric ReLU.
- **Why**: Fixed slope in LReLU.
- **How**: Scale negative linear outputs by a learnable $\alpha$.

```{note} Math
:class: dropdown
Forward:

$$
y=\mathrm{PReLU}(z)=\max{(\alpha z,z)}
$$
- $\alpha\in(0,1)$: Learnable negative slope param, default 0.25.

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\begin{cases}
\frac{\partial\mathcal{L}}{\partial y} & z\geq0 \\
\alpha\frac{\partial\mathcal{L}}{\partial y} & z<0
\end{cases}
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Adaptive param learned from data.

*Cons?*
- Computational cost ⬆️.
```

&nbsp;

#### RReLU
- **What**: Randomized ReLU.
- **Why**: Dying ReLU.
- **How**: Scale negative linear outputs by a random $ \alpha $.

```{note} Math
:class: dropdown
Forward:

$$
y=\mathrm{RReLU}(z)=\max{(\alpha z,z)}
$$
- $\alpha\sim\mathrm{Uniform}(l,u)$ in training.
- $\alpha=\frac{l+u}{2}$ in inference.
- $ l,u $: hyperparams (lower bound, upper bound).

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\begin{cases}
\frac{\partial\mathcal{L}}{\partial y} & z\geq0 \\
\alpha\frac{\partial\mathcal{L}}{\partial y} & z<0
\end{cases}
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Reduce overfitting by randomization.

*Cons?*
- Computational cost ⬆️.
```

&nbsp;

### ELU
- **What**: Exponential Linear Units.
- **Why**:
    - Dying ReLU.
    - Negative outputs push mean activations to 0 → bias shift ⬇️.
    - Smooth.
- **How**: Linear for positive, exponential saturation for negative.

```{note} Math
:class: dropdown
Forward:

$$
y=\text{ELU}(z)=\begin{cases}
z & z\ge 0\\
\alpha(e^z-1) & z<0
\end{cases}
$$
- $\alpha>0$: controls negative saturation level ($y\to -\alpha$ as $z\to -\infty$).

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\begin{cases}
\frac{\partial\mathcal{L}}{\partial y} & z\ge 0 \\
\frac{\partial\mathcal{L}}{\partial y}\cdot \alpha e^z & z<0
\end{cases}
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Non-zero gradient for $z<0$ → fewer “dead” neurons than ReLU.
- Negative outputs → activations closer to zero-mean.
- Smooth for $z<0$ (and continuous at $0$).

*Cons?*
- Saturation for very negative $z$: $\alpha e^z \to 0$ → gradients can still vanish on far negative side.
- Computational cost ⬆️ (exp) vs ReLU.
- Can be less GPU-friendly than piecewise-linear activations.
```

&nbsp;

#### SELU
- **What**: Scaled ELU.
- **Why**: Self-normalizing to $N(0,1)$ → Vanishing/Exploding activations ⬇️⬇️
- **How**: ELU scaled by fixed constants.

```{note} Math
:class: dropdown
Forward:

$$
y=\text{SELU}(z)=\lambda\begin{cases}
z & z\ge 0\\
\alpha(e^z-1) & z<0
\end{cases}
$$
- $\alpha \approx 1.6733, \lambda \approx 1.0507$


Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\begin{cases}
\lambda\frac{\partial\mathcal{L}}{\partial y} & z\ge 0 \\
\lambda\frac{\partial\mathcal{L}}{\partial y}\cdot \alpha e^z & z<0
\end{cases}
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Self-normalization.
- Reduce reliance on explicit normalization layers in some settings.

*Cons?*
- Useless with Normalization layers.
```

&nbsp;

#### CELU
- **What**: Continuously Differentiable Exponential Linear Unit.
- **Why**: ELU is NOT continuously differentiable at $0$ unless $\alpha=1$.
- **How**: Linear for positive, scaled exponential for negative with $z/\alpha$ inside the exp.

```{note} Math
:class: dropdown
Forward:

$$
y=\text{CELU}(z)=\begin{cases}
z & z\ge 0\\
\alpha\left(e^{z/\alpha}-1\right) & z<0
\end{cases}
$$
- $\alpha>0$: controls negative saturation level ($y\to -\alpha$ as $z\to -\infty$).

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\begin{cases}
\frac{\partial\mathcal{L}}{\partial y} & z\ge 0 \\
\frac{\partial\mathcal{L}}{\partial y}\cdot e^{z/\alpha} & z<0
\end{cases}
$$

- $\frac{d}{dz}\text{CELU}(0^-)=e^{0}=1=\frac{d}{dz}\text{CELU}(0^+)$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Continuously differentiable at $0$ for any $\alpha$ (smooth gradient through the transition).

*Cons?*
- Saturates for very negative $z$ ($e^{z/\alpha}\to 0$) → vanishing gradients far left.
- Computational cost ⬆️ (exp).
```

&nbsp;

### Other LUs
#### Softplus
- **What**: Smooth approximation to ReLU.
- **Why**: Differentiable everywhere (no kink at 0).
- **How**: Log-sum-exp smooth ramp.

```{note} Math
:class: dropdown
Forward:

$$
y=\text{Softplus}(z)=\ln(1+e^z)
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}
=\frac{\partial\mathcal{L}}{\partial y}\cdot \sigma(z)
=\frac{\partial\mathcal{L}}{\partial y}\cdot \frac{1}{1+e^{-z}}
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Smooth everywhere.
- No 0 gradient (unlike ReLU for $z<0$).

*Cons?*
- ❌ Sparsity.
- Computational cost ⬆️ (log/exp).
- Smaller gradients for large negative inputs, vs ReLU.
```

&nbsp;

#### SiLU
- **What**: Swish / Sigmoid Linear Unit.
- **Why**: Smooth and empirically competitive in CNN family.
- **How**: Gate input by its sigmoid.

```{note} Math
:class: dropdown
Forward:

$$
y=\text{SiLU}(z)=z\,\sigma(z)
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}
=\frac{\partial\mathcal{L}}{\partial y}\cdot \left(\sigma(z)+z\,\sigma(z)(1-\sigma(z))\right)
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Smooth.

*Cons?*
- Computational cost ⬆️ (sigmoid).
```

&nbsp;

#### GELU
- **What**: Gaussian Error Linear Unit.
- **Why**: Smooth, non-linear gating.
- **How**: Multiply input by probability it’s “kept” under a Gaussian.

```{note} Math
:class: dropdown
Forward (definition):

$$
y=\text{GELU}(z)=z\,\Phi(z)
$$

Backward (exact):

$$
\frac{\partial\mathcal{L}}{\partial z}
=\frac{\partial\mathcal{L}}{\partial y}\left(\Phi(z)+z\phi(z)\right)
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Empirically best in **transformers**.

*Cons?*
- Computational cost ⬆️ vs ReLU (though approximations help).
```

&nbsp;

### Softmax
- **What**: Numbers → Probabilities
- **Why**: Multiclass classification.
- **How**:
    1. Exponentiation: Larger/Smaller numbers → even larger/smaller numbers.
    2. Normalization: Numbers → Probabilities

```{note} Math
:class: dropdown
Forward:

$$
y_i=\text{softmax}(z_i)=\frac{\exp{(z_i)}}{\sum_j{\exp{(z_j)}}}
$$
- $i \& j$: Class indices. 

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z_i}=\begin{cases}
\frac{\partial\mathcal{L}}{\partial y_i}y_i(1-y_i) & i=j \\
-\frac{\partial\mathcal{L}}{\partial y_i}y_iy_j & i\neq j
\end{cases}
$$
```
