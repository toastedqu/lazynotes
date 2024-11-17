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
# Linear Models
Linear models predicts outputs by combining input features with assigned parameters linearly.

## Linear Regression
- **What**: Linear regression models a linear relationship between input & output.
- **Why**: The simplest & most interpretable supervised learning method.
- **How**:
    1. Draw a flat hyperplane between input & output.
    2. Measure the gap (residual/error) between predicted and actual outputs.
    3. Adjust the hyperplane to minimize the overall gap.
- **When**:
    - Linear relationship between input & output.
    - Independence between errors.
    - Homoskedasticity (constant error variance).
    - No multicollinearity between features.
- **Where**: Analytics, feature importance analysis, trend analysis, baseline models, etc.
- **Pros**:
    - Simple.
    - Interpretable.
    - High computational efficiency.
    - Wide applicability.
- **Cons**:
    - Assumptions fail IRL.
    - High sensitivity to outliers.
    - Limited to regression (continuous output).

```{admonition} Math
:class: note, dropdown
Notations:
- IO:
    - $X\in\mathbb{R}^{m\times n}$: Input matrix.
    - $\mathbf{x}_i\in\mathbb{R}^{n}$: Input features of sample $i$.
    - $\mathbf{y}\in\mathbb{R}^{n}$: Output vector.
    - $y_i\in\mathbb{R}$: Output value of sample $i$.
- Params:
    - $\mathbf{w}\in\mathbb{R}^n$: Param vector.
- Others:
    - $\varepsilon_i\in\mathbb{R}$: Error for sample $i$.

Model:

$$
y_i=\mathbf{x}_i\mathbf{w}+\varepsilon_i
$$

Training:
- Objective (OLS):

    $$\begin{align*}
    L(\mathbf{w})&=\sum_{i=1}^{m}(y_i-\mathbf{x}_i\mathbf{w})^2 \\
    &=||\mathbf{y}-X\mathbf{w}||_2^2
    \end{align*}$$

- Optimization (Exact solution with gradient descent):

    $$
    \hat{\mathbf{w}}=(X^TX)^{-1}X^T\mathbf{y}
    $$

Inference:

$$
\hat{y}=\mathbf{x}\mathbf{w}
$$

Others:
- **Invertibility & Multicollinearity**: The matrix $X^TX$ is ONLY invertible if all features/columns of $X$ are linearly independent.

- **MLE & Error Independence**: $\hat{\mathbf{w}}\sim N(\mathbf{w},\sigma^2(X^TX)^{-1})$

- **Gauss-Markov Theorem & Error Independence**: $\hat{\mathbf{w}}$ is BLUE (Best Linear Unbiased Estimator) of $\mathbf{w}$ as long as the following holds:
    - $\varepsilon_i$s are i.i.d.
    - $\varepsilon_i$s follow the same distribution where $\mu=0$ and $\sigma^2$ (NOT necessarily Normal distribution).
```

```{admonition} Derivation
:class: tip, dropdown
Exact Solution:
1. Compute gradient & Set to 0:

    $$
    \frac{\partial L}{\partial\mathbf{w}}=-2X^T(\mathbf{y}-X\mathbf{w})=0
    $$

2. Solve for $\mathbf{w}$:

    $$\begin{align*}
    X^TX\mathbf{w}&=X^T\mathbf{y} \\
    \hat{\mathbf{w}}&=(X^TX)^{-1}X^T\mathbf{y}
    \end{align*}$$
```
## Logistic Regression
## Generalized Linear Models
