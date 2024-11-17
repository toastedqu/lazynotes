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
- **What**: Models a linear relationship between input & output.
- **Why**: The simplest & most interpretable method for **regression**.
- **How**:
    - **Data**: Input features & output values.
    - **Model**: A linear hyperplane between input & output.
    - **Inference**:
        1. Assign weights to features.
        2. Add them up.
    - **Training**:
        1. Measure the gap between predicted & actual outputs.
        2. Adjust the hyperplane to minimize the overall gap.
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
    - Scale variant.
    - Assumptions fail IRL.
    - High sensitivity to outliers.
    - Limited to regression (continuous output).

```{admonition} Math
:class: note, dropdown
Notations:
- IO:
    - $X=[\mathbf{x}_1,\cdots,\mathbf{x}_m]^T\in\mathbb{R}^{m\times n}$: Input matrix.
    - $\mathbf{y}=[y_1,\cdots,y_m]^T\in\mathbb{R}^{n}$: Output vector.
- Params:
    - $\mathbf{w}\in\mathbb{R}^n$: Param vector.
- Hyperparams:
    - (optional) $\lambda_1$: L1 weight.
    - (optional) $\lambda_2$: L2 weight.
    - (optional) $\eta$: Learning rate if gradient descent.
- Others:
    - $\varepsilon_i\in\mathbb{R}$: Error for sample $i$.

Model:

$$
y_i=\mathbf{x}_i^T\mathbf{w}+\varepsilon_i
$$

Inference:

$$
\hat{y}=\mathbf{x}^T\mathbf{w}
$$

Training:
- Objective (OLS):

    $$\begin{align*}
    L(\mathbf{w})&=\sum_{i=1}^{m}(y_i-\mathbf{x}_i^T\mathbf{w})^2 \\
    &=||\mathbf{y}-X\mathbf{w}||_2^2
    \end{align*}$$

- Optimization:
    - Normal Equation (if $X^TX$ is invertible):

        $$
        \hat{\mathbf{w}}=(X^TX)^{-1}X^T\mathbf{y}
        $$

    - Gradient Descent:

        $$
        \mathbf{w}\leftarrow\mathbf{w}-\eta\nabla_\mathbf{w}L
        $$

        - $\eta$: learning rate.

- Regularization:
    - L2 (i.e., Ridge):

        $$\begin{align*}
        L(\mathbf{w})&=||\mathbf{y}-X\mathbf{w}||_2^2+\lambda_2||\mathbf{w}||_2^2 \\
        \hat{\mathbf{w}}&=(X^TX+\lambda_2 I)^{-1}X^T\mathbf{y}
        \end{align*}$$

    - L1 (i.e., Lasso):

        $$
        L(\mathbf{w})=||\mathbf{y}-X\mathbf{w}||_2^2+\lambda_1||\mathbf{w}||_1
        $$

    - Elastic Net:

        $$
        L(\mathbf{w})=||\mathbf{y}-X\mathbf{w}||_2^2+\lambda_1||\mathbf{w}||_1+\lambda||\mathbf{w}||_2^2
        $$

Others:
- Assumptions:
    - Error Independence: $\varepsilon_i$s are i.i.d.
    - Error Homoskedasticity: $\text{Var}[\varepsilon_i]=\sigma^2$.
    - Error Normality: $\varepsilon_i\sim N(0,\sigma^2)$.
- Properties:
    - Unbiasedness & Variance-Covariance: $\hat{\mathbf{w}}\sim N(\mathbf{w},\sigma^2(X^TX)^{-1})$.
    - Invertibility & Multicollinearity: The matrix $X^TX$ is ONLY invertible if all features/columns of $X$ are linearly independent.
    - Gauss-Markov Theorem: $\hat{\mathbf{w}}$ is BLUE (Best Linear Unbiased Estimator) of $\mathbf{w}$ as long as the following holds:
        - $\varepsilon_i$s are i.i.d.
        - $\varepsilon_i$s follow the same distribution where $\mu=0$ and $\sigma^2$ (NOT necessarily Normal distribution).
    - Orthogonality: $(\mathbf{y}-X\hat{\mathbf{w}})\perp X$.
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
<br>
<br>

## Logistic Regression
- **What**: Predicts the probability of a binary outcome using a logistic function on top of linear regression.
- **Why**: The simplest & most interpretable method for **binary clsasfication**.
- **How**:
    - **Data**: Input features & output binary labels.
    - **Model**: A logistic function on top of linear regression.
    - **Inference**:
        1. [Linear Regression](#linear-regression).
        2. Use logistic (i.e., sigmoid) function to convert the logit into a probability.
        3. Convert the probability into a label based on a threshold $p=0.5$.
    - **Training**:
        1. Measure the overall gap between predicted **probabilities** and the actual labels with Binary Cross-Entropy loss.
        2. Adjust the weights to narrow the gap.
- **When**: See [Linear Regression](#linear-regression).
- **Where**: Simple binary classification tasks with limited data and computational resources.
- **Pros**:
    - Simple.
    - Interpretable <- Probabilistic outputs.
    - High computational efficiency.
    - Wide applicability.
    - Applicable to multiclass classification.
- **Cons**:
    - Scale variant.
    - Assumptions fail IRL.
    - High sensitivity to outliers.
    - Limited to classification (discrete output).

```{admonition} Math
:class: note, dropdown
Notations:
- IO:
    - $X=[\mathbf{x}_1,\cdots,\mathbf{x}_m]^T\in\mathbb{R}^{m\times n}$: Input matrix.
    - $\mathbf{y}=[y_1,\cdots,y_m]^T\in\mathbb{R}^{n}$: Output vector, where $y_i\in\{0,1\}$.
- Params:
    - $\mathbf{w}\in\mathbb{R}^n$: Param vector.
- Hyperparams:
    - (optional) $\lambda_1$: L1 weight.
    - (optional) $\lambda_2$: L2 weight.
    - (optional) $\eta$: Learning rate if gradient descent.
- Others:
    - $p_i\in[0,1]$: Predicted probability for sample $i$.
    - $\varepsilon_i\in\mathbb{R}$: Error for sample $i$.

Model:
- Probability:

    $$
    p_i=P(y_i=1|\mathbf{x}_i)=\frac{1}{1+e^{-\mathbf{x}_i^T\mathbf{w}}}
    $$

- Log-odds:

    $$
    \log\left(\frac{p_i}{1-p_i}\right)=\mathbf{x}_i^T\mathbf{w}
    $$

Inference:

$$\begin{align*}
p&=\frac{1}{1+e^{-\mathbf{x}^T\mathbf{w}}} \\
\hat{y}&=\begin{cases}
1 & \text{if }p\geq0.5 \\
0 & \text{if }p<0.5
\end{cases}
\end{align*}$$

Training:
- Objective (NLL/BCE):

    $$\begin{align*}
    L(\mathbf{w})&=-\sum_{i=1}^{m}\left(y_i\log(p_i)+(1-y_i)\log(1-p_i)\right) \\
    &=\sum_{i=1}^{m}\left(\log\left(1+e^{\mathbf{x}_i^T\mathbf{w}}\right)+(1-y_i)(\mathbf{x}_i^T\mathbf{w})\right)
    \end{align*}$$

- Optimization:
    - Normal Equation (if $X^TX$ is invertible):

        $$
        \hat{\mathbf{w}}=(X^TX)^{-1}X^T\mathbf{y}
        $$

    - Gradient Descent:

        $$
        \mathbf{w}\leftarrow\mathbf{w}-\eta\frac{\partial L(\mathbf{w})}{\mathbf{w}} 
        $$

        - $\eta$: learning rate.

- Regularization:
    - L2 (i.e., Ridge):

        $$\begin{align*}
        L(\mathbf{w})&=||\mathbf{y}-X\mathbf{w}||_2^2+\lambda_2||\mathbf{w}||_2^2 \\
        \hat{\mathbf{w}}&=(X^TX+\lambda_2 I)^{-1}X^T\mathbf{y}
        \end{align*}$$

    - L1 (i.e., Lasso):

        $$
        L(\mathbf{w})=||\mathbf{y}-X\mathbf{w}||_2^2+\lambda_1||\mathbf{w}||_1
        $$

    - Elastic Net:

        $$
        L(\mathbf{w})=||\mathbf{y}-X\mathbf{w}||_2^2+\lambda_1||\mathbf{w}||_1+\lambda||\mathbf{w}||_2^2
        $$

```

```{admonition} Derivation
:class: tip, dropdown

```

## Generalized Linear Models
