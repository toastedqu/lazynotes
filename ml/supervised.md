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
# Supervised Learning
# Linear Models
Linear models predicts outputs by combining input features with assigned params linearly.

## Linear Regression
- **What**: Models a linear relationship between input & output.
- **Why**: The simplest & most interpretable method for **regression**.
- **How**: Make a linear hyperplane between input & output.
    - **Inference**:
        1. Assign weights to features.
        2. Add them up.
    - **Training**:
        1. Measure the gap between predicted & actual outputs.
        2. Adjust the hyperplane to minimize the overall gap.

```{note} Math
:class: dropdown
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
```

```{tip} Derivation
:class: dropdown
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

```{attention} Q&A
:class: dropdown
*Pros?*
- ✅Simple.
- ✅Interpretable.
- ⬇️Computational cost.

*Cons?*
- ✅Scale variance.
- Assumptions fail IRL.
- ⬆️Sensitivity to outliers.
- Limited to regression (continuous output).

*Assumptions?*
- Error Independence: $\varepsilon_i$s are i.i.d.
- Error Homoskedasticity: $\text{Var}[\varepsilon_i]=\sigma^2$.
- Error Normality: $\varepsilon_i\sim N(0,\sigma^2)$.
- No multicollinearity.

*Properties?*
- Unbiasedness & Variance-Covariance: $\hat{\mathbf{w}}\sim N(\mathbf{w},\sigma^2(X^TX)^{-1})$.
- Invertibility & Multicollinearity: The matrix $X^TX$ is ONLY invertible if all features/columns of $X$ are linearly independent.
- Gauss-Markov Theorem: $\hat{\mathbf{w}}$ is BLUE (Best Linear Unbiased Estimator) of $\mathbf{w}$ as long as the following holds:
    - $\varepsilon_i$s are i.i.d.
    - $\varepsilon_i$s follow the same distribution where $\mu=0$ and $\sigma^2$ (NOT necessarily Normal distribution).
- Orthogonality: $(\mathbf{y}-X\hat{\mathbf{w}})\perp X$.
```

<br>
<br>

## Logistic Regression
- **What**: Predicts the probability of a binary outcome using a logistic function on top of linear regression.
- **Why**: The simplest & most interpretable method for **binary clsasfication**.
- **How**: A logistic function on top of linear regression.
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

```{note} Math
:class: dropdown
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

```{tip} Derivation
:class: dropdown

```

## Generalized Linear Models


* **I. Linear Models**

  * **A. Regression**

    * Ordinary Least Squares (OLS)
    * Ridge Regression
    * Lasso Regression
    * Elastic Net
    * Bayesian Linear Regression
    * Partial Least Squares
  * **B. Classification**

    * Logistic Regression
    * Softmax (Multinomial Logistic) Regression
    * Linear Discriminant Analysis (LDA)
    * Quadratic Discriminant Analysis (QDA)

* **II. Kernel Methods**

  * Support Vector Machines (SVM)
  * Kernel Ridge Regression
  * Gaussian Process Regression
  * Kernel Discriminant Analysis

* **III. Tree‑Based Methods**

  * Decision Trees

    * CART (Classification and Regression Trees)
    * C4.5 / C5.0
  * Ensemble Trees

    * Bagging

      * Random Forest
      * Extra Trees
    * Boosting

      * AdaBoost
      * Gradient Boosting Machines (GBM)
      * XGBoost
      * LightGBM
      * CatBoost
    * Stacking (Stacked Generalization)

* **IV. Instance‑Based (Memory‑Based) Methods**

  * k‑Nearest Neighbors (k‑NN)
  * Radius Neighbors
  * Locally Weighted Regression
  * Kernel Density Estimation (for classification)

* **V. Probabilistic (Generative) Models**

  * Naive Bayes Variants

    * Gaussian Naive Bayes
    * Multinomial Naive Bayes
    * Bernoulli Naive Bayes
    * Complement Naive Bayes
  * Bayesian Networks
  * Hidden Markov Models (with supervised training)

* **VI. Neural Networks and Deep Learning**

  * Multilayer Perceptron (MLP)
  * Convolutional Neural Networks (CNN)
  * Recurrent Neural Networks (RNN)

    * Long Short‑Term Memory (LSTM)
    * Gated Recurrent Unit (GRU)
  * Transformer‑Based Models
  * Residual Networks (ResNets)
  * Graph Neural Networks (GNN)

* **VII. Discriminant and Factorization Methods**

  * Fisher’s Linear Discriminant
  * Canonical Correlation Analysis (CCA)
  * Non‑negative Matrix Factorization (for regression tasks)

* **VIII. Hybrid and Meta‑Learning Methods**

  * Semi‑Parametric Models (e.g., Partial Parametric + Nonparametric)
  * Meta‑Learners (e.g., Model Agnostic Meta‑Learning, MAML)
  * Multiple Kernel Learning

* **IX. Specialized Regression and Classification**

  * Ordinal Regression
  * Quantile Regression
  * Survival Analysis Models (e.g., Cox Proportional Hazards with supervision)
  * Multi‑Label and Multi‑Output Methods

* **X. Feature‑Based and Projection Methods**

  * k‑Dimensional Trees (k‑d Trees for nearest‑neighbor search)
  * Random Projection Methods (Random Kitchen Sinks)
  * Spectral Regression

* **XI. Miscellaneous and Emerging Methods**

  * Energy‑Based Models
  * Attention‑Augmented Models
  * Capsule Networks
  * Self‑Normalizing Neural Networks
