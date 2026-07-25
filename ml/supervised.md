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
Learn a mapping $f:\mathcal{X}\to\mathcal{Y}$ from labeled data $\{(\mathbf{x}_i,y_i)\}_{i=1}^m$.

This page covers prevalent traditional supervised methods ONLY.

Default notations:
- $X=[\mathbf{x}_1,\cdots,\mathbf{x}_m]^T\in\mathbb{R}^{m\times n}$: Input matrix ($m$ samples, $n$ features).
- $\mathbf{y}=[y_1,\cdots,y_m]^T$: Output vector.
- $\mathbf{w}\in\mathbb{R}^n$: Param vector. Bias absorbed by appending a constant-1 feature, unless stated otherwise.
- Penalties never apply to the bias — assume $X,\mathbf{y}$ centered & the bias split out wherever a penalty appears.

&nbsp;

## Linear Models
- **What**: Weighted sum of features, optionally passed through a link function.

### Linear Regression
- **What**: Weighted sum of features.
- **Why**: Simplest & most interpretable regressor → default baseline.
- **How**:
    - **Inference**: Weight each feature → Sum.
    - **Training**:
        1. Measure squared gap between predicted & actual outputs.
        2. Move the hyperplane to minimize the total gap.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $X\in\mathbb{R}^{m\times n}$: Input matrix.
    - $\mathbf{y}\in\mathbb{R}^{m}$: Output vector.
- Params:
    - $\mathbf{w}\in\mathbb{R}^n$: Param vector.
- Hyperparams:
    - (optional) $\eta$: Learning rate if gradient descent.
- Misc:
    - $\varepsilon_i\in\mathbb{R}$: Error/noise for sample $i$.

Model:

$$
y_i=\mathbf{x}_i^T\mathbf{w}+\varepsilon_i
$$

Inference:

$$
\hat{y}=\mathbf{x}^T\mathbf{w}
$$

Training:
- Objective (OLS = Ordinary Least Squares):

    $$\begin{align*}
    L(\mathbf{w})&=\sum_{i=1}^{m}(y_i-\mathbf{x}_i^T\mathbf{w})^2 \\
    &=||\mathbf{y}-X\mathbf{w}||_2^2
    \end{align*}$$

- Optimization:
    - Normal equation (iff $X^TX$ is invertible):

        $$
        \hat{\mathbf{w}}=(X^TX)^{-1}X^T\mathbf{y}
        $$

    - Gradient descent:

        $$
        \mathbf{w}\leftarrow\mathbf{w}+2\eta X^T(\mathbf{y}-X\mathbf{w})
        $$
```

```{tip} Derivation
:class: dropdown
*Where does the normal equation come from?*
1. Compute gradient & set to 0:

    $$
    \frac{\partial L}{\partial\mathbf{w}}=-2X^T(\mathbf{y}-X\mathbf{w})=0
    $$

2. Solve for $\mathbf{w}$:

    $$\begin{align*}
    X^TX\mathbf{w}&=X^T\mathbf{y} \\
    \hat{\mathbf{w}}&=(X^TX)^{-1}X^T\mathbf{y}
    \end{align*}$$

*Why squared error and not something else?*
1. Assume $\varepsilon_i\overset{iid}{\sim}N(0,\sigma^2)$ → $y_i\sim N(\mathbf{x}_i^T\mathbf{w},\sigma^2)$.
2. Log-likelihood:

    $$
    \log P(\mathbf{y}|X,\mathbf{w})=-\frac{m}{2}\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_{i=1}^m(y_i-\mathbf{x}_i^T\mathbf{w})^2
    $$

3. Only the last term depends on $\mathbf{w}$ → Maximizing likelihood $\Leftrightarrow$ Minimizing SSE.
4. → **OLS = MLE under Gaussian noise**.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.w = None

    def _design(self, X):
        ## absorb bias into w by appending a constant-1 column
        return np.hstack([np.ones((X.shape[0], 1)), X]) if self.fit_intercept else X

    def fit(self, X, y):
        X = self._design(X)
        ## lstsq == normal equation but numerically stable (QR/SVD, no explicit inverse)
        self.w = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def fit_gd(self, X, y, lr=1e-2, steps=1000):
        X = self._design(X)
        self.w = np.zeros(X.shape[1])
        for _ in range(steps):
            ## grad of ||y - Xw||^2 is -2 X^T (y - Xw); /m keeps lr scale-free in m
            self.w -= lr * (-2 / X.shape[0]) * X.T @ (y - X @ self.w)
        return self

    def predict(self, X):
        return self._design(X) @ self.w

## Example
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([2.0, 4.0, 6.0, 8.0])
print(LinearRegression().fit(X, y).predict(np.array([[5.0]])))  ## [10.]
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ✅Simple, ✅Interpretable, ✅Closed-form.
- ⬇️Compute, ⬇️Data requirement.
- Confidence intervals & p-values available → statistically testable.
- $w_j$ = marginal effect of feature $j$ → directly readable.

*Cons?*
- Underfits ← Assumes linearity & additivity.
- ⬆️Sensitivity to outliers ← Squared error penalizes large residuals quadratically.
- Breaks under multicollinearity ($X^TX$ singular).
- Extrapolates blindly outside the training range.

*Assumptions?*
- **Linearity**: $\mathbb{E}[y|\mathbf{x}]$ is linear in $\mathbf{w}$.
- **Independence**: $\varepsilon_i$s are independent.
- **Homoskedasticity**: $\text{Var}[\varepsilon_i]=\sigma^2$, constant.
- **Normality**: $\varepsilon_i\sim N(0,\sigma^2)$ — needed ONLY for CIs / p-values, NOT for OLS itself.
- **No multicollinearity**: Columns of $X$ linearly independent.
- **Exogeneity**: $\mathbb{E}[\varepsilon|\mathbf{x}]=0$.

*Properties?*
- **Gauss-Markov**: OLS is BLUE (Best Linear Unbiased Estimator) under zero-mean, uncorrelated, homoskedastic errors. ❌Normality needed.
- **Sampling distribution** (adds normality): $\hat{\mathbf{w}}\sim N(\mathbf{w},\sigma^2(X^TX)^{-1})$.
- **Orthogonality**: Residual $\perp$ column space of $X$, i.e., $X^T(\mathbf{y}-X\hat{\mathbf{w}})=\mathbf{0}$.
- **Invertibility**: $X^TX$ invertible $\Leftrightarrow$ $\text{rank}(X)=n$ $\Leftrightarrow$ no perfect multicollinearity.

*Does feature scaling matter?*
- ❌ for plain OLS predictions ← Rescaling feature $j$ just rescales $w_j$ inversely → fitted values unchanged.
- ✅ for regularized variants (penalty is scale-dependent) & for GD convergence speed.

*How to detect multicollinearity?*
- **VIF** (Variance Inflation Factor): $\text{VIF}_j=\frac{1}{1-R_j^2}$, where $R_j^2$ = $R^2$ of regressing feature $j$ on the rest.
    - VIF > 5–10 → problematic.
- Fix: Drop features / PCA / Ridge.

*Why is $R^2$ misleading?*
- $R^2$ never decreases when adding a feature, even a random one → use $R^2_\text{adj}$ or held-out error.
- $R^2$ can be negative on test data.

*Handling nonlinearity while staying "linear"?*
- Linear in **params**, not in features → basis expansion (polynomial, splines, interactions) keeps the closed form.
```

&nbsp;

#### Ridge Regression
- **What**: Linear regression + L2 penalty on the weights.
- **Why**:
    - Multicollinearity → $X^TX$ near-singular → huge $\text{Var}[\hat{\mathbf{w}}]$ → unstable coefficients.
    - $+\lambda I$ makes $X^TX+\lambda I$ positive definite for $\lambda>0$ → always solvable, even when $n>m$.
- **How**: Shrink all weights toward 0 proportionally to their size.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda\ge0$: L2 penalty weight.

Objective:

$$
L(\mathbf{w})=||\mathbf{y}-X\mathbf{w}||_2^2+\lambda||\mathbf{w}||_2^2
$$

Solution:

$$
\hat{\mathbf{w}}=(X^TX+\lambda I)^{-1}X^T\mathbf{y}
$$

Shrinkage in the SVD basis ($X=U\Sigma V^T$, singular values $d_j$):

$$
X\hat{\mathbf{w}}=\sum_{j=1}^n\mathbf{u}_j\frac{d_j^2}{d_j^2+\lambda}\mathbf{u}_j^T\mathbf{y}
$$
- $\mathbf{u}_j$: $j$-th left singular vector.
- Effective degrees of freedom: $\text{df}(\lambda)=\sum_j\frac{d_j^2}{d_j^2+\lambda}$.
```

```{tip} Derivation
:class: dropdown
*Why is Ridge always solvable?*
1. $X^TX$ is PSD → eigenvalues $\ge0$.
2. $X^TX+\lambda I$ has eigenvalues $\text{eig}(X^TX)+\lambda>0$ for $\lambda>0$ → positive definite → invertible.

*Why is Ridge = MAP with a Gaussian prior?*
1. Prior $\mathbf{w}\sim N(0,\tau^2I)$, likelihood $\mathbf{y}|X,\mathbf{w}\sim N(X\mathbf{w},\sigma^2I)$.
2. $-\log P(\mathbf{w}|X,\mathbf{y})\propto\frac{1}{2\sigma^2}||\mathbf{y}-X\mathbf{w}||_2^2+\frac{1}{2\tau^2}||\mathbf{w}||_2^2$.
3. → Ridge with $\lambda=\sigma^2/\tau^2$.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Trades a little bias for a lot of variance ⬇️.

*Why doesn't Ridge produce exact zeros?*
- Penalty gradient $2\lambda w_j\to0$ as $w_j\to0$ → no force strong enough to pin a weight at exactly 0.
- Geometrically: the L2 ball is smooth (no corners) → the contact point with the loss contour is generically off-axis.

*Must I standardize features?*
- ✅ Yes. The penalty is not scale-invariant → a feature measured in mm gets a 1000× smaller weight than one in m, so it gets penalized 1000× less.
- ❌ Never penalize the intercept (it just shifts the target's mean).

*What does $\lambda$ do?*
- $\lambda=0$ → OLS. $\lambda\to\infty$ → $\hat{\mathbf{w}}\to0$.
- $\lambda$⬆️ → Bias⬆️, Variance⬇️. Pick by CV.

*When Ridge over Lasso?*
- Many small, correlated effects, all genuinely useful → Ridge (keeps & averages them).
- Few large effects, most features irrelevant → Lasso.
```

&nbsp;

#### Lasso Regression
- **What**: Linear regression + L1 penalty on the weights.
- **Why**: Sparsity.
    - Irrelevant weights set to **exactly 0**.
    - → regularization + automatic feature selection in one shot.
- **How**: Penalize $|w_j|$ → constant pull of size $\lambda$ toward 0 regardless of magnitude → small weights get pinned to 0.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda\ge0$: L1 penalty weight.

Objective (❌closed form ← non-differentiable at 0):

$$
L(\mathbf{w})=\frac{1}{2}||\mathbf{y}-X\mathbf{w}||_2^2+\lambda||\mathbf{w}||_1
$$

Optimization: Coordinate descent — cycle over $j$, holding the rest fixed:

$$
w_j\leftarrow\frac{S(\mathbf{x}_{:j}^T\mathbf{r}_{-j},\ \lambda)}{\mathbf{x}_{:j}^T\mathbf{x}_{:j}}
$$
- $\mathbf{x}_{:j}$: $j$-th column of $X$.
- $\mathbf{r}_{-j}=\mathbf{y}-\sum_{l\neq j}\mathbf{x}_{:l}w_l$: Partial residual.
- $S(z,\lambda)=\text{sign}(z)\max(|z|-\lambda,0)$: **Soft-thresholding** operator.
```

```{tip} Derivation
:class: dropdown
*Why does L1 give exact zeros (and L2 doesn't)?*

Take orthonormal $X$ ($X^TX=I$) so coordinates decouple, and let $z_j=\mathbf{x}_{:j}^T\mathbf{y}$ be the OLS solution.

1. **L1**: minimize $\frac{1}{2}(w_j-z_j)^2+\lambda|w_j|$.
    - Subgradient: $w_j-z_j+\lambda\,\text{sign}(w_j)=0$ for $w_j\neq0$.
    - $0$ is optimal whenever $|z_j|\leq\lambda$ ← the subgradient interval $[-\lambda,\lambda]$ contains $z_j$.

    $$
    \hat{w}_j=\text{sign}(z_j)(|z_j|-\lambda)_+
    $$

2. **L2**: minimize $\frac{1}{2}(w_j-z_j)^2+\lambda w_j^2$.

    $$
    \hat{w}_j=\frac{z_j}{1+2\lambda}
    $$

3. → L1 **truncates** (a flat dead zone of width $2\lambda$), L2 **rescales** (never exactly 0).

*Bayesian view*: Lasso = MAP under a Laplace prior $P(w_j)\propto e^{-|w_j|/b}$. The prior puts zero point *mass* at 0 — exact zeros come from the kink in its log-density (a subgradient interval), not from atomic probability.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class Lasso:
    def __init__(self, lam=1.0, n_iter=100):
        self.lam, self.n_iter = lam, n_iter
        self.w, self.b = None, 0.0

    @staticmethod
    def _soft(z, lam):
        ## soft-threshold: the ONLY place sparsity comes from
        return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)

    def fit(self, X, y):
        ## center so the intercept is never penalized (penalty applies to slopes only)
        self.x_mean, self.y_mean = X.mean(axis=0), y.mean()
        X, y = X - self.x_mean, y - self.y_mean
        m, n = X.shape
        self.w = np.zeros(n)
        for _ in range(self.n_iter):
            for j in range(n):
                ## residual with feature j's own contribution removed
                r_j = y - X @ self.w + X[:, j] * self.w[j]
                self.w[j] = self._soft(X[:, j] @ r_j, self.lam) / (X[:, j] @ X[:, j])
        self.b = self.y_mean - self.x_mean @ self.w
        return self

    def predict(self, X):
        return X @ self.w + self.b

## Example: feature 1 is pure noise -> should be zeroed out
rng = np.random.default_rng(0)
X = rng.normal(size=(50, 2))
y = 3 * X[:, 0]
print(np.round(Lasso(lam=5.0).fit(X, y).w, 3))  ## [~2.9  0.   ]
```
````

```{attention} Q&A
:class: dropdown
*Cons?*
- $n>m$ → selects at most $m$ features.
- Correlated group of features → picks ONE arbitrarily, zeros the rest → unstable selection across resamples.
- ❌Closed form → iterative (coordinate descent / LARS).
- Biases the surviving (nonzero) coefficients toward 0 → some use Lasso to select, then refit OLS on the survivors ("relaxed Lasso").

*Path property?*
- Solution path is piecewise linear in $\lambda$ → LARS computes the ENTIRE path for the cost of one OLS fit.
- $\lambda\ge\max_j|\mathbf{x}_{:j}^T\mathbf{y}|$ → all weights 0.

*L1 vs L2 in one line?*
- L1 → sparse, feature selection, robust to irrelevant features.
- L2 → dense, stable, handles correlated features by averaging them.
```

&nbsp;

#### Elastic Net
- **What**: Linear regression + a convex mix of L1 & L2.
- **Why**:
    - Lasso fails on correlated features (picks 1 arbitrarily) & caps selection at $m$ features.
    - Adding L2 restores the **grouping effect** (correlated features get similar weights) & lifts the $m$-feature cap.
- **How**: Penalize $\lambda_1||\mathbf{w}||_1+\lambda_2||\mathbf{w}||_2^2$ → L1 selects, L2 stabilizes.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda_1$: L1 weight.
    - $\lambda_2$: L2 weight.
    - Equivalently $\alpha=\lambda_1+\lambda_2$ (strength) & $\rho=\frac{\lambda_1}{\lambda_1+\lambda_2}$ (L1 ratio).

Objective:

$$
L(\mathbf{w})=||\mathbf{y}-X\mathbf{w}||_2^2+\lambda_1||\mathbf{w}||_1+\lambda_2||\mathbf{w}||_2^2
$$

Special cases: $\rho=1$ → Lasso. $\rho=0$ → Ridge.
```

```{dropdown} Table: Regularized Linear Regression
| | Penalty | Closed form | Sparse | Correlated features | Bayesian prior |
|:--|:--|:--|:--|:--|:--|
| OLS | — | ✅ | ❌ | Breaks (singular $X^TX$) | Flat |
| Ridge | $\lambda\lVert\mathbf{w}\rVert_2^2$ | ✅ | ❌ | Shrinks the whole group | Gaussian |
| Lasso | $\lambda\lVert\mathbf{w}\rVert_1$ | ❌ | ✅ | Picks 1 arbitrarily | Laplace |
| Elastic Net | Both | ❌ | ✅ | Keeps the group together | Compromise |
```

&nbsp;

### Logistic Regression
- **What**: Sigmoid on a weighted sum of features.
- **Why**: Simplest & most interpretable **binary classifier**.
    - Outputs a **probability**, not just a label → threshold tunable post-hoc.
- **How**:
    - **Inference**:
        1. Linear score (logit) $z=\mathbf{x}^T\mathbf{w}$.
        2. Sigmoid squashes $z\in\mathbb{R}$ → $p\in(0,1)$.
        3. Threshold $p$ (default 0.5) → label.
    - **Training**: Minimize BCE.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $X\in\mathbb{R}^{m\times n}$: Input matrix.
    - $\mathbf{y}\in\{0,1\}^m$: Label vector.
- Params:
    - $\mathbf{w}\in\mathbb{R}^n$: Param vector.
- Hyperparams:
    - $\eta$: Learning rate.
    - (optional) $\lambda$: Penalty weight.
- Misc:
    - $p_i\in(0,1)$: Predicted probability for sample $i$.
    - $\mathbf{p}=[p_1,\cdots,p_m]^T$: Probability vector.
    - $\sigma(z)=\frac{1}{1+e^{-z}}$: Sigmoid.

Model:

$$
p_i=P(y_i=1|\mathbf{x}_i)=\sigma(\mathbf{x}_i^T\mathbf{w}),\qquad\log\frac{p_i}{1-p_i}=\mathbf{x}_i^T\mathbf{w}
$$

Inference:

$$
\hat{y}=\mathbb{1}[\sigma(\mathbf{x}^T\mathbf{w})\geq\tau],\quad\tau=0.5\text{ by default}
$$

Training:
- Objective (NLL = BCE):

    $$\begin{align*}
    L(\mathbf{w})&=-\sum_{i=1}^{m}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right] \\
    &=\sum_{i=1}^{m}\left[\log\left(1+e^{\mathbf{x}_i^T\mathbf{w}}\right)-y_i\mathbf{x}_i^T\mathbf{w}\right]
    \end{align*}$$

- Gradient & Hessian (❌closed form):

    $$
    \nabla_\mathbf{w}L=X^T(\mathbf{p}-\mathbf{y}),\qquad\nabla_\mathbf{w}^2L=X^TSX\succeq0
    $$

    - $S=\text{diag}(p_i(1-p_i))$.
    - Hessian PSD → $L$ convex → global optimum.

- Optimization:
    - GD: $\mathbf{w}\leftarrow\mathbf{w}-\eta X^T(\mathbf{p}-\mathbf{y})$
    - Newton / IRLS: $\mathbf{w}\leftarrow\mathbf{w}-(X^TSX)^{-1}X^T(\mathbf{p}-\mathbf{y})$

- Regularization (L2 shown; L1 & Elastic Net analogous):

    $$
    L(\mathbf{w})=-\sum_{i=1}^{m}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]+\lambda||\mathbf{w}||_2^2
    $$
```

```{tip} Derivation
:class: dropdown
*Where does BCE come from?*
1. $y_i|\mathbf{x}_i\sim\text{Bernoulli}(p_i)$ → $P(y_i|\mathbf{x}_i)=p_i^{y_i}(1-p_i)^{1-y_i}$.
2. Log-likelihood over i.i.d. samples:

    $$
    \ell(\mathbf{w})=\sum_{i=1}^m\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
    $$

3. Minimize $-\ell$ → **BCE = MLE for a Bernoulli response**.

*Where does the clean gradient come from?*
1. $\sigma'(z)=\sigma(z)(1-\sigma(z))=p(1-p)$.
2. Per-sample: $\frac{\partial}{\partial p}\left[-y\log p-(1-y)\log(1-p)\right]=\frac{p-y}{p(1-p)}$.
3. Chain rule: $\frac{\partial L_i}{\partial\mathbf{w}}=\frac{p_i-y_i}{p_i(1-p_i)}\cdot p_i(1-p_i)\cdot\mathbf{x}_i=(p_i-y_i)\mathbf{x}_i$.
4. Stack: $\nabla_\mathbf{w}L=X^T(\mathbf{p}-\mathbf{y})$ — identical in form to linear regression's gradient (this is a GLM property, not a coincidence).
```

````{important} Code
:class: dropdown
```python
import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.1, steps=1000, lam=0.0):
        self.lr, self.steps, self.lam = lr, steps, lam
        self.w = None

    @staticmethod
    def _sigmoid(z):
        ## branch on sign so exp() never sees a large positive argument
        out = np.empty_like(z, dtype=float)
        pos, neg = z >= 0, z < 0
        out[pos] = 1 / (1 + np.exp(-z[pos]))
        e = np.exp(z[neg])
        out[neg] = e / (1 + e)
        return out

    def fit(self, X, y):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.w = np.zeros(X.shape[1])
        for _ in range(self.steps):
            p = self._sigmoid(X @ self.w)
            grad = X.T @ (p - y)                    ## X^T (p - y): the whole gradient
            grad[1:] += 2 * self.lam * self.w[1:]   ## L2, never on the intercept
            self.w -= self.lr / X.shape[0] * grad
        return self

    def predict_proba(self, X):
        return self._sigmoid(np.hstack([np.ones((X.shape[0], 1)), X]) @ self.w)

    def predict(self, X, tau=0.5):
        return (self.predict_proba(X) >= tau).astype(int)

## Example
X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
y = np.array([0, 0, 1, 1])
print(LogisticRegression().fit(X, y).predict(np.array([[1.5]])))  ## [1]
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ✅Interpretable ← $e^{w_j}$ = odds ratio per unit increase in feature $j$.
- ✅Well-calibrated probabilities when the model is correctly specified (unlike SVM/NB scores).
- ✅Convex → global optimum, no restarts needed. ⬇️Compute. Extends to multiclass.

*Cons?*
- Linear decision boundary only → underfits (can't learn XOR without feature crosses).
- ⬆️Sensitivity to outliers & to perfectly separable data.
- Needs feature scaling for fast GD convergence & for fair regularization.

*Why not MSE as the loss?*
- MSE with a sigmoid is **non-convex** in $\mathbf{w}$ → local minima.
- Gradient carries a $\sigma'(z)=p(1-p)$ factor → vanishes when the model is confidently WRONG → glacial learning.
- BCE cancels that factor exactly (see Derivation) → gradient $\propto$ error.

*Why does it break on perfectly separable data?*
- Scaling $\mathbf{w}\to c\mathbf{w}$ with $c\to\infty$ pushes every $p_i$ to 0/1 → NLL $\to0$ but never attains it.
- → MLE does not exist, weights diverge. Fix: L2 regularization (always add some).

*Is it "regression" or "classification"?*
- Classification. It **regresses the log-odds**, which is where the name comes from.

*Threshold ≠ 0.5?*
- Imbalanced classes or asymmetric costs → tune $\tau$ on a validation set (or via the PR curve).
- Changing $\tau$ does NOT change ranking → AUC is unaffected.

*Interpreting coefficients?*
- $w_j$ = change in log-odds per unit of $x_j$. $e^{w_j}$ = odds multiplier.
- ⚠️ NOT a change in probability — that depends on where you sit on the sigmoid.

*Why is the Hessian PSD?*
- $\mathbf{v}^TX^TSX\mathbf{v}=\sum_i p_i(1-p_i)(\mathbf{x}_i^T\mathbf{v})^2\geq0$ ← $p_i(1-p_i)>0$.
```

&nbsp;

#### Softmax Regression
- **What**: Softmax over $K$ weighted sums of features.
- **Why**: Multiclass.
    - One-vs-Rest fits $K$ independent models → scores not comparable, need ad-hoc renormalization.
    - Softmax models the joint distribution → properly normalized, jointly trained.
- **How**: One weight vector per class → softmax over the $K$ logits → cross-entropy loss.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}_i\in\mathbb{R}^n$: Input.
    - $y_i\in\{1,\cdots,K\}$: Label.
- Params:
    - $W=[\mathbf{w}_1,\cdots,\mathbf{w}_K]\in\mathbb{R}^{n\times K}$: Param matrix.
- Misc:
    - $p_{ik}$: Predicted probability of class $k$ for sample $i$.
    - $y_{ik}=\mathbb{1}[y_i=k]$: One-hot label.

Model:

$$
p_{ik}=\frac{e^{\mathbf{x}_i^T\mathbf{w}_k}}{\sum_{l=1}^{K}e^{\mathbf{x}_i^T\mathbf{w}_l}}
$$

Objective (CE):

$$
L(W)=-\sum_{i=1}^m\sum_{k=1}^{K}y_{ik}\log p_{ik}
$$

Gradient:

$$
\frac{\partial L}{\partial\mathbf{w}_k}=\sum_{i=1}^m(p_{ik}-y_{ik})\mathbf{x}_i
$$
```

```{attention} Q&A
:class: dropdown
*Why is softmax over-parameterized?*
- Adding a constant $\mathbf{c}$ to EVERY $\mathbf{w}_k$ leaves all $p_{ik}$ unchanged → solution not unique.
- Fix: pin $\mathbf{w}_K=\mathbf{0}$ ($K-1$ free vectors, matches binary logistic at $K=2$), or add L2 (which makes the minimizer unique).

*Softmax vs One-vs-Rest?*
- Softmax: mutually exclusive classes, probabilities sum to 1, one joint fit.
- OvR: $K$ independent binary fits → multi-LABEL capable, trivially parallel, but scores need ad-hoc normalization & each fit sees an imbalanced problem.

*Softmax vs $K$ sigmoids?*
- Multi-class (exactly one label) → softmax. Multi-label (any number of labels) → independent sigmoids.

*Numerical stability?*
- Subtract $\max_l\mathbf{x}_i^T\mathbf{w}_l$ from all logits before exponentiating → identical output, no overflow.
```

&nbsp;

### GLM
- **Name**: Generalized Linear Model
- **What**: Linear predictor + a **link function** to the mean of an exponential-family response.
- **Why**: OLS assumes a Gaussian, unbounded, homoskedastic response.
    - False for counts (non-negative integers), binary outcomes ($[0,1]$), durations (positive, skewed).
    - → Swap in a response distribution & link that respect the target's actual range & variance.
- **How**: Pick 3 components → fit by IRLS.
    1. **Random**: response distribution from the exponential family.
    2. **Systematic**: linear predictor $\eta=\mathbf{x}^T\mathbf{w}$.
    3. **Link**: $g(\mu)=\eta$, where $\mu=\mathbb{E}[y|\mathbf{x}]$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mu=\mathbb{E}[y|\mathbf{x}]$: Conditional mean.
    - $\eta=\mathbf{x}^T\mathbf{w}$: Linear predictor.
    - $g$: Link function.
    - $g^{-1}$: Mean/response function.

Exponential family (canonical form):

$$
P(y|\theta,\phi)=\exp\left(\frac{y\theta-b(\theta)}{a(\phi)}+c(y,\phi)\right)
$$
- $\theta$: Natural param.
- $\phi$: Dispersion param.
- $\mu=b'(\theta)$, $\text{Var}[y]=a(\phi)b''(\theta)$.

Model:

$$
g(\mu)=\mathbf{x}^T\mathbf{w}
$$

**Canonical link**: $g$ such that $\theta=\eta$ → gradient collapses to $\frac{1}{a(\phi)}X^T(\boldsymbol{\mu}-\mathbf{y})$ for every member of the family.
```

```{dropdown} Table: Common GLMs
| Response | Distribution | Canonical link | $g^{-1}(\eta)$ | Model name |
|:--|:--|:--|:--|:--|
| Continuous, unbounded | Gaussian | Identity: $\mu$ | $\eta$ | Linear regression |
| Binary | Bernoulli | Logit: $\log\frac{\mu}{1-\mu}$ | $\frac{1}{1+e^{-\eta}}$ | Logistic regression |
| Categorical | Multinomial | Logit (generalized) | Softmax | Softmax regression |
| Count | Poisson | Log: $\log\mu$ | $e^\eta$ | Poisson regression |
| Positive, skewed | Gamma | Inverse: $\frac{1}{\mu}$ | $\frac{1}{\eta}$ | Gamma regression |
```

```{attention} Q&A
:class: dropdown
*Why the canonical link specifically?*
- Makes $\theta=\eta$ → log-likelihood concave in $\mathbf{w}$ → any stationary point is a global optimum.
    - ⚠️ Concave ≠ unique. Uniqueness also needs $\text{rank}(X)=n$, and a finite MLE needs no separation (see Logistic Regression).
- Makes the observed Hessian = expected Hessian → Newton = Fisher scoring = IRLS.
- Yields a sufficient statistic $X^T\mathbf{y}$.
- Not mandatory — probit and complementary log-log are non-canonical links for Bernoulli and are used routinely.

*Why is a link needed at all instead of just transforming $y$?*
- $g(\mathbb{E}[y])\neq\mathbb{E}[g(y)]$ (Jensen) → transforming $y$ models the wrong quantity, and blows up on $y=0$ for a log transform.
- GLM transforms the **mean**, not the data.

*Overdispersion?*
- Poisson forces $\text{Var}[y]=\mu$. Real counts usually have $\text{Var}[y]>\mu$ → underestimated standard errors.
- Fix: quasi-Poisson (free dispersion) or Negative Binomial.
```

&nbsp;

### LDA
- **Name**: Linear Discriminant Analysis
- **What**: Gaussian per class + one **shared** covariance → linear boundary.
- **Why**: Statistical efficiency on small $m$.
    - Logistic regression fits $P(y|\mathbf{x})$ only → wastes data when the Gaussian assumption roughly holds.
    - Stable when classes are well separated (where logistic regression's weights diverge).
    - Doubles as a **supervised** dimensionality reducer → ≤ $K-1$ discriminant directions.
- **How**:
    1. Estimate per-class mean $\boldsymbol{\mu}_k$, prior $\pi_k$, and ONE pooled covariance $\Sigma$.
    2. Apply Bayes' rule → the quadratic terms cancel (shared $\Sigma$) → linear discriminant score per class.
    3. Predict $\arg\max_k$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}\in\mathbb{R}^n$: Input.
    - $y\in\{1,\cdots,K\}$: Label.
- Params:
    - $\boldsymbol{\mu}_k\in\mathbb{R}^n$: Class-$k$ mean.
    - $\Sigma\in\mathbb{R}^{n\times n}$: Pooled covariance.
    - $\pi_k$: Class prior.
- Misc:
    - $m_k$: #samples in class $k$.
    - $S_W,S_B$: Within-/Between-class scatter matrices.

Model (class-conditional):

$$
P(\mathbf{x}|y=k)=N(\mathbf{x}|\boldsymbol{\mu}_k,\Sigma)
$$

Estimation:

$$
\hat{\pi}_k=\frac{m_k}{m},\quad\hat{\boldsymbol{\mu}}_k=\frac{1}{m_k}\sum_{i:y_i=k}\mathbf{x}_i,\quad\hat{\Sigma}=\frac{1}{m-K}\sum_{k=1}^K\sum_{i:y_i=k}(\mathbf{x}_i-\hat{\boldsymbol{\mu}}_k)(\mathbf{x}_i-\hat{\boldsymbol{\mu}}_k)^T
$$

Inference (linear in $\mathbf{x}$):

$$
\hat{y}=\arg\max_k\ \delta_k(\mathbf{x}),\qquad\delta_k(\mathbf{x})=\mathbf{x}^T\Sigma^{-1}\boldsymbol{\mu}_k-\frac{1}{2}\boldsymbol{\mu}_k^T\Sigma^{-1}\boldsymbol{\mu}_k+\log\pi_k
$$

Fisher's criterion (dimensionality reduction view):

$$
\max_\mathbf{w}\frac{\mathbf{w}^TS_B\mathbf{w}}{\mathbf{w}^TS_W\mathbf{w}}\ \Rightarrow\ S_W^{-1}S_B\mathbf{w}=\lambda\mathbf{w}
$$
- $S_W=\sum_k\sum_{i:y_i=k}(\mathbf{x}_i-\boldsymbol{\mu}_k)(\mathbf{x}_i-\boldsymbol{\mu}_k)^T$: Within-class scatter.
- $S_B=\sum_k m_k(\boldsymbol{\mu}_k-\bar{\boldsymbol{\mu}})(\boldsymbol{\mu}_k-\bar{\boldsymbol{\mu}})^T$: Between-class scatter, $\bar{\boldsymbol{\mu}}$ = global mean.
- $\text{rank}(S_B)\leq K-1$ → at most $K-1$ useful directions.
- Binary case: $\mathbf{w}\propto S_W^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)$.
```

```{tip} Derivation
:class: dropdown
*Why is the boundary linear?*
1. Bayes: $P(y=k|\mathbf{x})\propto\pi_kN(\mathbf{x}|\boldsymbol{\mu}_k,\Sigma)$.
2. Take logs:

    $$
    \log\pi_k-\frac{1}{2}\log|\Sigma|-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_k)^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}_k)
    $$

3. Expand the quadratic form:

    $$
    -\frac{1}{2}\mathbf{x}^T\Sigma^{-1}\mathbf{x}+\mathbf{x}^T\Sigma^{-1}\boldsymbol{\mu}_k-\frac{1}{2}\boldsymbol{\mu}_k^T\Sigma^{-1}\boldsymbol{\mu}_k
    $$

4. $-\frac{1}{2}\mathbf{x}^T\Sigma^{-1}\mathbf{x}$ and $\log|\Sigma|$ are **identical across $k$** ← shared $\Sigma$ → drop them.
5. → $\delta_k(\mathbf{x})$ is affine in $\mathbf{x}$ → boundaries $\delta_k=\delta_l$ are hyperplanes.
6. Per-class $\Sigma_k$ → the quadratic term survives → **QDA**.
```

```{attention} Q&A
:class: dropdown
*Assumptions?*
- Class-conditional Gaussian.
- **Equal covariance** across classes (this is the whole reason the boundary is linear).
- Full-rank $\Sigma$: the pooled scatter has rank $\leq m-K$ → needs $n\leq m-K$; else use regularized/shrinkage LDA.

*LDA vs PCA?*
- LDA: **supervised**, maximizes class separability, ≤ $K-1$ components.
- PCA: **unsupervised**, maximizes total variance, ≤ $n$ components.
- The top-variance direction is often NOT the most discriminative one.

*LDA vs Logistic Regression?*
- Both give linear boundaries; they differ in **how they fit**.
- LDA = generative (models $P(\mathbf{x},y)$), fit by moment estimates → ⬆️efficiency IF Gaussian holds, ⬆️stability on separable data & small $m$.
- LogReg = discriminative (models $P(y|\mathbf{x})$ only), fit by MLE → ⬆️robustness to non-Gaussian features & outliers. Usually the safer default.

*LDA vs QDA — which one?*
- LDA: $n(n+1)/2$ covariance params total. QDA: $K\cdot n(n+1)/2$.
- Small $m$ / large $n$ → LDA (⬇️variance). Large $m$ + genuinely different class shapes → QDA (⬇️bias).
- Regularized discriminant analysis interpolates: $\hat{\Sigma}_k(\alpha)=\alpha\hat{\Sigma}_k+(1-\alpha)\hat{\Sigma}$.

*Name clash warning*: LDA also = **Latent Dirichlet Allocation**, an unrelated unsupervised topic model. Context disambiguates.
```

&nbsp;

## Kernel Methods
- **What**: Inner products replaced by a kernel → operate in a high-dim (even infinite-dim) feature space without ever computing coordinates in it.

### SVM
- **Name**: Support Vector Machine
- **What**: Hyperplane maximizing the margin to the nearest points of each class.
- **Why**:
    - *Why do we need it?*
        - Infinitely many separating hyperplanes exist → which generalizes best?
        - The perceptron returns whichever one it stumbles into → poor margin → fragile.
    - *Why does it work?*
        - Generalization depends on the **margin**, not feature-space dimension → high-dim kernels don't automatically overfit.
        - Solution depends ONLY on the few **support vectors** → sparse, memory-light.
        - Convex QP → every local optimum is global ($\mathbf{w}$ unique; $b$ and $\boldsymbol{\alpha}$ need not be).
- **How**:
    1. **Hard margin**: maximize $\frac{2}{||\mathbf{w}||}$ s.t. every point is correctly classified with margin $\geq1$.
    2. **Soft margin**: add slack $\xi_i$ + penalty $C$ → tolerate violations for non-separable data.
    3. **Dual**: Lagrangian → the problem only touches data via inner products $\mathbf{x}_i^T\mathbf{x}_j$.
    4. **Kernel**: swap $\mathbf{x}_i^T\mathbf{x}_j\to K(\mathbf{x}_i,\mathbf{x}_j)$ → nonlinear boundary, same solver.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}_i\in\mathbb{R}^n$: Input.
    - $y_i\in\{-1,+1\}$: Label (NOT $\{0,1\}$).
- Params:
    - $\mathbf{w}\in\mathbb{R}^n,b\in\mathbb{R}$: Hyperplane.
    - $\alpha_i\geq0$: Dual variables.
- Hyperparams:
    - $C>0$: Slack penalty (inverse regularization).
    - $K(\cdot,\cdot)$: Kernel.
- Misc:
    - $\xi_i\geq0$: Slack for sample $i$.

Primal (soft margin):

$$
\min_{\mathbf{w},b,\boldsymbol{\xi}}\ \frac{1}{2}||\mathbf{w}||_2^2+C\sum_{i=1}^m\xi_i\quad\text{s.t.}\quad y_i(\mathbf{w}^T\mathbf{x}_i+b)\geq1-\xi_i,\ \ \xi_i\geq0
$$

Equivalent unconstrained (hinge loss + L2):

$$
\min_{\mathbf{w},b}\ \frac{1}{2}||\mathbf{w}||_2^2+C\sum_{i=1}^m\max\left(0,1-y_i(\mathbf{w}^T\mathbf{x}_i+b)\right)
$$

Dual:

$$
\max_{\boldsymbol{\alpha}}\ \sum_{i=1}^m\alpha_i-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_jy_iy_jK(\mathbf{x}_i,\mathbf{x}_j)\quad\text{s.t.}\quad0\leq\alpha_i\leq C,\ \sum_{i=1}^m\alpha_iy_i=0
$$

Decision function:

$$
f(\mathbf{x})=\sum_{i=1}^m\alpha_iy_iK(\mathbf{x}_i,\mathbf{x})+b,\qquad\hat{y}=\text{sign}(f(\mathbf{x}))
$$
- $\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i$ (linear kernel only).
- Margin width $=\frac{2}{||\mathbf{w}||_2}$.

KKT conditions → who is a support vector:

$$
\begin{cases}
\alpha_i=0 & y_if(\mathbf{x}_i)\geq1 \quad\text{(outside or on the margin, ignored)} \\
0<\alpha_i<C & y_if(\mathbf{x}_i)=1 \quad\text{(on the margin, "free" SV)} \\
\alpha_i=C & y_if(\mathbf{x}_i)\leq1 \quad\text{(inside margin or misclassified, "bounded" SV)}
\end{cases}
$$
```

```{tip} Derivation
:class: dropdown
*Why is the margin $\frac{2}{||\mathbf{w}||}$?*
1. Distance from $\mathbf{x}$ to the hyperplane $\mathbf{w}^T\mathbf{x}+b=0$ is $\frac{|\mathbf{w}^T\mathbf{x}+b|}{||\mathbf{w}||}$.
2. $(\mathbf{w},b)$ is scale-free → fix the scale by requiring $\min_i|\mathbf{w}^T\mathbf{x}_i+b|=1$ (canonical form).
3. → Nearest point of each class sits at distance $\frac{1}{||\mathbf{w}||}$ → total margin $\frac{2}{||\mathbf{w}||}$.
4. Maximize $\frac{2}{||\mathbf{w}||}$ $\Leftrightarrow$ minimize $\frac{1}{2}||\mathbf{w}||_2^2$ (smooth, convex QP).

*Where does the dual come from?*
1. Lagrangian with multipliers $\alpha_i\geq0$ (margin) and $\mu_i\geq0$ (slack):

    $$
    \mathcal{L}=\frac{1}{2}||\mathbf{w}||_2^2+C\sum_i\xi_i-\sum_i\alpha_i\left[y_i(\mathbf{w}^T\mathbf{x}_i+b)-1+\xi_i\right]-\sum_i\mu_i\xi_i
    $$

2. Stationarity:

    $$
    \frac{\partial\mathcal{L}}{\partial\mathbf{w}}=0\Rightarrow\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i,\quad\frac{\partial\mathcal{L}}{\partial b}=0\Rightarrow\sum_i\alpha_iy_i=0,\quad\frac{\partial\mathcal{L}}{\partial\xi_i}=0\Rightarrow\alpha_i+\mu_i=C
    $$

3. $\mu_i\geq0$ + $\alpha_i+\mu_i=C$ → **box constraint** $0\leq\alpha_i\leq C$.
4. Substitute back → all $\xi_i$, $\mathbf{w}$, $b$ vanish; $\mathbf{x}$ appears ONLY as $\mathbf{x}_i^T\mathbf{x}_j$ → kernelizable.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class LinearSVM:
    """Soft-margin SVM via subgradient descent on hinge loss + L2."""

    def __init__(self, C=1.0, lr=1e-3, steps=2000):
        self.C, self.lr, self.steps = C, lr, steps
        self.w, self.b = None, 0.0

    def fit(self, X, y):
        ## y must be in {-1, +1}, NOT {0, 1}
        y = np.where(y <= 0, -1.0, 1.0)
        m, n = X.shape
        self.w, self.b = np.zeros(n), 0.0
        for _ in range(self.steps):
            margin = y * (X @ self.w + self.b)
            viol = margin < 1                      ## only margin violators contribute
            ## d/dw [0.5||w||^2 + C*sum(max(0, 1 - y f(x)))]
            dw = self.w - self.C * (y[viol] @ X[viol])
            db = -self.C * y[viol].sum()
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

## Example
X = np.array([[-2.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 1.0]])
y = np.array([-1, -1, 1, 1])
print(LinearSVM().fit(X, y).predict(np.array([[1.5, 1.5]])))  ## [1.]
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ✅Effective in high dims, even $n>m$ ← margin-based, not dimension-based.
- ✅Sparse solution ← only SVs stored.
- ✅Convex → every local optimum is global, reproducible.
- ✅Nonlinear via kernels w/o explicit feature construction.

*Cons?*
- ❌Scales: training is $O(m^2)$–$O(m^3)$, kernel matrix is $O(m^2)$ memory → dead above ~$10^5$ samples.
- ❌Native probabilities → needs Platt scaling (an extra logistic fit on held-out scores).
- ⬆️Sensitivity to feature scaling (RBF distances are scale-dominated).
- ⬇️Interpretability with nonlinear kernels.
- $C$ & $\gamma$ need joint tuning → expensive grid/random search.

*What does $C$ do?*
- $C$⬆️ → slack expensive → narrow margin, fits training data hard → Overfit. ($C\to\infty$ → hard margin, IF the data is separable.)
- $C$⬇️ → slack cheap → wide margin, more violations tolerated → Underfit.
- $C$ is the INVERSE of regularization strength ($C\approx\frac{1}{2\lambda}$ vs. an L2-penalized hinge loss).

*vs. logistic regression on separable data?*
- Both diverge in norm, but unregularized LogReg trained by GD converges **in direction** to the max-margin separator — it does not settle on an arbitrary one.
- SVM gets there directly, in finite time, with a sparse solution and an explicit $C$ knob.

*What does $\gamma$ do (RBF)?*
- $\gamma$⬆️ → narrow bumps → each SV influences only its immediate neighborhood → wiggly boundary → Overfit.
- $\gamma$⬇️ → wide bumps → boundary approaches linear → Underfit.

*Why solve the dual instead of the primal?*
- The dual is the ONLY form where the kernel trick applies (data enters only via inner products).
- Dual has $m$ variables vs primal's $n$ → win when $n\gg m$.
- Modern linear SVMs (LIBLINEAR) solve the **primal** — faster when $m\gg n$ and no kernel is needed.

*What exactly is a support vector?*
- Any $\mathbf{x}_i$ with $\alpha_i>0$ → sits ON or INSIDE the margin.
- Deleting a non-SV and retraining gives the identical model.
- #SVs ⬆️ → memory & inference cost ⬆️, and it's a rough overfitting signal.

*Hinge loss vs log loss?*
- Hinge: exactly 0 once $y f(\mathbf{x})\geq1$ → sparsity; not differentiable at the kink; no probabilities.
- Log: never exactly 0 → every point contributes → dense; smooth; calibrated probabilities.

*Multiclass?*
- Not native. OvO ($\frac{K(K-1)}{2}$ models, libsvm's default) or OvR ($K$ models).

*Does the RBF kernel overfit because its feature space is infinite-dim?*
- No — capacity is controlled by the margin (and $C$, $\gamma$), not by feature-space dimension. That's the whole point of margin theory.
```

&nbsp;

#### Kernel Trick
- **What**: Inner products in a feature space computed directly via $K(\mathbf{x},\mathbf{z})=\phi(\mathbf{x})^T\phi(\mathbf{z})$, without ever evaluating $\phi$.
- **Why**: $\phi$ is intractable or infinite.
    - Degree-$d$ polynomial expansion over $n$ features costs $O(\binom{n+d}{d})$ dims; RBF's $\phi$ is infinite-dim.
    - → Any algorithm expressible purely in inner products becomes nonlinear for free.
- **How**: Rewrite the algorithm so data appears only as $\mathbf{x}_i^T\mathbf{x}_j$ → substitute $K(\mathbf{x}_i,\mathbf{x}_j)$.

```{note} Math
:class: dropdown
Mercer's condition: $K$ is a valid kernel $\Leftrightarrow$ $K$ is symmetric and the Gram matrix $[K(\mathbf{x}_i,\mathbf{x}_j)]_{ij}$ is PSD for every finite sample.

Example (degree-2 polynomial, $n=2$, $K(\mathbf{x},\mathbf{z})=(\mathbf{x}^T\mathbf{z})^2$):

$$
\phi(\mathbf{x})=[x_1^2,\ \sqrt{2}x_1x_2,\ x_2^2]^T\ \Rightarrow\ \phi(\mathbf{x})^T\phi(\mathbf{z})=(x_1z_1+x_2z_2)^2=(\mathbf{x}^T\mathbf{z})^2
$$
- Cost: $O(n)$ via $K$ vs $O(n^2)$ via $\phi$.

Closure properties: if $K_1,K_2$ are kernels, so are $K_1+K_2$, $cK_1$ ($c>0$), $K_1K_2$, $f(\mathbf{x})K_1f(\mathbf{z})$, and $\exp(K_1)$.
```

```{dropdown} Table: Common Kernels
| Kernel | $K(\mathbf{x},\mathbf{z})$ | Hyperparams | Feature space | Use |
|:--|:--|:--|:--|:--|
| Linear | $\mathbf{x}^T\mathbf{z}$ | — | $n$-dim | $n\gg m$, text, sparse features |
| Polynomial | $(\gamma\mathbf{x}^T\mathbf{z}+r)^d$ | $\gamma,r,d$ | $O(\binom{n+d}{d})$ | Explicit feature interactions |
| RBF / Gaussian | $\exp(-\gamma\lVert\mathbf{x}-\mathbf{z}\rVert_2^2)$ | $\gamma$ | Infinite | Default when unsure |
| Sigmoid | $\tanh(\gamma\mathbf{x}^T\mathbf{z}+r)$ | $\gamma,r$ | — | Rarely; NOT PSD for all params |

- $\gamma=\frac{1}{2\sigma^2}$ for the RBF; sklearn's `gamma='scale'` sets $\gamma=\frac{1}{n\cdot\text{Var}[X]}$.
```

```{attention} Q&A
:class: dropdown
*Which kernel first?*
- $n\gg m$ (text, genomics) → linear; data is already nearly separable in high dims.
- $m\gg n$ → RBF, then tune $(C,\gamma)$.
- RBF with tiny $\gamma$ ≈ linear → linear is a special case worth trying first (much faster).

*Cost?*
- Gram matrix: $O(m^2)$ memory, $O(m^2n)$ to build → the real reason kernel methods die on large $m$.
- Workaround: Nyström approximation / Random Fourier Features → explicit low-dim $\hat{\phi}$, then train a LINEAR model.

*Why does the kernel matrix have to be PSD?*
- PSD $\Leftrightarrow$ $K$ is an inner product in some feature space (an RKHS) → the QP is convex.
- Non-PSD → no such feature space, the dual is non-convex, and solvers may not converge.
- ⚠️ A kernel is a similarity, NOT a distance; the metric it induces, $\sqrt{K(\mathbf{x},\mathbf{x})+K(\mathbf{z},\mathbf{z})-2K(\mathbf{x},\mathbf{z})}$, is only a pseudometric in general.
```

&nbsp;

#### SVR
- **Name**: Support Vector Regression
- **What**: Regression ignoring errors inside an $\epsilon$-tube, penalizing only what falls outside.
- **Why**: Sparsity + noise robustness.
    - Squared error lets every point, noise included, pull the fit.
    - → $\epsilon$-insensitive loss makes only outside-tube points SVs.
- **How**: Fit the flattest function whose tube of width $2\epsilon$ contains as much data as possible; slack for the rest.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\epsilon\geq0$: Tube half-width.
    - $C$: Slack penalty.

Objective:

$$
\min_{\mathbf{w},b,\boldsymbol{\xi},\boldsymbol{\xi}^*}\frac{1}{2}||\mathbf{w}||_2^2+C\sum_{i=1}^m(\xi_i+\xi_i^*)\quad\text{s.t.}\quad\begin{cases}y_i-\mathbf{w}^T\mathbf{x}_i-b\leq\epsilon+\xi_i \\ \mathbf{w}^T\mathbf{x}_i+b-y_i\leq\epsilon+\xi_i^* \\ \xi_i,\xi_i^*\geq0\end{cases}
$$

$\epsilon$-insensitive loss:

$$
L_\epsilon(y,f(\mathbf{x}))=\max(0,|y-f(\mathbf{x})|-\epsilon)
$$
```

```{attention} Q&A
:class: dropdown
*$\epsilon$ vs $C$?*
- $\epsilon$⬆️ → wider tube → fewer SVs → flatter, sparser, more bias.
- $C$⬆️ → outside-tube errors expensive → tighter fit → overfit risk.

*$\nu$-SVR?*
- Reparametrizes $\epsilon$ as $\nu\in(0,1]$, which upper-bounds the fraction of training errors and lower-bounds the fraction of SVs → easier to set than a raw $\epsilon$ in target units.

*Why not just use kernel ridge regression?*
- Same hypothesis space; KRR uses squared loss → dense (ALL points are "SVs") and has a closed form; SVR uses $\epsilon$-insensitive loss → sparse but needs a QP solver.
```

&nbsp;

## Tree-Based Methods
- **What**: Feature space recursively partitioned by axis-aligned splits.

### Decision Tree
- **What**: Recursive splits on one feature at a time → constant prediction per region.
- **Why**: Nonlinearity + interactions for free.
    - ❌Feature engineering, ❌scaling, ❌distribution assumptions.
    - ✅Mixed numerical/categorical types, ✅missing values.
    - Fully interpretable → a root-to-leaf path IS the explanation.
- **How**:
    1. At each node, search every (feature, threshold) pair → pick the one with the largest impurity decrease.
    2. Split → recurse on both children.
    3. Stop at a stopping rule (max depth, min samples, zero impurity gain).
    4. Prune back (cost-complexity) to cut variance.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $p_k$: Fraction of class $k$ at a node.
    - $D$: Samples at the parent.
    - $D_L,D_R$: Samples at the children.
    - $|T|$: #leaves of tree $T$.

Impurity (classification):

$$
\text{Gini}(D)=1-\sum_{k=1}^{K}p_k^2,\qquad H(D)=-\sum_{k=1}^{K}p_k\log_2p_k
$$

Impurity (regression):

$$
\text{MSE}(D)=\frac{1}{|D|}\sum_{i\in D}(y_i-\bar{y}_D)^2
$$
- $\bar{y}_D$: Mean target in $D$ = the leaf's prediction.

Split criterion (maximize):

$$
\Delta=I(D)-\frac{|D_L|}{|D|}I(D_L)-\frac{|D_R|}{|D|}I(D_R)
$$
- $I$: Any impurity above. With $I=H$, $\Delta$ = **Information Gain**.

Gain Ratio (C4.5, corrects IG's bias toward many-valued features):

$$
\text{GainRatio}=\frac{\text{IG}}{\text{SplitInfo}},\qquad\text{SplitInfo}=-\sum_{v}\frac{|D_v|}{|D|}\log_2\frac{|D_v|}{|D|}
$$
- $D_v$: Samples taking the $v$-th value of the split feature.

Cost-complexity pruning (CART):

$$
R_\alpha(T)=R(T)+\alpha|T|
$$
- $R(T)$: Training error/impurity of $T$.
- $\alpha\geq0$: Complexity penalty, chosen by CV.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth, self.min_samples_split = max_depth, min_samples_split
        self.tree = None

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        p = np.bincount(y) / len(y)
        return 1.0 - (p ** 2).sum()

    def _best_split(self, X, y):
        best = (None, None, 0.0)                   ## (feature, threshold, gain)
        parent = self._gini(y)
        for j in range(X.shape[1]):
            ## midpoints between consecutive unique values are the only useful thresholds
            vals = np.unique(X[:, j])
            for t in (vals[:-1] + vals[1:]) / 2:
                left = X[:, j] <= t
                if left.sum() == 0 or (~left).sum() == 0:
                    continue
                wl = left.mean()
                gain = parent - wl * self._gini(y[left]) - (1 - wl) * self._gini(y[~left])
                if gain > best[2]:
                    best = (j, t, gain)
        return best

    def _build(self, X, y, depth):
        ## leaf: majority class, stored as a plain int
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
            return int(np.bincount(y).argmax())
        j, t, gain = self._best_split(X, y)
        if j is None or gain <= 0:
            return int(np.bincount(y).argmax())
        left = X[:, j] <= t
        return (j, t, self._build(X[left], y[left], depth + 1),
                      self._build(X[~left], y[~left], depth + 1))

    def fit(self, X, y):
        self.tree = self._build(X, np.asarray(y), 0)
        return self

    def _walk(self, node, x):
        if not isinstance(node, tuple):
            return node
        j, t, l, r = node
        return self._walk(l if x[j] <= t else r, x)

    def predict(self, X):
        return np.array([self._walk(self.tree, x) for x in X])

## Example: needs depth 2 -- the second split only pays off inside the right child
X = np.array([[1.0, 1.0], [1.0, 5.0], [5.0, 1.0], [5.0, 5.0]])
y = np.array([0, 0, 1, 0])
print(DecisionTreeClassifier(max_depth=2).fit(X, y).predict(X))  ## [0 0 1 0]
```
````

```{dropdown} Table: Decision Tree Algorithms
| | ID3 | C4.5 | CART |
|:--|:--|:--|:--|
| Split criterion | Information Gain | Gain Ratio | Gini (clf) / MSE (reg) |
| Split arity | Multi-way | Multi-way | **Binary only** |
| Numerical features | ❌ | ✅ | ✅ |
| Regression | ❌ | ❌ | ✅ |
| Missing values | ❌ | ✅ (fractional instances) | ✅ (surrogate splits) |
| Pruning | ❌ | Error-based | Cost-complexity |

sklearn implements an optimized CART (numerical features only, no native categorical support).
```

```{attention} Q&A
:class: dropdown
*Pros?*
- ✅Interpretable, ✅Fast inference ($O(\text{depth})$).
- ❌Scaling, ❌Normalization, ❌Distribution assumptions.
- ✅Mixed types, ✅Nonlinearity, ✅Interactions, ✅Multiclass natively.

*Cons?*
- ⬆️⬆️Variance → tiny data change → completely different tree. (This is why ensembles exist.)
- Overfits without depth/leaf constraints or pruning.
- Axis-aligned splits only → needs a staircase to model a diagonal boundary.
- Regression can't extrapolate — predictions are constants, bounded by the training target range.
- Greedy → no global optimum.
- Biased toward features with many distinct values / high cardinality.
- Unstable & biased on imbalanced classes.

*Gini vs Entropy?*
- Nearly identical trees in practice; they rarely disagree on the chosen split.
- Gini is cheaper ← no logarithm. sklearn's default.
- Both peak at a uniform class distribution and are 0 at purity.
- Gini range: $[0,1-\frac{1}{K}]$. Entropy range: $[0,\log_2K]$.

*Why greedy instead of the optimal tree?*
- Finding the globally optimal tree is NP-complete → greedy top-down (recursive binary splitting) is the practical compromise.

*Why does Information Gain favor high-cardinality features?*
- A feature with a unique value per sample (e.g., an ID) splits into perfectly pure singleton leaves → maximal IG, zero generalization.
- Mitigations: Gain Ratio (normalize by SplitInfo); binary splits avoid unrestricted multiway singleton splits but do NOT eliminate the bias ← many candidate thresholds still give more chances to look good.
- Real fix: permutation-based / unbiased split selection (conditional inference trees).

*Can a greedy tree learn XOR?*
- Not in one split. On balanced XOR, EVERY single split has impurity gain **exactly 0** → a greedy learner stops at the root.
- Depth-2 trees represent XOR fine — greedy search just can't find it, because the payoff only appears one level down (the horizon effect).
- Workarounds: lookahead search, random splits (Extra Trees), or an ensemble.

*Pre-pruning vs post-pruning?*
- Pre (early stopping: `max_depth`, `min_samples_leaf`, `min_impurity_decrease`) → cheap, but can stop before a good split that only pays off one level down (the "horizon effect").
- Post (grow full, then cost-complexity prune with $\alpha$ by CV) → better trees, more compute.

*Is feature importance trustworthy?*
- MDI (mean impurity decrease, sklearn's `feature_importances_`) is **biased** toward high-cardinality & continuous features, and is computed on TRAINING data.
- Permutation importance on held-out data is the safer alternative; both are misleading with correlated features.

*Handling missing values?*
- CART: surrogate splits (a backup feature that mimics the primary split).
- C4.5: send the instance down all branches with fractional weights.
- XGBoost/LightGBM: learn a default direction per node from the data.
```

&nbsp;

### Bagging
- **What**: Same learner trained on many bootstrap resamples → predictions averaged.
- **Why**: Variance⬇️.
    - High-variance/low-bias learners (deep trees) overfit the specific training sample.
    - Averaging $B$ i.i.d. estimators divides variance by $B$ w/o touching bias.
    - Bootstrap manufactures the "many datasets" you don't have.
- **How**:
    1. Draw $B$ bootstrap samples (size $m$, **with replacement**).
    2. Train one unpruned learner per sample.
    3. Aggregate: average (regression) / majority vote (classification).

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $B$: #base learners.
- Misc:
    - $\sigma^2$: Variance of a single base learner.
    - $\rho$: Pairwise correlation between learners.

Prediction:

$$
\hat{f}_\text{bag}(\mathbf{x})=\frac{1}{B}\sum_{b=1}^{B}\hat{f}_b(\mathbf{x})
$$

Variance of the average:

$$
\text{Var}[\hat{f}_\text{bag}]=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2
$$
- $B\to\infty$ → variance floor $\rho\sigma^2$ → **decorrelation ($\rho$⬇️), not more trees, is the real lever**. This is exactly what Random Forest attacks.

OOB (Out-of-Bag) fraction: probability a given sample is NEVER drawn in one bootstrap:

$$
\left(1-\frac{1}{m}\right)^m\xrightarrow{m\to\infty}\frac{1}{e}\approx0.368
$$
- → ~36.8% of samples are OOB for each learner → free validation set, no CV needed.
```

```{attention} Q&A
:class: dropdown
*Why does bagging need HIGH-variance base learners?*
- Bagging reduces variance, not bias. Bagging a linear model (already low-variance, high-bias) buys almost nothing.
- → Use **deep, unpruned** trees as base learners.

*Does bagging ever hurt?*
- Yes on stable, low-variance learners: each bootstrap sees only ~63.2% unique samples → slightly worse base fits with no variance payoff.

*Bagging vs Boosting?*
- Bagging: parallel, i.i.d. resamples, targets **variance**, base learners are strong/deep.
- Boosting: sequential, reweighted/residual-fitted data, targets **bias**, base learners are weak/shallow.

*Pasting?*
- Same idea but sampling WITHOUT replacement → more diversity per subsample, no OOB estimate.
```

&nbsp;

#### Random Forest
- **What**: Bagged decision trees + a random feature subset per split.
- **Why**: Decorrelation.
    - Plain bagged trees stay highly correlated ← a dominant feature is split on first by every tree.
    - → Variance floor $\rho\sigma^2$ barely drops. Hiding most features breaks that correlation.
- **How**: Bagging + at every node, sample $m_\text{try}$ of $n$ features and split only among those.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $m_\text{try}$: #features sampled per split.

Common defaults:

$$
m_\text{try}=\begin{cases}\lfloor\sqrt{n}\rfloor & \text{classification} \\ \lfloor n/3\rfloor & \text{regression}\end{cases}
$$
- Breiman's original recommendation. sklearn uses $\sqrt{n}$ for classification but ALL $n$ features for regression by default.

Prediction: majority vote (classification) / mean (regression) over $B$ trees.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- ✅Strong out-of-the-box accuracy with near-zero tuning.
- ✅Parallel training.
- ✅Free validation via OOB error.
- ✅Robust to outliers & noise; ❌scaling needed.
- ✅Handles $n\gg m$ and irrelevant features gracefully.

*Cons?*
- ❌Interpretability (hundreds of trees).
- ⬆️Memory & inference latency.
- Usually loses to tuned gradient boosting on tabular data.
- Regression still can't extrapolate beyond the training target range.
- MDI feature importance biased toward high-cardinality features.

*Does adding more trees overfit?*
- **No.** Test error converges as $B\to\infty$ (it's an average of i.i.d.-ish estimators, not a fit to the data).
- $B$ trades compute for a smaller Monte-Carlo error only. Overfitting comes from tree depth / too-large $m_\text{try}$, not from $B$.
- Contrast with boosting, where more rounds absolutely DO overfit.

*What does $m_\text{try}$ do?*
- $m_\text{try}$⬇️ → trees more decorrelated ($\rho$⬇️) but individually weaker ($\sigma^2$⬆️). $m_\text{try}=n$ → plain bagging.
- The sweet spot balances the two terms in $\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$.

*Extra Trees (Extremely Randomized Trees)?*
- Adds a second randomization: thresholds are drawn at RANDOM instead of being optimized, and (by default) it uses the whole training set, no bootstrap.
- → ⬇️Variance, ⬆️Bias, much faster training (no threshold search).
```

&nbsp;

### Boosting
- **What**: Weak learners fit sequentially, each correcting the current ensemble's errors, then summed.
- **Why**: Bias⬇️.
    - Bagging can't fix bias — averaging identically-biased models keeps the bias.
    - A weak learner can be boosted to arbitrarily low training error by re-focusing on what's still wrong.
- **How**: Additive model $F_t=F_{t-1}+\eta h_t$, where $h_t$ is fit to whatever the previous ensemble got wrong.

```{dropdown} Table: Bagging vs Boosting
| | Bagging | Boosting |
|:--|:--|:--|
| Learner order | Parallel, independent | Sequential, dependent |
| Data per learner | Bootstrap resample | Reweighted / residual-fitted |
| Targets | Variance | Bias (and variance, via shrinkage) |
| Base learner | Strong, deep, low-bias | Weak, shallow (depth 1–8) |
| Aggregation | Uniform vote/average | Weighted sum |
| More rounds → overfit? | ❌ No | ⚠️ Can (monitor validation) |
| Outlier/noise robustness | ✅ High | ❌ Low (keeps chasing hard points) |
| Typical accuracy (tabular) | Good | Best, if tuned |
```

&nbsp;

#### AdaBoost
- **What**: Boosting by reweighting samples, up-weighting whatever the ensemble misclassifies.
- **Why**: First proof that weak learnability implies strong learnability → no strong base model needed.
- **How**:
    1. Start with uniform sample weights.
    2. Fit a weak learner (usually a depth-1 **stump**) on the weighted data.
    3. Score it: lower weighted error → larger vote weight $\alpha_t$.
    4. Multiply the weights of misclassified samples up, correct ones down; renormalize.
    5. Repeat; final prediction = sign of the $\alpha$-weighted vote.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $y_i\in\{-1,+1\}$: Label.
- Hyperparams:
    - $T$: #rounds.
- Misc:
    - $w_i^{(t)}$: Weight of sample $i$ at round $t$.
    - $\epsilon_t$: Weighted error.
    - $\alpha_t$: Learner weight.
    - $Z_t$: Normalizer.

Process:
1. Init: $w_i^{(1)}=\frac{1}{m}$.
2. For $t=1,\cdots,T$:
    1. Fit $h_t$ on $\{(\mathbf{x}_i,y_i)\}$ weighted by $\mathbf{w}^{(t)}$.
    2. Weighted error:

        $$
        \epsilon_t=\sum_{i=1}^{m}w_i^{(t)}\mathbb{1}[h_t(\mathbf{x}_i)\neq y_i]
        $$

    3. Learner weight:

        $$
        \alpha_t=\frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}
        $$

    4. Reweight & normalize:

        $$
        w_i^{(t+1)}=\frac{w_i^{(t)}e^{-\alpha_ty_ih_t(\mathbf{x}_i)}}{Z_t}
        $$

3. Output:

    $$
    H(\mathbf{x})=\text{sign}\left(\sum_{t=1}^{T}\alpha_th_t(\mathbf{x})\right)
    $$

Equivalent objective — forward stagewise additive modeling on **exponential loss**:

$$
L=\sum_{i=1}^{m}e^{-y_iF(\mathbf{x}_i)},\qquad F(\mathbf{x})=\sum_t\alpha_th_t(\mathbf{x})
$$
```

```{tip} Derivation
:class: dropdown
*Where does $\alpha_t=\frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$ come from?*
1. At round $t$, minimize the exponential loss of $F_{t-1}+\alpha h_t$:

    $$
    L(\alpha)=\sum_iw_i^{(t)}e^{-\alpha y_ih_t(\mathbf{x}_i)},\qquad w_i^{(t)}\propto e^{-y_iF_{t-1}(\mathbf{x}_i)}
    $$

2. $y_ih_t(\mathbf{x}_i)\in\{-1,+1\}$ → split the sum into correct & incorrect (weights normalized, $\sum_iw_i^{(t)}=1$):

    $$
    L(\alpha)=e^{-\alpha}(1-\epsilon_t)+e^{\alpha}\epsilon_t
    $$

3. Set $\frac{dL}{d\alpha}=-e^{-\alpha}(1-\epsilon_t)+e^{\alpha}\epsilon_t=0$:

    $$
    e^{2\alpha}=\frac{1-\epsilon_t}{\epsilon_t}\ \Rightarrow\ \alpha_t=\frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}
    $$

4. Sanity check: $\epsilon_t=0.5$ → $\alpha_t=0$ (a coin flip gets no vote). $\epsilon_t\to0$ → $\alpha_t\to\infty$. $\epsilon_t>0.5$ → $\alpha_t<0$ (flip the learner).
5. → AdaBoost's reweighting is NOT a heuristic; it is exact coordinate descent on exponential loss.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class AdaBoost:
    """Discrete AdaBoost with decision stumps."""

    def __init__(self, n_rounds=20):
        self.n_rounds = n_rounds
        self.learners = []                        ## (feature, threshold, sign, alpha)

    @staticmethod
    def _best_stump(X, y, w):
        best, best_err = None, np.inf
        for j in range(X.shape[1]):
            for t in np.unique(X[:, j]):
                for s in (1, -1):
                    pred = np.where(s * X[:, j] <= s * t, -1, 1)
                    err = w[pred != y].sum()      ## WEIGHTED error, not plain error rate
                    if err < best_err:
                        best, best_err = (j, t, s), err
        return best, best_err

    def fit(self, X, y):
        y = np.where(y <= 0, -1, 1)
        w = np.full(len(y), 1 / len(y))
        for _ in range(self.n_rounds):
            (j, t, s), err = self._best_stump(X, y, w)
            err = np.clip(err, 1e-10, 1 - 1e-10)  ## avoid log(0) on a perfect stump
            alpha = 0.5 * np.log((1 - err) / err)
            pred = np.where(s * X[:, j] <= s * t, -1, 1)
            ## up-weight what this stump got wrong, down-weight what it got right
            w *= np.exp(-alpha * y * pred)
            w /= w.sum()
            self.learners.append((j, t, s, alpha))
        return self

    def predict(self, X):
        F = np.zeros(len(X))
        for j, t, s, alpha in self.learners:
            F += alpha * np.where(s * X[:, j] <= s * t, -1, 1)
        return np.sign(F)

## Example
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([-1, -1, 1, 1, 1])
print(AdaBoost(n_rounds=5).fit(X, y).predict(X))  ## [-1. -1.  1.  1.  1.]
```
````

```{attention} Q&A
:class: dropdown
*Why depth-1 stumps?*
- Boosting needs WEAK learners — strong ones drive $\epsilon_t\to0$ immediately, $\alpha_t\to\infty$, and the ensemble degenerates into one overfit model.
- Depth controls the max interaction order captured: stumps = additive model, depth-2 = pairwise interactions, etc.

*Why is it so sensitive to noise & outliers?*
- Exponential loss grows as $e^{-yF(\mathbf{x})}$ → a permanently misclassified (mislabeled) point gets exponentially growing weight → the ensemble ends up fitting the label noise.
- Fix: LogitBoost / GBM with log loss (linear tail growth), or cap the #rounds.

*Does AdaBoost overfit?*
- It CAN, but famously resists: test error often keeps dropping after training error hits 0 ← the **margin** keeps increasing.
- On noisy labels it overfits badly. Not a free lunch — monitor validation loss rather than assuming either outcome.

*AdaBoost vs Gradient Boosting?*
- AdaBoost = gradient boosting with exponential loss, where the "gradient" is expressed as sample reweighting.
- GBM generalizes to any differentiable loss by fitting the negative gradient directly.

*What if $\epsilon_t>0.5$?*
- $\alpha_t<0$ → the learner's vote is inverted. Binary implementations usually just restart or stop.
- Multiclass uses **SAMME**, which drops the $\frac{1}{2}$ and adds $\log(K-1)$:

    $\alpha_t=\log\frac{1-\epsilon_t}{\epsilon_t}+\log(K-1)$

- → Requirement relaxes from $\epsilon_t<\frac{1}{2}$ to $\epsilon_t<1-\frac{1}{K}$, i.e., accuracy better than random guessing at $\frac{1}{K}$.
```

&nbsp;

#### GBDT
- **Name**: Gradient Boosting Decision Tree
- **What**: Boosting where each tree fits the negative gradient of the loss w.r.t. the current predictions.
- **Why**: Any differentiable (or subdifferentiable) loss.
    - AdaBoost's reweighting is hard-wired to exponential loss → binary classification only, noise-fragile.
    - → Reframe boosting as **gradient descent in function space** (squared, absolute, Huber, log, Poisson, ranking, quantile).
    - MAE & quantile losses use a negative **sub**gradient at the kink.
- **How**:
    1. Initialize with a constant $F_0$ (the loss-optimal constant).
    2. Compute pseudo-residuals = negative gradient of the loss at the current predictions.
    3. Fit a shallow regression tree to those residuals.
    4. Solve for the optimal constant per leaf (line search).
    5. Add it to the ensemble, scaled by learning rate $\eta$ (**shrinkage**).
    6. Repeat.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $F_t$: Ensemble after $t$ rounds.
    - $h_t$: Tree fit at round $t$.
- Hyperparams:
    - $T$: #rounds.
    - $\eta\in(0,1]$: Learning rate / shrinkage.
- Misc:
    - $r_{it}$: Pseudo-residual for sample $i$ at round $t$.
    - $R_{jt}$: Region (leaf) $j$ of tree $t$.
    - $\gamma_{jt}$: Output value of leaf $R_{jt}$.

Process:
1. Init:

    $$
    F_0(\mathbf{x})=\arg\min_\gamma\sum_{i=1}^{m}L(y_i,\gamma)
    $$

2. For $t=1,\cdots,T$:
    1. Pseudo-residuals:

        $$
        r_{it}=-\left[\frac{\partial L(y_i,F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F=F_{t-1}}
        $$

    2. Fit a regression tree $h_t$ to $\{(\mathbf{x}_i,r_{it})\}$ → leaves $\{R_{jt}\}$.
    3. Leaf values by line search:

        $$
        \gamma_{jt}=\arg\min_\gamma\sum_{\mathbf{x}_i\in R_{jt}}L\left(y_i,F_{t-1}(\mathbf{x}_i)+\gamma\right)
        $$

    4. Update:

        $$
        F_t(\mathbf{x})=F_{t-1}(\mathbf{x})+\eta\sum_j\gamma_{jt}\mathbb{1}[\mathbf{x}\in R_{jt}]
        $$

Special case (squared loss $L=\frac{1}{2}(y-F)^2$): $r_{it}=y_i-F_{t-1}(\mathbf{x}_i)$ → **fitting the plain residuals**.
```

````{important} Code
:class: dropdown
```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor   ## tree is scaffolding; boosting is the point

class GradientBoostingRegressor:
    def __init__(self, n_rounds=50, lr=0.1, max_depth=2):
        self.n_rounds, self.lr, self.max_depth = n_rounds, lr, max_depth
        self.F0, self.trees = 0.0, []

    def fit(self, X, y):
        ## squared loss -> the optimal constant is the mean
        self.F0 = y.mean()
        F = np.full(len(y), self.F0)
        for _ in range(self.n_rounds):
            residual = y - F                     ## = -dL/dF for L = 0.5(y-F)^2
            tree = DecisionTreeRegressor(max_depth=self.max_depth).fit(X, residual)
            F += self.lr * tree.predict(X)       ## shrinkage: never take the full step
            self.trees.append(tree)
        return self

    def predict(self, X):
        return self.F0 + self.lr * sum(t.predict(X) for t in self.trees)

## Example
rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, size=(200, 1))
y = np.sin(X[:, 0])                              ## nonlinear -> one stump can't do it
model = GradientBoostingRegressor().fit(X, y)
print(np.abs(model.predict(X) - y).mean() < 0.1)  ## True
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ✅Best-in-class accuracy on tabular data.
- ✅Any differentiable loss → regression, classification, ranking, quantiles, survival.
- ✅Handles mixed types, ❌scaling.

*Cons?*
- ❌Parallel across rounds (sequential by construction) → slower to train than RF.
- ⬆️Overfits with too many rounds / too-deep trees.
- ⬆️Hyperparameter sensitivity → needs real tuning.
- ⬆️Sensitivity to noisy labels (less than AdaBoost, more than RF).

*Why does shrinkage ($\eta$) help?*
- Small $\eta$ → each tree contributes a sliver → the ensemble explores many small corrections instead of committing to a few big ones → ⬇️variance.
- Empirically $\eta\lesssim0.1$ + more rounds > $\eta=1$ + few rounds.
- $\eta$ and $T$ trade off roughly as $\eta\cdot T\approx\text{const}$ → halve $\eta$, double $T$.

*Key hyperparameters, ranked?*
1. $T$ (#rounds) — set by early stopping on a validation set.
2. $\eta$ — 0.01–0.1.
3. `max_depth` (3–8) or `num_leaves` — controls interaction order.
4. `subsample` (0.5–0.8) → **Stochastic Gradient Boosting**: row sampling per round → ⬇️variance + faster.
5. `min_samples_leaf`, L1/L2 on leaf weights.

*GBDT vs Random Forest?*
- RF: parallel, variance-focused, deep trees, near-tuning-free, more rounds never hurt.
- GBDT: sequential, bias-focused, shallow trees, tuning-hungry, more rounds DO overfit.
- GBDT usually wins on accuracy; RF wins on robustness & time-to-first-model.

*Why shallow trees here but deep trees in RF?*
- GBDT reduces bias by stacking many weak corrections → each tree must stay weak or the ensemble overfits instantly.
- RF reduces variance by averaging → each tree should be as unbiased (deep) as possible.

*Can it extrapolate?*
- No. Trees are piecewise constant → the model cannot continue a trend past the outermost split of the training range; it flattens.
- ⚠️ Unlike RF (an average, hence bounded by $[\min y,\max y]$), GBDT is a **sum** of corrections → its output CAN land outside the training target range.
```

&nbsp;

#### XGBoost
- **What**: GBDT + 2nd-order Taylor loss approximation + explicit tree regularization.
- **Why**:
    - 1st derivative only → the split criterion is a proxy (residual variance), not the actual loss reduction.
    - Regularization only implicit (depth, shrinkage) → no principled leaf-weight penalty.
    - Exhaustive threshold scans choke on sparse/large data.
- **How**:
    1. Taylor-expand the loss to 2nd order around the current prediction → per-sample $g_i$ (gradient) and $h_i$ (hessian).
    2. Add $\Omega(f)=\gamma T+\frac{1}{2}\lambda||\mathbf{w}||_2^2$ over leaves.
    3. → Closed-form optimal leaf weight & an exact **gain** formula → use it as the split criterion, with $\gamma$ acting as a minimum-gain pruning threshold.
    4. Systems: weighted quantile sketch for candidate splits, sparsity-aware default direction, column subsampling, cache-aware & out-of-core execution.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $w_j$: Weight (output) of leaf $j$.
    - $T$: #leaves.
- Hyperparams:
    - $\gamma$: Per-leaf complexity penalty (min split gain).
    - $\lambda$: L2 on leaf weights.
- Misc:
    - $g_i=\partial_{\hat{y}^{(t-1)}}L(y_i,\hat{y}_i^{(t-1)})$: Gradient.
    - $h_i=\partial^2_{\hat{y}^{(t-1)}}L(y_i,\hat{y}_i^{(t-1)})$: Hessian.
    - $I_j$: Sample set of leaf $j$.
    - $G_j=\sum_{i\in I_j}g_i$, $H_j=\sum_{i\in I_j}h_i$.

Objective at round $t$ (2nd-order approximation, constants dropped):

$$
\tilde{\mathcal{L}}^{(t)}=\sum_{i=1}^{m}\left[g_if_t(\mathbf{x}_i)+\frac{1}{2}h_if_t^2(\mathbf{x}_i)\right]+\gamma T+\frac{1}{2}\lambda\sum_{j=1}^{T}w_j^2
$$

Optimal leaf weight & resulting objective (fixed tree structure):

$$
w_j^*=-\frac{G_j}{H_j+\lambda},\qquad\tilde{\mathcal{L}}^{(t)*}=-\frac{1}{2}\sum_{j=1}^{T}\frac{G_j^2}{H_j+\lambda}+\gamma T
$$

Split gain:

$$
\text{Gain}=\frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma
$$
- Gain $<0$ → don't split → $\gamma$ IS the pruning knob.
```

```{tip} Derivation
:class: dropdown
*Where do $w_j^*$ and the gain come from?*
1. A tree is constant on each leaf → group the sum by leaf:

    $$
    \tilde{\mathcal{L}}^{(t)}=\sum_{j=1}^{T}\left[G_jw_j+\frac{1}{2}(H_j+\lambda)w_j^2\right]+\gamma T
    $$

2. Each leaf is now an independent 1-D quadratic $aw+\frac{1}{2}bw^2$ with $b=H_j+\lambda>0$.
3. Minimizer: $w_j^*=-\frac{a}{b}=-\frac{G_j}{H_j+\lambda}$; minimum value: $-\frac{a^2}{2b}=-\frac{G_j^2}{2(H_j+\lambda)}$.
4. Substitute back → $\tilde{\mathcal{L}}^{(t)*}=-\frac{1}{2}\sum_j\frac{G_j^2}{H_j+\lambda}+\gamma T$.
5. Gain of a split = (objective before) − (objective after) → the bracketed expression, minus $\gamma$ for the one extra leaf.
6. Note $\lambda$ appears in the DENOMINATOR → it shrinks leaf weights AND makes low-hessian (low-confidence) leaves less attractive to split.
```

```{attention} Q&A
:class: dropdown
*Why 2nd order?*
- Newton step $-\frac{g}{h}$ converges faster & needs no per-leaf line search (it's already the closed-form optimum of the approximation).
- The gain formula measures the decrease in the **2nd-order surrogate**, a much tighter proxy than plain residual variance.
    - ⚠️ Not the exact loss decrease — exact only for squared loss, where higher derivatives vanish.
- For squared loss $h_i=1$, so XGBoost's gain reduces to the classic variance-reduction criterion (with $\lambda=0$).

*What does each regularizer do?*
- $\lambda$ (L2 on leaf weights) → shrinks predictions, especially in small/low-hessian leaves.
- $\gamma$ (`min_split_loss`) → a split must earn at least $\gamma$ → post-pruning built into the split search.
- $\alpha$ (L1 on leaf weights) → can zero out leaf outputs.
- Plus `subsample`, `colsample_bytree`, `min_child_weight` ($=\min\sum h_i$ per leaf).

*Sparsity-aware split finding?*
- Each node learns a **default direction**; missing (or zero, in sparse format) values go there.
- Only non-missing values are enumerated → complexity scales with #non-missing entries, not $m\times n$ → the source of the big speedup on sparse data.

*Approximate split finding?*
- Exact greedy enumerates every threshold → requires sorting each feature, and doesn't fit in memory at scale.
- Approx: propose candidate percentiles via a **weighted quantile sketch** (weighted by $h_i$, because the objective is an $h_i$-weighted squared loss) → bucket into histograms.

*XGBoost vs GBM in one line?*
- Same algorithm; XGBoost adds 2nd-order gain, explicit $\gamma/\lambda$ regularization, sparsity handling, and systems engineering.
```

&nbsp;

#### LightGBM
- **What**: GBDT + histogram binning + leaf-wise growth + gradient-based subsampling + feature bundling.
- **Why**: Speed.
    - Split finding costs $O(\#\text{data}\times\#\text{features})$.
    - → Attack BOTH factors: drop low-gradient samples (GOSS), merge mutually exclusive sparse features (EFB).
- **How**:
    1. **Histogram binning**: bucket continuous features into a few hundred bins → split search becomes $O(\#\text{bins})$; one child's histogram comes free by subtracting its sibling's from the parent's.
    2. **Leaf-wise growth**: split the leaf with the highest gain globally, not level-by-level → lower loss per leaf added.
    3. **GOSS** (Gradient-based One-Side Sampling): keep all large-gradient samples, randomly sample the small-gradient ones and up-weight them to keep the gain estimate unbiased.
    4. **EFB** (Exclusive Feature Bundling): sparse features that are rarely nonzero simultaneously get packed into one → $\#\text{features}$⬇️.

```{attention} Q&A
:class: dropdown
*Leaf-wise vs level-wise growth?*
- Level-wise (XGBoost's default): grow all nodes at a depth → balanced, more regularized, wastes splits on low-gain nodes.
- Leaf-wise (LightGBM): always split the best leaf → deeper, asymmetric trees, lower loss for the same #leaves.
- ⚠️ Leaf-wise **overfits small datasets** → control with `num_leaves` (< $2^{\text{max\_depth}}$) and `min_data_in_leaf`.

*Why is it so much faster than XGBoost's exact mode?*
- Histograms: $O(\#\text{bins})$ split search instead of a sort over every distinct value.
- Sibling histogram by subtraction → costs $O(\#\text{bins})$ regardless of how many samples that child holds (only ONE child is actually scanned).
- GOSS: fewer rows. EFB: fewer columns.
- (XGBoost also ships a `hist` tree method now → the gap is much smaller than the original paper's benchmarks.)

*Why is GOSS "one-side"?*
- Small gradients = already well-fit samples. Dropping them loses little information, but dropping them naively would bias the data distribution → the retained small-gradient samples are up-weighted to compensate.

*Categorical features?*
- Native support: sorts categories by accumulated gradient statistics and splits that ordering → avoids one-hot's deep, unbalanced trees.

*When NOT LightGBM?*
- Small data (< a few thousand rows) → leaf-wise growth overfits; use XGBoost/RF or heavily constrain `num_leaves`.
```

&nbsp;

#### CatBoost
- **What**: GBDT + permutation-based ordering to kill target leakage & prediction shift.
- **Why**:
    - **Target statistics leakage**: encoding a category by its mean target uses that row's OWN label → worst for rare/high-cardinality categories.
    - **Prediction shift**: gradients estimated on the SAME data used to fit previous trees → $g_i$ biased → ensemble systematically off.
- **How**:
    1. **Ordered target statistics**: fix a random permutation; encode each row using only rows BEFORE it (plus a prior) → no self-leakage.
    2. **Ordered boosting**: maintain models trained only on preceding rows; compute each row's residual from a model that has never seen it.
    3. **Oblivious (symmetric) trees**: the same (feature, threshold) at every node of a level → strong regularization + branch-free, very fast inference.
    4. Automatic feature combinations of categoricals during tree construction.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\sigma$: Random permutation of the training rows.
    - $P$: Prior.
    - $a>0$: Prior weight.

Ordered target statistic for a categorical feature $j$:

$$
\hat{x}_{\sigma(i),j}=\frac{\sum_{p<i}\mathbb{1}[x_{\sigma(p),j}=x_{\sigma(i),j}]\cdot y_{\sigma(p)}+a\cdot P}{\sum_{p<i}\mathbb{1}[x_{\sigma(p),j}=x_{\sigma(i),j}]+a}
$$
- $P$ is typically the global target mean → smooths rare categories toward it.
- Only samples earlier in the permutation contribute → the target of sample $i$ never enters its own encoding.
```

```{dropdown} Table: XGBoost vs LightGBM vs CatBoost
| | XGBoost | LightGBM | CatBoost |
|:--|:--|:--|:--|
| Tree growth | Level-wise (leaf-wise also available) | Leaf-wise (best-first) | Oblivious / symmetric |
| Split finding | Exact greedy or histogram | Histogram + GOSS + EFB | Histogram |
| Categoricals | One-hot / external encoding (native support added later) | Native (gradient-sorted) | Native (ordered target statistics) |
| Overfitting control | $\gamma$, $\lambda$, $\alpha$, depth, subsample | `num_leaves`, `min_data_in_leaf` | Ordered boosting, symmetric trees |
| Speed (large data) | Fast | Fastest | Medium (fastest at inference) |
| Small / noisy data | Good | Overfit-prone | Best |
| Best at | General default | Speed & scale | High-cardinality categoricals |
```

```{attention} Q&A
:class: dropdown
*Why is mean target encoding dangerous?*
- A category appearing once gets encoded as exactly its own label → the tree can memorize the target through the feature → perfect train accuracy, garbage test accuracy.
- Standard fixes: K-fold/leave-one-out target encoding + smoothing toward a prior. CatBoost's ordering scheme is a systematic version of the same idea.

*What is prediction shift?*
- The gradient's conditional distribution estimated on training data $\neq$ its true conditional distribution, because $F_{t-1}$ was fit on those very rows → gradients are biased → the bias compounds over rounds.
- Ordered boosting breaks the dependency by evaluating each residual with a model that excluded that row.

*Cost of oblivious trees?*
- Less expressive per tree (one split condition per level) → needs more trees, but each is far cheaper, heavily regularized, and inference is a branch-free index lookup.
```

&nbsp;

### Stacking
- **What**: Meta-learner trained on the out-of-fold predictions of several base learners.
- **Why**: Learned combination.
    - Simple averaging weights every model equally, even the bad ones.
    - Different models are accurate in **different regions** → a meta-learner learns per-region trust.
    - Diverse families (trees + linear + kNN + SVM) make different errors → combining beats any single one.
- **How**:
    1. K-fold split the training data.
    2. For each base model: train on K−1 folds, predict the held-out fold → assemble a full column of **out-of-fold (OOF)** predictions.
    3. Meta-learner (usually something simple: regularized linear / logistic regression) trains on the matrix of OOF predictions.
    4. Refit base models on all data; at test time feed their predictions to the meta-learner.

```{attention} Q&A
:class: dropdown
*Why must the meta-features be out-of-fold?*
- In-sample predictions are optimistically accurate → the meta-learner learns to trust an overfit base model that will not be that accurate at test time → severe leakage & a broken blend.

*Stacking vs Blending?*
- Blending: hold out ONE validation split for meta-features. Simpler, faster, no leakage risk from fold reuse, but the meta-learner sees less data and wastes the holdout.
- Stacking: K-fold → uses all data, higher variance in meta-features, more compute.

*Why keep the meta-learner simple?*
- It trains on very few effective features (#base models) with correlated columns → a complex meta-learner overfits instantly.
- Common choice: non-negative least squares or L2 logistic regression, which also keeps the blend interpretable.

*Voting vs Stacking?*
- Hard voting: majority label. Soft voting: average probabilities (usually better ← uses confidence). Both are fixed, untrained rules.
- Stacking learns the weights → strictly more powerful, strictly more overfit-prone.

*How to pick base models?*
- Maximize **diversity**, not individual accuracy: different inductive biases (linear, tree, distance-based, kernel), different feature subsets, different seeds.
- Highly correlated base models add cost without adding information.
```

&nbsp;

## Instance-Based Methods
- **What**: No training phase — memorize the data, defer all computation to query time ("lazy" learning).

### KNN
- **Name**: k-Nearest Neighbors
- **What**: Prediction from the labels of the $k$ closest training points.
- **Why**: Zero distributional assumptions.
    - Non-parametric → fits any decision boundary given enough data.
    - Zero training cost → instantly absorbs new data (online/streaming).
    - As $m\to\infty$, 1-NN's error is at most twice the Bayes error.
- **How**:
    1. Compute the distance from the query to every training point.
    2. Take the $k$ smallest.
    3. Majority vote (classification) / mean (regression), optionally weighted by $\frac{1}{d}$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $k$: #neighbors.
    - $p$: Minkowski order.
- Misc:
    - $N_k(\mathbf{x})$: Index set of the $k$ nearest training points to $\mathbf{x}$.

Distance (Minkowski, $p\geq1$ for it to be a metric):

$$
d_p(\mathbf{x},\mathbf{z})=\left(\sum_{j=1}^{n}|x_j-z_j|^p\right)^{1/p}
$$
- $p=1$: Manhattan.
- $p=2$: Euclidean.
- $p\to\infty$: Chebyshev.

Prediction:

$$
\hat{y}=\begin{cases}\arg\max_c\sum_{i\in N_k(\mathbf{x})}\mathbb{1}[y_i=c] & \text{classification} \\ \frac{1}{k}\sum_{i\in N_k(\mathbf{x})}y_i & \text{regression}\end{cases}
$$

Distance-weighted variant: weight neighbor $i$ by $\frac{1}{d(\mathbf{x},\mathbf{x}_i)+\epsilon}$.

Complexity (brute force): Train $O(1)$, Predict $O(mn)$ per query, Memory $O(mn)$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X, self.y = None, None

    def fit(self, X, y):
        ## "training" = memorizing. This is what makes KNN a lazy learner.
        self.X, self.y = np.asarray(X), np.asarray(y)
        return self

    def predict(self, X):
        ## (q, 1, n) - (1, m, n) -> (q, m, n) -> pairwise distances (q, m)
        d = np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2)
        idx = np.argsort(d, axis=1)[:, :self.k]        ## k nearest per query
        return np.array([np.bincount(self.y[row]).argmax() for row in idx])

## Example
X = np.array([[0.0, 0.0], [0.1, 0.1], [3.0, 3.0], [3.1, 3.1]])
y = np.array([0, 0, 1, 1])
print(KNNClassifier(k=3).fit(X, y).predict(np.array([[2.9, 3.0]])))  ## [1]
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ✅Zero training, ✅Trivially incremental.
- ✅Non-parametric → arbitrary decision boundaries.
- ✅Naturally multiclass, ✅Works for regression too.
- ✅Local explanations for free ("these 5 neighbors decided it").

*Cons?*
- ❌Inference cost & memory: $O(mn)$ per query, stores the entire dataset.
- ❌Curse of dimensionality.
- ⬆️Sensitivity to feature scaling & irrelevant features.
- ⬆️Sensitivity to class imbalance (the majority class dominates any neighborhood).
- ❌Interpretable global model.

*How to pick $k$?*
- $k$⬇️ (→1) → jagged boundary, ⬆️variance, fits noise. $k$⬆️ (→$m$) → smooth boundary, ⬆️bias, converges to predicting the majority class.
- CV. Rule of thumb $k\approx\sqrt{m}$. Use ODD $k$ for binary classification to avoid ties.

*Why does the curse of dimensionality kill KNN?*
- As $n$⬆️, the ratio $\frac{d_{\max}-d_{\min}}{d_{\min}}\to0$ → all points become equidistant → "nearest" stops meaning anything.
- To keep a fixed fraction of the data within a neighborhood, the neighborhood's edge length must grow as $r^{1/n}$ → "local" neighborhoods stop being local.
- Fix: dimensionality reduction (PCA), feature selection, or a learned metric.

*Must I scale features?*
- Yes, always. An unscaled feature with a large range dominates the Euclidean distance → the other features are effectively ignored.

*How to make it fast?*
- KD-tree: sub-linear per query in low dims but degrades to brute force as $n$ grows (~20+).
- Ball-tree: better in moderate dims / non-Euclidean metrics.
- Approximate NN (HNSW, LSH, IVF): sub-linear, the only option at scale.

*Is KNN parametric?*
- No. #params grows with $m$ — the model IS the data.
- ⚠️ Non-parametric ≠ assumption-free: it assumes labels are **locally smooth** under a metric that actually reflects similarity.

*Does KNN have a training-time "fit"?*
- Only index construction. There is no loss, no optimization, no learned params. All of the inductive bias sits in $k$ and the metric.
```

&nbsp;

## Probabilistic Methods
- **What**: $P(\mathbf{x},y)$ or $P(y|\mathbf{x})$ modeled explicitly, classified by Bayes' rule.

### Naive Bayes
- **What**: Bayes' rule + features conditionally independent given the class.
- **Why**: Param count.
    - Full joint $P(x_1,\cdots,x_n|y)$ over $n$ binary features needs $O(2^n)$ params.
    - → Conditional independence collapses it to $O(nK)$ → one pass, closed-form, tiny data.
- **How**:
    1. Estimate class priors $P(y_k)$ by counting.
    2. Estimate each per-feature likelihood $P(x_j|y_k)$ independently (counts, or a Gaussian fit).
    3. Predict $\arg\max_k$ of the product (computed in log space).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\alpha$: Smoothing param ($\alpha=1$ → Laplace, $0<\alpha<1$ → Lidstone).
    - $N_{kj}$: Total count of feature $j$ in class $k$.
    - $N_k=\sum_jN_{kj}$: Total count in class $k$.

Bayes' rule:

$$
P(y_k|\mathbf{x})=\frac{P(y_k)P(\mathbf{x}|y_k)}{P(\mathbf{x})}\propto P(y_k)P(\mathbf{x}|y_k)
$$

Naive assumption:

$$
P(\mathbf{x}|y_k)=\prod_{j=1}^{n}P(x_j|y_k)
$$

Inference (log space → avoid underflow):

$$
\hat{y}=\arg\max_k\left[\log P(y_k)+\sum_{j=1}^{n}\log P(x_j|y_k)\right]
$$

Likelihoods:
- **Multinomial** (word counts): parameter $\theta_{kj}$ = probability that a token drawn from class $k$ is word $j$, smoothed:

    $$
    \theta_{kj}=\frac{N_{kj}+\alpha}{N_k+\alpha n}
    $$

    - Scoring uses the counts as **exponents**, so the log-score is count-weighted:

    $$
    \log P(\mathbf{x}|y_k)=\sum_{j=1}^{n}x_j\log\theta_{kj}\quad(+\text{const indep. of }k)
    $$

- **Bernoulli** (binary presence/absence): $P(x_j|y_k)$ is a smoothed presence rate, and ABSENT features contribute $(1-P(x_j|y_k))$ explicitly.
- **Gaussian** (continuous):

    $$
    P(x_j|y_k)=\frac{1}{\sqrt{2\pi\sigma_{kj}^2}}\exp\left(-\frac{(x_j-\mu_{kj})^2}{2\sigma_{kj}^2}\right)
    $$
```

````{important} Code
:class: dropdown
```python
import numpy as np

class GaussianNB:
    def __init__(self, eps=1e-9):
        self.eps = eps
        self.classes, self.log_prior, self.mu, self.var = None, None, None, None

    def fit(self, X, y):
        self.classes = np.unique(y)
        ## one (mean, var) pair PER CLASS PER FEATURE -- no covariance -> the "naive" part
        self.mu = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.var = np.array([X[y == c].var(axis=0) for c in self.classes]) + self.eps
        self.log_prior = np.log([np.mean(y == c) for c in self.classes])
        return self

    def _joint_log_likelihood(self, X):
        ## (q, 1, n) against (K, n) -> (q, K, n), summed over features
        ll = -0.5 * (np.log(2 * np.pi * self.var) +
                     (X[:, None, :] - self.mu) ** 2 / self.var)
        return self.log_prior + ll.sum(axis=2)   ## sum of logs == product of probs

    def predict(self, X):
        return self.classes[self._joint_log_likelihood(X).argmax(axis=1)]

## Example
X = np.array([[1.0, 1.0], [1.2, 0.9], [5.0, 5.0], [5.2, 4.8]])
y = np.array([0, 0, 1, 1])
print(GaussianNB().fit(X, y).predict(np.array([[5.1, 5.1]])))  ## [1]
```
````

```{dropdown} Table: Naive Bayes Variants
| Variant | Feature type | Per-feature parameter | Typical use |
|:--|:--|:--|:--|
| Gaussian | Continuous | $N(\mu_{kj},\sigma_{kj}^2)$ density | Generic numeric features |
| Multinomial | Counts / TF-IDF | $\theta_{kj}$, smoothed token frequency; counts enter as exponents | Text classification |
| Bernoulli | Binary | Smoothed presence rate; absences penalized | Short text, presence matters |
| Complement | Counts | Statistics from the COMPLEMENT of each class | Imbalanced text |
```

```{attention} Q&A
:class: dropdown
*Pros?*
- ⬇️⬇️Train & inference cost — one pass, closed-form, $O(mn)$.
- ✅Works with very little data, ✅Scales to huge $n$ (text) — a strong baseline even where the independence assumption is false.
- ✅Naturally multiclass, ✅Online-updatable (just update counts).
- ✅Robust to irrelevant features (they contribute near-equally to all classes).

*Cons?*
- ❌Feature interactions ← independence assumption.
- ❌Calibrated probabilities — outputs are pushed to 0/1 (see below).
- Continuous features force a distribution choice (usually Gaussian, often wrong) → discretize or transform.
- Correlated/duplicated features get their evidence double-counted.

*Why does it work despite an obviously false assumption?*
- Classification only needs the correct $\arg\max$, not correct probabilities. The independence violation distorts the magnitudes but often preserves the ranking.
- The estimator is very low-variance (few params, no interactions) → it beats correctly-specified but high-variance models on small $m$.

*Why are the probabilities badly calibrated?*
- Correlated features act as independent votes → the same evidence is counted many times → the product saturates → outputs near 0 or 1.
- Fix: isotonic regression / Platt scaling if you need real probabilities.

*Zero-frequency problem?*
- An unseen (feature, class) pair gives $P(x_j|y_k)=0$ → the whole product is 0, regardless of all other evidence.
- Fix: additive (Laplace/Lidstone) smoothing.

*Why work in log space?*
- Multiplying thousands of small probabilities underflows float64 to exactly 0. $\log$ turns it into a sum, and $\arg\max$ is unchanged (log is monotone).

*Naive Bayes vs Logistic Regression?*
- Same parametric form of the decision boundary (linear, for multinomial NB and for Gaussian NB with class-independent variances), but different fitting: NB is generative (MLE on the joint), LogReg is discriminative (MLE on the conditional).
- Ng & Jordan: NB approaches its asymptotic error faster (roughly $O(\log n)$ vs $O(n)$ samples), but that asymptote is usually higher → NB often wins on small $m$, LogReg overtakes as $m$⬆️.
- ⚠️ A tendency under their modelling assumptions, not a law — the crossover point is dataset-dependent.

*Is Gaussian NB the same as LDA/QDA?*
- Close cousins: Gaussian NB = QDA restricted to DIAGONAL covariance matrices. Dropping the off-diagonals is precisely the naive assumption.
```

&nbsp;