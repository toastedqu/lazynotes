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
# Objective
What a model minimizes during training. **Objective = Loss + Regularizer**; *how* it gets minimized is [optimization](optim.md).

Task-generic **training** objectives ONLY.
- ❌ Evaluation metrics (RMSE, accuracy, P/R/F1, AUC, $R^2$) — reported, not optimized.
- ❌ Field-specific objectives (CTC, Dice/IoU, GAN, ELBO, RL return, DPO) — on their own pages.

Default notations:
- $m$: #samples. $K$: #classes.
- $y_i$: True target for sample $i$. $\hat{y}_i$: Prediction.
- $e_i=y_i-\hat{y}_i$: Residual.
- $\mathbf{z}_i\in\mathbb{R}^K$: Logits for sample $i$. $\hat{p}_{ik}$: Predicted probability of class $k$.
- $\ell$: Per-sample loss. $\mathcal{L}$: Reduced objective over the batch/dataset.
- Reduction is `mean` unless stated otherwise.

&nbsp;

## Framework
### ERM
- **Name**: Empirical Risk Minimization
- **What**: Minimizing average training loss as a stand-in for expected loss on the true distribution.
- **Why**: The quantity we care about, $\mathbb{E}_{P}[\ell]$, is uncomputable ← $P$ unknown → replace the expectation with a sample average.
- **How**:
    1. Fix a hypothesis class $\mathcal{F}$.
    2. Fix a per-sample loss $\ell$.
    3. Average $\ell$ over the training set.
    4. Minimize over $\mathcal{F}$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}\in\mathcal{X}$: Input.
    - $y\in\mathcal{Y}$: Target.
- Misc:
    - $P$: True (unknown) joint distribution over $\mathcal{X}\times\mathcal{Y}$.
    - $\mathcal{F}$: Hypothesis class.
    - $f\in\mathcal{F}$: Hypothesis.
    - $\ell(y,f(\mathbf{x}))$: Per-sample loss.
    - $R^*=\inf_{f}R(f)$: Bayes risk (infimum over ALL measurable $f$).

Definition:

$$\begin{align*}
R(f)&=\mathbb{E}_{(\mathbf{x},y)\sim P}\left[\ell(y,f(\mathbf{x}))\right] &&\text{(true risk)}\\
\hat{R}(f)&=\frac{1}{m}\sum_{i=1}^{m}\ell(y_i,f(\mathbf{x}_i)) &&\text{(empirical risk)}
\end{align*}$$

$$
\hat{f}=\arg\min_{f\in\mathcal{F}}\hat{R}(f)
$$

Properties:
- $\mathbb{E}[\hat{R}(f)]=R(f)$ for **fixed** $f$ → unbiased.
- $\hat{R}(\hat{f})$ is optimistically biased ← $\hat{f}$ was chosen using the same sample.
- Excess risk splits:

$$
R(\hat{f})-R^*=\underbrace{\left[R(\hat{f})-\inf_{f\in\mathcal{F}}R(f)\right]}_{\text{estimation}}+\underbrace{\left[\inf_{f\in\mathcal{F}}R(f)-R^*\right]}_{\text{approximation}}
$$
```

```{attention} Q&A
:class: dropdown
*Why doesn't low training error guarantee low test error?*
- $\hat{R}$ is unbiased for a **fixed** $f$, but $\hat{f}$ is selected by minimizing $\hat{R}$ on the same data → the minimum is biased downward.
- Gap grows with the capacity of $\mathcal{F}$ → [overfitting](misc.md#overfitting).

*Assumptions?*
- Samples i.i.d. from a fixed $P$.
- Train & test drawn from the SAME $P$ → breaks under [covariate shift](../dl/issues.md#covariate-shift).
- $\ell$ decomposes over samples → ❌F1, ❌AUC.

*What controls the generalization gap?*
- Uniform deviation $\sup_{f\in\mathcal{F}}|R(f)-\hat{R}(f)|$, bounded by capacity measures (VC dimension, Rademacher complexity).
- Gap $\propto\sqrt{\text{capacity}/m}$ → $m$⬆️ or capacity⬇️ → gap⬇️.

*Estimation vs approximation error?*
- Estimation = finite data. $m$⬆️ → ⬇️. $\mathcal{F}$⬆️ → ⬆️. Analogous to variance.
- Approximation = $\mathcal{F}$ can't represent the Bayes rule. $\mathcal{F}$⬆️ → ⬇️. Analogous to bias.
- → Capacity trades one against the other. NOT literally the bias-variance decomposition — estimation error itself carries both.

*Is the ERM solution unique?*
- ❌ in general. Non-convex $\hat{R}$ (NNs) → many minima; convex-but-flat directions (e.g. $m<n$) → a solution set, not a point.
```

&nbsp;

### SRM
- **Name**: Structural Risk Minimization
- **What**: ERM plus an explicit complexity penalty.
- **Why**: ERM alone always prefers the most flexible hypothesis, which fits noise → the fit term has nothing pushing back on capacity.
- **How**:
    1. Order hypothesis classes by capacity: $\mathcal{F}_1\subset\mathcal{F}_2\subset\cdots$.
    2. Bound true risk by empirical risk + a capacity term.
    3. Minimize the bound instead of $\hat{R}$ alone.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda\ge0$: Penalty weight.
- Misc:
    - $\Omega(f)\ge0$: Complexity penalty.
    - $t$: Complexity budget.

Objective:

$$
\mathcal{L}(f)=\underbrace{\hat{R}(f)}_{\text{fit}}+\lambda\underbrace{\Omega(f)}_{\text{capacity}}
$$

Equivalent constrained form:

$$
\min_{f\in\mathcal{F}}\hat{R}(f)\quad\text{s.t.}\quad\Omega(f)\le t
$$
- Convex $\hat{R},\Omega$ + a constraint qualification (Slater) → each budget $t$ admits a $\lambda$ whose penalized solution set contains the constrained one. Convexity ALONE is not enough.
```

```{attention} Q&A
:class: dropdown
*How is the penalty actually implemented?*
- Explicit norm penalty ([L2](#l2), [L1](#l1)).
- Hard constraint (norm ball, max-norm clipping).
- Implicit: [early stopping](../dl/train.md#early-stopping), dropout, data augmentation, smaller architecture, ensembling.

*How is $\lambda$ chosen?*
- Validation set / CV. NEVER training loss ← $\lambda=0$ always wins on training loss.

*Why not just pick the smallest class that fits?*
- Capacity is not one-dimensional in practice (depth, width, weight norm, LR schedule all move it) → tuning a continuous $\lambda$ is easier than enumerating a nested family.

*Does more capacity always mean worse generalization?*
- ❌ Overparameterized NNs interpolate the training set and still generalize (double descent) → classical uniform-capacity bounds are loose here; implicit bias of SGD toward low-norm solutions does part of the regularizing.
```

&nbsp;

### Surrogate Loss
- **What**: Convex, gradient-friendly stand-in for a discrete target objective.
- **Why**: The objective we care about in classification (0-1 loss) is piecewise constant → gradient 0 almost everywhere, non-convex, NP-hard to minimize directly.
- **How**:
    1. Write the target objective as a function of the margin $u=yf(\mathbf{x})$.
    2. Upper-bound it by a convex, decreasing function of $u$.
    3. Minimize the bound.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $y\in\{-1,+1\}$: Label (sign convention).
- Misc:
    - $f(\mathbf{x})\in\mathbb{R}$: Real-valued score.
    - $u=yf(\mathbf{x})$: Margin. $u>0$ $\Leftrightarrow$ correct.

Margin losses:

$$\begin{align*}
\ell_{0/1}(u)&=\mathbb{1}[u\le0] \\
\ell_\text{hinge}(u)&=\max(0,1-u) \\
\ell_\text{log}(u)&=\log(1+e^{-u}) \\
\ell_\text{exp}(u)&=e^{-u} \\
\ell_\text{sq}(u)&=(1-u)^2
\end{align*}$$

Upper bounds on $\ell_{0/1}$:

$$
\ell_{0/1}(u)\le\ell_\text{hinge}(u),\quad
\ell_{0/1}(u)\le\ell_\text{exp}(u),\quad
\ell_{0/1}(u)\le\frac{\ell_\text{log}(u)}{\log 2}
$$
- Natural-log logistic loss is NOT itself an upper bound ← $\ell_\text{log}(0)=\log2<1$.

Classification calibration: a convex margin loss drives the sign of its population minimizer to the Bayes rule $\Leftrightarrow$ it is differentiable at $0$ with $\ell'(0)<0$.
```

```{dropdown} Table: Surrogate Losses
| | $\ell(u)$ | Grad as $u\to-\infty$ | Zero past margin | Probabilities | Model |
|:--|:--|:--|:--|:--|:--|
| 0-1 | $\mathbb{1}[u\le0]$ | 0 | ✅ | ❌ | — (evaluation only) |
| Hinge | $\max(0,1-u)$ | $-1$ (bounded) | ✅ ($u\ge1$) | ❌ | [SVM](supervised.md#svm) |
| Logistic | $\log(1+e^{-u})$ | $-1$ (bounded) | ❌ | ✅ | [Logistic Regression](supervised.md#logistic-regression) |
| Exponential | $e^{-u}$ | $-\infty$ | ❌ | ✅ (half log-odds) | [AdaBoost](supervised.md#adaboost) |
| Squared | $(1-u)^2$ | $-\infty$ | ❌ | ❌ | LS-SVM |

- Squared loss also penalizes $u>1$ → punishes confidently-correct predictions.
```

```{attention} Q&A
:class: dropdown
*Why not optimize accuracy directly?*
- Piecewise constant → $\nabla=0$ a.e., undefined at the boundary → gradient methods have nothing to follow.
- Non-convex, and minimizing 0-1 error over linear classifiers is NP-hard.

*Why must the surrogate upper-bound the 0-1 loss?*
- Then surrogate risk $\to0$ forces 0-1 risk $\to0$ → minimizing the bound is a valid proxy.
- Bound alone is not enough — calibration is what guarantees the sign of the minimizer matches the Bayes rule.

*What's the cost of the substitution?*
- Under model misspecification the surrogate minimizer $\neq$ 0-1 minimizer → the loss picks WHICH errors to trade away.
- Tail behavior differs sharply: exponential $\gg$ hinge $\approx$ logistic in outlier sensitivity.

*Does regression need surrogates?*
- ❌ Usually — squared/absolute error is already convex & differentiable (a.e.) in the prediction.
- ✅ For rank-based objectives (AUC, NDCG), same story as classification.

*Why can't set-level objectives like F1 or AUC be surrogated away as easily?*
- **Non-decomposable**: they depend on the whole set, not on a sum of per-sample terms → no per-sample gradient to hand SGD.
- Relaxations exist (soft-F1, pairwise logistic surrogate for AUC) but each optimizes a different function than the one being reported.
```

&nbsp;

### Reduction
- **What**: Aggregation of per-sample losses into one scalar.
- **Why**: Gradient-based optimizers need a scalar; the aggregation choice silently rescales every gradient.
- **How**: Sum, or divide by a normalizer (#samples, #valid tokens, #positive pairs).

```{note} Math
:class: dropdown
Definition:

$$
\mathcal{L}_\text{sum}=\sum_{i=1}^{m}\ell_i,\qquad
\mathcal{L}_\text{mean}=\frac{1}{m}\sum_{i=1}^{m}\ell_i
$$

Relation:

$$
\nabla\mathcal{L}_\text{sum}=m\,\nabla\mathcal{L}_\text{mean}
$$
- → Switching sum $\to$ mean at fixed LR is equivalent to dividing the LR by $m$.
```

```{attention} Q&A
:class: dropdown
*Sum vs mean?*
- Mean: batch-size invariant → LR transfers across batch sizes. Default.
- Sum: gradient $\propto$ batch size → LR must be retuned whenever the batch size changes.
- ❌ Neither fixes class imbalance — sum vs mean is a **uniform** rescale of every sample's gradient → relative influence is identical. That needs class/sample weighting or resampling.

*Gradient accumulation gotcha?*
- Accumulating $k$ mean-reduced micro-batch losses sums them → effective LR is $k\times$ too large.
- Fix: divide each micro-batch loss by $k$. → [Gradient Accumulation](../dl/train.md#gradient-accumulation)

*Masked / padded sequences?*
- Normalize by #**valid** elements, not by tensor size, or padding silently shrinks the loss.
- Mean-per-token vs mean-per-sequence weights long sequences differently → they are different objectives, not implementation details.

*Does reduction change the optimum?*
- ❌ For a fixed dataset & a scale-free optimizer, sum & mean share the same argmin.
- ✅ In practice: it rescales the gradient relative to a FIXED penalty term, LR, gradient-clipping threshold, and $\epsilon$ in adaptive optimizers.
```

&nbsp;

### Composite Objective
- **What**: Weighted sum of several loss terms.
- **Why**: Multi-task & auxiliary-loss setups mix terms with different units and gradient magnitudes → the largest term dictates the update and the rest are effectively ignored.
- **How**:
    1. Put every term on a comparable scale.
    2. Weight each term.
    3. Tune or learn the weights.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda_t\ge0$: Weight of task $t$.
- Misc:
    - $\mathcal{L}_t$: Loss of task $t$.
    - $T$: #tasks.
    - $\sigma_t$: Learned noise scale for task $t$.

Objective:

$$
\mathcal{L}=\sum_{t=1}^{T}\lambda_t\mathcal{L}_t
$$

Learned weighting via homoscedastic uncertainty:

$$
\mathcal{L}=\sum_{t=1}^{T}\left(\frac{1}{2\sigma_t^2}\mathcal{L}_t+\log\sigma_t\right)
$$
- $\frac{1}{2\sigma_t^2}$: Task weight, learned jointly with the model.
- $\log\sigma_t$: Prevents the degenerate solution $\sigma_t\to\infty$.
- Optimize $\log\sigma_t^2$ instead of $\sigma_t$ for numerical stability.
```

```{attention} Q&A
:class: dropdown
*How to set the weights?*
- Normalize each term by its running magnitude → all terms $O(1)$.
- Grid search on validation (expensive: $T$ dimensions).
- Learn them (uncertainty weighting above).
- Equalize gradient norms w.r.t. the shared trunk.

*Failure modes?*
- One term dominates → other tasks under-trained (negative transfer).
- Terms with conflicting gradients cancel on the shared trunk.
- A term that can be driven to 0 for free (e.g. a collapsed auxiliary head) gets all the weight it wants.

*Why is a fixed weighted sum limited?*
- Sweeping $\lambda$ only traces the **convex hull** of the Pareto front → non-convex regions of the trade-off surface are unreachable at any $\lambda$.

*Is a regularizer just another term?*
- ✅ Mathematically. ❌ Practically — it has no data-fit gradient, so it does not compete for capacity the same way; and its scale interacts with [reduction](#reduction).
```

&nbsp;

## Probabilistic View
### MLE
- **Name**: Maximum Likelihood Estimation
- **What**: Params maximizing the probability of the observed data.
- **Why**: Loss functions look arbitrary until the noise model is named — MLE derives them instead of postulating them.
- **How**:
    1. Assume a conditional density $p(y|\mathbf{x};\theta)$.
    2. Take the log-likelihood over i.i.d. samples.
    3. Negate → NLL = the loss.
    4. Minimize.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\theta$: Model params.
- Misc:
    - $\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^m$: Dataset.
    - $\hat{P}$: Empirical distribution of $\mathcal{D}$.
    - $P_\theta$: Model distribution.

Objective:

$$
\hat{\theta}_\text{MLE}=\arg\max_\theta\sum_{i=1}^{m}\log p(y_i|\mathbf{x}_i;\theta)=\arg\min_\theta\underbrace{-\sum_{i=1}^{m}\log p(y_i|\mathbf{x}_i;\theta)}_{\text{NLL}}
$$

Equivalence:

$$
\arg\max_\theta\frac{1}{m}\sum_{i=1}^{m}\log p(y_i|\mathbf{x}_i;\theta)=\arg\min_\theta D_\text{KL}(\hat{P}\|P_\theta)
$$
```

```{dropdown} Table: Noise Model → Loss
| Assumed $p(y|\mathbf{x})$ | NLL becomes | Estimates |
|:--|:--|:--|
| Gaussian, fixed $\sigma$ | [MSE](#mse) | Conditional mean |
| Gaussian, learned $\sigma(\mathbf{x})$ | $\frac{e^2}{2\sigma^2}+\log\sigma$ (heteroscedastic) | Mean + noise scale |
| Laplace | [MAE](#mae) | Conditional median |
| Asymmetric Laplace | [Quantile](#quantile-loss) | Conditional $\tau$-quantile |
| Bernoulli | [BCE](#bce) | $P(y=1\|\mathbf{x})$ |
| Categorical | [Cross Entropy](#cross-entropy) | $P(y=k\|\mathbf{x})$ |
| Poisson | $\hat{y}-y\log\hat{y}$ | Conditional rate |
| Student-$t$ | Robust NLL (heavy tails) | Conditional location ($=$ mean iff $\nu>1$) |
```

```{attention} Q&A
:class: dropdown
*Properties?*
- **Consistent**: $\hat{\theta}\to\theta^*$ as $m\to\infty$ (well-specified model, regularity conditions).
- **Asymptotically normal & efficient**: attains the Cramér-Rao lower bound asymptotically.
- **Invariant to reparametrization**: $\widehat{g(\theta)}=g(\hat{\theta})$.
- ❌ NOT unbiased in general — e.g. MLE of Gaussian variance is $\frac{m-1}{m}\sigma^2$.

*Why is MLE the same as minimizing KL / cross entropy?*
- $D_\text{KL}(\hat{P}\|P_\theta)=\underbrace{H(\hat{P},P_\theta)}_{\text{avg NLL}}-\underbrace{H(\hat{P})}_{\text{const in }\theta}$ → same argmin.
- Exact for discrete $P_\theta$. For a continuous density $\hat{P}$ is a sum of point masses → $D_\text{KL}$ and $H(\hat{P})$ are ill-defined, but the cross-entropy term is still exactly the average NLL → the argmin statement survives.

*When does MLE break?*
- **Separable data** in logistic regression → $\|\mathbf{w}\|\to\infty$, no finite optimum → needs a penalty.
- **Small $m$ / large capacity** → fits noise.
- **Misspecified model** → converges to the KL-closest wrong model, and the efficiency guarantee is void.
- **Unbounded likelihood** (e.g. a GMM component collapsing onto one point → $\sigma\to0$, likelihood $\to\infty$).

*Why log?*
- Products of $m$ probabilities underflow; sums don't.
- $\log$ is monotone → argmax preserved.
- Turns the i.i.d. product into a sum → decomposable → mini-batch SGD is unbiased.
```

&nbsp;

### MAP
- **Name**: Maximum A Posteriori
- **What**: Params maximizing the posterior.
- **Why**: MLE has no channel for prior belief → nothing stops it from choosing extreme params when data is scarce or the model is separable.
- **How**: Add $-\log p(\theta)$ to the NLL → the prior IS the regularizer.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $p(\theta)$: Prior over params.
    - $\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^m$: Dataset.
    - $\tau^2$: Gaussian prior variance.
    - $b$: Laplace prior scale.
    - $I$: Identity matrix.

Objective:

$$
\hat{\theta}_\text{MAP}=\arg\max_\theta\underbrace{\left[\log p(\mathcal{D}|\theta)+\log p(\theta)\right]}_{\log p(\theta|\mathcal{D})\ +\ \text{const}}
$$

$$
\Leftrightarrow\quad\mathcal{L}(\theta)=\underbrace{-\log p(\mathcal{D}|\theta)}_{\text{loss}}\ \underbrace{-\log p(\theta)}_{\text{regularizer}}
$$

Prior $\to$ penalty:

$$\begin{align*}
\theta\sim N(0,\tau^2I)&\ \Rightarrow\ -\log p(\theta)=\tfrac{1}{2\tau^2}\|\theta\|_2^2+\text{const} &&\to\text{ L2} \\
\theta\sim\text{Laplace}(0,b)&\ \Rightarrow\ -\log p(\theta)=\tfrac{1}{b}\|\theta\|_1+\text{const} &&\to\text{ L1}
\end{align*}$$
```

```{attention} Q&A
:class: dropdown
*MLE vs MAP vs full Bayes?*
- MLE: point estimate, no prior.
- MAP: point estimate, prior as a penalty. Still a single $\theta$ → no uncertainty.
- Bayes: keeps $p(\theta|\mathcal{D})$ and marginalizes → predictive uncertainty, but needs an intractable normalizer.

*Is MAP invariant to reparametrization?*
- ❌ Unlike MLE. A nonlinear change of variables introduces a Jacobian into the prior density → the mode moves.
- → MAP is a property of the parametrization, not just the model.

*What happens as $m\to\infty$?*
- Likelihood term is $O(m)$, prior term is $O(1)$ → MAP $\to$ MLE. The prior only matters in the small-data regime.

*Is every regularizer a prior?*
- ✅ Any $\Omega$ with $\int e^{-\lambda\Omega(\theta)}d\theta<\infty$ defines a proper prior.
- ❌ Not the practically important ones — dropout, early stopping, and data augmentation have no clean prior interpretation.
```

&nbsp;

## Regression
### MSE
- **Name**: Mean Squared Error
- **What**: Mean of squared residuals.
- **Why**: Additive Gaussian noise → NLL $\propto$ squared residual.
- **How**: Residual → square → average.

```{note} Math
:class: dropdown
Forward:

$$
\mathcal{L}=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial\hat{y}_i}=\frac{2}{m}(\hat{y}_i-y_i)
$$

Population minimizer:

$$
\arg\min_{c}\mathbb{E}\left[(y-c)^2\big|\mathbf{x}\right]=\mathbb{E}[y|\mathbf{x}]
$$
```

```{tip} Derivation
:class: dropdown
*Where does the square come from?*
1. Noise model: $y_i=\hat{y}_i+\varepsilon_i$, $\varepsilon_i\overset{iid}{\sim}N(0,\sigma^2)$ → $y_i\sim N(\hat{y}_i,\sigma^2)$.
2. Log-likelihood:

    $$
    \log L=-\frac{m}{2}\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
    $$

3. Only the last term depends on $\hat{y}$ → $\arg\max\log L=\arg\min\sum_i(y_i-\hat{y}_i)^2$.
4. → **MSE = MLE under Gaussian noise with fixed $\sigma$**.

*Why the conditional mean?*
1. $\mathbb{E}[(y-c)^2|\mathbf{x}]=\text{Var}[y|\mathbf{x}]+(\mathbb{E}[y|\mathbf{x}]-c)^2$.
2. First term is free of $c$ → minimized at $c=\mathbb{E}[y|\mathbf{x}]$.
```

````{important} Code
:class: dropdown
```python
import torch

def mse(y, yhat):
    return (y - yhat).pow(2).mean()

def mae(y, yhat):
    return (y - yhat).abs().mean()

## Example: sample 4 is a 10x outlier -> compare how much pull it exerts
y = torch.tensor([1.0, 2.0, 3.0, 40.0])
for fn in (mse, mae):
    yhat = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    fn(y, yhat).backward()
    print(fn.__name__, fn(y, yhat).item(), yhat.grad)

## mse 324.0 tensor([ -0.,  -0.,  -0., -18.])          <- gradient scales with the error
## mae   9.0 tensor([-0., -0., -0., -0.2500])          <- gradient capped at 1/m
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Smooth & convex in $\hat{y}$ → unique minimizer, closed form for linear models.
- Gradient $\propto$ error → large errors get corrected fast.

*Cons?*
- Outlier-dominated ← a $10\times$ residual contributes $100\times$ the loss and $10\times$ the gradient.
- Units are the target's **squared** → the loss scale grows quadratically with target scale → LR, penalty weight $\lambda$ & clipping thresholds all need retuning when the target is rescaled.
- Wrong NLL under heavy-tailed noise.

*What does the minimizer estimate?*
- The conditional **mean** $\mathbb{E}[y|\mathbf{x}]$. → skewed $p(y|\mathbf{x})$ makes MSE predict a value no sample looks like.

*Why NOT MSE for classification?*
1. Wrong likelihood — labels are Bernoulli/Categorical, not Gaussian.
2. Saturation: $\frac{\partial}{\partial z}\frac{1}{2}(\sigma(z)-y)^2=(\sigma(z)-y)\sigma'(z)$, and $\sigma'(z)\to0$ when $|z|$ is large → a confidently WRONG prediction produces a near-zero gradient. Cross entropy's $\hat{p}-y$ has no such factor.
3. MSE composed with a sigmoid is non-convex in $\mathbf{w}$; BCE is convex.

*Does the $\frac{1}{m}$ matter?*
- Only through the gradient scale → see [Reduction](#reduction).
```

&nbsp;

### MAE
- **Name**: Mean Absolute Error
- **What**: Mean of absolute residuals.
- **Why**: A single outlier dominates MSE ← its contribution grows as $e^2$.
- **How**: Residual → absolute value → average.

```{note} Math
:class: dropdown
Forward:

$$
\mathcal{L}=\frac{1}{m}\sum_{i=1}^{m}|y_i-\hat{y}_i|
$$

Backward (subgradient at $0$):

$$
\frac{\partial\mathcal{L}}{\partial\hat{y}_i}=\frac{1}{m}\text{sign}(\hat{y}_i-y_i)
$$

Population minimizer:

$$
\arg\min_{c}\mathbb{E}\left[|y-c|\big|\mathbf{x}\right]=\text{median}(y|\mathbf{x})
$$
```

```{tip} Derivation
:class: dropdown
*Where does the absolute value come from?*
1. Noise model: $\varepsilon_i\overset{iid}{\sim}\text{Laplace}(0,b)$ → $p(y_i|\hat{y}_i)=\frac{1}{2b}\exp\left(-\frac{|y_i-\hat{y}_i|}{b}\right)$.
2. Log-likelihood: $\log L=-m\log(2b)-\frac{1}{b}\sum_{i=1}^{m}|y_i-\hat{y}_i|$.
3. → **MAE = MLE under Laplace noise**.

*Why the conditional median?*
1. $\frac{\partial}{\partial c}\mathbb{E}|y-c|=P(y<c)-P(y>c)$.
2. Set to $0$ → $P(y<c)=P(y>c)=\frac{1}{2}$ → $c$ = median.
    - Continuous $y$. With atoms the stationarity condition becomes an inclusion $0\in[P(y<c)-P(y>c)]$ → a median **interval**.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Robust ← every sample contributes gradient magnitude $\frac{1}{m}$ regardless of error size.
- Gradient magnitude is **scale-invariant** ($\frac{1}{m}$ regardless of target scale) → rescaling the target does NOT force an LR retune, unlike MSE.

*Cons?*
- Non-differentiable at $0$ → subgradient; no closed form.
- **Constant** gradient magnitude → the step size doesn't shrink near the optimum → oscillation unless the LR decays.
- Minimizer can be non-unique (any point between the two middle residuals when $m$ is even).

*What does the minimizer estimate?*
- The conditional **median** → robust, but blind to the tail of $p(y|\mathbf{x})$.

*MAE or MSE?*
- Outliers are corrupt data → MAE.
- Outliers are real & expensive → MSE.
- Both → [Huber](#huber-loss).
```

&nbsp;

### Huber Loss
- **What**: Quadratic within $\delta$ of zero, linear outside.
- **Why**: MSE lets outliers dominate; MAE has a kink at $0$ and a gradient that never shrinks near the optimum.
- **How**:
    1. Threshold the residual at $\delta$.
    2. Below → squared error.
    3. Above → linear, matched to the quadratic in both value & slope at $|e|=\delta$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\delta>0$: Transition threshold.
- Misc:
    - $e_i=y_i-\hat{y}_i$: Residual.

Forward:

$$
\ell_\delta(e)=\begin{cases}
\frac{1}{2}e^2 & |e|\le\delta \\
\delta\left(|e|-\frac{1}{2}\delta\right) & |e|>\delta
\end{cases}
\qquad
\mathcal{L}=\frac{1}{m}\sum_{i=1}^{m}\ell_\delta(e_i)
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial\hat{y}_i}=\frac{1}{m}\begin{cases}
\hat{y}_i-y_i & |e_i|\le\delta \\
\delta\,\text{sign}(\hat{y}_i-y_i) & |e_i|>\delta
\end{cases}
$$
- → Gradient magnitude is **capped at $\frac{\delta}{m}$**.

Limits: $\delta\to\infty$ → $\frac{1}{2}$MSE (the quadratic branch swallows every finite residual). $\delta\to0$ → $\delta\cdot$MAE.
```

```{tip} Derivation
:class: dropdown
*Why that specific linear piece?*
1. Require **continuity** at $|e|=\delta$: quadratic gives $\frac{1}{2}\delta^2$.
2. Require **matching slope** at $|e|=\delta$: quadratic gives $\delta$ → the line must be $\delta|e|+c$.
3. Solve $\delta\cdot\delta+c=\frac{1}{2}\delta^2$ → $c=-\frac{1}{2}\delta^2$.
4. → $\ell=\delta\left(|e|-\frac{1}{2}\delta\right)$, which is $C^1$ everywhere (but NOT $C^2$ — $\ell''$ jumps from $1$ to $0$ at $|e|=\delta$).
```

````{important} Code
:class: dropdown
```python
import torch

class HuberLoss:
    def __init__(self, delta=1.0):
        self.delta = delta

    def __call__(self, y, yhat):
        e = y - yhat
        a = e.abs()
        quad = 0.5 * e.pow(2)
        ## the -0.5*delta offset is what makes the two pieces meet in value
        lin = self.delta * (a - 0.5 * self.delta)
        return torch.where(a <= self.delta, quad, lin).mean()

## Example: same 10x outlier as in the MSE block
y = torch.tensor([1.0, 2.0, 3.0, 40.0])
yhat = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(HuberLoss(delta=1.0)(y, yhat).item())  ## 8.875  (MSE: 324.0, MAE: 9.0)
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Robust (linear tail) AND smooth at $0$ (quadratic core) → bounded gradient, no oscillation near the optimum.
- One knob interpolates the whole MSE↔MAE family.

*Cons?*
- $\delta$ is scale-dependent → must be re-tuned whenever the target is rescaled.
- Only $C^1$ → second-order methods see a discontinuous Hessian at $|e|=\delta$.
- Minimizer is neither the mean nor the median → a $\delta$-dependent M-estimator, harder to state.

*How to pick $\delta$?*
- Standardize the target, then $\delta\approx1$.
- Or set it from the residual distribution: $\delta=1.345\hat{\sigma}$ gives ~95% asymptotic efficiency relative to OLS when the noise really is Gaussian.
- Or use a residual quantile (e.g. the 90th percentile) → the top 10% of residuals are treated as outliers.

*Why not just clip the gradient instead?*
- Gradient clipping caps the UPDATE, not the objective → the thing being minimized is no longer well-defined. Huber caps the gradient by construction while remaining a genuine loss.
```

&nbsp;

#### Smooth L1
- **What**: Huber rescaled so the linear region has slope 1.
- **Why**: Huber's loss magnitude & gradient cap both scale with $\delta$ → changing $\delta$ silently changes the effective LR.
- **How**: Divide Huber by $\delta$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta>0$: Transition threshold.

Forward:

$$
\ell_\beta(e)=\begin{cases}
\frac{1}{2\beta}e^2 & |e|<\beta \\
|e|-\frac{1}{2}\beta & \text{otherwise}
\end{cases}
$$

Relation:

$$
\ell^\text{Huber}_\delta(e)=\delta\cdot\ell^\text{SmoothL1}_{\beta=\delta}(e)
$$
```

```{attention} Q&A
:class: dropdown
*Difference from Huber in one line?*
- Identical shape; Smooth L1 caps the gradient at $1$ instead of $\delta$, so $\beta$ changes only the transition point, not the loss scale.

*Which one do frameworks give you?*
- `nn.SmoothL1Loss(beta=...)` and `nn.HuberLoss(delta=...)` differ by exactly the factor $\delta$.
- $\beta=\delta=1$ → the two coincide.
- $\beta\to0$ → L1; PyTorch's Smooth L1 converges to L1 as $\beta\to0$ whereas Huber converges to the constant $0$.
```

&nbsp;

### Log-Cosh
- **What**: $\log\cosh$ of the residual.
- **Why**: Huber is only $C^1$ and needs a threshold; second-order solvers want a Hessian that exists everywhere.
- **How**: Apply $\log\cosh$, which is $\approx\frac{e^2}{2}$ near $0$ and $\approx|e|-\log2$ far from it.

```{note} Math
:class: dropdown
Forward:

$$
\mathcal{L}=\frac{1}{m}\sum_{i=1}^{m}\log\cosh(e_i)
$$

Backward:

$$
\frac{\partial\ell}{\partial\hat{y}_i}=-\tanh(e_i),\qquad
\frac{\partial^2\ell}{\partial\hat{y}_i^2}=\text{sech}^2(e_i)=1-\tanh^2(e_i)
$$
- Gradient magnitude bounded by $1$; Hessian is continuous & strictly positive everywhere.

Asymptotics:

$$
\log\cosh(e)\approx\begin{cases}
\frac{e^2}{2} & |e|\to0 \\
|e|-\log2 & |e|\to\infty
\end{cases}
$$

Overflow-safe form ($\cosh$ overflows for $|e|\gtrsim710$ in float64):

$$
\log\cosh(e)=|e|+\log\left(1+e^{-2|e|}\right)-\log2
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- $C^\infty$ → clean Hessian for Newton / second-order boosting.
- Robust like Huber, with NO threshold to tune.

*Cons?*
- Naive $\log(\cosh(e))$ overflows → needs the stable form above.
- Robustness is fixed — no knob to trade sensitivity vs robustness the way $\delta$ does.
- Its noise model is the hyperbolic secant $p(e)=\frac{1}{\pi\cosh e}$ — a real NLL, but an unfamiliar one → no Gaussian/Laplace-style intuition to lean on.

*Log-Cosh vs Huber?*
- Log-Cosh $\approx$ Huber with $\delta=1$, minus the kink in the second derivative.
- → Use Log-Cosh when the target is standardized and you want to skip tuning $\delta$.
```

&nbsp;

### Quantile Loss
- **What**: Absolute residual weighted asymmetrically by $\tau$ vs $1-\tau$. Also called **pinball loss**.
- **Why**: MSE/MAE return only the conditional mean/median → no prediction intervals, and no way to say "under-predicting costs more than over-predicting".
- **How**:
    1. Compute the residual $e=y-\hat{y}$.
    2. Under-prediction ($e>0$) → charge $\tau|e|$.
    3. Over-prediction ($e<0$) → charge $(1-\tau)|e|$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\tau\in(0,1)$: Target quantile.

Forward:

$$
\ell_\tau(e)=\max\left(\tau e,\ (\tau-1)e\right)=\begin{cases}
\tau e & e\ge0 \\
(1-\tau)|e| & e<0
\end{cases}
$$

Backward:

$$
\frac{\partial\ell_\tau}{\partial\hat{y}}=\begin{cases}
-\tau & e>0 \\
1-\tau & e<0
\end{cases}
$$

Population minimizer:

$$
\arg\min_c\mathbb{E}\left[\ell_\tau(y-c)\big|\mathbf{x}\right]=Q_\tau(y|\mathbf{x})
$$
- $\tau=0.5$ → $\ell=\frac{1}{2}|e|$ → MAE (up to the factor $\frac{1}{2}$) → median.
```

````{important} Code
:class: dropdown
```python
import torch

class QuantileLoss:
    def __init__(self, tau=0.5):
        self.tau = tau

    def __call__(self, y, yhat):
        e = y - yhat
        ## max() picks tau*e when e>0 and (1-tau)*|e| when e<0 -- no branching needed
        return torch.maximum(self.tau * e, (self.tau - 1) * e).mean()

## Example: tau=0.9 -> under-prediction costs 9x more than over-prediction
y = torch.tensor([0.0, 0.0])
yhat = torch.tensor([-1.0, 1.0])          ## sample 1 under-predicts, sample 2 over-predicts
print(QuantileLoss(0.9)(y, yhat).item())  ## 0.5  == (0.9 + 0.1) / 2
```
````

```{attention} Q&A
:class: dropdown
*What is it actually used for?*
- Prediction intervals: fit $\tau=0.05$ and $\tau=0.95$ → an 90% interval, with NO Gaussian assumption.
- Asymmetric business cost (stock-outs vs overstock, under-forecasting demand).

*Cons?*
- One output head per quantile → $Q$ heads for $Q$ levels (one shared trunk suffices; separate models are not required).
- **Quantile crossing**: independently fit quantiles can violate $\hat{Q}_{0.05}\le\hat{Q}_{0.95}$.
- Non-differentiable at $e=0$; gradient magnitude is constant → same LR-decay issue as MAE.

*Why does it recover the quantile?*
- At the optimum the subgradient vanishes: $-\tau P(e>0)+(1-\tau)P(e<0)=0$ → $P(y\le\hat{y})=\tau$.

*Relation to expectiles?*
- Replace $|e|$ with $e^2$ in the same asymmetric weighting → expectile regression: smooth & differentiable, but the estimand is no longer a quantile.
```

```{dropdown} Table: Regression Losses
| | $\ell(e)$ | Estimates | Grad magnitude | Outlier-robust | Smooth at 0 | Hyperparam |
|:--|:--|:--|:--|:--|:--|:--|
| MSE | $e^2$ | Mean | $\propto\lvert e\rvert$ | ❌ | ✅ | — |
| MAE | $\lvert e\rvert$ | Median | $1$ | ✅ | ❌ | — |
| Huber | piecewise | M-estimate | $\le\delta$ | ✅ | ✅ | $\delta$ |
| Smooth L1 | piecewise | M-estimate | $\le1$ | ✅ | ✅ | $\beta$ |
| Log-Cosh | $\log\cosh e$ | $\approx$ Huber($\delta{=}1$) | $\le1$ | ✅ | ✅ | — |
| Quantile | $\max(\tau e,(\tau{-}1)e)$ | $Q_\tau$ | $\tau$ or $1-\tau$ | ✅ | ❌ | $\tau$ |

- $e=y-\hat{y}$: Residual.
- "Grad magnitude" is $\left\lvert\frac{\partial\ell}{\partial\hat{y}}\right\rvert$ before reduction.
```

&nbsp;

## Classification
### 0-1 Loss
- **What**: Indicator of a wrong label.
- **Why**: The objective every classifier actually wants to minimize — every practical classification loss is a stand-in for it.
- **How**: 1 if the predicted label differs from the true one, else 0.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $y\in\{-1,+1\}$: Label.
- Misc:
    - $f(\mathbf{x})$: Score; predicted label is $\text{sign}(f(\mathbf{x}))$.

Forward:

$$
\ell_{0/1}=\mathbb{1}\left[\text{sign}(f(\mathbf{x}))\neq y\right]=\mathbb{1}\left[yf(\mathbf{x})\le0\right]
$$

Risk:

$$
R_{0/1}(f)=P\left(yf(\mathbf{x})\le0\right)
$$

Bayes-optimal rule:

$$
f^*(\mathbf{x})=\arg\max_k P(y=k|\mathbf{x})
$$

Cost-sensitive generalization ($C_{k,k'}$ = cost of predicting $k$ when the truth is $k'$):

$$
f^*(\mathbf{x})=\arg\min_k\sum_{k'=1}^{K}C_{k,k'}P(y=k'|\mathbf{x})
$$
```

```{attention} Q&A
:class: dropdown
*Why is it never minimized directly?*
- Piecewise constant → $\nabla=0$ a.e., undefined at the boundary.
- Non-convex → no optimization guarantees.
- Minimizing it exactly over linear classifiers is NP-hard.
- → [Surrogate Loss](#surrogate-loss).

*Where does it still appear inside an objective?*
- AdaBoost's weighted error $\epsilon_t=\sum_iw_i\mathbb{1}[y_i\neq h_t(\mathbf{x}_i)]$, which sets the stump weight $\alpha_t$.
- Decision theory: the Bayes classifier is by definition the 0-1 risk minimizer → it defines what every surrogate is trying to reach.

*What does it throw away?*
- Confidence. A prediction at $\hat{p}=0.51$ and one at $\hat{p}=0.99$ cost the same → no gradient signal to sharpen a barely-correct prediction.
- Symmetric by default → FP and FN priced identically unless generalized to a cost matrix.

*Is it robust?*
- ✅ Extremely — an outlier can cost at most $\frac{1}{m}$. That robustness is exactly what every convex surrogate gives up in exchange for a usable gradient.
```

&nbsp;

### Cross Entropy
- **What**: Negative log-probability assigned to the true class.
- **Why**: Categorical likelihood → NLL; and the 0-1 loss it replaces is unoptimizable.
- **How**:
    1. Logits → softmax → probabilities.
    2. Take the probability of the true class.
    3. $-\log$ it.
    4. Average.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $y_{ik}\in[0,1]$: Target probability for sample $i$, class $k$. One-hot in the standard case, $\sum_ky_{ik}=1$.
- Misc:
    - $z_{ik}$: Logit for sample $i$, class $k$.
    - $\hat{p}_{ik}=\frac{e^{z_{ik}}}{\sum_{c=1}^{K}e^{z_{ic}}}$: Softmax probability.

Forward:

$$
\mathcal{L}=-\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}y_{ik}\log\hat{p}_{ik}
$$

Logit form (never form $\hat{p}$ explicitly):

$$
-\log\hat{p}_{ik}=\log\sum_{c=1}^{K}e^{z_{ic}}-z_{ik}
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial\hat{p}_{ik}}=-\frac{1}{m}\frac{y_{ik}}{\hat{p}_{ik}},
\qquad
\frac{\partial\mathcal{L}}{\partial z_{ik}}=\frac{1}{m}\left(\hat{p}_{ik}-y_{ik}\right)
$$

Log-sum-exp (numerically safe, $M_i=\max_c z_{ic}$):

$$
\log\sum_{c}e^{z_{ic}}=M_i+\log\sum_{c}e^{z_{ic}-M_i}
$$
```

```{tip} Derivation
:class: dropdown
*Where does the formula come from?*
1. $y_i|\mathbf{x}_i\sim\text{Categorical}(\hat{\mathbf{p}}_i)$ → $p(y_i|\mathbf{x}_i)=\prod_{k=1}^{K}\hat{p}_{ik}^{y_{ik}}$ (only the true class's factor survives ← $y_{ik}\in\{0,1\}$).
2. Log-likelihood: $\log L=\sum_{i=1}^{m}\sum_{k=1}^{K}y_{ik}\log\hat{p}_{ik}$.
3. Negate → **CE = NLL of a Categorical likelihood**.

*Where does the clean $\hat{p}-y$ gradient come from?*
1. Softmax Jacobian:

    $$
    \frac{\partial\hat{p}_c}{\partial z_k}=\hat{p}_c(\delta_{ck}-\hat{p}_k)
    $$

2. Chain:

    $$\begin{align*}
    \frac{\partial\ell}{\partial z_k}&=-\sum_{c=1}^{K}\frac{y_c}{\hat{p}_c}\cdot\hat{p}_c(\delta_{ck}-\hat{p}_k) \\
    &=-\sum_{c=1}^{K}y_c(\delta_{ck}-\hat{p}_k) \\
    &=-y_k+\hat{p}_k\underbrace{\sum_{c=1}^{K}y_c}_{=1}=\hat{p}_k-y_k
    \end{align*}$$

3. → The softmax derivative **cancels** the $\frac{1}{\hat{p}}$ from the log → no saturation, unlike MSE + sigmoid.
```

````{important} Code
:class: dropdown
```python
import torch

class CrossEntropy:
    def __call__(self, z, y):
        ## log-sum-exp: subtract the row max so exp() can never overflow
        M = z.max(dim=-1, keepdim=True).values
        logZ = M + (z - M).exp().sum(dim=-1, keepdim=True).log()
        logp = z - logZ                                      ## log-softmax, computed in ONE step
        ## pick the log-prob of the true class per row
        return -logp.gather(1, y[:, None]).squeeze(1).mean()

## Example
z = torch.tensor([[2.0, 1.0, 0.1], [0.5, 3.0, 0.2]])
y = torch.tensor([0, 1])
print(CrossEntropy()(z, y).item())                           ## 0.2753
print(torch.nn.functional.cross_entropy(z, y).item())        ## 0.2753 -- same
```
````

```{attention} Q&A
:class: dropdown
*Why CE instead of MSE for classification?*
1. Correct NLL — labels are Categorical, not Gaussian.
2. $\frac{\partial\ell}{\partial z}=\hat{p}-y$ has NO $\sigma'(z)$ factor → a confidently-wrong prediction still produces a large gradient. MSE + sigmoid saturates exactly there.
3. CE + softmax is convex in the logits; MSE + softmax is not.

*Numerical stability?*
- ❌ `log(softmax(z))` — softmax underflows to 0 → $\log0=-\infty$.
- ✅ Feed **logits** to a fused op: `nn.CrossEntropyLoss`, `nn.BCEWithLogitsLoss`.

*CE vs NLL vs BCE?*
- Same object, different entry point: `CrossEntropyLoss` = log-softmax + `NLLLoss`; `NLLLoss` expects log-probabilities; `BCEWithLogitsLoss` = sigmoid + BCE per element.

*Relation to KL?*
- $H(P,\hat{P})=H(P)+D_\text{KL}(P\|\hat{P})$.
- One-hot targets → $H(P)=0$ → CE $=$ KL exactly. Soft targets → they differ by a constant, same gradient.

*Is CE bounded?*
- ❌ $-\log\hat{p}\to\infty$ as $\hat{p}\to0$ → ONE confidently-wrong (or mislabeled) sample can dominate the batch → CE is not robust to label noise.

*Behavior under class imbalance?*
- The sum is dominated by the majority class → the model learns the prior and stops. → class weights, resampling, or [Focal Loss](#focal-loss).

*Is CE a proper scoring rule?*
- ✅ Its expectation is uniquely minimized at $\hat{p}=P(y|\mathbf{x})$ → optimizing it targets **calibrated** probabilities, not just correct argmax.

*Multi-class vs multi-label?*
- Mutually exclusive → softmax + CE (probabilities sum to 1).
- Independent labels → per-label sigmoid + BCE ($K$ independent Bernoullis).
```

&nbsp;

#### BCE
- **Name**: Binary Cross Entropy
- **What**: Cross entropy for a single Bernoulli output.
- **Why**: $K=2$ makes the softmax over-parameterized ← only the logit **difference** matters.
- **How**: Sigmoid the single logit → $-\log$ of the probability assigned to the observed label.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $y_i\in\{0,1\}$: Label.
- Misc:
    - $z_i$: Single logit.
    - $\hat{p}_i=\sigma(z_i)=\frac{1}{1+e^{-z_i}}$.

Forward:

$$
\mathcal{L}=-\frac{1}{m}\sum_{i=1}^{m}\left[y_i\log\hat{p}_i+(1-y_i)\log(1-\hat{p}_i)\right]
$$

Logit form:

$$
\ell_i=\log\left(1+e^{z_i}\right)-y_iz_i
$$

Stable form (what `BCEWithLogitsLoss` computes):

$$
\ell_i=\max(z_i,0)-y_iz_i+\log\left(1+e^{-|z_i|}\right)
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z_i}=\frac{1}{m}\left(\hat{p}_i-y_i\right)
$$
```

```{attention} Q&A
:class: dropdown
*BCE vs 2-class softmax CE?*
- Identical function class. Softmax with $K=2$ has one redundant degree of freedom ($z_1-z_0$ is all that matters) → BCE just fixes the gauge.

*When is BCE used with $K>2$?*
- **Multi-label**: $K$ independent sigmoids, one BCE per label, summed → labels are no longer mutually exclusive.

*How to handle imbalance?*
- `pos_weight` $w$ multiplies the positive term: $-\left[wy\log\hat{p}+(1-y)\log(1-\hat{p})\right]$, typically $w=\frac{\#\text{neg}}{\#\text{pos}}$.
- It rebalances the gradient but also **decalibrates** the output probabilities → recalibrate before using them as probabilities.

*Why is the stable form written with $\max(z,0)$ and $|z|$?*
- $\log(1+e^{z})$ overflows for large $z$. Factoring out $e^{\max(z,0)}$ leaves $\log(1+e^{-|z|})$, whose argument is always in $(1,2]$.
```

&nbsp;

#### Label Smoothing
- **What**: One-hot target replaced by a mixture with the uniform distribution.
- **Why**: A one-hot target is unreachable by a softmax → CE keeps pushing the true logit up forever → $\|\mathbf{w}\|$⬆️, overconfidence, poor calibration.
- **How**:
    1. Take mass $\epsilon$ off the true class.
    2. Spread it uniformly over all $K$ classes.
    3. Run ordinary CE against the smoothed target.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\epsilon\in[0,1)$: Smoothing strength. Typically $0.1$.
- Misc:
    - $u_k=\frac{1}{K}$: Uniform distribution over classes.

Target:

$$
y_k^\text{LS}=(1-\epsilon)y_k+\frac{\epsilon}{K}
$$

Objective (decomposes into two CE terms):

$$
\mathcal{L}_\text{LS}=(1-\epsilon)\underbrace{H(y,\hat{p})}_{\text{ordinary CE}}+\epsilon\underbrace{H(u,\hat{p})}_{\text{pull toward uniform}}
$$

Optimum (finite, unlike plain CE):

$$
\hat{p}_\text{true}=1-\epsilon+\frac{\epsilon}{K},\qquad
z_\text{true}-z_\text{other}=\log\frac{K(1-\epsilon)+\epsilon}{\epsilon}
$$
- $\epsilon=0$ → gap $\to\infty$ → logits diverge.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

class LabelSmoothingCE:
    def __init__(self, eps=0.1):
        self.eps = eps

    def __call__(self, z, y):
        K = z.size(-1)
        logp = F.log_softmax(z, dim=-1)
        nll = -logp.gather(1, y[:, None]).squeeze(1)   ## ordinary CE term
        uniform = -logp.mean(dim=-1)                   ## H(u, p_hat), the smoothing term
        return ((1 - self.eps) * nll + self.eps * uniform).mean()

## Example
z = torch.tensor([[2.0, 1.0, 0.1], [0.5, 3.0, 0.2]])
y = torch.tensor([0, 1])
print(LabelSmoothingCE(0.1)(z, y).item())                 ## 0.4120
print(F.cross_entropy(z, y, label_smoothing=0.1).item())  ## 0.4120 -- same
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Bounded optimal logit gap → ⬇️overconfidence, ⬆️calibration.
- Acts as a regularizer on the output distribution → small but consistent accuracy gains on large-$K$ problems.
- Softens the effect of mislabeled data ← the true class no longer demands $\hat{p}\to1$.

*Cons?*
- Distorts the probabilities → they are deliberately NOT the true posteriors any more.
- Erases the relative similarity structure among the wrong classes (it forces them all to the same value) → measurably hurts **knowledge distillation** from a smoothed teacher.
- $\epsilon$ is another hyperparam; too large → underconfident and slower to fit.

*How does it differ from a confidence penalty?*
- Label smoothing $\approx$ adding $\epsilon\,H(u,\hat{p})$, i.e. a CE toward uniform.
- A confidence penalty adds $-\beta H(\hat{p})$, i.e. the reverse direction (an entropy bonus). Similar effect, different divergence.

*Interaction with temperature / distillation?*
- Both flatten the target. Stacking smoothing on top of a soft teacher target double-counts the flattening → usually pick one.
```

&nbsp;

### Focal Loss
- **What**: Cross entropy scaled down by $(1-\hat{p}_t)^\gamma$.
- **Why**: Under extreme imbalance the many easy negatives, each with a small CE, still SUM to more gradient than the few hard positives.
- **How**:
    1. Compute $\hat{p}_t$, the probability given to the true class.
    2. Multiply CE by the modulating factor $(1-\hat{p}_t)^\gamma$ → easy examples ($\hat{p}_t\to1$) vanish.
    3. Optionally weight by class frequency $\alpha_t$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\gamma\ge0$: Focusing param. $\gamma=2$ works best in the original experiments.
    - $\alpha\in[0,1]$: Class weight for the positive class. $\alpha=0.25$ pairs with $\gamma=2$.
- Misc:
    - $\hat{p}_t=\begin{cases}\hat{p} & y=1\\ 1-\hat{p} & y=0\end{cases}$: Probability of the TRUE class.
    - $\alpha_t$: $\alpha$ if $y=1$, else $1-\alpha$.

Forward:

$$
\ell=-\alpha_t\left(1-\hat{p}_t\right)^\gamma\log\hat{p}_t
$$

Special case: $\gamma=0$ → $\alpha$-weighted CE.

Down-weighting at $\gamma=2$:

$$
\hat{p}_t=0.9\ \Rightarrow\ 100\times\text{ smaller than CE},\qquad
\hat{p}_t\approx0.968\ \Rightarrow\ 1000\times\text{ smaller}
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

class FocalLoss:
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma, self.alpha = gamma, alpha

    def __call__(self, z, y):
        y = y.float()
        p = torch.sigmoid(z)
        ## p_t and alpha_t written arithmetically to avoid any dtype branching
        p_t = y * p + (1 - y) * (1 - p)
        a_t = y * self.alpha + (1 - y) * (1 - self.alpha)
        ce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
        ## (1 - p_t)^gamma: ~0 for easy examples, ~1 for hard ones
        return (a_t * (1 - p_t).pow(self.gamma) * ce).mean()

## Example: an easy negative and a hard positive
z = torch.tensor([-4.0, 0.2])
y = torch.tensor([0, 1])
print(FocalLoss()(z, y).item())  ## 0.0152 -- the easy negative contributes ~4e-6 of it
```
````

```{attention} Q&A
:class: dropdown
*$\alpha$ vs $\gamma$ — why both?*
- $\alpha$ rebalances by **class frequency** (static, per class).
- $\gamma$ rebalances by **example difficulty** (dynamic, per sample, changes during training).
- They are orthogonal; $\alpha$ alone does not stop easy negatives from swamping the sum.

*What happens as $\gamma$⬆️?*
- Easy examples are suppressed harder → gradient concentrates on the decision boundary.
- Too large → almost everything is ignored → training stalls / becomes noisy.

*Cons?*
- 2 hyperparams, and the best $\alpha$ moves when $\gamma$ moves (larger $\gamma$ → smaller $\alpha$).
- Down-weights easy-but-informative examples too — it cannot tell "easy" from "already correct and still worth reinforcing".
- Degrades probability calibration (it is not a proper scoring rule).
- Training is unstable at initialization ← a default init makes $\hat{p}\approx0.5$ everywhere, so the enormous background set generates a huge first-step loss. Fix: set the final-layer bias to $b=-\log\frac{1-\pi}{\pi}$ so the initial foreground probability is $\pi$ (RetinaNet uses $0.01$; the value is task-specific, not universal).

*Focal loss vs resampling vs class weights?*
- Resampling changes the data distribution → risks discarding data or overfitting duplicated minorities.
- Class weights are static → still swamped by many easy negatives.
- Focal is per-example & dynamic → best when the imbalance is between *easy* and *hard*, not just between classes.
```

&nbsp;

### Hinge Loss
- **What**: Linear penalty on margins below 1, zero above.
- **Why**: Log loss keeps pushing on points that are already correct and far from the boundary → the solution depends on every sample.
- **How**: Charge $1-u$ while the margin $u=yf(\mathbf{x})$ is under 1; charge nothing once it clears.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $y\in\{-1,+1\}$: Label.
- Misc:
    - $u=yf(\mathbf{x})$: Margin.

Forward:

$$
\ell_\text{hinge}(u)=\max(0,1-u)
$$

Backward (subgradient w.r.t. $\mathbf{w}$ for $f=\mathbf{x}^T\mathbf{w}$):

$$
\frac{\partial\ell}{\partial\mathbf{w}}=\begin{cases}
-y\mathbf{x} & u<1 \\
0 & u>1
\end{cases}
$$

Multi-class extensions:

$$\begin{align*}
\ell_\text{WW}&=\sum_{k\neq y}\max\left(0,f_k(\mathbf{x})-f_y(\mathbf{x})+1\right) &&\text{(Weston-Watkins)}\\
\ell_\text{CS}&=\max\left(0,\max_{k\neq y}f_k(\mathbf{x})-f_y(\mathbf{x})+1\right) &&\text{(Crammer-Singer)}
\end{align*}$$
```

```{attention} Q&A
:class: dropdown
*Hinge vs log loss?*
- Hinge: exactly 0 past the margin → only violators & boundary points affect $\mathbf{w}$ → **sparse support**. Non-differentiable at the kink. ❌probabilities.
- Log: never 0 → every sample keeps contributing, forever. Smooth. ✅probabilities.

*Why is the margin fixed at 1?*
- $f$ and the margin can be scaled together without changing the classifier → the scale is absorbed into $\|\mathbf{w}\|$. Fixing the margin to 1 removes the redundancy; the L2 penalty then controls the geometric margin $\frac{1}{\|\mathbf{w}\|}$.

*Relation to SVM?*
- Soft-margin SVM = hinge + L2, with $C\approx\frac{1}{2\lambda}$. → [SVM](supervised.md#svm)

*Is hinge classification-calibrated?*
- ✅ Its population minimizer has the sign of the Bayes rule.
- ❌ But it estimates $\text{sign}(2\eta(\mathbf{x})-1)$, not $\eta(\mathbf{x})=P(y=1|\mathbf{x})$ → no probabilities without a post-hoc fit (Platt scaling).

*Robustness?*
- Linear tail → an outlier's gradient is bounded → more robust than exponential loss, comparable to log loss.
```

&nbsp;

#### Squared Hinge
- **What**: Hinge squared.
- **Why**: Hinge's kink at $u=1$ blocks second-order methods and any solver that needs a gradient everywhere.
- **How**: Square the hinge → the kink becomes $C^1$.

```{note} Math
:class: dropdown
Forward:

$$
\ell(u)=\max(0,1-u)^2
$$

Backward:

$$
\frac{\partial\ell}{\partial u}=\begin{cases}
-2(1-u) & u<1 \\
0 & u\ge1
\end{cases}
$$
- Continuous at $u=1$ (both sides $\to0$) → $C^1$, unlike plain hinge.
```

```{attention} Q&A
:class: dropdown
*Trade-off vs plain hinge?*
- ✅ Differentiable everywhere → smooth solvers work; often converges faster.
- ❌ Quadratic in the violation → a badly misclassified point dominates → less robust to label noise & outliers.

*Does it keep sparsity?*
- ✅ Still exactly 0 for $u\ge1$ → support-vector sparsity survives; only the penalty *shape* on violators changes.
```

&nbsp;

### Exponential Loss
- **What**: $e^{-u}$ on the margin.
- **Why**: The objective that makes AdaBoost's multiplicative sample reweighting fall out of plain forward stagewise fitting.
- **How**: Penalize the negative margin exponentially → each round's sample weights are the current per-sample losses.

```{note} Math
:class: dropdown
Forward:

$$
\ell_\text{exp}(u)=e^{-u},\qquad u=yf(\mathbf{x})
$$

Population minimizer:

$$
f^*(\mathbf{x})=\frac{1}{2}\log\frac{P(y=1|\mathbf{x})}{P(y=-1|\mathbf{x})}
$$
- → **Half** the log-odds. Logistic loss gives the full log-odds.
```

```{tip} Derivation
:class: dropdown
*Why half the log-odds?*
1. Let $p=P(y=1|\mathbf{x})$. Conditional risk:

    $$
    \mathbb{E}\left[e^{-yf}\big|\mathbf{x}\right]=pe^{-f}+(1-p)e^{f}
    $$

2. Set the derivative to 0:

    $$
    -pe^{-f}+(1-p)e^{f}=0\ \Rightarrow\ e^{2f}=\frac{p}{1-p}
    $$

3. → $f^*=\frac{1}{2}\log\frac{p}{1-p}$ → recover probabilities via $\hat{p}=\sigma(2f)$.
```

```{attention} Q&A
:class: dropdown
*Why is AdaBoost so sensitive to label noise?*
- $e^{-u}$ grows **without bound** as $u\to-\infty$ → a mislabeled sample's weight explodes round after round → the ensemble spends its capacity on it.
- → [AdaBoost](supervised.md#adaboost). GBDT with log loss or Huber is the standard fix.

*Exponential vs log loss?*
- Same population minimizer up to a factor of 2 → same Bayes-optimal decision.
- Difference is entirely in the tail: $e^{-u}$ vs $\approx-u$ for $u\ll0$ → noise sensitivity, not asymptotic target.

*Why is it used at all then?*
- Closed-form stage weights: minimizing exponential loss at each round gives $\alpha_t=\frac{1}{2}\log\frac{1-\epsilon_t}{\epsilon_t}$ and multiplicative reweighting for free — no line search.
```

&nbsp;

## Similarity & Ranking
### Contrastive Loss
- **What**: Pull matching pairs together, push non-matching pairs apart until a margin.
- **Why**: Classification objectives need a fixed label set → useless when classes are unbounded/unknown at train time (faces, retrieval, dedup) and when the goal is an **embedding space**, not a label.
- **How**:
    1. Embed both items of a pair.
    2. Similar pair → penalize distance directly.
    3. Dissimilar pair → penalize only the distance still below the margin $M$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}_1,\mathbf{x}_2$: Input pair.
    - $Y\in\{0,1\}$: $0$ if similar, $1$ if dissimilar.
- Hyperparams:
    - $M>0$: Margin — the distance beyond which a negative pair is "far enough".
- Misc:
    - $D=\|f(\mathbf{x}_1)-f(\mathbf{x}_2)\|_2$: Embedding distance.

Forward:

$$
\ell=(1-Y)\underbrace{\tfrac{1}{2}D^2}_{\text{pull}}+Y\underbrace{\tfrac{1}{2}\max(0,M-D)^2}_{\text{push}}
$$

Backward:

$$
\frac{\partial\ell}{\partial D}=\begin{cases}
D & Y=0 \\
-(M-D) & Y=1,\ D<M \\
0 & Y=1,\ D\ge M
\end{cases}
$$
- → Negatives already beyond $M$ contribute **nothing**.
```

```{attention} Q&A
:class: dropdown
*Why is the positive term unbounded but the negative term hinged?*
- Positives should collapse to $D=0$ → no reason to stop pulling.
- Negatives only need to be *distinguishable*; without a margin the push term would be unbounded and would blow the embedding scale up forever.

*What does the margin control?*
- $M$ too small → negatives are "satisfied" while still nearly on top of each other → no separation.
- $M$ too large → the push term dominates, the pull term is sacrificed → **embedding collapse** (all embeddings drift to satisfy the margin, positives no longer cluster).

*Cons?*
- Only pairwise → no notion of "closer than", so it enforces absolute distances that must transfer across the whole dataset.
- $M$ is in raw distance units → must be retuned whenever the embedding norm changes. Fixed by L2-normalizing embeddings onto the unit sphere, which bounds $D\in[0,2]$.
- Needs an explicit similar/dissimilar label per pair.
- Most random pairs become trivially easy → the gradient dies unless negatives are mined.
```

&nbsp;

### Triplet Loss
- **What**: Anchor must be closer to its positive than to its negative by at least a margin.
- **Why**: Contrastive loss fixes an **absolute** distance threshold, which cannot be right for every class at once ← intra-class spread varies wildly.
- **How**:
    1. Sample (anchor, positive, negative).
    2. Charge the amount by which $d(a,p)+\alpha$ exceeds $d(a,n)$.
    3. Zero once the ordering holds with slack $\alpha$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}^a,\mathbf{x}^p,\mathbf{x}^n$: Anchor, positive (same class), negative (different class).
- Hyperparams:
    - $\alpha>0$: Margin.
- Misc:
    - $d(\mathbf{u},\mathbf{v})=\|f(\mathbf{u})-f(\mathbf{v})\|_2^2$: Squared embedding distance.

Forward:

$$
\ell=\max\left(0,\ d(\mathbf{x}^a,\mathbf{x}^p)-d(\mathbf{x}^a,\mathbf{x}^n)+\alpha\right)
$$

Constraint being enforced:

$$
d(\mathbf{x}^a,\mathbf{x}^p)+\alpha\le d(\mathbf{x}^a,\mathbf{x}^n)
$$

Triplet categories:

$$\begin{align*}
\text{easy}&:\ d(a,n)>d(a,p)+\alpha &&\to\ \ell=0,\ \text{no gradient} \\
\text{semi-hard}&:\ d(a,p)<d(a,n)<d(a,p)+\alpha &&\to\ \text{correct order, margin violated} \\
\text{hard}&:\ d(a,n)<d(a,p) &&\to\ \text{ordering itself violated}
\end{align*}$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

class TripletLoss:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, a, p, n):
        ## L2-normalize -> distances live in [0, 4], so alpha transfers across runs
        a, p, n = (F.normalize(t, dim=-1) for t in (a, p, n))
        d_ap = (a - p).pow(2).sum(-1)
        d_an = (a - n).pow(2).sum(-1)
        ## relu() is what makes already-satisfied triplets contribute exactly 0
        return F.relu(d_ap - d_an + self.alpha).mean()

## Example: triplet 1 already satisfies the margin, triplet 2 does not
a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
p = torch.tensor([[1.0, 0.1], [0.0, 1.0]])
n = torch.tensor([[0.0, 1.0], [1.0, 0.1]])
print(TripletLoss()(a, p, n))  ## tensor(1.0950) -- entirely from triplet 2
```
````

```{attention} Q&A
:class: dropdown
*Why is it better than contrastive loss?*
- Enforces a **relative** ordering, not an absolute distance → each class keeps its own scale.
- The anchor gives the comparison a shared reference point, so the margin means the same thing everywhere.

*Why does naive training stall?*
- Random triplets are overwhelmingly **easy** → $\ell=0$ → zero gradient → the loss flatlines while nothing is learned.
- → mining is not an optimization; it is required for the objective to have signal at all.

*Why not always mine the HARDEST negatives?*
- The hardest negatives are frequently mislabeled or genuinely ambiguous → they produce huge, wrong gradients early in training → the model **collapses** ($f(\mathbf{x})=\mathbf{0}$ makes every distance 0 and satisfies nothing, but is a strong local attractor).
- → **semi-hard** mining: pick negatives that are farther than the positive but still inside the margin.

*Cons?*
- $O(m^3)$ candidate triplets → mining strategy becomes a core design decision, not a detail.
- Sensitive to batch composition — needs multiple samples per class in each batch (batch-hard mining).
- Slower convergence than a softmax classifier when the label set IS fixed and small.

*Why L2-normalize the embeddings?*
- Scaling every embedding by $c$ turns the loss into $\max(0,c^2(d_{ap}-d_{an})+\alpha)$ → once the ordering is correct, the model can **inflate** $\|f\|$ to clear any fixed $\alpha$ without improving the ordering → $\alpha$ stops meaning anything.
- Normalization pins the scale → $\alpha$ is comparable across runs & datasets, and squared distances live in $[0,4]$.
- On the unit sphere, $\|u-v\|_2^2=2-2u^Tv$ → distance and cosine similarity become interchangeable.
```

&nbsp;

### InfoNCE
- **Name**: Information Noise-Contrastive Estimation
- **What**: Cross entropy over one positive against $N-1$ negatives, on temperature-scaled similarities.
- **Why**: Triplet loss uses ONE negative per update → weak, high-variance signal; and it needs explicit labels.
- **How**:
    1. Build a positive pair (two views of the same item).
    2. Treat every other item in the batch as a negative.
    3. Softmax over similarities → CE with the positive as the "correct class".

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\tau>0$: Temperature.
    - $N$: #candidates (1 positive + $N-1$ negatives).
- Misc:
    - $\mathbf{q}$: Query/anchor embedding.
    - $\mathbf{k}^+$: Positive key.
    - $\mathbf{k}_j$: $j$-th candidate key.
    - $\text{sim}(\mathbf{u},\mathbf{v})=\frac{\mathbf{u}^T\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$: Cosine similarity.

Forward:

$$
\ell=-\log\frac{\exp\left(\text{sim}(\mathbf{q},\mathbf{k}^+)/\tau\right)}{\sum_{j=1}^{N}\exp\left(\text{sim}(\mathbf{q},\mathbf{k}_j)/\tau\right)}
$$
- → Identical in form to [Cross Entropy](#cross-entropy) over $N$ "classes", where the classes are *instances*, not labels.

Mutual information bound:

$$
I(\mathbf{q};\mathbf{k}^+)\ge\log N-\mathcal{L}_\text{InfoNCE}
$$
- Holds when the positive is drawn from the joint $p(\mathbf{q},\mathbf{k})$ and the $N-1$ negatives i.i.d. from the marginal $p(\mathbf{k})$. In-batch negatives violate this → the bound is only approximate in practice.
- → The bound is capped at $\log N$ → more negatives ⬆️ tightens it.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

class InfoNCE:
    def __init__(self, tau=0.07):
        self.tau = tau

    def __call__(self, q, k):
        ## normalize -> the dot product IS cosine similarity
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        logits = q @ k.T / self.tau          ## (B, B): row i vs every key
        ## the positive for row i sits on the diagonal -> label i
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)

## Example: 4 items, each paired with its own (slightly perturbed) view
torch.manual_seed(0)
q = torch.randn(4, 8)
k = q + 0.01 * torch.randn(4, 8)
print(InfoNCE()(q, k).item())  ## 0.0163 -- positives already dominate their rows
```
````

```{attention} Q&A
:class: dropdown
*Where does the name come from?*
- **NCE** ← it discriminates the positive from "noise" samples, exactly as in noise-contrastive estimation.
- **Info** ← the loss lower-bounds the mutual information between anchor and positive.

*What does the temperature do?*
- $\tau$⬇️ → sharper softmax → gradient concentrates on the hardest negatives → stronger separation, but noise-sensitive & can break the embedding's local structure.
- $\tau$⬆️ → flatter → all negatives weighted alike → weaker, more uniform repulsion.
- It is one of the most sensitive hyperparams in contrastive learning.

*Why do more negatives help?*
- The bound is capped at $\log N$ → small $N$ caps how much MI the objective can **certify** (not how much the encoder can actually learn).
- → large batches (SimCLR), memory queues (MoCo), or negative sampling from a bank.

*How does it beat triplet loss?*
- $N-1$ negatives per anchor per step instead of 1 → lower-variance gradient, implicit hard-negative weighting via the softmax (the hardest negatives automatically get the largest weight), and NO mining heuristic.

*Cons?*
- Performance scales with batch size → memory-bound.
- **False negatives**: batch items of the same true class are treated as negatives → the objective actively pushes apart things that should be together.
- $\tau$ requires tuning and interacts with the LR and batch size.
- The MI bound is loose in practice → the number should not be read as an MI estimate.

*Relation to cross entropy?*
- Exactly CE with instance-level "labels". Everything true of CE (log-sum-exp stability, $\hat{p}-y$ gradient) carries over unchanged.
```

&nbsp;

### Cosine Embedding Loss
- **What**: Penalty on cosine similarity — maximize it for similar pairs, drive it below a margin for dissimilar ones.
- **Why**: Euclidean distance conflates **direction** with **magnitude** → in high dimensions the norm often encodes frequency or confidence, not semantics.
- **How**: Similar → charge $1-\cos$. Dissimilar → charge $\cos$ only while it exceeds the margin.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $y\in\{-1,+1\}$: $+1$ if similar, $-1$ if dissimilar.
- Hyperparams:
    - $\text{margin}\in[-1,1]$: Similarity ceiling for dissimilar pairs. Default $0$.

Forward:

$$
\ell=\begin{cases}
1-\cos(\mathbf{x}_1,\mathbf{x}_2) & y=1 \\
\max\left(0,\ \cos(\mathbf{x}_1,\mathbf{x}_2)-\text{margin}\right) & y=-1
\end{cases}
$$

Relation to L2 on the unit sphere:

$$
\|\mathbf{u}-\mathbf{v}\|_2^2=2-2\cos(\mathbf{u},\mathbf{v})\quad\text{when}\quad\|\mathbf{u}\|=\|\mathbf{v}\|=1
$$
- → On normalized embeddings, cosine and squared-L2 objectives are **monotonically equivalent**.
```

```{attention} Q&A
:class: dropdown
*When cosine over Euclidean?*
- Embedding norm is uninformative or nuisance (TF-IDF counts, word vectors, retrieval).
- ❌ When magnitude carries real signal (e.g. regression outputs).

*Why is the default margin 0 and not something larger?*
- $\cos<0$ already means "pointing apart" → for most tasks pushing past orthogonality is unnecessary and over-constrains the space.

*Does it collapse?*
- ✅ Same failure as contrastive loss if the dissimilar term dominates. Bounded range $[-1,1]$ makes it milder than raw distance.
```

&nbsp;

### Pairwise Ranking Loss
- **What**: Penalty on the score gap of a (preferred, non-preferred) pair.
- **Why**: Metric-learning losses enforce *distances*; ranking needs only the **order** of two items — and absolute relevance labels are usually unavailable, while pairwise preferences (clicks, comparisons) are cheap.
- **How**:
    1. Score both items with the same model.
    2. Charge a penalty while the preferred item's score does not exceed the other's.
    3. Hinge form → zero past a margin. Logistic form → never exactly zero.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}^+,\mathbf{x}^-$: Preferred & non-preferred item.
- Hyperparams:
    - $\text{margin}\ge0$: Required score gap (hinge form).
    - $\sigma_0>0$: Slope (logistic form).
- Misc:
    - $s^+=f(\mathbf{x}^+),\ s^-=f(\mathbf{x}^-)$: Scores.
    - $\Delta=s^+-s^-$: Score gap.
    - $\sigma(z)=\frac{1}{1+e^{-z}}$: Sigmoid.

Margin ranking (hinge):

$$
\ell=\max\left(0,\ -\Delta+\text{margin}\right)
$$

RankNet (logistic):

$$
\ell=-\log\sigma\!\left(\sigma_0\Delta\right)=\log\left(1+e^{-\sigma_0\Delta}\right)
$$
- $\sigma_0$: Slope constant (written $\sigma$ in the original; renamed to avoid clashing with the sigmoid).
- → Identical in form to [BCE](#bce) on the single "is $\mathbf{x}^+$ ranked higher" event.

Backward (RankNet):

$$
\frac{\partial\ell}{\partial s^+}=-\sigma_0\left(1-\sigma(\sigma_0\Delta)\right)=-\frac{\partial\ell}{\partial s^-}
$$
- → Equal & opposite → the pair is pushed apart symmetrically.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

class PairwiseRankingLoss:
    def __init__(self, kind="hinge", margin=1.0, sigma0=1.0):
        self.kind, self.margin, self.sigma0 = kind, margin, sigma0

    def __call__(self, s_pos, s_neg):
        delta = s_pos - s_neg                      ## only the GAP matters, not the levels
        if self.kind == "hinge":
            return F.relu(self.margin - delta).mean()
        ## logistic form == BCE on "is the positive ranked higher"
        return F.softplus(-self.sigma0 * delta).mean()

## Example: pair 1 is correctly ordered by a wide gap, pair 2 is inverted
s_pos = torch.tensor([3.0, 0.0])
s_neg = torch.tensor([0.0, 2.0])
print(PairwiseRankingLoss("hinge")(s_pos, s_neg).item())     ## 1.5
print(PairwiseRankingLoss("logistic")(s_pos, s_neg).item())  ## 1.0878
```
````

```{attention} Q&A
:class: dropdown
*Why pairwise instead of pointwise regression on relevance scores?*
- Pointwise forces the model to predict absolute relevance values that are arbitrary & inconsistent across queries.
- Ranking only needs the ordering → pairwise removes the per-query scale entirely (any monotone shift of $f$ leaves $\Delta$ unchanged).

*Hinge vs logistic form?*
- Hinge: exactly 0 past the margin → ignores already-correct pairs → sparse gradient.
- Logistic: smooth, never 0 → keeps refining confident pairs; gives a probabilistic reading $P(\mathbf{x}^+\succ\mathbf{x}^-)=\sigma(\sigma_0\Delta)$.

*Cons?*
- $O(m^2)$ pairs per query → sampling required.
- Treats every inversion as equally costly, but real ranking measures weight the top of the list far more → LambdaRank fixes this by scaling each pair's gradient by the metric change from swapping it.
- Pairwise consistency does not guarantee a globally consistent total order when the scorer is noisy.

*How is this different from triplet loss?*
- Triplet: anchor-relative **distances** in an embedding space, one shared metric.
- Pairwise ranking: absolute **scores** from one scoring function, no anchor, no metric.
```

&nbsp;

## Distributional
### KL Divergence
- **Name**: Kullback-Leibler Divergence
- **What**: Expected extra nats from coding $P$ with a code optimized for $Q$ (bits if $\log_2$).
- **Why**: Matching a full **distribution** (distillation, variational inference, policy regularization) needs a target that is a distribution, not a label.
- **How**: Compute $\mathbb{E}_P\left[\log\frac{P}{Q}\right]$ and minimize w.r.t. $Q$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $P$: Target distribution.
    - $Q$: Model distribution.
    - $H(P)=-\sum_xP(x)\log P(x)$: Entropy.
    - $H(P,Q)=-\sum_xP(x)\log Q(x)$: Cross entropy.

Definition:

$$
D_\text{KL}(P\|Q)=\sum_xP(x)\log\frac{P(x)}{Q(x)}=H(P,Q)-H(P)
$$

Properties:
- $D_\text{KL}\ge0$, with equality iff $P=Q$ (Gibbs' inequality, via Jensen).
- **Asymmetric**: $D_\text{KL}(P\|Q)\neq D_\text{KL}(Q\|P)$ → not a metric, violates the triangle inequality.
- $\infty$ whenever $Q(x)=0$ while $P(x)>0$ → requires $P\ll Q$ (absolute continuity).

Gradient w.r.t. $Q$'s softmax logits (identical to CE):

$$
\frac{\partial D_\text{KL}(P\|Q)}{\partial z_k}=\hat{q}_k-P_k
$$
- $z_k$: Logit $k$ of $Q$.
- $\hat{q}_k=\text{softmax}(\mathbf{z})_k$: Probability $Q$ assigns to $k$.
- $H(P)$ is constant in $Q$ → it vanishes under differentiation.
```

```{tip} Derivation
:class: dropdown
*Why is $D_\text{KL}\ge0$?*
1. $-\log$ is convex → Jensen's inequality:

    $$
    D_\text{KL}(P\|Q)=\mathbb{E}_P\left[-\log\frac{Q}{P}\right]\ge-\log\mathbb{E}_P\left[\frac{Q}{P}\right]
    $$

2. $\mathbb{E}_P\left[\frac{Q}{P}\right]=\sum_{x:P(x)>0}P(x)\frac{Q(x)}{P(x)}=\sum_{x:P(x)>0}Q(x)\le1$.
    - $\le$ rather than $=$ ← $Q$ may place mass outside $\text{supp}(P)$.
3. → $D_\text{KL}\ge-\log1=0$, with equality iff $\frac{Q}{P}$ is constant $\Leftrightarrow P=Q$.
```

```{attention} Q&A
:class: dropdown
*Forward vs reverse KL?*
- **Forward** $D_\text{KL}(P\|Q)$ — **mass-covering / mean-seeking**. $P(x)>0$ with $Q(x)\to0$ costs $\infty$ → $Q$ must cover every mode, smearing across gaps. Used by MLE.
- **Reverse** $D_\text{KL}(Q\|P)$ — **mode-seeking / zero-forcing**. $Q(x)>0$ where $P(x)\approx0$ costs $\infty$ → $Q$ collapses onto one mode. Used by variational inference & policy optimization.

*If CE and KL have the same gradient, why bother with KL?*
- With one-hot targets they are equal up to a constant → CE is simpler.
- With **soft** targets $H(P)\neq0$ → KL reports the actual excess, so 0 means "matched" while CE bottoms out at $H(P)$ → KL is the readable diagnostic.
- KL is the object that appears in the ELBO, in trust-region/policy constraints, and in distillation — CE is the special case.

*Why can it be infinite, and how is that handled?*
- Support mismatch. Fixes: smooth the target, clamp $Q$ away from 0, or switch to a bounded divergence.

*Implementation gotchas?*
- `nn.KLDivLoss` expects `input` as **log-probabilities** and `target` as probabilities → forgetting `log_softmax` silently gives a wrong loss.
- Use `reduction='batchmean'` to get the mathematically correct per-sample KL; `'mean'` divides by $m\times K$ instead of $m$.

*Symmetric alternatives?*
- **JS divergence**: $\frac{1}{2}D_\text{KL}(P\|M)+\frac{1}{2}D_\text{KL}(Q\|M)$ with $M=\frac{P+Q}{2}$ → symmetric, bounded by $\log2$, finite even under disjoint support.
- **Wasserstein**: uses the geometry of the sample space → gives useful gradients even when the supports do not overlap at all.
```

&nbsp;

### Distillation Loss
- **What**: KL between a temperature-softened teacher and student, blended with ordinary CE on the hard labels.
- **Why**: A one-hot label carries at most $\log_2K$ bits; the teacher's full distribution also encodes **inter-class similarity** ("this 7 looks somewhat like a 1") — the "dark knowledge" a hard label throws away.
- **How**:
    1. Divide both logit vectors by $T$ → softened distributions.
    2. KL(teacher $\|$ student) on the softened pair.
    3. Rescale by $T^2$, then blend with hard-label CE.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $T>1$: Temperature.
    - $\lambda\in[0,1]$: Weight on the soft term.
- Misc:
    - $\mathbf{z}^t,\mathbf{z}^s$: Teacher & student logits.
    - $\hat{p}^{t,T}_k=\text{softmax}(\mathbf{z}^t/T)_k$: Softened teacher distribution.
    - $\hat{p}^{s,T}_k=\text{softmax}(\mathbf{z}^s/T)_k$: Softened student distribution.
    - $\hat{\mathbf{p}}^{s}=\text{softmax}(\mathbf{z}^s)$: Unsoftened student distribution.
    - $\mathbf{y}$: One-hot hard label.
    - $H(\cdot,\cdot)$: Cross entropy.
    - $K$: #classes.

Forward:

$$
\mathcal{L}=\lambda T^2\,D_\text{KL}\left(\hat{\mathbf{p}}^{t,T}\big\|\hat{\mathbf{p}}^{s,T}\right)+(1-\lambda)\,H\left(\mathbf{y},\hat{\mathbf{p}}^{s}\right)
$$
- Hard-label term uses $T=1$.

Why the $T^2$:

$$
\frac{\partial}{\partial z^s_k}\left[D_\text{KL}\right]=\frac{1}{T}\left(\hat{p}^{s,T}_k-\hat{p}^{t,T}_k\right)\approx\frac{1}{T^2}\cdot\frac{z^s_k-z^t_k}{K}\ \ (\text{large }T,\ \text{zero-meaned logits})
$$
- → Soft-term gradients shrink as $\frac{1}{T^2}$ → multiplying by $T^2$ keeps the two terms comparable when $T$ changes.
- The approximation needs $\sum_kz^s_k=\sum_kz^t_k=0$; the exact $\frac{1}{T}\left(\hat{p}^{s,T}-\hat{p}^{t,T}\right)$ gradient holds unconditionally.
```

```{attention} Q&A
:class: dropdown
*What does $T$ actually do?*
- $T$⬆️ → flattens the teacher → the small logits (the similarity structure) become visible instead of being crushed to ~0 by the softmax.
- $T\to\infty$ → the KL term degenerates to matching **logit differences** (least squares on centered logits).
- $T=1$ → just training on the teacher's raw output.

*Why does the soft target help at all?*
- More information per sample → the student gets a full $K$-dim regression target rather than a single class identity → effectively acts as a regularizer and lets the student learn from far less data.

*Why keep the hard-label term?*
- The teacher is wrong sometimes. The hard term anchors the student to ground truth and bounds how far teacher error can propagate.

*Interaction with label smoothing?*
- A teacher trained WITH label smoothing distills worse — smoothing deliberately collapses the inter-class similarity structure that distillation is trying to transfer.

*Is this the same as the KL block above?*
- Same divergence, but the target is a learned model rather than the empirical distribution, and both sides are temperature-scaled → a distinct objective in practice.
```

&nbsp;

## Regularizers
- **What**: Penalty added alongside the data-fit term to constrain the hypothesis.

### L2
- **What**: Penalty on the squared L2 norm of the weights. Also called **ridge**, **weight decay**, **Tikhonov**.
- **Why**: Unpenalized fitting drives $\|\mathbf{w}\|$ up until the model interpolates noise; near-collinear features additionally make $\hat{\mathbf{w}}$ enormous & unstable.
- **How**: Add $\lambda\|\mathbf{w}\|_2^2$ → every gradient step also shrinks $\mathbf{w}$ multiplicatively toward 0.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda\ge0$: Penalty weight.
    - $\eta$: Learning rate.
- Misc:
    - $\hat{R}(\mathbf{w})$: Empirical risk (data-fit term).
    - $X\in\mathbb{R}^{m\times n}$: Input matrix.
    - $I$: Identity matrix.
    - $\tau^2$: Gaussian prior variance.
    - $\sigma^2$: Gaussian noise variance.

Objective & gradient:

$$
\mathcal{L}=\hat{R}(\mathbf{w})+\lambda\|\mathbf{w}\|_2^2,
\qquad
\nabla\mathcal{L}=\nabla\hat{R}+2\lambda\mathbf{w}
$$

GD update (the "decay"):

$$
\mathbf{w}\leftarrow\underbrace{(1-2\eta\lambda)}_{\text{shrink}}\mathbf{w}-\eta\nabla\hat{R}
$$

Closed form (linear regression):

$$
\hat{\mathbf{w}}=(X^TX+\lambda I)^{-1}X^T\mathbf{y}
$$

Bayesian reading: MAP under $\mathbf{w}\sim N(0,\tau^2I)$ with Gaussian noise $\sigma^2$ → $\lambda=\frac{\sigma^2}{\tau^2}$ for $\hat{R}=\|\mathbf{y}-X\mathbf{w}\|_2^2$.
```

```{attention} Q&A
:class: dropdown
*Why doesn't L2 give exact zeros?*
- Penalty gradient $2\lambda w_j\to0$ as $w_j\to0$ → the shrinking force vanishes exactly where a zero would be needed.
- Geometrically: the L2 ball is smooth → its contact point with the loss contour is generically off-axis.

*Is L2 penalty the same as weight decay?*
- ✅ For plain SGD — identical up to the constant $2\eta\lambda$.
- ❌ For **adaptive** optimizers. Adam divides the whole gradient (penalty included) by $\sqrt{v_t}$ → weights with large gradient history get decayed LESS. AdamW instead applies the shrink directly to $\mathbf{w}$, outside the adaptive rescaling. → [AdamW](../dl/optim.md#adamw)

*What must NOT be penalized?*
- The **bias/intercept** — penalizing it breaks **translation equivariance**: adding a constant $c$ to every $y_i$ should just shift the intercept by $c$, but a penalized intercept shrinks toward 0 → the fit depends on where the target's origin happens to sit.
- Normalization scale/shift params — decaying them fights the normalizer's job.

*Does it need feature scaling?*
- ✅ The penalty is not scale-invariant → a feature in mm gets a $1000\times$ smaller weight than the same feature in m, hence $10^6\times$ less penalty.

*Effect on the bias-variance trade-off?*
- $\lambda$⬆️ → bias⬆️, variance⬇️. $\lambda=0$ → ERM. $\lambda\to\infty$ → $\hat{\mathbf{w}}\to\mathbf{0}$.
- Pick by CV, never by training loss ($\lambda=0$ always wins there).
```

&nbsp;

### L1
- **What**: Penalty on the L1 norm of the weights. Also called **lasso**.
- **Why**: L2 shrinks everything but zeroes nothing → the model keeps every irrelevant feature, so no selection happens and the result stays uninterpretable.
- **How**: Add $\lambda\|\mathbf{w}\|_1$ → a constant pull of size $\lambda$ toward 0 regardless of magnitude → weights whose data-fit gradient can't beat $\lambda$ get pinned at exactly 0.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda\ge0$: Penalty weight.
    - $\eta$: Learning rate.
- Misc:
    - $\hat{R}(\mathbf{w})$: Empirical risk (data-fit term).
    - $S(z,t)$: Soft-thresholding operator.
    - $b$: Laplace prior scale.
    - $\sigma^2$: Gaussian noise variance.

Objective:

$$
\mathcal{L}=\hat{R}(\mathbf{w})+\lambda\|\mathbf{w}\|_1
$$

Subgradient (❌closed form, ❌differentiable at 0):

$$
\partial_{w_j}\|\mathbf{w}\|_1=\begin{cases}
\text{sign}(w_j) & w_j\neq0 \\
[-1,1] & w_j=0
\end{cases}
$$

Proximal / soft-thresholding step:

$$
w_j\leftarrow S\left(w_j-\eta\nabla_j\hat{R},\ \eta\lambda\right),\qquad
S(z,t)=\text{sign}(z)\max(|z|-t,0)
$$

Bayesian reading: MAP under $w_j\overset{iid}{\sim}\text{Laplace}(0,b)$ → $\lambda=\frac{2\sigma^2}{b}$ for the same $\hat{R}$.
```

```{tip} Derivation
:class: dropdown
*Why does L1 zero out weights and L2 doesn't?*

Take an orthonormal design ($X^TX=I$) so coordinates decouple; let $z_j$ be the unpenalized solution.

1. **L1**: minimize $\frac{1}{2}(w_j-z_j)^2+\lambda|w_j|$.
    - Optimality at $w_j=0$ requires $0\in-z_j+\lambda[-1,1]$ → holds whenever $|z_j|\le\lambda$.

    $$
    \hat{w}_j=\text{sign}(z_j)\left(|z_j|-\lambda\right)_+
    $$

2. **L2**: minimize $\frac{1}{2}(w_j-z_j)^2+\lambda w_j^2$.

    $$
    \hat{w}_j=\frac{z_j}{1+2\lambda}
    $$

3. → L1 **truncates** (a dead zone of width $2\lambda$ from the subgradient interval at the kink); L2 **rescales** (never exactly 0).
```

```{attention} Q&A
:class: dropdown
*Why does the "corner" explanation work geometrically?*
- The L1 ball has vertices ON the axes → a generic loss contour first touches the constraint set at a vertex → some coordinates are exactly 0. The L2 ball has no vertices.

*Cons?*
- ❌Closed form → proximal / coordinate descent / LARS.
- Correlated group of features → picks ONE arbitrarily → selection is unstable across resamples.
- $n>m$ → selects at most $m$ features (for a unique solution; duplicate columns admit non-unique solutions with more nonzeros).
- Biases the surviving nonzero weights toward 0 → sometimes refit unpenalized on the selected support.

*Does plain subgradient descent produce true zeros?*
- ❌ It lands *near* 0 and jitters. Exact zeros require a **proximal** step (soft-thresholding) or coordinate descent.

*L1 vs L2 in one line?*
- L1 → sparse, selects features, robust to irrelevant inputs.
- L2 → dense, stable, splits weight evenly among correlated features.

*Combine them?*
- **Elastic Net**: $\lambda_1\|\mathbf{w}\|_1+\lambda_2\|\mathbf{w}\|_2^2$ → L1 selects, L2 restores the grouping effect & lifts the $m$-feature cap. → [Elastic Net](supervised.md#elastic-net)
```

```{dropdown} Table: L1 vs L2
| | L1 | L2 |
|:--|:--|:--|
| Penalty | $\lambda\lVert\mathbf{w}\rVert_1$ | $\lambda\lVert\mathbf{w}\rVert_2^2$ |
| Gradient | $\lambda\,\text{sign}(\mathbf{w})$ (constant) | $2\lambda\mathbf{w}$ (proportional) |
| Solution | Sparse | Dense |
| Closed form | ❌ | ✅ |
| Differentiable at 0 | ❌ | ✅ |
| Convex | ✅ | ✅ (strictly) |
| Correlated features | Picks 1 arbitrarily | Splits weight evenly |
| Prior | Laplace | Gaussian |
| Update | Soft-threshold | Multiplicative shrink |
```

&nbsp;

### Entropy Regularization
- **What**: Bonus on the entropy of the model's output distribution.
- **Why**: Objectives that reward confidence (CE, policy gradients) collapse onto a single high-confidence output → overconfidence in supervised learning, premature loss of exploration in RL.
- **How**: Subtract $\beta H(\hat{p})$ from the loss → high entropy is rewarded.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta\ge0$: Entropy weight.
- Misc:
    - $H(\hat{\mathbf{p}})=-\sum_{k=1}^{K}\hat{p}_k\log\hat{p}_k$: Predictive entropy.
    - $H(\mathbf{y},\hat{\mathbf{p}})=-\sum_{k=1}^{K}y_k\log\hat{p}_k$: Cross entropy against the label.
    - $u_k=\frac{1}{K}$: Uniform distribution over classes.
    - $\epsilon$: Label-smoothing strength.

Objective (confidence penalty):

$$
\mathcal{L}=H(\mathbf{y},\hat{\mathbf{p}})-\beta H(\hat{\mathbf{p}})
$$

Relation to label smoothing:

$$\begin{align*}
\text{Label smoothing}&:\ +\epsilon\,D_\text{KL}(u\|\hat{\mathbf{p}})+\text{const} \\
\text{Confidence penalty}&:\ +\beta\,D_\text{KL}(\hat{\mathbf{p}}\|u)+\text{const}
\end{align*}$$
- Same fixed point ($\hat{\mathbf{p}}$ pulled toward uniform $u$), **opposite KL direction** → different gradients off the optimum.
```

```{attention} Q&A
:class: dropdown
*Where is it actually used?*
- RL policy objectives — without it a policy commits to one action early and stops exploring.
- Supervised classification as a confidence penalty (an alternative to label smoothing).
- Max-entropy RL, where it is part of the objective's definition rather than an add-on.

*Entropy bonus vs label smoothing?*
- Smoothing modifies the **target**; the entropy bonus modifies the **loss**.
- Smoothing pushes toward a specific distribution ($u$); the entropy bonus just rewards flatness — with a nonuniform optimum, they differ.

*What breaks if $\beta$ is too large?*
- The model is paid more for being uncertain than for being right → predictions converge to uniform and the fit term is abandoned.
```

&nbsp;
