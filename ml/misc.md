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
# Misc
Model-agnostic theory belonging to no single model family.

❌ Deployment concerns (drift, monitoring, serving, latency) → [Overview](intro.md#production).

Default notations:
- $m$: #samples.
- $n$: #features.
- $\mathbf{x}$: Input.
- $y$: Target.
- $\mathcal{D}$: Training set.
- $\hat{f}$: Model fitted on $\mathcal{D}$.

&nbsp;

## Generalization
- **What**: Error on data not used for fitting.

### No Free Lunch
- **What**: Identical average performance of every learner over all possible problems.
- **Why**: "Which algorithm is best?" is posed as if it had a data-free answer → need a statement of what is achievable with NO assumptions.
- **How**:
    1. A "problem" = one target function $f$. On a finite $\mathcal{X}$ with binary labels there are $2^{|\mathcal{X}|}$ of them.
    2. Off the training set, whatever a learner predicts at $\mathbf{x}$, exactly half of all $f$ agree & half disagree.
    3. → Summing error uniformly over all $f$ cancels the learner out.
    4. → Any gain on one set of problems is repaid exactly on its complement.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{X}$: Finite input space.
    - $\mathcal{Y}=\{0,1\}$: Binary output space.
    - $f:\mathcal{X}\to\mathcal{Y}$: Target function, i.e., "the problem".
    - $\mathcal{L}_a,\mathcal{L}_b$: Learning algorithms.
    - $h$: Hypothesis returned by a learner.
    - $P(h|\mathcal{D},\mathcal{L}_a)$: Probability that $\mathcal{L}_a$ returns $h$ from $\mathcal{D}$.
    - $\mathcal{X}_\mathcal{D}\subset\mathcal{X}$: Inputs appearing in $\mathcal{D}$.
    - $P(\mathbf{x})$: Sampling distribution over inputs.

Off-training-set error:

$$
E_\text{ote}(\mathcal{L}_a|\mathcal{D},f)=\sum_h\sum_{\mathbf{x}\in\mathcal{X}\setminus\mathcal{X}_\mathcal{D}}P(\mathbf{x})\mathbb{1}[h(\mathbf{x})\neq f(\mathbf{x})]P(h|\mathcal{D},\mathcal{L}_a)
$$

Theorem:

$$
\sum_f E_\text{ote}(\mathcal{L}_a|\mathcal{D},f)=\sum_f E_\text{ote}(\mathcal{L}_b|\mathcal{D},f)\qquad\forall\mathcal{L}_a,\mathcal{L}_b
$$
- The sum is **uniform** over all $2^{|\mathcal{X}|}$ target functions — that uniform prior is the entire content of the theorem.
```

```{tip} Derivation
:class: dropdown
*Why does the learner cancel?*
1. Sum the off-training-set error over every target function:

$$
\sum_fE_\text{ote}(\mathcal{L}_a|\mathcal{D},f)=\sum_f\sum_h\sum_{\mathbf{x}\notin\mathcal{X}_\mathcal{D}}P(\mathbf{x})\mathbb{1}[h(\mathbf{x})\neq f(\mathbf{x})]P(h|\mathcal{D},\mathcal{L}_a)
$$

2. Reorder the finite sums, pushing $\sum_f$ innermost:

$$
=\sum_{\mathbf{x}\notin\mathcal{X}_\mathcal{D}}P(\mathbf{x})\sum_hP(h|\mathcal{D},\mathcal{L}_a)\sum_f\mathbb{1}[h(\mathbf{x})\neq f(\mathbf{x})]
$$

3. Fix $\mathbf{x}$ and $h(\mathbf{x})$. Among all $2^{|\mathcal{X}|}$ functions, exactly half disagree at that single point:

$$
\sum_f\mathbb{1}[h(\mathbf{x})\neq f(\mathbf{x})]=\tfrac{1}{2}\cdot2^{|\mathcal{X}|}
$$

4. Substitute, then use $\sum_hP(h|\mathcal{D},\mathcal{L}_a)=1$:

$$
\sum_fE_\text{ote}(\mathcal{L}_a|\mathcal{D},f)=2^{|\mathcal{X}|-1}\sum_{\mathbf{x}\notin\mathcal{X}_\mathcal{D}}P(\mathbf{x})
$$

5. $\mathcal{L}_a$ has vanished → the total depends only on $\mathcal{X}_\mathcal{D}$ and $P(\mathbf{x})$. $\blacksquare$
```

```{attention} Q&A
:class: dropdown
*Why is this not a reason to stop comparing algorithms?*
- The uniform average over ALL target functions is the load-bearing assumption, and it is false for reality.
- Real problems are a vanishingly small, highly structured subset: smooth, compositional, low intrinsic dimension, few relevant features.
- → A non-uniform prior over problems is legitimate → so is a favorite algorithm.

*What does it actually forbid?*
- ❌ A learner that beats another with **zero** assumptions about the data.
- ✅ "GBDT beats KNN on tabular data" — a claim about a restricted, structured problem class, untouched by the theorem.

*What is it really an argument for?*
- Every performance claim must name its problem class.
- Superiority comes from matching [inductive bias](#inductive-bias) to the domain, never from the algorithm alone.

*Does an analogous result hold for optimization?*
- ✅ Over any set of objective functions **closed under permutation** of the search space, all black-box search algorithms have identical average performance.
- → The same escape applies: real objectives are not permutation-closed.

*Why "off-training-set" error specifically?*
- On the training points a memorizer is unbeatable → the interesting quantity is behavior at unseen $\mathbf{x}$, which is exactly where the assumptions do all the work.
```

&nbsp;

### Inductive Bias
- **What**: Assumptions a learner uses to extrapolate beyond the training data.
- **Why**: A finite $\mathcal{D}$ is consistent with infinitely many hypotheses that disagree everywhere else → nothing in the data alone picks one ([No Free Lunch](#no-free-lunch)).
- **How**:
    - **Restriction bias**: shrink the hypothesis class. Linear form, bounded tree depth, conv weight sharing, #components.
    - **Preference bias**: keep the class, order it. L2 → small norm, L1 → sparsity, max-margin, shortest tree, SGD's drift toward low-norm/flat solutions.
    - Both are the $\mathcal{F}$ and the $\Omega$ of [SRM](obj.md#srm).

```{dropdown} Table: Inductive Bias by Model
| Model | Assumption it encodes |
|:--|:--|
| [Linear/Logistic Regression](supervised.md#linear-regression) | Target is linear in the features, in link space |
| [Ridge](supervised.md#ridge-regression) | Small $\ell_2$ norm; coefficients shrink together |
| [Lasso](supervised.md#lasso-regression) | Few features matter |
| [KNN](supervised.md#knn) | Labels are locally smooth under the chosen metric |
| [SVM (RBF)](supervised.md#svm) | Large margin; smooth boundary at a fixed length scale |
| [Decision Tree](supervised.md#decision-tree) | Axis-aligned, piecewise-constant regions; shallow is better |
| [Naive Bayes](supervised.md#naive-bayes) | Features conditionally independent given the class |
| [GMM](unsupervised.md#gmm) | Data generated by $k$ Gaussian blobs |
| [PCA](unsupervised.md#pca) | Signal lies in the high-variance directions |
| CNN | Locality, translation equivariance, hierarchical composition |
| RNN | Sequential order; state summarizes the past |
| Transformer | All-pairs interaction; permutation equivariance until positions are injected |
```

```{attention} Q&A
:class: dropdown
*Strong vs weak bias?*
- Strong → ⬇️data needed, ⬆️bias, catastrophic if wrong.
- Weak → ⬇️bias, ⬆️data needed, ⬆️variance.
- → The choice IS the [bias-variance tradeoff](#bias-variance-tradeoff), stated before seeing the data.

*Can a learner have none?*
- ❌ Impossible. Even rote memorization is a bias — it asserts that unseen points are unrelated to seen ones.
- It also hides outside the model: feature engineering, the loss, the optimizer, augmentation, the train/test split all inject assumptions.

*Where does it live in a deep net?*
- Architecture (conv, attention, recurrence) + initialization + optimizer's implicit bias + augmentation + tokenization.
- → Explains why architecture search matters so much despite universal approximation: capacity is shared, bias is not.

*Why does the "best" model differ across domains?*
- Images → locality & translation invariance hold → CNN's bias is nearly free.
- Tabular → features are heterogeneous, unordered, and interact sparsely → axis-aligned splits fit; conv/attention biases are wrong and pay for it.

*How do you tell a wrong bias from too little data?*
- Wrong bias → training error plateaus high & extra data does NOT help (bias-limited).
- Too little data → training error low, gap large, and the gap shrinks as $m$⬆️ (variance-limited).
```

&nbsp;

### Occam's Razor
- **What**: Preference for the simplest hypothesis consistent with the data.
- **Why**: Many hypotheses fit the training set equally well → need a tie-breaker, and the fit term itself cannot supply one.
- **How**:
    1. Fix a complexity measure: #params, norm, depth, description length, prior.
    2. Among hypotheses with roughly equal fit, take the least complex.
    3. Implemented as a penalty ([L1](obj.md#l1)/[L2](obj.md#l2)), a hard constraint, pruning, an information criterion (AIC/BIC/MDL), or a prior.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $h$: Hypothesis.
    - $\mathcal{M}$: Model class.
    - $\theta$: Params of $\mathcal{M}$.
    - $L(\cdot)$: Code length in bits.

MDL, i.e., the two-part code:

$$
h^*=\arg\min_h\underbrace{L(h)}_{\text{model}}+\underbrace{L(\mathcal{D}|h)}_{\text{data given model}}
$$

Shannon's $L(\cdot)=-\log_2P(\cdot)$ turns it into MAP:

$$
h^*=\arg\max_hP(\mathcal{D}|h)P(h)
$$
- → A code length on hypotheses $\Leftrightarrow$ a prior $P(h)\propto2^{-L(h)}$. Simplicity is a prior, not a fact.

Bayesian Occam's razor:

$$
P(\mathcal{D}|\mathcal{M})=\int P(\mathcal{D}|\theta,\mathcal{M})P(\theta|\mathcal{M})d\theta
$$
- $P(\mathcal{D}|\mathcal{M})$ is a distribution over **datasets** → it normalizes over the data space at fixed sample size, $\sum_\mathcal{D}P(\mathcal{D}|\mathcal{M})=1$.
- → A flexible $\mathcal{M}$ spreads that unit mass over many conceivable $\mathcal{D}$ → ⬇️mass on the observed one → complexity is penalized with NO explicit penalty term.
- ⚠️ Not automatic: what is penalized is prior mass wasted on datasets that did not occur, so a flexible $\mathcal{M}$ with a sharply concentrated prior can still win.
```

```{attention} Q&A
:class: dropdown
*Is it a theorem?*
- ❌ It is a prior. Under [No Free Lunch](#no-free-lunch)'s uniform prior it buys exactly nothing.
- It works because real data is compressible, and short descriptions are scarce: few hypotheses are simple, so a simple one fitting the data by luck is unlikely.

*Simple in what sense?*
- Undefined without a description language. #params, weight norm, depth & VC dimension disagree, and each can be gamed.
- → Two models with equal #params can have wildly different effective capacity → parameter counting is the weakest of the measures.

*Biggest counterexample?*
- Overparameterized nets: #params ≫ $m$, they interpolate the training set, and they still generalize (double descent).
- → Not a refutation of the razor, but of "#params = complexity". Norm, flatness & the optimizer's implicit bias are the operative measures.

*AIC vs BIC?*
- AIC $=-2\log\hat{L}+2d$; BIC $=-2\log\hat{L}+d\log m$, with $d$ = #params.
- $m>7$ → $\log m>2$ → BIC penalizes harder → picks smaller models.
- AIC targets predictive risk (asymptotically efficient); BIC targets recovering the true model (consistent, if it is in the candidate set). Different goals → they are allowed to disagree.

*Why prefer regularization over enumerating simpler models?*
- Capacity is not one-dimensional → a continuous knob is easier to tune than a nested family. See [SRM](obj.md#srm).
```

&nbsp;

### Bias-Variance Tradeoff
- **What**: Split of expected squared test error into bias², variance & irreducible noise, with the first two moving oppositely in capacity.
- **Why**: Training error alone cannot say *why* a model generalizes badly → wrong diagnosis → the fix (more data vs more capacity) is a coin flip.
- **How**:
    1. Fix a test point $\mathbf{x}$; imagine refitting on many independent training sets of size $m$.
    2. **Bias** = gap between the *average* prediction and the truth → rigidity, wrong assumptions.
    3. **Variance** = spread of predictions across training sets → sensitivity to which samples were drawn.
    4. **Noise** = randomness of $y$ given $\mathbf{x}$ → irreducible, a floor no model beats.
    5. Capacity⬆️ → bias⬇️, variance⬆️ → total error is U-shaped in capacity at fixed $m$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $f$: True regression function, $y=f(\mathbf{x})+\epsilon$.
    - $\epsilon$: Noise, $\mathbb{E}[\epsilon]=0$, $\text{Var}[\epsilon]=\sigma^2$, independent of $\mathcal{D}$.
    - $\hat{f}_\mathcal{D}$: Model fitted on a random training set $\mathcal{D}$.
    - $\bar{f}(\mathbf{x})=\mathbb{E}_\mathcal{D}[\hat{f}_\mathcal{D}(\mathbf{x})]$: Average prediction over training sets.

Decomposition at a fixed $\mathbf{x}$, under squared loss:

$$
\mathbb{E}_{\mathcal{D},\epsilon}\left[(y-\hat{f}_\mathcal{D}(\mathbf{x}))^2\right]=\underbrace{(\bar{f}(\mathbf{x})-f(\mathbf{x}))^2}_{\text{Bias}^2}+\underbrace{\mathbb{E}_\mathcal{D}\left[(\hat{f}_\mathcal{D}(\mathbf{x})-\bar{f}(\mathbf{x}))^2\right]}_{\text{Variance}}+\underbrace{\sigma^2}_{\text{Noise}}
$$

Properties:
- All three terms $\ge0$ → $\sigma^2$ is a hard lower bound on test MSE.
- Expectation is over **training sets**, not over test points → not measurable from a single fit.
- Holds pointwise; integrate over $P(\mathbf{x})$ for the full risk.
```

```{tip} Derivation
:class: dropdown
*Where do the three terms come from?*
1. Split off the noise, using $y=f+\epsilon$ with $\epsilon$ independent of $\mathcal{D}$ and $\mathbb{E}[\epsilon]=0$:

$$
\mathbb{E}\left[((f-\hat{f}_\mathcal{D})+\epsilon)^2\right]=\mathbb{E}_\mathcal{D}\left[(f-\hat{f}_\mathcal{D})^2\right]+2\underbrace{\mathbb{E}[\epsilon]}_{0}\mathbb{E}_\mathcal{D}[f-\hat{f}_\mathcal{D}]+\underbrace{\mathbb{E}[\epsilon^2]}_{\sigma^2}
$$

2. Add & subtract $\bar{f}$ inside the remaining term:

$$
\mathbb{E}_\mathcal{D}\left[\left((f-\bar{f})+(\bar{f}-\hat{f}_\mathcal{D})\right)^2\right]
$$

3. Expand. The cross term dies because $f-\bar{f}$ is deterministic and $\mathbb{E}_\mathcal{D}[\bar{f}-\hat{f}_\mathcal{D}]=0$:

$$
=(f-\bar{f})^2+\mathbb{E}_\mathcal{D}\left[(\hat{f}_\mathcal{D}-\bar{f})^2\right]
$$

4. Recombine → Bias² + Variance + $\sigma^2$. $\blacksquare$
```

```{dropdown} Table: Diagnosis
| Train error | Val error | Diagnosis | Fix |
|:--|:--|:--|:--|
| High | High, $\approx$ train | Underfit — bias-limited | ⬆️capacity, better features, ⬇️$\lambda$, train longer |
| Low | High | Overfit — variance-limited | ⬆️$m$, ⬆️$\lambda$, ⬇️capacity, bagging, early stopping |
| Low | Low | Done | Ship |
| High | Low | Impossible on i.i.d. splits | Bug: leaky/easy val set, mismatched preprocessing, train-time-only noise (dropout) |
```

```{attention} Q&A
:class: dropdown
*Which knobs move which term?*
- Bias⬇️: ⬆️capacity, richer features, boosting, ⬇️regularization.
- Variance⬇️: ⬆️$m$, ⬆️regularization, bagging/averaging, feature selection, early stopping.
- Noise⬇️: ❌ impossible from the model side — only better labels or better features can move it, and then it is a *different* $\sigma^2$.

*Where do ensembles sit?*
- [Bagging](supervised.md#bagging) averages decorrelated high-variance learners → variance⬇️, bias $\approx$ unchanged → pair it with deep trees.
- [Boosting](supervised.md#boosting) fits residuals sequentially → bias⬇️ primarily, variance⬆️ with #rounds → pair it with shallow trees.

*Does the decomposition hold for 0-1 loss?*
- ❌ Not additively. On points where the average prediction is already wrong, variance can *lower* error by flipping some fits to the correct class.
- → "High variance is bad" is a squared-loss statement, not a universal one.

*Can you measure bias and variance?*
- Not from one fit. Approximate by refitting over bootstrap samples & inspecting the spread of predictions at fixed $\mathbf{x}$; the bias term additionally needs $f$, so it is only exactly computable in simulation.

*Is the U-curve universal?*
- ❌ Double descent: past the interpolation threshold, test error falls again as capacity grows → the classic curve describes the underparameterized regime only.
- The tradeoff is over **capacity at fixed $m$**. ⬆️$m$ shrinks variance — and, for a regularized fit at fixed $\lambda$, bias along with it → adding data is never a tradeoff.

*Does regularization always help?*
- Optimal $\lambda>0$ exactly when the variance it removes exceeds the bias² it adds. On an already bias-limited model it strictly hurts.
```

&nbsp;

### Overfitting
- **What**: Low training error, high test error.
- **Why**:
    - *Why does it happen?*
        - Capacity ≫ information in the data → spare capacity is spent memorizing noise as if it were signal.
        - $m$ too small, or unrepresentative of the deployment distribution.
        - Noisy/mislabeled targets, or features that are pure noise.
        - Training too long past the point where validation error turns.
    - *Why does it hide?*
        - Repeated selection on the same validation set → the *validation* score itself becomes optimistic → the gap only shows up on truly fresh data.
        - [Data leakage](#data-leakage) → both train & val look great, and only production fails.
- **How**:
    - **More information**: collect data, augment, add weak labels, merge related tasks.
    - **Less capacity**: fewer features, prune trees, ⬇️degree, ⬇️width/depth, ⬆️$k$ in [KNN](supervised.md#knn), ⬆️bandwidth.
    - **Regularization**: [L1](obj.md#l1)/[L2](obj.md#l2), dropout, weight decay, max-norm, label smoothing.
    - **Early stopping**: halt at the validation minimum.
    - **Averaging**: [bagging](supervised.md#bagging), ensembling, weight averaging.
    - **Honest protocol**: [cross-validation](#cross-validation) for selection + a test set touched once.

```{attention} Q&A
:class: dropdown
*How do you detect it?*
- Train-val gap, tracked over training epochs or over model capacity.
- Learning curve in $m$: val error still falling as $m$⬆️ and the gap closing → variance-limited → more data pays. Both curves flat & high → bias-limited → more data does nothing.

*Is zero training error overfitting?*
- ❌ Overfitting is defined by the **gap**, not by memorization. 1-NN and overparameterized nets interpolate and can still generalize.

*Can a 2-parameter model overfit?*
- ✅ Capacity is not parameter count. Two params fit on 3 noisy samples still track the noise.
- ✅ And a 2-param model *selected* from thousands of candidate feature pairs has consumed far more degrees of freedom than 2 — the search is part of the model, and it never appears in the parameter count.

*What is validation-set overfitting?*
- Every comparison extracts information from the validation set → after hundreds of configurations, the winner's validation score is optimistically biased by roughly the spread of the search.
- Fixes: [nested CV](#nested-cv), a locked test set, fewer configurations, or a coarse-then-fine search.

*Why does early stopping act like a penalty?*
- Gradient descent from a small initialization grows the parameter norm gradually → stopping early caps that norm → an implicit norm ball, i.e., [SRM](obj.md#srm) with an implicit $\Omega$.

*Overfitting vs leakage vs distribution shift?*
- Overfitting: train ≪ val. Fit problem.
- Leakage: train ≈ val ≪ production. Protocol problem.
- Shift: train ≈ val ≪ production, and the *inputs* have moved. Data problem.
- → Same production symptom, three different fixes → never skip the diagnosis.
```

&nbsp;

#### Underfitting
- **What**: High training error AND high test error.
- **Why**:
    - Capacity < complexity of the signal, e.g., a linear model on an XOR-shaped boundary.
    - Over-regularization: $\lambda$ too large, tree depth too small, too few components.
    - Features carry no signal, or the signal is only in interactions that were never constructed.
    - Optimization failure, NOT capacity: LR too small/large, too few steps, bad conditioning, dead units.
- **How**:
    - ⬆️capacity, or switch to a nonlinear model/kernel.
    - Add features, interactions, basis expansions.
    - ⬇️$\lambda$, train longer, retune the optimizer.
    - Fix conditioning: scale/standardize features, whiten, precondition.

```{attention} Q&A
:class: dropdown
*How do you separate underfitting from an optimization failure?*
- Try to overfit a tiny subset (~20 samples) with regularization off. It is a smoke test, not a proof.
- ❌ It cannot → optimization or implementation is broken, or the labels are self-contradictory. Stop tuning capacity.
- ✅ It reaches ~0 error → that failure mode is ruled out, nothing more. Memorizing 20 points says nothing about capacity for the full relationship → next check the learning curve in $m$.

*Why is high train error sometimes NOT underfitting?*
- Irreducible noise sets a floor. A model at the Bayes error looks "bad" and is optimal → compare against the noise level or a human baseline, never against 0.

*Which is the worse failure?*
- Underfitting: visible immediately & cheap to fix.
- Overfitting: invisible until deployment if the protocol is sloppy → far more expensive.
```

&nbsp;

### Curse of Dimensionality
- **What**: Data becomes sparse & distances lose contrast as $n$⬆️.
- **Why**:
    - **Volume**: volume grows as $r^n$ → fixed $m$ covers an exponentially shrinking fraction → keeping density constant needs $m\propto c^n$.
    - **Concentration**: for roughly independent features, $\|\mathbf{x}_i-\mathbf{x}_j\|$ concentrates → nearest & farthest neighbors become indistinguishable → "nearest" stops meaning "similar".
    - **Geometry**: almost all the mass of a ball sits in a thin shell near its surface, and the inscribed ball occupies a vanishing fraction of the cube → every point is an edge case, so every prediction is an extrapolation.
    - **Estimation**: nonparametric rates degrade as $m^{-\Theta(1/n)}$; a full covariance needs $O(n^2)$ params.
    - **Noise accumulation**: each irrelevant feature adds variance to the distance while adding no signal → contrast is destroyed by the features you did not need.
- **How**:
    - Feature selection; sparsity ([L1](obj.md#l1)).
    - Dimensionality reduction ([PCA](unsupervised.md#pca), autoencoders; [t-SNE/UMAP](unsupervised.md#umap) for visualization ONLY).
    - Strong structural priors: linear/additive models, convolutions, factorization.
    - Better geometry: learned metrics, cosine on normalized vectors, domain-specific distances.
    - Exploit the manifold hypothesis — model the intrinsic, not the ambient, dimension.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $n$: #dims.
    - $m$: #samples.
    - $r$: Target fraction of the data to enclose.
    - $e_n(r)$: Edge length of a hypercube neighborhood capturing a fraction $r$.
    - $d_{\max},d_{\min}$: Farthest & nearest distance from a query to the $m$ points.
    - $V_n$: Volume of the unit-radius ball in $\mathbb{R}^n$.
    - $\mathbf{x}_n$: Query-to-point difference vector in $\mathbb{R}^n$, so $\|\mathbf{x}_n\|$ is a distance.

Neighborhood size for uniform data in $[0,1]^n$:

$$
e_n(r)=r^{1/n}
$$
- "Local" is a lie in high dims: a neighborhood holding 1% of the data spans 63% of each axis at $n=10$.

Ball vs cube:

$$
V_n=\frac{\pi^{n/2}}{\Gamma\left(\frac{n}{2}+1\right)},\qquad\frac{V_n}{2^n}\xrightarrow{n\to\infty}0
$$
- The ball inscribed in the cube occupies almost none of it → the mass lives in the corners.

Distance concentration, at fixed $m$:

$$
\lim_{n\to\infty}\text{Var}\left[\frac{\|\mathbf{x}_n\|}{\mathbb{E}\|\mathbf{x}_n\|}\right]=0\quad\Rightarrow\quad\frac{d_{\max}-d_{\min}}{d_{\min}}\xrightarrow{P}0
$$
- Relative contrast vanishes → nearest-neighbor queries become meaningless, however fast you compute them.
```

```{note} Example
:class: dropdown
$m$ points uniform in the unit ball in $\mathbb{R}^n$. Median distance from the center to the CLOSEST point:

$$
d(n,m)=\left(1-2^{-1/m}\right)^{1/n}
$$

| $n$ | $m$ | $d$ |
|:--|:--|:--|
| 2 | 500 | 0.04 |
| 10 | 500 | 0.52 |
| 100 | 500 | 0.94 |

At $n=10$ with 500 samples, the nearest neighbor is already past halfway to the boundary → "local averaging" averages over the whole space.
```

```{attention} Q&A
:class: dropdown
*Which models suffer most?*
- Worst: [KNN](supervised.md#knn), kernel density estimation, RBF kernels with a small bandwidth, Euclidean [K-Means](unsupervised.md#k-means), [LOF](unsupervised.md#lof) — anything that averages locally.
- Mildest: sparse linear models & regularized parametric models — they only need a low-dimensional projection to be right.
- Trees are in between: robust to feature *scale*, still degraded by many irrelevant features, since every split is chosen on noisy gains.

*Is adding features always bad?*
- ❌ A genuinely informative feature adds signal faster than it adds noise.
- ⚠️ But the curse is NOT only about irrelevant features: the nonparametric rate $m^{-\Theta(1/n)}$ degrades even when every coordinate carries signal. Irrelevant dimensions merely make it worse for free.
- → Feature selection, not feature starvation, is the answer.

*Then why do image & text models work at all?*
- Manifold hypothesis: intrinsic dimension ≪ ambient dimension — real images occupy a tiny structured sliver of pixel space.
- Features are strongly dependent, so the i.i.d. premise of the concentration result fails.
- Architectural priors (locality, weight sharing) cut the effective dimension before any distance is computed.

*Does high dimension ever help?*
- ✅ Blessing of dimensionality. Random directions are nearly orthogonal → the Johnson-Lindenstrauss lemma preserves all pairwise distances among $m$ points in $O(\epsilon^{-2}\log m)$ dims, independent of $n$.
- ✅ Cover's theorem: at fixed $m$ with points in general position, the fraction of labelings that are linearly separable rises to 1 as $n$⬆️, hitting 1 once $n\ge m-1$ → the [kernel trick](supervised.md#kernel-trick) deliberately raises dimension.
- → The curse hits *density & distance* estimation; separation gets easier.

*Why does normalization matter so much here?*
- Distances are dominated by the widest-scaled features → in high dims a single unscaled feature can set every neighbor ranking by itself.
```

&nbsp;

## Model Taxonomy
### Parametric vs Non-Parametric
- **What**: Fixed-size vs data-growing model representation.
- **Why**: The two families fail in opposite ways — one from a wrong functional form, one from insufficient local data — so the choice must be made from $m$, $n$ & how much is known about the shape of $f$, before any tuning.
- **How**:
    - **Parametric**: assume a functional form with $d$ params, $d$ fixed independent of $m$ → fit $\hat\theta$ → the data can be thrown away.
    - **Non-parametric**: effective #params grows with $m$; the data (or a data-sized summary) IS part of the model, and complexity is set by a smoothing knob ($k$, bandwidth, depth).

```{dropdown} Table: Parametric vs Non-Parametric
| | Parametric | Non-parametric |
|:--|:--|:--|
| Capacity | Fixed before seeing data | Grows with $m$ |
| Examples | Linear/Logistic Regression, [LDA](supervised.md#lda), [Naive Bayes](supervised.md#naive-bayes), [GMM](unsupervised.md#gmm) at fixed $k$, a fixed NN | [KNN](supervised.md#knn), KDE, kernel [SVM](supervised.md#svm), unpruned [trees](supervised.md#decision-tree), [GP](optim.md#gp) |
| Data needed | Less | More, exponentially so in $n$ |
| Inference cost | $O(d)$, data-independent | Grows with $m$ |
| Bias / Variance | ⬆️ / ⬇️ | ⬇️ / ⬆️ |
| Extrapolation | Follows the assumed form | No mechanism — falls back to the nearest data, a constant, or the prior mean |
| Interpretability | Coefficients | Instances, regions, neighbors |
| Fails when | The form is misspecified | $m$ small or $n$ large |
| Consistency | Only if $f$ is in the class | Universally consistent, but only under conditions — KNN needs $k\to\infty$ AND $k/m\to0$ |
```

```{attention} Q&A
:class: dropdown
*Is a neural net parametric?*
- ✅ By the definition: fixed architecture → fixed #params, independent of $m$.
- ⚠️ It behaves non-parametrically in practice ← width is chosen after seeing the data size, and the infinite-width limit is literally a [GP](optim.md#gp).

*Is an SVM parametric?*
- Linear → ✅, $n$ weights.
- Kernel → ❌, the solution is $\sum_i\alpha_iK(\mathbf{x},\mathbf{x}_i)$ over support vectors, whose count generally grows with $m$.
- → The same algorithm changes category with the kernel → the label describes the representation, not the method.

*Does "non-parametric" mean no parameters, or no assumptions?*
- ❌ Both readings are wrong. It means the #params is not fixed in advance.
- KNN still assumes local smoothness under a metric; KDE still assumes a kernel & bandwidth.

*Which do you pick?*
- Small $m$, large $n$, need extrapolation, need interpretable coefficients → parametric.
- Large $m$, small $n$, unknown & irregular $f$ → non-parametric.

*What is semi-parametric?*
- A parametric component of interest + a non-parametric nuisance component, e.g., Cox proportional hazards (parametric coefficients, unspecified baseline hazard) or partially linear models.
- → Keeps the interpretable part interpretable without assuming the rest.
```

&nbsp;

### Discriminative vs Generative
- **What**: Modeling $P(y|\mathbf{x})$ (or a bare decision rule) vs modeling the joint $P(\mathbf{x},y)$.
- **Why**: Classification only ever consumes $P(y|\mathbf{x})$, so modeling $P(\mathbf{x})$ spends capacity & assumptions on something the decision never reads — but without $P(\mathbf{x})$ there is no way to sample, to score novelty, to marginalize a missing feature, or to use unlabeled data.
- **How**:
    - **Generative**: estimate $P(\mathbf{x}|y)$ & $P(y)$ from the data → invert with Bayes' rule to classify.
    - **Discriminative**: estimate $P(y|\mathbf{x})$ directly, never modeling how $\mathbf{x}$ arose.
    - **Discriminant function**: skip probability entirely, learn $\mathbf{x}\mapsto y$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $P(\mathbf{x}|y=k)$: Class-conditional density.
    - $P(y=k)$: Class prior.
    - $f(\mathbf{x})$: Decision function.
    - $K$: #classes.

Generative → posterior by Bayes' rule:

$$
P(y=k|\mathbf{x})=\frac{P(\mathbf{x}|y=k)P(y=k)}{\sum_{j=1}^{K}P(\mathbf{x}|y=j)P(y=j)}
$$

Three levels of modeling, most to least assumptions:

$$
\underbrace{P(\mathbf{x},y)}_{\text{generative}}\;\longrightarrow\;\underbrace{P(y|\mathbf{x})}_{\text{discriminative}}\;\longrightarrow\;\underbrace{f(\mathbf{x})}_{\text{discriminant function}}
$$
- Each arrow discards information that is irrelevant to the decision, and with it the ability to answer questions other than "which class".
```

```{dropdown} Table: Generative vs Discriminative
| | Generative | Discriminative |
|:--|:--|:--|
| Models | $P(\mathbf{x},y)$ | $P(y\mid\mathbf{x})$ or $f(\mathbf{x})$ |
| Examples | [Naive Bayes](supervised.md#naive-bayes), [LDA](supervised.md#lda)/QDA, [GMM](unsupervised.md#gmm), HMM, VAE, diffusion, LM | Logistic Regression, [SVM](supervised.md#svm), [trees](supervised.md#decision-tree)/[GBDT](supervised.md#gbdt), [KNN](supervised.md#knn), CRF, most NN classifiers |
| Asymptotic error | Higher when the density is misspecified | Lower — it optimizes the quantity being scored |
| Small $m$ | Converges faster to its own ceiling | Needs more data |
| Sample new $\mathbf{x}$ | ✅ | ❌ |
| Missing features | ✅ Marginalize them out | ❌ Requires [imputation](#missing-data) |
| Unlabeled data | ✅ Contributes to $P(\mathbf{x})$ | ❌ Needs extra machinery |
| Novelty detection | ✅ Low $P(\mathbf{x})$ | ❌ No notion of an unlikely input |
| Class prior change | ✅ Swap $P(y)$, keep the rest | ❌ Refit or recalibrate |
```

```{attention} Q&A
:class: dropdown
*Why is the discriminative version usually more accurate?*
- It spends all its capacity on the decision boundary. A generative model must also get $P(\mathbf{x})$ right, and errors there leak into the posterior even where they change no decision.
- Formally: ERM on $P(y|\mathbf{x})$ optimizes the objective actually being evaluated. See [Metric vs Loss](eval.md#metric-vs-loss).

*Then when does generative win?*
- Small $m$: a strong (even wrong) density assumption acts as a prior → it approaches its own ceiling with far fewer samples — Naive Bayes needs $O(\log n)$ samples to get there where Logistic Regression needs $O(n)$.
- Plus anything requiring $P(\mathbf{x})$: sampling, anomaly detection, missing features, semi-supervised learning.
- → Often a crossover: NB ahead at small $m$, LogReg ahead at large $m$. Not guaranteed ← a correctly specified generative model reaches the Bayes error and never has to lose. See [Naive Bayes](supervised.md#naive-bayes).

*Are generative classifiers ever forced?*
- Streaming class priors, extreme imbalance where the negative class is "everything else", and open-set problems where a test point may belong to no training class.

*Does "generative model" mean the same thing in GenAI?*
- ✅ Same definition. An LLM models $P(\text{token}|\text{context})$, i.e., a distribution over the data itself, so it can sample → generative over sequences.
- ⚠️ It is still trained by a *discriminative-looking* next-token cross-entropy — the objective's form does not decide the category, what is being modeled does.

*Which side is KNN on?*
- Discriminative. It estimates $P(y|\mathbf{x})$ by a local vote and never models $P(\mathbf{x})$ — despite storing all the data.

*Is low $P(\mathbf{x})$ a reliable novelty detector?*
- ✅ For low-dim, well-specified density models.
- ❌ For deep generative models: they routinely assign HIGHER likelihood to out-of-distribution inputs than to their own training distribution → likelihood alone is not an OOD score.
```

&nbsp;

## Data Issues
### Data Leakage
- **What**: Information unavailable at prediction time entering training → optimistic validation, collapse in production.
- **Why**:
    - **Target leakage**: a feature is a consequence or proxy of the label, populated only after the event, e.g., `treatment_prescribed` when predicting diagnosis.
    - **Preprocessing contamination**: any `fit`-like transform run before the split — scaler, imputer, feature selector, PCA, target encoder, resampler — carries test statistics into training.
    - **Temporal leakage**: random split on time-ordered data → training on the future.
    - **Group leakage**: the same user/patient/session/document in both sides of a split.
    - **Duplicates**: exact or near-duplicate rows straddling the split → measured memorization.
    - **Selection leakage**: features chosen, or thresholds tuned, using all labels — including the test ones.
- **How**:
    - Split FIRST. Every fitted transform lives inside a pipeline fitted on training folds only.
    - Split by time for temporal data, by group for grouped data, and deduplicate before splitting.
    - For each feature, ask: does this value exist, with this value, at prediction time?
    - Audit implausibly high scores & single dominant features. A label-permutation check (shuffle $y$, rerun the whole pipeline) must fall back to the no-information baseline.
    - Lock a final test set & touch it once.

```{attention} Q&A
:class: dropdown
*What is the loudest symptom?*
- A validation score far better than any credible baseline, or one feature dominating importance.
- → Treat "too good" as a bug report, not a result.

*Why is fitting the scaler on all data actually a problem?*
- The validation fold stops being independent → its score is no longer an estimate of unseen performance.
- Magnitude varies wildly: standardization on large $m$ leaks almost nothing; supervised transforms (target encoding, feature selection by correlation with $y$, SMOTE, iterative imputation) leak enormously and can manufacture near-perfect scores from pure noise.

*Where exactly must resampling go?*
- Inside the CV loop, on the training fold only.
- Oversampling before splitting copies minority samples into both sides → the model is scored on rows it trained on → recall looks superb & production recall is near zero.

*Is leakage just overfitting?*
- ❌ [Overfitting](#overfitting) shows as a train-val **gap**; leakage keeps validation looking excellent and fails only after deployment.
- → Leakage is invisible to every offline diagnostic that trusts the split.

*Is a leaky feature ever acceptable?*
- Only if it is genuinely available at inference with the same timing & semantics. The test is the deployment timeline, not the correlation.
```

&nbsp;

### Class Imbalance
- **What**: Skewed class prior → the majority class dominates the loss & the metric.
- **Why**:
    - *Why does it happen?*
        - The base rate is genuinely small: fraud, disease, defects, clicks, churn.
        - Sampling or labeling bias: rare events are under-collected or under-annotated.
    - *Why does it hurt?*
        - Additive losses sum over samples → the majority's gradient swamps the minority's.
        - [Accuracy](eval.md#accuracy) hands a constant majority predictor the majority rate for free → the metric endorses a model that has learned nothing.
        - $\tau=0.5$ is calibrated-optimal only under symmetric costs → at low prevalence almost no sample's probability clears it → the default threshold predicts the majority everywhere, even when the *ranking* is perfect.
        - The minority has few samples in **absolute** terms → high variance exactly where it matters.
- **How**:
    - **Metric**: [PR-AUC](eval.md#pr-auc), macro-F1, [MCC](eval.md#mcc), recall at fixed precision. ❌accuracy, ⚠️[ROC-AUC](eval.md#roc-auc).
    - **Threshold**: tune $\tau$ on validation, or set it from the cost ratio — the cheapest fix, and often sufficient alone.
    - **Loss**: class weights $\propto1/m_k$, [focal loss](obj.md#focal-loss), explicit cost matrix.
    - **Data**: undersample the majority (ensemble over several subsets to avoid discarding data), oversample the minority, SMOTE — all strictly inside the CV loop.
    - **Reframe**: too few positives in absolute count, or positives that do not form a coherent class → [anomaly detection](unsupervised.md#anomaly-detection) instead of classification.
    - **Recalibrate** afterwards if the probabilities feed a decision.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\pi$: True positive-class prior in deployment.
    - $\pi'$: Positive-class prior after resampling/weighting.
    - $p$: Correctly calibrated $P(y=1|\mathbf{x})$ at prior $\pi$.
    - $p'$: Model output, calibrated at prior $\pi'$.

Prior correction, assuming $P(\mathbf{x}|y)$ is unchanged by resampling:

$$
\text{logit}(p)=\text{logit}(p')+\log\frac{\pi}{1-\pi}-\log\frac{\pi'}{1-\pi'}
$$

Equivalently:

$$
p=\frac{\frac{\pi}{\pi'}p'}{\frac{\pi}{\pi'}p'+\frac{1-\pi}{1-\pi'}(1-p')}
$$
- Balancing the training set shifts the intercept by exactly $\log\frac{\pi'(1-\pi)}{\pi(1-\pi')}$ → rankings are untouched, probabilities are not.
```

```{attention} Q&A
:class: dropdown
*Does imbalance always need fixing?*
- ❌ What hurts is (a) too few minority samples in absolute count, (b) a metric or threshold mismatched to the costs.
- 90:10 on 1M rows is usually a non-issue; 90:10 on 100 rows is fatal. The ratio alone is not the diagnosis.

*Oversampling vs class weights?*
- For any additive loss, duplicating a sample $c$ times and weighting it by $c$ give the same gradient direction → the same intervention, up to the loss normalization (`sum` → identical; `mean` → they differ by a global scale the LR absorbs) and minibatch composition.
- SMOTE is genuinely different: it fabricates new points rather than reweighting old ones.

*Why is SMOTE risky?*
- Interpolates in feature space → invalid points for categorical/ordinal features and for non-convex class manifolds.
- Interpolating near the boundary amplifies label noise into the region the classifier cares most about.
- Degrades in high dims ← the neighbors it interpolates between are not actually near.
- → Threshold tuning + class weights is the stronger default; SMOTE must clear that bar to justify itself.

*What does resampling do to the outputs?*
- Calibrates them to the resampled prior, not the deployment one → probabilities are systematically too high for the minority → correct via the prior-shift formula, or re-calibrate on an unresampled validation set.
- → Harmless if only the ranking is used; wrong if the score feeds an expected-cost decision.

*Does undersampling waste data?*
- ✅ Recover it by bagging over several disjoint majority subsets, each paired with the full minority set → uses all the data & decorrelates the members.

*Why is ROC-AUC misleading here?*
- FPR's denominator is all the negatives → thousands of false positives barely move it while precision collapses. See [PR-AUC](eval.md#pr-auc).
```

&nbsp;

### Missing Data
- **What**: Feature values absent for part of the samples.
- **Why**:
    - **MCAR** (Missing Completely At Random): the missingness pattern $R$ is independent of everything, $P(R|Z_\text{obs},Z_\text{mis})=P(R)$, where $Z=(\mathbf{x},y)$ covers features AND targets. Sensor dropout, random transmission loss.
    - **MAR** (Missing At Random): missingness explained by the OBSERVED data, $P(R|Z_\text{obs},Z_\text{mis})=P(R|Z_\text{obs})$. A test ordered only for older patients.
    - **MNAR** (Missing Not At Random): missingness depends on the unobserved value itself. High earners refusing to state income.
    - Mechanically: schema changes, failed joins, optional fields, "not applicable" encoded as blank, features that did not exist before some date.
- **How**:
    - **Deletion**: listwise (safe under MCAR, and for complete-case regression under some MAR patterns), or drop the feature if mostly empty.
    - **Simple imputation**: mean/median/mode, ALWAYS paired with a binary missingness indicator.
    - **Model-based**: KNN imputation, MICE/iterative imputation, [EM](optim.md#em), low-rank matrix completion.
    - **Multiple imputation**: draw $M$ completed datasets → fit $M$ models → pool by Rubin's rules → uncertainty from imputing is carried through instead of hidden.
    - **Native handling**: CART surrogate splits; XGBoost/LightGBM learn a default direction per split; CatBoost/LightGBM accept missing as its own category.
    - Fit every imputer inside the training fold ← otherwise [leakage](#data-leakage).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $R$: Missingness indicator matrix, $R_{ij}=1$ if $x_{ij}$ observed.
    - $M$: #imputed datasets.
    - $\hat{Q}_j$: Estimate from imputed dataset $j$.
    - $U_j$: Within-imputation variance of $\hat{Q}_j$.
    - $\bar{Q}$: Pooled estimate.
    - $B$: Between-imputation variance.
    - $T$: Total variance.

Rubin's rules for multiple imputation:

$$\begin{align*}
\bar{Q}&=\frac{1}{M}\sum_{j=1}^{M}\hat{Q}_j\\
\bar{U}&=\frac{1}{M}\sum_{j=1}^{M}U_j\\
B&=\frac{1}{M-1}\sum_{j=1}^{M}(\hat{Q}_j-\bar{Q})^2\\
T&=\bar{U}+\left(1+\frac{1}{M}\right)B
\end{align*}$$
- $B$ is the price of not knowing the missing values → single imputation sets $B=0$ and understates every standard error.
```

```{attention} Q&A
:class: dropdown
*Can you test which mechanism you have?*
- MCAR → testable, e.g., compare the observed-data distribution across missing & non-missing groups.
- MAR vs MNAR → **not** testable from the observed data, ever. The evidence needed is exactly what is missing.
- → MAR is an assumption you argue for from domain knowledge, not one you verify.

*Why is mean imputation dangerous?*
- Shrinks the feature's variance & pulls its correlations toward 0 → attenuated coefficients & understated standard errors.
- Creates an artificial spike at the mean → a tree will happily split on it.
- Acceptable for pure prediction with an indicator flag; not acceptable for inference.

*Why add a missingness indicator?*
- Under MNAR the *fact* of missingness carries signal, which imputation destroys → the flag preserves it and lets the model separate "unknown" from "average".

*Does dropping rows ever bias the model?*
- Under MCAR → unbiased, merely wasteful.
- Under MNAR → complete cases are a non-random subsample → the fitted relationship applies to a population that is not the deployment one.
- Under MAR → ⚠️ NOT automatically biased. Complete-case **regression** stays unbiased when missingness is independent of the target given the covariates already in the model — a genuinely weaker requirement than the one imputation needs.

*Single or multiple imputation?*
- Prediction only, evaluated end-to-end → single imputation inside the pipeline is usually enough.
- Coefficients, CIs, or p-values → multiple imputation, or the uncertainty is fiction.

*What if a feature is missing at inference but never in training?*
- The pipeline has no imputation path for it → decide the policy explicitly (default value + flag, a fallback model, or refuse to score). This is a modeling decision, not a preprocessing detail.
```

&nbsp;

## Validation
### Cross-Validation
- **What**: Rotating $K$ train/validation splits over the same data to estimate out-of-sample error.
- **Why**: A single hold-out both wastes data & inherits the luck of one split → its score has high variance, and model selection made on it is selection on noise.
- **How**:
    1. Shuffle, then partition into $K$ folds.
    2. For fold $k$: fit on the other $K-1$, score on $k$.
    3. Aggregate — mean ± sd of the $K$ scores, or pool the out-of-fold predictions & score once.
    4. Refit on all $m$ samples with the chosen configuration; that refit is the deliverable, $\text{CV}_K$ is the estimate of its error.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $K$: #folds.
    - $\mathcal{D}_k$: Validation fold $k$.
    - $\hat{f}^{-k}$: Model fit on all folds except $k$.
    - $\ell$: Per-sample loss.
    - $H$: Smoother matrix of a linear fit, $\hat{\mathbf{y}}=H\mathbf{y}$.
    - $X\in\mathbb{R}^{m\times n}$: Design matrix.
    - $h_{ii}$: $i$-th diagonal entry of $H$, i.e., the leverage of sample $i$.
    - $\hat{y}_i$: In-sample fitted value from the single full-data fit.

Estimator:

$$
\text{CV}_K=\frac{1}{K}\sum_{k=1}^{K}\frac{1}{|\mathcal{D}_k|}\sum_{i\in\mathcal{D}_k}\ell\left(y_i,\hat{f}^{-k}(\mathbf{x}_i)\right)
$$

LOOCV in closed form, for a linear fit under squared loss:

$$
\text{CV}_m=\frac{1}{m}\sum_{i=1}^{m}\left(\frac{y_i-\hat{y}_i}{1-h_{ii}}\right)^2
$$
- OLS → $H=X(X^TX)^{-1}X^T$; ridge → $H=X(X^TX+\lambda I)^{-1}X^T$.
- All $m$ leave-one-out errors from ONE fit → LOOCV costs nothing there, and $h_{ii}\to1$ marks the points the model cannot afford to lose.

Properties:
- Each $\hat{f}^{-k}$ sees only $\frac{K-1}{K}m$ samples → $\text{CV}_K$ is pessimistic for the final refit; the bias shrinks as $K$⬆️.
- Training sets overlap heavily → the $K$ scores are dependent → their sample variance understates $\text{Var}[\text{CV}_K]$, and the fold sd is a measure of fold heterogeneity, NOT a standard error.
- No **universal** unbiased estimator of $\text{Var}[\text{CV}_K]$ exists, i.e., none valid for every distribution & learner.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class KFold:
    def __init__(self, k=5, shuffle=True, seed=0):
        self.k, self.shuffle, self.seed = k, shuffle, seed

    def split(self, m):
        idx = np.arange(m)
        if self.shuffle:
            np.random.default_rng(self.seed).shuffle(idx)
        ## array_split handles m % k != 0 -> the first m % k folds get one extra sample
        for val in np.array_split(idx, self.k):
            yield np.setdiff1d(idx, val), val

def cross_val_score(fit_predict, X, y, k=5):
    scores, oof = [], np.empty(len(y))
    for tr, va in KFold(k).split(len(y)):
        ## refit from scratch every fold -> no state (scaler, imputer, params) survives a fold
        oof[va] = fit_predict(X[tr], y[tr], X[va])
        scores.append(np.mean((y[va] - oof[va]) ** 2))
    return np.array(scores), oof

def ridge(Xtr, ytr, Xva, lam=1.0):
    w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
    return Xva @ w

## Example
rng = np.random.default_rng(0)
X = rng.normal(size=(50, 3))
y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=50)

scores, oof = cross_val_score(ridge, X, y, k=5)
print(scores.mean(), np.mean((y - oof) ** 2))  ## equal: MSE is decomposable & folds are equal-sized
print(scores.std())                            ## the spread is the point -- report it, not just the mean
```
````

```{dropdown} Table: Variants
| Variant | Split rule | Use it when |
|:--|:--|:--|
| Hold-out | One split | $m$ huge, or a fit is expensive |
| K-Fold | $K$ rotating folds | Default, $K=5$ or $10$ |
| Stratified K-Fold | Preserve class ratios per fold | Any classification, mandatory under imbalance |
| Repeated K-Fold | K-Fold $\times R$ different shuffles | Small $m$, the fold split itself is noisy |
| LOOCV | $K=m$ | Tiny $m$, or a linear model where it is closed-form |
| Group K-Fold | No group spans two folds | Repeated users/patients/sessions |
| Time-Series | Forward chaining, train only on the past | Any temporal ordering |
| Nested | CV inside CV | Reporting a score for a **tuned** pipeline |
```

```{attention} Q&A
:class: dropdown
*How do you choose $K$?*
- $K$⬆️ → training sets closer to $m$ → ⬇️pessimistic bias, but the models become nearly identical → their errors correlate → variance need NOT shrink, and cost is $K$ fits.
- LOOCV: near-zero bias, and usually high variance ← the $m$ fits are almost the same model. Not a theorem: the direction depends on the learner's stability.
- → $K=5$/$10$ is the standard compromise, not a ritual.

*Is $\text{CV}_K$ unbiased for the deployed model?*
- ❌ Slightly pessimistic — every fold model is trained on less data than the final refit. The gap matters at small $m$ and vanishes at large $m$.

*Why stratify?*
- Keeps each fold's class ratio equal to the population's → ⬇️variance of the estimate, and it prevents folds with zero minority samples, which make several metrics undefined.

*When does plain K-Fold silently lie?*
- Temporal data → training on the future.
- Grouped data → the same entity on both sides.
- Spatially or serially correlated data → neighboring points act as duplicates.
- → All three inflate the score without producing any visible symptom.

*Fold-mean or pooled out-of-fold score?*
- Decomposable metrics (accuracy, MSE, log loss) → identical for equal folds.
- Non-decomposable (F1, AUC, $R^2$) → different. Pooling is usually preferred, but it assumes fold models are comparable & similarly calibrated.

*What must go inside the loop?*
- Every `fit`-like step: scaling, imputation, encoding, feature selection, PCA, resampling, threshold tuning. Anything fit outside is [leakage](#data-leakage).

*How do you compare two models across folds?*
- Compare **paired** per-fold differences, not two independent means.
- Folds share training data → the naive paired $t$-test is anticonservative → use repeated CV with a variance correction for the train/test overlap, or a paired bootstrap on a held-out set.
```

&nbsp;

#### Nested CV
- **What**: CV inside CV — inner loop selects hyperparams, outer loop scores the selection procedure.
- **Why**: Tuning and reporting on the same folds reuses the validation data twice → the reported score is the maximum over configurations, which is optimistically biased, and the bias grows with the size of the search.
- **How**:
    1. Split into $K_\text{out}$ outer folds.
    2. On each outer training portion: run a complete inner CV over the hyperparam grid.
    3. Refit on the whole outer training portion using the inner winner.
    4. Score once on the untouched outer fold.
    5. Average the $K_\text{out}$ outer scores.

````{important} Code
:class: dropdown
```python
import numpy as np
## Reuses KFold and ridge from the Cross-Validation block above

def nested_cv(fit_predict, X, y, grid, k_out=5, k_in=4):
    out = []
    for tr, te in KFold(k_out, seed=0).split(len(y)):
        best, best_err = None, np.inf
        for lam in grid:
            errs = []
            ## INNER: split the OUTER-TRAINING portion only -> te is never touched here
            for itr, iva in KFold(k_in, seed=1).split(len(tr)):
                p = fit_predict(X[tr[itr]], y[tr[itr]], X[tr[iva]], lam)
                errs.append(np.mean((y[tr[iva]] - p) ** 2))
            if np.mean(errs) < best_err:
                best, best_err = lam, np.mean(errs)
        ## OUTER: refit on all of tr with the inner winner, score once on te
        p = fit_predict(X[tr], y[tr], X[te], best)
        out.append((best, np.mean((y[te] - p) ** 2)))
    return out

## Example
rng = np.random.default_rng(0)
X = rng.normal(size=(50, 3))
y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=50)

res = nested_cv(ridge, X, y, grid=[0.01, 0.1, 1.0, 10.0])
print([lam for lam, _ in res])                  ## folds may disagree -- expected, not a bug
print(np.mean([e for _, e in res]))             ## honest error of the TUNING PROCEDURE
```
````

```{attention} Q&A
:class: dropdown
*What exactly does the outer score estimate?*
- The error of the whole **procedure** — "run this grid search, then ship the winner" — NOT of one fixed hyperparam setting.
- Trained on $\frac{K_\text{out}-1}{K_\text{out}}m$ samples, not $m$ → like plain CV, it is pessimistic for the final all-data refit.
- → Outer folds selecting different hyperparams is expected. Disagreement means the objective is flat there, not that the protocol failed.

*Then which hyperparams do you ship?*
- Re-run the inner selection on ALL the data. That single refit is the model; the nested score is only its honest error bar.

*How much does skipping it cost?*
- The bias scales with #configurations tried and with $1/m_\text{val}$ → a large random search on a small validation set can inflate the reported score by several points.

*When can you skip it?*
- Large $m$ with a locked test set never used for tuning — the test set already plays the outer role.
- A handful of configurations → the selection bias is negligible relative to the estimate's own noise.

*What is the cost?*
- $K_\text{out}\times K_\text{in}\times|\Lambda|$ fits. Cut it with a coarse inner grid, successive halving, or fewer inner folds — never by moving the tuning outside.
```

&nbsp;

### Online A/B Testing
- **What**: Randomized controlled experiment comparing two variants on live traffic.
- **Why**: Offline metrics score predictions on logged data, not the user behavior being optimized, and the two routinely disagree ← the logs were generated under the old system. Comparing before/after or self-selected groups confounds the change with who saw it; randomization is what makes the measured difference causal.
- **How**:
    1. Pre-register ONE decision metric plus guardrails, and fix $\alpha$, power & the MDE.
    2. Compute the sample size per arm → derive the runtime from traffic.
    3. Randomize by hashing a stable unit ID into buckets: control gets the current system, treatment the change.
    4. Run to the pre-set horizon, ≥1 full week for weekly seasonality. ❌stopping the moment it turns significant.
    5. Check sample-ratio mismatch → test the difference → read the CI on the effect, not just the p-value → ship, iterate, or roll back.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $p_C,p_T$: True conversion rate of control & treatment.
    - $\hat{p}_C,\hat{p}_T$: Observed rates.
- Hyperparams:
    - $\alpha$: Significance level, $=P(\text{Type I error})$, i.e., rejecting a true $H_0$.
    - $\beta$: $P(\text{Type II error})$, i.e., failing to reject a false $H_0$. Power $=1-\beta$.
    - $\delta$: MDE (Minimum Detectable Effect), the smallest **absolute** lift worth detecting.
- Misc:
    - $m$: Samples per arm.
    - $z_q$: Upper-$q$ standard-normal quantile, $\Phi(z_q)=1-q$.
    - $\hat{p}$: Pooled rate under $H_0$.

Sample size per arm, two-sided, equal split:

$$
m=\frac{2\bar{p}(1-\bar{p})\left(z_{\alpha/2}+z_{\beta}\right)^2}{\delta^2}
$$
- $\bar{p}=\frac{p_C+p_T}{2}=p_C+\frac{\delta}{2}$, i.e., the MIDPOINT rate. Plugging in the baseline $p_C$ instead silently under-powers the test.
- $\delta$ is absolute: a 10% baseline with a 20% relative target → $\delta=0.02$.
- Equal-variance approximation. The unpooled form $\left[z_{\alpha/2}\sqrt{2\bar{p}(1-\bar{p})}+z_\beta\sqrt{p_C(1-p_C)+p_T(1-p_T)}\right]^2/\delta^2$ agrees to within a sample here.
- $m\propto\delta^{-2}$ → halving the detectable effect quadruples the traffic. This is the entire cost structure of experimentation.

Two-proportion $z$-test of $H_0:p_T=p_C$:

$$
z=\frac{\hat{p}_T-\hat{p}_C}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{m_T}+\frac{1}{m_C}\right)}},\qquad\hat{p}=\frac{x_C+x_T}{m_C+m_T}
$$
- $x_C,x_T$: Conversion counts. Reject when $|z|>z_{\alpha/2}$.
- The variance is **pooled** ← $H_0$ asserts one common rate. The CI on the difference is unpooled ← it must not assume $H_0$.

CUPED variance reduction, using a pre-experiment covariate $X$:

$$
Y^\text{cuped}=Y-\theta\left(X-\mathbb{E}[X]\right),\qquad\theta^*=\frac{\text{Cov}(Y,X)}{\text{Var}(X)}
$$
- $Y$: The experiment metric per unit.
- $X$: The same unit's metric in the pre-period.
- $\rho=\text{corr}(Y,X)$.
- Unbiased ← $X$ predates the assignment → the treatment cannot have moved it → its mean is equal across arms.
- $\text{Var}[Y^\text{cuped}]=(1-\rho^2)\text{Var}[Y]$ → $\rho=0.7$ removes half the required traffic.
```

````{important} Code
:class: dropdown
```python
import numpy as np
from scipy.stats import norm

def sample_size(p_c, mde_abs, alpha=0.05, power=0.8):
    ## two-sided test -> alpha is split across both tails; power enters as a one-sided quantile
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    ## pool at the MIDPOINT rate, not the baseline -- p_c alone gives 14128 and only 78.3% power
    p_bar = p_c + mde_abs / 2
    return int(np.ceil(2 * p_bar * (1 - p_bar) * (z_a + z_b) ** 2 / mde_abs ** 2))

def two_proportion_z(x_c, m_c, x_t, m_t, alpha=0.05):
    p_c, p_t = x_c / m_c, x_t / m_t
    ## H0 says both arms share one rate -> pool them for the test's standard error
    p = (x_c + x_t) / (m_c + m_t)
    z = (p_t - p_c) / np.sqrt(p * (1 - p) * (1 / m_c + 1 / m_t))
    pval = 2 * (1 - norm.cdf(abs(z)))
    ## the interval must NOT assume H0 -> unpooled standard error here
    se = np.sqrt(p_c * (1 - p_c) / m_c + p_t * (1 - p_t) / m_t)
    half = norm.ppf(1 - alpha / 2) * se
    return z, pval, (p_t - p_c - half, p_t - p_c + half)

## Example: 10% baseline, detect an absolute +1pt lift
print(sample_size(0.10, 0.01))                       ## 14752 per arm
print(two_proportion_z(1500, 15000, 1650, 15000))    ## z=2.83, p=0.005, CI=(0.003, 0.017)
```
````

```{dropdown} Table: Pitfalls
| Pitfall | What goes wrong | Fix |
|:--|:--|:--|
| Peeking | Testing repeatedly as data arrives → true $\alpha$ ≫ nominal | Fixed horizon, or sequential tests / always-valid p-values |
| Multiple comparisons | $k$ metrics or arms → $P(\ge1\text{ FP})=1-(1-\alpha)^k$ if independent, and bounded by $k\alpha$ otherwise | One pre-registered decision metric; Bonferroni / BH-FDR on the rest |
| Sample ratio mismatch | Observed split deviates from the intended ratio by more than chance → assignment or logging is broken | $\chi^2$ on the bucket counts; ❌interpret the result until fixed |
| Novelty / primacy | Users react to *change*, not to the design | Run longer; segment new vs returning users |
| Interference | Units affect each other (social graph, marketplace supply) | Cluster/geo randomization, switchback tests |
| Simpson's paradox | Allocation ratio changes mid-test → the arms get different traffic mixes → the aggregate reverses every segment | Fixed allocation; always inspect segments |
| Dilution | Only a fraction of the arm is exposed to the change | Trigger-based analysis, valid ONLY if triggering itself is unaffected by the assignment |
| Twyman's law | A spectacular result | Assume instrumentation bug until an A/A test says otherwise |
```

```{attention} Q&A
:class: dropdown
*User-level or request-level randomization?*
- User-level: consistent experience, measures cumulative & long-term effects, ⬆️variance → ⬆️traffic needed. Mandatory whenever the change has memory (UI, personalization, learning systems).
- Request-level: ⬇️variance & faster, but the same user sees both arms → invalid for anything with carryover, and it corrupts user-level metrics.

*Why an A/A test?*
- Two identical arms → validates randomization, logging & the variance estimate. About $\alpha$ of A/A tests should come out "significant"; systematically more means the pipeline is broken, systematically fewer means the variance is overestimated.

*Why is peeking so destructive?*
- Each look is another chance to cross the threshold → checking daily for two weeks pushes the true false-positive rate several times above $\alpha$.
- Fix properly: sequential testing (SPRT, mSPRT, always-valid confidence sequences), which is designed for continuous monitoring, not a Bonferroni patch.

*How do you get power without more traffic?*
- CUPED with a pre-period covariate — usually the single biggest win.
- Cap/winsorize heavy-tailed metrics; stratified or blocked assignment; a more sensitive proxy metric; longer runtime.

*What if the metric is not a proportion?*
- Means → two-sample $t$/Welch, valid by CLT even for skewed data at large $m$.
- Revenue-style heavy tails → winsorize or bootstrap; the CLT is slow when a handful of users dominate the sum.
- Ratio metrics (clicks per session) → the randomization unit ≠ the metric unit → use the delta method or bootstrap over units, never the naive per-event variance.

*Significant but should you ship?*
- A p-value is not an effect size. Read the CI: a significant +0.02% lift may not cover the maintenance cost.
- Symmetrically, a non-significant result under low power is NOT evidence of no effect — report the CI and say what was ruled out.

*Why at least a full week?*
- Day-of-week seasonality & a different user mix on weekends → any partial week weights some populations more than others.

*Why do offline and online results disagree?*
- Logs come from the incumbent policy → offline scoring is off-policy and biased toward it.
- Offline metrics measure prediction quality; the online metric measures behavior after the prediction is acted on, including UI, latency & feedback loops. See [Metric vs Loss](eval.md#metric-vs-loss).
```

&nbsp;
