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
# Bayesian
Study notes from {cite:t}`bayes,bayes2`.
Inference → model construction → prediction & decisions.
Posterior computation (MCMC, VI, …) lives in [Sampling & Approximate Inference](sampling.md).

Notations:
- $\mathcal{D}$: Observed dataset; regression conditions on the observed inputs.
- $\theta$: Unknown parameter, scalar or vector.
- $d$: Dimension of $\theta$ when vector-valued.
- $p(\mathcal{D}\mid\theta)$: Sampling density/mass, viewed as a likelihood after observing $\mathcal{D}$.
- $\pi(\theta)$: Prior density/mass.
- $\pi(\theta\mid\mathcal{D})$: Posterior density/mass.
- $Z=p(\mathcal{D})$: Marginal likelihood, also called evidence.
- $\tilde p(\theta)=p(\mathcal{D}\mid\theta)\pi(\theta)$: Unnormalized posterior.
- $\ell(\theta)=\log\tilde p(\theta)$: Log unnormalized posterior.
- $h(\theta)$: Scalar quantity of interest.
- $S$: Number of simulation draws.
- $\theta^{(s)}$: Draw indexed by $s$.
- $\tilde y$: Future observation.
- $\alpha\in(0,1)$: Excluded probability for a credible set.

Parameter vectors use $\theta$ rather than the global bold-vector convention. Integrals become sums for discrete unknowns; $\mathcal{N}(\mu,v)$ uses variance $v$, not standard deviation.

&nbsp;

## Bayesian Inference

### Bayes' Rule
- **What**: Conditional distribution of unknowns given observations.
- **Why**: Observations constrain unknowns without usually determining them.
- **How**:
    1. **Prior**: Encode uncertainty before conditioning on this dataset.
    2. **Likelihood**: Score each parameter by the sampling density/mass it assigns to the observed data.
    3. **Posterior**: Multiply prior by likelihood, then normalize.

```{note} Math
:class: dropdown

$$
\pi(\theta\mid\mathcal{D})
=\frac{p(\mathcal{D}\mid\theta)\pi(\theta)}{Z},
\qquad
Z=\int p(\mathcal{D}\mid\vartheta)\pi(\vartheta)\,d\vartheta.
$$

- Proper posterior requires $0<Z<\infty$.
- For events: $P(A\mid B)=P(B\mid A)P(A)/P(B)$ when $P(B)>0$.
- For continuous observations, condition through densities/regular conditional distributions; do not divide by the probability of a singleton.
```

```{tip} Derivation
:class: dropdown
1. Factor the joint distribution in both directions:

    $$
    p(\mathcal{D},\theta)
    =p(\mathcal{D}\mid\theta)\pi(\theta)
    =\pi(\theta\mid\mathcal{D})p(\mathcal{D}).
    $$

2. Integrate out $\theta$ to obtain $Z$.
3. Divide by $Z$; normalization changes total mass, not relative parameter weights.
```

```{attention} Q&A
:class: dropdown
*Is a likelihood a probability distribution over parameters?*

- No. It is normalized over possible datasets for fixed $\theta$, not over $\theta$ for fixed data.
- A density value can exceed $1$; only integrated probabilities must lie in $[0,1]$.

*Can zero prior probability be rescued by data?*

- A prior-null set remains posterior-null under ordinary conditioning with a proper posterior.
- A continuous parameter's individual points are usually null; learning concerns neighborhoods, not point masses.

*Does Bayesian probability require the physical parameter to be random?*

- No. The distribution can represent uncertainty about a fixed unknown.
- All conclusions remain conditional on the likelihood, prior, and observation process.
```

&nbsp;

### Likelihood Principle
- **What**: Proportional likelihoods convey the same evidence about $\theta$.
- **Why**: Parameter-independent factors cannot change relative support for parameter values.
- **How**: Hold the prior fixed; remove only likelihood factors independent of every unknown being inferred.

```{note} Math
:class: dropdown

$$
p(\mathcal{D}_1\mid\theta)=c\,p(\mathcal{D}_2\mid\theta)
\quad\Longrightarrow\quad
\pi(\theta\mid\mathcal{D}_1)=\pi(\theta\mid\mathcal{D}_2),
\qquad c>0.
$$

- $c$: Constant independent of $\theta$; the two normalizing constants satisfy $Z_1=cZ_2$, not generally $Z_1=Z_2$.

$$
\frac{\pi(\theta_1\mid\mathcal{D})}{\pi(\theta_2\mid\mathcal{D})}
=\frac{p(\mathcal{D}\mid\theta_1)}{p(\mathcal{D}\mid\theta_2)}
\frac{\pi(\theta_1)}{\pi(\theta_2)}.
$$

- For discrete hypotheses: posterior odds = likelihood ratio × prior odds.
- For continuous $\theta$: ratios of density values, not odds of singleton events.
```

```{attention} Q&A
:class: dropdown
*Does a stopping rule matter?*

- Not for the posterior if its contribution to the observed-data likelihood is parameter-independent and the prior is unchanged.
- Informative selection, censoring, or unrecorded stopping information can change the likelihood → model the actual observation process.
- Posterior equivalence does not imply equal repeated-sampling error rates across experimental designs.

*Can all constants be dropped?*

- For a fixed model's posterior kernel: only factors independent of its unknowns.
- For evidence/model comparison: retain model-dependent normalizing constants.
```

&nbsp;

### Sufficient Statistics
- **What**: Data summaries preserving all likelihood information about $\theta$.
- **Why**: Compress observations without changing posterior inference under the specified model.
- **How**: Factor the likelihood into a parameter-free part and a part depending on data only through the summary.

```{note} Math
:class: dropdown
For a dominated family, the Fisher–Neyman factorization criterion is

$$
p(\mathcal{D}\mid\theta)=a(\mathcal{D})\,b(T(\mathcal{D}),\theta).
$$

- $T(\mathcal{D})$: Sufficient statistic.
- $a(\mathcal{D})\geq0$: Parameter-independent factor.
- $b\geq0$: Remaining factor.
- Equivalently, the conditional law of the full data given $T$ is parameter-independent.
- Consequently $\pi(\theta\mid\mathcal{D})=\pi(\theta\mid T(\mathcal{D}))$ for compatible priors, wherever the posteriors are defined.
```

```{note} Example
:class: dropdown
- Bernoulli observations: $T=\sum_i y_i$ is sufficient when sample size $m$ is fixed.
- Normal observations, known variance: $T=\bar y$.
- Normal observations, unknown mean & variance: $T=(\sum_i y_i,\sum_i y_i^2)$.
- Bernoulli counts alone cannot diagnose serial dependence; sufficiency is relative to the assumed independent Bernoulli model.
```

```{attention} Q&A
:class: dropdown
*Does any factorization produce a minimal sufficient statistic?*

- No. The entire dataset is sufficient too.
- A minimal sufficient statistic removes all remaining redundancy; under standard common-support conditions, equal statistic values correspond exactly to proportional likelihoods.

*Is posterior preservation for one prior enough to prove sufficiency?*

- No. A degenerate prior can make every dataset yield the same posterior.
- Sufficiency is a property of the sampling family, not a convenient prior.
```

&nbsp;

### Identifiability
- **What**: Distinct parameters imply distinct sampling distributions.
- **Why**: Data cannot distinguish parameter values that generate the same observable law.
- **How**: Find parameter symmetries or redundant coordinates; infer identifiable quantities or impose justified constraints.

```{note} Math
:class: dropdown

$$
p(\cdot\mid\theta_1)=p(\cdot\mid\theta_2)
\quad\Longrightarrow\quad \theta_1=\theta_2.
$$

- Equality means equality of distributions over all possible observations, not merely equal likelihood at the observed dataset.
```

```{note} Example
:class: dropdown
- Model $y_i\sim\mathcal{N}(a+b,1)$: data identify $a+b$, not $a$ and $b$ separately.
- A proper prior can produce a proper joint posterior; allocation between $a$ and $b$ remains prior-driven along the likelihood ridge.
- Mixture labels can be unidentifiable while the mixture density is learnable.
```

```{attention} Q&A
:class: dropdown
*Does nonidentifiability make all Bayesian learning impossible?*

- No. Identifiable functions and predictions may still be learned.
- A prior can resolve posterior ambiguity but does not make the likelihood identifiable.

*Identifiability versus weak identification?*

- Identifiability: a population property of the model.
- Weak identification: distinct parameters produce nearly indistinguishable distributions at the available sample size.
```

&nbsp;

### Priors
- **What**: Probability distributions over unknowns before observing $\mathcal{D}$.
- **Why**: Encode domain constraints & plausible scales where data leave ambiguity.
- **How**:
    1. Choose support from the parameter's meaning.
    2. Elicit plausible effects on the observation scale, not only coefficient scales.
    3. Simulate from the prior predictive distribution; revise implausible implications.

```{note} Math
:class: dropdown

$$
p(\tilde y)=\int p(\tilde y\mid\theta)\pi(\theta)\,d\theta.
$$

- **Proper prior**: $\int\pi(\theta)\,d\theta=1$.
- **Improper prior**: A nonnegative prior measure with infinite total mass; may yield a proper posterior, but has no ordinary prior predictive distribution.
- Even a proper prior requires checking $0<Z<\infty$.

For independent observations, sequential updating is

$$
\pi(\theta\mid y_{1:t})
\propto p(y_t\mid\theta)\pi(\theta\mid y_{1:t-1}).
$$

- $t$: Number of observations processed.
- Dependent observations require $p(y_t\mid y_{1:t-1},\theta)$ instead.
```

```{dropdown} Table: Prior Choices
| Choice | Encodes | Failure to watch |
|:--|:--|:--|
| Informative | Substantial external knowledge | Prior-data conflict; transported knowledge may not apply |
| Weakly informative | Plausible scale, excluding extreme effects | Must be calibrated to units, link function & number of predictors |
| Flat | Constant density in chosen coordinates | Not invariant to reparameterization; often improper |
| Gaussian shrinkage | Most coefficients near a center | Does not put mass at exact zero |
| Spike-and-slab | Positive probability of exact exclusion | Discrete model search; inclusion probabilities depend on slab scale |
```

```{attention} Q&A
:class: dropdown
*Is a diffuse prior uninformative?*

- Not necessarily. Wide coefficient priors can imply extreme predictions; in many dimensions, small effects can accumulate.
- A flat prior on a probability is not flat on its log-odds.

*When is reusing old data legitimate?*

- Previous posterior → new prior, then condition only on new information under a compatible model.
- Using the same observations to construct a prior and again as a fresh likelihood double-counts information unless the procedure explicitly accounts for that reuse.

*Can an improper prior be used for Bayes factors?*

- Not naively: its arbitrary multiplicative constant enters the evidence.
- A proper posterior within each model does not resolve undefined model odds.
```

&nbsp;

#### Jeffreys Prior
- **What**: Prior measure proportional to Fisher-information volume.
- **Why**: Flat density depends on the choice of coordinates.
- **How**: Weight parameter regions by how distinguishable nearby sampling distributions are.

```{note} Math
:class: dropdown

$$
\pi_J(\theta)\propto\sqrt{\det I(\theta)},
\qquad
I(\theta)=
\mathbb{E}_{Y\mid\theta}
\left[
\nabla_\theta\log p(Y\mid\theta)
\left(\nabla_\theta\log p(Y\mid\theta)\right)^\top
\right].
$$

- $I(\theta)$: Fisher information matrix under a regular, identifiable model.
- $Y$: Random observation under the sampling model.
- $\pi_J$: Jeffreys prior density.
- Scalar Bernoulli probability: $I(\theta)=1/[\theta(1-\theta)]$ → $\operatorname{Beta}(1/2,1/2)$.
- Normal mean with known variance: constant density on $\mathbb{R}$ → improper.
```

```{attention} Q&A
:class: dropdown
*What is invariant?*

- The probability measure under smooth one-to-one reparameterization, not the numeric density value.
- The Fisher-information determinant transforms by the square of the Jacobian determinant.

*Is it a universal objective prior?*

- No. It depends on the sampling model/design, can be improper, and can behave poorly with nuisance parameters.
- Reference priors optimize a different information criterion and need not equal the joint Jeffreys prior.
```

&nbsp;

### Conjugacy
- **What**: Prior families closed under Bayesian updating for a likelihood.
- **Why**: Exact posteriors expose how data and prior information combine.
- **How**: Match the prior kernel to the parameter-dependent factors in the likelihood.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $a>0$: First prior shape parameter.
    - $b>0$: Second shape parameter for Beta, rate for Gamma.
- Misc:
    - $r=\sum_i y_i$: Total successes/counts.

For $y_i\mid\theta\overset{\mathrm{iid}}{\sim}\operatorname{Bernoulli}(\theta)$,

$$
\begin{align*}
\theta&\sim\operatorname{Beta}(a,b),\\
\theta\mid\mathcal{D}&\sim\operatorname{Beta}(a+r,b+m-r),\\
\mathbb{E}[\theta\mid\mathcal{D}]
&=\frac{a+r}{a+b+m}
=\frac{a+b}{a+b+m}\frac{a}{a+b}
+\frac{m}{a+b+m}\frac{r}{m}.
\end{align*}
$$

- The weighted-mean identity uses $m>0$.
- Prior strength $a+b$ acts as a pseudo-sample size for this posterior-mean identity; it is not literally the number of previously observed trials.
```

```{tip} Derivation
:class: dropdown
1. Bernoulli likelihood kernel: $\theta^r(1-\theta)^{m-r}$.
2. Beta prior kernel: $\theta^{a-1}(1-\theta)^{b-1}$.
3. Add exponents → $\theta^{a+r-1}(1-\theta)^{b+m-r-1}$.
4. Recognize the updated Beta density; normalization is its beta-function constant.
```

```{note} Example
:class: dropdown
- Prior $\operatorname{Beta}(2,2)$; observe $7$ successes in $10$ trials.
- Posterior $\operatorname{Beta}(9,5)$; mean $9/14$, between prior mean $1/2$ and observed fraction $7/10$.
- Sequential batches give the same result as one batch: success/failure counts add.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class BetaBernoulli:
    def __init__(self, a, b):
        if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
            raise ValueError("Beta shapes must be finite and positive")
        self.a, self.b = float(a), float(b)

    def step(self, y):
        y = np.asarray(y)
        if y.ndim != 1 or not np.all((y == 0) | (y == 1)):
            raise ValueError("Expected a vector of binary observations")
        self.a += y.sum()
        self.b += y.size - y.sum()
        return self.a / (self.a + self.b)

## Example
model = BetaBernoulli(2, 2)
mean = model.step([1] * 7 + [0] * 3)
assert np.isclose(mean, 9 / 14)
print(mean)
```
````

```{dropdown} Table: Conjugate Updates
| Sampling model | Prior | Posterior |
|:--|:--|:--|
| $y_i\sim\operatorname{Bernoulli}(\theta)$ | $\theta\sim\operatorname{Beta}(a,b)$ | $\operatorname{Beta}(a+r,b+m-r)$ |
| $y_i\sim\operatorname{Poisson}(\lambda)$ | $\lambda\sim\operatorname{Gamma}(a,b)$ | $\operatorname{Gamma}(a+r,b+m)$ |
| $y_i\sim\operatorname{Categorical}(\mathbf{w})$ | $\mathbf{w}\sim\operatorname{Dirichlet}(\mathbf{a})$ | $\operatorname{Dirichlet}(\mathbf{a}+\mathbf{c})$ |
| $y_i\sim\mathcal{N}(\mu,\sigma^2)$, known $\sigma^2$ | $\mu\sim\mathcal{N}(\mu_0,v_0)$ | $\mathcal{N}(\mu_m,v_m)$ |

- $\lambda>0$: Poisson rate.
- $\operatorname{Gamma}(a,b)$: Shape-rate convention, density proportional to $\lambda^{a-1}e^{-b\lambda}$.
- $\mathbf{w}$: Category probabilities, summing to $1$.
- $\mathbf{a}$: Positive Dirichlet concentration vector.
- $\mathbf{c}$: Category counts.
- $\mu$: Normal mean.
- $\sigma^2>0$: Known sampling variance.
- $\mu_0$: Prior mean.
- $v_0>0$: Prior variance.
- $v_m=(v_0^{-1}+m/\sigma^2)^{-1}$: Posterior variance.
- $\mu_m=v_m(\mu_0/v_0+\sum_i y_i/\sigma^2)$: Posterior mean.
```

```{attention} Q&A
:class: dropdown
*Why is conjugacy common in exponential families?*

- A likelihood kernel $\exp\{\eta(\theta)^\top\sum_i T(y_i)-mA(\theta)\}$ updates a prior kernel $\exp\{\eta(\theta)^\top\chi-\nu A(\theta)\}$ by $\chi\leftarrow\chi+\sum_iT(y_i)$ and $\nu\leftarrow\nu+m$.
- $\eta$: Natural parameter.
- $T$: Sufficient statistic.
- $A$: Log-normalizer.
- $\chi$: Prior statistic hyperparameter.
- $\nu$: Prior weight hyperparameter; choose it jointly with $\chi$ so the prior is proper.

*Conjugate versus conditionally conjugate?*

- Conjugate: the full posterior remains in the prior family.
- Conditionally conjugate: a block's posterior given other unknowns has a tractable family → useful for Gibbs sampling, even when the marginal posterior is not closed form.

*Should convenience determine the prior?*

- Only if the resulting assumptions are defensible; conjugacy is algebraic convenience, not evidence of model adequacy.
```

&nbsp;

### Marginalization & Reparameterization
- **What**: Posterior distributions for subsets or functions of unknowns.
- **Why**: Joint parameters are rarely the final scientific quantity of interest.
- **How**:
    - **Marginalization**: Average over nuisance unknowns.
    - **Transformation**: Map posterior mass into the new coordinates.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\psi$: Parameter of interest.
    - $\lambda$: Nuisance parameter in $\theta=(\psi,\lambda)$.
- Misc:
    - $\phi=g(\theta)$: Smooth one-to-one transformation.

$$
\pi(\psi\mid\mathcal{D})
=\int\pi(\psi,\lambda\mid\mathcal{D})\,d\lambda.
$$

$$
\pi_\phi(\phi\mid\mathcal{D})
=\pi_\theta(g^{-1}(\phi)\mid\mathcal{D})
\left|\det\frac{\partial g^{-1}(\phi)}{\partial\phi}\right|.
$$

- Subscripts on $\pi$ identify the coordinate system.
- For any measurable $h$, transform draws directly: $h^{(s)}=h(\theta^{(s)})$.
- A Jacobian is needed when writing a transformed density, not when merely transforming draws.
```

```{note} Example
:class: dropdown
- Positive $\theta$ → unconstrained $\phi=\log\theta$.
- Transformed log posterior: $\ell_\phi(\phi)=\ell_\theta(e^\phi)+\phi$.
- Omitting $+\phi$ samples a different distribution.
```

```{attention} Q&A
:class: dropdown
*Why not fix a nuisance parameter at its estimate?*

- Plug-in conditioning discards its uncertainty and dependence with the target; marginalization retains both.
- Profiling maximizes over nuisance parameters; it is not integration.

*Does the likelihood itself acquire a parameter Jacobian?*

- No: $p(\mathcal{D}\mid\phi)=p(\mathcal{D}\mid g^{-1}(\phi))$.
- A parameter Jacobian belongs to the prior/posterior density because those are densities over parameters.

*What if the transformation is many-to-one?*

- Use the pushforward distribution; sum/integrate contributions from all preimages.
- The single inverse-Jacobian formula above does not apply unchanged.
```

&nbsp;

### Posterior Summaries & Credible Sets
- **What**: Point or set summaries of posterior uncertainty.
- **Why**: Reporting a full distribution is often impractical; summaries must preserve the relevant decision or uncertainty.
- **How**: Choose a target quantity first, then a loss-optimal point estimate or a set containing the desired posterior mass.

```{note} Math
:class: dropdown

$$
\begin{align*}
\hat\theta_{\mathrm{mean}}&=\mathbb{E}[\theta\mid\mathcal{D}],\\
\hat\theta_{\mathrm{MAP}}&\in\arg\max_\theta\pi(\theta\mid\mathcal{D}),\\
P(\theta\in C(\mathcal{D})\mid\mathcal{D})&\geq1-\alpha.
\end{align*}
$$

- $\hat\theta_{\mathrm{MAP}}$: Maximum a posteriori density estimate; requires a specified coordinate system.
- $C(\mathcal{D})$: Credible set; equality is usually available for continuous posteriors.
- A posterior mean requires an existing expectation; a mode need not exist or be unique.

Scalar continuous posterior:

$$
\begin{align*}
C_{\mathrm{ET}}&=[F^{-1}(\alpha/2),F^{-1}(1-\alpha/2)],\\
C_{\mathrm{HPD}}&=\{\theta:\pi(\theta\mid\mathcal{D})\geq c_\alpha\}.
\end{align*}
$$

- $F$: Posterior cumulative distribution function.
- $C_{\mathrm{ET}}$: Equal-tailed interval.
- $C_{\mathrm{HPD}}$: Highest posterior density set.
- $c_\alpha$: Density threshold chosen for the desired mass, with boundary handling if needed.
```

```{tip} Derivation
:class: dropdown
1. Squared-error posterior risk decomposes as

    $$
    \mathbb{E}[(\theta-a)^2\mid\mathcal{D}]
    =\operatorname{Var}(\theta\mid\mathcal{D})
    +(a-\mathbb{E}[\theta\mid\mathcal{D}])^2.
    $$

    - $a$: Reported scalar estimate.

2. The first term does not depend on $a$ → posterior mean minimizes risk when the second moment is finite.
3. Absolute-error risk has a minimum at a posterior median when the risk is finite.
4. Discrete zero-one loss selects the most probable state; continuous exact-match zero-one loss assigns risk $1$ to every point of a nonatomic posterior.
```

```{attention} Q&A
:class: dropdown
*Credible interval versus confidence interval?*

- Credible: posterior probability of the unknown lying in the reported set, conditional on this data & model.
- Confidence: repeated-sampling coverage of a procedure at a fixed true parameter.
- Bayesian credible sets have nominal coverage averaged over the proper prior predictive experiment, not necessarily at each fixed parameter.

*Is an HPD set always an interval?*

- No. Multimodality can require disconnected regions.
- It minimizes volume in the chosen coordinates under suitable density regularity; it is not invariant to nonlinear reparameterization.
- Equal-tailed intervals transform correctly under strictly monotone scalar transformations; MAP generally does not.

*Can a credible interval answer a point-null hypothesis?*

- Excluding zero answers an interval/tail-probability question.
- Under a continuous prior, $P(\theta=0\mid\mathcal{D})=0$ regardless of whether an interval contains zero.
- Positive posterior probability for an exact null requires prior mass on that null.
```

&nbsp;

### Posterior Concentration
- **What**: Shrinking posterior mass around data-supported parameter regions as sample size grows.
- **Why**: Clarify when prior effects vanish and Bayesian intervals resemble frequentist ones.
- **How**: Accumulated log-likelihood differences dominate fixed prior log-density differences when the model is sufficiently identifiable and regular.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\theta_0$: True parameter under a correctly specified model.
- Misc:
    - $I(\theta_0)$: Per-observation Fisher information.
    - $\hat\theta_{\mathrm{MLE}}$: Maximum likelihood estimate.

For a regular, fixed-dimensional, correctly specified iid model, a Bernstein–von Mises conclusion is

$$
\pi(\theta\mid\mathcal{D})
\approx
\mathcal{N}\!\left(\hat\theta_{\mathrm{MLE}},
\frac{1}{m}I(\theta_0)^{-1}\right).
$$

- Requires an interior identifiable truth, nonsingular information, local smoothness, a prior continuous & positive near truth, and conditions ensuring concentration/no competing distant mass.
- This is a large-sample approximation, not an identity for arbitrary models.
```

```{attention} Q&A
:class: dropdown
*Is consistency the same as asymptotic normality?*

- No. Consistency places posterior mass near truth; asymptotic normality specifies the local shape and $\sqrt m$ scale.
- Neither follows merely from writing Bayes' rule.

*What breaks the usual approximation?*

- Boundary parameters, mixture symmetries, singular information, growing dimension, or priors excluding truth.
- Under misspecification, the posterior may concentrate near a Kullback–Leibler minimizing model rather than the true distribution.
- Under regular misspecification, posterior curvature and the estimator's sampling covariance generally differ → naive credible intervals need not have nominal frequentist coverage.
```

&nbsp;

## Bayesian Models

### Bayesian Linear Regression
- **What**: Linear Gaussian observation model with a distribution over coefficients.
- **Why**: Least-squares coefficients alone omit uncertainty in the fitted relationship.
- **How**: Combine prior precision with data precision; retain the coefficient posterior instead of only its optimizer.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $X\in\mathbb{R}^{m\times n}$: Design matrix; include a constant column if fitting an intercept.
    - $\mathbf{y}\in\mathbb{R}^m$: Observed responses.
    - $\mathbf{x}_*\in\mathbb{R}^n$: New feature vector.
- Params:
    - $\boldsymbol{\beta}\in\mathbb{R}^n$: Regression coefficients.
- Hyperparams:
    - $\sigma^2>0$: Known observation-noise variance.
    - $\mathbf{b}_0\in\mathbb{R}^n$: Prior coefficient mean.
    - $V_0\in\mathbb{R}^{n\times n}$: Positive-definite prior covariance.

Model:

$$
\mathbf{y}\mid\boldsymbol{\beta},X\sim\mathcal{N}(X\boldsymbol{\beta},\sigma^2I_m),
\qquad
\boldsymbol{\beta}\sim\mathcal{N}(\mathbf{b}_0,V_0).
$$

Inference:

$$
\begin{align*}
V_m&=(V_0^{-1}+X^\top X/\sigma^2)^{-1},\\
\mathbf{b}_m&=V_m(V_0^{-1}\mathbf{b}_0+X^\top\mathbf{y}/\sigma^2),\\
\boldsymbol{\beta}\mid\mathcal{D}&\sim\mathcal{N}(\mathbf{b}_m,V_m),\\
\tilde y\mid\mathbf{x}_*,\mathcal{D}
&\sim\mathcal{N}(\mathbf{x}_*^\top\mathbf{b}_m,
\sigma^2+\mathbf{x}_*^\top V_m\mathbf{x}_*).
\end{align*}
$$

- $I_m$: Identity matrix.
- $V_m$: Posterior coefficient covariance.
- $\mathbf{b}_m$: Posterior coefficient mean.
- $\mathbf{x}_*^\top V_m\mathbf{x}_*$: Uncertainty in the latent mean; observation prediction adds $\sigma^2$.
```

```{tip} Derivation
:class: dropdown
1. Add negative log likelihood and negative log prior, dropping coefficient-independent terms:

    $$
    -2\ell(\boldsymbol{\beta})
    =\frac{\|\mathbf{y}-X\boldsymbol{\beta}\|^2}{\sigma^2}
    +(\boldsymbol{\beta}-\mathbf{b}_0)^\top V_0^{-1}
    (\boldsymbol{\beta}-\mathbf{b}_0)+C.
    $$

    - $C$: Coefficient-independent constant.

2. Collect the quadratic coefficient $V_m^{-1}$ and linear coefficient $V_m^{-1}\mathbf{b}_m$.
3. Complete the square → $(\boldsymbol{\beta}-\mathbf{b}_m)^\top V_m^{-1}(\boldsymbol{\beta}-\mathbf{b}_m)$ plus a constant.
4. A linear function of Gaussian coefficients is Gaussian; independent Gaussian observation noise adds variances.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class BayesianLinearRegression:
    def __init__(self, prior_mean, prior_cov, noise_var):
        self.b0 = np.asarray(prior_mean, dtype=float)
        self.v0 = np.asarray(prior_cov, dtype=float)
        if not np.isfinite(noise_var) or noise_var <= 0:
            raise ValueError("noise_var must be finite and positive")
        self.noise_var = noise_var

    def __call__(self, x, y, x_new):
        x, y, x_new = map(np.asarray, (x, y, x_new))
        eye = np.eye(self.b0.size)
        prior_precision = np.linalg.solve(self.v0, eye)
        precision = prior_precision + x.T @ x / self.noise_var
        rhs = prior_precision @ self.b0 + x.T @ y / self.noise_var
        mean = np.linalg.solve(precision, rhs)
        ## Solve against test vectors rather than explicitly invert precision.
        projected_cov = np.linalg.solve(precision, x_new.T)
        variance = self.noise_var + np.sum(x_new.T * projected_cov, axis=0)
        return x_new @ mean, variance

## Example: intercept-only model, prior variance = noise variance = 1
model = BayesianLinearRegression([0.0], [[1.0]], 1.0)
mean, variance = model(np.ones((2, 1)), [1.0, 1.0], np.ones((1, 1)))
assert np.allclose(mean, [2 / 3])
assert np.allclose(variance, [4 / 3])
print(mean, variance)
```
````

```{attention} Q&A
:class: dropdown
*Connection to ridge regression?*

- $\mathbf{b}_0=0$, $V_0=\tau^2I$ → posterior mean = MAP = minimizer of $\|\mathbf{y}-X\boldsymbol{\beta}\|^2+(\sigma^2/\tau^2)\|\boldsymbol{\beta}\|^2$.
- $\tau^2$: Prior coefficient variance.
- Ridge alone returns a point; Bayesian inference also retains $V_m$.

*What if $X$ is rank-deficient?*

- A positive-definite proper Gaussian prior still makes posterior precision positive definite.
- Data do not identify null-space directions; posterior information there comes from the prior.

*What assumptions matter?*

- Conditional linear mean, independent Gaussian errors, and constant known noise variance in this version.
- Feature scaling changes the meaning of a coefficient prior; an intercept often needs a separate prior scale.
```

&nbsp;

#### Unknown Noise Variance
- **What**: Joint coefficient & variance inference with a Normal–Inverse-Gamma prior.
- **Why**: Plugging in noise variance omits uncertainty in the residual scale.
- **How**: Condition coefficients on variance, then integrate variance out to obtain Student-$t$ uncertainty.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $a_0>0$: Inverse-Gamma shape.
    - $b_0>0$: Inverse-Gamma scale.
    - $\Lambda_0\in\mathbb{R}^{n\times n}$: Positive-definite relative coefficient precision.
- Misc:
    - $v$: Positive scalar variance value.

Model:

$$
\begin{align*}
\sigma^2&\sim\operatorname{InvGamma}(a_0,b_0),&
p(v)&=\frac{b_0^{a_0}}{\Gamma(a_0)}v^{-a_0-1}e^{-b_0/v},\\
\boldsymbol{\beta}\mid\sigma^2
&\sim\mathcal{N}(\mathbf{b}_0,\sigma^2\Lambda_0^{-1}),&
\mathbf{y}\mid\boldsymbol{\beta},\sigma^2,X
&\sim\mathcal{N}(X\boldsymbol{\beta},\sigma^2I_m).
\end{align*}
$$

- Overrides the parent's fixed $\sigma^2$ and variance-independent coefficient prior; $X,\mathbf{y},\mathbf{x}_*,\boldsymbol{\beta},\mathbf{b}_0$ retain their meanings.

Inference:

$$
\begin{align*}
\Lambda_m&=\Lambda_0+X^\top X,\\
\mathbf{b}_m&=\Lambda_m^{-1}(\Lambda_0\mathbf{b}_0+X^\top\mathbf{y}),\\
a_m&=a_0+m/2,\\
b_m&=b_0+\frac12\left[
\|\mathbf{y}-X\mathbf{b}_m\|^2+
(\mathbf{b}_m-\mathbf{b}_0)^\top\Lambda_0(\mathbf{b}_m-\mathbf{b}_0)
\right],\\
\sigma^2\mid\mathcal{D}&\sim\operatorname{InvGamma}(a_m,b_m),\\
\boldsymbol{\beta}\mid\sigma^2,\mathcal{D}
&\sim\mathcal{N}(\mathbf{b}_m,\sigma^2\Lambda_m^{-1}),\\
\tilde y\mid\mathbf{x}_*,\mathcal{D}
&\sim t_{2a_m}\!\left(\mathbf{x}_*^\top\mathbf{b}_m,
\frac{b_m}{a_m}(1+\mathbf{x}_*^\top\Lambda_m^{-1}\mathbf{x}_*)\right).
\end{align*}
$$

- $\Lambda_m$: Posterior relative coefficient precision.
- $\mathbf{b}_m$: Updated coefficient mean.
- $a_m$: Posterior inverse-Gamma shape.
- $b_m$: Posterior inverse-Gamma scale, distinct from bold $\mathbf{b}_m$.
- $t_\nu(\mu,s^2)$: Student-$t$ with degrees of freedom $\nu$, location $\mu$, and squared scale $s^2$; variance $s^2\nu/(\nu-2)$ only for $\nu>2$.
```

```{attention} Q&A
:class: dropdown
*Why is $a_m=a_0+m/2$, not $a_0+(m+n)/2$?*

- Integrating $n$ coefficients contributes a factor $(\sigma^2)^{n/2}$ that cancels their Gaussian normalization.
- The full conditional $\sigma^2\mid\boldsymbol{\beta},\mathcal{D}$ instead has shape $a_0+(m+n)/2$.

*Does this include the unknown Normal mean & variance model?*

- Yes: $X=\mathbf{1}_m$, $n=1$ gives the scalar Normal–Inverse-Gamma model.
- Independent priors on coefficients and variance are valid alternatives, but not this joint conjugate family.
```

&nbsp;

### Bayesian GLM
- **Name**: Bayesian Generalized Linear Model
- **What**: Exponential-family response model with a linked linear mean & coefficient priors.
- **Why**: Gaussian responses cannot represent binary outcomes or nonnegative counts correctly.
- **How**: Choose an observation family, link its mean to predictors, and infer the coefficients jointly with any dispersion parameters.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\boldsymbol{\beta}\in\mathbb{R}^n$: Regression coefficients.
- Misc:
    - $\eta_i=\mathbf{x}_i^\top\boldsymbol{\beta}$: Linear predictor.
    - $\mu_i=\mathbb{E}[y_i\mid\mathbf{x}_i,\boldsymbol{\beta}]$: Conditional mean.
    - $g$: Link function satisfying $g(\mu_i)=\eta_i$.

Model:

$$
\begin{align*}
\text{Binary:}\quad
y_i\mid\boldsymbol{\beta}
&\sim\operatorname{Bernoulli}(\operatorname{sigmoid}(\eta_i)),\\
\text{Counts:}\quad
y_i\mid\boldsymbol{\beta}
&\sim\operatorname{Poisson}(e^{\eta_i}).
\end{align*}
$$

- $\operatorname{sigmoid}(u)=1/(1+e^{-u})$.
- Multiclass: softmax category probabilities; fix a reference coefficient vector or impose another identifiability constraint.

Inference:

$$
\pi(\boldsymbol{\beta}\mid\mathcal{D})
\propto\pi(\boldsymbol{\beta})
\prod_{i=1}^m p(y_i\mid\mathbf{x}_i,\boldsymbol{\beta}).
$$

- Gaussian coefficient priors are generally not conjugate for logistic or log-Poisson likelihoods.
- Laplace, MCMC, or VI approximate this posterior.
```

```{attention} Q&A
:class: dropdown
*What changes under complete logistic separation?*

- An unregularized finite maximum likelihood estimate may not exist.
- A proper Gaussian coefficient prior yields a proper posterior and regularizes divergent coefficients; the likelihood itself remains separated.

*What if counts are more variable than Poisson allows?*

- Poisson imposes conditional variance = mean.
- A Gamma mixture of Poisson rates yields a Negative Binomial family; infer the additional dispersion.
- Excess zeros may motivate a hurdle or zero-inflated model, but those encode different data-generating mechanisms.

*Why not predict from the posterior mean coefficient?*

- Nonlinearity: $\mathbb{E}[\operatorname{sigmoid}(\eta)\mid\mathcal{D}]\neq\operatorname{sigmoid}(\mathbb{E}[\eta\mid\mathcal{D}])$ in general.
- Average response probabilities/rates over posterior draws.
```

&nbsp;

### Hierarchical Models
- **What**: Group-specific parameters drawn from a shared population distribution.
- **Why**: Complete pooling ignores group differences; separate fits waste shared information.
- **How**:
    1. Give each group local parameters.
    2. Link them through shared hyperparameters.
    3. Infer local and population parameters jointly → partial pooling.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathcal{D}_g$: Data in group $g$.
- Params:
    - $\theta_g$: Group-specific parameter.
    - $\phi$: Shared population hyperparameters.
- Misc:
    - $G$: Number of groups.

Model:

$$
\phi\sim\pi(\phi),
\qquad
\theta_g\mid\phi\sim\pi(\theta_g\mid\phi),
\qquad
\mathcal{D}_g\mid\theta_g\sim p(\mathcal{D}_g\mid\theta_g).
$$

Inference:

$$
\pi(\phi,\theta_{1:G}\mid\mathcal{D})
\propto\pi(\phi)\prod_{g=1}^G
\pi(\theta_g\mid\phi)p(\mathcal{D}_g\mid\theta_g).
$$

For Normal group estimates with known standard errors:

$$
\begin{align*}
\bar y_g\mid\theta_g&\sim\mathcal{N}(\theta_g,s_g^2),
&\theta_g\mid\mu,\tau&\sim\mathcal{N}(\mu,\tau^2),\\
\mathbb{E}[\theta_g\mid\bar y_g,\mu,\tau]
&=w_g\bar y_g+(1-w_g)\mu,
&w_g&=\frac{\tau^2}{\tau^2+s_g^2}.
\end{align*}
$$

- $\bar y_g$: Observed group estimate.
- $s_g>0$: Known standard error.
- $\mu$: Population mean.
- $\tau>0$: Between-group standard deviation.
- $w_g$: Conditional weight on the group estimate.
- Full Bayes averages this conditional answer over $\pi(\mu,\tau\mid\mathcal{D})$; assigning proper hyperpriors completes the model.
```

```{note} Example
:class: dropdown
- Conditional on $\mu=0,\tau=1$, a group estimate $\bar y_g=4$ with $s_g=2$ has posterior mean $4/5$.
- With $s_g=1/2$, the same estimate has posterior mean $16/5$.
- Weak/strong group evidence → more/less shrinkage toward the shared mean.
```

```{attention} Q&A
:class: dropdown
*What is exchangeability?*

- Invariance of a joint distribution to permutations of the group labels before conditioning on group-specific information.
- The hierarchy makes groups conditionally independent given hyperparameters, but generally dependent after integrating hyperparameters out.
- Group covariates can explain systematic differences before assuming residual exchangeability.

*Where are complete pooling and no pooling?*

- $\tau\to0$ forces group parameters toward a common mean.
- For fixed $s_g$, $\tau\to\infty$ makes the conditional mean approach the separate group estimate.
- These limits describe shrinkage, not permission to substitute an improper infinite-scale hyperprior.

*What changes for a new group?*

- Existing group: use its local posterior.
- New group: draw hyperparameters from their posterior, then draw a new local parameter from the population distribution.
```

&nbsp;

#### Empirical Bayes
- **What**: Plug-in estimation of prior hyperparameters from the observed data.
- **Why**: Full hyperparameter integration may be expensive or unnecessary for a chosen approximation.
- **How**: Fit hyperparameters by marginal likelihood, then condition on their estimate.

```{note} Math
:class: dropdown

$$
\begin{align*}
\hat\phi&\in\arg\max_\phi
\int p(\mathcal{D}\mid\theta)\pi(\theta\mid\phi)\,d\theta,\\
\pi_{\mathrm{EB}}(\theta\mid\mathcal{D})
&=\pi(\theta\mid\mathcal{D},\hat\phi).
\end{align*}
$$

- $\phi$: Prior hyperparameters, as in the parent hierarchy.
- $\hat\phi$: Marginal-likelihood estimate.
- $\pi_{\mathrm{EB}}$: Empirical Bayes posterior approximation.
```

```{attention} Q&A
:class: dropdown
*Is this the same as full Bayes?*

- No. Full Bayes integrates over $\pi(\phi\mid\mathcal{D})$; plug-in empirical Bayes ignores hyperparameter uncertainty and may understate uncertainty.
- Few groups or boundary variance estimates make that omission especially important.
- It is an explicit data-adaptive procedure, not a prior specified independently of the data.
```

&nbsp;

#### Non-Centered Parameterization
- **What**: Hierarchical effects expressed through independent standardized latent variables. {cite:p}`papaspiliopoulos2007general`
- **Why**: Small population scales can couple local effects tightly to hyperparameters.
- **How**: Sample standardized deviations, then transform them into group effects.

```{note} Math
:class: dropdown

$$
z_g\sim\mathcal{N}(0,1),
\qquad
\theta_g=\mu+\tau z_g,
\qquad \tau>0.
$$

- $z_g$: Standardized group deviation.
- $\theta_g,\mu,\tau$: Same group effect, population mean, and scale as in the parent model.
- This construction induces $\theta_g\mid\mu,\tau\sim\mathcal{N}(\mu,\tau^2)$.
- If transforming an existing density from $\theta_{1:G}$ to $z_{1:G}$, the Jacobian $\tau^G$ cancels the Normal prior's $\tau^{-G}$ factor; do not add it again to the direct standard-Normal construction.
```

```{attention} Q&A
:class: dropdown
*Which parameterization is better?*

- Weak group data often favor non-centering; strong group data can favor centering.
- They represent the same model but different computational geometry.
- Neither fixes a misspecified model or a genuinely unidentified likelihood.
```

&nbsp;

### Bayesian Mixture Models
- **What**: Mixture distributions with posterior uncertainty over weights, components & assignments.
- **Why**: A heterogeneous population may not be explained by one observation family.
- **How**: Introduce latent component labels, put priors on component parameters, and integrate or sample the labels.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{w}$: Nonnegative component weights summing to $1$.
    - $\psi_k$: Component $k$ parameters.
- Hyperparams:
    - $\mathbf{a}$: Positive Dirichlet concentrations.
- Misc:
    - $K$: Fixed number of components.
    - $z_i\in\{1,\ldots,K\}$: Latent assignment.

Model:

$$
\mathbf{w}\sim\operatorname{Dirichlet}(\mathbf{a}),
\qquad
z_i\mid\mathbf{w}\sim\operatorname{Categorical}(\mathbf{w}),
\qquad
y_i\mid z_i,\psi_{1:K}\sim p(y_i\mid\psi_{z_i}).
$$

- Specify proper component priors $\pi(\psi_k)$; Gaussian components require priors on means and positive-definite covariances.

Inference:

$$
p(y_i\mid\mathbf{w},\psi_{1:K})
=\sum_{k=1}^K w_k p(y_i\mid\psi_k).
$$

- Summing labels out gives a differentiable mixture likelihood when the component densities are differentiable.
- Evaluate its logarithm with log-sum-exp, not a sum of log component densities.
```

```{attention} Q&A
:class: dropdown
*How does this differ from EM for a GMM?*

- [GMM fitting by EM](../../ml/unsupervised.md#gmm) gives a likelihood/MAP point estimate with conditional assignment probabilities.
- Full Bayes also integrates uncertainty in mixture weights and component parameters.

*Why do component posterior means sometimes coincide?*

- Exchangeable component priors allow label switching: all permutations describe the same mixture.
- Prefer label-invariant targets such as predictive density or co-clustering probabilities; ordering constraints change label interpretation.
- A sampler trapped in one labeling is not necessarily exploring the full parameter posterior.

*Does a finite mixture infer the number of components?*

- Not automatically. Here $K$ is fixed.
- Random-$K$ and Dirichlet-process mixture models add different priors over complexity; number of occupied components is not automatically the number of real-world populations.
```

&nbsp;

### GP
- **Name**: Gaussian Process
- **What**: Function prior whose every finite set of evaluations is jointly Gaussian.
- **Why**: Model uncertainty in nonlinear functions without fixing a finite coefficient basis.
- **How**: Encode covariance through a kernel, then condition on observations; [Gaussian conditioning formulas & code](../../ml/optim.md#gp).

```{note} Math
:class: dropdown
Notations:
- Params:
    - $f$: Latent regression function.
    - $\phi$: Kernel & observation-noise hyperparameters.
- Misc:
    - $\mu_\phi$: Prior mean function.
    - $k_\phi$: Positive-semidefinite covariance kernel.
    - $f_*=f(\mathbf{x}_*)$: Latent value at a new input.

Model:

$$
f\mid\phi\sim\mathcal{GP}(\mu_\phi,k_\phi),
\qquad
y_i=f(\mathbf{x}_i)+\epsilon_i,
\qquad
\epsilon_i\overset{\mathrm{iid}}{\sim}\mathcal{N}(0,\sigma^2).
$$

- $\sigma^2$: Noise variance included in $\phi$.

Inference:

$$
p(f_*\mid\mathcal{D})
=\int p(f_*\mid\mathcal{D},\phi)\pi(\phi\mid\mathcal{D})\,d\phi.
$$

- Fixed $\phi$ + Gaussian observations → Gaussian conditional posterior.
- Hyperparameter integration generally gives a non-Gaussian mixture of those conditionals.
```

```{attention} Q&A
:class: dropdown
*Why call a GP nonparametric?*

- Its latent function representation can grow with observed input locations; it still has kernel hyperparameters.
- Optimizing those hyperparameters is empirical Bayes, not full hyperparameter integration.

*Are Gaussian observations required?*

- No. Classification can use Bernoulli/softmax likelihoods, but the posterior over function values is then generally non-Gaussian.
- Kernel choice expresses substantive assumptions about smoothness, periodicity & extrapolation.
```

&nbsp;

### Bayesian Neural Networks
- **What**: Neural observation models with a joint posterior over weights.
- **Why**: A single fitted network does not represent weight uncertainty.
- **How**: Put a prior on weights, infer their joint posterior, and average predictions across weight draws.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{w}$: All network weights.
- Misc:
    - $f_{\mathbf{w}}$: Network function.

Model:

$$
\mathbf{w}\sim\pi(\mathbf{w}),
\qquad
y_i\mid\mathbf{x}_i,\mathbf{w}
\sim p(y_i\mid f_{\mathbf{w}}(\mathbf{x}_i)).
$$

Inference:

$$
\pi(\mathbf{w}\mid\mathcal{D})
\propto\pi(\mathbf{w})\prod_i p(y_i\mid f_{\mathbf{w}}(\mathbf{x}_i)).
$$

- A Gaussian prior gives weight-decay MAP under a matching scaling, not a full posterior.
- VI, Laplace approximations, and MCMC make different compromises on this high-dimensional distribution.
```

```{attention} Q&A
:class: dropdown
*Does Bayesian imply calibrated out-of-distribution uncertainty?*

- No. Prior, architecture, likelihood & approximation can all be wrong for the shifted data.
- Weight uncertainty is not the same as uncertainty over every plausible function or model family.

*Do weight symmetries prevent useful prediction?*

- Permuting hidden units can leave the function unchanged → multiple parameter modes.
- Function-space predictions can be meaningful even when individual weight summaries are not.
```

&nbsp;

### Bayesian Networks
- **What**: Directed acyclic graphical factorizations of joint distributions.
- **Why**: Make conditional-independence assumptions explicit in multivariable models.
- **How**: Assign a conditional distribution to each node given its parents; condition on observations and marginalize hidden nodes.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{v}=(v_1,\ldots,v_J)$: Node values.
- Misc:
    - $J$: Number of nodes.
    - $\operatorname{pa}(j)$: Parents of node $j$ in the graph.

Model:

$$
p(\mathbf{v}\mid\theta)
=\prod_{j=1}^J p(v_j\mid\mathbf{v}_{\operatorname{pa}(j)},\theta).
$$

Inference:

$$
p(\mathbf{v}_{\mathrm{mis}}\mid\mathbf{v}_{\mathrm{obs}},\mathcal{D})
=\int p(\mathbf{v}_{\mathrm{mis}}\mid\mathbf{v}_{\mathrm{obs}},\theta)
\pi(\theta\mid\mathbf{v}_{\mathrm{obs}},\mathcal{D})\,d\theta.
$$

- $\mathbf{v}_{\mathrm{obs}}$: Observed nodes for the queried case.
- $\mathbf{v}_{\mathrm{mis}}$: Unobserved nodes.
- $\mathcal{D}$: Training cases; avoid conditioning on an observation twice if already included.
```

```{attention} Q&A
:class: dropdown
*Does “Bayesian network” mean parameters have priors?*

- No. The graph is a factorization; parameters can be fixed, fitted by maximum likelihood, or inferred Bayesianly.
- [Naive Bayes](../../ml/supervised.md#naive-bayes) is a restricted graph with features conditionally independent given class.

*Does an arrow establish causation?*

- No. Observational factorization alone does not justify intervention semantics.
- Causal interpretation needs additional structural assumptions; Bayesian updating does not supply them.

*Can missing observations simply be ignored?*

- Integrate missing values under the joint model.
- Ignoring the missingness mechanism additionally needs suitable ignorability conditions; informative missingness can require an explicit model.
```

&nbsp;

#### State-Space Models
- **What**: Sequential latent-state models with noisy observations.
- **Why**: Time-dependent data violate the iid likelihood and contain hidden evolving structure.
- **How**: Alternate state prediction through a transition model and correction by the new observation.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $z_t$: Latent state at time $t$.
- Misc:
    - $T$: Final observation time.

Model:

$$
p(z_{1:T},y_{1:T}\mid\theta)
=p(z_1\mid\theta)
\prod_{t=2}^T p(z_t\mid z_{t-1},\theta)
\prod_{t=1}^T p(y_t\mid z_t,\theta).
$$

Inference, conditional on $\theta$:

$$
p(z_t\mid y_{1:t},\theta)
\propto p(y_t\mid z_t,\theta)
\int p(z_t\mid z_{t-1},\theta)
p(z_{t-1}\mid y_{1:t-1},\theta)\,dz_{t-1}.
$$

- **Filtering**: $p(z_t\mid y_{1:t},\theta)$.
- **Smoothing**: $p(z_t\mid y_{1:T},\theta)$ for $t<T$.
- Full Bayesian inference also integrates unknown static $\theta$.
```

```{attention} Q&A
:class: dropdown
*Which major families fit here?*

- Finite discrete states → hidden Markov models, with exact forward/backward recursions.
- Linear transitions & observations, Gaussian initial state/noises, known static parameters → Kalman filtering/smoothing.
- Nonlinear or non-Gaussian models → approximations such as particle filters.

*Can random train/test splits evaluate forecasting?*

- Not safely: they can expose future information.
- Evaluate the intended forecast horizon using temporally ordered holdouts.
```

&nbsp;

### Robust Regression
- **What**: Regression with an observation model less sensitive to large residuals.
- **Why**: Outliers are not automatically handled by putting a prior on coefficients.
- **How**: Replace Gaussian residuals with a heavier-tailed model, such as a Student-$t$ scale mixture.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mu_i$: Conditional location for observation $i$.
    - $\sigma>0$: Residual scale.
    - $\nu>0$: Student-$t$ degrees of freedom.
    - $\lambda_i>0$: Latent relative precision.

Model:

$$
\lambda_i\sim\operatorname{Gamma}(\nu/2,\nu/2),
\qquad
y_i\mid\lambda_i\sim\mathcal{N}(\mu_i,\sigma^2/\lambda_i)
\quad\Longrightarrow\quad
y_i\sim t_\nu(\mu_i,\sigma^2).
$$

- Gamma uses shape-rate; $\sigma^2$ is Student-$t$ squared scale, not generally its variance.
- Set $\mu_i=\mathbf{x}_i^\top\boldsymbol{\beta}$ for linear regression; assign proper priors to coefficients, scale & any unknown degrees of freedom.
```

```{attention} Q&A
:class: dropdown
*Why does the Student-$t$ resist outliers?*

- An unusually large residual can be explained by small latent precision rather than forcing a large change in the shared location.
- It does not repair dependence, selection bias, or a wrong conditional mean.

*Is the scale-mixture representation a different model?*

- No. Integrating out the independent Gamma precisions gives exactly the Student-$t$ observation model.
- Gaussian errors are recovered as $\nu\to\infty$; small $\nu$ gives heavier tails, and moments may not exist.
```

&nbsp;

### Survival & Censoring
- **What**: Event-time distributions with likelihoods for partially observed outcomes.
- **Why**: A censored time bounds the event time; treating it as an exact event misrepresents the data.
- **How**: Use event densities for observed events and survival/interval probabilities for censored records.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $u_i$: Observed event/censoring time.
    - $\delta_i$: Event indicator, $1$ for observed event, $0$ for right censoring.

Model, under conditionally independent, noninformative right censoring:

$$
p(\mathcal{D}\mid\theta)\propto
\prod_i f_\theta(u_i)^{\delta_i}
S_\theta(u_i)^{1-\delta_i},
\qquad
S_\theta(u)=P(T_{\mathrm{event}}>u\mid\theta).
$$

- $f_\theta$: Event-time density.
- $S_\theta$: Survival function.
- $T_{\mathrm{event}}$: Random event time.
- Omitted censoring factors must be independent of event-model unknowns; specify priors on those unknowns.
```

```{attention} Q&A
:class: dropdown
*Which event-time models?*

- Exponential: constant hazard. Weibull: monotone hazard. Lognormal: a Normal model for log event time.
- Add covariates through a specified regression structure and put priors on its coefficients & baseline-distribution parameters.

*Censoring versus truncation?*

- Censoring: a record exists, but its value is only partially known.
- Truncation: records outside a selection region are absent; normalize the likelihood by the probability of inclusion.
- Informative censoring requires additional modeling, not merely the survival factor above.
```

&nbsp;

## Prediction, Evaluation, and Decisions

### Posterior Predictive Distribution
- **What**: Distribution of future observations after integrating posterior uncertainty.
- **Why**: Parameter estimates are not the same object as predictions.
- **How**: Draw parameters from the posterior, then draw new observations from their conditional sampling model.

```{note} Math
:class: dropdown

$$
p(\tilde y\mid\mathcal{D})
=\int p(\tilde y\mid\theta,\mathcal{D})\pi(\theta\mid\mathcal{D})\,d\theta.
$$

- Under conditional independence of future and observed data given $\theta$, use $p(\tilde y\mid\theta)$.
- For regression, condition additionally on the new input.

When second moments exist:

$$
\begin{align*}
\mathbb{E}[\tilde y\mid\mathcal{D}]
&=\mathbb{E}_{\theta\mid\mathcal{D}}
[\mathbb{E}[\tilde y\mid\theta,\mathcal{D}]],\\
\operatorname{Var}(\tilde y\mid\mathcal{D})
&=\mathbb{E}_{\theta\mid\mathcal{D}}
[\operatorname{Var}(\tilde y\mid\theta,\mathcal{D})]\\
&\quad+\operatorname{Var}_{\theta\mid\mathcal{D}}
(\mathbb{E}[\tilde y\mid\theta,\mathcal{D}]).
\end{align*}
$$

- First variance term: conditional observation noise.
- Second variance term: uncertainty in the conditional mean.
- This decomposition is relative to the model and its conditioning information.
```

```{note} Example
:class: dropdown
For the earlier $\operatorname{Beta}(9,5)$ posterior:

$$
\begin{align*}
P(\tilde y=1\mid\mathcal{D})&=9/14,\\
P(\tilde r=k\mid\mathcal{D})
&=\binom{N}{k}\frac{B(9+k,5+N-k)}{B(9,5)},\\
\operatorname{Var}(\tilde r\mid\mathcal{D})
&=N\frac9{14}\frac5{14}\frac{14+N}{15}.
\end{align*}
$$

- $\tilde r$: Success count in $N$ future conditionally independent trials sharing the same unknown probability.
- $N$: Number of future trials.
- $B$: Beta function.
- The Beta–Binomial variance exceeds plug-in Binomial variance for $N>1$; future trials become dependent after integrating their shared probability.
```

```{attention} Q&A
:class: dropdown
*How do joint predictive draws differ from separate marginal draws?*

- For a future batch sharing parameters, draw one $\theta^{(s)}$, then all batch outcomes conditional on it.
- Redrawing an independent parameter for every outcome removes the dependence induced by shared uncertainty.

*How is predictive log density estimated?*

- Compute $\operatorname{LSE}_{s}\log p(\tilde y\mid\theta^{(s)},\mathcal{D})-\log S$.
- Averaging log likelihoods estimates a different quantity; log of the average is not average of the logs.

*Predictive versus credible interval?*

- Predictive interval concerns a future observation.
- A credible interval for a latent mean concerns only that mean's posterior uncertainty.
```

&nbsp;

### Model Checking
- **What**: Comparing observable model implications with actual or plausible data.
- **Why**: A mathematically correct posterior can answer the wrong question under a bad likelihood or prior.
- **How**:
    1. **Prior predictive**: Check the generative model before fitting.
    2. **Posterior predictive**: Compare replicated datasets with the observed one.
    3. Target discrepancies tied to the intended use; revise the model when important structure is unexplained.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{D}^{\mathrm{rep}}$: Replicated dataset with a specified design.
    - $T$: Diagnostic statistic, such as a tail count or residual autocorrelation.
    - $p_B$: Posterior predictive tail probability.

$$
\begin{align*}
\theta^{(s)}&\sim\pi(\theta\mid\mathcal{D}),\\
\mathcal{D}^{\mathrm{rep},(s)}&\sim p(\cdot\mid\theta^{(s)}),\\
p_B&=P\!\left(T(\mathcal{D}^{\mathrm{rep}})\geq T(\mathcal{D})
\mid\mathcal{D}\right).
\end{align*}
$$

- Match replication to the target: same observed inputs, new inputs, existing groups, or new groups imply different checks.
- Parameter-dependent discrepancies can use $T(\mathcal{D},\theta)$ and $T(\mathcal{D}^{\mathrm{rep}},\theta)$ with the same posterior draw.
```

```{attention} Q&A
:class: dropdown
*Which discrepancies reveal useful failures?*

- Tail frequency, zero counts, skewness, residual patterns versus predictors, within-group variation & serial dependence.
- A fitted mean can match well while these features are wrong.

*Is $p_B$ an ordinary uniform-null p-value?*

- No. Data are used to fit the posterior and assess discrepancy; its null distribution is not generally uniform.
- It is not the posterior probability that the model is true.

*Does posterior predictive checking measure out-of-sample accuracy?*

- Not by itself. It checks compatibility; held-out predictive scoring answers a different question.
- Passing a chosen set of checks does not prove the model correct.
```

&nbsp;

#### SBC
- **Name**: Simulation-Based Calibration {cite:p}`talts2018validating`
- **What**: Prior-predictive simulation check of posterior computation.
- **Why**: A computational approximation can be wrong even when the statistical model is correctly implemented on paper.
- **How**: Simulate parameters and data, refit, and rank each generating quantity among its posterior draws.

```{note} Math
:class: dropdown

$$
\theta_0\sim\pi(\theta),
\qquad
\mathcal{D}_0\sim p(\mathcal{D}\mid\theta_0),
\qquad
\theta^{(s)}\overset{\mathrm{iid}}{\sim}\pi(\theta\mid\mathcal{D}_0).
$$

- $\theta_0$: Generating parameter draw.
- $\mathcal{D}_0$: Simulated dataset.

For a continuous test quantity:

$$
r=\sum_{s=1}^S
\mathbf{1}\{h(\theta^{(s)})<h(\theta_0)\}
\sim\operatorname{Uniform}\{0,\ldots,S\}.
$$

- $r$: Rank; uniformity is across repeated simulated experiments under exact iid posterior sampling.
- Ties require randomized ranking; correlated draws require appropriate dependence handling.
```

```{attention} Q&A
:class: dropdown
*Why uniform ranks?*

- Conditional on simulated data, the generating parameter and exact independent posterior draws are exchangeable.

*What can SBC not establish?*

- Real-world model adequacy; its data were generated from the assumed model.
- Uniform marginal ranks are not sufficient to prove a correct joint posterior. For example, ignoring data and drawing from the prior can pass parameter-rank checks.
- Use informative test quantities and complementary checks.
```

&nbsp;

### Bayes Factors & Model Averaging
- **What**: Evidence-based posterior weighting of competing probabilistic models.
- **Why**: Uncertainty may concern model structure as well as parameters within one structure.
- **How**: Integrate each model's likelihood over its prior, multiply by prior model probability, and normalize.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $M_k$: Candidate model indexed by $k$.
    - $\theta_k$: Its unknown parameters.
    - $Z_k$: Its evidence.
    - $B_{12}$: Bayes factor comparing models $1$ and $2$.

$$
\begin{align*}
Z_k&=\int p(\mathcal{D}\mid\theta_k,M_k)
\pi(\theta_k\mid M_k)\,d\theta_k,\\
P(M_k\mid\mathcal{D})
&=\frac{Z_kP(M_k)}{\sum_j Z_jP(M_j)},\\
\frac{P(M_1\mid\mathcal{D})}{P(M_2\mid\mathcal{D})}
&=\underbrace{\frac{Z_1}{Z_2}}_{B_{12}}
\frac{P(M_1)}{P(M_2)},\\
p(\tilde y\mid\mathcal{D})
&=\sum_k p(\tilde y\mid\mathcal{D},M_k)P(M_k\mid\mathcal{D}).
\end{align*}
$$

- Model averaging additionally includes parameter uncertainty within each predictive component.
- Evidence is prior-average fit, not maximized likelihood or posterior-average likelihood.
```

```{tip} Derivation
:class: dropdown
1. Treat the model index as a discrete unknown.
2. Integrate its model-specific parameters to obtain the likelihood for that index.
3. Apply Bayes' rule to the index.
4. Sum over the posterior model index for prediction.
```

```{note} Example
:class: dropdown
- $M_0$: Bernoulli probability fixed at $1/2$.
- $M_1$: Bernoulli probability uniform on $[0,1]$.
- Observe the ordered sequence success, success:

    $$
    Z_0=(1/2)^2=1/4,\qquad
    Z_1=\int_0^1\theta^2\,d\theta=1/3,\qquad
    B_{10}=4/3.
    $$

- Equal prior model probabilities → $P(M_1\mid\mathcal{D})=4/7$.
- The flexible model's maximum likelihood is $1$, but its evidence is $1/3$.
```

```{attention} Q&A
:class: dropdown
*Where is the complexity penalty?*

- Broad prior regions predicting the observed data poorly dilute the average likelihood.
- This is prior-volume sensitivity, not a universal fixed penalty per parameter.
- More diffuse alternative priors can favor a point null even when a frequentist test rejects it.

*Can evidence establish which model is true?*

- Only relative to the supplied candidate models & priors; all candidates may be wrong.
- High evidence is not a substitute for predictive checks or evaluation against the actual deployment target.

*Is predictive stacking Bayesian model averaging?*

- No. Stacking chooses mixture weights for predictive performance, typically from cross-validation; those weights are not posterior model probabilities.
```

&nbsp;

### Predictive Scoring & Cross-Validation
- **What**: Assessment of predictive distributions on observations excluded from fitting.
- **Why**: In-sample fit rewards adaptation to noise and does not estimate future performance directly.
- **How**: Define the prediction unit, hold it out, and score its full predictive distribution.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $p_*$: Actual data-generating distribution.
    - $q$: Candidate predictive distribution.
    - $\mathcal{D}_{-i}$: Dataset excluding prediction unit $i$.
    - $\operatorname{elpd}_{\mathrm{LOO}}$: Leave-one-out estimate of expected log predictive density, on a total-data scale.

Log score:

$$
\mathbb{E}_{p_*}[\log p_*(Y)]
-\mathbb{E}_{p_*}[\log q(Y)]
=\operatorname{KL}(p_*\|q)\geq0.
$$

- $Y$: Fresh outcome; common dominating measure and finite terms are assumed.
- Log score is strictly proper: the true distribution maximizes expected score, up to almost-everywhere equality.

Leave-one-out cross-validation:

$$
\operatorname{elpd}_{\mathrm{LOO}}
=\sum_{i=1}^m\log p(y_i\mid\mathcal{D}_{-i}),
\qquad
p(y_i\mid\mathcal{D}_{-i})
=\int p(y_i\mid\theta)\pi(\theta\mid\mathcal{D}_{-i})\,d\theta.
$$

- Factorized conditional likelihood assumed here; inputs are conditioned on where appropriate.
- Larger elpd is better. It estimates prediction after training on $m-1$ units, not a model probability.
```

```{attention} Q&A
:class: dropdown
*Which unit should be held out?*

- New row in an existing group → row-level holdout may be appropriate.
- New group → hold out groups and integrate unseen group effects.
- Forecasting → leave future data out; random splitting can leak future information.
- Feature transformations and hyperparameter selection must respect the same split.

*How uncertain is a model comparison?*

- Compare paired pointwise score differences, not unrelated standard errors for each model.
- Similar predictions, small samples, dependence & model selection can make simple uncertainty summaries unreliable.
- Repeatedly selecting against one validation set can overfit that set.

*Calibration versus sharpness?*

- Calibration: predictive probabilities/intervals agree with observed frequencies in the relevant assessment.
- Sharpness: concentration of predictions; narrow wrong intervals are not better forecasts.
```

&nbsp;

#### PSIS-LOO
- **Name**: Pareto-Smoothed Importance Sampling Leave-One-Out Cross-Validation {cite:p}`vehtari2017practical`
- **What**: Approximate leave-one-out prediction by reweighting full-posterior draws.
- **Why**: Refitting once per observation can be expensive.
- **How**: Undo each held-out likelihood contribution with importance weights, smooth extreme weights, and diagnose unreliable approximations.

```{note} Math
:class: dropdown
For a factorized likelihood:

$$
\pi(\theta\mid\mathcal{D}_{-i})
\propto\frac{\pi(\theta\mid\mathcal{D})}{p(y_i\mid\theta)}.
$$

For full-posterior draws, raw ratios and smoothed predictions are

$$
r_{is}=\frac1{p(y_i\mid\theta^{(s)})},
\qquad
\hat p(y_i\mid\mathcal{D}_{-i})
=\sum_s\bar w_{is}p(y_i\mid\theta^{(s)}).
$$

- $r_{is}$: Raw importance ratio for observation $i$ and draw $s$.
- $\bar w_{is}$: Normalized Pareto-smoothed weight.
- [PSIS](sampling.md#psis) regularizes the upper weight tail; its $\hat k$ diagnoses tail difficulty.
```

```{attention} Q&A
:class: dropdown
*What if a point is influential?*

- Leaving it out can change the posterior enough to produce unstable importance weights.
- Large $\hat k$ → inspect that point/model and use exact refitting or suitable $K$-fold validation when needed.
- More posterior draws do not cure missing proposal support.

*Can latent variables leak the held-out answer?*

- Yes, if the scored predictive quantity conditions on a local latent effect estimated using the held-out outcome.
- Match the likelihood contribution and latent-variable integration to the intended prediction task.
```

&nbsp;

#### WAIC
- **Name**: Widely Applicable Information Criterion {cite:p}`watanabe2010asymptotic`
- **What**: Posterior-variance correction to in-sample log predictive density.
- **Why**: Predictive accuracy needs a correction for fitting and scoring the same observations.
- **How**: Subtract the sum of pointwise posterior log-likelihood variances from fitted log predictive density.

```{note} Math
:class: dropdown

$$
\begin{align*}
\operatorname{lppd}
&=\sum_i\log\left(\frac1S\sum_s p(y_i\mid\theta^{(s)})\right),\\
p_{\mathrm{WAIC}}
&=\sum_i\widehat{\operatorname{Var}}_s
\left(\log p(y_i\mid\theta^{(s)})\right),\\
\widehat{\operatorname{elpd}}_{\mathrm{WAIC}}
&=\operatorname{lppd}-p_{\mathrm{WAIC}},\\
\operatorname{WAIC}
&=-2\widehat{\operatorname{elpd}}_{\mathrm{WAIC}}.
\end{align*}
$$

- $\operatorname{lppd}$: Log pointwise predictive density.
- $p_{\mathrm{WAIC}}$: Effective complexity correction, not generally the literal parameter count.
- Lower WAIC corresponds to higher estimated elpd; pointwise factorization and prediction unit must be meaningful.
```

```{attention} Q&A
:class: dropdown
*Is WAIC exactly leave-one-out?*

- No. Their asymptotic equivalence needs conditions; finite-sample discrepancies can be large.
- Influential observations or weakly constrained posteriors are reasons to inspect diagnostics and refit rather than trust a single corrected score.

*Why not always use BIC or DIC?*

- BIC approximates log evidence under regular large-sample assumptions; it targets a different quantity.
- DIC relies on a point-summary notion of deviance/complexity that can be problematic for mixtures and other nonregular posteriors.
```

&nbsp;

### Sensitivity Analysis
- **What**: Assessment of how conclusions change under defensible modeling alternatives.
- **Why**: A narrow posterior conditional on one specification can hide substantial uncertainty about that specification.
- **How**: Vary priors, likelihoods, influential observations & identification assumptions; compare the actual target summaries and decisions.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\pi_0$: Baseline prior.
    - $\pi_1$: Alternative prior.
    - $w(\theta)=\pi_1(\theta)/\pi_0(\theta)$: Prior density ratio.
    - $\mathbb{E}_0$: Expectation under the baseline posterior.

With the same likelihood and compatible support:

$$
\mathbb{E}_1[h(\theta)\mid\mathcal{D}]
=
\frac{\mathbb{E}_0[h(\theta)w(\theta)\mid\mathcal{D}]}
{\mathbb{E}_0[w(\theta)\mid\mathcal{D}]}.
$$

- $\mathbb{E}_1$: Expectation under the alternative posterior.
- Requires finite normalizers and no alternative posterior mass outside baseline support.
- Large or unstable weights → refit; reweighting cannot create unvisited regions.
```

```{attention} Q&A
:class: dropdown
*Which alternatives deserve attention?*

- Prior scale & tail shape; alternative links/noise families; pooling assumptions; selection/missingness mechanisms; plausible influential-data treatment.
- Study weakly identified quantities and Bayes factors especially carefully.

*Is sensitivity a defect?*

- Sometimes it correctly reveals that data cannot resolve assumptions.
- Report the range of defensible conclusions rather than tune assumptions until the desired result appears.

*How do computational and model sensitivity differ?*

- Different valid samplers targeting the same posterior should agree within numerical error.
- Different plausible models can legitimately disagree even with exact computation.
```

&nbsp;

### Bayesian Decision Theory
- **What**: Action selection by minimizing posterior expected loss.
- **Why**: Probabilities alone do not specify which errors or outcomes matter.
- **How**: Define feasible actions and their consequences, average loss over uncertainty, then choose the lowest-risk action.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{A}$: Feasible action set.
    - $a\in\mathcal{A}$: Action.
    - $L(a,\theta)$: Loss if action $a$ is taken and state is $\theta$.
    - $\rho(a\mid\mathcal{D})$: Posterior expected loss.

Objective:

$$
\rho(a\mid\mathcal{D})
=\int L(a,\theta)\pi(\theta\mid\mathcal{D})\,d\theta,
\qquad
a^*(\mathcal{D})\in\arg\min_{a\in\mathcal{A}}\rho(a\mid\mathcal{D}).
$$

- $a^*$: Bayes action, when a minimizer exists.
- If loss depends on a future outcome, integrate over its posterior predictive distribution instead.
- Risk must exist for the comparison being made.
```

```{tip} Derivation
:class: dropdown
1. For a binary state, let $p=P(\theta=1\mid\mathcal{D})$ and assign zero loss to correct decisions.
2. Let $C_{\mathrm{FP}}>0$ and $C_{\mathrm{FN}}>0$ be false-positive and false-negative costs.
3. Risks are $\rho(1)=C_{\mathrm{FP}}(1-p)$ and $\rho(0)=C_{\mathrm{FN}}p$.
4. Choose action $1$ exactly when

    $$
    p>\frac{C_{\mathrm{FP}}}{C_{\mathrm{FP}}+C_{\mathrm{FN}}},
    $$

    with either action allowed at equality.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class BayesAction:
    def __init__(self, losses):
        self.losses = np.asarray(losses, dtype=float)  ## [actions, states]
        if self.losses.ndim != 2 or min(self.losses.shape) == 0 or not np.isfinite(self.losses).all():
            raise ValueError("Expected a nonempty finite loss matrix")

    def __call__(self, probabilities):
        p = np.asarray(probabilities, dtype=float)
        if p.shape != (self.losses.shape[1],) or np.any(p < 0) or not np.isclose(p.sum(), 1):
            raise ValueError("Expected state probabilities summing to one")
        risk = self.losses @ p
        return int(np.argmin(risk)), risk

## Example: false negative costs 4, false positive costs 1
decision = BayesAction([[0, 4], [1, 0]])
action, risk = decision([0.7, 0.3])
assert action == 1
assert np.allclose(risk, [1.2, 0.7])
print(action, risk)
```
````

```{attention} Q&A
:class: dropdown
*Which posterior point summary is the right decision?*

- Squared error → posterior mean.
- Absolute error → posterior median.
- Asymmetric absolute loss → the quantile determined by the cost ratio.
- Discrete zero-one loss → most probable state; continuous MAP requires a different, coordinate-sensitive justification.

*Why can the same posterior lead to different actions?*

- Different costs, constraints, or utilities define different optimization problems.
- A probability threshold of $1/2$ is justified only by the corresponding symmetric binary costs.

*Does a predictive association justify an intervention?*

- No. Action-dependent outcomes require a defensible intervention model, not merely $p(y\mid x,\mathcal{D})$.
- A posterior cannot compensate for an unidentified causal effect.
```

&nbsp;

#### Value of Information
- **What**: Expected reduction in optimal decision loss from additional information.
- **Why**: More accurate inference is valuable only insofar as it changes consequential decisions.
- **How**: Average the best achievable risk after a proposed experiment, then compare it with current risk and experiment cost.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $e$: Proposed experiment/design.
    - $Y_e$: Future result under design $e$.
    - $R(\mathcal{D})=\min_a\rho(a\mid\mathcal{D})$: Current optimal risk.
    - $c(e)\geq0$: Experiment cost in loss units.

Objective:

$$
\operatorname{EVSI}(e)
=R(\mathcal{D})
-\mathbb{E}_{Y_e\mid\mathcal{D},e}
\left[R(\mathcal{D},Y_e,e)\right].
$$

- $\operatorname{EVSI}$: Expected value of sample information.
- Net value is $\operatorname{EVSI}(e)-c(e)$.
- With the same action set/loss and information that can be ignored, $\operatorname{EVSI}\geq0$: acting as before remains feasible after observing the result.
```

```{attention} Q&A
:class: dropdown
*Is maximum information gain the same objective?*

- No. Expected KL gain measures how much the posterior changes; EVSI measures expected improvement in decisions.
- Information about an irrelevant parameter can be statistically large and decision-theoretically worthless.

*What bounds the value of an experiment?*

- Under the same decision problem, perfect knowledge of the unknown state cannot be less useful than a noisy measurement of it.
- An experiment that changes the underlying state or feasible actions must model those consequences, not only its informational benefit.
```

&nbsp;