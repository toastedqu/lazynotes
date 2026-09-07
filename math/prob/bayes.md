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
Study notes from {cite:t}`bayes,bayes2`; variational inference follows the review by {cite:t}`blei2017variational`.
Inference → model construction → computation → prediction & decisions.

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

## Computation

### Numerical Integration
- **What**: Deterministic approximation of posterior integrals by weighted evaluations.
- **Why**: Bayes' rule defines the answer; its normalizer and expectations may lack closed forms.
- **How**: Evaluate the unnormalized posterior on integration nodes, then normalize their weighted mass.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\theta_r$: Integration node indexed by $r$.
    - $w_r>0$: Quadrature weight, including local volume.
    - $R$: Number of nodes.

Process:

$$
\begin{align*}
Z&\approx\sum_{r=1}^R w_r\tilde p(\theta_r),\\
\mathbb{E}[h(\theta)\mid\mathcal{D}]
&\approx
\frac{\sum_r w_r\tilde p(\theta_r)h(\theta_r)}
{\sum_r w_r\tilde p(\theta_r)}.
\end{align*}
$$

- Stable normalized log weights: $\log\bar w_r=\ell(\theta_r)+\log w_r-\operatorname{LSE}_{u}(\ell(\theta_u)+\log w_u)$.
- $\bar w_r$: Approximate posterior mass at a node, not a density value.
- $\operatorname{LSE}(a_1,\ldots,a_R)=a_{\max}+\log\sum_r e^{a_r-a_{\max}}$: Log-sum-exp.
- $a_{\max}=\max_r a_r$: Shift preventing overflow.
```

```{attention} Q&A
:class: dropdown
*When does a grid work?*

- Low dimension, with tails covered and resolution fine enough near concentrated mass.
- $R_0$ nodes per coordinate require $R_0^d$ evaluations → dimensionality rapidly defeats tensor grids.
- Refine spacing and expand bounds separately; a normalized grid can still miss most posterior mass.

*Which error does more computation reduce?*

- Integration/approximation error, not posterior uncertainty from limited data.
- Use a conjugate model as an exact reference before trusting the same implementation on an intractable model.
```

&nbsp;

### Laplace Approximation
- **What**: Local Gaussian approximation around an interior posterior mode.
- **Why**: Replace a difficult density with one determined by its local curvature.
- **How**: Find a mode, evaluate negative log-posterior curvature there, and use inverse curvature as covariance.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\hat\theta$: Interior mode.
    - $H=-\nabla^2\ell(\hat\theta)$: Positive-definite negative Hessian.
    - $q_L$: Laplace approximating density.

Process:

$$
\begin{align*}
\ell(\theta)&\approx\ell(\hat\theta)
-\frac12(\theta-\hat\theta)^\top H(\theta-\hat\theta),\\
q_L(\theta)&=\mathcal{N}(\theta;\hat\theta,H^{-1}),\\
Z&\approx
\tilde p(\hat\theta)(2\pi)^{d/2}|H|^{-1/2}.
\end{align*}
$$

- The evidence formula requires the actual joint density $\tilde p$, including all normalizing constants.
- Constrained parameters may need transformation to unconstrained coordinates, including the prior/posterior Jacobian.
```

```{tip} Derivation
:class: dropdown
1. Taylor-expand the log posterior at an interior stationary point; the first derivative vanishes.
2. A negative-definite Hessian yields a Gaussian-shaped quadratic log density.
3. Diagonalize $H$; each coordinate integrates to a one-dimensional Gaussian factor → $(2\pi)^{d/2}|H|^{-1/2}$.
```

```{attention} Q&A
:class: dropdown
*When does local curvature mislead?*

- Skewness, heavy tails, boundaries, ridges, or multiple important modes.
- One accurate local expansion can still miss most global posterior mass.
- A mode need not lie in a high-probability-mass region in high dimensions.

*Laplace versus Bernstein–von Mises?*

- Laplace is a computational approximation around a mode.
- Bernstein–von Mises is an asymptotic theorem requiring statistical regularity; it does not certify every finite-sample Laplace approximation.
```

&nbsp;

### Monte Carlo Integration
- **What**: Integral estimation by averages of random draws.
- **Why**: Random sampling can avoid the exponential node count of dense multidimensional grids.
- **How**: Draw from the target, evaluate the quantity of interest, and average.

```{note} Math
:class: dropdown
For iid posterior draws:

$$
\hat\mu_h=\frac1S\sum_{s=1}^S h(\theta^{(s)})
\xrightarrow{\mathrm{a.s.}}
\mu_h=\mathbb{E}[h(\theta)\mid\mathcal{D}].
$$

- $\mu_h$: Target expectation.
- $\hat\mu_h$: Monte Carlo estimate.
- Almost-sure convergence requires $\mathbb{E}[|h|\mid\mathcal{D}]<\infty$.

With finite variance:

$$
\sqrt S(\hat\mu_h-\mu_h)
\xrightarrow{d}\mathcal{N}(0,\sigma_h^2),
\qquad
\widehat{\operatorname{MCSE}}(\hat\mu_h)
=\sqrt{\frac{\hat\sigma_h^2}{S}}.
$$

- $\sigma_h^2=\operatorname{Var}(h(\theta)\mid\mathcal{D})$: Posterior variance of the target.
- $\hat\sigma_h^2$: Sample variance with divisor $S-1$.
- $\operatorname{MCSE}$: Monte Carlo standard error.
- The $S^{-1/2}$ error rate has no dimension exponent, but sampling cost and variance can deteriorate strongly with dimension.
```

```{attention} Q&A
:class: dropdown
*Posterior standard deviation versus Monte Carlo standard error?*

- Posterior standard deviation: uncertainty about the target given the data/model.
- MCSE: numerical uncertainty in an estimated posterior summary.
- More draws reduce MCSE; they do not create more observations.

*What changes with MCMC draws?*

- Dependence alters the variance of the average.
- Replace $S$ by a quantity-specific effective sample size only under a suitable Markov-chain central limit theorem.
```

&nbsp;

#### Rao–Blackwellization
- **What**: Replacing a sampled quantity by its computable conditional expectation.
- **Why**: Sampling a nuisance variable adds avoidable Monte Carlo noise.
- **How**: Integrate analytically where possible, simulate only the remaining unknowns.

```{note} Math
:class: dropdown
For $\theta=(\psi,\lambda)$, define

$$
g(\psi)=\mathbb{E}[h(\psi,\lambda)\mid\psi,\mathcal{D}].
$$

- $\psi$: Simulated unknowns.
- $\lambda$: Analytically integrated unknowns.
- $g$: Conditional expectation.

$$
\begin{align*}
\mathbb{E}[g(\psi)\mid\mathcal{D}]&=\mathbb{E}[h(\theta)\mid\mathcal{D}],\\
\operatorname{Var}(h\mid\mathcal{D})
&=\mathbb{E}[\operatorname{Var}(h\mid\psi,\mathcal{D})\mid\mathcal{D}]
+\operatorname{Var}(g(\psi)\mid\mathcal{D}).
\end{align*}
$$

- For iid draws, averaging $g$ has no greater variance than averaging $h$ for the same $S$.
```

```{attention} Q&A
:class: dropdown
*A predictive example?*

- Estimate $P(\tilde y=1\mid\mathcal{D})$ by averaging conditional probabilities $P(\tilde y=1\mid\theta^{(s)})$.
- Simulating binary outcomes and averaging their indicators has extra simulation noise.

*Does collapsing always improve an MCMC estimator?*

- Not automatically. Removing variables can alter mixing and cost; the iid variance comparison alone does not establish lower Markov-chain asymptotic variance.
```

&nbsp;

### Importance Sampling
- **What**: Monte Carlo integration with density-ratio correction for proposal draws.
- **Why**: Sampling the posterior directly may be hard while sampling a nearby distribution is easy.
- **How**: Sample from a proposal, upweight draws underrepresented by that proposal, and normalize if evidence is unknown.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $q$: Normalized proposal density, positive wherever the target has mass.
    - $w_s$: Unnormalized importance weight.
    - $\bar w_s$: Normalized importance weight.

Process:

$$
\begin{align*}
\theta^{(s)}&\overset{\mathrm{iid}}{\sim}q,
&w_s&=\frac{\tilde p(\theta^{(s)})}{q(\theta^{(s)})},
&\bar w_s&=\frac{w_s}{\sum_r w_r},\\
\hat Z&=\frac1S\sum_s w_s,
&\hat\mu_{\mathrm{IS}}&=\sum_s\bar w_s h(\theta^{(s)}),
&\widehat{\operatorname{ESS}}_w&=\frac{1}{\sum_s\bar w_s^2}.
\end{align*}
$$

- $\hat Z$: Unbiased evidence estimator when the proposal covers the target and the exact joint density is evaluable.
- $\hat\mu_{\mathrm{IS}}$: Self-normalized estimate; generally biased at finite $S$, consistent under integrability conditions.
- $\widehat{\operatorname{ESS}}_w$: Weight-concentration heuristic, not the MCMC effective sample size and not a proof of accuracy.
```

```{tip} Derivation
:class: dropdown
1. Multiply and divide by $q$:

    $$
    \mathbb{E}_{\pi}[h]
    =\frac{\int q(\theta)\,\tilde p(\theta)h(\theta)/q(\theta)\,d\theta}
    {\int q(\theta)\,\tilde p(\theta)/q(\theta)\,d\theta}.
    $$

2. Approximate numerator and denominator with averages under $q$.
3. Their ratio yields normalized importance weights; this ratio is why unbiased numerator/denominator estimates do not imply an unbiased final estimate.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class ImportanceSampling:
    def __init__(self, log_joint, log_proposal):
        self.log_joint = log_joint
        self.log_proposal = log_proposal

    def __call__(self, draws, values):
        log_w = self.log_joint(draws) - self.log_proposal(draws)
        values = np.asarray(values)
        if log_w.ndim != 1 or values.shape != log_w.shape:
            raise ValueError("Expected one log weight and value per draw")
        if not np.any(np.isfinite(log_w)) or np.any(np.isnan(log_w) | np.isposinf(log_w)):
            raise ValueError("Importance weights are undefined")
        ## Normalize in log space; the unknown evidence cancels.
        w = np.exp(log_w - np.max(log_w))
        w /= w.sum()
        return w @ values, 1 / (w @ w)

## Example: target N(0, 1), proposal N(0, 4); constants cancel in normalized weights
rng = np.random.default_rng(0)
draws = rng.normal(0, 2, size=1000)
estimate = ImportanceSampling(lambda z: -z**2 / 2, lambda z: -z**2 / 8)
print(estimate(draws, draws**2))  ## posterior second moment, weight ESS
assert np.allclose(estimate(np.array([-1.0, 1.0]), np.ones(2)), (1.0, 2.0))
```
````

```{attention} Q&A
:class: dropdown
*Why are thin-tailed proposals dangerous?*

- Rare tail draws can receive enormous weights; even finite evidence need not imply finite weight variance.
- Missing a mode is not repaired by renormalizing observed weights.
- A large empirical ESS can be falsely reassuring if problematic regions were never visited.

*How does rejection sampling differ?*

- With a finite envelope $\tilde p(\theta)\leq Mq(\theta)$, accept a proposal with probability $\tilde p(\theta)/(Mq(\theta))$.
- Accepted draws are iid from the target; acceptance rate is $Z/M$.
- $M$: Envelope constant. Finding a tight global bound is often the bottleneck.
```

&nbsp;

#### PSIS
- **Name**: Pareto-Smoothed Importance Sampling {cite:p}`vehtari2024pareto`
- **What**: Importance sampling with regularized extreme weights & a tail diagnostic.
- **Why**: A few enormous weights can dominate an importance estimate.
- **How**: Fit a generalized Pareto distribution to the upper ratio tail, replace extreme ratios by smoothed order statistics, then normalize.

```{attention} Q&A
:class: dropdown
*What does the fitted tail shape diagnose?*

- $\hat k$: Estimated Pareto tail-shape parameter; larger values indicate more difficult importance sampling.
- It diagnoses proposal-target mismatch and weight-tail behavior, not posterior correctness.

*Can smoothing repair arbitrary proposals?*

- No. Missing target support or important unvisited modes still require a better proposal or refitting.
- Smoothing trades finite-sample bias against instability; it is not an exact independent sampler.
```

&nbsp;

### MCMC
- **Name**: Markov Chain Monte Carlo
- **What**: Monte Carlo integration through a Markov chain with the posterior as invariant distribution.
- **Why**: Direct independent sampling is unavailable for many joint posteriors.
- **How**: Construct valid local transitions, run the chain, and use ergodic averages after addressing initialization and mixing.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $K(\theta,A)$: Probability of moving from $\theta$ into measurable set $A$.
    - $\Pi$: Posterior probability measure.

$$
\Pi(A)=\int K(\theta,A)\,\Pi(d\theta).
$$

- This is **invariance**: one transition preserves the posterior when already at stationarity.
- Positive Harris recurrence gives convergence of ergodic averages for posterior-integrable quantities; aperiodicity additionally supports ordinary convergence in distribution.
- Invariance alone does not guarantee exploration or practical convergence.
```

```{attention} Q&A
:class: dropdown
*Is detailed balance necessary?*

- No. It is a convenient sufficient condition for invariance.
- Valid nonreversible chains also exist.

*Does discarding warmup guarantee convergence?*

- No. Warmup is an opportunity to reduce initialization effects and adapt tuning parameters, not a certificate.
- Standard adaptive samplers freeze adaptation before retained sampling; indefinite arbitrary adaptation can invalidate the target.

*Why keep repeated states?*

- Rejections are part of the transition kernel and its stationary weighting.
- Keeping accepted proposals only generally samples a different distribution.
```

&nbsp;

#### Metropolis–Hastings
- **What**: Proposal-and-correction transition preserving the target distribution.
- **Why**: A convenient proposal generally does not preserve posterior probability mass.
- **How**: Propose a move; accept according to target density and reverse/forward proposal probability.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $q(\theta'\mid\theta)$: Proposal transition density.
    - $a(\theta,\theta')$: Acceptance probability.

Process:

$$
a(\theta,\theta')
=\min\left(1,
\frac{\tilde p(\theta')q(\theta\mid\theta')}
{\tilde p(\theta)q(\theta'\mid\theta)}\right).
$$

- Propose $\theta'\sim q(\cdot\mid\theta)$; move with probability $a$, otherwise retain $\theta$.
- Symmetric random-walk proposals cancel the $q$ ratio.
- Unknown evidence cancels in the target ratio.
```

```{tip} Derivation
:class: dropdown
1. For distinct states, accepted flow equals

    $$
    \pi(\theta\mid\mathcal{D})q(\theta'\mid\theta)a(\theta,\theta')
    =
    \min\{\pi(\theta\mid\mathcal{D})q(\theta'\mid\theta),
    \pi(\theta'\mid\mathcal{D})q(\theta\mid\theta')\}.
    $$

2. Swapping $\theta,\theta'$ leaves this unchanged → detailed balance.
3. Rejected probability remains at the current state → the complete transition kernel preserves the target.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class RandomWalkMetropolis:
    def __init__(self, log_target, initial, scale, seed=0):
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Proposal scale must be finite and positive")
        self.log_target, self.x = log_target, float(initial)
        self.log_p = float(log_target(self.x))
        if not np.isfinite(self.log_p):
            raise ValueError("Initial state must have finite log density")
        self.scale, self.rng = scale, np.random.default_rng(seed)

    def step(self):
        proposal = self.x + self.rng.normal(0, self.scale)
        log_p = float(self.log_target(proposal))
        if np.isnan(log_p) or np.isposinf(log_p):
            raise ValueError("Invalid proposed log density")
        ## -Exponential(1) has the same law as log Uniform(0, 1).
        if -self.rng.exponential() < min(0.0, log_p - self.log_p):
            self.x, self.log_p = proposal, log_p
        return self.x

## Example: retain repeated states after rejection
chain = RandomWalkMetropolis(lambda z: -z**2 / 2, initial=0, scale=1)
draws = np.array([chain.step() for _ in range(1000)])
print(draws.shape)  ## (1000,); diagnose mixing before using summaries
```
````

```{attention} Q&A
:class: dropdown
*Is high acceptance always good?*

- Tiny moves can yield high acceptance and very slow exploration.
- Huge moves can yield frequent rejection.
- Judge efficiency by target-specific MCSE per compute cost, not acceptance alone.

*Can random walks handle a correlated posterior?*

- They can be valid but mix slowly; transform coordinates, block updates, or use gradient-informed proposals.
- Local proposals may fail to cross low-density gaps between modes.
```

&nbsp;

#### Gibbs Sampling
- **What**: MCMC updates from exact full conditional distributions.
- **Why**: Joint sampling can be difficult even when each parameter block has a tractable conditional.
- **How**: Cycle through blocks, conditioning each update on the newest values of the others.

```{note} Math
:class: dropdown
For blocks $\theta=(\theta_1,\ldots,\theta_B)$:

$$
\theta_b^{(s+1)}
\sim
\pi\!\left(\theta_b\mid
\theta_{1:b-1}^{(s+1)},\theta_{b+1:B}^{(s)},\mathcal{D}\right).
$$

- $B$: Number of blocks.
- $b$: Current block index.
- Exact full-conditional updates preserve the joint posterior; compositions preserve invariance even when a systematic sweep is not reversible.
- A Metropolis update inside a block gives Metropolis-within-Gibbs, not an exact conditional draw.
```

```{attention} Q&A
:class: dropdown
*Why can an always-accepted sampler be slow?*

- Strong dependence can make each conditional narrow → coordinate updates barely traverse the joint distribution.
- Deterministic constraints can destroy irreducibility.
- Blocking, marginalization, or reparameterization can help more than additional iterations.
```

&nbsp;

#### HMC
- **Name**: Hamiltonian Monte Carlo {cite:p}`duane1987hybrid`
- **What**: Gradient-driven MCMC using auxiliary momentum & approximate Hamiltonian trajectories.
- **Why**: Random walks waste proposals exploring correlated high-dimensional posteriors.
- **How**: Draw momentum, integrate a trajectory, then correct numerical energy error with a Metropolis step.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\epsilon>0$: Leapfrog step size.
    - $L$: Number of leapfrog steps.
- Misc:
    - $\mathbf{r}\in\mathbb{R}^d$: Auxiliary momentum.
    - $M$: Symmetric positive-definite mass matrix.
    - $H(\theta,\mathbf{r})$: Hamiltonian.

Process:

$$
H(\theta,\mathbf{r})=-\ell(\theta)+\tfrac12\mathbf{r}^\top M^{-1}\mathbf{r},
\qquad
\mathbf{r}\sim\mathcal{N}(0,M).
$$

Repeat the leapfrog step $L$ times:

$$
\begin{align*}
\mathbf{r}&\leftarrow\mathbf{r}+\tfrac{\epsilon}{2}\nabla\ell(\theta),\\
\theta&\leftarrow\theta+\epsilon M^{-1}\mathbf{r},\\
\mathbf{r}&\leftarrow\mathbf{r}+\tfrac{\epsilon}{2}\nabla\ell(\theta).
\end{align*}
$$

Flip final momentum for a reversible proposal; accept with probability

$$
\min\{1,\exp[H(\theta,\mathbf{r})-H(\theta',\mathbf{r}')]\}.
$$

- Primed values: proposed endpoint; unprimed values in the acceptance formula: pre-trajectory state.
- Leapfrog is reversible & volume-preserving; correction removes integration bias under ordinary MCMC conditions, not finite-run error.
```

```{attention} Q&A
:class: dropdown
*What can HMC not handle directly?*

- Discrete unknowns lack the required gradients; marginalize them or use another update.
- Funnels and separated modes remain difficult; gradients do not guarantee global exploration.
- Work in unconstrained coordinates with the correct transformed density.
```

&nbsp;

##### NUTS
- **Name**: No-U-Turn Sampler {cite:p}`hoffman2014no`
- **What**: HMC variant with dynamically selected trajectory length.
- **Why**: A fixed trajectory can stop too soon or retrace already explored regions.
- **How**: Expand a trajectory in both time directions, stop on a valid U-turn criterion, and select a state using a balance-preserving rule.

```{attention} Q&A
:class: dropdown
*Why not just stop ordinary HMC when it turns?*

- State-dependent stopping can bias sampling; valid tree construction & state selection are essential.
- NUTS is not ordinary HMC with an arbitrary early-exit condition.

*Does it remove all tuning?*

- No. Step size and mass matrix still need adaptation.
- Divergences signal unreliable numerical trajectories; merely increasing the iteration count is not a remedy.
- Reaching a tree-depth limit can indicate inefficiency without being the same failure as a divergence.
```

&nbsp;

### MCMC Diagnostics
- **What**: Empirical assessments of chain agreement & Monte Carlo precision. {cite:p}`vehtari2021rank`
- **Why**: Correct transition formulas do not establish that a finite run explored the posterior.
- **How**: Compare dispersed chains, inspect rank/trace plots, and estimate precision for every reported quantity.

```{note} Math
:class: dropdown
For a stationary scalar chain satisfying a central limit theorem:

$$
\operatorname{Var}(\hat\mu_h)
\approx\frac{\sigma_h^2}{S}
\left(1+2\sum_{t=1}^{\infty}\rho_h(t)\right),
\qquad
S_{\mathrm{eff},h}
=\frac{S}{1+2\sum_{t=1}^{\infty}\rho_h(t)}.
$$

- $\sigma_h^2$: Posterior variance of $h$.
- $\rho_h(t)$: Lag-$t$ autocorrelation of $h(\theta^{(s)})$.
- $S_{\mathrm{eff},h}$: Effective sample size for estimating the mean of $h$; requires a finite positive asymptotic variance.
- Estimate mean MCSE using $\hat\sigma_h/\sqrt{\hat S_{\mathrm{eff},h}}$.
```

```{attention} Q&A
:class: dropdown
*Which diagnostics?*

- Rank-normalized split/folded $\widehat R$: between-chain disagreement, drift & scale mismatch.
- Bulk/tail effective sample sizes: central summaries versus tail/quantile precision.
- HMC divergences: unresolved trajectory geometry; inspect alongside chain diagnostics.

*Does $\widehat R$ near $1$ prove convergence?*

- No. All chains can miss the same mode.
- Diagnose transformed quantities and predictions too; diagnostics cannot establish model adequacy.

*Should samples be thinned?*

- Usually not solely to reduce autocorrelation: discarding computed draws generally wastes information.
- Storage constraints can justify thinning; negative autocorrelation can even give effective sample size above the retained draw count.
```

&nbsp;

### VI
- **Name**: Variational Inference
- **What**: Posterior approximation by optimization over a restricted distribution family.
- **Why**: Repeated posterior integration or MCMC can be costly.
- **How**: Choose a tractable family and optimize its evidence lower bound.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\lambda$: Variational parameters, not model parameters.
- Misc:
    - $q_\lambda(\theta)$: Normalized approximating density.
    - $\mathcal{Q}$: Allowed variational family.
    - $\mathcal{L}(\lambda)$: Evidence lower bound, abbreviated ELBO.
    - $\operatorname{KL}(q\|p)=\mathbb{E}_q[\log(q/p)]$: Kullback–Leibler divergence.

Objective:

$$
\begin{align*}
\lambda^*&\in\arg\min_\lambda
\operatorname{KL}(q_\lambda\|\pi(\cdot\mid\mathcal{D})),\\
\mathcal{L}(\lambda)&=
\mathbb{E}_{q_\lambda}[\ell(\theta)-\log q_\lambda(\theta)],\\
\log Z&=\mathcal{L}(\lambda)
+\operatorname{KL}(q_\lambda\|\pi(\cdot\mid\mathcal{D})).
\end{align*}
$$

- The identity assumes the terms exist and $q_\lambda$ gives no mass where the posterior has zero density.
- Maximizing the ELBO minimizes this reverse KL; it need not recover the exact posterior unless the family contains it and optimization finds the global optimum.

For a mean-field family:

$$
q(\theta)=\prod_{b=1}^B q_b(\theta_b),
\qquad
\log q_b^*(\theta_b)
=\mathbb{E}_{q_{-b}}[\ell(\theta)]+\text{constant}.
$$

- $B$: Number of parameter blocks.
- $q_b$: Variational density for block $b$.
- $q_{-b}$: Product of the other block factors.
- The coordinate optimum additionally requires the resulting factor to normalize.
```

```{tip} Derivation
:class: dropdown
1. Substitute $\log\pi(\theta\mid\mathcal{D})=\ell(\theta)-\log Z$ into the definition of KL.
2. Rearrange to obtain the ELBO identity; nonnegative KL gives the lower bound.
3. Holding $q_{-b}$ fixed, the ELBO terms involving $q_b$ are

    $$
    \int q_b(\theta_b)
    \left\{\mathbb{E}_{q_{-b}}[\ell(\theta)]-\log q_b(\theta_b)\right\}\,d\theta_b.
    $$

4. Add a Lagrange multiplier for $\int q_b=1$; functional differentiation yields the coordinate update.
```

```{note} Example
:class: dropdown
- Target: zero-mean bivariate Normal with marginal variances $1$ and correlation $\rho$, $|\rho|<1$.
- Reverse-KL-optimal independent Gaussian factors each have variance $1-\rho^2$, not $1$.
- Mean-field removes covariance and, in this example, understates marginal uncertainty increasingly as $|\rho|$ grows.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class MeanFieldGaussian(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(dim))
        self.log_std = nn.Parameter(torch.zeros(dim))

    def forward(self, log_joint, samples):
        q = torch.distributions.Normal(self.mean, self.log_std.exp())
        ## Reparameterized draws let gradients pass through the sampled weights.
        theta = self.mean + self.log_std.exp() * torch.randn(
            samples, self.mean.numel(), device=self.mean.device, dtype=self.mean.dtype
        )
        log_q = q.log_prob(theta).sum(dim=-1)
        return (log_joint(theta) - log_q).mean()

## Example: one ascent step toward a shifted unit Gaussian target
q = MeanFieldGaussian(2)
target = torch.distributions.Normal(torch.tensor([1.0, -1.0]), torch.ones(2))
optimizer = torch.optim.SGD(q.parameters(), lr=0.01)
optimizer.zero_grad()
elbo = q(lambda theta: target.log_prob(theta).sum(dim=-1), samples=16)
(-elbo).backward()
optimizer.step()
print(q.mean.shape)  ## torch.Size([2]); one update, not a converged fit
```
````

```{attention} Q&A
:class: dropdown
*Why can reverse KL miss modes?*

- It penalizes placing $q$ mass where posterior density is small, but evaluates the penalty only where $q$ visits.
- A restricted family may prefer one mode over spreading density across a low-density gap.
- Underdispersion is common, not a theorem for every target and variational family.

*Is a high ELBO enough?*

- It compares approximations for the same normalized model target; additive constants must match.
- A local optimum or restrictive family can still give poor tail probabilities and decisions.
- Richer covariance/flow families improve expressiveness but do not guarantee successful optimization.

*How does stochastic VI use minibatches?*

- For factorized likelihoods, an unbiased log-joint estimator scales the minibatch log-likelihood sum by $m/$batch size.
- Include the prior term once; do not scale it as if it were per-observation data.
- Unbiased stochastic gradients do not make the approximate posterior unbiased.
```

&nbsp;

### SMC
- **Name**: Sequential Monte Carlo
- **What**: Weighted-particle approximation to a sequence of target distributions.
- **Why**: Streaming observations or staged targets favor updating an existing approximation.
- **How**: Propagate particles, update weights, and resample when weight concentration becomes excessive.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $z_t^{(s)}$: State particle at time $t$.
    - $w_t^{(s)}$: Unnormalized particle weight.
    - $q_t$: Proposal for the next state.

For the [state-space model](#state-space-models), without resampling at the current propagation step:

$$
\begin{align*}
z_t^{(s)}
&\sim q_t(\cdot\mid z_{t-1}^{(s)},y_t),\\
w_t^{(s)}
&=w_{t-1}^{(s)}
\frac{
p(y_t\mid z_t^{(s)})
p(z_t^{(s)}\mid z_{t-1}^{(s)})
}{
q_t(z_t^{(s)}\mid z_{t-1}^{(s)},y_t)
}.
\end{align*}
$$

- Condition on fixed static parameters here.
- Transition proposal $q_t=p(z_t\mid z_{t-1})$ → incremental weight is just the observation likelihood.
- Resampling selects ancestors using normalized weights and resets particle weights to equal values before the next update.
```

```{attention} Q&A
:class: dropdown
*Does resampling add information?*

- No. It reallocates computation toward supported particles and duplicates ancestors.
- Repeated resampling can cause path degeneracy; rejuvenation moves or better proposals may be needed.

*Can SMC infer static parameters too?*

- Yes, using staged/tempered targets and appropriate mutation moves.
- Treating a static parameter as a particle that never moves typically degenerates under repeated resampling.
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
- [PSIS](#psis) regularizes the upper weight tail; its $\hat k$ diagnoses tail difficulty.
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