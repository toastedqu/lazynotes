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
# Statistics
Frequentist inference: estimate parameters, quantify uncertainty, test hypotheses, run experiments.
Point estimation → interval estimation → hypothesis testing → multiple testing → online experiments.
Bayesian counterparts live in [Bayesian](bayes.md).

Notations:
- $X_1,\dots,X_n$: Sample, i.i.d. from the population unless stated.
- $\theta$: True (unknown) parameter, scalar or vector.
- $\Theta$: Parameter space.
- $\hat\theta_n$: Estimator of $\theta$ from $n$ samples.
- $f(x;\theta)$: Model PDF/PMF.
- $L(\theta)=\prod_{i=1}^n f(X_i;\theta)$: Likelihood.
- $\ell(\theta)=\log L(\theta)$: Log-likelihood.
- $\bar{X}_n=\frac{1}{n}\sum_{i=1}^nX_i$: Sample mean.
- $s^2=\frac{1}{n-1}\sum_{i=1}^n(X_i-\bar{X}_n)^2$: Sample variance.
- $I(\theta)$: Fisher information of one observation.
- $H_0$: Null hypothesis.
- $H_1$: Alternative hypothesis.
- $\alpha$: Significance level (type I error budget).
- $\Phi$: Standard Normal CDF.
- $z_q=\Phi^{-1}(q)$: Standard Normal $q$-quantile.

Overrides Tier 1: $n$ = #samples here (not #features).

&nbsp;

## Point Estimation

### Estimator
- **What**: Function of the sample that guesses $\theta$: $\hat\theta_n=T(X_1,\dots,X_n)$.
- **Why**: $\theta$ is unknown → only a random sample is seen → need a rule & a way to grade the rule.
- **How**:
    1. Fix the rule $T$ before seeing data.
    2. Sample is random → $\hat\theta_n$ is a random variable w/ a **sampling distribution**.
    3. Grade the rule by that distribution: center (bias), spread (variance), total error (MSE), behavior as $n\to\infty$ (consistency, efficiency).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbb{E}_\theta,\operatorname{Var}_\theta$: Expectation / variance w/ $X_i\sim f(\cdot;\theta)$.
    - $\xrightarrow{P}$: Convergence in probability.
    - $\xrightarrow{d}$: Convergence in distribution.

Definitions:

$$
\begin{align*}
\operatorname{Bias}_\theta(\hat\theta_n)&=\mathbb{E}_\theta[\hat\theta_n]-\theta\\
\operatorname{se}(\hat\theta_n)&=\sqrt{\operatorname{Var}_\theta(\hat\theta_n)}\\
\operatorname{MSE}_\theta(\hat\theta_n)&=\mathbb{E}_\theta\left[(\hat\theta_n-\theta)^2\right]=\operatorname{Bias}_\theta(\hat\theta_n)^2+\operatorname{Var}_\theta(\hat\theta_n)
\end{align*}
$$

- $\operatorname{se}$: Standard error = SD of the estimator, not of the data.
- **Unbiased**: $\operatorname{Bias}_\theta(\hat\theta_n)=0$ for all $\theta\in\Theta$.
- Vector $\theta$: $\mathbb{E}_\theta[(\hat\theta_n-\theta)(\hat\theta_n-\theta)^\top]=\operatorname{Cov}_\theta(\hat\theta_n)+\mathbf{b}\mathbf{b}^\top$.
    - $\mathbf{b}$: Bias vector.

Asymptotic properties:
- **Consistent**: $\hat\theta_n\xrightarrow{P}\theta$ for all $\theta\in\Theta$, i.e., $P_\theta(|\hat\theta_n-\theta|>\varepsilon)\to0$ for all $\varepsilon>0$.
    - Sufficient: $\operatorname{Bias}\to0$ & $\operatorname{Var}\to0$ → $\operatorname{MSE}\to0$ → consistent.
- **Asymptotically normal**: $\sqrt{n}(\hat\theta_n-\theta)\xrightarrow{d}\mathcal{N}(0,v(\theta))$.
    - $v(\theta)$: Asymptotic variance.
- **Asymptotically efficient**: $v(\theta)=I(\theta)^{-1}$ → smallest possible for regular estimators ([Cramér–Rao Bound](#cram-r-rao-bound)).
- **Asymptotic relative efficiency** of $T_1$ vs $T_2$: $v_2(\theta)/v_1(\theta)$ = ratio of sample sizes for equal precision.
```

```{tip} Derivation
:class: dropdown
*Why $\operatorname{MSE}=\operatorname{Bias}^2+\operatorname{Var}$?*

1. Add & subtract the mean: $\hat\theta_n-\theta=(\hat\theta_n-\mathbb{E}\hat\theta_n)+(\mathbb{E}\hat\theta_n-\theta)$.
2. Square & take expectation:

    $$
    \operatorname{MSE}=\mathbb{E}[(\hat\theta_n-\mathbb{E}\hat\theta_n)^2]+2(\mathbb{E}\hat\theta_n-\theta)\,\mathbb{E}[\hat\theta_n-\mathbb{E}\hat\theta_n]+(\mathbb{E}\hat\theta_n-\theta)^2
    $$

3. Middle term: $\mathbb{E}[\hat\theta_n-\mathbb{E}\hat\theta_n]=0$ → vanishes.
4. → $\operatorname{Var}+\operatorname{Bias}^2$.

*Why does $\operatorname{MSE}\to0$ imply consistency?*

1. [Markov](limits.md#markovs-inequality) on $(\hat\theta_n-\theta)^2$: $P(|\hat\theta_n-\theta|>\varepsilon)\le\operatorname{MSE}/\varepsilon^2\to0$.
```

```{note} Example
:class: dropdown
- **Unbiased, not consistent**: $\hat\mu_n=X_1$.
    - $\mathbb{E}[X_1]=\mu$, but $\operatorname{Var}=\sigma^2$ for every $n$ → never concentrates.
- **Consistent, biased**: $\hat\sigma^2_\text{MLE}=\frac1n\sum_i(X_i-\bar X_n)^2$.
    - Bias $=-\sigma^2/n\to0$, $\operatorname{Var}\to0$ → consistent.
- **Consistent, bias $\not\to0$**: $\hat\theta_n=\bar X_n$ w/ prob $1-\frac1n$, $=n^2$ w/ prob $\frac1n$.
    - $P(\hat\theta_n\ne\bar X_n)=\frac1n\to0$ → consistent.
    - $\mathbb{E}[\hat\theta_n]\approx\mu+n\to\infty$ → bias explodes.
- **Relative efficiency**: Normal data, median vs mean.
    - $\operatorname{Var}(\text{median})\approx\frac{\pi\sigma^2}{2n}$ vs $\operatorname{Var}(\bar X_n)=\frac{\sigma^2}{n}$ → ARE $=2/\pi\approx0.64$.
    - → median needs $\approx57\%$ more samples for the same precision.
- $\mu,\sigma^2$: Population mean & variance.
```

```{attention} Q&A
:class: dropdown
*Does unbiased imply consistent, or vice versa?*

- No, in both directions. See Example.
- Unbiased + $\operatorname{Var}\to0$ → consistent.

*Why accept a biased estimator?*

- MSE, not bias, is the total error → small bias can buy a large variance cut.
- Normal $\sigma^2$: among $c\sum_i(X_i-\bar X_n)^2$, divisor $n+1$ minimizes MSE (not $n-1$, not $n$).
- **Stein's paradox**: Normal mean (known covariance $\sigma^2\mathbf{I}$) in $\ge3$ dims → shrinkage (James–Stein) has lower total MSE than $\bar X_n$ for every $\theta$.
- Same tradeoff as in learning: [Bias-Variance Tradeoff](../../ml/misc.md#bias-variance-tradeoff), ridge, early stopping.

*Is unbiasedness invariant under reparametrization?*

- No. $\mathbb{E}[g(\hat\theta_n)]\ne g(\mathbb{E}[\hat\theta_n])$ unless $g$ is affine ← [Jensen](limits.md#jensens-inequality).
- e.g., $s^2$ unbiased for $\sigma^2$ → $s$ biased for $\sigma$.

*Does an unbiased estimator always exist?*

- No. $\operatorname{Binomial}(n,p)$, target $1/p$: $\mathbb{E}_p[T]$ is a polynomial in $p$ of degree $\le n$ → never equals $1/p$.

*Can an estimator beat $I(\theta)^{-1}$ asymptotically?*

- At isolated points, yes: **Hodges' estimator** (threshold $\bar X_n$ to $0$ near $0$) → superefficient at $\theta=0$.
- Price: bad risk at nearby $\theta$ for finite $n$. Regular estimators cannot beat $I(\theta)^{-1}$ (Hájek–Le Cam).
```

&nbsp;

### Sample Mean & Sample Variance
- **What**: $\bar X_n$ & $s^2$: unbiased estimators of the population mean & variance.
- **Why**: Default estimators behind SE, $z$/$t$ intervals & tests.
    - Divide-by-$n$ variance underestimates $\sigma^2$ ← $\bar X_n$ is fitted to the same data.
- **How**:
    1. Average → $\bar X_n$. Spread shrinks as $1/\sqrt{n}$.
    2. Sum squared residuals around $\bar X_n$.
    3. Divide by $n-1$ = #free residuals (**Bessel's correction**), not $n$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mu=\mathbb{E}[X]$: Population mean.
    - $\sigma^2=\operatorname{Var}[X]<\infty$: Population variance.
    - $\mu_4=\mathbb{E}[(X-\mu)^4]$: 4th central moment.
    - $\chi^2_\nu$: Chi-squared w/ $\nu$ degrees of freedom (df).

Properties (i.i.d., finite $\sigma^2$; $\operatorname{Var}(s^2)$ also needs $\mu_4<\infty$):

$$
\begin{align*}
\mathbb{E}[\bar X_n]&=\mu, &\operatorname{Var}(\bar X_n)&=\frac{\sigma^2}{n}, &\operatorname{se}(\bar X_n)&=\frac{\sigma}{\sqrt n}\approx\frac{s}{\sqrt n}\\
\mathbb{E}[s^2]&=\sigma^2, &\operatorname{Var}(s^2)&=\frac1n\left(\mu_4-\frac{n-3}{n-1}\sigma^4\right)
\end{align*}
$$

- MLE (Normal): $\hat\sigma^2_\text{MLE}=\frac1n\sum_i(X_i-\bar X_n)^2=\frac{n-1}{n}s^2$ → bias $=-\frac{\sigma^2}{n}$.
- $\mathbb{E}[s]<\sigma$ when $\operatorname{Var}(s)>0$ ← Jensen on concave $\sqrt{\cdot}$.

Extra facts for Normal data:
- $\bar X_n\sim\mathcal{N}(\mu,\sigma^2/n)$.
- $(n-1)s^2/\sigma^2\sim\chi^2_{n-1}$ → $\operatorname{Var}(s^2)=\frac{2\sigma^4}{n-1}$.
- $\bar X_n\perp s^2$. Holds for i.i.d. data **only** if Normal.
- $\mathbb{E}[s]=c_4(n)\,\sigma$.
    - $c_4(n)=\sqrt{\frac{2}{n-1}}\,\frac{\Gamma(n/2)}{\Gamma((n-1)/2)}$: Bias factor, e.g., $c_4(2)\approx0.80$, $\to1$ as $n\to\infty$.
```

```{tip} Derivation
:class: dropdown
*Why $\operatorname{Var}(\bar X_n)=\sigma^2/n$?*

1. Uncorrelated $X_i$ → $\operatorname{Var}(\sum_iX_i)=n\sigma^2$.
2. Scale by $1/n$ → variance scales by $1/n^2$ → $\sigma^2/n$.

*Why $\mathbb{E}[s^2]=\sigma^2$?*

1. Center at $\mu$ & expand:

    $$
    \sum_{i=1}^n(X_i-\bar X_n)^2=\sum_{i=1}^n(X_i-\mu)^2-n(\bar X_n-\mu)^2
    $$

    - Cross term: $-2(\bar X_n-\mu)\sum_i(X_i-\mu)=-2n(\bar X_n-\mu)^2$.
2. Take expectation: $n\sigma^2-n\cdot\frac{\sigma^2}{n}=(n-1)\sigma^2$.
3. Divide by $n-1$ → $\sigma^2$.
- Uses only common mean, equal variances & zero correlation. Independence & Normality not needed.

*Why "$n-1$ free residuals"?*

1. Residuals $r_i=X_i-\bar X_n$ satisfy $\sum_ir_i=0$.
2. → residual vector lives in the $(n-1)$-dim subspace $\perp\mathbf{1}$.
3. Projecting onto that subspace drops one dim of noise → $\mathbb{E}\|\mathbf{r}\|^2=(n-1)\sigma^2$.
```

```{note} Example
:class: dropdown
- $n=2$: $s^2=\frac{(X_1-X_2)^2}{2}$ → $\mathbb{E}=\frac{2\sigma^2}{2}=\sigma^2$. MLE $=\frac{(X_1-X_2)^2}{4}$ → expects $\sigma^2/2$.
- Sample $\{1,2,3,4,5\}$: $\bar X_n=3$, $\sum_i(X_i-\bar X_n)^2=10$.
    - $s^2=10/4=2.5$, $\hat\sigma^2_\text{MLE}=10/5=2$.
    - $\widehat{\operatorname{se}}(\bar X_n)=\sqrt{2.5/5}\approx0.71$.
```

```{attention} Q&A
:class: dropdown
*SD vs SE?*

- SD $\sigma$: spread of the data. Does not shrink w/ $n$.
- SE $\sigma/\sqrt n$: spread of $\bar X_n$. $\to0$ as $n\to\infty$.
- 4× samples → ½ SE.

*Is $s$ unbiased for $\sigma$?*

- No. $\mathbb{E}[s]<\sigma$ ← Jensen. Bias $\to0$ as $n\to\infty$.
- Unbiased $s^2$ ≠ unbiased $s$ → unbiasedness does not survive nonlinear maps.

*Is $s^2$ the best variance estimator?*

- Best among unbiased ones for Normal data (UMVU).
- Not best in MSE: divisor $n+1$ wins for Normal data. See [Estimator](#estimator).

*Library defaults?*

- `numpy.var` / `np.std`: `ddof=0` → divide by $n$.
- `torch.var` / `torch.std`: `correction=1` → divide by $n-1$.
- `pandas.Series.var`: `ddof=1` → divide by $n-1$.

*What if samples are correlated?*

- Stationary w/ lag-$k$ correlation $\rho_k$: $\operatorname{Var}(\bar X_n)=\frac{\sigma^2}{n}\left[1+2\sum_{k=1}^{n-1}\left(1-\frac kn\right)\rho_k\right]$.
- $\rho_k>0$ (time series, clustered users) → true SE > $s/\sqrt n$ → overconfident intervals.
- Effective sample size $n_\text{eff}<n$. See also [block bootstrap](#bootstrap).

*What do heavy tails do?*

- $\bar X_n$ stays unbiased if $\mu$ exists, but $\operatorname{Var}(s^2)$ grows w/ $\mu_4$ → $s^2$ is noisy.
- Cauchy: no mean → $\bar X_n$ has the same Cauchy distribution for every $n$ → never converges.
```

&nbsp;

### Method of Moments
- **What**: Set the first $d$ population moments equal to sample moments → solve for $\theta$.
- **Why**: Simple estimator, often closed-form, w/o the full likelihood.
    - Common use: initial value for MLE.
- **How**:
    1. Write population moments as functions of $\theta$: $m_j(\theta)=\mathbb{E}_\theta[X^j]$, $j=1..d$.
    2. Compute sample moments $\hat m_j=\frac1n\sum_iX_i^j$.
    3. Solve the $d$ equations $m_j(\hat\theta)=\hat m_j$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $d=\dim\theta$: #params.
    - $\mathbf{m}(\theta)=(m_1(\theta),\dots,m_d(\theta))$: Population moment map.
    - $\hat{\mathbf{m}}=(\hat m_1,\dots,\hat m_d)$: Sample moments.
    - $M=\partial\mathbf{m}/\partial\theta\in\mathbb{R}^{d\times d}$: Jacobian of the moment map.
    - $\Sigma=\operatorname{Cov}_\theta\left[(X,X^2,\dots,X^d)\right]$: Moment covariance.

Estimator:

$$
\hat\theta_n=\mathbf{m}^{-1}(\hat{\mathbf{m}})
$$

Properties (moments up to order $2d$ finite, $\mathbf{m}$ invertible & $\mathbf{m}^{-1}$ smooth):
- Consistent ← LLN: $\hat{\mathbf{m}}\xrightarrow{P}\mathbf{m}(\theta)$ + continuous mapping.
- Asymptotically normal ← CLT on $\hat{\mathbf{m}}$ + delta method:

    $$
    \sqrt n(\hat\theta_n-\theta)\xrightarrow{d}\mathcal{N}\left(\mathbf{0},\,M^{-1}\Sigma M^{-\top}\right)
    $$

- Regular model: $M^{-1}\Sigma M^{-\top}\succeq I(\theta)^{-1}$ → usually less efficient than MLE.
```

```{note} Example
:class: dropdown
- Normal: $\hat\mu=\bar X_n$, $\hat\sigma^2=\hat m_2-\bar X_n^2=\frac1n\sum_i(X_i-\bar X_n)^2$ → same as MLE (biased).
- Gamma (shape $a$, rate $b$): mean $=a/b$, variance $=a/b^2$.
    - → $\hat b=\bar X_n/\hat\sigma^2$, $\hat a=\bar X_n^2/\hat\sigma^2$.
    - MLE for $a$ has no closed form → MoM gives the starting point.
- $\operatorname{Uniform}(0,\theta)$: $\mathbb{E}[X]=\theta/2$ → $\hat\theta=2\bar X_n$.
    - Can give $\hat\theta<\max_iX_i$ → impossible value.
    - $\operatorname{MSE}=\frac{\theta^2}{3n}$ vs MLE $\max_iX_i$: $\operatorname{MSE}=\frac{2\theta^2}{(n+1)(n+2)}$ → MoM is much worse.
```

```{attention} Q&A
:class: dropdown
*Pros?*

- Closed form for many families → cheap.
- Needs only moment formulas, not the full density.
- Consistent under weak conditions.

*Cons?*

- Usually less efficient than MLE ← uses only $d$ moments, not the full likelihood.
- Can leave $\Theta$ (e.g., $\hat\theta<\max_iX_i$, negative variance components).
- Choice of moments is arbitrary → different estimators.
- Needs moments to exist (Cauchy → no moments → no MoM).

*When does MoM equal MLE?*

- Full regular exponential family w/ natural sufficient statistic $T(x)$, interior solution → MLE solves $\mathbb{E}_\theta[T(X)]=\frac1n\sum_iT(X_i)$ → MoM on $T$.
- e.g., Normal, Poisson, Bernoulli, Exponential.

*More moment equations than params?*

- Overidentified → minimize a weighted distance between $\hat{\mathbf{m}}$ & $\mathbf{m}(\theta)$ → **generalized method of moments** (econometrics).
```

&nbsp;

### MLE
- **Name**: Maximum Likelihood Estimation
- **What**: Parameter value maximizing the likelihood of the observed sample: $\hat\theta_n=\arg\max_{\theta\in\Theta}\ell(\theta)$.
- **Why**: General recipe w/ the best possible large-sample variance under regularity conditions.
    - Loss view (NLL → MSE / cross entropy): [MLE in Objective](../../ml/obj.md#mle).
- **How**:
    1. Write $\ell(\theta)=\sum_i\log f(X_i;\theta)$.
    2. Solve the score equation $\nabla\ell(\theta)=\mathbf{0}$ in closed form, or optimize numerically.
    3. Check that the solution is the global max. Check boundaries too: the score equation finds only interior candidates.
    4. Get the SE from the curvature at the peak ([Fisher Information](#fisher-information)).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $J_n(\theta)=-\nabla^2\ell(\theta)$: Observed information.
    - $\hat P_n$: Empirical distribution of the sample.
    - $P_\theta$: Model distribution.
    - $\theta^*$: KL-closest model param under misspecification.

Model:

$$
\hat\theta_n=\arg\max_{\theta\in\Theta}\ell(\theta)=\arg\min_{\theta\in\Theta}D_\text{KL}(\hat P_n\|P_\theta)
$$

- KL form: exact for discrete models. For densities, use the cross-entropy form (see [MLE in Objective](../../ml/obj.md#mle)).

Properties (i.i.d., well-specified, regular):
- **Consistent**: $\hat\theta_n\xrightarrow{P}\theta$.
- **Asymptotically normal & efficient**:

    $$
    \sqrt n(\hat\theta_n-\theta)\xrightarrow{d}\mathcal{N}\left(\mathbf{0},I(\theta)^{-1}\right)
    $$

    - → $\hat\theta_n\approx\mathcal{N}\left(\theta,(nI(\theta))^{-1}\right)$.
    - Plug-in SE: $\widehat{\operatorname{se}}=\left(nI(\hat\theta_n)\right)^{-1/2}$ or $J_n(\hat\theta_n)^{-1/2}$ (scalar).
- **Invariant**: MLE of $g(\theta)$ is $g(\hat\theta_n)$.
    - Delta method: $\sqrt n\left(g(\hat\theta_n)-g(\theta)\right)\xrightarrow{d}\mathcal{N}\left(0,\nabla g^\top I(\theta)^{-1}\nabla g\right)$.
- **Misspecified** (true distribution $P\notin\{P_\theta\}$):
    - $\hat\theta_n\xrightarrow{P}\theta^*=\arg\min_\theta D_\text{KL}(P\|P_\theta)$.
    - $\sqrt n(\hat\theta_n-\theta^*)\xrightarrow{d}\mathcal{N}(\mathbf{0},A^{-1}BA^{-1})$ (**sandwich**).
    - $A=-\mathbb{E}_P[\nabla^2\log f(X;\theta^*)]$: Expected negative Hessian.
    - $B=\operatorname{Cov}_P[\nabla\log f(X;\theta^*)]$: Score covariance.
    - Well-specified → $A=B=I(\theta)$ → $I(\theta)^{-1}$.
```

```{tip} Derivation
:class: dropdown
*Why consistent?*

1. LLN: $\frac1n\ell(\vartheta)\xrightarrow{P}\mathbb{E}_\theta[\log f(X;\vartheta)]$ for each candidate $\vartheta$.
2. $\mathbb{E}_\theta[\log f(X;\vartheta)]=\mathbb{E}_\theta[\log f(X;\theta)]-D_\text{KL}(P_\theta\|P_\vartheta)$.
3. KL $\ge0$, $=0$ iff $P_\vartheta=P_\theta$ → unique max at $\vartheta=\theta$ (identifiability).
4. Uniform convergence (regularity) → argmax of the limit = limit of the argmax.
- $\vartheta$: Generic candidate param.

*Why asymptotically $\mathcal{N}(0,I^{-1})$?* (scalar)

1. Score at the MLE is $0$: $\ell'(\hat\theta_n)=0$.
2. Taylor around $\theta$: $0\approx\ell'(\theta)+\ell''(\theta)(\hat\theta_n-\theta)$.
3. Rearrange:

    $$
    \sqrt n(\hat\theta_n-\theta)\approx\frac{\frac{1}{\sqrt n}\ell'(\theta)}{-\frac1n\ell''(\theta)}
    $$

4. Numerator: sum of $n$ i.i.d. scores, mean $0$, variance $I(\theta)$ → [CLT](limits.md#central-limit-theorem) → $\mathcal{N}(0,I(\theta))$.
5. Denominator: [LLN](limits.md#law-of-large-numbers) → $-\mathbb{E}[\partial_\theta^2\log f]=I(\theta)$.
6. Slutsky → $\mathcal{N}(0,I/I^2)=\mathcal{N}(0,I^{-1})$.
- Misspecified: numerator variance $B$, denominator $A$ → $B/A^2$ = sandwich.
```

```{note} Example
:class: dropdown
- **Bernoulli**: $\ell=S\log p+(n-S)\log(1-p)$, $S=\sum_iX_i$.
    - $\ell'=0$ → $\hat p=S/n=\bar X_n$.
    - $I(p)=\frac{1}{p(1-p)}$ → $\widehat{\operatorname{se}}=\sqrt{\hat p(1-\hat p)/n}$.
- **Exponential** (rate $\lambda$): $\ell=n\log\lambda-\lambda\sum_iX_i$ → $\hat\lambda=1/\bar X_n$.
    - Invariance: MLE of the mean $1/\lambda$ is $\bar X_n$ (unbiased).
    - $\hat\lambda$ itself is biased: $\mathbb{E}[\hat\lambda]=\frac{n}{n-1}\lambda$.
- **Normal**: $\hat\mu=\bar X_n$, $\hat\sigma^2=\frac1n\sum_i(X_i-\bar X_n)^2$ → biased by $-\sigma^2/n$.
```

````{important} Code
:class: dropdown
```python
import torch

class MLE:
    def __init__(self, log_f, theta0):
        self.log_f = log_f                                ## log f(x; theta), vectorized over x
        self.theta = theta0.clone().requires_grad_(True)

    def nll(self, theta, x):
        return -self.log_f(x, theta).sum()                ## -ell(theta): i.i.d. -> sum of logs

    def fit(self, x, steps=100):
        opt = torch.optim.LBFGS([self.theta], max_iter=steps, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            loss = self.nll(self.theta, x)
            loss.backward()
            return loss
        opt.step(closure)                                 ## argmax ell = argmin NLL
        theta_hat = self.theta.detach()
        ## observed information J_n = Hessian of the NLL at the peak; Cov(theta_hat) ~ J_n^{-1}
        J = torch.autograd.functional.hessian(lambda t: self.nll(t, x), theta_hat)
        se = torch.linalg.inv(J).diagonal().sqrt()
        return theta_hat, se

## Example: Normal, theta = (mu, log sigma) -> log keeps sigma > 0 during optimization
def log_normal(x, theta):
    mu, log_sigma = theta
    return torch.distributions.Normal(mu, log_sigma.exp()).log_prob(x)

x = torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float64)
theta_hat, se = MLE(log_normal, torch.zeros(2, dtype=torch.float64)).fit(x)
print(theta_hat[0], theta_hat[1].exp())  ## ~3.0, ~1.414 = sqrt(10/5): divide by n, not n-1
print(se)                                ## ~[0.632, 0.316] = [sigma_hat/sqrt(n), 1/sqrt(2n)]
```
````

```{attention} Q&A
:class: dropdown
*Regularity conditions?*

- Identifiable: $\theta\ne\theta'$ → $P_\theta\ne P_{\theta'}$.
- True $\theta$ in the interior of $\Theta$.
- Support of $f(\cdot;\theta)$ does not depend on $\theta$.
- $\log f$ smooth enough to differentiate under the integral.
- $I(\theta)$ finite & positive definite.
- #params fixed as $n\to\infty$.

*What breaks without them?*

- **Support depends on $\theta$**: $\operatorname{Uniform}(0,\theta)$ → $\hat\theta_n=\max_iX_i$.
    - $\frac{n}{\theta}(\theta-\hat\theta_n)\xrightarrow{d}\operatorname{Exp}(1)$ → rate $1/n$, not $1/\sqrt n$; limit is not Normal.
- **Boundary**: $\theta$ on the edge of $\Theta$ (e.g., variance component $=0$) → limit is a mixture w/ a point mass, not Normal.
- **#params grows w/ $n$** (Neyman–Scott): pairs $X_{i1},X_{i2}\sim\mathcal{N}(\mu_i,\sigma^2)$ → $\hat\sigma^2\xrightarrow{P}\sigma^2/2$ → inconsistent.
- **Unbounded likelihood / multiple local maxima**: e.g., mixture components collapsing onto one point → see [MLE in Objective](../../ml/obj.md#mle).

*Is the MLE unbiased?*

- Not in general. See Example.
- Bias is usually $O(1/n)$, SE is $O(1/\sqrt n)$ → bias is negligible for large $n$.

*Why is the MLE invariant, while unbiasedness is not?*

- MLE is an argmax: the peak of the likelihood moves w/ any one-to-one relabeling of $\theta$.
- Unbiasedness is an expectation: $\mathbb{E}[g(\hat\theta)]\ne g(\mathbb{E}[\hat\theta])$ for nonlinear $g$.
- Non-injective $g$: define the MLE of $g(\theta)$ via the induced (profile) likelihood → still $g(\hat\theta_n)$.

*Observed vs expected information for the SE?*

- Both are consistent: $J_n(\hat\theta_n)/n\xrightarrow{P}I(\theta)$.
- Observed $J_n$ needs only the Hessian at the peak → what autograd / optimizers give.

*How does the MLE use the data?*

- Only through any sufficient statistic $T$: $\ell(\theta)=\log a(\mathbf{x})+\log b(T,\theta)$ → argmax depends on $T$ only. See [Sufficient Statistics](bayes.md#sufficient-statistics).

*How does it relate to Bayesian estimates?*

- MAP w/ flat prior = MLE (in that parametrization).
- Bernstein–von Mises: posterior $\approx\mathcal{N}(\hat\theta_n,(nI)^{-1})$ → credible ≈ confidence intervals for large $n$. See [Posterior Concentration](bayes.md#posterior-concentration).

*What if the model is wrong?*

- MLE still converges, to the KL-closest model $\theta^*$.
- $I(\theta)^{-1}$-based SE is then wrong → use the sandwich $A^{-1}BA^{-1}$ ("robust SE").
```

&nbsp;

#### Fisher Information
- **What**: Variance of the score = expected negative curvature of the log-likelihood.
- **Why**: Measure how much one observation tells about $\theta$.
    - Sharp log-likelihood peak → $\theta$ well determined → small estimator variance.
- **How**:
    1. **Score** = slope of $\log f(x;\theta)$ w.r.t. $\theta$.
    2. At the true $\theta$, the slope averages to $0$.
    3. Large spread of the slope across samples = data reacts strongly to $\theta$ = much information.
    4. Equivalently: average curvature of the log-likelihood peak.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{u}(\theta;x)=\nabla_\theta\log f(x;\theta)$: Score of one observation.
    - $I_n(\theta)$: Fisher information of $n$ observations.

Definition:

$$
\begin{align*}
\mathbb{E}_\theta[\mathbf{u}(\theta;X)]&=\mathbf{0}\\
I(\theta)&=\mathbb{E}_\theta\left[\mathbf{u}\mathbf{u}^\top\right]=\operatorname{Cov}_\theta(\mathbf{u})=-\mathbb{E}_\theta\left[\nabla^2_\theta\log f(X;\theta)\right]
\end{align*}
$$

- Last equality: regular model (differentiate under the integral twice).

Properties:
- $I(\theta)\succeq0$. Singular → some direction of $\theta$ is locally not identifiable.
- **Additivity**: independent scores add → $I_n(\theta)=nI(\theta)$.
- **Reparametrization** $\theta=h(\eta)$: $I_\eta=H^\top I_\theta H$.
    - $H=\partial\theta/\partial\eta$: Jacobian. Scalar: $I(\eta)=I(\theta)\left(\frac{d\theta}{d\eta}\right)^2$.
- **Data processing**: statistic $T$ → $I_T(\theta)\preceq I_n(\theta)$. Equality for all $\theta$ iff $T$ sufficient ([Sufficient Statistics](bayes.md#sufficient-statistics)).
- Observed info: $J_n(\hat\theta_n)/n\xrightarrow{P}I(\theta)$.
- Geometry: $I(\theta)$ = metric tensor of the model manifold = Hessian of KL ([Fisher-Rao Metric](../geo/info.md#fisher-rao-metric)).
- Prior: $\sqrt{\det I(\theta)}$ → [Jeffreys Prior](bayes.md#jeffreys-prior).
```

```{tip} Derivation
:class: dropdown
*Why $\mathbb{E}[\text{score}]=0$?* (scalar)

1. $\int f(x;\theta)\,dx=1$ → differentiate → $\int\partial_\theta f\,dx=0$.
2. $\partial_\theta f=f\,\partial_\theta\log f$ → $\mathbb{E}_\theta[u]=\int f\,u\,dx=0$.

*Why $\operatorname{Var}(\text{score})=-\mathbb{E}[\partial^2_\theta\log f]$?*

1. Differentiate $\int f\,u\,dx=0$ again:

    $$
    \int f\,u^2\,dx+\int f\,\partial_\theta u\,dx=0
    $$

    - First term: $\partial_\theta f\cdot u=f\,u^2$.
2. → $\mathbb{E}[u^2]=-\mathbb{E}[\partial_\theta^2\log f]$.
3. $\mathbb{E}[u]=0$ → $\mathbb{E}[u^2]=\operatorname{Var}(u)$.

*Why $I_n=nI$?*

1. $\ell'(\theta)=\sum_iu(\theta;X_i)$, independent terms.
2. Variance of an independent sum = sum of variances → $nI(\theta)$.
```

```{note} Example
:class: dropdown
- **Bernoulli**($p$): $u=\frac{x}{p}-\frac{1-x}{1-p}=\frac{x-p}{p(1-p)}$.
    - $I(p)=\frac{\operatorname{Var}(X)}{p^2(1-p)^2}=\frac{1}{p(1-p)}$.
    - Largest near $p\in\{0,1\}$ ← $\bar X_n$ has tiny variance there.
- **Normal**($\mu$, $\sigma^2$ known): $u=\frac{x-\mu}{\sigma^2}$ → $I(\mu)=\frac1{\sigma^2}$.
- **Normal**($\mu,\sigma^2$): $I=\operatorname{diag}\left(\frac1{\sigma^2},\frac1{2\sigma^4}\right)$ in coordinates $(\mu,\sigma^2)$.
- **Poisson**($\lambda$): $I=\frac1\lambda$. **Exponential**($\lambda$): $I=\frac1{\lambda^2}$.
```

```{attention} Q&A
:class: dropdown
*Why does high information mean low variance?*

- High information = sharp log-likelihood peak → small $\Delta\theta$ changes the likelihood a lot → data pins $\theta$ down.
- Formalized by the [Cramér–Rao Bound](#cram-r-rao-bound): $\operatorname{Var}\ge(nI)^{-1}$.

*Expected $I(\theta)$ vs observed $J_n$?*

- $I$: average over hypothetical data at a fixed $\theta$. Needs the true $\theta$ → plug in $\hat\theta_n$.
- $J_n$: curvature of *this* sample's log-likelihood. Free from autograd.
- Exponential families in natural params: $J_n(\theta)=nI(\theta)$ exactly.

*When does $I_n=nI$ fail?*

- Dependent data → scores correlated → information is not additive.

*Is Fisher information parametrization-invariant?*

- No. It transforms like a metric (Jacobian on both sides).
- → $\sqrt{\det I}\,d\theta$ is invariant → basis of the Jeffreys prior.

*Can $I(\theta)=0$?*

- Yes. $X\sim\mathcal{N}(\theta^2,1)$: $I(\theta)=4\theta^2=0$ at $\theta=0$ ← $\theta$ & $-\theta$ give the same model.
- Singular $I$ → standard $\sqrt n$-Normal theory fails → often slower rates (here $|\hat\theta|$ at $n^{-1/4}$) or non-identifiability.

*Fisher information vs Shannon entropy?*

- Fisher: local sensitivity of the model to $\theta$.
- Shannon: uncertainty of $X$ itself.
- Link: $D_\text{KL}(P_\theta\|P_{\theta+\delta})\approx\frac12\delta^\top I(\theta)\delta$.
```

&nbsp;

#### Cramér–Rao Bound
- **What**: Lower bound $(nI(\theta))^{-1}$ on the variance of any unbiased estimator.
- **Why**: Benchmark for estimator quality → defines efficiency.
    - Tells when to stop looking for a better unbiased estimator.
- **How**:
    1. Unbiasedness → estimator has covariance exactly $1$ w/ the total score.
    2. Cauchy–Schwarz: $\operatorname{Cov}^2\le\operatorname{Var}(\text{estimator})\cdot\operatorname{Var}(\text{score})$.
    3. $\operatorname{Var}(\text{score})=nI(\theta)$ → $\operatorname{Var}(\text{estimator})\ge1/(nI(\theta))$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $T=T(X_1,\dots,X_n)$: Estimator.
    - $\psi(\theta)=\mathbb{E}_\theta[T]$: Quantity $T$ is unbiased for.
    - $\nabla\psi$: Jacobian of $\psi$.
    - $S_n(\theta)=\nabla_\theta\ell(\theta)$: Total score.

Scalar:

$$
\operatorname{Var}_\theta(T)\ge\frac{\psi'(\theta)^2}{nI(\theta)}
\quad\xrightarrow{\ \psi(\theta)=\theta\ }\quad
\operatorname{Var}_\theta(T)\ge\frac{1}{nI(\theta)}
$$

- Biased $T$ w/ bias $b(\theta)$: $\psi=\theta+b$ → $\operatorname{MSE}\ge\frac{(1+b'(\theta))^2}{nI(\theta)}+b(\theta)^2$.

Matrix:

$$
\operatorname{Cov}_\theta(T)\succeq\nabla\psi\,\left(nI(\theta)\right)^{-1}\nabla\psi^\top
\quad\xrightarrow{\ \text{unbiased}\ }\quad
\operatorname{Cov}_\theta(T)\succeq\left(nI(\theta)\right)^{-1}
$$

- $\succeq$: Loewner order → $\mathbf{a}^\top\operatorname{Cov}(T)\mathbf{a}\ge\mathbf{a}^\top(nI)^{-1}\mathbf{a}$ for every direction $\mathbf{a}$.
- Single coordinate: $\operatorname{Var}(T_j)\ge[(nI)^{-1}]_{jj}\ge1/(nI_{jj})$.

Efficiency:
- $e(T)=\frac{1/(nI(\theta))}{\operatorname{Var}_\theta(T)}\in(0,1]$. **Efficient** iff $e(T)=1$.
- Equality iff $S_n(\theta)=a(\theta)\,(T-\psi(\theta))$ a.s.
    - ⇔ one-param exponential family $f=h(x)\exp(\eta(\theta)t(x)-A(\theta))$ & $T$ affine in $\frac1n\sum_it(X_i)$.

Conditions: regular model, $0<I(\theta)<\infty$, $\psi$ differentiable under the integral.
```

```{tip} Derivation
:class: dropdown
*Scalar bound via Cauchy–Schwarz*

1. $\psi(\theta)=\int T(\mathbf{x})\,f_n(\mathbf{x};\theta)\,d\mathbf{x}$.
    - $f_n=\prod_if(x_i;\theta)$: Joint density.
2. Differentiate, use $\partial_\theta f_n=f_nS_n$:

    $$
    \psi'(\theta)=\mathbb{E}_\theta[T\,S_n]=\operatorname{Cov}_\theta(T,S_n)
    $$

    - Second equality ← $\mathbb{E}[S_n]=0$.
3. [Cauchy–Schwarz](limits.md#cauchy-schwarz-inequality): $\operatorname{Cov}(T,S_n)^2\le\operatorname{Var}(T)\operatorname{Var}(S_n)=\operatorname{Var}(T)\,nI(\theta)$.
4. → $\operatorname{Var}(T)\ge\psi'(\theta)^2/(nI(\theta))$.
5. Equality in Cauchy–Schwarz iff $S_n$ & $T$ are affinely related.

*Matrix bound*

1. Stack $(T,S_n)$. Its covariance is PSD:

    $$
    \begin{pmatrix}\operatorname{Cov}(T)&\nabla\psi\\\nabla\psi^\top&nI(\theta)\end{pmatrix}\succeq0
    $$

    - Off-diagonal: $\operatorname{Cov}(T,S_n)=\nabla\psi$ (step 2, per coordinate).
2. Schur complement → $\operatorname{Cov}(T)-\nabla\psi\,(nI)^{-1}\nabla\psi^\top\succeq0$.
```

```{note} Example
:class: dropdown
- **Attained**: Normal mean, $\sigma^2$ known → bound $\sigma^2/n$ = $\operatorname{Var}(\bar X_n)$ → $\bar X_n$ efficient.
- **Not attained, $s^2$**: Normal $\sigma^2$ → bound $\frac{2\sigma^4}{n}$ vs $\operatorname{Var}(s^2)=\frac{2\sigma^4}{n-1}$.
    - $s^2$ is still the best unbiased estimator (UMVU) → bound is not tight here.
- **Parametrization matters**: Exponential, $n$ samples.
    - Mean $1/\lambda$: $\bar X_n$ attains $\frac{1}{n\lambda^2}$.
    - Rate $\lambda$: best unbiased $\frac{n-1}{\sum_iX_i}$ has $\operatorname{Var}=\frac{\lambda^2}{n-2}>\frac{\lambda^2}{n}$ → bound not attained.
- **Non-regular**: $\operatorname{Uniform}(0,\theta)$ → $\frac{n+1}{n}\max_iX_i$ is unbiased w/ $\operatorname{Var}=\frac{\theta^2}{n(n+2)}=O(1/n^2)$.
    - Support depends on $\theta$ → bound does not apply.
```

```{attention} Q&A
:class: dropdown
*Does the bound hold for biased estimators?*

- Not the $1/(nI)$ form. Constant $T\equiv\theta_0$ has variance $0$.
- Use the biased form ($\psi'^2/(nI)$, or the MSE version).

*When is the bound not attainable?*

- At finite $n$: whenever the model is not an exponential family w/ $T$ affine in its sufficient statistic.
- Then the best unbiased estimator (UMVU) has variance strictly above the bound.
- Asymptotically: MLE attains it under regularity.

*How to improve an unbiased estimator toward the bound?*

- **Rao–Blackwell**: $\tilde T=\mathbb{E}[T\mid\text{sufficient statistic}]$ → still unbiased, $\operatorname{Var}\le\operatorname{Var}(T)$. See [Sufficient Statistics](bayes.md#sufficient-statistics).
- **Lehmann–Scheffé**: if that statistic is also complete → $\tilde T$ is the unique UMVU.

*Does an unknown nuisance param cost precision?*

- Yes, in general: $(I^{-1})_{11}=\frac{1}{I_{11}-I_{12}I_{22}^{-1}I_{21}}\ge\frac{1}{I_{11}}$.
- No cost iff $I_{12}=0$ (orthogonal params, e.g., Normal $\mu$ & $\sigma^2$).

*Does it apply when the support depends on $\theta$?*

- No. Differentiation under the integral fails → estimators can reach $O(1/n^2)$ variance (Uniform example).
```

&nbsp;

## Interval Estimation

### Confidence Interval
- **What**: Random interval that covers the fixed $\theta$ w/ probability $\ge1-\alpha$ over repeated sampling.
- **Why**: Point estimate hides its uncertainty → report a range w/ a guaranteed long-run coverage rate.
- **How**:
    1. Find a **pivot** $Q(X,\theta)$: distribution known & free of $\theta$.
    2. Take its quantiles: $P(a\le Q\le b)=1-\alpha$.
    3. Invert $a\le Q\le b$ for $\theta$ → $[L,U]$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $L(X),U(X)$: Random endpoints.
    - $\mu=\mathbb{E}[X]$: Population mean.
    - $\sigma^2=\operatorname{Var}[X]$: Population variance.
    - $t_{\nu,q}$: $q$-quantile of Student $t$ w/ $\nu$ df.
    - $\widehat{\operatorname{se}}$: Estimated SE of $\hat\theta_n$.

Definition:

$$
P_\theta\left(L(X)\le\theta\le U(X)\right)\ge1-\alpha\quad\forall\theta\in\Theta
$$

- Randomness is in $L,U$, not in $\theta$.
- **Asymptotic** CI: $\lim_{n\to\infty}P_\theta(\cdot)=1-\alpha$.

Standard intervals (two-sided, level $1-\alpha$):

$$
\begin{align*}
z\text{-interval }(\sigma\text{ known})&: &&\bar X_n\pm z_{1-\alpha/2}\,\frac{\sigma}{\sqrt n}\\
t\text{-interval }(\sigma\text{ unknown})&: &&\bar X_n\pm t_{n-1,1-\alpha/2}\,\frac{s}{\sqrt n}\\
\text{Wald}&: &&\hat\theta_n\pm z_{1-\alpha/2}\,\widehat{\operatorname{se}},\quad\widehat{\operatorname{se}}=\left(nI(\hat\theta_n)\right)^{-1/2}
\end{align*}
$$

- $z$: exact for Normal data, asymptotic otherwise (CLT).
- $t$: exact for Normal data. Pivot:

    $$
    \frac{\bar X_n-\mu}{s/\sqrt n}\sim t_{n-1}
    $$

- Wald: asymptotic, built on [MLE](#mle) normality.

**CI ↔ test duality**:

$$
C(X)=\{\theta_0:\ \text{level-}\alpha\text{ test of }H_0:\theta=\theta_0\text{ does not reject}\}
$$

- $C(X)$ has coverage $\ge1-\alpha$. Conversely: reject $H_0:\theta=\theta_0$ iff $\theta_0\notin C(X)$.
```

```{tip} Derivation
:class: dropdown
*Why is the $t$-pivot $t_{n-1}$?* (Normal data)

1. $Z=\frac{\bar X_n-\mu}{\sigma/\sqrt n}\sim\mathcal{N}(0,1)$.
2. $V=\frac{(n-1)s^2}{\sigma^2}\sim\chi^2_{n-1}$.
3. $Z\perp V$ ← $\bar X_n\perp s^2$ for Normal data.
4. $\frac{Z}{\sqrt{V/(n-1)}}=\frac{\bar X_n-\mu}{s/\sqrt n}$ → $\sigma$ cancels → definition of $t_{n-1}$.

*Why does inverting a test give coverage?*

1. Let $A(\theta_0)$ = acceptance region of a level-$\alpha$ test of $\theta=\theta_0$.
2. $\theta_0\in C(X)\iff X\in A(\theta_0)$.
3. → $P_{\theta_0}(\theta_0\in C(X))=P_{\theta_0}(X\in A(\theta_0))\ge1-\alpha$.
```

```{note} Example
:class: dropdown
- $n=25$, $\bar X_n=10$, $s=5$ → $\widehat{\operatorname{se}}=1$.
    - $t$: $t_{24,0.975}\approx2.064$ → $[7.94,\,12.06]$.
    - $z$ (wrong here, $\sigma$ unknown & $n$ small): $1.96$ → $[8.04,\,11.96]$ → too narrow.
- Proportion: $\hat p=0.5$, $n=100$ → Wald $0.5\pm1.96\cdot0.05=[0.40,\,0.60]$.
- Proportion: $\hat p=0$, $n=20$ → Wald $=[0,0]$ → absurd. Wilson / Clopper–Pearson stay sensible.
```

```{dropdown} Table: Common Intervals
| Interval | Target | Assumptions | Notes |
|:--|:--|:--|:--|
| $z$ | Mean | $\sigma$ known; Normal or large $n$ | Rarely realistic |
| $t$ | Mean | Normal data (exact) or large $n$ | Default for means |
| Wald | Any MLE | Large $n$, regular model | Not parametrization-invariant; poor near boundaries |
| Wilson | Proportion | Large $n$ | Inverts the score test; good near $0/1$ |
| Clopper–Pearson | Proportion | Binomial sampling (exact) | Conservative: coverage $\ge1-\alpha$ |
| [Bootstrap](#bootstrap) | Any statistic | Smooth statistic, i.i.d. | No formula needed |

- Wilson: $\dfrac{\hat p+\frac{z^2}{2n}}{1+\frac{z^2}{n}}\pm\dfrac{z}{1+\frac{z^2}{n}}\sqrt{\dfrac{\hat p(1-\hat p)}{n}+\dfrac{z^2}{4n^2}}$.
- Clopper–Pearson: $\left[B_{\alpha/2}(k,\,n-k+1),\ B_{1-\alpha/2}(k+1,\,n-k)\right]$.
    - $z=z_{1-\alpha/2}$: Normal critical value.
    - $k$: #successes.
    - $k=0$ → lower $=0$. $k=n$ → upper $=1$.
    - $B_q(a,b)$: $q$-quantile of $\operatorname{Beta}(a,b)$.
```

```{attention} Q&A
:class: dropdown
*Correct interpretation of a 95% CI?*

- ✅ The *procedure* covers $\theta$ in 95% of repeated samples.
- ❌ "$P(\theta\in[7.94,12.06])=0.95$". After the data are seen, the interval is fixed → it covers $\theta$ or it does not.
- ❌ "95% of the data lie in the CI". That describes a prediction / tolerance interval.

*CI vs Bayesian credible interval?*

- CI: coverage over repeated samples at fixed $\theta$.
- Credible: posterior probability given this sample & prior. See [Credible Sets](bayes.md#posterior-summaries-credible-sets).
- Numerically equal for a Normal mean: flat prior ($\sigma$ known → $z$), or prior $\propto1/\sigma$ ($\sigma$ unknown → $t$). Differ w/ informative priors or small $n$.

*Why $t$ instead of $z$?*

- $s$ estimates $\sigma$ → extra randomness → heavier tails than Normal.
- $t_{n-1}\to\mathcal{N}(0,1)$ as $n\to\infty$. Difference matters for small $n$ (rule of thumb: $n\lesssim30$).

*Is the $t$-interval robust?*

- Exact only for Normal data. CLT makes it approximately right for large $n$.
- Strong skew + small $n$ → actual coverage < nominal.

*How does width scale?*

- Width $\propto1/\sqrt n$ → 4× samples for ½ width.
- Confidence ⬆️ → quantile ⬆️ → width ⬆️.

*Why can Wald intervals fail?*

- Relies on the Normal approximation at $\hat\theta_n$ → poor at small $n$ & near boundaries ($\hat p\approx0$).
- Not invariant: Wald on $\theta$ ≠ transformed Wald on $\log\theta$.
- Fix: transform to an unbounded scale (log / logit), build the interval, map back. Or invert a score / likelihood-ratio test.

*Two 95% CIs overlap → no significant difference?*

- ❌ Overlap does not imply non-significance.
- Equal SEs: non-overlap needs a gap $>2\cdot1.96\,\operatorname{se}$, a test needs only $>1.96\sqrt2\,\operatorname{se}$.
- → test the difference directly; its CI uses $\operatorname{se}_\text{diff}=\sqrt{\operatorname{se}_1^2+\operatorname{se}_2^2}$ (independent estimates).

*How do CIs connect to tests?*

- Duality: $\theta_0\notin$ 95% CI ⇔ two-sided test rejects $H_0:\theta=\theta_0$ at $\alpha=0.05$ (for the inverted test).
```

&nbsp;

### Bootstrap
- **What**: Resample the data w/ replacement to approximate the sampling distribution of an estimator. {cite:p}`efron1979bootstrap`
- **Why**: SE & CI for complex statistics w/o a closed-form formula or a Normal assumption.
    - Sampling distribution needs fresh samples from the unknown population $F$ → use the sample as a stand-in for $F$.
- **How**:
    1. Replace $F$ by the empirical distribution $\hat F_n$ (**plug-in principle**).
    2. Draw $B$ resamples of size $n$ from the data, w/ replacement.
    3. Recompute the statistic on each → $\hat\theta^*_1,\dots,\hat\theta^*_B$.
    4. SD of $\hat\theta^*_b$ → SE. Quantiles of $\hat\theta^*_b$ → CI.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $B$: #bootstrap resamples.
- Misc:
    - $F$: Population distribution.
    - $\hat F_n$: Empirical distribution, mass $\frac1n$ on each $X_i$.
    - $X^*_1,\dots,X^*_n\overset{\text{iid}}{\sim}\hat F_n$: One resample.
    - $\hat\theta^*_b$: Statistic on resample $b$.
    - $\bar\theta^*=\frac1B\sum_b\hat\theta^*_b$: Bootstrap mean.
    - $\hat q_q$: Empirical $q$-quantile of $\{\hat\theta^*_b\}$.

Principle:

$$
\left(\hat\theta^*-\hat\theta_n\right)\mid\hat F_n\ \approx\ \left(\hat\theta_n-\theta\right)\mid F
$$

Process:
1. For $b=1..B$: draw $X^*\sim\hat F_n$, compute $\hat\theta^*_b$.
2. SE: $\widehat{\operatorname{se}}_\text{boot}=\sqrt{\frac{1}{B-1}\sum_b(\hat\theta^*_b-\bar\theta^*)^2}$.
3. Bias: $\widehat{\operatorname{Bias}}=\bar\theta^*-\hat\theta_n$.
4. CIs (level $1-\alpha$):
    - **Percentile**: $[\hat q_{\alpha/2},\ \hat q_{1-\alpha/2}]$.
    - **Basic** (pivotal): $[2\hat\theta_n-\hat q_{1-\alpha/2},\ 2\hat\theta_n-\hat q_{\alpha/2}]$.
    - **Normal**: $\hat\theta_n\pm z_{1-\alpha/2}\,\widehat{\operatorname{se}}_\text{boot}$.

Two error sources:
- $\hat F_n\ne F$: statistical error. Shrinks only w/ $n$.
- Finite $B$: Monte Carlo error. Shrinks w/ $B$.
    - SE: $B\sim10^2$ is usually enough. Percentile CI: $B\gtrsim10^3$ ← tail quantiles are noisy.
```

```{tip} Derivation
:class: dropdown
*Where does the basic interval come from?*

1. Approximate the law of $\hat\theta_n-\theta$ by the law of $\hat\theta^*-\hat\theta_n$.
2. → $P\left(\hat q_{\alpha/2}-\hat\theta_n\le\hat\theta_n-\theta\le\hat q_{1-\alpha/2}-\hat\theta_n\right)\approx1-\alpha$.
3. Solve for $\theta$ → $[2\hat\theta_n-\hat q_{1-\alpha/2},\ 2\hat\theta_n-\hat q_{\alpha/2}]$.

*How many distinct points does a resample contain?*

1. $P(X_i\notin\text{resample})=(1-\frac1n)^n\to e^{-1}\approx0.368$.
2. → each resample holds $\approx63.2\%$ of the distinct points. Rest = duplicates.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class Bootstrap:
    def __init__(self, stat, B=2000, seed=0):
        self.stat = stat                                   ## statistic T: sample -> scalar
        self.B = B
        self.rng = np.random.default_rng(seed)

    def __call__(self, x, alpha=0.05):
        n = len(x)
        ## B resamples of size n, WITH replacement = i.i.d. draws from the empirical dist
        idx = self.rng.integers(0, n, size=(self.B, n))
        theta_star = np.array([self.stat(x[i]) for i in idx])          ## [B]
        se = theta_star.std(ddof=1)                                    ## bootstrap SE
        lo, hi = np.quantile(theta_star, [alpha / 2, 1 - alpha / 2])   ## percentile CI
        return se, (lo, hi)

## Example: median of a skewed sample -> no simple SE formula
x = np.random.default_rng(1).exponential(scale=1.0, size=50)
se, ci = Bootstrap(np.median)(x)
print(se, ci)          ## se of order 1/sqrt(50) ~ 0.14 (asymptotic SE of the Exp(1) median)

## Sanity check: for the mean, ideal bootstrap SE = plug-in sigma_hat / sqrt(n) (ddof=0)
se_mean, _ = Bootstrap(np.mean)(x)
print(se_mean, x.std(ddof=0) / np.sqrt(len(x)))   ## ~equal
```
````

```{attention} Q&A
:class: dropdown
*Why does it work?*

- $\hat F_n\to F$ (Glivenko–Cantelli) → for smooth statistics, resampling from $\hat F_n$ mimics sampling from $F$.
- Consistent for smooth functionals (e.g., mean w/ finite variance, smooth functions of means).

*When does it fail?*

- **Extremes**: $\hat\theta_n=\max_iX_i$. $P(\hat\theta^*=\hat\theta_n)=1-(1-\frac1n)^n\to0.632$ → bootstrap law has a big atom; true law is continuous → inconsistent.
- **Heavy tails**: infinite variance → bootstrap of the mean is inconsistent. Fix: $m$-out-of-$n$ bootstrap ($m\ll n$).
- **Dependent data**: i.i.d. resampling destroys autocorrelation → SE wrong (too small under positive autocorrelation). Fix: **block bootstrap** (resample contiguous blocks).
- **Small $n$**: $\hat F_n$ is a poor proxy → percentile CIs undercover.
- **Boundary params / some non-smooth statistics** (e.g., max; the median still works): limit is non-Normal or depends on $\theta$.

*Does a larger $B$ fix a failing bootstrap?*

- No. $B$ only removes Monte Carlo error. The gap $\hat F_n\ne F$ stays.

*Percentile vs basic vs Normal interval?*

- Percentile: transformation-respecting; correct if some monotone transform makes $\hat\theta$ symmetric around $\theta$.
- Basic: right if $\hat\theta_n-\theta$ is approximately pivotal.
- Normal: right if the sampling distribution is near-Normal; uses only the SE.
- **BCa** (bias-corrected & accelerated) adjusts the percentile quantiles for bias & skew → better coverage.

*Parametric vs nonparametric bootstrap?*

- Nonparametric: resample from $\hat F_n$.
- Parametric: simulate from the fitted model $f(\cdot;\hat\theta_n)$ → more efficient if the model is right, wrong if it is not.

*Bootstrap vs jackknife?*

- Jackknife: $n$ leave-one-out refits → linear approximation.
- Jackknife SE is inconsistent for non-smooth statistics (e.g., median). Bootstrap SE works there.

*Relation to ML?*

- [Bagging](../../ml/supervised.md#bagging) = bootstrap resamples + averaged models.
- Left-out $\approx36.8\%$ per resample → out-of-bag error estimate.
```

&nbsp;

## Hypothesis Testing

### Hypothesis Test
- **What**: Data-based rule to reject $H_0$ or not, w/ type I error rate $\le\alpha$.
- **Why**: Data are noisy → need a rule that separates real effects from chance at a controlled false-alarm rate.
- **How**:
    1. State $H_0$ (default claim, e.g., no effect) & $H_1$ (claim to detect). Fix $\alpha$ & sidedness before seeing data.
    2. Pick a test statistic $T$ w/ known distribution under $H_0$ (exact or asymptotic).
    3. Set rejection region $R$ = values of $T$ most extreme under $H_0$, w/ $P_{H_0}(T\in R)\le\alpha$.
    4. Compute $T$ on data → reject $H_0$ iff $T\in R$ ⟺ [p-value](#p-value) $\le\alpha$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\Theta_0$: Null param set.
    - $\Theta_1$: Alternative param set, $\Theta_0\cap\Theta_1=\emptyset$.
    - $T=T(X_1,\dots,X_n)$: Test statistic.
    - $R$: Rejection region.
    - $\gamma(\theta)$: Power function.
    - $\beta$: Type II error rate.

Definition:

$$
H_0:\theta\in\Theta_0\qquad\text{vs.}\qquad H_1:\theta\in\Theta_1
$$

- Simple hypothesis: a single point ($\Theta_0=\{\theta_0\}$). Composite: more than one point.

$$
\gamma(\theta)=P_\theta(T\in R)
$$

- Level-$\alpha$ test: $\sup_{\theta\in\Theta_0}\gamma(\theta)\le\alpha$. Size = the sup itself.
- Type II error at $\theta\in\Theta_1$: $\beta(\theta)=1-\gamma(\theta)$. Power = $\gamma(\theta)$.

Sidedness, for $H_0:\theta=\theta_0$ & $T=(\hat\theta-\theta_0)/\operatorname{se}\sim N(0,1)$ under $H_0$ (large $T$ → large $\theta$):

$$
\begin{cases}
H_1:\theta>\theta_0 & R=\{T>z_{1-\alpha}\}\\
H_1:\theta<\theta_0 & R=\{T<z_{\alpha}\}\\
H_1:\theta\ne\theta_0 & R=\{|T|>z_{1-\alpha/2}\}
\end{cases}
$$

CI ↔ test duality:

$$
\text{Reject }H_0:\theta=\theta_0\text{ at level }\alpha\iff\theta_0\notin C_{1-\alpha}
$$

- $C_{1-\alpha}$: $(1-\alpha)$ [confidence set](#confidence-interval) for $\theta$.
- Both directions hold: invert a family of level-$\alpha$ tests → a $(1-\alpha)$ confidence set; a $(1-\alpha)$ confidence set → a level-$\alpha$ test.
```

```{dropdown} Table: Error Types
| Decision | $H_0$ true | $H_0$ false |
|:--|:--|:--|
| Reject $H_0$ | Type I error (false positive), rate $\le\alpha$ | Correct, rate = power $1-\beta$ |
| Fail to reject $H_0$ | Correct, rate $\ge1-\alpha$ | Type II error (false negative), rate $\beta$ |
```

```{attention} Q&A
:class: dropdown
*Why "fail to reject" and not "accept"?*

- Non-rejection = data compatible w/ $H_0$, not evidence for $H_0$.
- Low power → non-rejection is likely even if $H_1$ is true.
- To claim "no meaningful effect": show the CI lies inside a pre-set equivalence margin (equivalence test, TOST).

*Why does $H_0$ hold the "no effect" claim?*

- The test controls only the type I rate → put the claim that is costly to make wrongly (a discovery) in $H_1$.
- → Asymmetric: strong conclusion only on rejection.

*How do $\alpha$ & $\beta$ trade off?*

- Fixed $n$: shrink $R$ → $\alpha$⬇️ → $\beta$⬆️.
- Lower both → $n$⬆️ or noise⬇️.

*What drives power?*

- Effect size⬆️, $n$⬆️, noise $\sigma$⬇️, $\alpha$⬆️ → power⬆️.
- One-sided test in the correct direction → power⬆️; in the wrong direction → power $\le\alpha$.

*When is a one-sided test valid?*

- Only if the other direction cannot change the decision, & you fix the side before seeing data.
- Pick the side after seeing data → real type I rate doubles.

*Statistical vs practical significance?*

- Large $n$ → tiny, useless effects become significant.
- Report the effect size & its CI, not only reject/not reject.

*What does the CI ↔ test duality give you?*

- One CI = all $\theta_0$ not rejected → more information than one test.
- CI excludes $\theta_0$ ⟺ two-sided test rejects at matching $\alpha$ (same statistic & SE).
```

&nbsp;

### p-value
- **What**: Probability, under $H_0$, of a test statistic at least as extreme as the observed one.
- **Why**: One evidence scale for any test → each reader applies their own $\alpha$.
- **How**:
    1. Compute observed statistic $t_\text{obs}$.
    2. Take the tail probability of $T$ under $H_0$ beyond $t_\text{obs}$, in the direction(s) of $H_1$.
    3. Small p → data unusual under $H_0$ → reject iff $p\le\alpha$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $T$: Test statistic, large = extreme.
    - $t_\text{obs}$: Observed value of $T$.
    - $\Theta_0$: Null param set.
    - $F_0$: CDF of $T$ under a simple $H_0$.

Definition:

$$
p=\sup_{\theta\in\Theta_0}P_\theta\left(T\ge t_\text{obs}\right)
$$

- Two-sided $z$-statistic: $p=2\left(1-\Phi(|z_\text{obs}|)\right)$.

Properties:
- $p$ = smallest $\alpha$ at which the test rejects (nested rejection regions).
- Validity: $P_{H_0}(p\le u)\le u$ $\forall u\in[0,1]$ → "reject iff $p\le\alpha$" is a level-$\alpha$ test.
- Continuous $T$ & simple $H_0$ → $p\sim U(0,1)$ exactly.
- Discrete $T$ → $P_{H_0}(p\le u)\le u$, often strict → conservative.
- Under $H_1$ → mass shifts toward 0.
```

```{tip} Derivation
:class: dropdown
*Why is $p$ uniform under a continuous simple null?*

1. $F_0$ continuous → $p=P(T\ge t_\text{obs})=1-F_0(t_\text{obs})$ → as a RV, $p=1-F_0(T)$.
2. Probability integral transform: $F_0$ continuous → $F_0(T)\sim U(0,1)$.
3. → $P(p\le u)=P\left(F_0(T)\ge1-u\right)=u$.
```

```{note} Example
:class: dropdown
- 10 coin flips, 9 heads. $H_0$: fair coin. Two-sided.
- Outcomes at least as extreme: $\{0,1,9,10\}$ heads.

$$
p=\frac{\binom{10}{0}+\binom{10}{1}+\binom{10}{9}+\binom{10}{10}}{2^{10}}=\frac{22}{1024}\approx0.021
$$

- $p\le0.05$ → reject at $\alpha=0.05$.
```

```{attention} Q&A
:class: dropdown
*Is the p-value the probability that $H_0$ is true?*

- No. $p=P(\text{data this extreme}\mid H_0)$, not $P(H_0\mid\text{data})$.
- $P(H_0\mid\text{data})$ needs a prior → [Bayes' rule](bayes.md#bayes-rule).
- Low prior on $H_1$ → many results w/ $p<0.05$ are still false positives.

*Does a small p mean a large effect?*

- No. $p$ mixes effect size & $n$.
- Huge $n$ → tiny effect, $p\approx0$.

*Does $p>\alpha$ show no effect?*

- No. Absence of evidence ≠ evidence of absence ← low power.

*What is p-hacking?*

- Try many analyses (metrics, subgroups, outlier rules, covariates) → report the one w/ $p\le\alpha$ → real type I rate ≫ $\alpha$.
- Fix: pre-register the analysis; correct for [multiple comparisons](#multiple-comparisons).

*What is optional stopping?*

- Peek after each batch, stop at the first $p\le\alpha$ → each look = new chance to cross the threshold.
- Unlimited peeking → type I rate → 1 ← law of the iterated logarithm.
- Fix: fixed horizon, or sequential tests built for repeated looks.

*Why does uniformity under $H_0$ matter?*

- Basis of multiple-testing procedures.
- p-value histogram: flat part ≈ true nulls; spike near 0 ≈ true effects.

*Is the p-value itself stable?*

- No. $p$ is a RV → replications of a true effect at moderate power give widely spread p-values.
```

&nbsp;

### Neyman–Pearson Lemma
- **What**: For simple $H_0$ vs. simple $H_1$, the likelihood-ratio test is the most powerful test of its size.
- **Why**: Defines the optimal test → foundation of LR-based tests.
    - *Why does it work?* Spend the $\alpha$ budget on outcomes w/ the most power per unit of type I probability.
    - Ratio = $L(\theta_1)/L(\theta_0)$ → fill $R$ from the highest ratio down until $\alpha$ is spent (fractional knapsack).
- **How**:
    1. Compute $\Lambda=L(\theta_1)/L(\theta_0)$.
    2. Reject $H_0$ when $\Lambda>c$. Pick $c$ so that size = $\alpha$.
    3. Any other test w/ size $\le\alpha$ → power $\le$ LR test power.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\theta_0$: Null param value.
    - $\theta_1$: Alternative param value.
    - $x$: Full observed sample.
    - $\phi(x)\in[0,1]$: Test function = prob. of rejecting $H_0$ given $x$.
    - $c\ge0$: LR threshold.
    - $\tau\in[0,1]$: Rejection prob. on the boundary.

Statement: $H_0:\theta=\theta_0$ vs. $H_1:\theta=\theta_1$. Define

$$
\phi^*(x)=\begin{cases}
1 & L(\theta_1)>c\,L(\theta_0)\\
\tau & L(\theta_1)=c\,L(\theta_0)\\
0 & L(\theta_1)<c\,L(\theta_0)
\end{cases}
\qquad\text{w/ }E_{\theta_0}[\phi^*]=\alpha
$$

- $L(\theta)$ = joint density of $x$ under $\theta$.
- $\tau$ matters only for discrete data, where $P(\Lambda=c)>0$.

Then for every test $\phi$ w/ $E_{\theta_0}[\phi]\le\alpha$:

$$
E_{\theta_1}[\phi^*]\ge E_{\theta_1}[\phi]
$$
```

```{tip} Derivation
:class: dropdown
1. Pointwise, for all $x$:

    $$
    \left(\phi^*(x)-\phi(x)\right)\left(L(\theta_1)-c\,L(\theta_0)\right)\ge0
    $$

    - $L(\theta_1)>cL(\theta_0)$ → $\phi^*=1\ge\phi$ → both factors $\ge0$.
    - $L(\theta_1)<cL(\theta_0)$ → $\phi^*=0\le\phi$ → both factors $\le0$.
    - Equality → second factor $=0$.
2. Integrate over $x$:

    $$
    E_{\theta_1}[\phi^*-\phi]-c\,E_{\theta_0}[\phi^*-\phi]\ge0
    $$

3. Rearrange:

    $$
    E_{\theta_1}[\phi^*]-E_{\theta_1}[\phi]\ge c\left(\alpha-E_{\theta_0}[\phi]\right)\ge0
    $$
```

```{note} Example
:class: dropdown
- $X_i\sim N(\theta,\sigma^2)$, $\sigma$ known. $H_0:\theta=0$ vs. $H_1:\theta=\theta_1>0$.

$$
\log\Lambda=\frac{\theta_1}{\sigma^2}\sum_{i=1}^nX_i-\frac{n\theta_1^2}{2\sigma^2}
$$

- Increasing in $\bar{X}_n$ → reject when $\bar{X}_n>c'$.
- Size $\alpha$ → $c'=z_{1-\alpha}\,\sigma/\sqrt{n}$ → the one-sided $z$-test.
- $c'$ does not depend on $\theta_1$ → same test is most powerful for every $\theta_1>0$ → UMP for $H_1:\theta>0$.
```

```{attention} Q&A
:class: dropdown
*Does it give the best test for composite hypotheses?*

- Not directly.
- One-sided $H_1$ & monotone likelihood ratio family → LR region does not depend on $\theta_1$ → UMP test (Karlin–Rubin).
- Two-sided $H_1$ → best regions for $\theta_1>\theta_0$ & $\theta_1<\theta_0$ differ → no UMP test in general → restrict to unbiased tests.

*How does it connect to classification & ROC?*

- Binary classification w/ known class densities = simple vs. simple test.
- $\alpha$ = FPR. Power = TPR.
- Thresholding the LR at varying $c$ → an ROC curve that no other classifier's ROC exceeds.

*Why randomize on the boundary?*

- Discrete data → $P(\Lambda>c)$ jumps → exact $\alpha$ may be unreachable.
- Practice: skip randomization, accept size $<\alpha$.

*Does the lemma assume i.i.d. data?*

- No. It needs only the two joint densities of the full sample.
```

&nbsp;

### Likelihood Ratio Test
- **What**: Test comparing the max likelihood under $H_0$ to the max likelihood under the full model.
- **Why**: Real hypotheses are composite & have nuisance params → Neyman–Pearson does not apply → maximize the likelihood within each hypothesis.
- **How**:
    1. Fit the model restricted to $\Theta_0$ → max likelihood $L(\hat\theta_0)$.
    2. Fit the full model → [MLE](#mle) $\hat\theta_n$ → $L(\hat\theta_n)$.
    3. Ratio $\Lambda_n\in(0,1]$. Small $\Lambda_n$ → $H_0$ fits much worse → reject.
    4. Calibrate w/ Wilks: $-2\log\Lambda_n\approx\chi^2_r$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\Theta_0\subset\Theta$: Null param set, nested in the full model.
    - $\hat\theta_n$: Unrestricted MLE over $\Theta$.
    - $\hat\theta_0$: Restricted MLE over $\Theta_0$.
    - $r$: Degrees of freedom.
    - $\chi^2_{r,q}$: $q$-quantile of $\chi^2_r$.

Statistic:

$$
\Lambda_n=\frac{\sup_{\theta\in\Theta_0}L(\theta)}{\sup_{\theta\in\Theta}L(\theta)}=\frac{L(\hat\theta_0)}{L(\hat\theta_n)}
$$

Wilks' theorem, under $H_0$ & regularity:

$$
-2\log\Lambda_n=2\left[\ell(\hat\theta_n)-\ell(\hat\theta_0)\right]\xrightarrow{d}\chi^2_r,\qquad r=\dim\Theta-\dim\Theta_0
$$

- Reject iff $-2\log\Lambda_n>\chi^2_{r,1-\alpha}$.
- Regularity: true $\theta$ in the interior of $\Theta$; identifiable; smooth $\ell$; finite, nonsingular $I(\theta)$; consistent MLE.

Trinity, scalar $\theta$, $H_0:\theta=\theta_0$:

$$\begin{align*}
&\text{LR}:    &&2\left[\ell(\hat\theta_n)-\ell(\theta_0)\right]\\
&\text{Wald}:  &&n\,I(\hat\theta_n)\left(\hat\theta_n-\theta_0\right)^2\\
&\text{Score}: &&\frac{\ell'(\theta_0)^2}{n\,I(\theta_0)}
\end{align*}$$

- $\ell'(\theta_0)$: Score at $\theta_0$.
- All three $\xrightarrow{d}\chi^2_1$ under $H_0$ → asymptotically equivalent.
- Observed information may replace $nI(\cdot)$.
```

```{tip} Derivation
:class: dropdown
*Why $\chi^2$? (scalar sketch, $H_0:\theta=\theta_0$)*

1. Taylor-expand $\ell(\theta_0)$ around $\hat\theta_n$. Use $\ell'(\hat\theta_n)=0$:

    $$
    \ell(\theta_0)\approx\ell(\hat\theta_n)+\frac12\ell''(\hat\theta_n)(\theta_0-\hat\theta_n)^2
    $$

2. → $2\left[\ell(\hat\theta_n)-\ell(\theta_0)\right]\approx-\ell''(\hat\theta_n)(\hat\theta_n-\theta_0)^2$.
3. LLN & consistency → $-\ell''(\hat\theta_n)/n\xrightarrow{P}I(\theta_0)$.
4. MLE asymptotic normality → $\sqrt{n\,I(\theta_0)}(\hat\theta_n-\theta_0)\xrightarrow{d}N(0,1)$.
5. Slutsky → statistic $\approx Z^2\sim\chi^2_1$.
6. Vector case: same quadratic form; the restricted fit removes $\dim\Theta_0$ free directions → $\chi^2_r$.
```

```{note} Example
:class: dropdown
- $X_i\sim N(\mu,\sigma^2)$, $\sigma$ known, $H_0:\mu=\mu_0$. MLE $\hat\mu=\bar{X}_n$.

$$
2\left[\ell(\bar{X}_n)-\ell(\mu_0)\right]=\frac{\sum_i(X_i-\mu_0)^2-\sum_i(X_i-\bar{X}_n)^2}{\sigma^2}=\frac{n(\bar{X}_n-\mu_0)^2}{\sigma^2}=Z^2
$$

- Exactly $\chi^2_1$ → LRT = two-sided $z$-test.
```

```{dropdown} Table: LR vs. Wald vs. Score
| Test | Fits needed | Invariant to reparameterization | Typical use |
|:--|:--|:--|:--|
| LR | Restricted & full | ✅ | Nested model comparison, GLM deviance |
| Wald | Full only | ❌ | Coefficient p-values in regression output, CI $\hat\theta\pm z\cdot\text{SE}$ |
| Score (Lagrange multiplier) | Restricted only | ✅ | Pearson $\chi^2$, cheap check for adding a variable |
```

```{attention} Q&A
:class: dropdown
*When does Wilks fail?*

- $H_0$ on the boundary of $\Theta$ (e.g., variance component $=0$) → limit = mixture, e.g., $\frac12\chi^2_0+\frac12\chi^2_1$ for one param → naive $\chi^2_1$ p-value too large (conservative).
- Param not identifiable under $H_0$ (e.g., #mixture components) → no $\chi^2$ limit.
- Non-nested models → LRT does not apply → use AIC/BIC.
- Small $n$ → $\chi^2$ approximation poor.

*Why $-2\log$?*

- $\log$ → sums over samples.
- Factor 2 cancels the $\frac12$ of the quadratic Taylor term → limit is exactly $\chi^2$.

*How does it relate to GLM deviance?*

- Deviance difference of nested GLMs = $-2\log\Lambda_n$ (up to dispersion).

*When do Wald & LR disagree?*

- Small $n$, $\hat\theta_n$ near the boundary, strong curvature of $\ell$.
- Logistic regression w/ near-separation → Wald SE blows up → Wald p-value can rise as the effect grows (Hauck–Donner). Prefer LR.

*Is the LRT optimal?*

- Simple vs. simple → yes, = Neyman–Pearson test.
- Composite → no finite-sample optimality guarantee.
```

&nbsp;

### z-test & t-test
- **What**: Tests on means that compare (estimate − null value) / standard error to a reference distribution.
- **Why**: Most common inference question: is a mean, or a difference of means, different from a reference value?
- **How**:
    1. Compute the standardized difference: (estimate − null value) / SE.
    2. **z-test**: $\sigma$ known, or $n$ large → compare to $N(0,1)$.
    3. **t-test**: $\sigma$ estimated by $s$ & Normal data → compare to $t_\nu$ (heavier tails than $N(0,1)$).
    4. Pick the design: one-sample, two-sample (pooled or [Welch](#welchs-t-test)), [paired](#paired-t-test).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $Y_1,\dots,Y_{n_2}$: Second sample (two-sample tests).
- Params:
    - $\mu$: Population mean.
    - $\sigma^2$: Population variance.
    - $\mu_X$: Mean of $X$ population.
    - $\mu_Y$: Mean of $Y$ population.
- Misc:
    - $\mu_0$: Null value of $\mu$.
    - $n_1$: Size of $X$ sample (two-sample tests; replaces $n$).
    - $n_2$: Size of $Y$ sample.
    - $s_X^2$: Sample variance of $X$.
    - $s_Y^2$: Sample variance of $Y$.
    - $t_\nu$: Student $t$ distribution w/ $\nu$ df.

One-sample, $H_0:\mu=\mu_0$:

$$
Z=\frac{\bar{X}_n-\mu_0}{\sigma/\sqrt{n}}\sim N(0,1),\qquad T=\frac{\bar{X}_n-\mu_0}{s/\sqrt{n}}\sim t_{n-1}
$$

Two-sample, $H_0:\mu_X=\mu_Y$, independent samples:

$$
Z=\frac{\bar{X}_{n_1}-\bar{Y}_{n_2}}{\sqrt{\sigma_X^2/n_1+\sigma_Y^2/n_2}}\sim N(0,1)
$$

- $\sigma_X^2$: Known variance of $X$ population.
- $\sigma_Y^2$: Known variance of $Y$ population.
- Large $n_1,n_2$ → plug in $s_X^2,s_Y^2$.

Two-sample pooled $t$, equal variances assumed:

$$
T=\frac{\bar{X}_{n_1}-\bar{Y}_{n_2}}{s_p\sqrt{1/n_1+1/n_2}}\sim t_{n_1+n_2-2},\qquad s_p^2=\frac{(n_1-1)s_X^2+(n_2-1)s_Y^2}{n_1+n_2-2}
$$

- $s_p^2$: Pooled variance.
- Exact $t$ distributions need Normal data. Otherwise asymptotic $N(0,1)$ by [CLT](limits.md#central-limit-theorem).
```

```{tip} Derivation
:class: dropdown
*Why $t_{n-1}$ in the one-sample case?*

1. Normal data, $H_0$ → $Z=\sqrt{n}(\bar{X}_n-\mu_0)/\sigma\sim N(0,1)$.
2. $V=(n-1)s^2/\sigma^2\sim\chi^2_{n-1}$, independent of $\bar{X}_n$ (property specific to the Normal).
3. $T=Z/\sqrt{V/(n-1)}$ → definition of $t_{n-1}$; $\sigma$ cancels.
4. df $=n-1$ ← $s^2$ uses 1 df to estimate the mean.
```

```{attention} Q&A
:class: dropdown
*Assumptions?*

- Independent observations. Most critical.
- Normality: needed for exact $t$ at small $n$.
- Equal variances: pooled two-sample $t$ only.

*Is Normality required?*

- Small $n$: approximately. Robust to mild skew; sensitive to heavy tails & outliers.
- Large $n$ & finite variance: CLT → $T\approx N(0,1)$ under $H_0$ → valid approximately.
- Skew or heavy tails → need larger $n$ for the CLT to apply.

*$t$ or $z$?*

- $n\to\infty$ → $t_{n-1}\to N(0,1)$.
- Small $n$ → $t$ has wider critical values, e.g., $t_{9,0.975}=2.262$ vs. $z_{0.975}=1.960$ ← extra uncertainty from estimating $\sigma$.

*What if observations are dependent?*

- Many rows per user, time series, clusters → SE too small → type I rate inflated.
- Fix: aggregate to the independent unit, cluster-robust SE, block [bootstrap](#bootstrap).

*How do these tests relate to linear regression?*

- One-sample $t$ = $t$-test on the intercept of an intercept-only OLS.
- Pooled two-sample $t$ = $t$-test on the slope of $y\sim1+\text{group indicator}$ (homoskedastic SE).

*Non-parametric alternatives?*

- [Permutation test](#permutation-test): any statistic, exact under exchangeability.
- Mann–Whitney U / Wilcoxon signed-rank: rank-based; test stochastic ordering, not means.

*How do proportions fit?*

- Binary data → $\sigma^2=p(1-p)$ → two-proportion $z$-test. See [A/B Testing](#a-b-testing).
```

&nbsp;

#### Welch's t-test
- **What**: Two-sample $t$-test w/o the equal-variance assumption.
- **Why**: Pooled $t$ w/ unequal variances & unequal $n$ → wrong type I rate.
    - Smaller group has larger variance → $s_p$ too small → type I rate too high.
- **How**: Use the unpooled SE. Approximate the df by Welch–Satterthwaite.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\nu$: Welch–Satterthwaite df, non-integer.

$$
T=\frac{\bar{X}_{n_1}-\bar{Y}_{n_2}}{\sqrt{s_X^2/n_1+s_Y^2/n_2}}\ \dot\sim\ t_\nu
$$

$$
\nu=\frac{\left(s_X^2/n_1+s_Y^2/n_2\right)^2}{\frac{\left(s_X^2/n_1\right)^2}{n_1-1}+\frac{\left(s_Y^2/n_2\right)^2}{n_2-1}}
$$

- $\min(n_1,n_2)-1\le\nu\le n_1+n_2-2$.
```

```{tip} Derivation
:class: dropdown
*Where does $\nu$ come from?*

1. $W=s_X^2/n_1+s_Y^2/n_2$ is not a scaled $\chi^2$. Approximate $W\approx\frac{E[W]}{\nu}\chi^2_\nu$.
2. Match the first two moments: $\text{Var}(W)=2E[W]^2/\nu$ → $\nu=2E[W]^2/\text{Var}(W)$.
3. Normal data → $\text{Var}(s^2)=2\sigma^4/(n-1)$ →

    $$
    \nu=\frac{\left(\sigma_X^2/n_1+\sigma_Y^2/n_2\right)^2}{\frac{\sigma_X^4}{n_1^2(n_1-1)}+\frac{\sigma_Y^4}{n_2^2(n_2-1)}}
    $$

4. Plug in $s_X^2,s_Y^2$ for $\sigma_X^2,\sigma_Y^2$.
5. Bounds: Cauchy–Schwarz → $\nu\le(n_1-1)+(n_2-1)$; denominator $\le(s_X^2/n_1+s_Y^2/n_2)^2/\min(n_1-1,n_2-1)$ → lower bound.
```

````{important} Code
:class: dropdown
```python
import numpy as np
from scipy import stats

class WelchTTest:
    def __call__(self, x, y):
        n1, n2 = len(x), len(y)
        ## variance of each sample mean, unpooled (ddof=1 -> s^2)
        a, b = x.var(ddof=1) / n1, y.var(ddof=1) / n2
        t = (x.mean() - y.mean()) / np.sqrt(a + b)
        ## Welch-Satterthwaite: moment-match a + b to a scaled chi^2
        nu = (a + b) ** 2 / (a ** 2 / (n1 - 1) + b ** 2 / (n2 - 1))
        p = 2 * stats.t.sf(abs(t), nu)  ## two-sided
        return t, nu, p

## Example
x = np.array([5.1, 4.9, 5.6, 5.8, 6.0, 5.4])
y = np.array([4.2, 6.5, 3.1, 5.9, 4.0])
print(WelchTTest()(x, y))  ## t ≈ 1.11, nu ≈ 4.59, p ≈ 0.32
```
````

```{attention} Q&A
:class: dropdown
*Why use it as the default?*

- Equal variances → loses little power vs. pooled $t$.
- Unequal variances → keeps the type I rate near $\alpha$.
- R's `t.test` uses Welch by default.

*Does Welch fix non-Normality?*

- No. It fixes only unequal variances. Small skewed samples still need care.
```

&nbsp;

#### Paired t-test
- **What**: One-sample $t$-test on within-pair differences.
- **Why**: Same unit measured twice (before/after; two models on the same test examples) → pairs correlated → two-sample test wastes the correlation.
- **How**: $D_i=X_i-Y_i$ → one-sample $t$-test of $H_0:\mu_D=0$, df $=n-1$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $D_i=X_i-Y_i$: Within-pair difference, $i=1,\dots,n$.
    - $\mu_D$: Mean of $D_i$.
    - $\bar{D}_n$: Sample mean of $D_i$.
    - $s_D$: Sample STD of $D_i$.

$$
T=\frac{\bar{D}_n}{s_D/\sqrt{n}}\sim t_{n-1}
$$

$$
\text{Var}(\bar{D}_n)=\frac{\sigma_X^2+\sigma_Y^2-2\,\text{Cov}(X,Y)}{n}
$$

- $\text{Cov}(X,Y)>0$ → variance⬇️ → power⬆️ vs. two-sample test.
- Normality is needed for $D_i$ only, not for $X_i$ or $Y_i$.
```

```{attention} Q&A
:class: dropdown
*What if you analyze paired data w/ a two-sample test?*

- Ignores positive covariance → SE overestimated → conservative, power⬇️.

*Can you compare two ML models w/ a paired test?*

- Per-example losses on one fixed test set → ✅ paired $t$ on per-example differences.
- Scores across CV folds → ❌ naive paired $t$ ← overlapping training sets → correlated fold scores → SE too small.
```

&nbsp;

### Chi-Squared Test
- **What**: Test comparing observed category counts to expected counts under $H_0$.
- **Why**: Categorical data → means & $t$-tests do not apply → need a test on count distributions & associations.
- **How**:
    1. Get expected counts $E$ under $H_0$.
    2. Sum the squared, scaled gaps $(O-E)^2/E$ over cells.
    3. Large sum → reject. Compare to $\chi^2_\text{df}$.
    - **Goodness-of-fit**: one categorical variable vs. a hypothesized distribution.
    - **Independence**: two categorical variables in a contingency table.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $K$: #categories.
    - $O_k$: Observed count in category $k$.
    - $E_k$: Expected count in category $k$ under $H_0$.
    - $p_{0k}$: $H_0$ probability of category $k$.
    - $d$: #params estimated from data to get $E_k$.
    - $r$: #rows of contingency table.
    - $c$: #columns of contingency table.
    - $i$: Row idx (overrides Tier 1 sample idx).
    - $j$: Column idx (overrides Tier 1 feature idx).
    - $O_{ij}$: Observed count in cell $(i,j)$.
    - $O_{i\cdot}$: Row total.
    - $O_{\cdot j}$: Column total.

Goodness-of-fit:

$$
X^2=\sum_{k=1}^K\frac{(O_k-E_k)^2}{E_k}\xrightarrow{d}\chi^2_{K-1-d},\qquad E_k=n\,p_{0k}
$$

Independence:

$$
X^2=\sum_{i=1}^r\sum_{j=1}^c\frac{(O_{ij}-E_{ij})^2}{E_{ij}}\xrightarrow{d}\chi^2_{(r-1)(c-1)},\qquad E_{ij}=\frac{O_{i\cdot}O_{\cdot j}}{n}
$$

- $n$: Total count.
- $\chi^2_{K-1-d}$ needs $d$ params fitted by MLE on the binned counts. MLE on raw data → limit lies between $\chi^2_{K-1-d}$ & $\chi^2_{K-1}$.
- Valid approximation: all $E\ge5$, or $\ge80\%$ of cells $\ge5$ & none $<1$. Else merge categories, or use Fisher's exact test ($2\times2$).
```

```{tip} Derivation
:class: dropdown
*Where do the df come from?* Free params of the full model − free params under $H_0$ (Wilks).

1. GOF: $K$ probabilities sum to 1 → $K-1$ free. $H_0$ w/ $d$ fitted params → $d$ free.
    - → df $=K-1-d$.
2. Independence: full table → $rc-1$ free. $H_0$: row marginals $(r-1)$ + column marginals $(c-1)$.
    - → df $=rc-1-(r-1)-(c-1)=(r-1)(c-1)$.
```

```{note} Example
:class: dropdown
- Treatment: 30/100 convert. Control: 20/100 convert.
- Row totals 100, 100. Column totals 50, 150. $n=200$.
- $E$: 25 & 75 in each row.

$$
X^2=\frac{5^2}{25}+\frac{5^2}{75}+\frac{5^2}{25}+\frac{5^2}{75}\approx2.667,\qquad\text{df}=1,\quad p\approx0.102
$$

- Two-proportion pooled $z$: $\hat p=0.25$, SE $=\sqrt{0.25\cdot0.75\cdot(2/100)}\approx0.0612$ → $z\approx1.633$ → $z^2\approx2.667=X^2$.
```

````{important} Code
:class: dropdown
```python
import numpy as np
from scipy import stats

class ChiSquaredIndependence:
    def __call__(self, O):
        n = O.sum()
        ## H0 independence -> expected = row total * col total / n
        E = np.outer(O.sum(axis=1), O.sum(axis=0)) / n
        x2 = ((O - E) ** 2 / E).sum()
        df = (O.shape[0] - 1) * (O.shape[1] - 1)
        return x2, df, stats.chi2.sf(x2, df)

## Example: rows = treatment / control, cols = converted / not
O = np.array([[30, 70], [20, 80]])
print(ChiSquaredIndependence()(O))  ## (2.667, 1, 0.102)
```
````

```{attention} Q&A
:class: dropdown
*How does the $2\times2$ test relate to the two-proportion $z$-test?*

- $X^2=z^2$ (pooled SE, no continuity correction) → same two-sided p-value.

*How does it relate to the LRT?*

- $G=2\sum O\log(O/E)$ = LRT statistic.
- Pearson $X^2$ = score test.
- Both $\xrightarrow{d}$ same $\chi^2$ under $H_0$.

*Why subtract one df per estimated param?*

- Each fitted param pulls $E$ toward $O$ → $X^2$⬇️ → fewer free directions.

*Does a large $X^2$ show a strong association?*

- No. Fixed proportions → $X^2\propto n$.
- Effect size: Cramér's $V=\sqrt{X^2/\left(n(\min(r,c)-1)\right)}$.

*Assumptions?*

- Independent observations; each unit in exactly one cell.
- Raw counts, not percentages.
- Expected counts large enough.
```

&nbsp;

### Permutation Test
- **What**: Test that builds the null distribution of a statistic by reshuffling group labels.
- **Why**: Exact p-value w/o Normality or large $n$; works for any statistic.
    - *Why does it work?* $H_0$ → labels exchangeable → every relabeling equally likely → observed statistic = uniform draw from the permutation distribution.
- **How**:
    1. Compute $T_\text{obs}$ on the real labels.
    2. Shuffle labels $B$ times → $T_b$ each time.
    3. p-value = fraction of shuffles w/ $T_b\ge T_\text{obs}$, counting the observed labeling.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $Y_1,\dots,Y_{n_2}$: Second sample.
    - $W=(X_1,\dots,X_{n_1},Y_1,\dots,Y_{n_2})$: Pooled sample.
- Misc:
    - $T$: Test statistic, large = extreme, e.g., $|\bar{X}_{n_1}-\bar{Y}_{n_2}|$.
    - $T_\text{obs}$: $T$ on the observed labels.
    - $B$: #random permutations.
    - $T_b$: $T$ on the $b$-th permutation.

Null: $H_0$: $X$ & $Y$ have the same distribution → $W$ exchangeable.

Exact, over all $\binom{n_1+n_2}{n_1}$ splits $\mathcal{S}$:

$$
p=\frac{\#\{S\in\mathcal{S}:T(S)\ge T_\text{obs}\}}{\binom{n_1+n_2}{n_1}}
$$

- $T(S)$: $T$ w/ the split $S$ as group labels.

Monte Carlo:

$$
p=\frac{1+\sum_{b=1}^B\mathbb{1}\left[T_b\ge T_\text{obs}\right]}{B+1}
$$

- $+1$: observed labeling is one of the permutations → $P_{H_0}(p\le\alpha)\le\alpha$ for any $B$; $p\ne0$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class PermutationTest:
    def __init__(self, B=9999, seed=0):
        self.B = B
        self.rng = np.random.default_rng(seed)

    def stat(self, x, y):
        return abs(x.mean() - y.mean())  ## two-sided diff in means

    def __call__(self, x, y):
        ## H0 -> labels exchangeable -> pool, then relabel at random
        w = np.concatenate([x, y])
        t_obs = self.stat(x, y)
        count = 0
        for _ in range(self.B):
            w_perm = self.rng.permutation(w)
            count += self.stat(w_perm[:len(x)], w_perm[len(x):]) >= t_obs
        ## +1: observed labeling counts as one permutation -> valid, never 0
        return (1 + count) / (self.B + 1)

## Example
x = np.array([2.1, 3.4, 2.9, 3.8])
y = np.array([1.2, 1.9, 2.4])
print(PermutationTest()(x, y))  ## ≈ 0.086 (exact: 3 of 35 splits)
```
````

```{attention} Q&A
:class: dropdown
*What exactly does it test?*

- $H_0$ = equal distributions (exchangeability), not only equal means.
- Equal means & unequal variances → can reject → not a pure test of means.
- Studentized statistic (e.g., Welch $T$) → asymptotically valid for equal means.

*How does it differ from the bootstrap?*

- Permutation: relabel w/o replacement under $H_0$ → tests; exact.
- [Bootstrap](#bootstrap): resample w/ replacement from the data → estimates the sampling distribution → CIs; approximate.

*Paired version?*

- Under $H_0$: $D_i=X_i-Y_i$ symmetric about 0 → randomly flip the sign of each $D_i$.

*Cons?*

- Compute: $B$ refits of the statistic.
- Needs exchangeability → fails for time series & clustered data → permute whole blocks/clusters.
- Tests exchangeability (equal distributions), not only equal means. A CI needs test inversion.

*Relation to randomized experiments?*

- Randomization test: randomness comes from the assignment itself → valid w/o any population model.
```

&nbsp;

## Multiple Testing

### Multiple Comparisons
- **What**: Inflation of false positives when many hypotheses are tested.
- **Why**: Each test at $\alpha$ → many true nulls → some reject by chance.
- **How**: Pick the error rate to control → adjust the thresholds.
    - **FWER**: P(≥1 false positive) → [Bonferroni](#bonferroni-correction), Holm. Use when any false claim is costly (confirmatory studies).
    - **FDR**: expected share of false discoveries among discoveries → [BH](#benjamini-hochberg-procedure). Use for screening many hypotheses.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $m$: #hypotheses (overrides Tier 1 #samples).
    - $m_0$: #true nulls.
    - $V$: #false rejections.
    - $R$: #rejections.
    - $p_1,\dots,p_m$: p-values.

All $m$ nulls true & independent, each tested at $\alpha$:

$$
P(V\ge1)=1-(1-\alpha)^m
$$

- $m=20$, $\alpha=0.05$ → $\approx0.64$.

Error rates:

$$
\text{FWER}=P(V\ge1),\qquad\text{FDR}=E\left[\frac{V}{\max(R,1)}\right]
$$

- $\text{FDR}\le\text{FWER}$ ← $V/\max(R,1)\le\mathbb{1}[V\ge1]$.
- $m_0=m$ → $V=R$ → $\text{FDR}=\text{FWER}$.
- Weak FWER control: only when all nulls are true. Strong: under every configuration of true/false nulls.
```

```{dropdown} Table: Outcomes of $m$ Tests
| | $H_0$ true | $H_0$ false | Total |
|:--|:--|:--|:--|
| Rejected | $V$ | $S$ | $R$ |
| Not rejected | $U$ | $T$ | $m-R$ |
| Total | $m_0$ | $m-m_0$ | $m$ |

- $S$: #true discoveries.
- $U$: #correct non-rejections.
- $T$: #missed effects.
```

```{attention} Q&A
:class: dropdown
*Does dependence between tests matter?*

- Positive dependence → fewer effectively independent tests → $1-(1-\alpha)^m$ overstates FWER.
- Bonferroni stays valid under any dependence, but gets more conservative.

*Where does it show up in ML?*

- A/B test w/ many metrics, variants, or segments.
- Pick the best of many models on one test set → test score optimistic.
- Univariate feature screening.

*Which procedures control FWER strongly vs. weakly?*

- Bonferroni & Holm: strong.
- BH: weak only ← all nulls true → FDR = FWER.
```

&nbsp;

#### Bonferroni Correction
- **What**: Test each of $m$ hypotheses at level $\alpha/m$.
- **Why**: Control FWER $\le\alpha$ w/o any assumption on dependence.
- **How**: Reject $H_i$ iff $p_i\le\alpha/m$ ⟺ adjusted $\min(m\,p_i,1)\le\alpha$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{I}_0$: Index set of true nulls, $|\mathcal{I}_0|=m_0$.
    - $p_{(i)}$: $i$-th smallest p-value.
    - $H_{(i)}$: Hypothesis of $p_{(i)}$.

FWER bound, via [union bound](limits.md#booles-inequality-union-bound) & valid p-values:

$$
\text{FWER}=P\left(\bigcup_{i\in\mathcal{I}_0}\left\{p_i\le\frac{\alpha}{m}\right\}\right)\le\sum_{i\in\mathcal{I}_0}P\left(p_i\le\frac{\alpha}{m}\right)\le\frac{m_0\alpha}{m}\le\alpha
$$

Holm (step-down):
1. Sort $p_{(1)}\le\dots\le p_{(m)}$.
2. For $i=1,2,\dots$: reject $H_{(i)}$ if $p_{(i)}\le\frac{\alpha}{m-i+1}$. Else stop; keep all remaining.

- Controls FWER strongly under any dependence.
- Rejects every hypothesis Bonferroni rejects → power $\ge$ Bonferroni.

Šidák: per-test level $1-(1-\alpha)^{1/m}$ → FWER $=\alpha$ exactly if all nulls are true & tests independent. Slightly larger than $\alpha/m$.
```

```{attention} Q&A
:class: dropdown
*Cons?*

- Conservative for large $m$ or positively correlated tests → power⬇️ → many missed effects.

*Why is Holm never worse than Bonferroni?*

- Step 1 uses the same threshold $\alpha/m$.
- Later thresholds $\alpha/(m-i+1)$ grow → more rejections.

*Why does the bound hold under any dependence?*

- The union bound needs no independence. Overlaps only reduce the true probability.
```

&nbsp;

#### Benjamini–Hochberg Procedure
- **What**: Step-up procedure that controls FDR at level $q$. {cite:p}`benjamini1995controlling`
- **Why**: FWER control → power collapses for large $m$. FDR tolerates a controlled share of false discoveries → more discoveries.
    - *Why does it work?* At threshold $t$: expected #false rejections $\le mt$ (null p-values uniform); actual rejections $R(t)$.
    - → Estimated false discovery proportion $=mt/R(t)$. BH picks the largest $t$ w/ estimate $\le q$.
- **How**:
    1. Sort $p_{(1)}\le\dots\le p_{(m)}$.
    2. Find the largest $k$ w/ $p_{(k)}\le kq/m$.
    3. Reject $H_{(1)},\dots,H_{(k)}$, also those w/ $p_{(i)}>iq/m$ for $i<k$.

```{note} Math
:class: dropdown
Inherits $m$ (#hypotheses), $m_0$, $V$, $R$ from [Multiple Comparisons](#multiple-comparisons).

Notations:
- Misc:
    - $q$: Target FDR level.
    - $p_{(i)}$: $i$-th smallest p-value.
    - $H_{(i)}$: Hypothesis of $p_{(i)}$.

Process:

$$
k=\max\left\{i:\ p_{(i)}\le\frac{i}{m}q\right\}
$$

- Reject $H_{(1)},\dots,H_{(k)}$. Empty set → reject none.

Guarantee, continuous null p-values:

$$
\text{FDR}\le\frac{m_0}{m}q\le q
$$

- Independent test statistics → equality $\text{FDR}=\frac{m_0}{m}q$.
- Positive regression dependence on the subset of true nulls (PRDS) → bound still holds.
- Arbitrary dependence → replace $q$ by $q/\sum_{i=1}^m\frac1i$ (Benjamini–Yekutieli).

BH-adjusted p-value:

$$
\tilde p_{(i)}=\min_{j\ge i}\min\left(\frac{m\,p_{(j)}}{j},1\right)
$$

- Reject $H_{(i)}$ iff $\tilde p_{(i)}\le q$.
```

```{note} Example
:class: dropdown
- $m=5$, $q=0.05$. Sorted p-values & thresholds $iq/m$:

| $i$ | 1 | 2 | 3 | 4 | 5 |
|:--|:--|:--|:--|:--|:--|
| $p_{(i)}$ | 0.001 | 0.008 | 0.035 | 0.039 | 0.6 |
| $iq/m$ | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 |
| $p_{(i)}\le iq/m$ | ✅ | ✅ | ❌ | ✅ | ❌ |

- $k=4$ → reject $H_{(1)},\dots,H_{(4)}$, also $H_{(3)}$ ← step-up.
- Bonferroni ($0.01$) & Holm → reject 2.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class BenjaminiHochberg:
    def __init__(self, q=0.05):
        self.q = q

    def __call__(self, p):
        m = len(p)
        order = np.argsort(p)
        ## compare i-th smallest p-value to i*q/m
        below = p[order] <= np.arange(1, m + 1) * self.q / m
        ## step-up: LARGEST i that passes, not the first failure
        k = below.nonzero()[0].max() + 1 if below.any() else 0
        reject = np.zeros(m, dtype=bool)
        reject[order[:k]] = True
        return reject

## Example
p = np.array([0.039, 0.001, 0.6, 0.008, 0.035])
print(BenjaminiHochberg(0.05)(p))  ## [ True  True False  True  True]
```
````

```{attention} Q&A
:class: dropdown
*What does FDR $=5\%$ mean?*

- On average, $\le5\%$ of rejections are false.
- An average over repeated experiments: the realized share in one run can exceed $q$.
- Not a per-discovery error probability.

*Why "step-up"?*

- Scan from the largest p-value down. The first $p_{(i)}\le iq/m$ → reject it & all smaller.
- Step-down (Holm-like) scans from the smallest & stops at the first failure → fewer rejections.

*Pros vs. Bonferroni?*

- Threshold grows w/ rank → power⬆️, esp. when many effects are real.
- Adapts: few true effects → behaves like Bonferroni; many → far less strict.

*Cons?*

- Controls an average, not the per-run worst case.
- Arbitrary dependence → needs the more conservative Benjamini–Yekutieli version.
- No strong FWER control.
```

&nbsp;

## Experiments

### A/B Testing
- **What**: Randomized controlled experiment on live traffic, analyzed as a two-sample test of a metric.
- **Why**: Randomization → arms differ only by treatment, in expectation → metric difference = causal effect.
- **How**: Operational checklist & pitfall table → [Online A/B Testing](../../ml/misc.md#online-a-b-testing).
    1. Fix the metric, $\alpha$, power $1-\beta$, MDE $\delta$ → sample size per arm.
    2. Randomize units (usually users) to control & treatment.
    3. Run to the fixed horizon.
    4. Check SRM → two-sample test ($z$ / [Welch](#welchs-t-test)) on the metric difference → report the effect & its CI.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\bar{Y}_T$: Treatment-arm metric mean.
    - $\bar{Y}_C$: Control-arm metric mean.
- Params:
    - $\mu_T$: Treatment-arm population mean.
    - $\mu_C$: Control-arm population mean.
    - $\Delta=\mu_T-\mu_C$: True effect (average treatment effect).
    - $\sigma^2$: Per-unit metric variance, equal across arms.
- Hyperparams:
    - $\beta$: Type II error rate. Power $=1-\beta$.
    - $\delta$: MDE (minimum detectable effect), absolute.
- Misc:
    - $n$: Units per arm, equal split (overrides page $n$).
    - $s_T^2$: Sample variance of treatment-arm metric.
    - $s_C^2$: Sample variance of control-arm metric.

Model: $H_0:\Delta=0$ vs. $H_1:\Delta\ne0$.

$$
\hat\Delta=\bar{Y}_T-\bar{Y}_C,\qquad\text{Var}(\hat\Delta)=\frac{2\sigma^2}{n},\qquad Z=\frac{\hat\Delta}{\sqrt{s_T^2/n+s_C^2/n}}
$$

- Reject iff $|Z|>z_{1-\alpha/2}$.

Sample size per arm:

$$
n=\frac{2\sigma^2\left(z_{1-\alpha/2}+z_{1-\beta}\right)^2}{\delta^2}
$$

- $\alpha=0.05$, $\beta=0.2$ → $2(1.960+0.842)^2\approx15.7$ → $n\approx16\,\sigma^2/\delta^2$.
- Unequal arm variances → replace $2\sigma^2$ by $\sigma_T^2+\sigma_C^2$.
- Binary metric → $\sigma^2=p(1-p)$.
- $n\propto\sigma^2/\delta^2$ → half the MDE → 4× the traffic.

MDE for a given $n$:

$$
\delta=\left(z_{1-\alpha/2}+z_{1-\beta}\right)\sqrt{\frac{2\sigma^2}{n}}
$$
```

```{tip} Derivation
:class: dropdown
*Where does the sample-size formula come from?*

1. $\Delta=\delta>0$ → $Z\approx N(\delta/\text{SE},1)$, $\text{SE}=\sqrt{2\sigma^2/n}$.
2. Power $=P(|Z|>z_{1-\alpha/2})\approx P(Z>z_{1-\alpha/2})=\Phi\left(\delta/\text{SE}-z_{1-\alpha/2}\right)$.
    - Dropped term: rejection in the wrong tail, prob. $<\alpha/2$, ≈ 0 at useful power.
3. Set power $=1-\beta$ → $\delta/\text{SE}-z_{1-\alpha/2}=z_{1-\beta}$.
4. → $\delta\sqrt{n/(2\sigma^2)}=z_{1-\alpha/2}+z_{1-\beta}$ → square → $n$.
```

```{note} Example
:class: dropdown
- Baseline conversion $p_C=0.10$. Detect $\delta=0.01$ absolute. $\alpha=0.05$, power $0.8$.
- $\sigma_C^2+\sigma_T^2=0.10\cdot0.90+0.11\cdot0.89=0.1879$.

$$
n=\frac{0.1879\cdot7.849}{0.01^2}\approx14{,}749\text{ per arm}
$$

- Plug in $2p_C(1-p_C)=0.18$ instead → $\approx14{,}128$ → slightly under-powered.
```

```{attention} Q&A
:class: dropdown
*Why does peeking break the test?*

- Repeated looks w/ fixed-horizon thresholds → [optional stopping](#p-value) → type I rate ≫ $\alpha$.
- Fix: fixed horizon; group-sequential designs (alpha spending); always-valid p-values.

*What is sample ratio mismatch (SRM)?*

- Observed split ≠ designed split beyond chance, e.g., 50.5/49.5 w/ millions of users.
- Detect: [$\chi^2$ goodness-of-fit](#chi-squared-test) on arm counts, w/ a strict threshold.
- Causes: assignment bugs, bot filtering, redirect loss, logging loss → results untrustworthy until fixed.

*What are novelty & primacy effects?*

- Novelty: users try a feature because it is new → early lift decays.
- Primacy: users resist change → early drop recovers.
- Check: plot effect over time; compare new vs. returning users; run longer.

*What is SUTVA & how does interference break it?*

- SUTVA: no interference (a unit's outcome depends only on its own assignment) + no hidden treatment versions.
- Social networks, two-sided marketplaces (shared supply), shared ad budgets → treatment spills over to control → biased $\hat\Delta$.
- Fix: cluster/graph randomization, geo experiments, switchback designs.

*Many metrics or variants?*

- Each one is another test → [multiple comparisons](#multiple-comparisons).
- Fix: one pre-registered primary metric; Bonferroni/BH for the rest.

*How do you test a ratio metric?*

- E.g., CTR $=\sum\text{clicks}/\sum\text{pageviews}$, randomized by user.
- Pageviews of one user are correlated → per-pageview i.i.d. variance → SE too small.
- Fix: user-level numerator $A$ & denominator $B$ → [delta method](limits.md#delta-method):

    $$
    \text{Var}\left(\frac{\bar{A}}{\bar{B}}\right)\approx\frac{1}{n\mu_B^2}\left(\sigma_A^2-2\frac{\mu_A}{\mu_B}\sigma_{AB}+\frac{\mu_A^2}{\mu_B^2}\sigma_B^2\right)
    $$

    - $\mu_A$: Mean of per-user $A$.
    - $\mu_B$: Mean of per-user $B$.
    - $\sigma_A^2$: Variance of per-user $A$.
    - $\sigma_B^2$: Variance of per-user $B$.
    - $\sigma_{AB}$: Covariance of per-user $A$ & $B$.
- Alternative: [bootstrap](#bootstrap) over users.

*Heavy-tailed metric (revenue)?*

- CLT converges slowly → winsorize/cap, or reduce variance w/ [CUPED](#cuped).
```

&nbsp;

#### CUPED
- **Name**: Controlled-experiment Using Pre-Experiment Data {cite:p}`deng2013improving`
- **What**: Subtract from the metric the part that a pre-experiment covariate predicts.
- **Why**: Metric variance high → weak power → long or large experiments.
    - *Why does it work?* Much of the variance = persistent unit differences, visible before the experiment (heavy users stay heavy) → remove it → noise⬇️.
    - *Why is it unbiased?* $X$ is measured before assignment → treatment cannot change it → randomization → $E[\bar{X}_T-\bar{X}_C]=0$.
- **How**:
    1. Pick covariate $X$: usually the same metric in a pre-period.
    2. Estimate $\theta=\text{Cov}(X,Y)/\text{Var}(X)$ on pooled data (both arms).
    3. Adjust each unit: $Y_\text{cv}=Y-\theta(X-\bar{X})$, $\bar{X}$ = pooled mean.
    4. Run the usual two-sample test on $Y_\text{cv}$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $Y$: Per-unit metric during the experiment.
    - $X$: Per-unit pre-experiment covariate.
- Misc:
    - $\theta$: Adjustment coeff (overrides page-level $\theta$ = model param).
    - $\rho=\text{Corr}(X,Y)$.
    - $\bar{Y}_T$: Treatment-arm mean of $Y$.
    - $\bar{Y}_C$: Control-arm mean of $Y$.
    - $\bar{X}_T$: Treatment-arm mean of $X$.
    - $\bar{X}_C$: Control-arm mean of $X$.

One arm:

$$
\bar{Y}_\text{cv}=\bar{Y}-\theta\left(\bar{X}-E[X]\right)
$$

- $E[\bar{Y}_\text{cv}]=E[\bar{Y}]$ for any fixed $\theta$.

Optimal $\theta$ & variance:

$$
\theta^*=\frac{\text{Cov}(X,Y)}{\text{Var}(X)},\qquad\text{Var}\left(\bar{Y}_\text{cv}\right)\big|_{\theta^*}=(1-\rho^2)\,\text{Var}(\bar{Y})
$$

Treatment effect, same $\theta$ in both arms → $E[X]$ cancels:

$$
\hat\Delta_\text{cv}=\left(\bar{Y}_T-\bar{Y}_C\right)-\theta\left(\bar{X}_T-\bar{X}_C\right)
$$

- Estimated $\theta$ → bias $O(1/n)$ ← product of two $O_p(n^{-1/2})$ terms; negligible at A/B scale.
```

```{tip} Derivation
:class: dropdown
*Why $\theta^*=\text{Cov}/\text{Var}$ & factor $1-\rho^2$?*

1. Variance as a function of $\theta$:

    $$
    \text{Var}(\bar{Y}-\theta\bar{X})=\text{Var}(\bar{Y})-2\theta\,\text{Cov}(\bar{X},\bar{Y})+\theta^2\text{Var}(\bar{X})
    $$

2. Quadratic in $\theta$ → minimizer $\theta^*=\text{Cov}(\bar{X},\bar{Y})/\text{Var}(\bar{X})=\text{Cov}(X,Y)/\text{Var}(X)$ ← both terms scale by $1/n$.
3. Plug in:

    $$
    \text{Var}(\bar{Y})-\frac{\text{Cov}(\bar{X},\bar{Y})^2}{\text{Var}(\bar{X})}=\text{Var}(\bar{Y})\left(1-\rho^2\right)
    $$
```

```{note} Example
:class: dropdown
- $\rho=0.7$ → factor $0.51$ → ≈ half the variance → ≈ half the users for the same power ← $n\propto\sigma^2$.
- $\rho=0.9$ → factor $0.19$ → ≈ 5× fewer users.
- Bing (paper): ≈50% variance reduction w/ 1 week of pre-period data; revenue-per-user $<5\%$ ← low pre/post correlation.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class CUPED:
    def __call__(self, y_t, x_t, y_c, x_c):
        x, y = np.concatenate([x_t, x_c]), np.concatenate([y_t, y_c])
        ## one pooled theta for both arms -> E[X] term cancels in the difference
        theta = np.cov(x, y)[0, 1] / x.var(ddof=1)
        ## remove what each unit's pre-period predicts
        adj_t = y_t - theta * (x_t - x.mean())
        adj_c = y_c - theta * (x_c - x.mean())
        delta = adj_t.mean() - adj_c.mean()
        se = np.sqrt(adj_t.var(ddof=1) / len(adj_t) + adj_c.var(ddof=1) / len(adj_c))
        return delta, se

## Example: Var(X)=9, noise var=1 -> rho^2 = 0.9; true effect 0.5
rng = np.random.default_rng(0)
n = 1000
x_t, x_c = rng.normal(10, 3, n), rng.normal(10, 3, n)
y_t, y_c = x_t + rng.normal(0.5, 1, n), x_c + rng.normal(0.0, 1, n)
raw_se = np.sqrt(y_t.var(ddof=1) / n + y_c.var(ddof=1) / n)
print(CUPED()(y_t, x_t, y_c, x_c), raw_se)  ## delta ≈ 0.5, se ≈ 0.045 vs. raw se ≈ 0.14
```
````

```{attention} Q&A
:class: dropdown
*Why must $X$ be pre-experiment?*

- Post-treatment $X$ can move w/ treatment → $\theta(\bar{X}_T-\bar{X}_C)$ absorbs part of the effect → biased $\hat\Delta_\text{cv}$.

*Does it change what is estimated?*

- No. Same average treatment effect; only the variance drops.

*How does it relate to regression & control variates?*

- ≈ OLS of $Y$ on treatment indicator + $X$ (ANCOVA); asymptotically the same estimator.
- = control variates from Monte Carlo: subtract a zero-mean term correlated w/ the estimator.

*Which covariate?*

- Same metric in the pre-period → usually the highest $\rho$.
- Any variable fixed before assignment is valid.

*When does it help little?*

- Low pre/post correlation, e.g., sparse or volatile metrics.
- Many units w/o pre-period data (new users) → no reduction for them.
```

&nbsp;
