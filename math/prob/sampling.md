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
# Sampling & Approximate Inference
Computing intractable expectations & normalizers, mainly under Bayesian posteriors.
Study notes from {cite:t}`bayes,bayes2`; variational inference follows the review by {cite:t}`blei2017variational`.
Deterministic approximation → Monte Carlo → MCMC → SMC → variational inference.
Models & targets live in [Bayesian](bayes.md); Markov-chain theory lives in [Stochastic Processes](process.md#markov-chains).

Notations:
- $\mathcal{D}$: Observed dataset.
- $\theta$: Unknown parameter, scalar or vector.
- $d$: Dimension of $\theta$ when vector-valued.
- $p(\mathcal{D}\mid\theta)$: Likelihood.
- $\pi(\theta)$: Prior density/mass.
- $\pi(\theta\mid\mathcal{D})$: Posterior density/mass (the target).
- $Z=p(\mathcal{D})$: Marginal likelihood (normalizer of the target).
- $\tilde p(\theta)=p(\mathcal{D}\mid\theta)\pi(\theta)$: Unnormalized posterior.
- $\ell(\theta)=\log\tilde p(\theta)$: Log unnormalized posterior.
- $h(\theta)$: Scalar quantity of interest.
- $S$: Number of simulation draws.
- $\theta^{(s)}$: Draw indexed by $s$.

Parameter vectors use $\theta$ rather than the global bold-vector convention. Integrals become sums for discrete unknowns; $\mathcal{N}(\mu,v)$ uses variance $v$, not standard deviation.

&nbsp;

## Deterministic Approximation

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

## Monte Carlo

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

## Markov Chain Monte Carlo

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

## Sequential Monte Carlo

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

For the [state-space model](bayes.md#state-space-models), without resampling at the current propagation step:

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

## Variational Inference

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
