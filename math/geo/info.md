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
# Information Geometry
Families of probability distributions as Riemannian manifolds: Fisher metric, dual affine structure, divergences, & the optimizers & integration measures built on them.
Study notes from {cite:t}`amari2016information`. General Riemannian tools → [Riemannian Metric](manifold.md#riemannian-metric) & [Geodesic](manifold.md#geodesic).

Notations:
- $\theta\in\Theta\subseteq\mathbb{R}^d$: Param vector = coordinates of one distribution.
- $d$: #params = dimension of the manifold.
- $p_\theta$: Distribution w/ param $\theta$.
- $\ell_\theta(x)=\log p_\theta(x)$: Log-likelihood of outcome $x$.
- $\partial_i=\partial/\partial\theta_i$: Partial derivative along coordinate $i$.
- $G(\theta)$: Fisher information matrix = metric tensor.
- $g_{ij}$: Entry $(i,j)$ of $G$.
- $D_{KL}(p\|q)$: [KL divergence](../info.md#kl-divergence).
- $\mathcal{L}(\theta)$: Loss to minimize.
- $\alpha$: Step size (LR).
- $\psi(\theta)$: Log-partition function of an exponential family.
- $\eta=\nabla\psi(\theta)$: Expectation params.
- $\varphi(\eta)$: Legendre dual of $\psi$ (negative entropy).

- **Overrides**: $x$ = outcome (sample) of $p_\theta$, ❌model input. $i,j$ = param coordinate idx, ❌sample/feature idx. $\eta$ = expectation params, ❌LR.
- **Conventions**: $\log=\ln$ (nats) unless stated. Vectors are columns.

&nbsp;

## Statistical Manifolds

### Statistical Manifold
- **What**: Parametric family $\mathcal{M}=\{p_\theta:\theta\in\Theta\}$ viewed as a smooth manifold w/ params as coordinates.
- **Why**: Param values are arbitrary labels.
    - New param ($\sigma$ vs $\sigma^2$ vs $\log\sigma$) → new Euclidean distances & gradients for the same distributions.
    - Goal: distance, gradient & volume that depend only on the distributions → reparametrization invariance.
- **How**:
    1. Point = one distribution.
    2. Coordinates = params. Reparametrization = change of chart.
    3. Tangent vector = direction of change of the log-density (a score).
    4. Add a metric ([Fisher-Rao](#fisher-rao-metric)) & a pair of dual connections ([Dual Geometry](#dual-geometry)) → geometry.

```{note} Math
:class: dropdown
Definition (regular model):
- $\theta\mapsto p_\theta$ is smooth & injective (identifiable).
- Support of $p_\theta$ does not depend on $\theta$.
- Scores $\partial_1\ell_\theta,\dots,\partial_d\ell_\theta$ are linearly independent → $G(\theta)\succ0$.

Tangent space:

$$
T_\theta\mathcal{M}\cong\operatorname{span}\{\partial_1\ell_\theta,\dots,\partial_d\ell_\theta\},\qquad
\mathbb{E}_{p_\theta}[\partial_i\ell_\theta]=0
$$

- $T_\theta\mathcal{M}$: Tangent space at $p_\theta$.
- Tangent vector $\mathbf{v}\in\mathbb{R}^d$ ↔ zero-mean random variable $\sum_iv_i\partial_i\ell_\theta(x)$.
- Inner product of two tangent vectors = covariance of their scores under $p_\theta$ → [Fisher-Rao metric](#fisher-rao-metric).
```

```{dropdown} Table: Common Statistical Manifolds
| Family | Coordinates | Dim | Geometry under Fisher-Rao |
|:--|:--|:--|:--|
| Bernoulli | $p$ | 1 | Arc of a circle of radius 2 |
| Categorical, $C$ outcomes | $(p_1,\dots,p_{C-1})$ | $C-1$ | Positive orthant of a sphere of radius 2 (curvature $+\frac14$) |
| Gaussian $\mathcal{N}(\mu,\sigma^2)$ | $(\mu,\sigma)$ | 2 | Hyperbolic half-plane (curvature $-\frac12$) |
| Gaussian $\mathcal{N}(\mathbf{0},\Sigma)$ | $\Sigma$ | $\frac{n(n+1)}{2}$ | SPD matrices w/ $\frac12\times$ affine-invariant metric |
| NN classifier $p_\theta(y\mid\mathbf{x})$ | Weights | #weights | Architecture-dependent; continuous redundancies & singular regions can make $G$ degenerate |
```

```{attention} Q&A
:class: dropdown
*Why not use the Euclidean distance between params?*

- $\mathcal{N}(0,0.01^2)$ vs $\mathcal{N}(0,0.02^2)$: easy to tell apart.
- $\mathcal{N}(0,10^2)$ vs $\mathcal{N}(0,10.01^2)$: almost identical.
- Same $\Delta\sigma=0.01$ → param distance ignores distinguishability.
- Fisher-Rao distance at fixed $\mu$: $\sqrt2\,|\log(\sigma_2/\sigma_1)|$ → $0.98$ vs $0.0014$.

*Is a neural net a regular statistical manifold?*

- ❌. Permutation symmetries → many $\theta$ give one $p_\theta$ → globally non-identifiable. $G$ need not be singular.
- Continuous redundancies (e.g., ReLU rescaling) → null directions of $G$.
- NNs can contain singular regions → singular statistical model. Watanabe's singular learning theory studies this case.
- Natural-gradient methods still work locally w/ damping ($G+\lambda I$).

*What does this tell us about intelligence/AC?*

- Learning = curve on $\mathcal{M}$. What changes is the agent's distribution (its world model), not the param vector.
- Param count & coordinates = implementation details. Natural-gradient flow is coordinate-free; plain SGD is not.
```

&nbsp;

### Fisher-Rao Metric
- **What**: Riemannian metric on a statistical manifold = covariance of the score. {cite:p}`rao1945information`
- **Why**: Measure distance by statistical distinguishability, not by param difference.
    - Local KL between neighbors = $\frac12$ squared Fisher length.
    - Invariant under reparametrization & sufficient statistics → [Cencov's Theorem](#cencovs-theorem).
- **How**:
    1. Perturb $\theta$ by $d\theta$.
    2. Measure the change of $\ell_\theta(x)$, averaged over $x\sim p_\theta$.
    3. Large change → easy to distinguish → long distance.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\xi$: Alternative coordinates, $\theta=h(\xi)$.
    - $J=\partial\theta/\partial\xi$: Jacobian of the reparametrization.
    - $\hat\theta$: Unbiased estimator of $\theta$ from $m$ iid samples.

Definition:

$$
g_{ij}(\theta)=\mathbb{E}_{x\sim p_\theta}\left[\partial_i\ell_\theta(x)\,\partial_j\ell_\theta(x)\right]=-\mathbb{E}_{x\sim p_\theta}\left[\partial_i\partial_j\ell_\theta(x)\right]
$$

- Second equality: regular model (differentiation under the integral).

Properties:
- $G\succeq0$ always. $G\succ0$ for a regular model.
- KL expansion (either direction):

    $$
    D_{KL}(p_\theta\|p_{\theta+d\theta})=\tfrac12\,d\theta^\top G(\theta)\,d\theta+O(\|d\theta\|^3)
    $$

- Line element $ds^2=d\theta^\top G\,d\theta$. Fisher-Rao distance = length of the shortest curve ([Geodesic](manifold.md#geodesic)).
- Reparametrization:

    $$
    G_\xi=J^\top G_\theta J
    $$

    - → $d\theta^\top G_\theta d\theta=d\xi^\top G_\xi d\xi$ → lengths are invariant.
- $m$ iid samples → Fisher $=mG$.
- **Cramér–Rao**: $\operatorname{Cov}(\hat\theta)\succeq(mG)^{-1}$.
- Statistic $T(x)$ → $G_T\preceq G$. Equality for all $\theta$ ⇔ $T$ sufficient.

Examples:
- Bernoulli: $g=\dfrac{1}{p(1-p)}$.
- Gaussian $(\mu,\sigma)$: $G=\operatorname{diag}\left(\dfrac{1}{\sigma^2},\dfrac{2}{\sigma^2}\right)$ → $ds^2=\dfrac{d\mu^2+2\,d\sigma^2}{\sigma^2}$.
    - Substitute $\mu=\sqrt2\,u$ → $ds^2=2\,\dfrac{du^2+d\sigma^2}{\sigma^2}$ → Poincaré half-plane scaled by 2 → constant curvature $-\frac12$ ([Hyperbolic Space](manifold.md#hyperbolic-space)).
- Categorical: $ds^2=\sum_k\dfrac{dp_k^2}{p_k}$.
    - Substitute $u_k=2\sqrt{p_k}$ → $ds^2=\sum_kdu_k^2$ w/ $\sum_ku_k^2=4$ → sphere of radius 2.
    - Distance: $d_{FR}(p,q)=2\arccos\sum_k\sqrt{p_kq_k}$.
- Softmax $p=\operatorname{softmax}(\mathbf{z})$, w.r.t. logits $\mathbf{z}$: $G_\mathbf{z}=\operatorname{diag}(p)-pp^\top$ (rank $C-1$; null vector $\mathbf{1}$).
```

```{tip} Derivation
:class: dropdown
*Why does $\mathbb{E}[\text{score}\cdot\text{score}^\top]=-\mathbb{E}[\text{Hessian}]$?*

1. $\int p_\theta\,dx=1$ → $\int\partial_ip_\theta\,dx=0$ → $\mathbb{E}[\partial_i\ell_\theta]=0$ ← $\partial_ip_\theta=p_\theta\partial_i\ell_\theta$.
2. Differentiate $\int p_\theta\,\partial_i\ell_\theta\,dx=0$ by $\theta_j$:

    $$
    \int p_\theta\,\partial_j\ell_\theta\,\partial_i\ell_\theta\,dx+\int p_\theta\,\partial_i\partial_j\ell_\theta\,dx=0
    $$

*Why is $G$ the Hessian of KL?*

1. $f(\boldsymbol\delta)=D_{KL}(p_\theta\|p_{\theta+\boldsymbol\delta})=\mathbb{E}_{p_\theta}[\ell_\theta-\ell_{\theta+\boldsymbol\delta}]$.
2. $f(\mathbf{0})=0$.
3. $\nabla f(\mathbf{0})=-\mathbb{E}[\nabla\ell_\theta]=\mathbf{0}$.
4. $\nabla^2f(\mathbf{0})=-\mathbb{E}[\nabla^2\ell_\theta]=G$ → $f\approx\frac12\boldsymbol\delta^\top G\boldsymbol\delta$.
5. Reverse direction: Hessian at $\boldsymbol\delta=\mathbf{0}$ is also $G$ → same leading term.

*Why $G_\xi=J^\top G_\theta J$?*

1. Chain rule: $\nabla_\xi\ell=J^\top\nabla_\theta\ell$.
2. $G_\xi=\mathbb{E}[J^\top\nabla_\theta\ell\,\nabla_\theta\ell^\top J]=J^\top G_\theta J$.

*Gaussian Fisher?*

1. $\ell=-\log\sigma-\frac{(x-\mu)^2}{2\sigma^2}+\text{const}$. Let $z=(x-\mu)/\sigma\sim\mathcal{N}(0,1)$.
2. $\partial_\mu\ell=z/\sigma$ → $\mathbb{E}[z^2]/\sigma^2=1/\sigma^2$.
3. $\partial_\sigma\ell=(z^2-1)/\sigma$ → $\mathbb{E}[(z^2-1)^2]/\sigma^2=(3-2+1)/\sigma^2=2/\sigma^2$.
4. Cross term $\mathbb{E}[z(z^2-1)]/\sigma^2=0$ ← odd moments vanish.
```

````{important} Code
:class: dropdown
```python
import torch

class CategoricalFisher:
    ## Fisher of p = softmax(z) w.r.t. the logits z, one matrix per row
    def __call__(self, logits):
        p = logits.softmax(-1)                                        ## [m, C]
        ## E_{y~p}[(e_y - p)(e_y - p)^T] = diag(p) - p p^T  (y drawn from the MODEL)
        return torch.diag_embed(p) - p[:, :, None] * p[:, None, :]   ## [m, C, C]

## Example
F = CategoricalFisher()(torch.zeros(1, 3))  ## uniform p = 1/3
print(F[0])                    ## diagonal 2/9 = 0.2222, off-diagonal -1/9 = -0.1111
print(F[0] @ torch.ones(3))    ## zeros: adding a constant to all logits leaves p unchanged
```
````

```{attention} Q&A
:class: dropdown
*Fisher vs Hessian of the loss?*

- $G$ = expected Hessian of the NLL w/ targets drawn from the model.
- Observed Hessian (targets from data) can be indefinite. $G$ is always PSD.
- NLL w/ an exponential-family output in natural params → $G$ = generalized Gauss-Newton matrix ([Gauss-Newton](../../ml/optim.md#gauss-newton)).

*Fisher-Rao vs Wasserstein?*

- Fisher-Rao compares densities point by point ("vertical") → ignores the geometry of the sample space.
- Disjoint supports → categorical Fisher-Rao distance saturates at $\pi$; KL = $\infty$.
- [Wasserstein](ot.md#wasserstein-distance) moves mass across the sample space ("horizontal") → finite & grows w/ the shift.
- Fisher-Rao: invariant under relabeling outcomes. Wasserstein: needs a ground metric, by design.

*Why is the Gaussian family hyperbolic?*

- $\sigma\to0$ → a tiny shift in $\mu$ is easy to detect → metric $\propto1/\sigma^2$ blows up.
- → $\sigma=0$ is an infinitely distant boundary, like the edge of the Poincaré half-plane.

*Where does Fisher appear in ML?*

- [Natural gradient](#natural-gradient), [K-FAC](#k-fac), [natural policy gradient](../../rl/policy.md#natural-policy-gradient), [TRPO](../../rl/policy.md#trpo), [NES](../../ml/optim.md#nes).
- [Jeffreys prior](../prob/bayes.md#jeffreys-prior) $\propto\sqrt{\det G}$ = Fisher-Rao volume.
- [MLE](../../ml/obj.md#mle) asymptotics: $\operatorname{Cov}(\hat\theta)\to(mG)^{-1}$.
- EWC (continual learning): diagonal Fisher = param importance.

*What does this tell us about intelligence/AC?*

- $G$ = sensitivity of the distribution to its params.
- Gibbs family $p_\theta\propto\exp(\theta^\top\mathbf{t}(x))$ → $G=\operatorname{Cov}(\mathbf{t})$ = susceptibility → peaks at 2nd-order phase transitions.
- Hypothesis (contested): brain-criticality → cortex operates near a critical point → near-max Fisher information about its control params.
```

&nbsp;

#### Cencov's Theorem
- **What**: Fisher metric = unique Riemannian metric, up to scale, invariant under Markov morphisms. {cite:p}`cencov1982statistical`
- **Why**: Show Fisher is forced, not chosen.
    - Demand: a statistically reversible re-encoding of outcomes must not change distances.
    - → Only $c\cdot G$ survives.
- **How**:
    1. Markov morphism: split each coarse outcome into fine outcomes w/ fixed, $\theta$-independent probabilities.
    2. Coarse outcome = sufficient statistic of the fine one → no information is lost.
    3. Require lengths to match on both simplices, for every such split → Fisher metric up to a constant.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\Delta_n$: Interior of the simplex of distributions on $n$ outcomes.
    - $B_1,\dots,B_n$: Partition of $\{1,\dots,N\}$ into blocks, $N\geq n$.
    - $Q_{ka}>0$ for $a\in B_k$: Split weights, $\sum_{a\in B_k}Q_{ka}=1$.
    - $f:\Delta_n\to\Delta_N$: Markov morphism, $(f(p))_a=p_kQ_{ka}$ for $a\in B_k$.
    - $f_*$: Pushforward of tangent vectors by $f$.
    - $g^{(n)}$: Riemannian metric on $\Delta_n$.

Statement:

$$
g^{(N)}(f_*\mathbf{u},f_*\mathbf{v})=g^{(n)}(\mathbf{u},\mathbf{v})\ \ \forall f,\mathbf{u},\mathbf{v}
\;\Longrightarrow\;
g^{(n)}=c\cdot g^{\text{Fisher}},\quad c>0
$$

- $\mathbf{u},\mathbf{v}$: Tangent vectors at a point of $\Delta_n$.
- $c$: One constant for all $n$.
- Čencov also: invariant affine connections = 1-param family ($\alpha$-connections; e & m are $\alpha=\pm1$).
- Extensions beyond finite sample spaces: Ay–Jost–Lê–Schwachhöfer (2015); Bauer–Bruveris–Michor (2016, smooth densities).
```

```{attention} Q&A
:class: dropdown
*What breaks w/ another metric?*

- Distances would depend on the encoding of outcomes.
- E.g., split category "cat" into "cat-A"/"cat-B" by a fair coin → no new information, but a non-invariant metric reports new distances.

*Does it make Fisher-Rao the only useful geometry?*

- ❌. The theorem assumes invariance under Markov morphisms.
- [Wasserstein](ot.md#wasserstein-distance) drops that assumption on purpose: it uses the sample-space metric.

*How does this relate to divergences?*

- Invariant divergences ([f-divergences](#f-divergence)) all induce $c\cdot G$ locally → same local geometry.
```

&nbsp;

## Dual Geometry
- **What**: Fisher metric + 2 affine connections (e & m) that are dual w.r.t. it → 2 kinds of straight line.

### f-Divergence
- **What**: $Q$-average of a convex function $f$ of the likelihood ratio $p/q$. {cite:p}`csiszar1967information`
- **Why**: Unify KL, reverse KL, $\chi^2$, Hellinger, TV & JS in one family.
    - Every member obeys the data-processing inequality → invariant under sufficient statistics.
    - Every smooth member w/ $f''(1)>0$ induces a positive multiple of the Fisher metric locally.
- **How**:
    1. Compute the likelihood ratio $p(x)/q(x)$ per outcome.
    2. Apply convex $f$ w/ $f(1)=0$ → 0 where the distributions agree.
    3. Average under $q$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $f:(0,\infty)\to\mathbb{R}$: Convex generator, $f(1)=0$.
    - $K$: Markov kernel (channel).

Definition:

$$
D_f(p\|q)=\sum_xq(x)\,f\!\left(\frac{p(x)}{q(x)}\right)
$$

Properties:
- $D_f\geq0$ ← Jensen. $=0$ iff $p=q$ if $f$ is strictly convex at 1.
- Jointly convex in $(p,q)$.
- Data processing: $D_f(pK\|qK)\leq D_f(p\|q)$.
- $f(t)+c(t-1)$ → same $D_f$ → $f$ is fixed only up to a linear term.
- Dual generator $\tilde f(t)=tf(1/t)$ → $D_{\tilde f}(p\|q)=D_f(q\|p)$.
- Local form for a smooth regular model, w/ $f\in C^3$ near 1:

    $$
    D_f(p_\theta\|p_{\theta+d\theta})=\frac{f''(1)}{2}\,d\theta^\top G\,d\theta+O(\|d\theta\|^3)
    $$
```

```{tip} Derivation
:class: dropdown
*Why does every smooth $f$-divergence induce the Fisher metric?*

1. Let $p=p_\theta$, $q=p_{\theta+d\theta}$, $\varepsilon(x)=p(x)/q(x)-1$.
2. $\sum_xq\varepsilon=\sum_x(p-q)=0$.
3. Taylor: $f(1+\varepsilon)=f'(1)\varepsilon+\frac12f''(1)\varepsilon^2+O(\varepsilon^3)$.
4. Average under $q$: linear term vanishes ← step 2 → $D_f\approx\frac{f''(1)}{2}\sum_x\frac{(p-q)^2}{q}$.
5. $p-q\approx-\sum_i\partial_ip\,d\theta_i$ & $\partial_ip=p\,\partial_i\ell$ → $\sum_x\frac{(p-q)^2}{q}\approx d\theta^\top G\,d\theta$.
```

````{important} Code
:class: dropdown
```python
import torch

class FDivergence:
    GENERATORS = {
        "kl":         lambda t: t * t.log(),
        "reverse_kl": lambda t: -t.log(),
        "hellinger":  lambda t: (t.sqrt() - 1) ** 2,
        "chi2":       lambda t: (t - 1) ** 2,
        "tv":         lambda t: 0.5 * (t - 1).abs(),
    }

    def __init__(self, name):
        self.f = self.GENERATORS[name]

    def __call__(self, p, q):
        ## average a convex function of the likelihood ratio under q
        return (q * self.f(p / q)).sum(-1)

## Example
p, q = torch.tensor([0.2, 0.8]), torch.tensor([0.5, 0.5])
for name in FDivergence.GENERATORS:
    print(name, round(FDivergence(name)(p, q).item(), 4))
## kl 0.1927, reverse_kl 0.2231, hellinger 0.1026, chi2 0.36, tv 0.3
```
````

```{dropdown} Table: Common f-Divergences
| Divergence | $f(t)$ | $f''(1)$ |
|:--|:--|:--|
| KL $D_{KL}(p\Vert q)$ | $t\log t$ | 1 |
| Reverse KL $D_{KL}(q\Vert p)$ | $-\log t$ | 1 |
| Squared Hellinger $\sum(\sqrt p-\sqrt q)^2$ | $(\sqrt t-1)^2$ | $\frac12$ |
| Pearson $\chi^2$ | $(t-1)^2$ | 2 |
| Total variation | $\frac12\lvert t-1\rvert$ | undefined (kink) |
| Jensen–Shannon | $\frac12\left[t\log\frac{2t}{1+t}+\log\frac{2}{1+t}\right]$ | $\frac14$ |
```

```{attention} Q&A
:class: dropdown
*Why does TV have no Fisher local form?*

- $f(t)=\frac12|t-1|$ has a kink at 1 → TV $\approx\frac12\sum_x|\partial_\theta p\cdot d\theta|$ = first order in $d\theta$, ❌quadratic.

*How do $f$-divergences power GANs?*

- Variational form: $D_f(p\|q)=\sup_T\ \mathbb{E}_p[T(x)]-\mathbb{E}_q[f^\star(T(x))]$.
    - $f^\star$: Convex conjugate of $f$.
    - $T$: Critic function.
- Critic = NN → lower bound → f-GAN. Original GAN ↔ JS.

*Why are $f$-divergences "the" invariant divergences?*

- Divergences that are sums over outcomes & obey data processing = $f$-divergences.
- Holds for $\geq3$ outcomes. Binary alphabets admit exceptions.

*Which divergence is both $f$ & [Bregman](#bregman-divergence)?*

- In probability coordinates: only positive multiples of forward KL (Jiao et al., 2014).
- Reverse KL is Bregman in dual coordinates, ❌in probability coordinates.
- On positive measures, Amari (2009) permits nonlinear $\alpha$-representations → the $\alpha$-divergence family.
- → KL = invariant & dually flat at the same time → central role in information geometry.
```

&nbsp;

### Bregman Divergence
- **What**: Gap between a strictly convex function & its tangent plane. {cite:p}`bregman1967relaxation`
- **Why**: Generalize squared Euclidean distance to non-Euclidean geometries.
    - Keep its algebra: mean = best constant predictor, Pythagorean theorem, projections.
    - Every exponential family ↔ one Bregman divergence → MLE = Bregman loss minimization.
- **How**:
    1. Pick a strictly convex generator $F$.
    2. Linearize $F$ at $\mathbf{v}$.
    3. $D_F(\mathbf{u},\mathbf{v})$ = height of $F(\mathbf{u})$ above that tangent plane.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $F$: Strictly convex, differentiable generator.
    - $F^*$: Convex conjugate of $F$.
    - $\mathbf{u},\mathbf{v},\mathbf{w}$: Points in the domain of $F$.

Definition:

$$
D_F(\mathbf{u},\mathbf{v})=F(\mathbf{u})-F(\mathbf{v})-\langle\nabla F(\mathbf{v}),\mathbf{u}-\mathbf{v}\rangle
$$

Examples:
- $F(\mathbf{u})=\|\mathbf{u}\|^2$ → $\|\mathbf{u}-\mathbf{v}\|^2$.
- $F(\mathbf{u})=\sum_ju_j\log u_j$ (negative entropy) → $\sum_j\left(u_j\log\frac{u_j}{v_j}-u_j+v_j\right)$ → KL on the simplex.
- $F(\mathbf{u})=-\sum_j\log u_j$ (Burg) → Itakura–Saito: $\sum_j\left(\frac{u_j}{v_j}-\log\frac{u_j}{v_j}-1\right)$.

Properties:
- $D_F\geq0$. $=0$ iff $\mathbf{u}=\mathbf{v}$.
- Convex in $\mathbf{u}$. Not convex in $\mathbf{v}$ in general.
- ❌Symmetric. ❌Triangle inequality.
- Linear in $F$: $D_{aF_1+bF_2}=aD_{F_1}+bD_{F_2}$ for $a,b\geq0$.
- Local form ($F\in C^2$): $D_F(\mathbf{v}+d\mathbf{v},\mathbf{v})\approx\frac12d\mathbf{v}^\top\nabla^2F(\mathbf{v})\,d\mathbf{v}$.
    - Riemannian (Hessian) metric where $\nabla^2F\succ0$.
- Duality: $D_F(\mathbf{u},\mathbf{v})=D_{F^*}(\nabla F(\mathbf{v}),\nabla F(\mathbf{u}))$.
- Three-point identity:

    $$
    D_F(\mathbf{u},\mathbf{w})=D_F(\mathbf{u},\mathbf{v})+D_F(\mathbf{v},\mathbf{w})-\langle\nabla F(\mathbf{w})-\nabla F(\mathbf{v}),\mathbf{u}-\mathbf{v}\rangle
    $$

    - Inner product $=0$ → Pythagorean equality.
- **Mean minimizer**: $X$ random, $\boldsymbol\mu=\mathbb{E}[X]$ → $\arg\min_\mathbf{s}\mathbb{E}[D_F(X,\mathbf{s})]=\boldsymbol\mu$, unique.
```

```{tip} Derivation
:class: dropdown
*Why is the mean the unique minimizer of $\mathbb{E}[D_F(X,\mathbf{s})]$?*

1. $\mathbb{E}[D_F(X,\mathbf{s})]=\mathbb{E}[F(X)]-F(\mathbf{s})-\langle\nabla F(\mathbf{s}),\boldsymbol\mu-\mathbf{s}\rangle$.
2. $\mathbb{E}[D_F(X,\boldsymbol\mu)]=\mathbb{E}[F(X)]-F(\boldsymbol\mu)$.
3. Subtract: $\mathbb{E}[D_F(X,\mathbf{s})]-\mathbb{E}[D_F(X,\boldsymbol\mu)]=D_F(\boldsymbol\mu,\mathbf{s})\geq0$, $=0$ iff $\mathbf{s}=\boldsymbol\mu$.
4. Argument order matters: the mean minimizes over the **second** argument.
```

````{important} Code
:class: dropdown
```python
import torch

class BregmanDivergence:
    GENERATORS = {
        "sq_euclid":   lambda u: (u ** 2).sum(-1),
        "neg_entropy": lambda u: (u * u.log()).sum(-1),
        "burg":        lambda u: -u.log().sum(-1),
    }

    def __init__(self, name):
        self.F = self.GENERATORS[name]

    def __call__(self, u, v):
        v = v.detach().requires_grad_()
        Fv = self.F(v)
        ## tangent plane of F at v: slope = grad F(v), via autograd
        grad_v, = torch.autograd.grad(Fv.sum(), v)
        return self.F(u) - Fv.detach() - (grad_v * (u - v.detach())).sum(-1)

## Example
u, v = torch.tensor([0.2, 0.8]), torch.tensor([0.5, 0.5])
for name in BregmanDivergence.GENERATORS:
    print(name, round(BregmanDivergence(name)(u, v).item(), 4))
## sq_euclid 0.18, neg_entropy 0.1927 (= KL), burg 0.4463 (= Itakura-Saito)
print(round(BregmanDivergence("neg_entropy")(v, u).item(), 4))  ## 0.2231 -> asymmetric
```
````

```{attention} Q&A
:class: dropdown
*Is a Bregman divergence a metric?*

- ❌. Asymmetric & no triangle inequality.
- Only squared Mahalanobis distances ($F$ quadratic) are symmetric.

*Why does k-means work w/ any Bregman divergence?*

- Assign step: nearest center in $D_F(\mathbf{x},\mathbf{c})$.
- Update step: best center = cluster mean ← mean-minimizer property.
- → Same algorithm for squared error, KL, Itakura–Saito (Banerjee et al., 2005).
- Converse (Banerjee, Guo & Wang, 2005), under the theorem's regularity conditions:
    - Loss w/ the mean as unique optimum = Bregman divergence + a term that depends only on the observation.
    - Normalize the loss to 0 on the diagonal → exactly Bregman.

*How does it relate to exponential families & losses?*

- $\log p_\theta(x)=-D_\varphi(\mathbf{t}(x),\eta)+\text{terms free of }\theta$ → see [Dually Flat Manifold](#dually-flat-manifold).
- Gaussian → squared error. Poisson → $y\log\frac{y}{\lambda}-y+\lambda$. Bernoulli → binary cross-entropy.

*How does it relate to optimization?*

- Replace $\frac12\|\mathbf{w}-\mathbf{w}_t\|^2$ in the proximal step by $D_F$ → [Mirror Descent](#mirror-descent).
- Proximal point w/ KL → entropic & policy-mirror-descent updates.
```

&nbsp;

### Dually Flat Manifold
- **What**: Manifold w/ 2 affine coordinate systems $\theta$ & $\eta$ linked by a Legendre transform. {cite:p}`amari2016information`
- **Why**: Exponential families have 2 natural coordinate systems: natural params & means.
    - Make both "straight" → projections & a Pythagorean theorem hold exactly.
    - KL becomes a Bregman divergence → convex algebra replaces differential geometry.
- **How**:
    1. Convex potential $\psi(\theta)$ → $\theta$ = flat coordinates of the e-connection (e-flat).
    2. $\eta=\nabla\psi(\theta)$ → flat coordinates of the m-connection (m-flat).
    3. Metric = $\nabla^2\psi$ = Fisher.
    4. Divergence = Bregman divergence of $\psi$ = KL.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{t}(x)$: Sufficient statistic.
    - $h(x)$: Base measure.

Model (exponential family):

$$
p_\theta(x)=h(x)\exp\!\left(\theta^\top\mathbf{t}(x)-\psi(\theta)\right),\qquad
\psi(\theta)=\log\int h(x)\,e^{\theta^\top\mathbf{t}(x)}dx
$$

Dual coordinates:

$$
\begin{align*}
\eta&=\nabla\psi(\theta)=\mathbb{E}_\theta[\mathbf{t}(x)]\\
G(\theta)&=\nabla^2\psi(\theta)=\operatorname{Cov}_\theta[\mathbf{t}(x)]\\
\varphi(\eta)&=\sup_{\theta'}\left\{\theta'^\top\eta-\psi(\theta')\right\}=\theta^\top\eta-\psi(\theta)=\mathbb{E}_\theta\left[\log p_\theta(x)-\log h(x)\right]\\
\theta&=\nabla\varphi(\eta),\qquad G_\eta=\nabla^2\varphi(\eta)=G(\theta)^{-1}
\end{align*}
$$

KL = canonical (Bregman) divergence:

$$
D_{KL}(p_{\theta_1}\|p_{\theta_2})=D_\psi(\theta_2,\theta_1)=D_\varphi(\eta_1,\eta_2)=\psi(\theta_2)+\varphi(\eta_1)-\theta_2^\top\eta_1
$$

- Argument order flips between $\theta$ & $\eta$ coordinates.

Properties:
- **e-flat** submanifold: affine in $\theta$ (e.g., $\theta=\theta_0+B\mathbf{u}$ → exponential subfamily).
- **m-flat** submanifold: affine in $\eta$ (in the full simplex: mixture families & linear families $\{q:\mathbb{E}_q[\mathbf{f}]=\mathbf{c}\}$).
- e & m connections: flat, dual w.r.t. $G$. Average of the two = Levi-Civita connection of $G$.
- Flat ≠ zero Riemannian curvature: Gaussian family is dually flat, yet hyperbolic (curvature $-\frac12$) under Levi-Civita.
```

```{tip} Derivation
:class: dropdown
*Why is KL a Bregman divergence of $\psi$?*

1. $\log\frac{p_{\theta_1}}{p_{\theta_2}}=(\theta_1-\theta_2)^\top\mathbf{t}(x)-\psi(\theta_1)+\psi(\theta_2)$ ← $h$ cancels.
2. Take $\mathbb{E}_{\theta_1}$, use $\mathbb{E}_{\theta_1}[\mathbf{t}]=\eta_1=\nabla\psi(\theta_1)$:

    $$
    D_{KL}=\psi(\theta_2)-\psi(\theta_1)-\nabla\psi(\theta_1)^\top(\theta_2-\theta_1)=D_\psi(\theta_2,\theta_1)
    $$

3. Substitute $\psi(\theta_1)=\theta_1^\top\eta_1-\varphi(\eta_1)$ → $\psi(\theta_2)+\varphi(\eta_1)-\theta_2^\top\eta_1$.

*Why is $\log p_\theta(x)$ a negative Bregman loss?*

1. $D_\varphi(\mathbf{t},\eta)=\varphi(\mathbf{t})-\varphi(\eta)-\theta^\top(\mathbf{t}-\eta)=\varphi(\mathbf{t})+\psi(\theta)-\theta^\top\mathbf{t}$.
2. → $\log p_\theta(x)=-D_\varphi(\mathbf{t}(x),\eta)+\varphi(\mathbf{t}(x))+\log h(x)$. Last 2 terms are free of $\theta$.
```

```{note} Example
:class: dropdown
Bernoulli, $t(x)=x\in\{0,1\}$:
- $\theta=\log\frac{p}{1-p}$ (logit).
- $\psi(\theta)=\log(1+e^\theta)$ (softplus).
- $\eta=\sigma(\theta)=p$.
- $G_\theta=p(1-p)$. $G_\eta=\frac{1}{p(1-p)}$ → inverses ✓.
- $\varphi(\eta)=\eta\log\eta+(1-\eta)\log(1-\eta)$ = negative entropy.
- $D_\varphi(y,p)$ for $y\in\{0,1\}$ = binary cross-entropy.
```

```{dropdown} Table: Exponential Families as Bregman Losses
| Family | $\theta$ | $\psi(\theta)$ | $\eta$ | $D_\varphi(y,\eta)$ |
|:--|:--|:--|:--|:--|
| Gaussian, $\sigma=1$ | $\mu$ | $\theta^2/2$ | $\mu$ | $\frac12(y-\mu)^2$ |
| Bernoulli | logit | $\log(1+e^\theta)$ | $p$ | BCE (for $y\in\{0,1\}$) |
| Poisson | $\log\lambda$ | $e^\theta$ | $\lambda$ | $y\log\frac{y}{\lambda}-y+\lambda$ |
| Categorical | $\log\frac{p_k}{p_C}$, $k<C$ | $\log\left(1+\sum_{k<C}e^{\theta_k}\right)$ | $p_{1:C-1}$ | CE minus entropy of $y$ |
```

```{attention} Q&A
:class: dropdown
*Why "dual"?*

- Two connections $\nabla$, $\nabla^*$ satisfy $\partial_kg_{ij}=\langle\nabla_{\partial_k}\partial_i,\partial_j\rangle+\langle\partial_i,\nabla^*_{\partial_k}\partial_j\rangle$.
- Neither alone preserves $G$. Together they do.
- Dually flat: both flat → $\theta$ & $\eta$ are global affine coordinates.

*How does this relate to GLMs & output layers?*

- [GLM](../../ml/supervised.md#glm) w/ canonical link: linear predictor = $\theta$; mean = $\eta=\nabla\psi(\theta)$.
- Softmax/sigmoid/identity output + matching loss = $\eta$-map + Bregman loss.
- → Gradient w.r.t. $\theta$ = $\eta-\mathbf{t}(y)$ = "prediction − target" for all canonical pairs.

*What does $\varphi$ mean?*

- Negative entropy (relative to $h$) as a function of the means.
- Max-entropy dual: $\max_q-\int q\log\frac{q}{h}$ s.t. $\mathbb{E}_q[\mathbf{t}]=\eta$ → exponential-family member w/ that $\eta$.
    - Constant $h$ → plain Shannon max-entropy.
```

&nbsp;

#### e- and m-Geodesics
- **What**: Straight lines in $\theta$ (e-geodesic) & in $\eta$ (m-geodesic).
- **Why**: Two ways to interpolate distributions → each [projection](#information-projection) runs along one of them.
- **How**:
    - **e-geodesic**: interpolate log-densities, then renormalize.
    - **m-geodesic**: interpolate densities (mixture).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $t\in[0,1]$: Curve param.
    - $c(t)$: Log-normalizer of the e-geodesic.

On the full space of positive distributions:

$$
\log p^{(e)}_t=(1-t)\log p+t\log q-c(t),\qquad
p^{(m)}_t=(1-t)\,p+t\,q
$$

Properties:
- Inside an exponential family: e-geodesic = $\theta$-affine → stays in the family.
- m-geodesic of the family = $\eta$-affine. A mixture of 2 members leaves the family in general.
- **Projection theorem**:
    - m-projection onto an e-flat submanifold → unique, if attained.
    - e-projection onto an m-flat submanifold → unique, if attained.
- **Pythagorean theorem**: m-geodesic $r\to q$ ⊥ e-geodesic $q\to p$ at $q$ (in $G$) →

    $$
    D_{KL}(r\|p)=D_{KL}(r\|q)+D_{KL}(q\|p)
    $$
```

```{note} Example
:class: dropdown
$p=\mathcal{N}(-2,1)$, $q=\mathcal{N}(2,1)$, $t=\frac12$:
- m-geodesic: $\frac12p+\frac12q$ → bimodal, ❌Gaussian.
- e-geodesic: $\sqrt{pq}\propto\exp\left(-\frac{(x+2)^2+(x-2)^2}{4}\right)\propto e^{-x^2/2}$ → $\mathcal{N}(0,1)$.
```

```{attention} Q&A
:class: dropdown
*Is either one the Fisher-Rao geodesic?*

- ❌ in general. Fisher-Rao geodesic = Levi-Civita geodesic = $\alpha=0$, halfway between e ($\alpha=1$) & m ($\alpha=-1$).

*Where do the two interpolations show up in ML?*

- m: mixture models, data mixing, ensembles that average probabilities.
- e: product of experts, ensembles that average logits, annealed importance sampling ($p_0^{1-t}p_1^t$).
```

&nbsp;

### Information Projection
- **What**: Closest member of a model family to a given distribution, in KL. {cite:p}`csiszar1975divergence`
    - **I-projection**: $\arg\min_{q\in\mathcal{C}}D_{KL}(q\|p)$, $\mathcal{C}$ convex (often linear).
    - **M-projection** (reverse I-projection): $\arg\min_{q\in\mathcal{E}}D_{KL}(p\|q)$, $\mathcal{E}$ exponential family.
- **Why**: Many inference tasks = "nearest distribution that satisfies constraints".
    - Max-entropy, MLE, EM, iterative scaling & variational inference are projections.
    - Pythagorean theorem → projections compose & error splits exactly.
- **How**:
    - I-projection onto $\{q:\mathbb{E}_q[\mathbf{f}]=\mathbf{c}\}$: tilt $p$ exponentially, $q^*\propto p\,e^{\boldsymbol\lambda^\top\mathbf{f}}$. Tune $\boldsymbol\lambda$ to meet the constraints.
    - M-projection onto $\mathcal{E}$: match moments, $\mathbb{E}_{q^*}[\mathbf{t}]=\mathbb{E}_p[\mathbf{t}]$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $p$: Reference distribution.
    - $\mathbf{f}(x)$: Constraint features.
    - $\mathbf{c}$: Target moments.
    - $\mathcal{C}_\text{lin}=\{q:\mathbb{E}_q[\mathbf{f}]=\mathbf{c}\}$: Linear family (m-flat).
    - $\mathcal{E}$: Exponential family w/ sufficient statistic $\mathbf{t}$ (e-flat).
    - $\boldsymbol\lambda$: Lagrange multipliers.

I-projection onto a linear family (finite alphabet, $p>0$, $\mathcal{C}_\text{lin}$ contains a strictly positive distribution):

$$
q^*(x)=\frac{p(x)\,e^{\boldsymbol\lambda^\top\mathbf{f}(x)}}{Z(\boldsymbol\lambda)},\qquad
D_{KL}(r\|p)=D_{KL}(r\|q^*)+D_{KL}(q^*\|p)\quad\forall r\in\mathcal{C}_\text{lin}
$$

- $Z(\boldsymbol\lambda)$: Normalizer.
- General convex $\mathcal{C}$: "$\geq$" instead of "$=$".

M-projection onto a regular minimal exponential family ($\mathbb{E}_p[\mathbf{t}]$ inside its mean-param domain):

$$
\mathbb{E}_{q^*}[\mathbf{t}]=\mathbb{E}_p[\mathbf{t}],\qquad
D_{KL}(p\|q)=D_{KL}(p\|q^*)+D_{KL}(q^*\|q)\quad\forall q\in\mathcal{E}
$$

- Boundary solutions may require the closure of the family & divergent natural params.

Special cases:
- **Max-entropy**: $p$ = uniform → $D_{KL}(q\|p)=\log|\mathcal{X}|-H(q)$ → I-projection = max-entropy distribution under the constraints.
    - $|\mathcal{X}|$: #outcomes.
- **MLE**: $p$ = empirical distribution → M-projection onto $\mathcal{E}$ = MLE.
- **Variational inference**: $\arg\min_{q\in\mathcal{Q}}D_{KL}(q\|p(\cdot\mid\text{data}))$ = I-projection of the posterior onto $\mathcal{Q}$.
```

```{tip} Derivation
:class: dropdown
*Why the Pythagorean equality for a linear family?*

1. $\log\frac{q^*}{p}=\boldsymbol\lambda^\top\mathbf{f}-\log Z$.
2. $D_{KL}(r\|p)-D_{KL}(r\|q^*)=\mathbb{E}_r\left[\log\frac{q^*}{p}\right]=\boldsymbol\lambda^\top\mathbf{c}-\log Z$ ← $r\in\mathcal{C}_\text{lin}$.
3. Same value for $r=q^*$ ($q^*\in\mathcal{C}_\text{lin}$) → it equals $D_{KL}(q^*\|p)$.

*Why the Pythagorean equality for an exponential family?*

1. $q=h\,e^{\theta^\top\mathbf{t}-\psi(\theta)}$, $q^*$ w/ $\theta^*$.
2. $D_{KL}(p\|q)-D_{KL}(p\|q^*)=\mathbb{E}_p\left[\log\frac{q^*}{q}\right]=(\theta^*-\theta)^\top\mathbb{E}_p[\mathbf{t}]-\psi(\theta^*)+\psi(\theta)$.
3. Moment matching → $\mathbb{E}_p[\mathbf{t}]=\mathbb{E}_{q^*}[\mathbf{t}]$ → step 2 = $\mathbb{E}_{q^*}\left[\log\frac{q^*}{q}\right]=D_{KL}(q^*\|q)$.
```

```{note} Example
:class: dropdown
Jaynes's dice: die w/ faces $1..6$, observed mean $4.5$ (fair die: $3.5$).
- I-project uniform $p$ onto $\{q:\mathbb{E}_q[x]=4.5\}$ → $q^*(x)\propto e^{\lambda x}$, $\lambda\approx0.3710$.
- $q^*\approx(0.0544,0.0788,0.1142,0.1654,0.2398,0.3475)$.
- = least-committal die that matches the mean.
```

````{important} Code
:class: dropdown
```python
import torch

class IProjection:
    ## I-projection of p onto the linear family {q : E_q[f] = c}, scalar feature f
    def __init__(self, f, c):
        self.f, self.c = f, c

    def tilt(self, p, lam):
        w = p * torch.exp(lam * self.f)  ## exponential tilt of p
        return w / w.sum()

    def __call__(self, p, lo=-20.0, hi=20.0, iters=100):
        ## E_{q_lam}[f] increases in lam (its derivative = Var_{q_lam}[f] >= 0) -> bisection
        for _ in range(iters):
            mid = (lo + hi) / 2
            if (self.tilt(p, mid) * self.f).sum() < self.c:
                lo = mid
            else:
                hi = mid
        return self.tilt(p, (lo + hi) / 2)

## Example: Jaynes's dice
faces = torch.arange(1, 7, dtype=torch.float64)
uniform = torch.full((6,), 1 / 6, dtype=torch.float64)
print(IProjection(faces, 4.5)(uniform).round(decimals=4))
## tensor([0.0544, 0.0788, 0.1142, 0.1654, 0.2398, 0.3475], dtype=torch.float64)
```
````

```{attention} Q&A
:class: dropdown
*Which projection is e & which is m?*

- I-projection $\min_qD_{KL}(q\|p)$: $q^*$ joins $p$ by an e-geodesic (exponential tilt) → **e-projection**.
- M-projection $\min_qD_{KL}(p\|q)$: $q^*$ joins $p$ by an m-geodesic → **m-projection**.
- Unique, if attained, when the target set is flat in the opposite sense: e-projection onto m-flat, m-projection onto e-flat.

*Why do they behave differently?*

- M-projection averages over $p$ → covers all of $p$'s mass (moment matching).
- I-projection averages over $q$ → may drop modes of $p$ (mode seeking) → mean-field VI underestimates variance.
- Details → [KL Divergence](../info.md#kl-divergence) Q&A.

*How is EM a pair of projections?*

- Objective: $\min_{q,\theta}D_{KL}(q(x,z)\|p_\theta(x,z))$ over $q$ w/ $x$-marginal = data.
- E-step: minimize over $q$ → I-projection onto the data family → $q(z\mid x)=p_\theta(z\mid x)$ ← KL chain rule.
- M-step: minimize over $\theta$ → M-projection onto the model.
- Each step ⬇️ the same KL → monotone convergence → [EM](../../ml/optim.md#em).

*How does iterative scaling relate?*

- Iterative proportional fitting (IPF) = cyclic I-projections onto single marginal constraints. Csiszár (1975) proves convergence.
- [Sinkhorn](ot.md#sinkhorn-algorithm) = IPF w/ 2 marginal constraints.

*What does this tell us about intelligence/AC?*

- Variational inference = I-projection of the true posterior onto the agent's tractable belief family.
- Free-energy-principle / active-inference accounts model perception this way. Status as a theory of consciousness: contested.
- Max-entropy = "assume nothing beyond the evidence" → I-projection formalizes least-committal belief update.
```

&nbsp;

## Geometric Optimization

### Natural Gradient
- **What**: Steepest descent under the Fisher metric: $G^{-1}\nabla\mathcal{L}$. {cite:p}`amari1998natural`
- **Why**: Plain gradient = steepest descent under the Euclidean metric on $\theta$.
    - Same step in $\theta$ → tiny or huge change in $p_\theta$, depending on the parametrization.
    - Measure the step in KL → reparametrization-invariant (first order) & better conditioned.
- **How**:
    1. Compute $\nabla\mathcal{L}$.
    2. Estimate $G$ w/ targets sampled from the model.
    3. Solve $(G+\lambda I)\,\boldsymbol\delta=\nabla\mathcal{L}$: exact, conjugate gradient, or a structured approximation ([K-FAC](#k-fac)).
    4. $\theta\leftarrow\theta-\alpha\boldsymbol\delta$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda\geq0$: Damping.
    - $\varepsilon$: KL radius of one step.
- Misc:
    - $\mathbf{g}=\nabla_\theta\mathcal{L}$: Euclidean gradient.
    - $\tilde{\nabla}\mathcal{L}=G^{-1}\mathbf{g}$: Natural gradient.

Process:

$$
\theta_{t+1}=\theta_t-\alpha\,G(\theta_t)^{-1}\nabla\mathcal{L}(\theta_t)
$$

Fisher of a conditional model $p_\theta(y\mid\mathbf{x})$:

$$
G(\theta)=\mathbb{E}_{\mathbf{x}\sim\mathcal{D}}\,\mathbb{E}_{y\sim p_\theta(\cdot\mid\mathbf{x})}\left[\nabla_\theta\log p_\theta(y\mid\mathbf{x})\,\nabla_\theta\log p_\theta(y\mid\mathbf{x})^\top\right]
$$

- $\mathcal{D}$: Data distribution of inputs.
- $y$ drawn from the **model**. $y$ from data labels → **empirical Fisher**, a different matrix.

Objective (one step):

$$
\boldsymbol\delta^*=\arg\min_{\boldsymbol\delta}\ \mathbf{g}^\top\boldsymbol\delta\ \ \text{s.t.}\ \ \tfrac12\boldsymbol\delta^\top G\boldsymbol\delta\leq\varepsilon
\;\Rightarrow\;
\boldsymbol\delta^*=-\sqrt{\frac{2\varepsilon}{\mathbf{g}^\top G^{-1}\mathbf{g}}}\;G^{-1}\mathbf{g}
$$

Properties:
- $\tilde\nabla\mathcal{L}$ = Riemannian gradient under $G$ ([Riemannian Gradient Descent](manifold.md#riemannian-gradient-descent)). Update = first-order retraction, ❌exponential map.
- Exponential family in natural params: $G(\theta)^{-1}\nabla_\theta\mathcal{L}=\nabla_\eta\mathcal{L}$.
- **Fisher efficiency**: online NLL minimization w/ $\alpha_t=1/t$ → $\operatorname{Cov}(\theta_t)\approx(tG)^{-1}$ asymptotically = Cramér–Rao bound.
```

```{tip} Derivation
:class: dropdown
*Why is $G^{-1}\mathbf{g}$ the steepest direction?*

1. Lagrangian of the Objective: $\mathbf{g}^\top\boldsymbol\delta+\nu\left(\frac12\boldsymbol\delta^\top G\boldsymbol\delta-\varepsilon\right)$, $\nu>0$.
2. Stationarity: $\mathbf{g}+\nu G\boldsymbol\delta=\mathbf{0}$ → $\boldsymbol\delta=-\nu^{-1}G^{-1}\mathbf{g}$.
3. Active constraint: $\frac{1}{2\nu^2}\mathbf{g}^\top G^{-1}\mathbf{g}=\varepsilon$ → $\nu^{-1}=\sqrt{2\varepsilon/(\mathbf{g}^\top G^{-1}\mathbf{g})}$.
4. Euclidean ball $\frac12\|\boldsymbol\delta\|^2\leq\varepsilon$ instead → plain gradient.

*Why is it reparametrization-invariant?*

1. $\theta=h(\xi)$, $J=\partial\theta/\partial\xi$ invertible.
2. $\mathbf{g}_\xi=J^\top\mathbf{g}_\theta$. $G_\xi=J^\top G_\theta J$.
3. $\boldsymbol\delta_\xi=G_\xi^{-1}\mathbf{g}_\xi=J^{-1}G_\theta^{-1}\mathbf{g}_\theta$.
4. Map to $\theta$: $J\boldsymbol\delta_\xi=G_\theta^{-1}\mathbf{g}_\theta=\boldsymbol\delta_\theta$ → same first-order change of $p_\theta$.
5. Plain gradient: $J\mathbf{g}_\xi=JJ^\top\mathbf{g}_\theta\neq\mathbf{g}_\theta$ in general.

*Why $G^{-1}\nabla_\theta\mathcal{L}=\nabla_\eta\mathcal{L}$ for exponential families?*

1. $\eta=\nabla\psi(\theta)$ → $\partial\eta/\partial\theta=\nabla^2\psi=G$ (symmetric).
2. Chain rule: $\nabla_\theta\mathcal{L}=G\,\nabla_\eta\mathcal{L}$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

## Reuses CategoricalFisher from the Fisher-Rao Metric block above
class SoftmaxNaturalGradient:
    ## natural-gradient step for softmax regression p(y|x) = softmax(W x)
    def __init__(self, lr, damping=1e-3):
        self.lr, self.damping = lr, damping

    def step(self, W, X, y):
        W = W.detach().requires_grad_()
        logits = X @ W.T                                        ## [m, C]
        g = torch.autograd.grad(F.cross_entropy(logits, y), W)[0].reshape(-1)  ## row-major vec(W)
        Fz = CategoricalFisher()(logits.detach())               ## [m, C, C], y ~ model
        ## score wrt W = (e_y - p) x^T -> Fisher = E_x[F_z (kron) x x^T]
        G = torch.stack([torch.kron(Fz[s], torch.outer(X[s], X[s])) for s in range(len(X))]).mean(0)
        G = G + self.damping * torch.eye(G.shape[0])            ## softmax over-param -> G singular
        delta = torch.linalg.solve(G, g)                        ## G^{-1} g w/o forming the inverse
        return (W - self.lr * delta.view_as(W)).detach()

## Example
X, y = torch.randn(4, 3), torch.tensor([0, 1, 2, 0])
W = SoftmaxNaturalGradient(lr=0.5).step(torch.zeros(3, 3), X, y)
print(W.shape)  ## torch.Size([3, 3])
```
````

```{attention} Q&A
:class: dropdown
*Is it exactly parametrization-invariant?*

- Continuous-time flow $\dot\theta=-G^{-1}\nabla\mathcal{L}$: ✅ exactly.
- Discrete steps: only to first order. Curvature of the chart → 2nd-order differences.
- Damping $\lambda I$ & approximations (K-FAC, diagonal) break invariance further.

*Empirical Fisher vs Fisher?*

- Empirical Fisher: $y$ from data labels. Cheap ← reuses per-sample grads.
- ≠ Fisher away from a well-specified optimum → can mis-scale & mis-direct steps {cite:p}`martens2020new`.
- Near a well-specified optimum, both ≈ Hessian of the NLL.

*Natural gradient vs Newton?*

- Newton: $H^{-1}\mathbf{g}$. $H$ can be indefinite → can step uphill.
- Natural gradient: $G\succeq0$ → always a descent direction.
- NLL w/ exponential-family outputs → $G$ = generalized Gauss-Newton ≈ $H$ near a good fit ([Gauss-Newton](../../ml/optim.md#gauss-newton)).

*Is Adam a natural-gradient method?*

- Loosely. [Adam](../../dl/train/optim.md#adam-adaptive-moment-estimation) divides by $\sqrt{\text{EMA of squared mini-batch grads}}$.
- Differences: diagonal only, empirical (data labels), mini-batch-averaged grads, & a square root (Fisher has power 1).
- → "Adam ≈ diagonal natural gradient" is an analogy, ❌an identity.

*Where does it appear in RL & LLM training?*

- [Natural policy gradient](../../rl/policy.md#natural-policy-gradient): $G$ = Fisher of the policy.
- [TRPO](../../rl/policy.md#trpo): KL-ball constraint = Objective above; step length = $\sqrt{2\varepsilon/(\mathbf{g}^\top G^{-1}\mathbf{g})}$.
- [PPO](../../rl/policy.md#ppo): clipping as a cheap substitute for the KL ball.
- [RLHF](../../nlp/post/rl.md#rlhf): $\beta D_{KL}(\pi_\theta\|\pi_\text{ref})$ penalty ≈ $\frac\beta2\Delta\theta^\top G\Delta\theta$ near $\pi_\text{ref}$ only.
- [NES](../../ml/optim.md#nes): natural gradient on a search distribution.

*Cons?*

- $G$ is $d\times d$ → $O(d^2)$ memory, $O(d^3)$ solve → impossible for NNs w/o structure.
- Fisher estimate is noisy → damping & trust-region tuning.

*Why did Amari expect fewer plateaus?*

- MLP plateaus ← near-singular regions of $G$ (symmetric hidden units).
- $G^{-1}$ amplifies the small gradient along flat directions → Amari (1998) suggests plateaus become less severe.

*What does this tell us about intelligence/AC?*

- Learning rule that depends only on the distribution, not on the wiring/param code.
- Hypothesis: a substrate-independent learner should be natural-gradient-like. Unproven; brains show no evidence of an explicit $G^{-1}$.
```

&nbsp;

#### K-FAC
- **Name**: Kronecker-Factored Approximate Curvature {cite:p}`martens2015optimizing`
- **What**: Natural gradient w/ each layer's Fisher block approximated by a Kronecker product of 2 small matrices.
- **Why**: Exact $G$ is $d\times d$ → infeasible for NNs w/ $d\sim10^6$+.
- **How**:
    1. Drop cross-layer blocks → block-diagonal $G$.
    2. Per layer: Fisher ≈ (input-activation 2nd moment) ⊗ (output-grad 2nd moment).
    3. Invert 2 small matrices → update $\Gamma^{-1}(\nabla W)A^{-1}$.
    4. Damp the factors. Refresh them w/ an EMA. Re-invert every few steps.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\bar{\mathbf{a}}\in\mathbb{R}^{d_{in}}$: Layer input (append 1 for bias).
    - $\mathbf{s}=W\bar{\mathbf{a}}\in\mathbb{R}^{d_{out}}$: Pre-activation.
- Params:
    - $W\in\mathbb{R}^{d_{out}\times d_{in}}$: Layer weight.
- Hyperparams:
    - $\lambda$: Damping.
- Misc:
    - $\mathbf{g}_s=\partial\log p_\theta(y\mid\mathbf{x})/\partial\mathbf{s}$: Backprop grad, $y\sim$ model.
    - $A=\mathbb{E}[\bar{\mathbf{a}}\bar{\mathbf{a}}^\top]\in\mathbb{R}^{d_{in}\times d_{in}}$: Activation factor.
    - $\Gamma=\mathbb{E}[\mathbf{g}_s\mathbf{g}_s^\top]\in\mathbb{R}^{d_{out}\times d_{out}}$: Grad factor (paper's $G$; here $G$ = Fisher).
    - $\operatorname{vec}$: Column-stacking.
    - $\pi$: Damping split between factors.

Process:

$$
\begin{align*}
\nabla_W\log p&=\mathbf{g}_s\bar{\mathbf{a}}^\top\ \Rightarrow\ \operatorname{vec}(\nabla_W\log p)=\bar{\mathbf{a}}\otimes\mathbf{g}_s\\
G_W&=\mathbb{E}\left[\bar{\mathbf{a}}\bar{\mathbf{a}}^\top\otimes\mathbf{g}_s\mathbf{g}_s^\top\right]\approx A\otimes\Gamma\\
(A\otimes\Gamma)^{-1}\operatorname{vec}(V)&=\operatorname{vec}\left(\Gamma^{-1}VA^{-1}\right)\\
W&\leftarrow W-\alpha\,(\Gamma+\tfrac{\sqrt\lambda}{\pi}I)^{-1}\,\nabla_W\mathcal{L}\,(A+\pi\sqrt\lambda I)^{-1}
\end{align*}
$$

- $V$: Any $d_{out}\times d_{in}$ matrix.
- $\approx$: Assumes $\bar{\mathbf{a}}\bar{\mathbf{a}}^\top$ & $\mathbf{g}_s\mathbf{g}_s^\top$ are uncorrelated → expectation of product = product of expectations.
- Damping: factored Tikhonov. $\pi=\sqrt{\dfrac{\operatorname{tr}(A)/d_{in}}{\operatorname{tr}(\Gamma)/d_{out}}}$ balances the 2 factors.
- Cost per layer: invert $d_{in}^2$ & $d_{out}^2$ matrices, ❌a $(d_{in}d_{out})^2$ matrix.
```

```{tip} Derivation
:class: dropdown
*Why $(A\otimes\Gamma)^{-1}\operatorname{vec}(V)=\operatorname{vec}(\Gamma^{-1}VA^{-1})$?*

1. $(A\otimes\Gamma)^{-1}=A^{-1}\otimes\Gamma^{-1}$.
2. Identity: $(B^\top\otimes C)\operatorname{vec}(V)=\operatorname{vec}(CVB)$.
3. $B=A^{-1}$ (symmetric), $C=\Gamma^{-1}$ → $\operatorname{vec}(\Gamma^{-1}VA^{-1})$.
```

````{important} Code
:class: dropdown
```python
import torch

class KFACLinear:
    ## K-FAC preconditioner for one linear layer W: [d_out, d_in] (bias: append 1 to a)
    def __init__(self, d_in, d_out, damping=1e-3, decay=0.95):
        self.A, self.Gam = torch.eye(d_in), torch.eye(d_out)
        self.damping, self.decay = damping, decay

    def update(self, a, gs):
        ## a: [m, d_in] layer inputs; gs: [m, d_out] grads wrt pre-activations, y ~ MODEL
        m = a.shape[0]
        self.A = self.decay * self.A + (1 - self.decay) * a.T @ a / m
        self.Gam = self.decay * self.Gam + (1 - self.decay) * gs.T @ gs / m

    def precondition(self, grad_W):
        d_out, d_in = grad_W.shape
        ## factored Tikhonov damping, split by average eigenvalue (trace / dim)
        pi = ((self.A.trace() / d_in) / (self.Gam.trace() / d_out)).sqrt()
        A = self.A + pi * self.damping ** 0.5 * torch.eye(d_in)
        Gam = self.Gam + self.damping ** 0.5 / pi * torch.eye(d_out)
        ## (A kron Gam)^{-1} vec(grad) = vec(Gam^{-1} grad A^{-1})
        return torch.linalg.solve(Gam, grad_W) @ torch.linalg.inv(A)

## Example
a, gs = torch.randn(8, 3), torch.randn(8, 2)
kfac = KFACLinear(d_in=3, d_out=2)
kfac.update(a, gs)
print(kfac.precondition(gs.T @ a / 8).shape)  ## torch.Size([2, 3])
```
````

```{attention} Q&A
:class: dropdown
*Assumptions?*

- Cross-layer Fisher blocks ≈ 0 (block-diagonal). The paper also gives a block-tridiagonal variant.
- Activation outer products & pre-activation-grad outer products ≈ uncorrelated (independent) within each layer.
- Targets sampled from the model → true Fisher, ❌empirical Fisher.

*Why is it cheap?*

- Store $d_{in}^2+d_{out}^2$ numbers, ❌$(d_{in}d_{out})^2$.
- Invert small factors every few steps; reuse between refreshes.

*How does it relate to Shampoo?*

- Shampoo also preconditions $W$ from both sides w/ 2 small matrices.
- Uses gradient outer products $\nabla W\nabla W^\top$ & $\nabla W^\top\nabla W$ w/ inverse 4th roots, ❌Fisher factors.

*When is it NOT enough?*

- Strong cross-layer coupling, attention & weight-sharing layers → Kronecker form needs extra approximations.
- Small/noisy batches → factor estimates & damping dominate.
```

&nbsp;

### Mirror Descent
- **What**: Gradient step in the dual space of a convex mirror map $F$, then a Bregman projection back. {cite:p}`nemirovski1983problem`
- **Why**: Euclidean steps ignore the domain's geometry → bounds grow w/ dimension.
    - Simplex, sparse vectors, positive cones → non-Euclidean norms fit better.
    - Simplex w/ entropy: $\sqrt{\log d}$ instead of $\sqrt d$ dependence.
- **How**:
    1. Map the iterate to the dual: $\nabla F(\mathbf{w}_t)$.
    2. Take the gradient step in the dual.
    3. Map back w/ $(\nabla F)^{-1}$.
    4. Bregman-project onto the feasible set.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{w}\in\mathcal{W}\subseteq\mathbb{R}^d$: Iterate in a closed convex set.
    - $\mathbf{g}_t$: (Sub)gradient of $\mathcal{L}$ at $\mathbf{w}_t$.
- Hyperparams:
    - $F$: Mirror map, $\sigma$-strongly convex w.r.t. a norm $\|\cdot\|$.
- Misc:
    - $\|\cdot\|_*$: Dual norm.
    - $L_*$: Bound on $\|\mathbf{g}_t\|_*$.
    - $T$: #steps.

Process:

$$
\begin{align*}
\nabla F(\mathbf{v}_{t+1})&=\nabla F(\mathbf{w}_t)-\alpha\,\mathbf{g}_t\\
\mathbf{w}_{t+1}&=\arg\min_{\mathbf{w}\in\mathcal{W}}D_F(\mathbf{w},\mathbf{v}_{t+1})
\end{align*}
$$

Equivalent (proximal form):

$$
\mathbf{w}_{t+1}=\arg\min_{\mathbf{w}\in\mathcal{W}}\ \alpha\langle\mathbf{g}_t,\mathbf{w}\rangle+D_F(\mathbf{w},\mathbf{w}_t)
$$

Instances:
- $F=\frac12\|\mathbf{w}\|_2^2$ → projected (sub)gradient descent.
- $F=\sum_jw_j\log w_j$ on the simplex → **exponentiated gradient**:

    $$
    w_{t+1,j}=\frac{w_{t,j}\,e^{-\alpha g_{t,j}}}{\sum_kw_{t,k}\,e^{-\alpha g_{t,k}}}
    $$

Guarantee (convex $\mathcal{L}$, $\mathbf{w}^*$ optimal):

$$
\frac1T\sum_{t=1}^T\mathcal{L}(\mathbf{w}_t)-\mathcal{L}(\mathbf{w}^*)\leq\frac{D_F(\mathbf{w}^*,\mathbf{w}_1)}{\alpha T}+\frac{\alpha L_*^2}{2\sigma}
$$

- Simplex + entropy: $\sigma=1$ w.r.t. $\|\cdot\|_1$ (Pinsker), $\|\cdot\|_*=\|\cdot\|_\infty$, $D_F(\mathbf{w}^*,\text{uniform})\leq\log d$.
- Tuned $\alpha$ → $L_\infty\sqrt{2\log d/T}$. Projected GD on the simplex → $O(L_\infty\sqrt{d/T})$.
```

```{tip} Derivation
:class: dropdown
*Why are the two forms equal?*

1. $\alpha\langle\mathbf{g}_t,\mathbf{w}\rangle+D_F(\mathbf{w},\mathbf{w}_t)=F(\mathbf{w})-\langle\nabla F(\mathbf{w}_t)-\alpha\mathbf{g}_t,\mathbf{w}\rangle+\text{const}$.
2. $\nabla F(\mathbf{w}_t)-\alpha\mathbf{g}_t=\nabla F(\mathbf{v}_{t+1})$.
3. → $F(\mathbf{w})-\langle\nabla F(\mathbf{v}_{t+1}),\mathbf{w}\rangle+\text{const}=D_F(\mathbf{w},\mathbf{v}_{t+1})+\text{const}$.

*Why is mirror descent a natural gradient?*

1. Continuous time: $\frac{d}{dt}\nabla F(\mathbf{w})=-\nabla\mathcal{L}$ → $\nabla^2F(\mathbf{w})\,\dot{\mathbf{w}}=-\nabla\mathcal{L}$.
2. → $\dot{\mathbf{w}}=-[\nabla^2F]^{-1}\nabla\mathcal{L}$ = Riemannian gradient flow under the Hessian metric $\nabla^2F$.
3. Exponential family, $F=\psi$ on $\theta$: $\nabla^2\psi=G$ → natural-gradient flow.
4. Discrete: MD w/ $F=\psi$ → $\eta_{t+1}=\eta_t-\alpha\nabla_\theta\mathcal{L}=\eta_t-\alpha\,G_\eta^{-1}\nabla_\eta\mathcal{L}$ → exact natural-gradient step in $\eta$ coordinates.
    - ← $\nabla_\theta\mathcal{L}=G\nabla_\eta\mathcal{L}$ & $G_\eta=G^{-1}$ ([Dually Flat Manifold](#dually-flat-manifold)).
```

````{important} Code
:class: dropdown
```python
import torch

class ExponentiatedGradient:
    ## mirror descent on the simplex w/ negative-entropy mirror map
    def __init__(self, lr):
        self.lr = lr

    def step(self, w, g):
        ## dual step: log w - lr * g; map back: exp; Bregman (KL) projection: normalize
        w = w * torch.exp(-self.lr * g)
        return w / w.sum()

## Example: linear loss <c, w> on the simplex -> mass flows to the cheapest coordinate
c = torch.tensor([1.0, 2.0, 3.0])
w, opt = torch.full((3,), 1 / 3), ExponentiatedGradient(lr=0.5)
for _ in range(10):
    w = opt.step(w, c)
print([round(v, 4) for v in w.tolist()])  ## [0.9933, 0.0067, 0.0]
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Dimension dependence matched to the geometry: $\sqrt{\log d}$ on the simplex.
- Iterates stay feasible automatically (entropy → positive & normalized).
- Same template covers GD, EG, Hedge, AdaGrad.

*Cons?*

- Need a mirror map w/ computable $\nabla F$, $(\nabla F)^{-1}$ & Bregman projection.
- Wrong mirror map → worse than plain GD.

*How does it relate to projected subgradient?*

- Mirror descent = nonlinear projected subgradient method w/ a Bregman distance instead of $\frac12\|\cdot\|^2$ {cite:p}`beck2003mirror`.

*How does it relate to natural gradient?*

- Continuous time: mirror flow = Riemannian gradient flow under $\nabla^2F$ (Derivation).
- Discrete: MD w/ $F=\psi$ = natural gradient in $\eta$ coordinates; natural gradient in $\theta$ = MD w/ $\varphi$ in $\eta$ (Raskutti & Mukherjee, 2015).

*Where does it appear in ML/RL?*

- Online learning: Hedge / multiplicative weights = EG on expert weights.
- Tabular [natural policy gradient](../../rl/policy.md#natural-policy-gradient) = per-state EG w/ advantages.
- KL-regularized policy updates (TRPO, PPO, RLHF) = mirror-descent-style steps w/ KL as the Bregman term.
- AdaGrad = mirror descent w/ an adaptive quadratic mirror map $\frac12\mathbf{w}^\top H_t\mathbf{w}$.

*How does it relate to evolution?*

- Replicator dynamics $\dot w_j=w_j(f_j-\mathbf{w}^\top\mathbf{f})$ = continuous-time EG w/ fitness $\mathbf{f}=-\mathbf{g}$.
- → Selection = mirror ascent on the simplex of type frequencies.
```

&nbsp;

## Geometry of Integration

### Integrated Information as Projection
- **What**: Integration = min KL from a system's full model to a "disconnected" model w/ cross-part influences cut. {cite:p}`oizumi2016unified`
- **Why**: [IIT](../../ac/theories.md#integrated-information-theory-iit) needs a measure of "whole > sum of parts".
    - Older measures break natural bounds: stochastic interaction & summed transfer entropies can exceed $I(X;Y)$.
    - Projection onto nested submodels → one framework; bounds follow from set inclusion.
- **How**:
    1. Model past state $X$ & present state $Y$ of the system: $p(X,Y)$.
    2. Define a disconnected manifold by conditional-independence constraints.
    3. M-project $p$ onto it → $q^*$.
    4. Information loss $D_{KL}(p\|q^*)$ = strength of the cut interactions.
    5. Different constraints → [mutual information](../info.md#mutual-information), transfer entropy, $\Phi_G$, stochastic interaction.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $X=(x_1,\dots,x_N)$: Past state of $N$ units.
    - $Y=(y_1,\dots,y_N)$: Present state.
- Misc:
    - $P_1,\dots,P_K$: Partition of the units into $K$ parts.
    - $X[P_k]$: Past state of the units in part $P_k$.
    - $Y[P_k]$: Present state of the units in part $P_k$.
    - $X_{\setminus a}$: Past state of all units except $a$.
    - $\mathcal{M}_\cdot$: Manifold of disconnected models $q(X,Y)$.

Definition (one measure per constraint set):

$$
\Phi_\mathcal{M}=\min_{q\in\mathcal{M}}D_{KL}\big(p(X,Y)\,\|\,q(X,Y)\big)
$$

| Manifold | Constraint on $q$ | $\min_qD_{KL}(p\Vert q)$ |
|:--|:--|:--|
| $\mathcal{M}_I$ | $q(X,Y)=q(X)q(Y)$ | $I(X;Y)$ |
| $\mathcal{M}_{TE}$ (cut $x_a\to y_b$) | $q(y_b\mid X)=q(y_b\mid X_{\setminus a})$ | $TE(x_a\to y_b\mid X_{\setminus a})$ |
| $\mathcal{M}_G$ | $q(Y[P_k]\mid X)=q(Y[P_k]\mid X[P_k])\ \forall k$ | $\Phi_G$ |
| $\mathcal{M}_S$ | $q(Y\mid X)=\prod_kq(Y[P_k]\mid X[P_k])$ | $SI=\sum_kH(Y[P_k]\mid X[P_k])-H(Y\mid X)$ |

- $TE$: Conditional transfer entropy $H(y_b\mid X_{\setminus a})-H(y_b\mid X)$.
- $\Phi_G$: Geometric integrated information.
- $SI$: Stochastic interaction.

Properties:
- $\mathcal{M}_I\subset\mathcal{M}_G$ & $\mathcal{M}_S\subset\mathcal{M}_G$ → $0\leq\Phi_G\leq\min(I(X;Y),SI)$.
- $\mathcal{M}_S$ also cuts equal-time correlations among the $Y[P_k]$ → $SI$ can exceed $I(X;Y)$.
- $x_a,y_b$ in different parts → $\mathcal{M}_G\subset\mathcal{M}_{TE}$ → $TE\leq\Phi_G$.
- Larger manifold (fewer cuts) → smaller loss → all disconnected models form a partial order; top (all links cut) = $I(X;Y)$.
- Gaussian AR model $Y=AX+E$: $\mathcal{M}_G$ ⇔ $A'$ w/ cross-part blocks $=0$.
    - Find the projected params $A'$, $\Sigma_{E'}$ numerically, then evaluate:

    $$
    \Phi_G=\frac12\log\frac{|\Sigma_{E'}|}{|\Sigma_E|}
    $$

    - $A$: Connectivity matrix.
    - $E$: Gaussian noise w/ covariance $\Sigma_E$.
    - $\Sigma_{E'}$: Noise covariance of the projected model.
    - = generalized-variance form of multivariate Granger causality.
```

```{note} Example
:class: dropdown
2 binary units, $p(x_1,x_2)$ uniform, $y_1=y_2=x_1\oplus x_2$ (XOR), parts $\{1\}$ & $\{2\}$. In bits:
- $I(X;Y)=H(Y)-H(Y\mid X)=1-0=1$.
- $TE(x_1\to y_2\mid x_2)=H(y_2\mid x_2)-H(y_2\mid X)=1-0=1$. Same for $x_2\to y_1$.
- Sum of TEs $=2>I(X;Y)$ → summed TE breaks the bound.
- $SI=H(y_1\mid x_1)+H(y_2\mid x_2)-H(Y\mid X)=1+1-0=2>I(X;Y)$.
- $1=TE\leq\Phi_G\leq I(X;Y)=1$ → $\Phi_G=1$.
```

```{attention} Q&A
:class: dropdown
*Is $\Phi_G$ the $\Phi$ of IIT 3.0/4.0?*

- ❌. $\Phi_G$ is a measure "in the spirit of IIT".
- Uses the observed joint $p(X,Y)$, ❌perturbations or cause–effect repertoires.
- "Causal" = statistical (Granger-like), not interventional.
- Yields one number per partition, ❌a cause–effect structure.

*Why m-projection & not I-projection?*

- $\min_qD_{KL}(p\|q)$ → $q^*(X)=p(X)$ → "what the system would do if parts could not talk".
- Conditional model changes to satisfy the disconnection constraints.
    - $\mathcal{M}_S$: keeps each $p(Y[P_k]\mid X[P_k])$.
    - $\mathcal{M}_G$: need not keep them.
- $\mathcal{M}_I$ & $\mathcal{M}_S$: closed form (marginals / factorized conditionals).
- $\mathcal{M}_G$: numerical optimization in general, incl. Gaussian AR models.

*How is the partition chosen?*

- Typically the minimum information partition (MIP): the cut that loses least.
- #partitions grows faster than exponentially in $N$ → exact search infeasible for large systems.

*Does a feedforward coupling give $\Phi_G>0$?*

- ✅. One-way $x_2\to y_1$ dependence → $\Phi_G\geq TE(x_2\to y_1\mid x_1)>0$.
- IIT 3.0 assigns $\Phi=0$ to feedforward systems → the measures disagree on a core IIT prediction.

*What does this tell us about intelligence/AC?*

- Integration = geometric distance to the nearest "disconnected" model → coordinate-free & bounded by $I(X;Y)$.
- Computable for recorded neural/NN activity → test "integration tracks conscious level" empirically.
- Whether any $\Phi$ measures consciousness: contested ([IIT](../../ac/theories.md#integrated-information-theory-iit)).
```

&nbsp;
